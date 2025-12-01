"""
Pipeline for generating high-quality SFT datasets for SDS problems using ShinkaEvolve.

This script:
1. Generates SDS problem instances
2. Runs ShinkaEvolve evolution for each problem
3. Extracts best solutions from the database
4. Generates reasoning traces
5. Formats data for student model training
"""

import os
import json
import argparse
import tqdm
import sys
import random
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset

# ShinkaEvolve imports
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.launch import LocalJobConfig
from shinka.annotation.trace_generator import TraceGenerator

# Import problem generation (assumed to exist)
try:
    from gen_sds_dataset import sds_render_prompt, _instance_to_problem
    from gen_sds_dataset import make_dense_instance, make_tree_instance
except ImportError:
    print("❌ Could not import gen_sds_dataset.py. Make sure it is in the path.")
    sys.exit(1)


def create_sds_evolution_config(
    problem_data: dict,
    num_generations: int = 10,
    results_dir: str = None,
) -> EvolutionConfig:
    """Create EvolutionConfig for SDS problem."""
    
    search_task_sys_msg = """You are an expert algorithmist specializing in combinatorial optimization, particularly the Synergistic Dependency Selection (SDS) problem.

The SDS problem requires selecting a subset of variables that:
1. Satisfies cardinality bounds (min/max number of items)
2. Respects mutex constraints (cannot select both items in a mutex pair)
3. Respects group constraints (at most one item per group)
4. Satisfies precedence constraints (if j is selected, i must be selected)
5. Maximizes the objective: sum of selected item weights + pairwise interaction values

Key strategies to explore:
1. Greedy selection with constraint-aware ordering
2. Dynamic programming for small instances
3. Constraint satisfaction with backtracking
4. Weighted scoring that accounts for interactions
5. Precedence-aware selection ordering
6. Group-aware selection that maximizes diversity

The code should read JSON from stdin with "requirements" and "catalog", and output JSON to stdout with "selection" containing "variables" list.

Be creative and find efficient solutions that maximize the objective while respecting all constraints."""
    
    return EvolutionConfig(
        task_sys_msg=search_task_sys_msg,
        patch_types=["diff", "full", "cross"],
        patch_type_probs=[0.6, 0.3, 0.1],
        num_generations=num_generations,
        max_parallel_jobs=1,  # Sequential for dataset generation
        max_patch_resamples=3,
        max_patch_attempts=3,
        job_type="local",
        language="python",
        llm_models=["gpt-5-mini"],
        llm_kwargs=dict(
            temperatures=[0.0, 0.5, 0.7],
            max_tokens=8192,
        ),
        embedding_model="text-embedding-3-small",
        init_program_path="initial.py",  # Relative to current directory
        results_dir=results_dir,
    )


def create_sds_database_config() -> DatabaseConfig:
    """Create DatabaseConfig for SDS evolution."""
    return DatabaseConfig(
        db_path="evolution_db.sqlite",
        num_islands=2,
        archive_size=20,
        elite_selection_ratio=0.3,
        num_archive_inspirations=4,
        num_top_k_inspirations=2,
        migration_interval=10,
        migration_rate=0.1,
        island_elitism=True,
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )


def extract_best_code_from_database(db_path: str) -> tuple[str, float]:
    """
    Extract the best solution code and score from the evolution database.
    
    Returns:
        (code: str, combined_score: float)
    """
    db_config = DatabaseConfig(db_path=db_path)
    db = ProgramDatabase(config=db_config, read_only=True)
    
    best_program = db.get_best_program()
    if not best_program:
        return None, 0.0
    
    return best_program.code, best_program.combined_score or 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT dataset for SDS using ShinkaEvolve"
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Number of SFT samples to generate"
    )
    parser.add_argument(
        "--generations", type=int, default=10,
        help="Evolution generations per problem"
    )
    parser.add_argument(
        "--push_to", type=str,
        help="HuggingFace repo ID (e.g., 'Org/Dataset')"
    )
    parser.add_argument(
        "--api_key", type=str,
        help="OpenAI API Key (optional if in env)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="sds_dataset_output",
        help="Output directory for results"
    )
    args = parser.parse_args()
    
    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API Key required for SFT Data Generation")
    
    # Initialize trace generator
    tracer = TraceGenerator(api_key=api_key, model="gpt-4o")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_records = []
    
    print(f"🚀 Starting Pipeline: {args.samples} samples")
    print(f"📁 Output directory: {output_dir}")
    
    # Loop through desired samples
    for i in tqdm.tqdm(range(args.samples), desc="Generating samples"):
        # A. Generate a Problem Instance
        ptype = "dense" if i % 2 == 0 else "tree"
        if ptype == "dense":
            inst = make_dense_instance(n=random.randint(15, 25), seed=i)
        else:
            inst = make_tree_instance(n=random.randint(15, 25), seed=i)
        
        problem_dict = _instance_to_problem(inst, ptype, i)
        prompt_data = sds_render_prompt(problem_dict)
        
        # B. Set up evolution for this problem
        # Create unique results directory for this problem
        problem_results_dir = output_dir / f"problem_{i}"
        problem_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variable for evaluate.py to use
        os.environ["SDS_PROBLEM_DATA"] = json.dumps(problem_dict)
        
        # Create configs
        evo_config = create_sds_evolution_config(
            problem_dict,
            num_generations=args.generations,
            results_dir=str(problem_results_dir),
        )
        db_config = create_sds_database_config()
        db_config.db_path = str(problem_results_dir / "evolution_db.sqlite")
        
        job_config = LocalJobConfig(
            eval_program_path="evaluate.py"  # Relative to current directory
        )
        
        # C. Run Shinka Evolution
        print(f"\n🔬 Running evolution for problem {i} ({ptype})...")
        evo_runner = EvolutionRunner(
            evo_config=evo_config,
            job_config=job_config,
            db_config=db_config,
            verbose=False,  # Less verbose for batch processing
        )
        
        try:
            evo_runner.run()
        except Exception as e:
            print(f"   ⚠️ Evolution failed for problem {i}: {e}")
            continue
        
        # D. Extract best solution from database
        db_path = str(problem_results_dir / "evolution_db.sqlite")
        best_code, best_fitness = extract_best_code_from_database(db_path)
        
        if not best_code or best_fitness <= 0.01:
            print(f"   ⚠️ Skipping {i}: No valid solution found (fitness: {best_fitness:.4f})")
            continue
        
        # E. Generate Thinking Trace
        print(f"   📝 Generating trace for problem {i}...")
        trace = tracer.generate(prompt_data["problem"], best_code)
        
        # F. Format Entry for Student Training
        full_content = (
            f"<think>\n{trace}\n</think>\n\n"
            f"<code>\n{best_code}\n</code>"
        )
        
        dataset_records.append({
            "problem_id": prompt_data.get("uuid", f"problem_{i}"),
            "messages": [
                {"role": "user", "content": prompt_data["problem"]},
                {"role": "assistant", "content": full_content}
            ],
            "score": float(best_fitness),
            "type": ptype,
            "mission": json.dumps(problem_dict.get("mission", {}))
        })
        
        print(f"   ✅ Problem {i} completed (fitness: {best_fitness:.4f})")
    
    print(f"\n✅ Generated {len(dataset_records)} high-quality samples.")
    
    # G. Save or Push Dataset
    if args.push_to and len(dataset_records) > 0:
        print(f"📤 Pushing to HuggingFace: {args.push_to}...")
        ds = Dataset.from_list(dataset_records)
        ds.push_to_hub(args.push_to, private=True)
        print("✅ Done!")
    else:
        output_file = output_dir / "sft_dataset.jsonl"
        with open(output_file, "w") as f:
            for r in dataset_records:
                f.write(json.dumps(r) + "\n")
        print(f"💾 Saved to {output_file}")
        
        # Also save as JSON for easy inspection
        json_file = output_dir / "sft_dataset.json"
        with open(json_file, "w") as f:
            json.dump(dataset_records, f, indent=2)
        print(f"💾 Also saved to {json_file}")


if __name__ == "__main__":
    main()
