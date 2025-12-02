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
import numpy as np
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
        "--push_to", type=str, default=None,
        help="HuggingFace repo ID (e.g., 'Org/Dataset'). If not provided, will auto-generate "
             "names based on sample count and seed: SoheylM/ShinkaEvolve-SDS-{N}k-seed{seed} and "
             "IDEALLab/ShinkaEvolve-SDS-{N}k-seed{seed}"
    )
    parser.add_argument(
        "--api_key", type=str,
        help="OpenAI API Key (optional if in env)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="sds_dataset_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Master seed for reproducibility. Seeds problem generation deterministically. "
             "Evolution uses natural randomness to ensure diversity and avoid mode collapse."
    )
    args = parser.parse_args()
    
    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API Key required for SFT Data Generation")
    
    # Check for HF_TOKEN if pushing to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    
    # Set up seeding for reproducibility
    # Strategy: Seed problem generation deterministically, but let evolution
    # use natural randomness to avoid mode collapse while maintaining problem reproducibility
    master_seed = args.seed
    if master_seed is not None:
        # Seed Python's random for problem generation
        random.seed(master_seed)
        print(f"🌱 Using master seed: {master_seed} (problem generation seeded, evolution uses natural randomness)")
    else:
        print("⚠️  No seed provided - results will not be reproducible")
    
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
        # Seed problem generation deterministically (reproducible problems)
        if master_seed is not None:
            problem_seed = master_seed + i
            random.seed(problem_seed)  # Seed for this problem's generation
        
        # A. Generate a Problem Instance (deterministic when seed is set)
        ptype = "dense" if i % 2 == 0 else "tree"
        n_size = random.randint(15, 25)  # Deterministic when seed is set
        problem_gen_seed = problem_seed if master_seed is not None else i
        if ptype == "dense":
            inst = make_dense_instance(n=n_size, seed=problem_gen_seed)
        else:
            inst = make_tree_instance(n=n_size, seed=problem_gen_seed)
        
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
        # EvolutionRunner will prepend results_dir, so just use filename
        db_config.db_path = "evolution_db.sqlite"
        
        job_config = LocalJobConfig(
            eval_program_path="evaluate.py"  # Relative to current directory
        )
        
        # C. Run Shinka Evolution
        # Note: Evolution uses natural randomness to explore diverse solutions
        # and avoid mode collapse. Problem generation is seeded for reproducibility.
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
        # EvolutionRunner creates db at results_dir/db_path
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
    if len(dataset_records) > 0:
        ds = Dataset.from_list(dataset_records)
        
        # Determine repo names using same logic as push_sds_to_hf.py
        if args.push_to:
            # Use provided repo name
            repos_to_push = [args.push_to]
        else:
            # Auto-generate repo names based on sample count and seed
            total_samples = len(dataset_records)
            # Format sample count for repo name (e.g., 10000 -> "10k", 5000 -> "5k")
            if total_samples >= 1000:
                sample_suffix = f"{total_samples // 1000}k"
            else:
                sample_suffix = str(total_samples)
            
            # Use seed from args if available, otherwise use master_seed
            seed_value = args.seed if args.seed is not None else master_seed if master_seed is not None else 0
            
            # Create descriptive repo names for both organizations (matching push_sds_to_hf.py style)
            soheylm_repo = f"SoheylM/ShinkaEvolve-SDS-{sample_suffix}-seed{seed_value}"
            ideallab_repo = f"IDEALLab/ShinkaEvolve-SDS-{sample_suffix}-seed{seed_value}"
            repos_to_push = [soheylm_repo, ideallab_repo]
        
        # Push to HuggingFace
        if not hf_token:
            print("⚠️  HF_TOKEN not found in environment. Cannot push to HuggingFace.")
            print("   Saving dataset locally instead...")
            repos_to_push = []  # Skip pushing
        
        success_count = 0
        for repo_name in repos_to_push:
            print(f"📤 Pushing to HuggingFace: {repo_name}...")
            try:
                ds.push_to_hub(repo_name, token=hf_token, private=True)
                print(f"✅ Successfully pushed to {repo_name}")
                success_count += 1
            except Exception as e:
                print(f"❌ Error pushing to {repo_name}: {e}")
        
        if success_count > 0:
            print(f"✅ Successfully pushed to {success_count}/{len(repos_to_push)} repositories")
        else:
            print(f"❌ Failed to push to any repository")
        
        # Always save locally as backup
        output_file = output_dir / "sft_dataset.jsonl"
        with open(output_file, "w") as f:
            for r in dataset_records:
                f.write(json.dumps(r) + "\n")
        print(f"💾 Saved local backup to {output_file}")
        
        # Also save as JSON for easy inspection
        json_file = output_dir / "sft_dataset.json"
        with open(json_file, "w") as f:
            json.dump(dataset_records, f, indent=2)
        print(f"💾 Also saved to {json_file}")
    else:
        print("⚠️  No dataset records to save or push")


if __name__ == "__main__":
    main()
