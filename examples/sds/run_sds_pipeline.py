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
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Calculate paths: script is in deps/ShinkaEvolve/examples/sds/
# Project root is 4 levels up
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent.parent.parent

# Add project root to Python path for imports (like ShinkaEvolve tutorial)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ShinkaEvolve imports
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.launch import LocalJobConfig
from shinka.annotation.trace_generator import TraceGenerator

# Import problem generation from data directory
try:
    from data.gen_sds_dataset import sds_render_prompt, _instance_to_problem
    from data.gen_sds_dataset import make_dense_instance, make_tree_instance
except ImportError:
    print(f"❌ Could not import gen_sds_dataset.py.")
    print(f"   Script dir: {script_dir}")
    print(f"   Project root: {project_root}")
    print(f"   Looking for: {project_root / 'data' / 'gen_sds_dataset.py'}")
    sys.exit(1)


def create_sds_evolution_config(
    problem_data: dict,
    num_generations: int = 10,
    results_dir: str = None,
    script_dir: Path = None,
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
        llm_models=["gpt-5-nano","gpt-5-mini"],
        llm_kwargs=dict(
            temperatures=[0.0, 0.5, 0.7],
            max_tokens=8192,
        ),
        embedding_model="text-embedding-3-small",
        init_program_path=str(script_dir / "initial.py") if script_dir else "initial.py",
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


def get_deterministic_seed(*args):
    """
    Creates a unique 32-bit integer seed from any number of arguments.
    Ensures no overlap between runs (e.g., Seed 101 vs 202).
    
    This matches the implementation in rewards_unified_v2.py for consistency.
    
    Args:
        *args: Any number of arguments to combine into a seed
        
    Returns:
        int: 32-bit integer seed
    """
    # Combine all args into a single unique string
    seed_str = "_".join(str(arg) for arg in args)
    # Hash it (SHA256 is robust enough to prevent collision)
    hash_bytes = hashlib.sha256(seed_str.encode('utf-8')).digest()
    # Convert to integer and clip to 32-bit (standard for numpy/random)
    return int.from_bytes(hash_bytes[:4], byteorder='big')


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken if available, otherwise use approximation.
    
    Args:
        text: Text to count tokens for
    
    Returns:
        Approximate token count
    """
    if TIKTOKEN_AVAILABLE:
        try:
            # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo)
            # This is a reasonable approximation for most modern tokenizers
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fall back to approximation if encoding fails
            pass
    
    # Simple approximation: ~4 characters per token (common for English)
    # This is a rough estimate but better than nothing
    return len(text) // 4


def process_single_problem(
    i: int,
    master_seed: int,
    num_generations: int,
    script_dir: Path,
    output_dir: Path,
    api_key: str,
) -> dict:
    """
    Process a single SDS problem through ShinkaEvolve evolution.
    
    This function is designed to be called in parallel across multiple problems.
    Each problem gets its own isolated environment (results directory, database, etc.).
    
    Args:
        i: Problem index (0-based)
        master_seed: Master seed for reproducibility
        num_generations: Number of evolution generations
        script_dir: Path to script directory (for initial.py, evaluate.py)
        output_dir: Base output directory
        api_key: OpenAI API key for trace generation
    
    Returns:
        dict: Dataset record if successful, None if failed
    """
    try:
        # Calculate deterministic seed for this problem using hash-based seeding
        # This ensures:
        # 1. Problem i always gets the same seed regardless of execution order
        # 2. No overlap between different master seeds (e.g., seed 101 problem 102 vs seed 202 problem 1)
        # 3. Matches the approach used in generalization reward for consistency
        if master_seed is not None:
            problem_gen_seed = get_deterministic_seed(master_seed, i)
        else:
            problem_gen_seed = i
        
        # Use the seed to create a deterministic RNG for problem size selection
        rng_for_size = random.Random(problem_gen_seed)
        
        # A. Generate a Problem Instance (deterministic when seed is set)
        # Use same HARD parameters as GRPO dataset generation for consistency
        ptype = "dense" if i % 2 == 0 else "tree"
        n_size = rng_for_size.randint(15, 25)  # Deterministic when seed is set
        
        # Calculate cardinality bounds (matching GRPO dataset generation)
        if ptype == "dense":
            card = (max(3, n_size//3), min(n_size, n_size//2 + 4))
            # HARD: Weak unaries (2.0), Massive interactions (20.0)
            # Mixed signs (0.4/0.4) = Frustration (greedy fails)
            inst = make_dense_instance(
                n=n_size,
                card=card,
                weight_scale=2.0,    # Weak unaries (default is 8.0)
                pair_scale=20.0,    # Strong interactions (default is 6.0)
                pos_pair_frac=0.4,  # Mixed signs = Frustration
                neg_pair_frac=0.4,
                seed=problem_gen_seed
            )
        else:  # tree
            card = (max(3, n_size//3), min(n_size, n_size//2 + 4))
            # HARD: Very weak unaries (1.0), Very strong interactions (25.0)
            inst = make_tree_instance(
                n=n_size,
                card=card,
                weight_scale=1.0,   # Very weak unaries (default is 10.0)
                pair_scale=25.0,    # Very strong interactions (default is 6.0)
                seed=problem_gen_seed
            )
        
        problem_dict = _instance_to_problem(inst, ptype, i)
        prompt_data = sds_render_prompt(problem_dict)
        
        # B. Set up evolution for this problem
        # Create unique results directory for this problem
        problem_results_dir = output_dir / f"problem_{i}"
        problem_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variable for evaluate.py to use (isolated per worker)
        os.environ["SDS_PROBLEM_DATA"] = json.dumps(problem_dict)
        
        # Create configs with absolute paths
        # Ensure results_dir is absolute so it works regardless of working directory
        evo_config = create_sds_evolution_config(
            problem_dict,
            num_generations=num_generations,
            results_dir=str(problem_results_dir.resolve()),
            script_dir=script_dir,
        )
        db_config = create_sds_database_config()
        # EvolutionRunner will prepend results_dir, so just use filename
        db_config.db_path = "evolution_db.sqlite"
        
        job_config = LocalJobConfig(
            eval_program_path=str(script_dir / "evaluate.py")
        )
        
        # C. Run Shinka Evolution
        # Change to script directory so relative paths in evaluate.py work correctly
        # Note: Evolution uses natural randomness to explore diverse solutions
        # and avoid mode collapse. Problem generation is seeded for reproducibility.
        original_cwd = os.getcwd()
        try:
            os.chdir(script_dir)  # Change to script directory for evolution
            evo_runner = EvolutionRunner(
                evo_config=evo_config,
                job_config=job_config,
                db_config=db_config,
                verbose=False,  # Less verbose for batch processing
            )
            evo_runner.run()
        except Exception as e:
            print(f"   ⚠️ Evolution failed for problem {i}: {e}")
            return None
        finally:
            os.chdir(original_cwd)  # Restore original working directory
        
        # D. Extract best solution from database
        # EvolutionRunner creates db at results_dir/db_path (see runner.py line 140)
        # Since results_dir is absolute and db_config.db_path is "evolution_db.sqlite",
        # the final path is: results_dir / "evolution_db.sqlite"
        db_path = problem_results_dir.resolve() / "evolution_db.sqlite"
        
        # Verify the path exists (EvolutionRunner should have created it)
        if not db_path.exists():
            print(f"   ⚠️ Database not found at {db_path} for problem {i}")
            return None
        
        best_code, best_fitness = extract_best_code_from_database(str(db_path))
        
        if not best_code or best_fitness <= 0.01:
            print(f"   ⚠️ Skipping {i}: No valid solution found (fitness: {best_fitness:.4f})")
            return None
        
        # E. Generate Thinking Trace
        # Create trace generator per worker (thread-safe)
        tracer = TraceGenerator(api_key=api_key, model="gpt-5-mini")
        trace = tracer.generate(prompt_data["problem"], best_code)
        
        # F. Format Entry for Student Training
        full_content = (
            f"<think>\n{trace}\n</think>\n\n"
            f"<code>\n{best_code}\n</code>"
        )
        
        # Calculate token counts
        user_tokens = count_tokens(prompt_data["problem"])
        assistant_tokens = count_tokens(full_content)
        total_tokens = user_tokens + assistant_tokens
        
        return {
            "problem_id": prompt_data.get("uuid", f"problem_{i}"),
            "messages": [
                {"role": "user", "content": prompt_data["problem"]},
                {"role": "assistant", "content": full_content}
            ],
            "score": float(best_fitness),
            "type": ptype,
            "mission": json.dumps(problem_dict.get("mission", {})),
            "index": i,  # Keep track of original index for sorting
            "num_tokens": total_tokens,  # Total tokens (user + assistant)
            "num_tokens_user": user_tokens,  # User prompt tokens
            "num_tokens_assistant": assistant_tokens,  # Assistant response tokens
        }
    except Exception as e:
        print(f"   ❌ Error processing problem {i}: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_best_code_from_database(db_path: str) -> tuple[str, float]:
    """
    Extract the best solution code and score from the evolution database.
    
    Args:
        db_path: Absolute path to the database file
    
    Returns:
        (code: str, combined_score: float)
    """
    # Ensure db_path is absolute
    db_path_abs = Path(db_path).resolve()
    if not db_path_abs.exists():
        raise FileNotFoundError(f"Database file not found: {db_path_abs}")
    
    db_config = DatabaseConfig(db_path=str(db_path_abs))
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
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers for problem processing. "
             "If not specified, defaults to number of CPU cores. "
             "Set to 1 for sequential processing."
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_records = []
    
    # Determine number of workers
    if args.workers is None:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = max(1, args.workers)
    
    print(f"🚀 Starting Pipeline: {args.samples} samples")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚡ Using {num_workers} parallel workers")
    
    # Process problems in parallel
    if num_workers == 1:
        # Sequential processing (easier debugging)
        print("🔄 Running in sequential mode (workers=1)")
        for i in tqdm.tqdm(range(args.samples), desc="Generating samples"):
            result = process_single_problem(
                i=i,
                master_seed=master_seed,
                num_generations=args.generations,
                script_dir=script_dir,
                output_dir=output_dir,
                api_key=api_key,
            )
            if result is not None:
                dataset_records.append(result)
    else:
        # Parallel processing
        print(f"⚡ Running in parallel mode ({num_workers} workers)")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all problems
            future_to_index = {
                executor.submit(
                    process_single_problem,
                    i=i,
                    master_seed=master_seed,
                    num_generations=args.generations,
                    script_dir=script_dir,
                    output_dir=output_dir,
                    api_key=api_key,
                ): i
                for i in range(args.samples)
            }
            
            # Collect results as they complete (with progress bar)
            completed_results = {}
            with tqdm.tqdm(total=args.samples, desc="Generating samples") as pbar:
                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        result = future.result()
                        if result is not None:
                            completed_results[i] = result
                    except Exception as e:
                        print(f"   ❌ Problem {i} failed with exception: {e}")
                    finally:
                        pbar.update(1)
            
            # Sort by original index to maintain deterministic order
            dataset_records = [
                completed_results[i] 
                for i in sorted(completed_results.keys())
            ]
    
    print(f"\n✅ Generated {len(dataset_records)} high-quality samples out of {args.samples} attempted.")
    
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
