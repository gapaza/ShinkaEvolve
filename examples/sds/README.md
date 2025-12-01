# SDS (Synergistic Dependency Selection) Example

This example demonstrates how to use ShinkaEvolve for the SDS combinatorial optimization problem and generate high-quality SFT datasets.

## Files

- **`initial.py`**: Starting solution with `EVOLVE-BLOCK` markers. The `solve_sds()` function will be evolved.
- **`evaluate.py`**: Evaluation script using ShinkaEvolve's `run_shinka_eval` pattern. Validates feasibility and calculates fitness.
- **`run_evo.py`**: Example runner for single problem evolution using `EvolutionRunner`.
- **`sds_task.py`**: Deprecated (kept for reference only).

## Usage

### Single Problem Evolution

```bash
# Set problem data via environment variable
export SDS_PROBLEM_DATA='{"requirements": {...}, "catalog": {...}}'

# Run evolution
cd examples/sds
python run_evo.py
```

### Batch Dataset Generation

Use the pipeline script:

```bash
# From examples/sds directory
cd examples/sds
python run_sds_pipeline.py \
    --samples 100 \
    --generations 10 \
    --output_dir sds_dataset_output \
    --push_to YourOrg/SDS-Dataset  # Optional: push to HuggingFace

# Or from project root
python examples/sds/run_sds_pipeline.py --samples 100 --generations 10
```

## Integration with ShinkaEvolve

This example properly integrates with ShinkaEvolve's architecture:

- ✅ Uses `EvolutionRunner` with `EvolutionConfig`, `JobConfig`, `DatabaseConfig`
- ✅ Follows the `evaluate.py` pattern with `run_shinka_eval`
- ✅ Uses `EVOLVE-BLOCK` markers in `initial.py`
- ✅ Leverages database, islands, archives, and all ShinkaEvolve features
- ✅ Extracts best solutions from the database after evolution

## Problem Format

The SDS problem expects:
- **Input** (stdin or parameter): JSON with `requirements` and `catalog`
- **Output** (stdout or return): JSON with `selection.variables` list

The evaluator validates:
- Cardinality bounds
- Mutex constraints
- Group constraints
- Precedence constraints
- Objective maximization (weights + interactions)

