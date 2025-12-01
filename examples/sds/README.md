# SDS (Synergistic Dependency Selection) Example

This example demonstrates how to use ShinkaEvolve for the SDS combinatorial optimization problem and generate high-quality SFT datasets.

## Files

- **`initial.py`**: Starting solution with `EVOLVE-BLOCK` markers. The `solve_sds()` function will be evolved.
- **`evaluate.py`**: Evaluation script using ShinkaEvolve's `run_shinka_eval` pattern. Validates feasibility and calculates fitness.
- **`run_evo.py`**: Standard evolution runner for solving a single SDS problem (same pattern as other examples).
- **`run_sds_pipeline.py`**: **Optional** - Specialized pipeline for generating SFT training datasets. Only needed if you want to create datasets with reasoning traces for model training.

## Usage

### Standard Evolution (Single Problem)

For solving a single SDS problem, use `run_evo.py` just like other ShinkaEvolve examples:

```bash
# Set problem data via environment variable
export SDS_PROBLEM_DATA='{"requirements": {...}, "catalog": {...}}'

# Run evolution
cd examples/sds
python run_evo.py
```

This follows the standard ShinkaEvolve pattern: `initial.py` + `evaluate.py` + `run_evo.py`.

### Dataset Generation (Optional)

**Only needed if you want to generate SFT training datasets.** The pipeline script automates:
- Generating multiple problem instances
- Running evolution for each
- Extracting best solutions
- Generating reasoning traces
- Formatting training data

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

