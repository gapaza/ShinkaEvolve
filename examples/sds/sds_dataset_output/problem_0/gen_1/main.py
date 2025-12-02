"""
Initial solution for Synergistic Dependency Selection (SDS) problem.
The code in EVOLVE-BLOCK will be evolved by ShinkaEvolve.
"""

import json
import sys
import contextlib

# EVOLVE-BLOCK-START# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_sds(problem_data=None):
    """
    Main function called by the evaluator.
    Can accept problem_data as parameter or read from stdin.
    Returns JSON string (for ShinkaEvolve) or prints to stdout (for direct execution).
    """
    import sys
    import json
    import io
    
    if problem_data is None:
        # Read from stdin (for compatibility)
        input_data = json.load(sys.stdin)
    else:
        # Use provided problem data
        input_data = problem_data
    
    # Capture stdout from solve_sds
    stdout_capture = io.StringIO()
    old_stdin = sys.stdin
    
    try:
        # Create mock stdin
        sys.stdin = io.StringIO(json.dumps(input_data))
        with contextlib.redirect_stdout(stdout_capture):
            solve_sds()
    finally:
        sys.stdin = old_stdin
    
    result_str = stdout_capture.getvalue()
    return result_str

