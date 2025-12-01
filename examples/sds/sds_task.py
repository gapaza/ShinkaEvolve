import json
import sys
import io
import contextlib
import numpy as np
from typing import Dict, Any, Tuple

# We assume shinka.core.task is where the base class lives based on your file structure
from shinka.core.task import Task 

# Minimal mock for Syndeopt if not installed in the shinka env
# You should ideally install syndeopt in the env, but this makes it portable
try:
    from syndeopt.core.instance import SDSInstance
except ImportError:
    pass

class SDSTask(Task):
    """
    ShinkaEvolve Task for Synergistic Dependency Selection (SDS).
    """
    def __init__(self, problem_payload: Dict[str, Any]):
        """
        problem_payload: The full dict from your gen_sds_dataset.py output
        """
        super().__init__()
        self.problem_data = problem_payload
        self.requirements = problem_payload.get("requirements", {})
        self.catalog = problem_payload.get("catalog", {})
        
        # Pre-serialize input to speed up evaluation
        self.stdin_str = json.dumps({
            "requirements": self.requirements,
            "catalog": self.catalog
        })

    def evaluate(self, code: str) -> float:
        """
        Executes code against the SDS constraints.
        Returns fitness (0.0 to 1.0).
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # 1. Sandbox Execution
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture), \
                 self._mock_stdin(self.stdin_str):
                
                # Create isolated scope
                local_scope = {}
                exec(code, {'__name__': '__main__', 'print': print}, local_scope)
            
            # 2. Output Parsing
            output = stdout_capture.getvalue()
            if not output.strip():
                return 0.0
                
            try:
                out_json = json.loads(output)
                selection = out_json.get("selection", {}).get("variables", [])
            except json.JSONDecodeError:
                return 0.0

            # 3. Validation (Simplified logic for evolution speed)
            # Check Feasibility
            valid, violation_count = self._check_feasibility(selection)
            if not valid:
                # Soft penalty to guide evolution toward feasibility
                return 0.1 / (1 + violation_count)
            
            # Check Optimality
            raw_score = self._calculate_score(selection)
            
            # Normalize score for fitness (0-1 range helps Shinka)
            # Estimate max score as sum of positive weights (rough upper bound)
            max_est = sum(max(0, w) for w in self.requirements.get("weights", [])) 
            max_est += sum(max(0, v) for v in self.requirements.get("interactions", {}).values())
            if max_est == 0: max_est = 1.0
            
            fitness = max(0.1, min(1.0, raw_score / max_est))
            return fitness

        except Exception:
            return 0.0

    @contextlib.contextmanager
    def _mock_stdin(self, text):
        old_stdin = sys.stdin
        sys.stdin = io.StringIO(text)
        try: yield
        finally: sys.stdin = old_stdin

    def _check_feasibility(self, selection):
        """Minimal reimplementation of SDS constraints to avoid heavy dependencies."""
        sel_set = set(selection)
        violations = 0
        
        # A. Cardinality
        L, U = self.requirements.get("cardinality_bounds", [0, 9999])
        if not (L <= len(sel_set) <= U):
            violations += 1
            
        # B. Mutex
        for a, b in self.requirements.get("mutex", []):
            if a in sel_set and b in sel_set:
                violations += 1
                
        # C. Groups
        for grp_vars in self.requirements.get("groups", {}).values():
            if len(set(grp_vars).intersection(sel_set)) > 1:
                violations += 1
        
        # D. Precedence (i -> j means if j selected, i must be selected)
        for i, j in self.requirements.get("precedence", []):
            if j in sel_set and i not in sel_set:
                violations += 1
                
        return (violations == 0), violations

    def _calculate_score(self, selection):
        sel_set = set(selection)
        total = 0.0
        
        # Unaries
        weights = self.requirements.get("weights", [])
        for i in selection:
            if i < len(weights): total += weights[i]
            
        # Pairwise
        interactions = self.requirements.get("interactions", {})
        for k, v in interactions.items():
            try:
                u, w = map(int, k.split(","))
                if u in sel_set and w in sel_set:
                    total += v
            except: pass
            
        return total