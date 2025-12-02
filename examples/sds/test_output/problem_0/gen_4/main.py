"""
Initial solution for Synergistic Dependency Selection (SDS) problem.
The code in EVOLVE-BLOCK will be evolved by ShinkaEvolve.
"""

import json
import sys
import contextlib

# EVOLVE-BLOCK-START
def solve_sds():
    """
    Solve the Synergistic Dependency Selection problem.

    Reads from stdin: JSON with "requirements" and "catalog"
    Outputs to stdout: JSON with "selection" containing "variables" list

    Improved greedy with precedence-closure and small local search.
    """
    # Read input
    input_data = json.load(sys.stdin)
    requirements = input_data.get("requirements", {})
    catalog = input_data.get("catalog", {})

    # Extract constraints
    weights = requirements.get("weights", [])
    raw_interactions = requirements.get("interactions", {})
    cardinality_bounds = requirements.get("cardinality_bounds", [0, len(weights)])
    mutex = requirements.get("mutex", [])
    groups = requirements.get("groups", {})
    precedence = requirements.get("precedence", [])

    n = len(weights)

    # Parse interactions into a symmetric dict for quick lookup
    interactions = {}
    for k, v in raw_interactions.items():
        try:
            u, w = map(int, k.split(","))
            interactions[(u, w)] = float(v)
            interactions[(w, u)] = float(v)
        except Exception:
            # ignore malformed keys
            continue

    # Build precedence map: if (i, j) in precedence then selecting j requires i
    pred_map = {}  # pred_map[j] = set([i,...])
    for pair in precedence:
        try:
            i, j = pair
            pred_map.setdefault(j, set()).add(i)
        except Exception:
            continue

    # Compute transitive closure for a given item (its required predecessors)
    def closure_of(item):
        stack = [item]
        closure = set()
        while stack:
            cur = stack.pop()
            if cur in closure:
                continue
            closure.add(cur)
            for p in pred_map.get(cur, ()):
                if p not in closure:
                    stack.append(p)
        return closure

    # Objective for a set (weights + interactions among all unordered pairs)
    def objective_of(varset):
        s = 0.0
        for i in varset:
            if 0 <= i < n:
                s += weights[i]
        # pairwise interactions (count each pair once)
        seen = set()
        for i in varset:
            for j in varset:
                if j <= i:
                    continue
                if (i, j) in interactions:
                    s += interactions[(i, j)]
        return s

    # Check mutex and group feasibility for union selected U to_add
    def feasible_union(selected_set, to_add):
        union = set(selected_set) | set(to_add)
        # cardinality upper bound
        if len(union) > cardinality_bounds[1]:
            return False
        # mutex pairs
        for a, b in mutex:
            if a in union and b in union:
                return False
        # group constraints: at most one per group
        for grp_vars in groups.values():
            cnt = 0
            for v in grp_vars:
                if v in union:
                    cnt += 1
                    if cnt > 1:
                        return False
        # precedence: every selected element must have its preds in union
        for j in list(union):
            for p in pred_map.get(j, ()):
                if p not in union:
                    return False
        return True

    # Check if an item is removable (no selected item depends on it)
    # i.e., no other selected has i in its transitive predecessors
    # Precompute reverse closure check by scanning selected set when needed
    def removable(selected_set, item):
        for j in selected_set:
            if j == item:
                continue
            # if item is in closure of j, cannot remove
            if item in closure_of(j):
                return False
        return True

    # Greedy selection using closure-aware delta objective
    selected_set = set()
    # Precompute candidate base scores to order attempts (weight + small interaction potential)
    base_scores = []
    for i in range(n):
        score = weights[i]
        # add half of absolute interactions as potential
        for (u, v), val in interactions.items():
            if u == i or v == i:
                score += abs(val) * 0.25
        base_scores.append((score, i))
    base_scores.sort(reverse=True)
    candidates = [i for _, i in base_scores]

    # Iteratively add the best feasible closure that gives positive objective delta
    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_add = None
        cur_obj = objective_of(selected_set)
        for i in candidates:
            if i in selected_set:
                continue
            cl = closure_of(i)
            to_add = cl - selected_set
            if not to_add:
                continue
            if not feasible_union(selected_set, to_add):
                continue
            new_obj = objective_of(selected_set | to_add)
            delta = new_obj - cur_obj
            if delta > best_delta + 1e-9:
                best_delta = delta
                best_add = to_add
        if best_add:
            selected_set |= set(best_add)
            improved = True
        # stop if hit upper bound
        if len(selected_set) >= cardinality_bounds[1]:
            break

    # Local improvement: try beneficial removals and additions until no change
    changed = True
    while changed:
        changed = False
        cur_obj = objective_of(selected_set)
        # Attempt removals
        for i in list(selected_set):
            if not removable(selected_set, i):
                continue
            new_set = set(selected_set)
            new_set.remove(i)
            if not feasible_union(set(), new_set):  # new_set should be self-consistent
                continue
            new_obj = objective_of(new_set)
            if new_obj > cur_obj + 1e-9:
                selected_set = new_set
                changed = True
                break
        if changed:
            continue
        # Attempt any single-item addition (with closure) that improves objective
        cur_obj = objective_of(selected_set)
        best_delta = 0.0
        best_add = None
        for i in range(n):
            if i in selected_set:
                continue
            cl = closure_of(i)
            to_add = cl - selected_set
            if not to_add:
                continue
            if not feasible_union(selected_set, to_add):
                continue
            new_obj = objective_of(selected_set | to_add)
            delta = new_obj - cur_obj
            if delta > best_delta + 1e-9:
                best_delta = delta
                best_add = to_add
        if best_add:
            selected_set |= set(best_add)
            changed = True

    # Ensure minimum cardinality: add best feasible candidates (even if delta <= 0) until min satisfied
    while len(selected_set) < cardinality_bounds[0]:
        best_choice = None
        best_delta = None
        cur_obj = objective_of(selected_set)
        for i in range(n):
            if i in selected_set:
                continue
            cl = closure_of(i)
            to_add = cl - selected_set
            if not to_add:
                continue
            if not feasible_union(selected_set, to_add):
                continue
            new_obj = objective_of(selected_set | to_add)
            delta = new_obj - cur_obj
            if best_choice is None or delta > best_delta:
                best_choice = to_add
                best_delta = delta
        if best_choice:
            selected_set |= set(best_choice)
        else:
            # No feasible addition to reach min (shouldn't happen in feasible instances), break to avoid infinite loop
            break

    result = {
        "selection": {
            "variables": sorted(selected_set)
        }
    }
    print(json.dumps(result))
# EVOLVE-BLOCK-END


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
