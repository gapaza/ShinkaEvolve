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

    Improved greedy with precedence closure and small local search.
    """
    # Read input
    input_data = json.load(sys.stdin)
    requirements = input_data.get("requirements", {})
    catalog = input_data.get("catalog", {})

    # Extract constraints
    weights = requirements.get("weights", [])
    interactions_raw = requirements.get("interactions", {})
    cardinality_bounds = requirements.get("cardinality_bounds", [0, len(weights)])
    mutex = requirements.get("mutex", [])
    groups = requirements.get("groups", {})
    precedence = requirements.get("precedence", [])

    n = len(weights)

    # Parse interactions into a dict for quick lookup (make symmetric)
    interactions = {}
    for k, v in interactions_raw.items():
        try:
            parts = [p.strip() for p in k.split(",")]
            if len(parts) >= 2:
                u = int(parts[0]); w = int(parts[1])
                interactions[(u, w)] = float(v)
                interactions[(w, u)] = float(v)
        except Exception:
            # ignore malformed keys
            continue

    # Build mutex map for O(1) checks
    mutex_map = {i: set() for i in range(n)}
    for a, b in mutex:
        if 0 <= a < n and 0 <= b < n:
            mutex_map[a].add(b)
            mutex_map[b].add(a)

    # Build variable->group index (assumes groups is dict: groupname -> [vars])
    var_to_groups = {}
    for g, vars_in_g in groups.items():
        for v in vars_in_g:
            if 0 <= v < n:
                var_to_groups.setdefault(v, set()).add(g)

    # Build precedence graph: prereq_of[j] = list of items that must be present if j selected
    prereq_of = {}
    for a, b in precedence:
        # if b selected then a must be selected
        if 0 <= a < n and 0 <= b < n:
            prereq_of.setdefault(b, []).append(a)

    # Helper: compute closure of prerequisites for an item (recursive)
    def prerequisite_closure(start):
        closure = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            for p in prereq_of.get(cur, []):
                if p not in closure and p != cur:
                    closure.add(p)
                    stack.append(p)
        return closure

    # Objective evaluation
    def objective_of(selection_set):
        total = 0.0
        for i in selection_set:
            total += weights[i] if i < len(weights) else 0.0
        # pairwise interactions
        sel_list = list(selection_set)
        m = len(sel_list)
        for ii in range(m):
            u = sel_list[ii]
            for jj in range(ii + 1, m):
                v = sel_list[jj]
                total += interactions.get((u, v), 0.0)
        return total

    # Feasibility check for adding a set of new items given current selection
    def feasible_to_add(current_sel, to_add, max_card):
        # Check cardinality
        if len(current_sel | to_add) > max_card:
            return False
        # Check mutex: no pair (x,y) both in new selection or between new and current
        for x in to_add:
            # against current
            for y in mutex_map.get(x, ()):
                if y in current_sel:
                    return False
            # within to_add
            for y in mutex_map.get(x, ()):
                if y in to_add:
                    return False
        # Check group: at most one per group
        # Build counts per group from current and to_add
        group_counts = {}
        for v in current_sel:
            for g in var_to_groups.get(v, ()):
                group_counts[g] = group_counts.get(g, 0) + 1
        for v in to_add:
            for g in var_to_groups.get(v, ()):
                group_counts[g] = group_counts.get(g, 0) + 1
                if group_counts[g] > 1:
                    return False
        # Precedence: ensure for any item in to_add, its prerequisites are included in current_sel or to_add
        for v in list(to_add):
            needed = prerequisite_closure(v)
            if not needed.issubset(current_sel | to_add):
                return False
        return True

    # Compute marginal gain of adding item i (including its prerequisites) given current_sel
    def marginal_gain(current_sel, i):
        if i in current_sel:
            return -1e9, set()  # already present, no gain
        # compute closure (prereqs)
        req = prerequisite_closure(i) | {i}
        # Determine which of these are truly new
        new_items = set(x for x in req if x not in current_sel)
        if not new_items:
            return -1e9, set()
        # Check feasibility wrt constraints (we'll check cardinality externally)
        # For scoring, compute added weight + interactions with current selection and internal interactions
        added = 0.0
        for u in new_items:
            added += weights[u] if u < len(weights) else 0.0
            for v in current_sel:
                added += interactions.get((u, v), 0.0)
        # interactions among new_items (pairwise)
        new_list = list(new_items)
        for a_idx in range(len(new_list)):
            u = new_list[a_idx]
            for b_idx in range(a_idx + 1, len(new_list)):
                v = new_list[b_idx]
                added += interactions.get((u, v), 0.0)
        return added, new_items

    max_card = cardinality_bounds[1] if len(cardinality_bounds) > 1 else n
    min_card = cardinality_bounds[0] if len(cardinality_bounds) > 0 else 0

    selected_set = set()

    # Main greedy loop: at each step choose item (with its prereqs) that gives best positive marginal and is feasible
    while True:
        best_gain = 0.0
        best_item = None
        best_new_set = None
        for i in range(n):
            if i in selected_set:
                continue
            gain, new_items = marginal_gain(selected_set, i)
            if not new_items:
                continue
            # feasibility checks (mutex, groups, cardinality)
            if not feasible_to_add(selected_set, new_items, max_card):
                continue
            if gain > best_gain + 1e-9:
                best_gain = gain
                best_item = i
                best_new_set = new_items
        if best_item is None:
            break
        # Add best_new_set to selection
        selected_set |= best_new_set

    # If we are below min_card, add best feasible items (ignore positive gain requirement)
    if len(selected_set) < min_card:
        candidates = []
        for i in range(n):
            if i in selected_set:
                continue
            gain, new_items = marginal_gain(selected_set, i)
            if not new_items:
                continue
            # check feasibility w.r.t max cardinality
            if not feasible_to_add(selected_set, new_items, max_card):
                continue
            candidates.append((gain, i, new_items))
        # sort by gain descending, fall back to weight if ties
        candidates.sort(key=lambda x: (-x[0], -max(weights[x[1]] if x[1] < len(weights) else 0.0, 0.0)))
        for gain, i, new_items in candidates:
            if len(selected_set) >= min_card:
                break
            if feasible_to_add(selected_set, new_items, max_card):
                selected_set |= new_items

    # Local improvement: try single swaps (replace one selected with one candidate and its prereqs)
    improved = True
    iter_limit = 5
    iters = 0
    while improved and iters < iter_limit:
        improved = False
        iters += 1
        base_obj = objective_of(selected_set)
        # Try each unselected candidate
        unselected = [i for i in range(n) if i not in selected_set]
        for cand in unselected:
            gain, cand_new = marginal_gain(selected_set, cand)
            if not cand_new:
                continue
            # If adding directly is feasible and increases obj, do it
            if feasible_to_add(selected_set, cand_new, max_card):
                new_obj = base_obj + gain
                if new_obj > base_obj + 1e-9:
                    selected_set |= cand_new
                    improved = True
                    base_obj = new_obj
                    break
            # Otherwise attempt swap: remove one selected item s and add cand_new
            for s in list(selected_set):
                trial_sel = set(selected_set)
                trial_sel.remove(s)
                # Ensure removing s doesn't violate precedence (if some remaining item requires s)
                violated = False
                for rem in trial_sel:
                    needed = prerequisite_closure(rem)
                    if s in needed:
                        violated = True
                        break
                if violated:
                    continue
                # Now try to add cand_new to trial_sel
                if not feasible_to_add(trial_sel, cand_new, max_card):
                    continue
                trial_obj = objective_of(trial_sel | cand_new)
                if trial_obj > base_obj + 1e-9:
                    selected_set = trial_sel | cand_new
                    improved = True
                    base_obj = trial_obj
                    break
            if improved:
                break

    # Final safety: if still below min_card, fill with any feasible items greedily
    if len(selected_set) < min_card:
        for i in range(n):
            if len(selected_set) >= min_card:
                break
            if i in selected_set:
                continue
            _, new_items = marginal_gain(selected_set, i)
            if not new_items:
                continue
            if feasible_to_add(selected_set, new_items, max_card):
                selected_set |= new_items

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
