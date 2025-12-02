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
    Improved solver for the Synergistic Dependency Selection problem.

    - Reads JSON from stdin with "requirements" and "catalog".
    - Produces JSON to stdout with "selection": {"variables": [...] }.
    - Uses a precedence-aware marginal-gain greedy algorithm that:
        * when adding an item, also includes its required predecessors (closure),
        * computes the marginal objective (weights + pairwise interactions),
        * checks mutex/group/cardinality feasibility for the closure before adding.
    - Adds a lightweight one-step lookahead: when evaluating a candidate closure,
      also consider the best single subsequent closure that would be feasible after
      adding the candidate. A discounted portion of that subsequent gain is added
      to the candidate score to prefer moves that enable strong follow-ups.
    """
    import re

    # Read input
    input_data = json.load(sys.stdin)
    requirements = input_data.get("requirements", {})
    catalog = input_data.get("catalog", {})

    # Extract constraints
    weights = list(requirements.get("weights", []))
    interactions_raw = requirements.get("interactions", {})
    cardinality_bounds = requirements.get("cardinality_bounds", [0, len(weights)])
    min_card, max_card = (cardinality_bounds + [len(weights), len(weights)])[:2]
    mutex = [tuple(m) for m in requirements.get("mutex", [])]
    groups = requirements.get("groups", {})
    precedence = requirements.get("precedence", [])

    n = len(weights)

    # Parse interactions into dict of tuple->float for quick lookup (undirected)
    interactions = {}
    for k, v in interactions_raw.items():
        try:
            if isinstance(k, str):
                # allow "u,v" or "u v"
                parts = re.split(r'\s*,\s*|\s+', k.strip())
                if len(parts) >= 2:
                    u = int(parts[0]); w = int(parts[1])
                else:
                    continue
            elif isinstance(k, (list, tuple)) and len(k) >= 2:
                u = int(k[0]); w = int(k[1])
            else:
                continue
            interactions[(u, w)] = float(v)
            interactions[(w, u)] = float(v)
        except Exception:
            # skip malformed keys
            continue

    # Build precedence graph: if (p, j) then p is a prerequisite of j
    preds = {i: set() for i in range(n)}
    succs = {i: set() for i in range(n)}
    for entry in precedence:
        try:
            a, b = entry
            a = int(a); b = int(b)
            preds[b].add(a)
            succs[a].add(b)
        except Exception:
            continue

    # Precompute ancestors (transitive closure of preds) using DFS/memoization
    ancestor_cache = {}

    def get_ancestors(i):
        if i in ancestor_cache:
            return ancestor_cache[i].copy()
        res = set()
        stack = [i]
        while stack:
            cur = stack.pop()
            for p in preds.get(cur, []):
                if p not in res:
                    res.add(p)
                    stack.append(p)
        ancestor_cache[i] = res.copy()
        return res

    # Precompute closures (ancestors + self) and cache them for fast reuse
    closure_cache = {i: get_ancestors(i).union({i}) for i in range(n)}

    # Map variable -> group (assume at most one group membership; if multiple, first wins)
    var_to_group = {}
    for gname, vars_list in groups.items():
        for v in vars_list:
            try:
                var_to_group[int(v)] = gname
            except Exception:
                pass

    # Helper: check feasibility of adding a closure set given current selected_set
    def feasible_closure(closure_set, selected_set):
        # Cardinality
        if len(selected_set) + len(closure_set - selected_set) > max_card:
            return False
        # Mutex: none of closure can conflict with each other or with selected_set
        for a, b in mutex:
            if a in closure_set and b in closure_set:
                return False
            if (a in closure_set and b in selected_set) or (b in closure_set and a in selected_set):
                return False
        # Group: ensure at most one per group overall
        # Build per-group counts
        group_counts = {}
        for v in selected_set:
            g = var_to_group.get(v)
            if g is not None:
                group_counts[g] = group_counts.get(g, 0) + 1
        for v in closure_set:
            g = var_to_group.get(v)
            if g is not None:
                group_counts[g] = group_counts.get(g, 0) + 1
                if group_counts[g] > 1:
                    return False
        # Precedence: closure itself must include ancestors for each element (or they are already selected)
        for v in closure_set:
            if not get_ancestors(v).issubset(closure_set.union(selected_set)):
                return False
        return True

    # Compute marginal gain of adding closure_set (items that will be newly included)
    def marginal_gain(closure_set, selected_set):
        new_items = set(closure_set) - set(selected_set)
        if not new_items:
            return 0.0
        gain = 0.0
        for i in new_items:
            if 0 <= i < len(weights):
                gain += float(weights[i])
        # interactions among new items
        new_list = list(new_items)
        for a in range(len(new_list)):
            for b in range(a + 1, len(new_list)):
                gain += interactions.get((new_list[a], new_list[b]), 0.0)
        # interactions between new items and already selected
        for i in new_items:
            for j in selected_set:
                gain += interactions.get((i, j), 0.0)
        return gain

    # Greedy loop: at each step try to add an item (and its ancestors) with max marginal gain
    # Enhanced with small one-step lookahead: for each candidate closure consider the best
    # subsequent single-closure gain and add a discounted fraction to the candidate score.
    selected_set = set()
    improved = True
    LOOKAHEAD_DISCOUNT = 0.5  # weight of the best next-step gain when scoring current candidate
    while len(selected_set) < max_card and improved:
        improved = False
        best_score = float("-inf")
        best_gain = float("-inf")
        best_closure = None
        # Consider candidates not already selected
        for i in range(n):
            if i in selected_set:
                continue
            # closure includes i and its ancestors (cached)
            closure = closure_cache.get(i, {i})
            # Only consider closures that are feasible
            if not feasible_closure(closure, selected_set):
                continue
            gain = marginal_gain(closure, selected_set)
            # quick capacity pruning: if closure would add more items than remaining slots, skip
            if len(closure - selected_set) > (max_card - len(selected_set)):
                continue
            # Compute one-step lookahead: best feasible single closure after adding this closure
            lookahead_best = 0.0
            if len(selected_set) + len(closure - selected_set) < max_card:
                # simulate new selected set
                sim_selected = selected_set.union(closure)
                # find best single candidate (could be more expensive, so limit exploration)
                # We will search all but with a cheap break if we find strong positive gains.
                for j in range(n):
                    if j in sim_selected:
                        continue
                    closure_j = closure_cache.get(j, {j})
                    if not feasible_closure(closure_j, sim_selected):
                        continue
                    gain_j = marginal_gain(closure_j, sim_selected)
                    if gain_j > lookahead_best:
                        lookahead_best = gain_j
                        # small optimization: if it's very large, we can accept it
                        if lookahead_best > abs(gain) * 2 + 1e-9:
                            break
            total_score = gain + LOOKAHEAD_DISCOUNT * max(0.0, lookahead_best)
            # Prefer closures that add more items when scores equal (helps reach min_card)
            if (total_score > best_score) or (abs(total_score - best_score) < 1e-9 and (best_closure is None or len(closure - selected_set) > len(best_closure - selected_set))):
                best_score = total_score
                best_gain = gain
                best_closure = closure
        # If we found a non-negative-gain move, take it
        if best_closure is not None and best_gain > -1e-12:
            to_add = set(best_closure) - selected_set
            if to_add:
                selected_set.update(to_add)
                improved = True
        else:
            # No non-negative gain move; stop greedy improvements
            break

    # If we still haven't met minimum cardinality, greedily add best feasible closures regardless of gain
    # Use same closure cache and prefer closures that may enable good follow-ups.
    while len(selected_set) < min_card:
        best_score = float("-inf")
        best_gain = float("-inf")
        best_closure = None
        for i in range(n):
            if i in selected_set:
                continue
            closure = closure_cache.get(i, {i})
            if not feasible_closure(closure, selected_set):
                continue
            gain = marginal_gain(closure, selected_set)
            # lookahead as before (but smaller budget)
            lookahead_best = 0.0
            if len(selected_set) + len(closure - selected_set) < max_card:
                sim_selected = selected_set.union(closure)
                for j in range(n):
                    if j in sim_selected:
                        continue
                    closure_j = closure_cache.get(j, {j})
                    if not feasible_closure(closure_j, sim_selected):
                        continue
                    gain_j = marginal_gain(closure_j, sim_selected)
                    if gain_j > lookahead_best:
                        lookahead_best = gain_j
            total_score = gain + LOOKAHEAD_DISCOUNT * max(0.0, lookahead_best)
            if (total_score > best_score) or (abs(total_score - best_score) < 1e-9 and (best_closure is None or len(closure - selected_set) > len(best_closure - selected_set))):
                best_score = total_score
                best_gain = gain
                best_closure = closure
        if best_closure is None:
            # No feasible closure remains; break to avoid infinite loop
            break
        selected_set.update(set(best_closure) - selected_set)

    # Final trim if over max_card (shouldn't happen due to checks) - keep highest-weight items
    if len(selected_set) > max_card:
        # score items by marginal contribution relative to selected (approx): weight + interactions with others
        def item_score(i):
            s = float(weights[i]) if i < len(weights) else 0.0
            for j in selected_set:
                if j == i:
                    continue
                s += interactions.get((i, j), 0.0)
            return s
        sorted_items = sorted(selected_set, key=item_score, reverse=True)
        selected_set = set(sorted_items[:max_card])

    result = {
        "selection": {
            "variables": sorted(int(x) for x in selected_set)
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