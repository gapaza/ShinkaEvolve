"""
Initial solution for Synergistic Dependency Selection (SDS) problem.
The code in EVOLVE-BLOCK will be evolved by ShinkaEvolve.
"""

import json
import sys
import contextlib

# EVOLVE-BLOCK-START
Modular solver for Synergistic Dependency Selection (SDS).
The EVOLVE block contains the evolved solver implementation.
"""
import json
import sys

def solve_sds():
    import re
    from collections import defaultdict, deque

    # ---------- Parsing ----------
    data = json.load(sys.stdin)
    requirements = data.get("requirements", {}) or {}
    catalog = data.get("catalog", {}) or {}

    weights = list(requirements.get("weights", []) or [])
    raw_interactions = requirements.get("interactions", {}) or {}
    card_bounds = requirements.get("cardinality_bounds", [0, len(weights)]) or [0, len(weights)]
    min_card = int(card_bounds[0]) if len(card_bounds) > 0 else 0
    max_card = int(card_bounds[1]) if len(card_bounds) > 1 else len(weights)
    mutex_list = requirements.get("mutex", []) or []
    groups = requirements.get("groups", {}) or {}
    precedence = requirements.get("precedence", []) or []

    n = len(weights)

    # ---------- Preprocessing ----------
    # Build interaction adjacency: interactions[u][v] = value
    interactions = defaultdict(dict)
    def parse_interaction_key(k):
        try:
            if isinstance(k, str):
                parts = re.split(r'\s*,\s*|\s+', k.strip())
                if len(parts) >= 2:
                    return int(parts[0]), int(parts[1])
            elif isinstance(k, (list, tuple)) and len(k) >= 2:
                return int(k[0]), int(k[1])
        except Exception:
            return None
        return None

    for k, v in raw_interactions.items():
        parsed = parse_interaction_key(k)
        try:
            val = float(v)
        except Exception:
            continue
        if parsed is None:
            continue
        u, w = parsed
        if 0 <= u < n and 0 <= w < n:
            interactions[u][w] = interactions[u].get(w, 0.0) + val
            interactions[w][u] = interactions[w].get(u, 0.0) + val

    # Mutex map
    mutex_map = defaultdict(set)
    for pair in mutex_list:
        try:
            a, b = pair
            a = int(a); b = int(b)
            if 0 <= a < n and 0 <= b < n:
                mutex_map[a].add(b); mutex_map[b].add(a)
        except Exception:
            continue

    # Group map: variable -> group (first membership wins)
    var_to_group = {}
    for gname, vars_list in groups.items():
        for v in vars_list:
            try:
                idx = int(v)
                if 0 <= idx < n and idx not in var_to_group:
                    var_to_group[idx] = str(gname)
            except Exception:
                continue

    # Precedence graph
    preds = defaultdict(set)
    succs = defaultdict(set)
    for entry in precedence:
        try:
            a, b = entry
            a = int(a); b = int(b)
            if 0 <= a < n and 0 <= b < n:
                preds[b].add(a)
                succs[a].add(b)
        except Exception:
            continue

    # Compute transitive ancestors (closure of preds) using iterative DFS per node with memoization
    ancestor_cache = {}
    def compute_ancestors(i):
        if i in ancestor_cache:
            return ancestor_cache[i]
        res = set()
        stack = list(preds.get(i, []))
        seen = set(stack)
        while stack:
            cur = stack.pop()
            res.add(cur)
            for p in preds.get(cur, ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        ancestor_cache[i] = res
        return res

    closures = [None] * n
    closure_sizes = [0] * n
    closure_base_weight = [0.0] * n  # sum of weights in closure (may include out-of-range indices protection)
    for i in range(n):
        anc = compute_ancestors(i)
        c = set(anc)
        c.add(i)
        closures[i] = c
        closure_sizes[i] = len(c)
        s = 0.0
        for v in c:
            if 0 <= v < len(weights):
                try:
                    s += float(weights[v])
                except Exception:
                    pass
        closure_base_weight[i] = s

    # Utility functions for interactions and scoring
    def interaction(u, v):
        return interactions.get(u, {}).get(v, 0.0)

    def total_score(selected_set):
        # Sum weights
        s = 0.0
        for i in selected_set:
            if 0 <= i < len(weights):
                try:
                    s += float(weights[i])
                except Exception:
                    pass
        # Pairwise interactions (each pair counted once)
        sel_list = list(selected_set)
        L = len(sel_list)
        for a in range(L):
            u = sel_list[a]
            for b in range(a+1, L):
                v = sel_list[b]
                s += interaction(u, v)
        return s

    # Feasibility check for closure given a base selected_set
    def feasible_closure(closure_set, selected_set):
        # new items to add
        new_items = closure_set - selected_set
        # cardinality
        if len(selected_set) + len(new_items) > max_card:
            return False
        # mutex: none of closure can conflict with selected or each other
        for v in new_items:
            for c in mutex_map.get(v, ()):
                if c in selected_set or c in new_items:
                    return False
        # group counts
        group_counts = {}
        for v in selected_set:
            g = var_to_group.get(v)
            if g is not None:
                group_counts[g] = group_counts.get(g, 0) + 1
        for v in new_items:
            g = var_to_group.get(v)
            if g is not None:
                group_counts[g] = group_counts.get(g, 0) + 1
                if group_counts[g] > 1:
                    return False
        # precedence: closure must provide ancestors for each of its elements or they already in selected_set
        for v in closure_set:
            for a in preds.get(v, ()):
                if a not in closure_set and a not in selected_set:
                    return False
        return True

    # Compute marginal_gain for adding closure to selected_set (exact)
    def marginal_gain(closure_set, selected_set):
        new_items = closure_set - selected_set
        if not new_items:
            return 0.0
        gain = 0.0
        for i in new_items:
            if 0 <= i < len(weights):
                try:
                    gain += float(weights[i])
                except Exception:
                    pass
        # interactions among new items
        new_list = list(new_items)
        L = len(new_list)
        for a in range(L):
            u = new_list[a]
            for b in range(a+1, L):
                v = new_list[b]
                gain += interaction(u, v)
        # interactions between new_items and selected_set
        for u in new_items:
            for v in selected_set:
                gain += interaction(u, v)
        return gain

    # ---------- Greedy selection (closure-based) ----------
    selected_set = set()
    current_score = 0.0

    LOOKAHEAD_DISCOUNT = 0.45  # a tuned discount factor

    # Pre-rank candidates by a simple heuristic to limit expensive lookahead:
    # heuristic = closure_base_weight + sum(max interaction with closure) approx
    # We'll rank on-the-fly each iteration based on marginal_gain estimate without lookahead.
    while len(selected_set) < max_card:
        best_score = float("-inf")
        best_closure = None
        best_gain = 0.0

        # Evaluate all candidates but compute a cheap marginal first, then do lookahead for top-K
        candidate_estimates = []
        for i in range(n):
            if i in selected_set:
                continue
            closure = closures[i]
            # Quick capacity pruning
            new_items = closure - selected_set
            if len(selected_set) + len(new_items) > max_card:
                continue
            if not feasible_closure(closure, selected_set):
                continue
            # cheap marginal (weights sum + approx interactions with selected): use marginal_gain
            mg = marginal_gain(closure, selected_set)
            candidate_estimates.append((mg, i, closure))

        if not candidate_estimates:
            break

        # Sort descending by mg and prefer larger closures when tie
        candidate_estimates.sort(key=lambda x: (x[0], len(x[2] - selected_set)), reverse=True)

        # Limit lookahead to top_k candidates to control cost
        top_k = min(12, len(candidate_estimates))
        for idx in range(top_k):
            mg, i, closure = candidate_estimates[idx]
            # compute lookahead: best single closure after simulating adding this closure
            total_score_candidate = mg
            # if there is still capacity for one more closure
            if len(selected_set) + len(closure - selected_set) < max_card:
                sim_selected = selected_set.union(closure)
                best_follow = 0.0
                # consider a smaller subset of candidates for follow-up (next best excluding those in sim_selected)
                follow_k = min(10, n)
                # follow-up list reuse candidate_estimates order: evaluate top follow_k that are not in sim_selected
                fcount = 0
                for mg2, j, closure_j in candidate_estimates:
                    if j in sim_selected:
                        continue
                    if not feasible_closure(closure_j, sim_selected):
                        continue
                    g2 = marginal_gain(closure_j, sim_selected)
                    if g2 > best_follow:
                        best_follow = g2
                    fcount += 1
                    if fcount >= follow_k:
                        break
                if best_follow > 0:
                    total_score_candidate += LOOKAHEAD_DISCOUNT * best_follow
            # choose best by total_score_candidate, tie-breaker larger closure
            if (total_score_candidate > best_score) or (abs(total_score_candidate - best_score) < 1e-9 and (best_closure is None or len(closure - selected_set) > len(best_closure - selected_set))):
                best_score = total_score_candidate
                best_closure = closure
                best_gain = mg

        # Accept if non-negative gain (small negative tolerances avoided)
        if best_closure is None:
            break
        if best_gain > -1e-12:
            to_add = best_closure - selected_set
            if to_add:
                selected_set.update(to_add)
                current_score = total_score(selected_set)
                # continue greedy
                continue
            else:
                # nothing new to add
                break
        else:
            break

    # If min_card not reached, add best feasible closures regardless of sign (prefer larger)
    while len(selected_set) < min_card:
        best_gain = float("-inf")
        best_closure = None
        for i in range(n):
            if i in selected_set:
                continue
            closure = closures[i]
            if not feasible_closure(closure, selected_set):
                continue
            mg = marginal_gain(closure, selected_set)
            if (mg > best_gain) or (abs(mg - best_gain) < 1e-9 and (best_closure is None or len(closure - selected_set) > len(best_closure - selected_set))):
                best_gain = mg
                best_closure = closure
        if best_closure is None:
            break
        selected_set.update(best_closure - selected_set)
        current_score = total_score(selected_set)

    # ---------- Local improvement: removable-item swaps ----------
    # Identify removable items: selected items that have no selected successors (safe to remove alone)
    # We'll attempt limited number of simulated swaps: remove up to need_slots removable items to add a new closure if net score improves.
    improvement = True
    max_local_iters = 200
    iter_count = 0
    while improvement and iter_count < max_local_iters:
        improvement = False
        iter_count += 1
        # update removable set
        removable = set()
        for v in selected_set:
            # if no successor in selected_set, it's removable
            if not any((succ in selected_set) for succ in succs.get(v, ())):
                removable.add(v)
        # prepare list of candidate closures not already satisfied
        candidates = []
        for i in range(n):
            if i in selected_set:
                continue
            closure = closures[i]
            # compute new_items
            new_items = closure - selected_set
            if not new_items:
                continue
            candidates.append((i, closure, new_items))
        # sort candidates by marginal_gain descending to try promising first
        candidates.sort(key=lambda x: marginal_gain(x[1], selected_set), reverse=True)
        # try top some candidates
        tried = 0
        for i, closure, new_items in candidates:
            if tried >= 60:
                break
            tried += 1
            slots_avail = max_card - len(selected_set)
            need_slots = max(0, len(new_items) - slots_avail)
            if need_slots == 0:
                # can add directly if feasible
                if feasible_closure(closure, selected_set):
                    mg = marginal_gain(closure, selected_set)
                    if mg > 1e-12:
                        selected_set.update(closure - selected_set)
                        current_score = total_score(selected_set)
                        improvement = True
                        break
                continue
            # need to remove need_slots items from removable to make room
            if len(removable) < need_slots:
                continue
            # compute individual removal scores relative to current selected_set (loss if removed)
            removal_scores = []
            for r in removable:
                # compute score contribution of r in current selected_set
                # contribution = weight(r) + interactions(r, others in selected_set)
                contrib = 0.0
                if 0 <= r < len(weights):
                    try:
                        contrib += float(weights[r])
                    except Exception:
                        pass
                for o in selected_set:
                    if o == r:
                        continue
                    contrib += interaction(r, o)
                removal_scores.append((contrib, r))
            # choose candidates to remove with smallest contribution (min loss)
            removal_scores.sort(key=lambda x: x[0])
            # try a few small removal combinations (greedy pick lowest need_slots)
            removals = set([x[1] for x in removal_scores[:need_slots]])
            # Ensure none of removals are in closure (we don't want to remove items that the closure will re-add - though allowed, it's wasteful)
            if removals & closure:
                # prefer next best set that doesn't intersect closure if possible
                alt = []
                for cval, rv in removal_scores:
                    if rv not in closure:
                        alt.append(rv)
                        if len(alt) >= need_slots:
                            break
                if len(alt) < need_slots:
                    # can't find non-intersecting removals; skip this candidate
                    continue
                removals = set(alt[:need_slots])
            # simulate new selected
            sim_selected = set(selected_set) - removals
            if not feasible_closure(closure, sim_selected):
                continue
            sim_after = sim_selected.union(closure)
            # compute delta
            sim_score = total_score(sim_after)
            delta = sim_score - current_score
            if delta > 1e-9:
                # perform swap
                selected_set = sim_after
                current_score = sim_score
                improvement = True
                break
        # continue loop until no improvements
    # ---------- Final safety trim ----------
    if len(selected_set) > max_card:
        # keep top-scoring items by approximate contribution
        def item_contribution(i, sel):
            s = 0.0
            if 0 <= i < len(weights):
                try:
                    s += float(weights[i])
                except Exception:
                    pass
            for j in sel:
                if j == i: continue
                s += interaction(i, j)
            return s
        sorted_items = sorted(selected_set, key=lambda x: item_contribution(x, selected_set), reverse=True)
        selected_set = set(sorted_items[:max_card])
        current_score = total_score(selected_set)

    # Ensure final feasibility: if constraints violated (very unlikely), fallback to greedy simple trim
    def validate_and_fix(sel):
        # fix group conflicts: keep highest contribution per group
        sel = set(sel)
        groups_rev = {}
        for v in list(sel):
            g = var_to_group.get(v)
            if g is None: continue
            groups_rev.setdefault(g, []).append(v)
        for g, members in groups_rev.items():
            if len(members) <= 1:
                continue
            # sort members by contribution and keep best
            members.sort(key=lambda x: (weights[x] if 0<=x<len(weights) else 0.0) + sum(interaction(x,y) for y in sel if y!=x), reverse=True)
            for rm in members[1:]:
                sel.discard(rm)
        # fix mutex conflicts by dropping the lower-contributing member
        for a in range(n):
            for b in mutex_map.get(a, ()):
                if a < b and a in sel and b in sel:
                    # drop the one with smaller contribution
                    ca = (weights[a] if 0<=a<len(weights) else 0.0) + sum(interaction(a,y) for y in sel if y!=a)
                    cb = (weights[b] if 0<=b<len(weights) else 0.0) + sum(interaction(b,y) for y in sel if y!=b)
                    if ca >= cb:
                        sel.discard(b)
                    else:
                        sel.discard(a)
        # enforce precedence: if some selected lacks its ancestors, add ancestors if feasible, otherwise drop the descendant
        changed = True
        while changed:
            changed = False
            for v in list(sel):
                for a in preds.get(v, ()):
                    if a not in sel:
                        # try to add ancestor if feasible
                        if len(sel) < max_card and (a not in sel) and feasible_closure({a}, sel):
                            sel.add(a)
                            changed = True
                        else:
                            # cannot add ancestor: drop v
                            sel.discard(v)
                            changed = True
                            break
        # enforce cardinality <= max_card by dropping lowest contribution items
        while len(sel) > max_card:
            contribs = [( (weights[x] if 0<=x<len(weights) else 0.0) + sum(interaction(x,y) for y in sel if y!=x), x) for x in sel]
            contribs.sort()
            sel.discard(contribs[0][1])
        return sel

    selected_set = validate_and_fix(selected_set)

    result = {"selection": {"variables": sorted(int(x) for x in selected_set)}}
    print(json.dumps(result))


if __name__ == "__main__":
    solve_sds()
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