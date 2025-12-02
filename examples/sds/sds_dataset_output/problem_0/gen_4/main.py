"""
Initial solution for Synergistic Dependency Selection (SDS) problem.
The code in EVOLVE-BLOCK will be evolved by ShinkaEvolve.
"""

import json
import sys
import contextlib

# EVOLVE-BLOCK-START
A novel SDS solver: precedence-aware beam + simulated-annealing-style local search.
Reads JSON from stdin with "requirements" and "catalog", writes JSON to stdout:
{"selection":{"variables":[...]}}

This block is intended to be evolved by ShinkaEvolve.
"""
import sys
import json
import math
import random
import time

def solve_sds():
    input_data = json.load(sys.stdin)
    requirements = input_data.get("requirements", {})
    catalog = input_data.get("catalog", {})

    weights = requirements.get("weights", [])
    interactions = requirements.get("interactions", {})
    cardinality_bounds = requirements.get("cardinality_bounds", [0, len(weights)])
    min_card, max_card = cardinality_bounds[0], cardinality_bounds[1]
    mutex = requirements.get("mutex", [])
    groups = requirements.get("groups", {})
    precedence = requirements.get("precedence", [])

    n = len(weights)

    # Robust interaction parsing: handle "i,j" -> value or nested dicts
    interaction_map = {}
    # first try string keys like "i,j" or tuple-like
    if isinstance(interactions, dict):
        for k, v in interactions.items():
            if isinstance(k, str):
                if "," in k:
                    try:
                        a_s, b_s = k.split(",")
                        a, b = int(a_s), int(b_s)
                        if a > b:
                            a, b = b, a
                        interaction_map[(a, b)] = float(v)
                    except Exception:
                        continue
                else:
                    # maybe nested mapping case handled later
                    continue
            elif isinstance(k, (list, tuple)) and len(k) == 2:
                try:
                    a, b = int(k[0]), int(k[1])
                    if a > b:
                        a, b = b, a
                    interaction_map[(a, b)] = float(v)
                except Exception:
                    continue
    # try nested dict form interactions[i][j] = val
    if not interaction_map and isinstance(interactions, dict):
        try:
            for ik, row in interactions.items():
                ii = int(ik)
                if isinstance(row, dict):
                    for jk, vv in row.items():
                        jj = int(jk)
                        a, b = (ii, jj) if ii <= jj else (jj, ii)
                        interaction_map[(a, b)] = float(vv)
        except Exception:
            pass

    def get_interaction(i, j):
        if i > j:
            i, j = j, i
        return interaction_map.get((i, j), 0.0)

    # Precedence maps
    preds = {i: set() for i in range(n)}
    succs = {i: set() for i in range(n)}
    for p in precedence:
        try:
            a = int(p[0]); b = int(p[1])
            # if b selected then a must be selected => a is predecessor of b
            preds[b].add(a)
            succs[a].add(b)
        except Exception:
            continue

    # Compute transitive predecessor closure and successor closure
    pred_closure = {}
    for i in range(n):
        seen = set()
        stack = list(preds.get(i, []))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            for y in preds.get(x, ()):
                if y not in seen:
                    stack.append(y)
        pred_closure[i] = seen

    succ_closure = {}
    for i in range(n):
        seen = set()
        stack = list(succs.get(i, []))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            for y in succs.get(x, ()):
                if y not in seen:
                    stack.append(y)
        succ_closure[i] = seen

    # Precompute group and mutex quick checks
    # groups: dict name -> list of vars
    var_to_group = {}
    for gname, vars_list in groups.items():
        for v in vars_list:
            var_to_group[v] = gname

    mutex_pairs = set()
    mutex_map = {i: set() for i in range(n)}
    for a, b in mutex:
        mutex_pairs.add((a, b))
        mutex_pairs.add((b, a))
        if 0 <= a < n and 0 <= b < n:
            mutex_map[a].add(b)
            mutex_map[b].add(a)

    # Quick feasibility check
    def is_feasible_set(S):
        if len(S) > max_card or len(S) < 0:
            return False
        # precedence: all preds must be included
        for j in S:
            for p in preds.get(j, ()):
                if p not in S:
                    return False
        # mutex
        for a in S:
            for b in mutex_map.get(a, ()):
                if b in S:
                    return False
        # groups
        seen_groups = {}
        for v in S:
            g = var_to_group.get(v)
            if g is not None:
                if g in seen_groups:
                    return False
                seen_groups[g] = v
        return True

    # Objective computation optimized: sum weights + sum interactions for pairs (i<j)
    # We'll maintain an interactions adjacency list for non-zero interactions
    neighbors = {i: {} for i in range(n)}
    for (a, b), val in interaction_map.items():
        if 0 <= a < n and 0 <= b < n:
            neighbors[a][b] = val
            neighbors[b][a] = val

    def objective(S):
        total = 0.0
        for i in S:
            if 0 <= i < n:
                total += weights[i]
        # pairwise interactions
        # sum over each i neighbors j>i in S
        Sset = set(S)
        for i in Sset:
            for j, val in neighbors[i].items():
                if j in Sset and j > i:
                    total += val
        return total

    # Marginal gain for adding a set A to S (A disjoint or overlap handled)
    def marginal_gain(S, A):
        Sset = set(S)
        Aset = set(A) - Sset
        if not Aset:
            return 0.0
        gain = 0.0
        for i in Aset:
            if 0 <= i < n:
                gain += weights[i]
        # interactions between A and S
        for i in Aset:
            for j, val in neighbors[i].items():
                if j in Sset:
                    gain += val
        # interactions internal to A
        Alist = list(Aset)
        for ui in range(len(Alist)):
            ii = Alist[ui]
            for uj in range(ui + 1, len(Alist)):
                jj = Alist[uj]
                gain += neighbors[ii].get(jj, 0.0)
        return gain

    # closure_for: when adding i, include its predecessors (transitive)
    def closure_for_add(i, current_set):
        res = set([i])
        for p in pred_closure.get(i, set()):
            res.add(p)
        # exclude already present
        return res - set(current_set)

    # closure_for_remove: when removing i, need to remove all successors that depend on it (transitive)
    def closure_for_remove(i):
        res = set([i])
        for s in succ_closure.get(i, set()):
            res.add(s)
        return res

    # centrality heuristic: base for seeding
    centrality = []
    for i in range(n):
        s = weights[i] if 0 <= i < n else 0.0
        # sum of positive interactions
        pos_sum = 0.0
        for j, val in neighbors[i].items():
            if val > 0:
                pos_sum += val
            else:
                pos_sum += 0.1 * val  # small weight for negative links
        s += pos_sum
        centrality.append((s, i))
    centrality.sort(reverse=True)

    # Beam parameters scaled with n
    beam_width = max(5, min(40, n // 3 + 1))
    seeds_to_try = beam_width
    seed_indices = [i for _, i in centrality[:seeds_to_try]]

    # If too few seeds (small n), consider random seeds for diversity
    if len(seed_indices) < beam_width:
        others = [i for i in range(n) if i not in seed_indices]
        random.shuffle(others)
        seed_indices += others[:beam_width - len(seed_indices)]

    # Greedy augment function (respect closures, groups, mutex)
    def greedy_augment(initial_set):
        S = set(initial_set)
        # If initial violates precedence, fix by adding needed preds
        changed = True
        while changed:
            changed = False
            for j in list(S):
                for p in preds.get(j, ()):
                    if p not in S:
                        S.add(p)
                        changed = True
        # Greedily add items with best positive marginal gain
        while len(S) < max_card:
            best_gain = -1e18
            best_add = None
            for i in range(n):
                if i in S:
                    continue
                add_set = closure_for_add(i, S)
                if not add_set:
                    continue
                if len(S) + len(add_set) > max_card:
                    continue
                # check group/mutex feasibility quickly
                violated = False
                # group conflict
                existing_groups = set(var_to_group.get(v) for v in S if var_to_group.get(v) is not None)
                for v in add_set:
                    g = var_to_group.get(v)
                    if g is not None and g in existing_groups:
                        violated = True
                        break
                if violated:
                    continue
                # mutex: any v in add_set conflicting with S?
                for v in add_set:
                    for m in mutex_map.get(v, ()):
                        if m in S:
                            violated = True
                            break
                    if violated:
                        break
                if violated:
                    continue
                gain = marginal_gain(S, add_set)
                # prefer positive gains; otherwise small positive if need to reach min_card
                if gain > best_gain:
                    best_gain = gain
                    best_add = add_set
            if best_add is None or (best_gain <= 1e-12 and len(S) >= min_card):
                break
            S |= set(best_add)
        # If below min_card, force add highest marginal items (even negative if necessary)
        if len(S) < min_card:
            # consider all candidates sorted by marginal gain
            cand_list = []
            for i in range(n):
                if i in S:
                    continue
                add_set = closure_for_add(i, S)
                if not add_set:
                    continue
                if len(S) + len(add_set) > max_card:
                    continue
                # skip if group/mutex violate
                violated = False
                existing_groups = set(var_to_group.get(v) for v in S if var_to_group.get(v) is not None)
                for v in add_set:
                    g = var_to_group.get(v)
                    if g is not None and g in existing_groups:
                        violated = True
                        break
                    for m in mutex_map.get(v, ()):
                        if m in S:
                            violated = True
                            break
                    if violated:
                        break
                if violated:
                    continue
                cand_list.append((marginal_gain(S, add_set), add_set))
            cand_list.sort(reverse=True, key=lambda x: x[0])
            for gain, add_set in cand_list:
                if len(S) >= min_card:
                    break
                if len(S) + len(add_set) > max_card:
                    continue
                S |= set(add_set)
        # final feasibility guard
        if not is_feasible_set(S):
            # fallback: simple weight-based fill respecting closures
            S = set()
            for i in sorted(range(n), key=lambda x: weights[x], reverse=True):
                add_set = closure_for_add(i, S)
                if len(S) + len(add_set) > max_card:
                    continue
                if not is_feasible_set(S | set(add_set)):
                    continue
                S |= set(add_set)
                if len(S) >= max_card:
                    break
        return S

    # Build initial beam of solutions
    beam = []
    tried_seeds = set()
    for sidx in seed_indices:
        if sidx in tried_seeds:
            continue
        tried_seeds.add(sidx)
        init = closure_for_add(sidx, set())
        # if too large, skip seed
        if len(init) > max_card:
            continue
        sol = greedy_augment(init)
        beam.append(sol)
    # If beam too small, add some empties and singletons
    if not beam:
        beam.append(set())
    # ensure beam unique
    uniq = []
    seen = set()
    for sol in beam:
        key = tuple(sorted(sol))
        if key not in seen:
            uniq.append(sol)
            seen.add(key)
    beam = uniq[:beam_width]

    # Local search with simulated annealing for each beam member
    best_sol = None
    best_score = -1e99

    # Tune iterations by problem size
    base_iters = 300
    iter_scale = max(1, n // 10)
    max_iters_per_seed = min(2000, base_iters * iter_scale)

    rng = random.Random()
    # deterministic-ish seed based on weights to be reproducible across runs
    rng.seed(int(sum(weights) * 1000) if weights else 12345)

    start_time = time.time()
    time_limit = 1.8  # try to keep runtime modest; evaluator may limit
    for sol in beam:
        S = set(sol)
        S = greedy_augment(S)  # tighten
        cur_score = objective(S)
        if cur_score > best_score:
            best_score = cur_score
            best_sol = set(S)
        # SA parameters
        T0 = max(1.0, abs(cur_score) * 0.1 + 1.0)
        Tmin = 1e-3
        iters = 0
        last_improve = 0
        while iters < max_iters_per_seed:
            iters += 1
            # time cut
            if time.time() - start_time > time_limit:
                break
            # temperature schedule
            frac = iters / float(max_iters_per_seed + 1)
            temp = T0 * (1.0 - frac) + Tmin * frac
            op = rng.choice(['add', 'remove', 'swap'])
            accepted = False
            if op == 'add':
                # pick candidate item not in S that might improve
                candidates = list(set(range(n)) - S)
                rng.shuffle(candidates)
                for cand in candidates[:min(50, len(candidates))]:
                    add_set = closure_for_add(cand, S)
                    if not add_set:
                        continue
                    if len(S) + len(add_set) > max_card:
                        continue
                    # group/mutex check
                    violated = False
                    existing_groups = set(var_to_group.get(v) for v in S if var_to_group.get(v) is not None)
                    for v in add_set:
                        g = var_to_group.get(v)
                        if g is not None and g in existing_groups:
                            violated = True
                            break
                        for m in mutex_map.get(v, ()):
                            if m in S:
                                violated = True
                                break
                        if violated:
                            break
                    if violated:
                        continue
                    newS = set(S) | set(add_set)
                    # ensure not violating precedence (add_set includes preds)
                    if not is_feasible_set(newS):
                        continue
                    new_score = objective(newS)
                    delta = new_score - cur_score
                    if delta > 1e-12 or math.exp(min(50.0, delta / max(1e-9, temp))) > rng.random():
                        S = newS
                        cur_score = new_score
                        accepted = True
                        last_improve = iters
                        break
            elif op == 'remove':
                if len(S) == 0:
                    continue
                removes = list(S)
                rng.shuffle(removes)
                for rem in removes[:min(50, len(removes))]:
                    rem_set = closure_for_remove(rem)
                    newS = set(S) - rem_set
                    # enforce min_card or allow temporary below with small prob
                    if len(newS) < min_card:
                        # permit removal with low probability if helps and temperature high
                        pass_prob = math.exp(- (min_card - len(newS)) / (1.0 + temp))
                        if rng.random() > pass_prob:
                            continue
                    if not is_feasible_set(newS):
                        continue
                    new_score = objective(newS)
                    delta = new_score - cur_score
                    if delta > 1e-12 or math.exp(min(50.0, delta / max(1e-9, temp))) > rng.random():
                        S = newS
                        cur_score = new_score
                        accepted = True
                        last_improve = iters
                        break
            else:  # swap: remove some dependent cluster and add some other closure
                if len(S) == 0:
                    continue
                rem = rng.choice(list(S))
                rem_set = closure_for_remove(rem)
                base = set(S) - rem_set
                # consider candidates to add given base
                candidates = list(set(range(n)) - base)
                rng.shuffle(candidates)
                for cand in candidates[:min(80, len(candidates))]:
                    add_set = closure_for_add(cand, base)
                    if not add_set:
                        continue
                    if len(base) + len(add_set) > max_card:
                        continue
                    # group/mutex check
                    violated = False
                    existing_groups = set(var_to_group.get(v) for v in base if var_to_group.get(v) is not None)
                    for v in add_set:
                        g = var_to_group.get(v)
                        if g is not None and g in existing_groups:
                            violated = True
                            break
                        for m in mutex_map.get(v, ()):
                            if m in base:
                                violated = True
                                break
                        if violated:
                            break
                    if violated:
                        continue
                    newS = set(base) | set(add_set)
                    if len(newS) < min_card:
                        continue
                    if not is_feasible_set(newS):
                        continue
                    new_score = objective(newS)
                    delta = new_score - cur_score
                    if delta > 1e-12 or math.exp(min(50.0, delta / max(1e-9, temp))) > rng.random():
                        S = newS
                        cur_score = new_score
                        accepted = True
                        last_improve = iters
                        break
            # record best
            if cur_score > best_score + 1e-12:
                best_score = cur_score
                best_sol = set(S)
            # small early stop if no improvement for long
            if iters - last_improve > 400 and iters > 800:
                break
        # time limit guard
        if time.time() - start_time > time_limit:
            break

    # Final repair and ensure feasibility and cardinality
    if best_sol is None:
        best_sol = set()
    S = set(best_sol)
    # Ensure precedence closure included
    changed = True
    while changed:
        changed = False
        for j in list(S):
            for p in preds.get(j, ()):
                if p not in S:
                    S.add(p)
                    changed = True
    # If too large, try removing low marginal contributors while respecting precedence
    if len(S) > max_card:
        # rank items by marginal contribution (with respect to S)
        def marginal_contrib(i, curS):
            # contribution approximated as weight + sum interactions with curS - interactions lost from removal
            contrib = weights[i]
            for j, val in neighbors[i].items():
                if j in curS and j != i:
                    contrib += val
            return contrib
        # try removing items with smallest contrib but ensure we also remove dependents
        items = sorted(list(S), key=lambda x: marginal_contrib(x, S))
        for itv in items:
            if len(S) <= max_card:
                break
            # can't remove if it has predecessors that would remain invalid (we'll remove dependents)
            rem_cluster = closure_for_remove(itv)
            # ensure we don't violate min_card after removal
            if len(S) - len(rem_cluster) < min_card:
                continue
            S -= rem_cluster
        # final safety: if still too large, keep top by contrib
        if len(S) > max_card:
            items = sorted(list(S), key=lambda x: marginal_contrib(x, S), reverse=True)
            S = set(items[:max_card])

    # If too small, fill greedily
    if len(S) < min_card:
        # add best marginal additions
        while len(S) < min_card:
            best_gain = -1e99
            best_add = None
            for i in range(n):
                if i in S:
                    continue
                add_set = closure_for_add(i, S)
                if not add_set:
                    continue
                if len(S) + len(add_set) > max_card:
                    continue
                # group/mutex check
                violated = False
                existing_groups = set(var_to_group.get(v) for v in S if var_to_group.get(v) is not None)
                for v in add_set:
                    g = var_to_group.get(v)
                    if g is not None and g in existing_groups:
                        violated = True
                        break
                    for m in mutex_map.get(v, ()):
                        if m in S:
                            violated = True
                            break
                    if violated:
                        break
                if violated:
                    continue
                gain = marginal_gain(S, add_set)
                if gain > best_gain:
                    best_gain = gain
                    best_add = add_set
            if best_add is None:
                break
            S |= set(best_add)

    # Final enforced feasibility (last resort simple greedy)
    if not is_feasible_set(S):
        S = set()
        for i in sorted(range(n), key=lambda x: weights[x], reverse=True):
            add_set = closure_for_add(i, S)
            if len(S) + len(add_set) > max_card:
                continue
            if not is_feasible_set(S | set(add_set)):
                continue
            S |= set(add_set)
            if len(S) >= max_card:
                break
        i = 0
        while len(S) < min_card and i < n:
            if i not in S:
                add_set = closure_for_add(i, S)
                if len(S) + len(add_set) <= max_card and is_feasible_set(S | set(add_set)):
                    S |= set(add_set)
            i += 1

    result = {"selection": {"variables": sorted(list(S))}}
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
