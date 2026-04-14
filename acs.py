import random
from data import distance
from utils import build_heuristic_solution, evaluate

def run_acs(problem, iters=300, ants=40, seed=None):
    rng = random.Random(seed)
    customer_ids = [c.idx for c in problem.customers if c.idx != 0]
    customer_map = {c.idx: c for c in problem.customers}

    best_sol = build_heuristic_solution(problem)
    best_cost = evaluate(best_sol, problem)

    for _ in range(iters):
        for _ in range(ants):
            sol = []
            remaining = customer_ids[:]
            current = 0

            while remaining:
                # Soft nearest-neighbor choice for better route locality.
                weights = []
                cur_customer = customer_map[current]
                for nid in remaining:
                    d = distance(cur_customer, customer_map[nid])
                    weights.append(1.0 / (1.0 + d))

                total_w = sum(weights)
                r = rng.random() * total_w
                acc = 0.0
                next_node = remaining[-1]
                for nid, w in zip(remaining, weights):
                    acc += w
                    if acc >= r:
                        next_node = nid
                        break

                sol.append(next_node)
                remaining.remove(next_node)
                current = next_node

            cost = evaluate(sol, problem)

            if cost < best_cost:
                best_cost = cost
                best_sol = sol

    return best_cost, best_sol
