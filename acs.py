import random
from utils import build_heuristic_solution, evaluate

def run_acs(problem, iters=300, ants=100, seed=None, rho=0.3, q0=0.1, beta=2, top_k=1, early_stop=None):
    rng = random.Random(seed)
    customer_ids = [c.idx for c in problem.customers if c.idx != 0]
    all_nodes = [0] + customer_ids
    idx_map = {nid: i for i, nid in enumerate(all_nodes)}
    n = len(all_nodes)
    dist = problem.distance_matrix

    tau0 = 1.0
    tau = [[tau0] * n for _ in range(n)]

    best_sol = build_heuristic_solution(problem)
    best_cost = evaluate(best_sol, problem)
    cache = {}

    no_improve = 0

    for _ in range(iters):
        for _ in range(ants):
            sol = []
            remaining = customer_ids[:]
            current = 0

            while remaining:
                cur_idx = current
                desirability = []
                for nid in remaining:
                    j = nid
                    eta = 1.0 / (1.0 + dist[cur_idx][j])
                    desirability.append((nid, (tau[cur_idx][j] ** 1.0) * (eta ** beta)))

                if rng.random() < q0:
                    next_node = max(desirability, key=lambda x: x[1])[0]
                else:
                    total_w = sum(w for _, w in desirability)
                    r = rng.random() * total_w
                    acc = 0.0
                    next_node = desirability[-1][0]
                    for nid, w in desirability:
                        acc += w
                        if acc >= r:
                            next_node = nid
                            break

                sol.append(next_node)
                remaining.remove(next_node)
                current = next_node

            tsol = tuple(sol)
            if tsol not in cache:
                cache[tsol] = evaluate(sol, problem)
            cost = cache[tsol]

            if cost < best_cost:
                best_cost = cost
                best_sol = sol
                no_improve = 0
            else:
                no_improve += 1
                if early_stop is not None and no_improve >= early_stop:
                    return best_cost, best_sol

        # Global pheromone update using top-k (here k=1)
        best_iter_sol = best_sol
        edges = [(0, best_iter_sol[0])] if best_iter_sol else []
        edges += list(zip(best_iter_sol[:-1], best_iter_sol[1:]))
        edges += [(best_iter_sol[-1], 0)] if best_iter_sol else []
        for i, j in edges:
            tau[i][j] = (1 - rho) * tau[i][j] + rho * (1.0 / (1.0 + best_cost))
            tau[j][i] = tau[i][j]

    return best_cost, best_sol
