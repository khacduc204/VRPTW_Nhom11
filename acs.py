import random
from utils import build_heuristic_solution, evaluate

def run_acs(
    problem,
    iters=300,
    ants=100,
    seed=None,
    rho=0.3,
    q0=0.1,
    beta=2,
    top_k=1,
    early_stop=None,
    return_history=False,
    init_heuristic=True,
):
    rng = random.Random(seed)
    customer_ids = [c.idx for c in problem.customers if c.idx != 0]
    n = len(problem.customers)
    dist = problem.distance_matrix
    demand = problem.demand_arr
    ready = problem.ready_arr
    due = problem.due_arr
    service = problem.service_arr
    depot_due = problem.customers[0].due
    capacity = problem.capacity

    tau0 = 1.0
    tau = [[tau0] * n for _ in range(n)]

    if init_heuristic:
        best_sol = build_heuristic_solution(problem)
    else:
        best_sol = rng.sample(customer_ids, len(customer_ids))
    best_cost = evaluate(best_sol, problem)
    cache = {}

    no_improve = 0

    history = []

    for _ in range(iters):
        for _ in range(ants):
            sol = []
            remaining = customer_ids[:]
            current = 0
            load = 0.0
            time = 0.0

            while remaining:
                cur_idx = current
                desirability = []
                for nid in remaining:
                    if load + demand[nid] > capacity:
                        continue
                    travel = dist[cur_idx][nid]
                    arrival = time + travel
                    start_service = arrival if arrival > ready[nid] else ready[nid]
                    return_to_depot = start_service + service[nid] + dist[nid][0]
                    if start_service > due[nid] or return_to_depot > depot_due:
                        continue
                    eta = 1.0 / (1.0 + travel)
                    desirability.append((nid, (tau[cur_idx][nid] ** 1.0) * (eta ** beta)))

                if not desirability:
                    # Start a new route when no feasible customer fits.
                    current = 0
                    load = 0.0
                    time = 0.0
                    # If still no feasible, pick the nearest to avoid deadlock.
                    if not remaining:
                        break
                    next_node = min(remaining, key=lambda nid: dist[0][nid])
                    sol.append(next_node)
                    remaining.remove(next_node)
                    current = next_node
                    load = demand[next_node]
                    time = max(dist[0][next_node], ready[next_node]) + service[next_node]
                    continue

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
                travel = dist[cur_idx][next_node]
                arrival = time + travel
                start_service = arrival if arrival > ready[next_node] else ready[next_node]
                load += demand[next_node]
                time = start_service + service[next_node]
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
                    if return_history:
                        history.append(best_cost)
                        return best_cost, best_sol, history
                    return best_cost, best_sol

        # Global pheromone update using top-k (here k=1)
        best_iter_sol = best_sol
        edges = [(0, best_iter_sol[0])] if best_iter_sol else []
        edges += list(zip(best_iter_sol[:-1], best_iter_sol[1:]))
        edges += [(best_iter_sol[-1], 0)] if best_iter_sol else []
        for i, j in edges:
            tau[i][j] = (1 - rho) * tau[i][j] + rho * (1.0 / (1.0 + best_cost))
            tau[j][i] = tau[i][j]

        if return_history:
            history.append(best_cost)

    if return_history:
        return best_cost, best_sol, history
    return best_cost, best_sol
