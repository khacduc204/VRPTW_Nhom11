import random
from utils import build_heuristic_solution, evaluate

def run_pso(
    problem,
    iters=300,
    swarm=100,
    seed=None,
    w=0.1,
    early_stop=None,
    return_history=False,
    init_heuristic=True,
):
    rng = random.Random(seed)
    customer_ids = [c.idx for c in problem.customers if c.idx != 0]
    seed_solution = build_heuristic_solution(problem) if init_heuristic else None
    particles = []
    if seed_solution is not None:
        particles.append(seed_solution[:])
    while len(particles) < swarm:
        if seed_solution is not None and rng.random() < 0.5:
            p = seed_solution[:]
            i, j = rng.sample(range(len(p)), 2)
            p[i], p[j] = p[j], p[i]
            particles.append(p)
        else:
            particles.append(rng.sample(customer_ids, len(customer_ids)))

    best = None
    best_cost = float("inf")
    cache = {}

    no_improve = 0

    history = []

    for _ in range(iters):
        for i, p in enumerate(particles):
            tp = tuple(p)
            if tp not in cache:
                cache[tp] = evaluate(p, problem)
            cost = cache[tp]
            if cost < best_cost:
                best_cost = cost
                best = p[:]
                no_improve = 0
            else:
                no_improve += 1
                if early_stop is not None and no_improve >= early_stop:
                    if return_history:
                        history.append(best_cost)
                        return best_cost, best, history
                    return best_cost, best

            if rng.random() < w:
                i, j = rng.sample(range(len(p)), 2)
                p[i], p[j] = p[j], p[i]
            elif best is not None:
                # Pull particle toward global best by segment injection.
                cut = rng.randint(1, len(p) - 2)
                segment = best[:cut]
                p[:] = segment + [x for x in p if x not in segment]

        if return_history:
            history.append(best_cost)

    if return_history:
        return best_cost, best, history
    return best_cost, best
