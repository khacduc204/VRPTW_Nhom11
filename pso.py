import random
from utils import build_heuristic_solution, evaluate

def run_pso(problem, iters=300, swarm=40, seed=None):
    rng = random.Random(seed)
    customer_ids = [c.idx for c in problem.customers if c.idx != 0]
    seed_solution = build_heuristic_solution(problem)
    particles = [seed_solution[:]]
    while len(particles) < swarm:
        if rng.random() < 0.5:
            p = seed_solution[:]
            i, j = rng.sample(range(len(p)), 2)
            p[i], p[j] = p[j], p[i]
            particles.append(p)
        else:
            particles.append(rng.sample(customer_ids, len(customer_ids)))

    best = None
    best_cost = float("inf")

    for _ in range(iters):
        for p in particles:
            cost = evaluate(p, problem)
            if cost < best_cost:
                best_cost = cost
                best = p[:]

            i, j = rng.sample(range(len(p)), 2)
            p[i], p[j] = p[j], p[i]

            if best is not None and rng.random() < 0.15:
                # Pull particle toward global best by segment injection.
                cut = rng.randint(1, len(p) - 2)
                segment = best[:cut]
                p[:] = segment + [x for x in p if x not in segment]

    return best_cost, best
