import random
from utils import build_heuristic_solution, evaluate

def init_population(ids, size, rng, seed_solution=None):
    pop = []
    if seed_solution is not None:
        pop.append(seed_solution[:])

    while len(pop) < size:
        if seed_solution is not None and rng.random() < 0.5:
            child = seed_solution[:]
            i, j = rng.sample(range(len(child)), 2)
            child[i], child[j] = child[j], child[i]
            pop.append(child)
        else:
            pop.append(rng.sample(ids, len(ids)))
    return pop

def mutate(sol, rng):
    i, j = rng.sample(range(len(sol)), 2)
    sol[i], sol[j] = sol[j], sol[i]
    return sol

def crossover(p1, p2, rng):
    cut = rng.randint(1, len(p1)-1)
    child = p1[:cut] + [x for x in p2 if x not in p1[:cut]]
    return child

def run_ga(problem, iters=300, pop_size=60, seed=None):
    rng = random.Random(seed)
    customer_ids = [c.idx for c in problem.customers if c.idx != 0]
    seed_solution = build_heuristic_solution(problem)
    pop = init_population(customer_ids, pop_size, rng, seed_solution=seed_solution)

    best = None
    best_cost = float("inf")

    for _ in range(iters):
        scored = [(evaluate(sol, problem), sol) for sol in pop]
        scored.sort()

        if scored[0][0] < best_cost:
            best_cost = scored[0][0]
            best = scored[0][1]

        elite_size = min(8, len(scored))
        new_pop = [scored[i][1][:] for i in range(elite_size)]

        while len(new_pop) < pop_size:
            p1, p2 = rng.sample(new_pop, 2)
            child = crossover(p1, p2, rng)
            if rng.random() < 0.25:
                child = mutate(child, rng)
            new_pop.append(child)

        pop = new_pop

    return best_cost, best
