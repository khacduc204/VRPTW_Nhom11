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
    # PMX crossover (cycle-safe)
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [-1] * n
    child[a:b] = p1[a:b]

    for i in range(a, b):
        if p2[i] in child:
            continue
        val = p2[i]
        pos = i
        while True:
            mapped = p1[pos]
            pos = p2.index(mapped)
            if child[pos] == -1:
                child[pos] = val
                break

    for i in range(n):
        if child[i] == -1:
            child[i] = p2[i]

    return child

def run_ga(problem, iters=300, pop_size=100, seed=None, ts=0.2, mr=0.5, early_stop=None):
    rng = random.Random(seed)
    customer_ids = [c.idx for c in problem.customers if c.idx != 0]
    seed_solution = build_heuristic_solution(problem)
    pop = init_population(customer_ids, pop_size, rng, seed_solution=seed_solution)

    best = None
    best_cost = float("inf")

    cache = {}

    no_improve = 0

    for _ in range(iters):
        scored = []
        for sol in pop:
            tsol = tuple(sol)
            if tsol not in cache:
                cache[tsol] = evaluate(sol, problem)
            scored.append((cache[tsol], sol))
            
        scored.sort(key=lambda x: x[0])

        if scored[0][0] < best_cost:
            best_cost = scored[0][0]
            best = scored[0][1]
            no_improve = 0
        else:
            no_improve += 1
            if early_stop is not None and no_improve >= early_stop:
                break

        trunc_size = max(2, int(len(scored) * ts))
        pool = [scored[i][1][:] for i in range(trunc_size)]
        new_pop = [scored[0][1][:]]

        while len(new_pop) < pop_size:
            p1, p2 = rng.sample(pool, 2)
            child = crossover(p1, p2, rng)
            if rng.random() < mr:
                child = mutate(child, rng)
            new_pop.append(child)

        pop = new_pop

    return best_cost, best
