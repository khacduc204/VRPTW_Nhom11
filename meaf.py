from ga import run_ga
from pso import run_pso
from acs import run_acs

def run_meaf(problem, iters=300, seed=None):
    stage_iters = max(50, iters // 3)

    ga_cost, ga_sol = run_ga(problem, iters=stage_iters, seed=None if seed is None else seed + 11)
    pso_cost, pso_sol = run_pso(problem, iters=stage_iters, seed=None if seed is None else seed + 17)
    acs_cost, acs_sol = run_acs(problem, iters=stage_iters, seed=None if seed is None else seed + 23)

    best_cost = min(ga_cost, pso_cost, acs_cost)

    if best_cost == ga_cost:
        final_cost, final_sol = run_ga(problem, iters=iters, seed=None if seed is None else seed + 101)
        if final_cost < ga_cost:
            return final_cost, final_sol
        return ga_cost, ga_sol
    elif best_cost == pso_cost:
        final_cost, final_sol = run_pso(problem, iters=iters, seed=None if seed is None else seed + 103)
        if final_cost < pso_cost:
            return final_cost, final_sol
        return pso_cost, pso_sol
    else:
        final_cost, final_sol = run_acs(problem, iters=iters, seed=None if seed is None else seed + 107)
        if final_cost < acs_cost:
            return final_cost, final_sol
        return acs_cost, acs_sol
