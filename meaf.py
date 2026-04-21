from ga import run_ga
from pso import run_pso
from acs import run_acs

def run_meaf(
    problem,
    iters=300,
    seed=None,
    T=5,
    M=15,
    easthreshold=2,
    sditer=50,
    dfpercent=0.85,
    early_stop=None,
):
    algos = {
        "GA": (run_ga, 11),
        "PSO": (run_pso, 17),
        "ACS": (run_acs, 23),
    }
    active = set(algos.keys())
    best_cost = float("inf")
    best_sol = None
    last_improve = {k: 0 for k in algos}

    iter_now = 0
    while iter_now < iters and active:
        chunk = min(M, iters - iter_now)
        for name in list(active):
            func, offset = algos[name]
            cost, sol = func(
                problem,
                iters=chunk,
                seed=None if seed is None else seed + offset + iter_now,
                early_stop=early_stop,
            )
            if cost < best_cost:
                best_cost = cost
                best_sol = sol
                last_improve[name] = iter_now
            elif iter_now >= sditer and (iter_now - last_improve[name]) >= T:
                if len(active) > easthreshold and cost > best_cost / dfpercent:
                    active.remove(name)

        iter_now += chunk

    return best_cost, best_sol
