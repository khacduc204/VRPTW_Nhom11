import numpy as np
from data import distance_idx

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

USE_NUMBA = True


def set_use_numba(enabled):
    global USE_NUMBA
    USE_NUMBA = bool(enabled)


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _decode_giant_tour_numba(
        solution,
        dist,
        demand,
        ready,
        due,
        service,
        capacity,
        vehicle_count,
        depot_due,
        extra_penalty,
        infeasible_cost,
    ):
        routes = 0
        load = 0.0
        time = 0.0
        prev = 0
        total_dist = 0.0
        has_route = False

        for idx in solution:
            if idx == 0:
                continue

            travel = dist[prev][idx]
            arrival = time + travel
            start_service = arrival if arrival > ready[idx] else ready[idx]
            return_to_depot = start_service + service[idx] + dist[idx][0]

            can_append = (
                load + demand[idx] <= capacity
                and start_service <= due[idx]
                and return_to_depot <= depot_due
            )

            if (not can_append) and has_route:
                total_dist += dist[prev][0]
                routes += 1
                load = 0.0
                time = 0.0
                prev = 0
                has_route = False

                travel = dist[prev][idx]
                arrival = time + travel
                start_service = arrival if arrival > ready[idx] else ready[idx]
                return_to_depot = start_service + service[idx] + dist[idx][0]
                can_append = (
                    load + demand[idx] <= capacity
                    and start_service <= due[idx]
                    and return_to_depot <= depot_due
                )

            if not can_append:
                return infeasible_cost, 0, 0

            load += demand[idx]
            time = start_service + service[idx]
            total_dist += travel
            prev = idx
            has_route = True

        if has_route:
            total_dist += dist[prev][0]
            routes += 1

        if routes > vehicle_count:
            total_dist += extra_penalty * (routes - vehicle_count)
            return total_dist, routes, 0

        return total_dist, routes, 1

INFEASIBLE_COST = 10**12
HARD_VIOLATION_PENALTY = 10**6
EXTRA_ROUTE_PENALTY = 250.0


def _build_customer_map(customers):
    return {c.idx: c for c in customers}


def build_heuristic_solution(problem):
    customers = _build_customer_map(problem.customers)
    depot = customers[0]
    depot_idx = 0
    remaining = {c.idx for c in problem.customers if c.idx != 0}
    giant_tour = []

    while remaining:
        route = []
        load = 0.0
        time = 0.0
        prev_idx = depot_idx

        while True:
            candidates = []
            for idx in remaining:
                c = customers[idx]
                if load + c.demand > problem.capacity:
                    continue

                travel = distance_idx(problem, prev_idx, idx)
                arrival = time + travel
                start_service = max(arrival, c.ready)
                end_service = start_service + c.service
                if start_service > c.due:
                    continue
                if end_service + distance_idx(problem, idx, depot_idx) > depot.due:
                    continue

                # Favor low travel and early service times.
                score = travel + 0.2 * max(0.0, c.ready - arrival)
                candidates.append((score, idx, start_service, end_service))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0])
            _, chosen, start_service, end_service = candidates[0]
            c = customers[chosen]

            route.append(chosen)
            remaining.remove(chosen)
            load += c.demand
            time = end_service
            prev_idx = chosen

        if not route:
            # Fallback to avoid deadlock on very hard/tight instances.
            chosen = min(remaining)
            route = [chosen]
            remaining.remove(chosen)

        giant_tour.extend(route)

    return giant_tour


def decode_giant_tour(solution, problem, use_numba=True):
    if use_numba and NUMBA_AVAILABLE:
        sol_arr = np.asarray(solution, dtype=np.int64)
        total_dist, routes_count, feasible_int = _decode_giant_tour_numba(
            sol_arr,
            problem.distance_matrix,
            problem.demand_arr,
            problem.ready_arr,
            problem.due_arr,
            problem.service_arr,
            float(problem.capacity),
            int(problem.vehicle_count),
            float(problem.customers[0].due),
            float(EXTRA_ROUTE_PENALTY),
            float(INFEASIBLE_COST),
        )
        feasible = bool(feasible_int)
        routes = [None] * int(routes_count)
        return float(total_dist), routes, feasible

    customers = _build_customer_map(problem.customers)
    depot = customers[0]
    depot_idx = 0

    routes = []
    current_route = []
    load = 0.0
    time = 0.0
    prev_idx = depot_idx
    total_dist = 0.0

    for idx in solution:
        if idx == 0:
            continue

        c = customers[idx]

        travel = distance_idx(problem, prev_idx, idx)
        arrival = time + travel
        start_service = max(arrival, c.ready)
        return_to_depot = start_service + c.service + distance_idx(problem, idx, depot_idx)

        can_append = (
            load + c.demand <= problem.capacity
            and start_service <= c.due
            and return_to_depot <= depot.due
        )

        if not can_append and current_route:
            total_dist += distance_idx(problem, prev_idx, depot_idx)
            routes.append(current_route)
            current_route = []
            load = 0.0
            time = 0.0
            prev_idx = depot_idx

            travel = distance_idx(problem, prev_idx, idx)
            arrival = time + travel
            start_service = max(arrival, c.ready)
            return_to_depot = start_service + c.service + distance_idx(problem, idx, depot_idx)
            can_append = (
                load + c.demand <= problem.capacity
                and start_service <= c.due
                and return_to_depot <= depot.due
            )

        if not can_append:
            return INFEASIBLE_COST, [], False

        current_route.append(idx)
        load += c.demand
        time = start_service + c.service
        total_dist += travel
        prev_idx = idx

    if current_route:
        total_dist += distance_idx(problem, prev_idx, depot_idx)
        routes.append(current_route)

    feasible = len(routes) <= problem.vehicle_count
    if not feasible:
        extra = len(routes) - problem.vehicle_count
        penalized = total_dist + EXTRA_ROUTE_PENALTY * extra
        return penalized, routes, False

    return total_dist, routes, True


def evaluate(solution, problem):
    total_dist, routes, feasible = decode_giant_tour(solution, problem, use_numba=USE_NUMBA)
    if not feasible:
        # Hard decode violations are rare, but keep output bounded.
        if total_dist >= INFEASIBLE_COST:
            fallback = build_heuristic_solution(problem)
            fb_dist, _, fb_feasible = decode_giant_tour(fallback, problem)
            if fb_feasible and fb_dist < INFEASIBLE_COST:
                return fb_dist
            return HARD_VIOLATION_PENALTY
        return total_dist

    # Table-style reporting uses distance scale for feasible solutions.
    return total_dist
