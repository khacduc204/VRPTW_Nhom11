from data import distance

INFEASIBLE_COST = 10**12
HARD_VIOLATION_PENALTY = 10**6
EXTRA_ROUTE_PENALTY = 250.0


def _build_customer_map(customers):
    return {c.idx: c for c in customers}


def build_heuristic_solution(problem):
    customers = _build_customer_map(problem.customers)
    depot = customers[0]
    remaining = {c.idx for c in problem.customers if c.idx != 0}
    giant_tour = []

    while remaining:
        route = []
        load = 0.0
        time = 0.0
        prev = depot

        while True:
            candidates = []
            for idx in remaining:
                c = customers[idx]
                if load + c.demand > problem.capacity:
                    continue

                travel = distance(prev, c)
                arrival = time + travel
                start_service = max(arrival, c.ready)
                end_service = start_service + c.service
                if start_service > c.due:
                    continue
                if end_service + distance(c, depot) > depot.due:
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
            prev = c

        if not route:
            # Fallback to avoid deadlock on very hard/tight instances.
            chosen = min(remaining)
            route = [chosen]
            remaining.remove(chosen)

        giant_tour.extend(route)

    return giant_tour


def decode_giant_tour(solution, problem):
    customers = _build_customer_map(problem.customers)
    depot = customers[0]

    routes = []
    current_route = []
    load = 0.0
    time = 0.0
    prev = depot
    total_dist = 0.0

    for idx in solution:
        if idx == 0:
            continue

        c = customers[idx]

        travel = distance(prev, c)
        arrival = time + travel
        start_service = max(arrival, c.ready)
        return_to_depot = start_service + c.service + distance(c, depot)

        can_append = (
            load + c.demand <= problem.capacity
            and start_service <= c.due
            and return_to_depot <= depot.due
        )

        if not can_append and current_route:
            total_dist += distance(prev, depot)
            routes.append(current_route)
            current_route = []
            load = 0.0
            time = 0.0
            prev = depot

            travel = distance(prev, c)
            arrival = time + travel
            start_service = max(arrival, c.ready)
            return_to_depot = start_service + c.service + distance(c, depot)
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
        prev = c

    if current_route:
        total_dist += distance(prev, depot)
        routes.append(current_route)

    feasible = len(routes) <= problem.vehicle_count
    if not feasible:
        extra = len(routes) - problem.vehicle_count
        penalized = total_dist + EXTRA_ROUTE_PENALTY * extra
        return penalized, routes, False

    return total_dist, routes, True


def evaluate(solution, problem):
    total_dist, routes, feasible = decode_giant_tour(solution, problem)
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
