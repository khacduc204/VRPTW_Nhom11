import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Customer:
    idx: int
    x: float
    y: float
    demand: float
    ready: float
    due: float
    service: float


@dataclass(frozen=True)
class VRPTWProblem:
    name: str
    vehicle_count: int
    capacity: float
    customers: list


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def load_simple_data():
    customers = [
        Customer(0, 50, 50, 0, 0, 999, 0),
        Customer(1, 10, 10, 10, 0, 200, 10),
        Customer(2, 20, 20, 10, 0, 200, 10),
        Customer(3, 30, 30, 10, 0, 200, 10),
        Customer(4, 40, 40, 10, 0, 200, 10),
    ]
    return VRPTWProblem(
        name="simple_demo",
        vehicle_count=25,
        capacity=50,
        customers=customers,
    )


def _parse_solomon_vehicle_info(lines):
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("VEHICLE"):
            for j in range(i + 1, min(i + 6, len(lines))):
                parts = lines[j].split()
                if len(parts) >= 2 and parts[0].isdigit():
                    return int(parts[0]), float(parts[1])
    raise ValueError("Cannot parse VEHICLE section in Solomon instance.")


def _parse_solomon_customers(lines):
    start = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("CUSTOMER"):
            start = i
            break

    if start is None:
        raise ValueError("Cannot find CUSTOMER section in Solomon instance.")

    customers = []
    for line in lines[start + 1 :]:
        parts = line.split()
        if len(parts) < 7:
            continue
        if not parts[0].lstrip("+-").isdigit():
            continue

        idx = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        demand = float(parts[3])
        ready = float(parts[4])
        due = float(parts[5])
        service = float(parts[6])
        customers.append(Customer(idx, x, y, demand, ready, due, service))

    if not customers:
        raise ValueError("No customer rows were parsed from Solomon instance.")

    customers.sort(key=lambda c: c.idx)
    if customers[0].idx != 0:
        raise ValueError("Expected depot with index 0 in Solomon instance.")

    return customers


def load_solomon_instance(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    name = lines[0].strip() if lines else "solomon"
    vehicle_count, capacity = _parse_solomon_vehicle_info(lines)
    customers = _parse_solomon_customers(lines)
    return VRPTWProblem(
        name=name,
        vehicle_count=vehicle_count,
        capacity=capacity,
        customers=customers,
    )
