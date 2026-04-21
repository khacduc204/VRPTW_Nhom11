import argparse
import csv
import glob
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from acs import run_acs
from data import load_simple_data, load_solomon_instance
from ga import run_ga
from meaf import run_meaf
from pso import run_pso
from utils import decode_giant_tour, set_use_numba

ALGORITHMS = [
    ("GA", run_ga),
    ("PSO", run_pso),
    ("ACS", run_acs),
    ("MEAF", run_meaf),
]

DEFAULT_SECTION6_INSTANCES = ["C101", "R101", "RC101", "C201", "R201", "RC201"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VRPTW experiments (simple demo or Solomon benchmark format)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "section6"],
        help="single: run one instance; section6: run a benchmark list and export CSV.",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Path to Solomon .txt instance (e.g. data/solomon/C101.txt).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/solomon",
        help="Root directory containing Solomon instance .txt files.",
    )
    parser.add_argument(
        "--instances",
        type=str,
        default="all",
        help="Comma-separated instance names (without .txt), or 'all' to run every .txt under dataset-root.",
    )
    parser.add_argument(
        "--sample-instances",
        type=int,
        default=None,
        help="Run only first N instances (sorted).",
    )
    parser.add_argument(
        "--max-customers",
        type=int,
        default=None,
        help="Limit customers to N (keeps depot + first N customers).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/section6_results.csv",
        help="Output CSV path for section6 mode.",
    )
    parser.add_argument("--runs", type=int, default=21, help="Independent runs per algorithm.")
    parser.add_argument("--iters", type=int, default=500, help="Iterations for each run.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--early-stop", type=int, default=None, help="Stop after N iterations without improvement.")
    parser.add_argument("--use-numba", action="store_true", help="Enable Numba-accelerated decode.")
    parser.add_argument("--parallel", action="store_true", help="Run instances in parallel (section6 mode only).")
    parser.add_argument("--workers", type=int, default=None, help="Worker count for parallel runs.")
    parser.add_argument("--no-heuristic-init", action="store_true", help="Disable heuristic seeding in GA/PSO/ACS.")
    return parser.parse_args()


def load_problem(instance_path, max_customers=None):
    if instance_path:
        if not os.path.exists(instance_path):
            raise FileNotFoundError(f"Instance file not found: {instance_path}")
        return load_solomon_instance(instance_path, max_customers=max_customers)
    return load_simple_data()


def evaluate_algorithm(
    problem,
    func,
    name,
    runs,
    iters,
    seed,
    early_stop=None,
    return_runs=False,
    return_history=False,
    init_heuristic=True,
):
    results = []
    run_details = []
    history = None
    best_sol = None
    best_cost = float("inf")

    for r in range(runs):
        if return_history and r == 0:
            out = func(
                problem,
                iters=iters,
                seed=seed + r,
                early_stop=early_stop,
                return_history=True,
                init_heuristic=init_heuristic,
            )
            cost, sol, history = out
        else:
            cost, sol = func(
                problem,
                iters=iters,
                seed=seed + r,
                early_stop=early_stop,
                init_heuristic=init_heuristic,
            )
        results.append(cost)
        if return_runs:
            run_details.append({"seed": seed + r, "cost": cost})
        if best_sol is None or cost < best_cost:
            best_cost = cost
            best_sol = sol

    if best_sol is None:
        return {
            "algorithm": name,
            "avg_objective": float("inf"),
            "std_objective": float("inf"),
            "best_objective": float("inf"),
            "best_distance": float("inf"),
            "vehicles_used": -1,
            "feasible": False,
        }

    route_dist, routes, feasible = decode_giant_tour(best_sol, problem)

    payload = {
        "algorithm": name,
        "avg_objective": float(np.mean(results)),
        "std_objective": float(np.std(results)),
        "best_objective": float(best_cost),
        "best_distance": float(route_dist),
        "vehicles_used": len(routes),
        "feasible": bool(feasible),
    }
    if return_runs:
        payload["runs"] = run_details
    if return_history:
        payload["history"] = history
    return payload


def print_algorithm_result(row):
    print(f"{row['algorithm']}:")
    print(f"  Avg objective = {row['avg_objective']:.4f}")
    print(f"  Std objective = {row['std_objective']:.4f}")
    print(f"  Best objective = {row['best_objective']:.4f}")
    print(f"  Best distance = {row['best_distance']:.4f}")
    print(f"  Vehicles used = {row['vehicles_used']}")
    print(f"  Feasible = {row['feasible']}")
    print("-" * 42)


def run_single(problem, runs, iters, seed, early_stop=None, init_heuristic=True):
    print(f"Instance: {problem.name}")
    print(f"Customers: {len(problem.customers) - 1}, Capacity: {problem.capacity}")
    print("=" * 42)

    for name, func in ALGORITHMS:
        row = evaluate_algorithm(
            problem,
            func,
            name,
            runs,
            iters,
            seed,
            early_stop=early_stop,
            init_heuristic=init_heuristic,
        )
        print_algorithm_result(row)


def _find_instance_file(dataset_root, instance_name):
    exact = os.path.join(dataset_root, f"{instance_name}.txt")
    if os.path.exists(exact):
        return exact

    candidates = glob.glob(os.path.join(dataset_root, "**", "*.txt"), recursive=True)
    instance_name_l = instance_name.lower()
    for p in candidates:
        if os.path.splitext(os.path.basename(p))[0].lower() == instance_name_l:
            return p
    return None


def _discover_all_instances(dataset_root):
    candidates = glob.glob(os.path.join(dataset_root, "**", "*.txt"), recursive=True)
    names = sorted({os.path.splitext(os.path.basename(p))[0].upper() for p in candidates})
    return names


def _run_instance(args):
    dataset_root, ins_name, runs, iters, seed, early_stop, max_customers, init_heuristic = args
    ins_file = _find_instance_file(dataset_root, ins_name)
    if ins_file is None:
        return []
    problem = load_solomon_instance(ins_file, max_customers=max_customers)
    rows = []
    for algo_name, algo_func in ALGORITHMS:
        result = evaluate_algorithm(
            problem,
            algo_func,
            algo_name,
            runs,
            iters,
            seed,
            early_stop=early_stop,
            init_heuristic=init_heuristic,
        )
        row = {
            "instance": ins_name,
            "algorithm": algo_name,
            "avg_objective": result["avg_objective"],
            "std_objective": result["std_objective"],
            "best_objective": result["best_objective"],
            "best_distance": result["best_distance"],
            "vehicles_used": result["vehicles_used"],
            "feasible": result["feasible"],
            "vehicle_limit": problem.vehicle_count,
            "capacity": problem.capacity,
        }
        rows.append(row)
    return rows


def run_section6(
    dataset_root,
    instance_names,
    runs,
    iters,
    seed,
    output_csv,
    early_stop=None,
    parallel=False,
    workers=None,
    max_customers=None,
    init_heuristic=True,
):
    rows = []

    print("Section6 benchmark mode")
    print(f"Dataset root: {dataset_root}")
    print("=" * 42)

    if parallel:
        work = [(dataset_root, ins_name, runs, iters, seed, early_stop, max_customers, init_heuristic) for ins_name in instance_names]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, res in enumerate(ex.map(_run_instance, work)):
                print(f"Running instance {instance_names[i]} ({i + 1}/{len(instance_names)})")
                rows.extend(res)
    else:
        for i, ins_name in enumerate(instance_names):
            ins_file = _find_instance_file(dataset_root, ins_name)
            if ins_file is None:
                print(f"[WARN] Missing instance: {ins_name}.txt")
                continue

            problem = load_solomon_instance(ins_file, max_customers=max_customers)
            print(f"Running instance {ins_name} ({i + 1}/{len(instance_names)})")

            for algo_name, algo_func in ALGORITHMS:
                result = evaluate_algorithm(
                    problem,
                    algo_func,
                    algo_name,
                    runs,
                    iters,
                    seed,
                    early_stop=early_stop,
                    init_heuristic=init_heuristic,
                )
                row = {
                    "instance": ins_name,
                    "algorithm": algo_name,
                    "avg_objective": result["avg_objective"],
                    "std_objective": result["std_objective"],
                    "best_objective": result["best_objective"],
                    "best_distance": result["best_distance"],
                    "vehicles_used": result["vehicles_used"],
                    "feasible": result["feasible"],
                    "vehicle_limit": problem.vehicle_count,
                    "capacity": problem.capacity,
                }
                rows.append(row)

    if not rows:
        print("No section6 results generated.")
        print("Please put Solomon .txt files in dataset-root and run again.")
        return

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "algorithm",
                "avg_objective",
                "std_objective",
                "best_objective",
                "best_distance",
                "vehicles_used",
                "feasible",
                "vehicle_limit",
                "capacity",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_csv}")

    # Export Table-3 style summary layout (one row per instance).
    table3_path = os.path.splitext(output_csv)[0] + "_table3.csv"
    by_instance = {}
    for r in rows:
        by_instance.setdefault(r["instance"], {})[r["algorithm"]] = r

    ordered_instances = sorted(by_instance.keys())
    table3_rows = []
    for ins in ordered_instances:
        d = by_instance[ins]
        acs = d.get("ACS", {})
        pso = d.get("PSO", {})
        ga = d.get("GA", {})
        meaf = d.get("MEAF", {})

        acs_avg = acs.get("avg_objective", float("inf"))
        pso_avg = pso.get("avg_objective", float("inf"))
        ga_avg = ga.get("avg_objective", float("inf"))
        meaf_avg = meaf.get("avg_objective", float("inf"))

        baseline = min(acs_avg, pso_avg, ga_avg)
        if baseline == float("inf") or meaf_avg == float("inf"):
            reduction = float("nan")
        else:
            reduction = (baseline - meaf_avg) * 100.0 / baseline

        table3_rows.append(
            {
                "Inst.": ins,
                "ACS_Avg": acs_avg,
                "ACS_Std": acs.get("std_objective", float("inf")),
                "PSO_Avg": pso_avg,
                "PSO_Std": pso.get("std_objective", float("inf")),
                "GA_Avg": ga_avg,
                "GA_Std": ga.get("std_objective", float("inf")),
                "MEAVRPTW_Avg": meaf_avg,
                "MEAVRPTW_Std": meaf.get("std_objective", float("inf")),
                "cost_reduction_percent": reduction,
            }
        )

    with open(table3_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Inst.",
                "ACS_Avg",
                "ACS_Std",
                "PSO_Avg",
                "PSO_Std",
                "GA_Avg",
                "GA_Std",
                "MEAVRPTW_Avg",
                "MEAVRPTW_Std",
                "cost_reduction_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(table3_rows)

    print(f"Saved Table3-style CSV: {table3_path}")


def main():
    args = parse_args()
    set_use_numba(args.use_numba)
    init_heuristic = not args.no_heuristic_init
    if args.mode == "single":
        problem = load_problem(args.instance, max_customers=args.max_customers)
        run_single(
            problem,
            args.runs,
            args.iters,
            args.seed,
            early_stop=args.early_stop,
            init_heuristic=init_heuristic,
        )
    else:
        if args.instances.strip().lower() == "all":
            instance_names = _discover_all_instances(args.dataset_root)
            if not instance_names:
                # Keep prior behavior for empty folders while preserving old defaults.
                instance_names = DEFAULT_SECTION6_INSTANCES
        else:
            instance_names = [x.strip() for x in args.instances.split(",") if x.strip()]

        if args.sample_instances is not None and args.sample_instances > 0:
            instance_names = instance_names[: args.sample_instances]

        run_section6(
            dataset_root=args.dataset_root,
            instance_names=instance_names,
            runs=args.runs,
            iters=args.iters,
            seed=args.seed,
            output_csv=args.output_csv,
            early_stop=args.early_stop,
            parallel=args.parallel,
            workers=args.workers,
            max_customers=args.max_customers,
            init_heuristic=init_heuristic,
        )


if __name__ == "__main__":
    main()
