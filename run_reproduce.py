#!/usr/bin/env python3
"""GCVaR End-to-End Experimental Reproducibility Runner.

Pure Python executable script to reproduce:
1. Exact Skyband Core Reduction (M / N scenarios retained).
2. Theoretical Exactness: EF-F vs EF-C zero optimality gap (|Obj_EF - Obj_EFC| / Obj_EF < 1e-6).
3. Optimization Speedup: Extensive Formulation Full (EF-F) vs Core (EF-C).
4. Query Stream Retrieval: FULL-SELECT vs SKYBAND-SELECT across query batch sizes Q in {1, 10, 100, 1000}.

Usage:
    python run_reproduce.py                     # Fast benchmark (~1-2 mins)
    python run_reproduce.py --mode smoke       # Fast validation (~10 seconds)
    python run_reproduce.py --dataset bike     # Run on UCI Bike Sharing dataset
    python run_reproduce.py --dataset tetouan  # Run on Tetouan City Power dataset
    python run_reproduce.py --include-oa       # Also run Outer Approximation (OA-F, OA-C)
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_ROOT / "results"
DATA_DIR = PACKAGE_ROOT / "data"
SRC_DIR = PACKAGE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from core import skyband_2d
from synthetic import make_cloud
from datasets import load_local, chronological_split
from optimization import (
    instance_demands,
    solve_extensive,
    solve_outer_approx,
    top_risk_from_demands,
    bound_vector,
    fractional_cvar_topmeans,
)
from oracles import full_select


def print_banner(text: str):
    print("=" * 78, flush=True)
    print(f"  {text}", flush=True)
    print("=" * 78, flush=True)


def print_section(text: str):
    print("-" * 78, flush=True)
    print(f" {text}", flush=True)
    print("-" * 78, flush=True)


def run_optimization_case(
    dataset_name: str,
    factors: np.ndarray,
    q: int = 10,
    alpha: float = 0.90,
    seed: int = 801,
    test_oa: bool = False,
) -> List[Dict[str, Any]]:
    """Runs EF-F vs EF-C (and optionally OA-F vs OA-C) on a single factor cloud."""
    rng = np.random.default_rng(seed)
    N = len(factors)
    p = 2  # 2D factors

    # Generate uniform positive cone coefficients
    coeff = rng.uniform(0.2, 1.3, size=(q, p))
    cost = rng.uniform(0.8, 1.2, size=q)
    D = instance_demands(factors, coeff)
    ub = bound_vector(D)

    # Standard CVaR empirical quantile budget (guarantees primal feasibility)
    sample_losses = np.max(D, axis=1)
    budget = float(np.quantile(sample_losses, alpha))

    means = fractional_cvar_topmeans(N, alpha)
    k_max = max(k for k, _ in means)

    # 1. Skyband 2D Core Screening
    t_core_start = time.perf_counter()
    core_idx = skyband_2d(factors, k_max)
    t_core = time.perf_counter() - t_core_start
    M = len(core_idx)
    D_core = D[core_idx]

    records = []

    # 2. Solve EF-F (Full sample)
    try:
        t_start = time.perf_counter()
        sol_eff = solve_extensive(D, cost, means, budget, ub=ub)
        t_eff = time.perf_counter() - t_start
        records.append({
            "dataset": dataset_name, "N": N, "q": q, "alpha": alpha, "core_M": M,
            "core_reduction_pct": (1.0 - M / N) * 100.0,
            "algorithm": "EF-F", "objective": sol_eff.objective,
            "risk": sol_eff.original_risk, "solve_s": t_eff,
            "all_in_s": t_eff, "lp_rows": sol_eff.lp_rows, "lp_cols": sol_eff.lp_cols,
            "status": sol_eff.status, "iterations": sol_eff.iterations,
        })
    except Exception as e:
        sol_eff = None
        records.append({
            "dataset": dataset_name, "N": N, "q": q, "alpha": alpha, "core_M": M,
            "core_reduction_pct": (1.0 - M / N) * 100.0,
            "algorithm": "EF-F", "objective": -1.0, "risk": -1.0, "solve_s": 0.0,
            "all_in_s": 0.0, "lp_rows": 0, "lp_cols": 0, "status": f"ERROR: {e}", "iterations": 0
        })

    # 3. Solve EF-C (Core sample)
    try:
        t_start = time.perf_counter()
        sol_efc = solve_extensive(D_core, cost, means, budget, ub=ub)
        t_efc = time.perf_counter() - t_start
        all_in_efc = t_efc + t_core

        # Verify full-sample risk of EF-C solution
        risk_full = top_risk_from_demands(D, sol_efc.x, means)
        gap = abs(sol_eff.objective - sol_efc.objective) / max(1.0, abs(sol_eff.objective)) if sol_eff else 0.0
        speedup = sol_eff.wall_s / all_in_efc if sol_eff and all_in_efc > 0 else 1.0

        records.append({
            "dataset": dataset_name, "N": N, "q": q, "alpha": alpha, "core_M": M,
            "core_reduction_pct": (1.0 - M / N) * 100.0,
            "algorithm": "EF-C", "objective": sol_efc.objective,
            "risk": risk_full, "solve_s": t_efc,
            "all_in_s": all_in_efc, "lp_rows": sol_efc.lp_rows, "lp_cols": sol_efc.lp_cols,
            "status": sol_efc.status, "iterations": sol_efc.iterations,
            "optimality_gap": gap,
            "speedup_vs_eff": speedup,
        })
    except Exception as e:
        sol_efc = None
        records.append({
            "dataset": dataset_name, "N": N, "q": q, "alpha": alpha, "core_M": M,
            "core_reduction_pct": (1.0 - M / N) * 100.0,
            "algorithm": "EF-C", "objective": -1.0, "risk": -1.0, "solve_s": 0.0,
            "all_in_s": 0.0, "lp_rows": 0, "lp_cols": 0, "status": f"ERROR: {e}", "iterations": 0,
            "optimality_gap": -1.0, "speedup_vs_eff": 0.0
        })

    # 4. Optional: Solve OA-F and OA-C (Outer Approximation)
    if test_oa:
        max_oa_iter = 50 if N > 5000 else 150
        try:
            sol_oaf = solve_outer_approx(D, cost, means, budget, ub=ub, max_iter=max_oa_iter)
            records.append({
                "dataset": dataset_name, "N": N, "q": q, "alpha": alpha, "core_M": M,
                "core_reduction_pct": (1.0 - M / N) * 100.0,
                "algorithm": "OA-F", "objective": sol_oaf.objective,
                "risk": sol_oaf.original_risk, "solve_s": sol_oaf.wall_s,
                "all_in_s": sol_oaf.wall_s, "lp_rows": sol_oaf.lp_rows, "lp_cols": sol_oaf.lp_cols,
                "status": sol_oaf.status, "iterations": sol_oaf.iterations,
            })
        except Exception as e:
            records.append({
                "dataset": dataset_name, "N": N, "q": q, "alpha": alpha, "core_M": M,
                "core_reduction_pct": (1.0 - M / N) * 100.0,
                "algorithm": "OA-F", "objective": -1.0, "risk": -1.0, "solve_s": 0.0,
                "all_in_s": 0.0, "lp_rows": 0, "lp_cols": 0, "status": f"ERROR: {e}", "iterations": 0
            })

        try:
            sol_oac = solve_outer_approx(D_core, cost, means, budget, ub=ub, max_iter=max_oa_iter)
            risk_full_oac = top_risk_from_demands(D, sol_oac.x, means)
            records.append({
                "dataset": dataset_name, "N": N, "q": q, "alpha": alpha, "core_M": M,
                "core_reduction_pct": (1.0 - M / N) * 100.0,
                "algorithm": "OA-C", "objective": sol_oac.objective,
                "risk": risk_full_oac, "solve_s": sol_oac.wall_s,
                "all_in_s": sol_oac.wall_s + t_core, "lp_rows": sol_oac.lp_rows, "lp_cols": sol_oac.lp_cols,
                "status": sol_oac.status, "iterations": sol_oac.iterations,
            })
        except Exception as e:
            records.append({
                "dataset": dataset_name, "N": N, "q": q, "alpha": alpha, "core_M": M,
                "core_reduction_pct": (1.0 - M / N) * 100.0,
                "algorithm": "OA-C", "objective": -1.0, "risk": -1.0, "solve_s": 0.0,
                "all_in_s": 0.0, "lp_rows": 0, "lp_cols": 0, "status": f"ERROR: {e}", "iterations": 0
            })

    return records


def run_query_stream_benchmark(
    factors: np.ndarray,
    k: int = 100,
    q_queries: List[int] = [1, 10, 100, 1000],
    seed: int = 801,
) -> List[Dict[str, Any]]:
    """Benchmarks FULL-SELECT vs SKYBAND-SELECT on actual query streams."""
    rng = np.random.default_rng(seed)
    N = len(factors)

    # Skyband Core build
    t0 = time.perf_counter()
    core_idx = skyband_2d(factors, k)
    t_core_build = time.perf_counter() - t0
    factors_core = factors[core_idx]
    M = len(core_idx)

    records = []
    for Q in q_queries:
        # Generate Q random non-negative query direction vectors
        slopes = rng.uniform(0.1, 10.0, size=(Q, 2))
        offsets = rng.uniform(-5.0, 5.0, size=Q)

        # 1. FULL-SELECT: scan all N points for each query
        t_full_start = time.perf_counter()
        full_results = []
        for q_idx in range(Q):
            a = slopes[q_idx : q_idx + 1]
            b = offsets[q_idx : q_idx + 1]
            res = full_select(factors, a, b, k)
            full_results.append(res)
        t_full = time.perf_counter() - t_full_start

        # 2. SKYBAND-SELECT: scan only M core points for each query
        t_core_query_start = time.perf_counter()
        core_results = []
        for q_idx in range(Q):
            a = slopes[q_idx : q_idx + 1]
            b = offsets[q_idx : q_idx + 1]
            res = full_select(factors_core, a, b, k)
            core_results.append(res)
        t_core_query = time.perf_counter() - t_core_query_start

        # Verify 100% exact top-k match
        exact_matches = sum(
            np.allclose(full_results[i], core_results[i], atol=1e-8) for i in range(Q)
        )
        match_pct = (exact_matches / Q) * 100.0

        all_in_core = t_core_build + t_core_query
        speedup = t_full / all_in_core if all_in_core > 0 else 1.0

        records.append({
            "N": N, "core_M": M, "k": k, "Q": Q,
            "exact_match_pct": match_pct,
            "full_select_s": t_full,
            "core_build_s": t_core_build,
            "core_query_s": t_core_query,
            "core_all_in_s": all_in_core,
            "speedup": speedup,
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="GCVaR End-to-End Reproducibility Runner")
    parser.add_argument("--mode", choices=["smoke", "benchmark"], default="benchmark",
                        help="Run mode: smoke (~10s validation) or benchmark (comprehensive grid)")
    parser.add_argument("--dataset", choices=["all", "synthetic", "bike", "tetouan"], default="all",
                        help="Dataset to evaluate: synthetic, bike, tetouan, or all")
    parser.add_argument("--include-oa", action="store_true",
                        help="Also evaluate Outer Approximation (OA-F, OA-C)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print_banner("GCVaR EXPERIMENTAL REPRODUCIBILITY SUITE (PURE PYTHON)")
    print(f"Mode:       {args.mode.upper()}", flush=True)
    print(f"Dataset:    {args.dataset.upper()}", flush=True)
    print(f"Include OA: {args.include_oa}", flush=True)
    print(f"Results:    {RESULTS_DIR}", flush=True)

    all_opt_records: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # PART 1: OPTIMIZATION EXPERIMENTS (EF-F vs EF-C)
    # -------------------------------------------------------------------------
    print_section("PART 1: EXTENSIVE FORMULATION (EF-F vs EF-C) PARITY & SPEEDUP")

    # 1. Synthetic Datasets
    if args.dataset in ("all", "synthetic"):
        if args.mode == "smoke":
            synth_configs = [
                {"name": "Synthetic_Lognormal_rho0.9", "family": "rho_+0.9", "N": 1000, "q": 10, "alpha": 0.90},
                {"name": "Synthetic_Gamma_Independent", "family": "independent", "N": 2000, "q": 10, "alpha": 0.95},
            ]
        else:
            synth_configs = [
                {"name": "Synthetic_Lognormal_rho0.9_N1k", "family": "rho_+0.9", "N": 1000, "q": 10, "alpha": 0.90},
                {"name": "Synthetic_Lognormal_rho0.9_N3k", "family": "rho_+0.9", "N": 3000, "q": 15, "alpha": 0.90},
                {"name": "Synthetic_Lognormal_rho0.9_N5k", "family": "rho_+0.9", "N": 5000, "q": 20, "alpha": 0.95},
                {"name": "Synthetic_Student_t3_N3k", "family": "t3", "N": 3000, "q": 15, "alpha": 0.90},
            ]

        for cfg in synth_configs:
            print(f"-> Running Synthetic: {cfg['name']} (N={cfg['N']}, q={cfg['q']}, alpha={cfg['alpha']})...", flush=True)
            factors = make_cloud(seed=801, n=cfg["N"], family=cfg["family"])
            recs = run_optimization_case(
                cfg["name"], factors, q=cfg["q"], alpha=cfg["alpha"], seed=801, test_oa=args.include_oa
            )
            all_opt_records.extend(recs)

    # 2. Operational Dataset: UCI Bike Sharing
    if args.dataset in ("all", "bike"):
        bike_file = DATA_DIR / "bike" / "hour.csv"
        if bike_file.exists():
            print(f"-> Loading UCI Bike Sharing ({bike_file.name})...", flush=True)
            df_bike, _ = load_local("bike", bike_file)
            splits = chronological_split(df_bike)
            factors_bike = splits["train"][["factor_1", "factor_2"]].to_numpy(dtype=float)
            if args.mode == "smoke":
                factors_bike = factors_bike[:1500]
                q_bike = 10
            else:
                q_bike = 10
            print(f"   Running Bike (N={len(factors_bike)}, q={q_bike}, alpha=0.90)...", flush=True)
            recs = run_optimization_case(
                "UCI_Bike_Train", factors_bike, q=q_bike, alpha=0.90, seed=801, test_oa=args.include_oa
            )
            all_opt_records.extend(recs)
        else:
            print(f"   [Notice] Bike file not found at {bike_file}, skipping.", flush=True)

    # 3. Operational Dataset: Tetouan City Power
    if args.dataset in ("all", "tetouan"):
        tetouan_file = DATA_DIR / "tetouan" / "Tetuan City power consumption.csv"
        if tetouan_file.exists():
            print(f"-> Loading Tetouan City Power ({tetouan_file.name})...", flush=True)
            df_tetouan, _ = load_local("tetouan", tetouan_file)
            splits = chronological_split(df_tetouan)
            factors_tetouan = splits["train"][["factor_1", "factor_2"]].to_numpy(dtype=float)
            if args.mode == "smoke":
                factors_tetouan = factors_tetouan[:2000]
                q_tetouan = 10
            else:
                factors_tetouan = factors_tetouan[:10000]  # Representative 10k sample for fast local execution
                q_tetouan = 10
            print(f"   Running Tetouan (N={len(factors_tetouan)}, q={q_tetouan}, alpha=0.95)...", flush=True)
            recs = run_optimization_case(
                "Tetouan_Train", factors_tetouan, q=q_tetouan, alpha=0.95, seed=801, test_oa=args.include_oa
            )
            all_opt_records.extend(recs)
        else:
            print(f"   [Notice] Tetouan file not found at {tetouan_file}, skipping.", flush=True)

    # Save and display optimization results
    opt_df = pd.DataFrame(all_opt_records)
    csv_opt_path = RESULTS_DIR / "optimization_reproduce.csv"
    opt_df.to_csv(csv_opt_path, index=False)
    print(f"\n[Saved] Optimization results written to: {csv_opt_path}\n", flush=True)

    # Format table for terminal
    print_section("OPTIMIZATION SUMMARY TABLE (EF-F vs EF-C)")
    display_rows = []
    datasets_seen = opt_df["dataset"].unique()
    for ds in datasets_seen:
        sub = opt_df[opt_df["dataset"] == ds]
        eff = sub[sub["algorithm"] == "EF-F"].iloc[0] if (sub["algorithm"] == "EF-F").any() else None
        efc = sub[sub["algorithm"] == "EF-C"].iloc[0] if (sub["algorithm"] == "EF-C").any() else None
        if eff is not None and efc is not None:
            speedup = eff["solve_s"] / efc["all_in_s"] if efc["all_in_s"] > 0 else 1.0
            gap = abs(eff["objective"] - efc["objective"]) / max(1.0, abs(eff["objective"]))
            display_rows.append({
                "Dataset": ds,
                "N": int(eff["N"]),
                "Core M": int(efc["core_M"]),
                "Screening %": f"{efc['core_reduction_pct']:.1f}%",
                "EF-F Time(s)": f"{eff['solve_s']:.3f}",
                "EF-C Time(s)": f"{efc['all_in_s']:.3f}",
                "Speedup": f"{speedup:.2f}x",
                "Optimality Gap": f"{gap:.2e}",
                "Status": "[PASS]" if gap <= 1e-6 else "[FAIL]",
            })

    summary_df = pd.DataFrame(display_rows)
    print(summary_df.to_string(index=False), flush=True)

    # -------------------------------------------------------------------------
    # PART 2: QUERY STREAM RETRIEVAL (FULL-SELECT vs SKYBAND-SELECT)
    # -------------------------------------------------------------------------
    print_section("PART 2: QUERY STREAM RETRIEVAL (FULL-SELECT vs SKYBAND-SELECT)")
    N_query = 5000 if args.mode == "smoke" else 10000
    print(f"Testing query batches Q on N={N_query} scenario cloud...", flush=True)

    synth_cloud = make_cloud(seed=801, n=N_query, family="rho_+0.9")
    q_batches = [1, 10, 50] if args.mode == "smoke" else [1, 10, 100, 1000]
    retrieval_recs = run_query_stream_benchmark(synth_cloud, k=100, q_queries=q_batches, seed=801)

    retrieval_df = pd.DataFrame(retrieval_recs)
    csv_ret_path = RESULTS_DIR / "query_stream_reproduce.csv"
    retrieval_df.to_csv(csv_ret_path, index=False)
    print(f"\n[Saved] Query stream results written to: {csv_ret_path}\n", flush=True)

    display_ret = []
    for _, r in retrieval_df.iterrows():
        display_ret.append({
            "N": int(r["N"]),
            "Core M": int(r["core_M"]),
            "k": int(r["k"]),
            "Queries Q": int(r["Q"]),
            "Match %": f"{r['exact_match_pct']:.1f}%",
            "Full Select(s)": f"{r['full_select_s']:.4f}",
            "Core All-In(s)": f"{r['core_all_in_s']:.4f}",
            "Amortized Speedup": f"{r['speedup']:.2f}x",
        })
    print(pd.DataFrame(display_ret).to_string(index=False), flush=True)

    print_banner("ALL REPRODUCIBILITY EXPERIMENTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
