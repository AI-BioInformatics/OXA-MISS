"""
Module A, retrospective evaluation of modality acquisition (docs/design_rationale.md).

For the test patients of every fold with at least two modalities: start from the cheapest available
modality, then acquire the others one at a time, in the order chosen by each policy:
    voi             largest expected reduction of the risk variance (model.value_of_acquisition; fusion_type poe)
    voi_cost        the same divided by the cost of the exam
    random          random order (seeded)
    fixed           cost order (cheapest first)
    hindsight       the best single order for everyone, chosen on the fold's TRAINING patients (best mean
                    c-index over the acquisition steps) and applied unchanged to the test patients: the
                    achievable upper bound of a fixed policy
    oracle_outcome  per patient, the modality that most decreases its survival NLL, using its outcome: not
                    achievable (it can exceed the complete model), reported only as a bound
Only modalities the patient really has can be "acquired". After every step: c-index, integrated Brier score,
D-calibration, mean predicted risk std and mean cumulative cost over the test patients (a patient with fewer
modalities keeps its complete state). Costs are relative placeholders: set them with --costs.

Each patient's prediction is computed once for every subset of its modalities that contains the starting one
(at most 2^(n-1)); the policies then only look them up (voi also calls the cross-modal predictors).

Usage (GPU, SLURM):
    python utils/acquisition_eval.py --run_dir /work/.../results/<run> [--checkpoint last_epoch]
        [--costs '{"Clinical": 1, "CT": 5, "MRI": 8, "WSI": 10, "Genomics": 20}']
"""
import argparse
import contextlib
import io
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "utils"))
from dataloader.dataloader_utils import make_dataloader  # noqa: E402
from experiments.utils import extended_survival_metrics, move_to_device, safe_c_index  # noqa: E402
from shortcut_analysis import CHECKPOINTS, iterate_run_folds  # noqa: E402

STATUS_KEYS = {"WSI": "WSI_status", "Genomics": "genomics_status", "CNV": "cnv_status",
               "CT": "ct_status", "MRI": "mri_status", "Clinical": "clinical_status"}
DEFAULT_COSTS = {"Clinical": 1, "CT": 5, "MRI": 8, "WSI": 10, "Genomics": 20, "CNV": 20}
POLICIES = ["voi", "voi_cost", "random", "fixed", "hindsight", "oracle_outcome"]


def with_modalities(data, modalities):
    """Copy of a batch input where only `modalities` are available (the others: status False)."""
    data = dict(data)
    for modality, key in STATUS_KEYS.items():
        if key in data and modality not in modalities:
            data[key] = torch.zeros_like(data[key])
    return data


def nll(logits, label, censorship):
    """Discrete-time survival NLL of one patient (Zadeh & Schmid 2020, alpha = 0)."""
    h = 1 / (1 + np.exp(-np.asarray(logits, dtype=float)))
    h = np.clip(h, 1e-7, 1 - 1e-7)
    S = np.cumprod(1 - h)
    S_prev = np.concatenate([[1.0], S[:-1]])
    return float(-((1 - censorship) * np.log(S_prev[label] * h[label]) + censorship * np.log(S[label])))


def risk_of(logits):
    return -np.cumprod(1 - 1 / (1 + np.exp(-np.asarray(logits, dtype=float))), axis=-1).sum(axis=-1)


def collect(model, loader, modalities, costs, device, with_voi):
    """Per patient with >= 2 modalities: outcome, available modalities, the prediction (logits, risk std)
    for every subset containing the starting (cheapest) modality and, with_voi, the voi / voi_cost paths."""
    patients = []
    with torch.no_grad():
        for batch in loader:
            data = move_to_device(batch["input"], device)
            available = [m for m in modalities if STATUS_KEYS[m] in data and bool(data[STATUS_KEYS[m]].item())]
            if len(available) < 2:
                continue
            by_cost = sorted(available, key=lambda m: costs.get(m, 1))
            start, rest = by_cost[0], by_cost[1:]
            cache = {}
            for r in range(len(rest) + 1):
                for extra in itertools.combinations(rest, r):
                    subset = frozenset((start,) + extra)
                    out = model(with_modalities(data, subset))
                    cache[subset] = (out["output"][0].float().cpu().numpy(),
                                     float(out["risk_std"].item()) if "risk_std" in out else np.nan)
            p = {"patient": batch["patient_id"][0], "label": int(batch["label"].item()),
                 "censorship": float(batch["censorship"].item()), "time": float(batch["original_event_time"].item()),
                 "available": available, "by_cost": by_cost, "cache": cache, "paths": {}}
            if with_voi:
                for policy in ("voi", "voi_cost"):
                    acquired = [start]
                    while len(acquired) < len(available):
                        value = model.value_of_acquisition(with_modalities(data, acquired))["value"]
                        remaining = [m for m in available if m not in acquired]
                        score = {m: float(value[m].item()) / (costs.get(m, 1) if policy == "voi_cost" else 1)
                                 for m in remaining if m in value}
                        acquired.append(max(score, key=score.get) if score else remaining[0])
                    p["paths"][policy] = acquired
            patients.append(p)
    return patients


def path_for(p, policy, rng, order=None, costs=None):
    """Acquisition order (list of modalities, starting modality first) of a patient under a policy."""
    start = p["by_cost"][0]
    rest = [m for m in p["available"] if m != start]
    if policy in p["paths"]:
        return p["paths"][policy]
    if policy == "fixed":
        return p["by_cost"]
    if policy == "random":
        return [start] + list(rng.permutation(rest))
    if policy == "hindsight":
        return [start] + sorted(rest, key=order.index)
    if policy == "oracle_outcome":
        acquired = [start]
        while len(acquired) < len(p["available"]):
            remaining = [m for m in p["available"] if m not in acquired]
            acquired.append(min(remaining, key=lambda m: nll(p["cache"][frozenset(acquired + [m])][0],
                                                              p["label"], p["censorship"])))
        return acquired
    raise ValueError(policy)


def curve(patients, paths, max_step):
    """c-index of the patients at each step (a patient with fewer modalities keeps its complete state)."""
    out = []
    for step in range(max_step + 1):
        logits = np.stack([p["cache"][frozenset(paths[p["patient"]][:min(step, len(paths[p["patient"]]) - 1) + 1])][0]
                           for p in patients])
        out.append(safe_c_index([p["censorship"] for p in patients], [p["time"] for p in patients], risk_of(logits)))
    return out


def best_order_in_hindsight(patients, modalities, max_step):
    """The global modality order with the best mean c-index over the acquisition steps on these patients."""
    best, best_score = None, -np.inf
    for order in itertools.permutations(modalities):
        paths = {p["patient"]: [p["by_cost"][0]] + sorted([m for m in p["available"] if m != p["by_cost"][0]],
                                                            key=list(order).index) for p in patients}
        score = np.nanmean(curve(patients, paths, max_step)[1:])
        if score > best_score:
            best, best_score = list(order), score
    return best, best_score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default="last_epoch", choices=sorted(CHECKPOINTS))
    ap.add_argument("--costs", default=None, help="JSON {modality: cost}")
    args = ap.parse_args()
    costs = {**DEFAULT_COSTS, **(json.loads(args.costs) if args.costs else {})}
    rng = np.random.default_rng(0)
    rows, steps = [], []
    for k, config, dataset, model, train, val, test, device in iterate_run_folds(args.run_dir, args.checkpoint):
        modalities = list(config.model.kwargs.input_modalities)
        has_voi = hasattr(model, "value_of_acquisition") and getattr(model, "fusion_type", None) == "poe"
        policies = POLICIES if has_voi else [p for p in POLICIES if not p.startswith("voi")]
        reference = dataset.patient_df.loc[[p for p in np.concatenate([x for x in (train, val) if x is not None])
                                            if p in dataset.patient_df.index]]
        reference = (reference["censorship"].values, reference["time"].values)
        with contextlib.redirect_stdout(io.StringIO()):
            train_loader = make_dataloader(dataset, "train", train, modalities, config)
            test_loader = make_dataloader(dataset, "test", test, modalities, config)
        train_patients = collect(model, train_loader, modalities, costs, device, with_voi=False)
        test_patients = collect(model, test_loader, modalities, costs, device, with_voi=has_voi)
        max_step = max(len(p["available"]) for p in test_patients) - 1
        order, train_score = best_order_in_hindsight(train_patients, modalities,
                                                     max(len(p["available"]) for p in train_patients) - 1)
        print(f"fold {k + 1}: {len(test_patients)} test patients with >= 2 modalities; best order in hindsight "
              f"(training patients, mean c-index {train_score:.3f}): {' > '.join(order)}")
        for policy in policies:
            paths = {p["patient"]: path_for(p, policy, rng, order, costs) for p in test_patients}
            for step in range(max_step + 1):
                states = [paths[p["patient"]][:min(step, len(paths[p["patient"]]) - 1) + 1] for p in test_patients]
                cached = [p["cache"][frozenset(s)] for p, s in zip(test_patients, states)]
                logits = np.stack([c[0] for c in cached])
                metrics = extended_survival_metrics(logits, [p["censorship"] for p in test_patients],
                                                    [p["time"] for p in test_patients], config.data_loader.survival_bins,
                                                    *reference)
                rows.append({"fold": k + 1, "policy": policy, "step": step, "patients": len(test_patients),
                             "c-index": safe_c_index([p["censorship"] for p in test_patients],
                                                     [p["time"] for p in test_patients], risk_of(logits)),
                             "IBS": metrics["IBS"], "D-cal_p": metrics["D-cal_p"],
                             "risk_std": np.nanmean([c[1] for c in cached]),
                             "cost": np.mean([sum(costs.get(m, 1) for m in s) for s in states])})
                steps.extend({"fold": k + 1, "policy": policy, "step": step, "patient": p["patient"],
                              "acquired": "+".join(s), "risk_std": c[1]}
                             for p, s, c in zip(test_patients, states, cached))
            if policy == "hindsight":
                rows[-1]["order"] = " > ".join(order)
    curves = pd.DataFrame(rows)
    summary = curves.groupby(["policy", "step"])[["c-index", "IBS", "D-cal_p", "risk_std", "cost"]].agg(["mean", "std"]).round(3)
    print("\nAcquisition curves (mean, std over folds; oracle_outcome uses the outcome, not achievable):\n" + summary.to_string())
    out = os.path.join(args.run_dir, f"acquisition_{args.checkpoint}")
    curves.to_csv(out + "_curves.csv", index=False)
    pd.DataFrame(steps).to_csv(out + "_steps.csv", index=False)
    print(f"-> {out}_curves.csv, {out}_steps.csv")


if __name__ == "__main__":
    main()
