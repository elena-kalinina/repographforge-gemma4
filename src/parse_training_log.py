#!/usr/bin/env python3
"""Extract a loss curve from a training log, for manuscript §7.2.

The log may be a Kaggle notebook's captured stdout, HuggingFace `Trainer` dict lines,
JSONL, or our own `[HH:MM:SS] INFO - ...` format. We do not know which, so this parser
tries all of them and REPORTS what it matched rather than silently returning an empty
series — reporting a confident zero where the truth is unknown is failure F5.

Usage:
    python3 src/parse_training_log.py experiments/logs/train_log_sft_rft.log
    python3 src/parse_training_log.py <log> --plot experiments/plots/loss_curve.png
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

# {'loss': 2.046, ...} and {'loss': '2.046', ...} — this trainer quotes its numbers,
# which the first version of this pattern rejected, so it found nothing and said so.
DICT_LINE = re.compile(r"\{[^{}]*['\"]loss['\"]\s*:\s*['\"]?[-\d.eE+]+['\"]?[^{}]*\}")
FINAL_LINE = re.compile(r"\{[^{}]*['\"]train_loss['\"][^{}]*\}")
# A run boundary: a bare caption line the operator typed between pasted cell outputs.
NOISE_PREFIX = ("[", "{", "Loading", "Map:", "Generating", "trainable", "=", "-", "✓", "🚀")
DATASET_N = re.compile(r"(?:Prepared|Loaded)\s+(\d+)\s+(?:verifier-curated Best-of-N |training )?traject")
# loss=2.046 ... step=12 ... epoch=0.11   (any order, any separators)
KV_LOSS = re.compile(r"\bloss\s*[=:]\s*([-\d.]+(?:[eE][-+]?\d+)?)")
KV_STEP = re.compile(r"\b(?:global_step|step)\s*[=:]\s*(\d+)")
KV_EPOCH = re.compile(r"\bepoch\s*[=:]\s*([\d.]+)")


def _num(v: Any) -> Any:
    """This trainer emits every value as a string. Coerce what looks numeric."""
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _is_header(ln: str) -> bool:
    t = ln.strip()
    if not t or t.startswith(NOISE_PREFIX) or "|" in t or "%" in t:
        return False
    return not t.startswith(("Note:", "Active", "Base Model", "Training Data", "Output"))


def parse(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    points: List[Dict[str, Any]] = []
    matched_by = {"dict": 0, "json": 0, "kv": 0}
    runs: List[Dict[str, Any]] = []
    cur = {"label": "(unlabelled)", "points": [], "n_examples": None, "final": None}

    for ln in lines:
        if _is_header(ln):
            if cur["points"] or cur["final"]:
                runs.append(cur)
            cur = {"label": ln.strip(), "points": [], "n_examples": None, "final": None}
            continue
        dn = DATASET_N.search(ln)
        if dn:
            cur["n_examples"] = int(dn.group(1))
        fm = FINAL_LINE.search(ln)
        if fm:
            try:
                cur["final"] = {k: _num(v) for k, v in
                                ast.literal_eval(fm.group(0)).items()}
            except Exception:
                pass
        rec = None
        m = DICT_LINE.search(ln)
        if m:
            for loader in (ast.literal_eval, json.loads):
                try:
                    rec = loader(m.group(0).replace("'", '"') if loader is json.loads else m.group(0))
                    matched_by["dict" if loader is ast.literal_eval else "json"] += 1
                    break
                except Exception:
                    rec = None
        if rec is None and ln.strip().startswith("{"):
            try:
                cand = json.loads(ln)
                if isinstance(cand, dict) and "loss" in cand:
                    rec, _ = cand, matched_by.__setitem__("json", matched_by["json"] + 1)
            except Exception:
                rec = None
        if rec is None:
            lm = KV_LOSS.search(ln)
            if lm:
                rec = {"loss": float(lm.group(1))}
                sm, em = KV_STEP.search(ln), KV_EPOCH.search(ln)
                if sm:
                    rec["step"] = int(sm.group(1))
                if em:
                    rec["epoch"] = float(em.group(1))
                matched_by["kv"] += 1
        if isinstance(rec, dict):
            rec = {k: _num(v) for k, v in rec.items()}
        if isinstance(rec, dict) and isinstance(rec.get("loss"), (int, float)):
            pt = {
                "run": cur["label"],
                "step": rec.get("step") or rec.get("global_step"),
                "epoch": rec.get("epoch"),
                "loss": float(rec["loss"]),
                "learning_rate": rec.get("learning_rate"),
            }
            points.append(pt)
            cur["points"].append(pt)

    if cur["points"] or cur["final"]:
        runs.append(cur)

    for r in runs:                              # per-run step numbering
        for i, p in enumerate(r["points"], 1):
            if p["step"] is None:
                p["step"] = i
    for i, p in enumerate(points, 1):
        if p["step"] is None:
            p["step"] = i
    return {"points": points, "runs": runs, "matched_by": matched_by,
            "total_lines": len(lines)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--label", default="SFT", help="series label for the figure")
    args = ap.parse_args()

    res = parse(args.log)
    pts = res["points"]
    print(f"{args.log}: {res['total_lines']} lines, {len(pts)} loss point(s)")
    print(f"  matched by: {res['matched_by']}")
    if not pts:
        print("\n  NO LOSS POINTS FOUND. The format is not one of the four this parser")
        print("  knows. Paste ~20 representative lines and the pattern can be added —")
        print("  do not report an empty curve as evidence of anything.")
        return

    print("\n  Per-run breakdown:")
    print(f"    {'run':<52}{'n':>6}{'pts':>5}{'first':>9}{'last':>9}{'final':>9}{'sec':>8}")
    for r in res["runs"]:
        rp, fin = r["points"], (r["final"] or {})
        if not rp and not fin:
            continue
        print(f"    {r['label'][:52]:<52}{str(r['n_examples']):>6}{len(rp):>5}"
              f"{(f'{rp[0][chr(39)+chr(39)]}' if False else f'{rp[0]["loss"]:.4f}') if rp else '-':>9}"
              f"{f'{rp[-1]["loss"]:.4f}' if rp else '-':>9}"
              f"{fin.get('train_loss', '-'):>9}{fin.get('train_runtime', '-'):>8}")

    first, last = pts[0], pts[-1]
    lo = min(p["loss"] for p in pts)
    print(f"  first: step {first['step']} loss {first['loss']:.4f}")
    print(f"  last : step {last['step']} loss {last['loss']:.4f}")
    print(f"  min  : {lo:.4f}   range: {first['loss']:.4f} -> {last['loss']:.4f}")
    if first["loss"] > 0:
        print(f"  reduction: {100 * (first['loss'] - last['loss']) / first['loss']:.1f}%")

    out_csv = args.csv or args.log.with_suffix(".loss.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "step", "epoch", "loss", "learning_rate"])
        w.writeheader()
        w.writerows(pts)
    print(f"  wrote {out_csv}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (matplotlib unavailable; CSV written, plot skipped)")
            return
        runs = [r for r in res["runs"] if r["points"]]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
        for r in runs:
            pts = r["points"]
            is_rft = "RFT" in r["label"].upper()
            ax1.plot(range(1, len(pts) + 1), [p["loss"] for p in pts],
                     lw=2.2 if is_rft else 1.4,
                     ls="--" if is_rft else "-",
                     marker="o" if is_rft else None, ms=4,
                     label=f"{r['label'][:34]}"
                           + (f" (n={r['n_examples']})" if r["n_examples"] else ""))
        ax1.set_xlabel("logged step within run")
        ax1.set_ylabel("training loss")
        ax1.set_title("Every run converges")
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=6.2, loc="upper right")

        finals = [(i + 1, r["final"]["train_loss"], "RFT" in r["label"].upper(),
                   r["n_examples"]) for i, r in enumerate(runs) if r.get("final")]
        ax2.bar([f[0] for f in finals], [f[1] for f in finals],
                color=["#c0392b" if f[2] else "#2c6fbb" for f in finals])
        for x, y, is_rft, n in finals:
            ax2.text(x, y + 0.02, f"{y:.3f}" + (f"\nn={n}" if is_rft else ""),
                     ha="center", fontsize=7)
        ax2.set_xlabel("run, in chronological order")
        ax2.set_ylabel("final train_loss")
        ax2.set_title("Final loss per run (red = RFT)")
        ax2.grid(alpha=0.3, axis="y")
        fig.suptitle("Training loss across the campaign; end-task resolution stayed ~0", y=1.0)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=160)
        print(f"  wrote {args.plot}")


if __name__ == "__main__":
    main()
