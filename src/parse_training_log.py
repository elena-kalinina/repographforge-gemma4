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

# {'loss': 2.046, 'grad_norm': 1.2, 'learning_rate': 9e-05, 'epoch': 0.11}
DICT_LINE = re.compile(r"\{[^{}]*['\"]loss['\"]\s*:\s*[-\d.eE+]+[^{}]*\}")
# loss=2.046 ... step=12 ... epoch=0.11   (any order, any separators)
KV_LOSS = re.compile(r"\bloss\s*[=:]\s*([-\d.]+(?:[eE][-+]?\d+)?)")
KV_STEP = re.compile(r"\b(?:global_step|step)\s*[=:]\s*(\d+)")
KV_EPOCH = re.compile(r"\bepoch\s*[=:]\s*([\d.]+)")


def parse(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    points: List[Dict[str, Any]] = []
    matched_by = {"dict": 0, "json": 0, "kv": 0}

    for ln in lines:
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
        if isinstance(rec, dict) and isinstance(rec.get("loss"), (int, float)):
            points.append({
                "step": rec.get("step") or rec.get("global_step"),
                "epoch": rec.get("epoch"),
                "loss": float(rec["loss"]),
                "learning_rate": rec.get("learning_rate"),
            })

    for i, p in enumerate(points, 1):          # synthesise step when absent
        if p["step"] is None:
            p["step"] = i
    return {"points": points, "matched_by": matched_by, "total_lines": len(lines)}


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
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "loss", "learning_rate"])
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
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([p["step"] for p in pts], [p["loss"] for p in pts], lw=1.4, label=args.label)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("training loss")
        ax.set_title("Training loss converges while end-task resolution stays ~0")
        ax.grid(alpha=0.3)
        ax.legend()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=160)
        print(f"  wrote {args.plot}")


if __name__ == "__main__":
    main()
