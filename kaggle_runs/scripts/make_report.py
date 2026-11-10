"""Render markdown tables from stage1 metrics.json + stage2 adapt_report.json."""
import json, sys, argparse

KEYS = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]
VARIANT_LABEL = {
    ("base", "global"): "Baseline global (paper)",
    ("base", "local"): "Baseline + local fusion (no adapt)",
    ("base", "fuse"): "Baseline + 0.5 fusion (no adapt)",
    ("adapted", "global"): "Adapted global",
    ("adapted", "local"): "Adapted local fusion",
    ("adapted", "fuse"): "Adapted 0.5 fusion",
    ("extra", "global"): "Full-image extra global",
    ("extra", "local"): "Full-image extra local fusion",
    ("extra", "fuse"): "Full-image extra 0.5 fusion",
}


def pct(x):
    return f"{x*100:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="stage1 metrics.json")
    ap.add_argument("--adapt", required=True, help="stage2 adapt_report.json")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    m1 = json.load(open(args.base, encoding="utf-8"))
    r2 = json.load(open(args.adapt, encoding="utf-8"))
    lines = []

    # Stage 1
    lines.append("### Stage 1 — official-protocol 10k baseline (I-AUROC/I-AP/I-F1/P-AUROC/P-AP/P-F1/P-AUPRO, %)\n")
    lines.append("| iter | " + " | ".join(KEYS) + " |")
    lines.append("|---" * (len(KEYS) + 1) + "|")
    for it, m in sorted(m1.items(), key=lambda kv: int(kv[0])):
        lines.append(f"| {it} | " + " | ".join(pct(m["mean"][k]) for k in KEYS) + " |")

    # Stage 1 per class at the last iteration
    last = sorted(m1.items(), key=lambda kv: int(kv[0]))[-1]
    lines.append(f"\n### Stage 1 per class @ {last[0]}\n")
    lines.append("| class | " + " | ".join(KEYS) + " |")
    lines.append("|---" * (len(KEYS) + 1) + "|")
    for cls, m in last[1]["per_class"].items():
        lines.append(f"| {cls} | " + " | ".join(pct(m[k]) for k in KEYS) + " |")

    # Stage 2 means
    lines.append("\n### Stage 2 — 15-class means\n")
    lines.append("| variant | " + " | ".join(KEYS) + " |")
    lines.append("|---" * (len(KEYS) + 1) + "|")
    for (tag, v) in [("base", "global"), ("base", "local"), ("base", "fuse"),
                     ("adapted", "local"), ("adapted", "fuse"), ("adapted", "global"),
                     ("extra", "global"), ("extra", "local"), ("extra", "fuse")]:
        mean = r2["results"][tag]["mean"][v]["metrics"]
        lines.append(f"| {VARIANT_LABEL[(tag, v)]} | " + " | ".join(pct(mean[k]) for k in KEYS) + " |")

    # recall / regions
    lines.append("\n### Recall at fixed pixel FPR (class mean, %)\n")
    lines.append("| variant | R@FPR0.1% | R@FPR1% | R@FPR5% |")
    lines.append("|---|---|---|---|")
    for (tag, v) in [("base", "global"), ("base", "local"), ("base", "fuse"),
                     ("adapted", "local"), ("adapted", "fuse"),
                     ("extra", "local"), ("extra", "fuse")]:
        rec = r2["results"][tag]["mean"][v]["recall"]
        lines.append(f"| {VARIANT_LABEL[(tag, v)]} | " + " | ".join(pct(rec[k]) for k in
                     ("R@FPR0.001", "R@FPR0.01", "R@FPR0.05")) + " |")

    lines.append("\n### Mean anomaly score by region (class mean)\n")
    lines.append("| variant | interior | inner_edge | outer_neighborhood | far_normal |")
    lines.append("|---|---|---|---|---|")
    for (tag, v) in [("base", "global"), ("base", "local"), ("base", "fuse"),
                     ("adapted", "local"), ("adapted", "fuse"),
                     ("extra", "local"), ("extra", "fuse")]:
        reg = r2["results"][tag]["mean"][v]["regions"]
        lines.append(f"| {VARIANT_LABEL[(tag, v)]} | " + " | ".join(
            f"{reg[k]:.4f}" for k in ("interior", "inner_edge", "outer_neighborhood", "far_normal")) + " |")

    # deltas
    lines.append("\n### Headline deltas (percentage points, class-mean)\n")
    lines.append("| comparison | " + " | ".join(KEYS) + " |")
    lines.append("|---" * (len(KEYS) + 1) + "|")
    for name, d in r2["deltas"].items():
        lines.append(f"| {name} | " + " | ".join(f"{d[k]*100:+.2f}" for k in KEYS) + " |")

    # per-class deltas
    lines.append("\n### Per-class pixel AP delta: adapted local minus baseline global (pp)\n")
    lines.append("| class | P-AP Δ | P-F1 Δ | P-AUROC Δ | P-AUPRO Δ |")
    lines.append("|---|---|---|---|---|")
    for cls, d in r2["per_class_delta_adapted_local_minus_base_global"].items():
        lines.append(f"| {cls} | " + " | ".join(
            f"{d[k]*100:+.2f}" for k in ("P-AP", "P-F1", "P-AUROC", "P-AUPRO")) + " |")

    lines.append(f"\nIdentity check (global image scores, base vs adapted): max |Δ| = "
                 f"{r2['identity']['global_score_max_abs_diff_base_vs_adapted']:.3e}\n")
    lines.append("Train stats: " + json.dumps(r2["train"], indent=2, ensure_ascii=False))

    text = "\n".join(lines)
    if args.out == "-":
        print(text)
    else:
        open(args.out, "w", encoding="utf-8").write(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
