#!/usr/bin/env python3
"""Vyhodnocení soutěže — INSTRUKTORSKÝ skript.

Spočítá pro každý tým SOUTĚŽNÍ METRIKU = senzitivita při specificitě >= 95 %
(tj. bod ROC křivky na FPR = 5 %) na SKRYTÉM testu, seřadí žebříček a vykreslí
ROC křivky všech týmů do jednoho grafu pro finální vyhlášení.

Použití:
    python3 score_submission_INSTRUCTOR.py
    python3 score_submission_INSTRUCTOR.py --labels test_labels.npz --subs .

Očekává:
    - test_labels.npz  (klíč 'y' s pravými labely testu — máte jen vy)
    - predikce_*.csv   (odevzdání týmů: hlavička 'prob' + pravděpodobnosti, řádek na buňku)
"""
import argparse
import glob
import os

import numpy as np
from sklearn.metrics import roc_curve


def sensitivity_at_specificity(y_true, y_prob, min_spec=0.95):
    """Nejvyšší senzitivita, kterou lze dosáhnout při specificitě >= min_spec."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)        # fpr = 1 - specificita
    ok = fpr <= (1.0 - min_spec)
    best = int(np.argmax(np.where(ok, tpr, -1.0)))
    return float(tpr[best]), float(thr[best]), float(1.0 - fpr[best])


def load_submission(path):
    """Načte odevzdání (hlavička 'prob' + hodnoty) → 1D pole pravděpodobností."""
    return np.loadtxt(path, skiprows=1)


def team_name(path):
    base = os.path.basename(path)
    name = base[len("predikce_"):] if base.startswith("predikce_") else base
    return os.path.splitext(name)[0]


def main():
    ap = argparse.ArgumentParser(description="Vyhodnocení soutěže v detekci malárie")
    ap.add_argument("--labels", default="test_labels.npz", help="npz s pravými labely testu")
    ap.add_argument("--subs", default=".", help="adresář s odevzdáními predikce_*.csv")
    ap.add_argument("--min-spec", type=float, default=0.95, help="požadovaná specificita")
    ap.add_argument("--out", default="vysledky_soutez.png", help="kam uložit graf ROC")
    args = ap.parse_args()

    y_true = np.load(args.labels)["y"].astype(int)
    files = sorted(glob.glob(os.path.join(args.subs, "predikce_*.csv")))
    if not files:
        raise SystemExit(f"Nenašel jsem žádné predikce_*.csv v '{args.subs}'.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = []
    plt.figure(figsize=(7, 7))
    for path in files:
        name = team_name(path)
        prob = load_submission(path)
        if len(prob) != len(y_true):
            print(f"⚠️  {name}: {len(prob)} predikcí, ale test má {len(y_true)} buněk — přeskakuji.")
            continue
        sens, thr, spec = sensitivity_at_specificity(y_true, prob, args.min_spec)
        results.append((name, sens, thr, spec))
        fpr, tpr, _ = roc_curve(y_true, prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name}  (sens={sens:.3f})")

    plt.axvline(1 - args.min_spec, ls="--", color="gray",
                label=f"hranice specificity {args.min_spec:.0%}")
    plt.plot([0, 1], [0, 1], ls=":", color="lightgray")
    plt.xlabel("1 − specificita (falešné poplachy)")
    plt.ylabel("senzitivita (zachycení nemocní)")
    plt.title("Soutěž v detekci malárie — ROC křivky týmů")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nGraf uložen do: {args.out}")

    # žebříček
    results.sort(key=lambda r: r[1], reverse=True)
    print("\n" + "=" * 60)
    print(f"  ŽEBŘÍČEK — senzitivita při specificitě >= {args.min_spec:.0%}")
    print("=" * 60)
    for rank, (name, sens, thr, spec) in enumerate(results, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        print(f"  {medal} {rank}. {name:20s}  senzitivita = {sens:.3f}   "
              f"(práh {thr:.3f}, specificita {spec:.3f})")
    print("=" * 60)
    if results:
        print(f"\n🏆 Vítěz: {results[0][0]}  se senzitivitou {results[0][1]:.3f}")


if __name__ == "__main__":
    main()
