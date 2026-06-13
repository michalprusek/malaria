#!/usr/bin/env python3
"""Vyhodnocení soutěže — INSTRUKTORSKÝ skript.

Spočítá pro každý tým SOUTĚŽNÍ METRIKU = specificita při senzitivitě >= 99 %.
Logika: u smrtelné nemoci je nepodkročitelný požadavek NEPŘEHLÉDNOUT nemocného,
proto fixujeme senzitivitu vysoko (>= 99 %, tj. zachytíme aspoň 99 % napadených)
a soutěžíme v tom, kdo přitom udrží co nejvyšší specificitu (nejméně falešných
poplachů). Je to bod na ROC křivce při senzitivitě 99 %.

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


def specificity_at_sensitivity(y_true, y_prob, min_sens=0.99):
    """Nejvyšší specificita dosažitelná při senzitivitě (recall) >= min_sens."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)          # tpr = senzitivita
    ok = tpr >= min_sens
    best = int(np.argmin(np.where(ok, fpr, 2.0)))      # nejmenší fpr (=max specificita)
    return float(1.0 - fpr[best]), float(thr[best]), float(tpr[best])


def load_submission(path):
    return np.loadtxt(path, skiprows=1)


def team_name(path):
    base = os.path.basename(path)
    name = base[len("predikce_"):] if base.startswith("predikce_") else base
    return os.path.splitext(name)[0]


def main():
    ap = argparse.ArgumentParser(description="Vyhodnocení soutěže v detekci malárie")
    ap.add_argument("--labels", default="test_labels.npz", help="npz s pravými labely testu")
    ap.add_argument("--subs", default=".", help="adresář s odevzdáními predikce_*.csv")
    ap.add_argument("--min-sens", type=float, default=0.99, help="požadovaná senzitivita")
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
        spec, thr, sens = specificity_at_sensitivity(y_true, prob, args.min_sens)
        results.append((name, spec, thr, sens))
        fpr, tpr, _ = roc_curve(y_true, prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name}  (spec={spec:.3f})")

    plt.axhline(args.min_sens, ls="--", color="crimson",
                label=f"požadovaná senzitivita {args.min_sens:.0%}")
    plt.plot([0, 1], [0, 1], ls=":", color="lightgray")
    plt.xlabel("1 − specificita (falešné poplachy)")
    plt.ylabel("senzitivita (zachycení nemocní)")
    plt.title("Soutěž v detekci malárie — ROC křivky týmů")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nGraf uložen do: {args.out}")

    # žebříček: vyšší specificita = lepší
    results.sort(key=lambda r: r[1], reverse=True)
    print("\n" + "=" * 64)
    print(f"  ŽEBŘÍČEK — specificita při senzitivitě >= {args.min_sens:.0%}")
    print("=" * 64)
    for rank, (name, spec, thr, sens) in enumerate(results, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        print(f"  {medal} {rank}. {name:20s}  specificita = {spec:.3f}   "
              f"(práh {thr:.3f}, dosažená senzitivita {sens:.3f})")
    print("=" * 64)
    if results:
        print(f"\n🏆 Vítěz: {results[0][0]}  se specificitou {results[0][1]:.3f}")


if __name__ == "__main__":
    main()
