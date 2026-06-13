#!/usr/bin/env python3
"""Sestaví self-contained HTML prezentaci (prezentace.html).

Vygeneruje figury laděné do tmavého motivu (PCA projekce featur, ROC), spočítá
reálné srovnání k-NN vs MLP na testovací sadě, vše zakóduje do base64 a vloží do
jediného HTML souboru (žádné externí závislosti kromě webových fontů).
"""
import base64, glob, io, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

TEAL = "#35d6c0"
CRIMSON = "#f0476b"
INK = "#cdd6e2"
MUTED = "#8b98a8"
GRID = "#26303d"

torch.manual_seed(42)
np.random.seed(42)


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img_to_b64(path, size=240):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def style_axes(ax):
    ax.set_facecolor("none")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)


# ---------- data ----------
tr = np.load("train.npz"); te = np.load("test_features.npz"); tl = np.load("test_labels.npz")
Xtr, ytr = tr["X"].astype(np.float32), tr["y"]
Xte, yte = te["X"].astype(np.float32), tl["y"]
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

# ---------- PCA projekce featur (2D) ----------
idx = np.random.choice(len(Xtr), 4000, replace=False)
pts = PCA(n_components=2).fit_transform(Xtr_s[idx])
yy = ytr[idx]
fig, ax = plt.subplots(figsize=(7.4, 5.4))
ax.scatter(pts[yy == 0, 0], pts[yy == 0, 1], s=9, alpha=0.40, c=TEAL, label="zdravá", edgecolors="none")
ax.scatter(pts[yy == 1, 0], pts[yy == 1, 1], s=9, alpha=0.40, c=CRIMSON, label="napadená", edgecolors="none")
ax.set_xlabel("hlavní komponenta 1"); ax.set_ylabel("hlavní komponenta 2")
leg = ax.legend(loc="upper right", frameon=False, fontsize=12)
for t in leg.get_texts(): t.set_color(INK)
style_axes(ax)
PCA_B64 = fig_to_b64(fig)

# ---------- k-NN + ROC ----------
knn = KNeighborsClassifier(n_neighbors=31, weights="distance", n_jobs=-1).fit(Xtr_s, ytr)
p_knn = knn.predict_proba(Xte_s)[:, 1]
tn, fp, fn, tp = confusion_matrix(yte, (p_knn >= 0.5).astype(int), labels=[0, 1]).ravel()
knn_acc = (((p_knn >= 0.5).astype(int)) == yte).mean()
fpr, tpr, _ = roc_curve(yte, p_knn); ok = fpr <= 0.05; b = int(np.argmax(np.where(ok, tpr, -1)))
knn_sens95, knn_spec95 = tpr[b], 1 - fpr[b]; knn_auc = roc_auc_score(yte, p_knn)

# ---------- MLP (naučená hlava) na testu ----------
Xtr_t = torch.tensor(Xtr_s); ytr_t = torch.tensor(ytr, dtype=torch.float32)
Xte_t = torch.tensor(Xte_s)
mlp = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss()
for ep in range(30):
    mlp.train(); perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(Xtr_t), 256):
        j = perm[i:i + 256]; opt.zero_grad()
        lossf(mlp(Xtr_t[j]).squeeze(1), ytr_t[j]).backward(); opt.step()
mlp.eval()
with torch.no_grad():
    p_mlp = torch.sigmoid(mlp(Xte_t).squeeze(1)).numpy()
fpr_m, tpr_m, _ = roc_curve(yte, p_mlp); okm = fpr_m <= 0.05; bm = int(np.argmax(np.where(okm, tpr_m, -1)))
mlp_sens95 = tpr_m[bm]; mlp_auc = roc_auc_score(yte, p_mlp)
mlp_acc = (((p_mlp >= 0.5).astype(int)) == yte).mean()

# ROC obě křivky
fig, ax = plt.subplots(figsize=(7.0, 5.4))
ax.plot(fpr, tpr, lw=2.4, color=MUTED, label=f"k-NN  (AUC {knn_auc:.2f})")
ax.plot(fpr_m, tpr_m, lw=2.6, color=TEAL, label=f"MLP  (AUC {mlp_auc:.2f})")
ax.axvline(0.05, ls="--", lw=1.2, color=CRIMSON, alpha=0.8)
ax.scatter([1 - knn_spec95], [knn_sens95], color=MUTED, zorder=5, s=40)
ax.scatter([fpr_m[bm]], [tpr_m[bm]], color=TEAL, zorder=5, s=55, edgecolors="white", linewidths=0.6)
ax.plot([0, 1], [0, 1], ls=":", color=GRID)
ax.set_xlabel("1 − specificita  (falešné poplachy)"); ax.set_ylabel("senzitivita")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
leg = ax.legend(loc="lower right", frameon=False, fontsize=12)
for t in leg.get_texts(): t.set_color(INK)
ax.text(0.055, 0.05, "hranice 5 % falešných poplachů", color=CRIMSON, fontsize=10, rotation=90, va="bottom")
style_axes(ax)
ROC_B64 = fig_to_b64(fig)

# ---------- vzorové buňky ----------
inf = sorted(glob.glob("samples/Parasitized/*.png"))[0]
hea = sorted(glob.glob("samples/Uninfected/*.png"))[0]
CELL_INF = img_to_b64(inf); CELL_OK = img_to_b64(hea)

print(f"k-NN: acc={knn_acc:.3f} sens95={knn_sens95:.3f} auc={knn_auc:.3f} (tn,fp,fn,tp={tn,fp,fn,tp})")
print(f"MLP : acc={mlp_acc:.3f} sens95={mlp_sens95:.3f} auc={mlp_auc:.3f}")

vals = dict(
    KNN_SENS=f"{knn_sens95:.2f}".replace(".", ","),
    MLP_SENS=f"{mlp_sens95:.2f}".replace(".", ","),
    KNN_ACC=f"{knn_acc:.2f}".replace(".", ","),
    MLP_ACC=f"{mlp_acc:.2f}".replace(".", ","),
    KNN_AUC=f"{knn_auc:.2f}".replace(".", ","),
    MLP_AUC=f"{mlp_auc:.2f}".replace(".", ","),
    TN=str(tn), FP=str(fp), FN=str(fn), TP=str(tp),
    KNN_SENS_PCT=str(round(knn_sens95 * 100)),
    MLP_SENS_PCT=str(round(mlp_sens95 * 100)),
)

# =====================================================================
HTML = r"""<!DOCTYPE html>
<html lang="cs">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detekce malárie — hluboké učení v diagnostice</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#090c12; --bg2:#0d131d; --ink:#eaf0f7; --muted:#93a1b3;
  --teal:#35d6c0; --teal-d:#1ea08f; --crimson:#f0476b; --line:rgba(149,167,182,.16);
  --serif:"Fraunces",Georgia,serif; --sans:"IBM Plex Sans",sans-serif; --mono:"IBM Plex Mono",monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  background:var(--bg); color:var(--ink); font-family:var(--sans);
  overflow:hidden; -webkit-font-smoothing:antialiased;
}
/* atmosféra: záře + jemné "buňky" + zrno */
.bg{position:fixed;inset:0;z-index:0;pointer-events:none}
.bg .glow{position:absolute;border-radius:50%;filter:blur(120px);opacity:.5}
.bg .g1{width:60vw;height:60vw;top:-22vw;left:-14vw;background:radial-gradient(circle,rgba(53,214,192,.20),transparent 70%)}
.bg .g2{width:55vw;height:55vw;bottom:-24vw;right:-12vw;background:radial-gradient(circle,rgba(240,71,107,.18),transparent 70%)}
.bg .grain{position:absolute;inset:0;opacity:.05;mix-blend-mode:overlay;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}

.deck{position:relative;z-index:1;height:100%}
.slide{
  position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;
  padding:clamp(28px,6vw,96px); opacity:0; transform:translateY(24px) scale(.992);
  transition:opacity .55s ease, transform .55s cubic-bezier(.2,.7,.2,1); pointer-events:none;
}
.slide.active{opacity:1;transform:none;pointer-events:auto}
.wrap{width:100%;max-width:1180px;margin:0 auto}

.kicker{font-family:var(--mono);font-size:.74rem;letter-spacing:.32em;text-transform:uppercase;
  color:var(--teal);display:flex;align-items:center;gap:.7em;margin-bottom:1.4rem}
.kicker::before{content:"";width:34px;height:1px;background:var(--teal);opacity:.7}

h1{font-family:var(--serif);font-weight:900;line-height:1.02;letter-spacing:-.02em;
  font-size:clamp(2.4rem,6.4vw,5.4rem)}
h2{font-family:var(--serif);font-weight:600;line-height:1.05;letter-spacing:-.015em;
  font-size:clamp(1.9rem,4.6vw,3.6rem);margin-bottom:1.1rem}
h3{font-family:var(--serif);font-weight:600;font-size:1.5rem;margin-bottom:.5rem}
p{color:#c7d2de;font-size:clamp(1rem,1.5vw,1.22rem);line-height:1.6;max-width:62ch}
.lead{font-size:clamp(1.1rem,1.9vw,1.5rem);color:#d7e0ea}
strong{color:#fff;font-weight:600}
.tt{font-family:var(--mono)}
.teal{color:var(--teal)} .crim{color:var(--crimson)}

/* reveal stagger */
.reveal{opacity:0;transform:translateY(16px);transition:.6s cubic-bezier(.2,.7,.2,1)}
.slide.active .reveal{opacity:1;transform:none}
.slide.active .reveal{transition-delay:calc(var(--i,0) * 90ms + 120ms)}

/* chips */
.chips{display:flex;flex-wrap:wrap;gap:.6rem;margin-top:1.6rem}
.chip{font-family:var(--mono);font-size:.8rem;border:1px solid var(--line);border-radius:999px;
  padding:.5em 1em;color:var(--muted);background:rgba(255,255,255,.02)}
.chip b{color:var(--ink);font-weight:500}
.role{font-family:var(--mono);font-size:.82rem;color:var(--teal);letter-spacing:.04em;margin-top:.3rem}
.teamnote{margin-top:1.5rem;text-align:center;color:var(--muted);font-size:1.02rem;max-width:none}
.card .list li{font-size:1rem}

.grid2{display:grid;grid-template-columns:1.05fr .95fr;gap:clamp(24px,4vw,64px);align-items:center}
@media(max-width:820px){.grid2{grid-template-columns:1fr;gap:28px}}

.bignum{font-family:var(--serif);font-weight:900;font-size:clamp(3.6rem,11vw,8rem);
  line-height:.9;background:linear-gradient(180deg,#fff,var(--teal));-webkit-background-clip:text;background-clip:text;color:transparent}
.list{list-style:none;display:flex;flex-direction:column;gap:.85rem;margin-top:1.2rem}
.list li{display:flex;gap:.8rem;align-items:flex-start;color:#cdd8e3;font-size:1.08rem;line-height:1.45}
.list li::before{content:"";flex:none;width:9px;height:9px;margin-top:.55em;border-radius:50%;
  background:var(--teal);box-shadow:0 0 0 4px rgba(53,214,192,.14)}

.card{border:1px solid var(--line);border-radius:18px;padding:clamp(18px,2.4vw,30px);
  background:linear-gradient(160deg,rgba(255,255,255,.035),rgba(255,255,255,.01))}
.figframe{border-radius:16px;overflow:hidden;border:1px solid var(--line);background:rgba(0,0,0,.25)}
.figframe img{display:block;width:100%}

/* vzorové buňky */
.cells{display:flex;gap:1.4rem;flex-wrap:wrap}
.cell{width:200px;max-width:42vw}
.cell .pic{border-radius:14px;overflow:hidden;border:2px solid var(--c);box-shadow:0 16px 40px rgba(0,0,0,.45)}
.cell img{display:block;width:100%;aspect-ratio:1;object-fit:cover}
.cell .lab{font-family:var(--mono);font-size:.82rem;margin-top:.7rem;color:var(--c);letter-spacing:.06em}

/* pipeline */
.pipe{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin-top:1.4rem}
.node{border:1px solid var(--line);border-radius:14px;padding:.9rem 1.1rem;background:rgba(255,255,255,.03);
  font-family:var(--mono);font-size:.92rem;text-align:center;min-width:96px}
.node small{display:block;color:var(--muted);font-size:.72rem;margin-top:.25rem}
.node.acc{border-color:rgba(53,214,192,.5);box-shadow:0 0 28px rgba(53,214,192,.10)}
.arrow{color:var(--teal);font-size:1.3rem}

/* schémata SVG */
.svgbox{width:100%;height:auto}

/* matice záměn */
.cm{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;max-width:360px;margin-top:1rem}
.cm .c{border-radius:12px;padding:1.1rem;text-align:center;font-family:var(--mono);border:1px solid var(--line)}
.cm .c b{display:block;font-size:1.7rem;color:#fff}
.cm .c span{font-size:.72rem;color:var(--muted)}
.cm .ok{background:rgba(53,214,192,.10)} .cm .bad{background:rgba(240,71,107,.14);border-color:rgba(240,71,107,.4)}

/* srovnání barů */
.bars{display:flex;flex-direction:column;gap:1.1rem;margin-top:.4rem}
.bar{}
.bar .top{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.92rem;margin-bottom:.35rem}
.bar .track{height:16px;border-radius:999px;background:rgba(255,255,255,.06);overflow:hidden}
.bar .fill{height:100%;border-radius:999px;transform-origin:left;transform:scaleX(0);transition:transform 1s cubic-bezier(.2,.7,.2,1) .3s}
.slide.active .bar .fill{transform:scaleX(var(--v))}
.fill.knn{background:linear-gradient(90deg,#5b6675,#93a1b3)}
.fill.mlp{background:linear-gradient(90deg,var(--teal-d),var(--teal))}

/* HUD */
.hud{position:fixed;z-index:5;left:0;right:0;bottom:0;height:3px;background:rgba(255,255,255,.06)}
.hud .pb{height:100%;background:linear-gradient(90deg,var(--teal),var(--crimson));width:0;transition:width .45s ease}
.counter{position:fixed;z-index:5;top:22px;right:26px;font-family:var(--mono);font-size:.78rem;color:var(--muted)}
.counter b{color:var(--ink)}
.brand{position:fixed;z-index:5;top:22px;left:26px;font-family:var(--mono);font-size:.72rem;letter-spacing:.2em;
  text-transform:uppercase;color:var(--muted)}
.dots{position:fixed;z-index:5;bottom:18px;left:50%;transform:translateX(-50%);display:flex;gap:9px}
.dots button{width:8px;height:8px;border-radius:50%;border:0;background:rgba(255,255,255,.18);cursor:pointer;padding:0;transition:.3s}
.dots button.on{background:var(--teal);transform:scale(1.35)}
.hint{position:fixed;z-index:5;bottom:16px;right:26px;font-family:var(--mono);font-size:.72rem;color:var(--muted);opacity:.7}
.foot{margin-top:1.6rem;font-family:var(--mono);font-size:.8rem;color:var(--muted)}
</style>
</head>
<body>
<div class="bg"><div class="glow g1"></div><div class="glow g2"></div><div class="grain"></div></div>
<div class="brand">Malárie · Deep&nbsp;Learning</div>
<div class="counter"><b id="cur">1</b> / <span id="tot">12</span></div>

<div class="deck" id="deck">

  <!-- 1 TITUL -->
  <section class="slide">
    <div class="wrap">
      <div class="kicker reveal" style="--i:0">Veletrh vědy · FJFI ČVUT</div>
      <h1 class="reveal" style="--i:1">Detekuje malárii lépe<br>člověk, nebo <span class="teal">notebook?</span></h1>
      <p class="lead reveal" style="--i:2;margin-top:1.2rem">Hluboké učení v diagnostice z mikroskopu — od pixelů buňky k rozhodnutí, které může zachránit život.</p>
      <div class="chips reveal" style="--i:3">
        <span class="chip"><b>Michal Průšek</b></span>
        <span class="chip"><b>Michal Bělohlávek</b></span>
        <span class="chip">školitelé · FJFI ČVUT</span>
      </div>
    </div>
  </section>

  <!-- 2 ŠKOLITELÉ (sdílený slide) -->
  <section class="slide">
    <div class="wrap">
      <div class="kicker reveal" style="--i:0">Kdo vás provede</div>
      <h2 class="reveal" style="--i:1">Dva školitelé, společný základ</h2>
      <div class="grid2" style="margin-top:.4rem">
        <div class="card reveal" style="--i:2">
          <h3>Michal Průšek</h3>
          <div class="role">PhD · FJFI ČVUT &nbsp;·&nbsp; ÚTIA AV ČR</div>
          <ul class="list" style="margin-top:.8rem">
            <li><span>Výzkum na <strong>ÚTIA AV ČR</strong> (Ústav teorie informace a automatizace).</span></li>
            <li><span><strong>Biomedicínská segmentace</strong> — učí počítače rozpoznávat buňky a struktury v mikroskopii.</span></li>
          </ul>
        </div>
        <div class="card reveal" style="--i:3">
          <h3>Michal Bělohlávek</h3>
          <div class="role">PhD · CIIRC ČVUT &nbsp;·&nbsp; Cisco</div>
          <ul class="list" style="margin-top:.8rem">
            <li><span><strong>Software Engineer v Cisco</strong> — detekce e-mailových hrozeb pomocí embeddingů.</span></li>
            <li><span>PhD na <strong>CIIRC</strong> (Formal Methods Group) — <strong>kontinuální učení</strong> v modelech pro uvažování (LLM reasoning).</span></li>
          </ul>
        </div>
      </div>
      <p class="reveal teamnote" style="--i:4">Spolužáci z oboru <strong>Aplikované matematicko-stochastické metody</strong> (Ing.) na FJFI ČVUT — odtud společná vášeň pro matematiku a strojové učení.</p>
    </div>
  </section>

  <!-- 3 PROBLÉM -->
  <section class="slide">
    <div class="wrap grid2">
      <div>
        <div class="kicker reveal" style="--i:0">Proč to řešíme</div>
        <div class="bignum reveal" style="--i:1">≈ 600 000</div>
        <p class="lead reveal" style="--i:2">úmrtí na malárii každý rok — většina tam, kde chybí vyškolení patologové.</p>
      </div>
      <div>
        <p class="reveal" style="--i:3">Diagnóza stojí na jediné, tisíckrát opakované otázce nad krevním nátěrem pod mikroskopem:</p>
        <h3 class="reveal teal" style="--i:4;margin-top:1rem">„Je tahle červená krvinka napadená parazitem?“</h3>
        <p class="reveal" style="--i:5;margin-top:1rem">Je to pomalé a vyžaduje experta. <strong>Co kdyby tu práci zvládl mobil s dobře natrénovaným algoritmem?</strong></p>
      </div>
    </div>
  </section>

  <!-- 4 DATA -->
  <section class="slide">
    <div class="wrap grid2">
      <div>
        <div class="kicker reveal" style="--i:0">Data</div>
        <h2 class="reveal" style="--i:1"><span class="tt">27 558</span> snímků buněk</h2>
        <p class="reveal" style="--i:2">Veřejný dataset Národního institutu zdraví (NIH). Každý snímek je jedna vysegmentovaná červená krvinka. Sada je <strong>dokonale vyvážená</strong> — polovina napadených, polovina zdravých.</p>
        <p class="reveal" style="--i:3;margin-top:.8rem;color:#93a1b3">U napadené buňky bývá vidět drobná fialová tečka — parazit <em>Plasmodium</em>.</p>
      </div>
      <div class="cells reveal" style="--i:3">
        <div class="cell" style="--c:var(--crimson)">
          <div class="pic"><img src="__CELL_INF__" alt="napadená buňka"></div>
          <div class="lab">● napadená</div>
        </div>
        <div class="cell" style="--c:var(--teal)">
          <div class="pic"><img src="__CELL_OK__" alt="zdravá buňka"></div>
          <div class="lab">● zdravá</div>
        </div>
      </div>
    </div>
  </section>

  <!-- 5 NÁPAD / PIPELINE -->
  <section class="slide">
    <div class="wrap">
      <div class="kicker reveal" style="--i:0">Nápad</div>
      <h2 class="reveal" style="--i:1">Nech „vidění“ na předtrénované síti,<br>uč jen rozhodování</h2>
      <p class="reveal" style="--i:2">Trénovat síť od nuly potřebuje miliony obrázků. My místo toho vezmeme hotový <strong>ResNet</strong>, použijeme ho jako překladač „obrázek → čísla“, a doučíme jen malý klasifikátor.</p>
      <div class="pipe reveal" style="--i:3">
        <div class="node">buňka<small>224×224 px</small></div>
        <div class="arrow">→</div>
        <div class="node">ResNet50<small>zmrazený</small></div>
        <div class="arrow">→</div>
        <div class="node acc">2048 čísel<small>featury</small></div>
        <div class="arrow">→</div>
        <div class="node">klasifikátor<small>k-NN / MLP</small></div>
        <div class="arrow">→</div>
        <div class="node">zdravá /<br>napadená</div>
      </div>
    </div>
  </section>

  <!-- 6 CNN / ResNet -->
  <section class="slide">
    <div class="wrap">
      <div class="kicker reveal" style="--i:0">Encoder · konvoluční síť</div>
      <h2 class="reveal" style="--i:1">ResNet: z pixelů postupně skládá smysl</h2>
      <div class="grid2" style="margin-top:.4rem">
        <div>
          <ul class="list">
            <li class="reveal" style="--i:2"><span><strong>Konvoluce</strong> klouže po obrázku a hledá vzory — nejdřív hrany, pak textury, pak tvary.</span></li>
            <li class="reveal" style="--i:3"><span><strong>Reziduální spojení</strong> (zkratky) umožní síť opravdu hluboko — 50 vrstev — aniž by se „ztratil signál“.</span></li>
            <li class="reveal" style="--i:4"><span>Na konci zůstane <span class="teal">2048 čísel</span> shrnujících, co na buňce je.</span></li>
          </ul>
        </div>
        <div class="reveal" style="--i:3">
          <svg class="svgbox" viewBox="0 0 460 220" fill="none" font-family="IBM Plex Mono">
            <defs><linearGradient id="gst" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stop-color="#1ea08f"/><stop offset="1" stop-color="#35d6c0"/></linearGradient></defs>
            <rect x="6" y="70" width="40" height="80" rx="6" fill="rgba(255,255,255,.06)" stroke="var(--line)"/>
            <text x="26" y="170" fill="#93a1b3" font-size="10" text-anchor="middle">vstup</text>
            <rect x="66" y="60" width="34" height="100" rx="6" fill="rgba(53,214,192,.10)" stroke="rgba(53,214,192,.4)"/>
            <rect x="112" y="74" width="34" height="72" rx="6" fill="rgba(53,214,192,.12)" stroke="rgba(53,214,192,.4)"/>
            <rect x="158" y="86" width="34" height="48" rx="6" fill="rgba(53,214,192,.14)" stroke="rgba(53,214,192,.4)"/>
            <rect x="204" y="94" width="34" height="32" rx="6" fill="rgba(53,214,192,.16)" stroke="rgba(53,214,192,.4)"/>
            <text x="150" y="200" fill="#93a1b3" font-size="10" text-anchor="middle">reziduální bloky (hloubka ↑, rozlišení ↓)</text>
            <path d="M70 50 q40 -22 80 0" stroke="#35d6c0" stroke-width="1.4" fill="none" opacity=".7"/>
            <path d="M116 52 q40 -20 80 0" stroke="#35d6c0" stroke-width="1.4" fill="none" opacity=".7"/>
            <circle cx="300" cy="110" r="26" fill="rgba(255,255,255,.05)" stroke="var(--line)"/>
            <text x="300" y="114" fill="#cdd6e2" font-size="9" text-anchor="middle">pool</text>
            <rect x="350" y="86" width="100" height="48" rx="8" fill="url(#gst)"/>
            <text x="400" y="114" fill="#06231f" font-size="12" font-weight="600" text-anchor="middle">2048-D</text>
          </svg>
        </div>
      </div>
    </div>
  </section>

  <!-- 7 TRANSFER LEARNING -->
  <section class="slide">
    <div class="wrap grid2">
      <div>
        <div class="kicker reveal" style="--i:0">Přenosové učení</div>
        <h2 class="reveal" style="--i:1">Zmrazený mozek,<br>učíme jen rozhodnutí</h2>
        <p class="reveal" style="--i:2">ResNet už někdo natrénoval na milionech běžných fotek (psi, auta, židle). Naučil se <strong>vidět</strong> — a to se hodí i na buňky.</p>
        <p class="reveal" style="--i:3;margin-top:.8rem">Jeho váhy <span class="crim">zmrazíme</span> (neučí se), takže nám stačí slabý notebook a žádné miliony medicínských snímků. Učí se jen malá hlava na konci.</p>
      </div>
      <div class="reveal" style="--i:3">
        <div class="card">
          <div style="display:flex;gap:1rem;align-items:center;font-family:var(--mono)">
            <span style="font-size:1.5rem">🧊</span>
            <div><strong>ResNet50 — zmrazený</strong><br><small style="color:#93a1b3">25 M vah se neučí</small></div>
          </div>
          <div style="height:1px;background:var(--line);margin:1.1rem 0"></div>
          <div style="display:flex;gap:1rem;align-items:center;font-family:var(--mono)">
            <span style="font-size:1.5rem">🔥</span>
            <div><strong class="teal">Hlava — trénovaná</strong><br><small style="color:#93a1b3">k-NN nebo malá MLP</small></div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- 8 PCA -->
  <section class="slide">
    <div class="wrap grid2">
      <div>
        <div class="kicker reveal" style="--i:0">Už to vidí</div>
        <h2 class="reveal" style="--i:1">Featury samy<br>tvoří dva shluky</h2>
        <p class="reveal" style="--i:2">2048 rozměrů promítneme metodou <span class="tt">PCA</span> do roviny a obarvíme podle skutečné třídy.</p>
        <p class="reveal" style="--i:3;margin-top:.8rem">Zdravé (<span class="teal">tyrkysové</span>) a napadené (<span class="crim">červené</span>) se oddělují <strong>ještě než cokoli trénujeme</strong>. ResNet už odvedl většinu práce — zbývá najít dělící čáru.</p>
      </div>
      <div class="figframe reveal" style="--i:3"><img src="__PCA__" alt="PCA projekce featur"></div>
    </div>
  </section>

  <!-- 9 kNN -->
  <section class="slide">
    <div class="wrap grid2">
      <div>
        <div class="kicker reveal" style="--i:0">Baseline · k-NN</div>
        <h2 class="reveal" style="--i:1">k nejbližších sousedů</h2>
        <p class="reveal" style="--i:2">Nejjednodušší možný model: <strong>vůbec se netrénuje</strong>. Novou buňku zařadí tak, že najde <span class="teal">k nejpodobnějších</span> a nechá je hlasovat.</p>
        <svg class="svgbox reveal" style="--i:3;max-width:340px;margin-top:1rem" viewBox="0 0 300 170">
          <circle cx="150" cy="85" r="58" fill="none" stroke="var(--line)" stroke-dasharray="4 4"/>
          <circle cx="120" cy="60" r="7" fill="var(--crimson)"/><circle cx="180" cy="70" r="7" fill="var(--crimson)"/>
          <circle cx="115" cy="110" r="7" fill="var(--crimson)"/><circle cx="185" cy="118" r="7" fill="var(--teal)"/>
          <circle cx="150" cy="48" r="7" fill="var(--crimson)"/>
          <circle cx="150" cy="85" r="9" fill="#fff" stroke="var(--crimson)" stroke-width="3"/>
          <text x="150" y="160" fill="#93a1b3" font-size="11" font-family="IBM Plex Mono" text-anchor="middle">4 z 5 sousedů → napadená</text>
        </svg>
      </div>
      <div class="reveal" style="--i:3">
        <h3>Ale pozor na „přesnost“</h3>
        <p style="margin-bottom:.4rem">Při výchozím prahu: přesnost <span class="tt">__KNN_ACC__</span> zní hezky…</p>
        <div class="cm">
          <div class="c ok"><b>__TN__</b><span>správně zdravá</span></div>
          <div class="c"><b>__FP__</b><span>falešný poplach</span></div>
          <div class="c bad"><b>__FN__</b><span>PŘEHLÉDNUTÝ nemocný</span></div>
          <div class="c ok"><b>__TP__</b><span>správně napadená</span></div>
        </div>
        <p style="margin-top:.9rem" class="crim"><strong>…ale přehlédne skoro polovinu nemocných.</strong> Přesnost klame.</p>
      </div>
    </div>
  </section>

  <!-- 10 METRIKY / ROC -->
  <section class="slide">
    <div class="wrap grid2">
      <div>
        <div class="kicker reveal" style="--i:0">Medicína ≠ přesnost</div>
        <h2 class="reveal" style="--i:1">Senzitivita vs.<br>specificita</h2>
        <ul class="list">
          <li class="reveal" style="--i:2"><span><strong>Senzitivita</strong> — kolik nemocných zachytíme. Nízká = přehlížíme pacienty. <span class="crim">Může stát život.</span></span></li>
          <li class="reveal" style="--i:3"><span><strong>Specificita</strong> — kolik zdravých správně propustíme. Nízká = zbytečné poplachy.</span></li>
          <li class="reveal" style="--i:4"><span>Hodnotíme proto <strong>senzitivitu při specificitě ≥ 95 %</strong> — bod na ROC křivce.</span></li>
        </ul>
      </div>
      <div class="figframe reveal" style="--i:3"><img src="__ROC__" alt="ROC křivka"></div>
    </div>
  </section>

  <!-- 11 MLP -->
  <section class="slide">
    <div class="wrap grid2">
      <div>
        <div class="kicker reveal" style="--i:0">Naučená hlava · MLP</div>
        <h2 class="reveal" style="--i:1">Malá síť, která se<br><span class="teal">naučí, na čem záleží</span></h2>
        <p class="reveal" style="--i:2">Na rozdíl od k-NN se MLP (vícevrstvý perceptron) učí vážit, které z 2048 featur rozhodují — proto baseline výrazně překoná.</p>
        <svg class="svgbox reveal" style="--i:3;max-width:380px;margin-top:1rem" viewBox="0 0 360 160">
          <g stroke="rgba(53,214,192,.25)" stroke-width="1">
            <line x1="60" y1="30" x2="170" y2="40"/><line x1="60" y1="30" x2="170" y2="80"/><line x1="60" y1="30" x2="170" y2="120"/>
            <line x1="60" y1="80" x2="170" y2="40"/><line x1="60" y1="80" x2="170" y2="80"/><line x1="60" y1="80" x2="170" y2="120"/>
            <line x1="60" y1="130" x2="170" y2="40"/><line x1="60" y1="130" x2="170" y2="80"/><line x1="60" y1="130" x2="170" y2="120"/>
            <line x1="170" y1="40" x2="290" y2="80"/><line x1="170" y1="80" x2="290" y2="80"/><line x1="170" y1="120" x2="290" y2="80"/>
          </g>
          <g fill="#9aa7b6"><circle cx="60" cy="30" r="8"/><circle cx="60" cy="80" r="8"/><circle cx="60" cy="130" r="8"/></g>
          <g fill="var(--teal)"><circle cx="170" cy="40" r="9"/><circle cx="170" cy="80" r="9"/><circle cx="170" cy="120" r="9"/></g>
          <circle cx="290" cy="80" r="11" fill="#fff" stroke="var(--crimson)" stroke-width="3"/>
          <text x="60" y="152" fill="#93a1b3" font-size="9" font-family="IBM Plex Mono" text-anchor="middle">2048 featur</text>
          <text x="170" y="152" fill="#93a1b3" font-size="9" font-family="IBM Plex Mono" text-anchor="middle">skrytá vrstva</text>
          <text x="290" y="152" fill="#93a1b3" font-size="9" font-family="IBM Plex Mono" text-anchor="middle">P(napadená)</text>
        </svg>
      </div>
      <div class="reveal" style="--i:3">
        <h3>Senzitivita při specificitě 95 %</h3>
        <div class="bars">
          <div class="bar">
            <div class="top"><span>k-NN baseline</span><span class="tt">__KNN_SENS__</span></div>
            <div class="track"><div class="fill knn" style="--v:__KNN_SENS_PCT__%"></div></div>
          </div>
          <div class="bar">
            <div class="top"><span class="teal">MLP (naučená hlava)</span><span class="tt teal">__MLP_SENS__</span></div>
            <div class="track"><div class="fill mlp" style="--v:__MLP_SENS_PCT__%"></div></div>
          </div>
        </div>
        <p style="margin-top:1rem">Z <span class="tt">__KNN_SENS__</span> na <span class="teal tt">__MLP_SENS__</span> — tatáž data, chytřejší rozhodnutí. <strong>Baseline překonán.</strong></p>
      </div>
    </div>
  </section>

  <!-- 12 ZÁVĚR -->
  <section class="slide">
    <div class="wrap">
      <div class="kicker reveal" style="--i:0">Shrnutí</div>
      <h2 class="reveal" style="--i:1">Mezi notebookem a dopadem<br>na zdravotnictví není daleko</h2>
      <div class="grid2" style="margin-top:.6rem">
        <ul class="list">
          <li class="reveal" style="--i:2"><span><strong>Přenosové učení</strong> — zmrazený ResNet jako extraktor featur.</span></li>
          <li class="reveal" style="--i:3"><span>Featury <strong>samy oddělují</strong> třídy (PCA).</span></li>
          <li class="reveal" style="--i:4"><span>V medicíně <strong>přesnost klame</strong> — počítá se senzitivita a specificita.</span></li>
          <li class="reveal" style="--i:5"><span>Naučená hlava (MLP) <span class="teal">překonává</span> jednoduchý baseline.</span></li>
        </ul>
        <div class="reveal" style="--i:4">
          <div class="card">
            <h3>Děkujeme za pozornost</h3>
            <p style="font-size:1rem">Michal Průšek &amp; Michal Bělohlávek<br><span style="color:#93a1b3">ÚTIA AV ČR · CIIRC · FJFI ČVUT</span></p>
            <p class="foot">Materiály a notebook:<br><span class="teal">github.com/michalprusek/malaria</span></p>
          </div>
        </div>
      </div>
    </div>
  </section>

</div>

<div class="dots" id="dots"></div>
<div class="hint">← →  /  mezerník</div>
<div class="hud"><div class="pb" id="pb"></div></div>

<script>
const slides=[...document.querySelectorAll('.slide')];
const dots=document.getElementById('dots');
const pb=document.getElementById('pb'), cur=document.getElementById('cur'), tot=document.getElementById('tot');
let i=0; tot.textContent=slides.length;
slides.forEach((_,k)=>{const b=document.createElement('button');b.onclick=()=>go(k);dots.appendChild(b);});
const dotEls=[...dots.children];
function go(n){
  i=Math.max(0,Math.min(slides.length-1,n));
  slides.forEach((s,k)=>s.classList.toggle('active',k===i));
  dotEls.forEach((d,k)=>d.classList.toggle('on',k===i));
  pb.style.width=((i+1)/slides.length*100)+'%';
  cur.textContent=i+1;
}
function next(){go(i+1)} function prev(){go(i-1)}
addEventListener('keydown',e=>{
  if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){e.preventDefault();next();}
  else if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){e.preventDefault();prev();}
  else if(e.key==='Home')go(0); else if(e.key==='End')go(slides.length-1);
});
let sx=0;
addEventListener('touchstart',e=>sx=e.touches[0].clientX,{passive:true});
addEventListener('touchend',e=>{const d=e.changedTouches[0].clientX-sx; if(Math.abs(d)>50)(d<0?next():prev());},{passive:true});
go(0);
</script>
</body>
</html>
"""

for k, v in vals.items():
    HTML = HTML.replace("__" + k + "__", v)
HTML = HTML.replace("__PCA__", PCA_B64).replace("__ROC__", ROC_B64)
HTML = HTML.replace("__CELL_INF__", CELL_INF).replace("__CELL_OK__", CELL_OK)

with open("prezentace.html", "w", encoding="utf-8") as f:
    f.write(HTML)
print("zapsáno: prezentace.html  (%.0f kB)" % (len(HTML.encode()) / 1024))
