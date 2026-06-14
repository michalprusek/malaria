#!/usr/bin/env python3
"""Generátor výukových notebooků pro miniprojekt 'Detekce malárie'.

Spuštění:  python3 tools/build_notebooks.py
Vytvoří v kořeni repozitáře:
  - malaria_classifier_STUDENT.ipynb     (hlavní notebook pro studenty)
  - prepare_features_INSTRUCTOR.ipynb    (příprava featur, školitel spustí jednou)

Obsah buněk je psaný jako obyčejné řetězce (žádné f-stringy), aby se nepletly
procenta, složené závorky a česká diakritika.
"""
import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def build(cells, path, kernel="python3"):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": kernel},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("zapsáno:", path)


# =====================================================================
#  STUDENTSKÝ NOTEBOOK  (B)
# =====================================================================
student = []

student.append(md(r"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/michalprusek/malaria/blob/main/malaria_classifier_STUDENT.ipynb)

# 🦟 Detekuje malárii lépe člověk, nebo notebook?

### Miniprojekt — Veletrh vědy · Katedra matematiky FJFI ČVUT

Malárie každý rok zabije přibližně **600 000 lidí** — hlavně tam, kde chybí vyškolení
patologové, kteří by pod mikroskopem poznali parazita *Plasmodium* v krevním nátěru. Celá
diagnóza stojí na jediné otázce, opakované u tisíců buněk: **je tahle červená krvinka napadená,
nebo zdravá?** Dnes ji zkusíme nechat řešit počítač — na **27 558 snímcích** buněk z datasetu
NIH (polovina zdravých, polovina napadených).

> **Připomenutí z přednášky:** zmrazený **ResNet** promění každý snímek buňky na **2048 čísel**
> (*featury*). My netrénujeme ResNet — učíme jen malou **klasifikační hlavu**, která z těch 2048
> čísel rozhodne *napadená / zdravá*. Proto vše běží i **bez grafické karty**, během pár sekund.

### Co dnes uděláte

1. Podíváte se na skutečné buňky a na to, jak z nich vzniká **2048 čísel**.
2. Postavíte jednoduchou **baseline (k-NN)** a překonáte ji **vlastní natrénovanou sítí**.
3. Zjistíte, proč v medicíně **nestačí „95 % přesnost"** — co je *senzitivita*, *specificita*,
   *matice záměn* a *ROC křivka*.
4. **Soutěž!** Vyhrává model, který zachytí nemocné, aniž by zbytečně strašil zdravé.

Vzhůru do toho. 🚀
"""))

student.append(md(r"""
## 0 · Jak tenhle notebook ovládat

- Notebook se skládá z **buněk**. Buňku spustíte klávesou **`Shift + Enter`**.
- Buňky spouštějte **odshora dolů** — pozdější počítají s výsledky dřívějších.
- Nepotřebujete grafickou kartu. (Pokud chcete, v menu *Runtime → Change runtime type* můžete
  zapnout GPU, ale není to nutné.)
- Místa označená **`# 🔬 EXPERIMENT`** jsou vaše hřiště — tam měňte hodnoty a zkoušejte, jak
  se model zlepší. Zbytek nechte běžet, jak je.

➡️ **Nejdřív vyplňte jméno svého týmu v další buňce.**
"""))

student.append(code(r"""
# ====== NASTAVENÍ ======
TEAM_NAME = "TYM_A"          # 🔬 napiš sem název svého týmu (např. "Plasmodium_Hunters")

# Odkud se stáhnou předpočítaná data (GitHub Release).
# Pokud nejsou dostupná (privátní repo), notebook si data obstará sám z veřejného NIH zdroje.
DATA_BASE_URL = "https://github.com/michalprusek/malaria/releases/download/data-v1"
# =======================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Pro reprodukovatelnost (stejný start → stejné výsledky)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch:", torch.__version__, "| zařízení:", device)
print("Tým:", TEAM_NAME)
"""))

student.append(md(r"""
## 1 · Získání dat

Notebook si data obstará automaticky a sám pozná, kterou ze dvou cest jít:

- **Rychlá cesta** — pokud školitel zveřejnil předpočítané featury, stáhnou se hotové
  soubory `train.npz`, `val.npz`, `test_features.npz` (pár sekund, GPU netřeba).
- **Soběstačná cesta** — když předpočítaná data nejsou veřejně dostupná, notebook si stáhne
  původní obrázky z veřejného serveru **NIH** a featury si **sám jednou spočítá** ResNetem.
  Trvá to ~5 minut a potřebuje zapnuté GPU: *Runtime → Change runtime type → GPU*.
  (Bonus: uvidíte ResNet skutečně běžet na obrázcích!)

Výsledek je v obou případech stejný. Spusťte obě následující buňky.
"""))

student.append(code(r"""
import os, zipfile

def _ok(f):
    return os.path.exists(f) and os.path.getsize(f) > 1000

REZIM = "predpocitana"
for fname in ["train.npz", "val.npz", "test_features.npz", "samples.zip"]:
    if not _ok(fname):
        os.system(f"wget -q {DATA_BASE_URL}/{fname} -O {fname}")
    if not _ok(fname):                       # rychlá cesta nedostupná (např. privátní repo)
        if os.path.exists(fname):
            os.remove(fname)
        REZIM = "sobestacna"
        break

if REZIM == "predpocitana" and _ok("samples.zip") and not os.path.isdir("samples"):
    with zipfile.ZipFile("samples.zip") as z:
        z.extractall(".")     # zip už obsahuje prefix samples/
print("Režim dat:", REZIM)
"""))

student.append(code(r"""
# Tahle buňka se spustí, jen když nejsou předpočítaná data. Pak si je notebook obstará sám.
# (Nemusíte jí rozumět — jen jednou připraví stejné soubory jako rychlá cesta.)
if REZIM == "sobestacna":
    import glob, shutil, urllib.request
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights
    from sklearn.model_selection import train_test_split

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("⚠️  GPU NENÍ zapnuté → Runtime → Change runtime type → GPU. Na CPU to potrvá ~20 min.")

    URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
    if not os.path.exists("cell_images.zip"):
        print("stahuji obrázky (~350 MB)…")
        urllib.request.urlretrieve(URL, "cell_images.zip")
    if not os.path.isdir("cell_images"):
        with zipfile.ZipFile("cell_images.zip") as z:
            z.extractall(".")

    items = []
    for p in glob.glob("**/*.png", recursive=True):
        par = os.path.basename(os.path.dirname(p)).lower()
        if "parasitized" in par:
            items.append((p, 1))
        elif "uninfected" in par:
            items.append((p, 0))
    items.sort(key=lambda t: (os.path.basename(t[0]), t[1]))   # deterministické pořadí
    print("obrázků:", len(items))

    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    class _DS(Dataset):
        def __init__(self, it): self.it = it
        def __len__(self): return len(self.it)
        def __getitem__(self, i):
            p, yy = self.it[i]
            return tfm(Image.open(p).convert("RGB")), yy

    enc = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    enc.fc = nn.Identity(); enc.eval().to(dev)
    feats, labs = [], []
    with torch.no_grad():
        for k, (xb, yb) in enumerate(DataLoader(_DS(items), batch_size=128, num_workers=2)):
            feats.append(enc(xb.to(dev)).cpu().numpy().astype(np.float16))
            labs.append(np.asarray(yb))
            if k % 20 == 0:
                print(f"  zpracováno {k*128}/{len(items)}")
    Xall = np.concatenate(feats).astype(np.float16)
    yall = np.concatenate(labs).astype(np.int64)

    Xtmp, Xte, ytmp, yte = train_test_split(Xall, yall, test_size=0.15,
                                            stratify=yall, random_state=42)
    Xtr_, Xva_, ytr_, yva_ = train_test_split(Xtmp, ytmp, test_size=0.1765,
                                              stratify=ytmp, random_state=42)
    np.savez_compressed("train.npz", X=Xtr_, y=ytr_)
    np.savez_compressed("val.npz", X=Xva_, y=yva_)
    np.savez_compressed("test_features.npz", X=Xte)   # bez labelů (soutěž)

    os.makedirs("samples/Parasitized", exist_ok=True)
    os.makedirs("samples/Uninfected", exist_ok=True)
    for p in [p for p, l in items if l == 1][:15]:
        shutil.copy(p, "samples/Parasitized/")
    for p in [p for p, l in items if l == 0][:15]:
        shutil.copy(p, "samples/Uninfected/")
    print("hotovo — data připravena.")
"""))

student.append(md(r"""
## 2 · Podívejme se na buňky

Než začneme cokoli trénovat, podívejme se vlastníma očima, co má model rozlišit.
U **napadených** buněk bývá vidět malá fialová tečka — to je parazit *Plasmodium* obarvený
při přípravě nátěru. **Zdravé** buňky jsou rovnoměrně růžové.
"""))

student.append(code(r"""
import glob
from PIL import Image

paras = sorted(glob.glob("samples/**/*arasitized*/*.png", recursive=True)) \
        or sorted(glob.glob("samples/**/Parasitized*/*.png", recursive=True))
healthy = sorted(glob.glob("samples/**/*ninfected*/*.png", recursive=True)) \
        or sorted(glob.glob("samples/**/Uninfected*/*.png", recursive=True))

fig, axes = plt.subplots(2, 6, figsize=(13, 4.5))
for j in range(6):
    if j < len(paras):
        axes[0, j].imshow(Image.open(paras[j]));
    axes[0, j].set_title("napadená", color="crimson", fontsize=10); axes[0, j].axis("off")
    if j < len(healthy):
        axes[1, j].imshow(Image.open(healthy[j]))
    axes[1, j].set_title("zdravá", color="seagreen", fontsize=10); axes[1, j].axis("off")
plt.suptitle("Vidíte u napadených buněk tu fialovou tečku (parazita)?", fontsize=12)
plt.tight_layout(); plt.show()
"""))

student.append(md(r"""
## 3 · Od obrázku k 2048 číslům

Počítač nevidí „fialovou tečku" — vidí čísla. **Zmrazený ResNet** (*encoder*) každý snímek
přeloží na **2048 featur** (příznaků): stručné shrnutí toho, co na obrázku je. S těmihle čísly
pak pracují všechny naše modely — obrázek už nevidí. Načteme si je a podíváme se, jak vypadají.
"""))

student.append(code(r"""
train = np.load("train.npz")
val   = np.load("val.npz")

X_train, y_train = train["X"].astype(np.float32), train["y"].astype(np.int64)
X_val,   y_val   = val["X"].astype(np.float32),   val["y"].astype(np.int64)

print("trénink:", X_train.shape, " (počet buněk, počet featur)")
print("ověření:", X_val.shape)
print("\nKaždá buňka =", X_train.shape[1], "čísel.")
print("Vyváženost tříd v tréninku:",
      "zdravých", int((y_train == 0).sum()), "| napadených", int((y_train == 1).sum()))
print("\nPrvních 8 featur první buňky:", np.round(X_train[0, :8], 2))
"""))

student.append(md(r"""
### Jak vypadá „obrázek → 2048 čísel"

Vlevo skutečná buňka, vpravo jejích 2048 featur poskládaných do mřížky. Model rozhoduje **jen
z těch čísel vpravo** — samotný obrázek nevidí. (Rozdíl mezi napadenou a zdravou je v těch
číslech jemný a roztroušený — proto na to potřebujeme model, ne jen oko.)
"""))

student.append(code(r"""
v_inf  = X_train[np.where(y_train == 1)[0][0]]      # featury jedné napadené buňky
v_heal = X_train[np.where(y_train == 0)[0][0]]      # featury jedné zdravé buňky

fig, ax = plt.subplots(2, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 2.4]})
for r, (imgs, vec, name, col) in enumerate([
        (paras, v_inf, "napadená", "crimson"),
        (healthy, v_heal, "zdravá", "seagreen")]):
    ax[r, 0].imshow(Image.open(imgs[0])); ax[r, 0].axis("off")
    ax[r, 0].set_title(name, color=col, fontsize=12)
    ax[r, 1].imshow(vec.reshape(32, 64), cmap="magma", aspect="auto")
    ax[r, 1].set_title("jejích 2048 featur (mřížka 32×64)", fontsize=10)
    ax[r, 1].set_xticks([]); ax[r, 1].set_yticks([])
fig.suptitle("Každou buňku ResNet promění na 2048 čísel (vpravo).\n"
             "Klasifikátor pak vidí jen ta čísla — obrázek už ne.", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.9]); plt.show()
"""))

student.append(md(r"""
### Magický okamžik: featury už třídy rozdělují

2048 rozměrů si nedokážeme představit. Promítneme je proto pomocí metody **PCA** do roviny
(2 rozměry) a obarvíme body podle skutečné třídy. Pokud encoder odvedl dobrou práci, uvidíme
**dva oddělené shluky** — ještě **předtím**, než jsme cokoli natrénovali. To je důkaz, že
předtrénovaná síť už „těžkou práci vidění" udělala za nás a nám zbývá jen najít dělící čáru.
"""))

student.append(code(r"""
from sklearn.decomposition import PCA

idx = np.random.choice(len(X_train), size=3000, replace=False)  # podvzorek kvůli rychlosti
pts = PCA(n_components=2).fit_transform(X_train[idx])

plt.figure(figsize=(7, 6))
for label, name, col in [(0, "zdravá", "seagreen"), (1, "napadená", "crimson")]:
    m = y_train[idx] == label
    plt.scatter(pts[m, 0], pts[m, 1], s=6, alpha=0.4, label=name, color=col)
plt.legend(); plt.title("PCA featur — buňky se samy rozdělily do dvou shluků")
plt.xlabel("hlavní komponenta 1"); plt.ylabel("hlavní komponenta 2"); plt.show()
"""))

student.append(md(r"""
## 4 · Standardizace featur

Drobný, ale užitečný krok. Různé featury mají různě velké hodnoty. Když je všechny převedeme
na společné měřítko (průměr 0, rozptyl 1), pomůže to **oběma** modelům, které za chvíli
postavíme: k-NN (počítá vzdálenosti — ty dávají smysl jen při srovnatelném měřítku) i
neuronové síti (učí se rychleji a stabilněji).

> ⚠️ Důležité pravidlo: měřítko spočítáme **jen z trénovacích dat** (`fit`) a stejné měřítko
> pak použijeme na ověřovací i testovací data (`transform`). Jinak bychom „nakoukli" do dat,
> která máme předstírat, že neznáme.
"""))

student.append(code(r"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit JEN na tréninku
X_val_s   = scaler.transform(X_val)

# převedeme na tensory pro PyTorch
Xtr = torch.tensor(X_train_s, dtype=torch.float32, device=device)
ytr = torch.tensor(y_train,   dtype=torch.float32, device=device)
Xva = torch.tensor(X_val_s,   dtype=torch.float32, device=device)
print("připraveno:", Xtr.shape, "na zařízení", device)
"""))

student.append(md(r"""
## 5 · Nejjednodušší učený model: jedna lineární vrstva

PCA kreslila featury „naslepo“ — bez znalosti správných odpovědí. Teď zkusíme opak: necháme
**jedinou lineární vrstvu** naučit se z 2048 featur **dvě čísla** (skóre pro „zdravá“ a pro
„napadená“) přímo z labelů. Je to nejjednodušší *učený* model — featury jen zváží a sečte —
a hned uvidíme, že třídy slušně oddělí. Dostaneme i názorný 2D obrázek s **dělící čarou**
(tam, kde se obě skóre rovnají).
"""))

student.append(code(r"""
ytr_long = torch.tensor(y_train, dtype=torch.long, device=device)

lin = nn.Linear(Xtr.shape[1], 2).to(device)          # 2048 featur → 2 skóre (zdravá, napadená)
opt_lin = torch.optim.Adam(lin.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
for epoch in range(15):
    lin.train()
    perm = torch.randperm(Xtr.shape[0], device=device)
    for i in range(0, Xtr.shape[0], 256):
        idx = perm[i:i + 256]
        opt_lin.zero_grad()
        ce(lin(Xtr[idx]), ytr_long[idx]).backward()
        opt_lin.step()

lin.eval()
with torch.no_grad():
    z_val = lin(Xva).cpu().numpy()                    # 2D skóre pro ověřovací buňky
lin_acc = (z_val.argmax(1) == y_val).mean()
print(f"Lineární vrstva — přesnost na ověření: {lin_acc:.4f}")
"""))

student.append(code(r"""
plt.figure(figsize=(7, 6))
for label, name, col in [(0, "zdravá", "seagreen"), (1, "napadená", "crimson")]:
    m = y_val == label
    plt.scatter(z_val[m, 0], z_val[m, 1], s=8, alpha=0.4, color=col, label=name)
lim = [float(z_val.min()), float(z_val.max())]
plt.plot(lim, lim, ls="--", color="gray", label="dělící čára (skóre stejné)")
plt.xlabel("skóre: zdravá"); plt.ylabel("skóre: napadená")
plt.legend(); plt.title(f"Jedna lineární vrstva — 2D projekce (přesnost {lin_acc:.3f})")
plt.show()
"""))

student.append(md(r"""
## 6 · Baseline: k nejbližších sousedů (k-NN)

Naše **baseline** bude co nejjednodušší model — laťka, kterou se pak budeme snažit překonat.
Použijeme **k-NN** (*k nearest neighbors*, k nejbližších sousedů):

> Chceš zařadit novou buňku? Najdi mezi trénovacími buňkami **k nejpodobnějších** (nejbližších
> v prostoru featur) a nech je **hlasovat**. Bližší sousedé váží víc. Podíl hlasů pro
> „napadená" bereme jako **pravděpodobnost**.

Zvláštnost k-NN: **vůbec se netrénuje** — jen si zapamatuje trénovací data a při dotazu počítá
vzdálenosti. Právě proto je dobrý jako výchozí bod. Uvidíme ale i jeho slabinu: ve **2048
rozměrech** přestávají být vzdálenosti spolehlivé (tzv. *prokletí dimenzionality*), takže k-NN
tady nebude žádný přeborník — a to je pro nás dobře, bude co překonávat.
"""))

student.append(code(r"""
from sklearn.neighbors import KNeighborsClassifier

K = 31    # 🔬 EXPERIMENT (lite): kolik sousedů hlasuje (zkuste 5, 15, 51…)

knn = KNeighborsClassifier(n_neighbors=K, weights="distance", n_jobs=-1)
knn.fit(X_train_s, y_train)                       # "trénink" = jen si zapamatuje data
knn_val_prob = knn.predict_proba(X_val_s)[:, 1]   # pravděpodobnost "napadená"

knn_acc = ((knn_val_prob > 0.5).astype(int) == y_val).mean()
print(f"k-NN baseline — přesnost na ověření: {knn_acc:.4f}")
print("To je laťka. V dalších sekcích ji zkusíme překonat vlastní neuronovou sítí.")
"""))

student.append(md(r"""
### A jak zvolit práh?

Z pravděpodobnosti uděláme rozhodnutí až **prahem**. Práh 0,5 není svatý — volíme ho podle
cíle. My chceme zachytit **aspoň 99 % nemocných**, takže práh *snižujeme*, dokud senzitivita
nevyleze na 99 % (na **ověřovací** sadě, nikdy na testu!). Pozor — **99 % záchytu k-NN splní**
(stačí dát práh skoro na nulu, pak označí za napadené skoro vše). Jenže tím označí i skoro
všechny zdravé, takže jeho **specificita v tom bodě spadne skoro k nule** — soutěžní skóre je
proto ≈ 0. Není to „nedosáhne", ale „na laťce skóruje nulu".
"""))

student.append(code(r"""
from sklearn.metrics import roc_curve

fpr, tpr, thr = roc_curve(y_val, knn_val_prob)      # tpr = senzitivita, fpr = 1 - specificita
ok = tpr >= 0.99
j = int(np.argmin(np.where(ok, fpr, 2.0)))          # nejmenší falešné poplachy při senz. >= 99 %
print(f"práh pro senzitivitu ≥ 99 %: {thr[j]:.3f}")
print(f"  → senzitivita = {tpr[j]:.3f},  specificita = {1 - fpr[j]:.3f}")

print("\npro srovnání pár pevných prahů:")
print(" práh | senzitivita | specificita")
for t in [0.3, 0.5, 0.7, 0.9]:
    yp = (knn_val_prob >= t).astype(int)
    se = yp[y_val == 1].mean()                      # podíl zachycených nemocných
    sp = 1 - yp[y_val == 0].mean()                  # podíl správně propuštěných zdravých
    print(f" {t:.2f} |    {se:.3f}    |    {sp:.3f}")
print("\nNižší práh = víc záchytů (↑ senzitivita), ale i víc falešných poplachů (↓ specificita).")
"""))

student.append(md(r"""
## 7 · Vaše vlastní klasifikační hlava 🔬

Teď to nejzábavnější — **navrhnete vlastní malou neuronovou síť**, která má překonat baseline
(k-NN). Dostane na vstupu 2048 featur a na výstupu vydá jediné číslo (*logit*), které po
převedení funkcí *sigmoid* říká **pravděpodobnost, že je buňka napadená**.

Výchozí síť níže má jednu skrytou vrstvu. **Tady experimentujte:**
- `hidden` — kolik neuronů má skrytá vrstva (víc = větší kapacita, ale i riziko přeučení),
- můžete přidat **další vrstvy** (`nn.Linear`, `nn.ReLU`).
"""))

student.append(code(r"""
class Hlava(nn.Module):
    def __init__(self, in_dim=2048, hidden=128):   # 🔬 EXPERIMENT: hidden
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),       # 🔬 EXPERIMENT: zkus přidat další vrstvy nad tenhle řádek
        )

    def forward(self, x):
        return self.net(x).squeeze(1)   # vrací logit, tvar (batch,)


model = Hlava(in_dim=Xtr.shape[1], hidden=128).to(device)
print(model)
print("počet trénovaných vah:", sum(p.numel() for p in model.parameters()))
"""))

student.append(md(r"""
## 8 · Trénink 🔬

Trénink znamená: model hádá, porovnáme jeho odhad se správnou odpovědí, spočítáme **chybu**
(*loss*) a malými krůčky vahy upravíme, aby chyba klesala. Jeden průchod všemi daty = jedna
**epocha**.

Páky, se kterými si můžete hrát:
- `EPOCHS` — kolikrát projdeme data (víc se naučí, ale může se přeučit),
- `LR` (learning rate) — velikost krůčku při učení,
- `POS_WEIGHT` — **jak draho stojí přehlédnutý nemocný**. Hodnota > 1 nutí model brát napadené
  buňky vážněji (zvýší senzitivitu na úkor falešných poplachů). To je přímo páka na medicínský
  kompromis, o kterém bude řeč v sekci 9!
"""))

student.append(code(r"""
# ====== 🔬 EXPERIMENT: hyperparametry ======
EPOCHS       = 25
BATCH        = 256
LR           = 1e-3
POS_WEIGHT   = 1.0      # > 1 = přísnější na přehlédnuté nemocné
# ===========================================

loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=device))
opt = torch.optim.Adam(model.parameters(), lr=LR)

hist = {"train_loss": [], "val_acc": []}
n = Xtr.shape[0]
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(n, device=device)
    running = 0.0
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = loss_fn(model(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()
        running += loss.item() * len(idx)
    # vyhodnocení na ověřovací sadě
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(Xva)).cpu().numpy()
    acc = ((val_prob > 0.5).astype(int) == y_val).mean()
    hist["train_loss"].append(running / n)
    hist["val_acc"].append(acc)
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"epocha {epoch:2d}  chyba={running/n:.4f}  přesnost na ověření={acc:.4f}")
print("\nHotovo. Nejlepší přesnost na ověření:", round(max(hist['val_acc']), 4))
"""))

student.append(code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(hist["train_loss"]); ax[0].set_title("Trénovací chyba (loss)")
ax[0].set_xlabel("epocha"); ax[0].set_ylabel("chyba")
ax[1].plot(hist["val_acc"], color="darkorange"); ax[1].set_title("Přesnost na ověření")
ax[1].set_xlabel("epocha"); ax[1].set_ylabel("přesnost"); ax[1].set_ylim(0, 1)
plt.tight_layout(); plt.show()
"""))

student.append(md(r"""
## 9 · Proč v medicíně nestačí „95 % přesnost" 🩺

Představte si model, který o **každé** buňce řekne „zdravá". Když je v nátěru 95 % buněk
zdravých, má tenhle hloupý model rovnou **95 % přesnost** — a přitom **nezachytí ani jednoho
nemocného**. Přesnost (accuracy) nás obelhala.

V medicíně rozlišujeme dva druhy chyb a každý má jinou cenu:

|  | model řekl **zdravá** | model řekl **napadená** |
|---|---|---|
| **opravdu zdravá** | ✅ správně negativní (TN) | ⚠️ **falešný poplach** (FP) |
| **opravdu napadená** | ☠️ **přehlédnutý nemocný** (FN) | ✅ správně pozitivní (TP) |

- **Senzitivita** = TP / (TP + FN) = *jaký podíl skutečně nemocných model zachytí.*
  Nízká senzitivita = přehlížíme nemocné. **Tohle může stát život.**
- **Specificita** = TN / (TN + FP) = *jaký podíl zdravých model správně propustí.*
  Nízká specificita = strašíme zdravé a zbytečně je posíláme na další vyšetření.

Spočítejme si je z **matice záměn**.
"""))

student.append(code(r"""
from sklearn.metrics import confusion_matrix

def matice_a_metriky(y_true, y_prob, prah=0.5):
    y_pred = (y_prob >= prah).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return (tn, fp, fn, tp), sens, spec

(tn, fp, fn, tp), sens, spec = matice_a_metriky(y_val, val_prob, prah=0.5)

cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.imshow(cm, cmap="Blues")
for (r, c), v in np.ndenumerate(cm):
    ax.text(c, r, str(v), ha="center", va="center", fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(["řekl zdravá", "řekl napadená"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["je zdravá", "je napadená"])
ax.set_title("Matice záměn (práh 0.5)"); plt.tight_layout(); plt.show()

print(f"senzitivita = {sens:.3f}  (podíl zachycených nemocných)")
print(f"specificita = {spec:.3f}  (podíl správně propuštěných zdravých)")
print(f"přehlédnutých nemocných (FN): {fn}   |   falešných poplachů (FP): {fp}")
"""))

student.append(md(r"""
### Práh rozhoduje o kompromisu

Model vydává **pravděpodobnost** 0–1. My musíme zvolit **práh**, nad kterým buňku prohlásíme
za napadenou. Posuneme-li práh **dolů**, zachytíme víc nemocných (↑ senzitivita), ale přibude
falešných poplachů (↓ specificita). A naopak. Celý ten kompromis zachycuje **ROC křivka**.

Naše **soutěžní metrika** je jeden konkrétní bod na téhle křivce:

> **Specificita při senzitivitě ≥ 99 %** — tedy: *jak málo zdravých zbytečně poplašíme, když
> musíme zachytit aspoň 99 % nemocných?* U smrtelné nemoci je „nepřehlédnout nemocného“ ta
> nepodkročitelná podmínka; soutěžíme pak v tom, kdo přitom udrží nejvyšší specificitu.
> Čím vyšší, tím lepší model.
"""))

student.append(code(r"""
from sklearn.metrics import roc_curve

def specificita_pri_senzitivite(y_true, y_prob, min_sens=0.99):
    fpr, tpr, thr = roc_curve(y_true, y_prob)          # tpr = senzitivita
    ok = tpr >= min_sens
    best = int(np.argmin(np.where(ok, fpr, 2.0)))      # nejmenší fpr = max specificita
    return 1 - fpr[best], thr[best], tpr[best]

fpr, tpr, thr = roc_curve(y_val, val_prob)
spec99, prah99, sens99 = specificita_pri_senzitivite(y_val, val_prob, 0.99)

# k-NN baseline pro srovnání
fpr_k, tpr_k, _ = roc_curve(y_val, knn_val_prob)
spec99_knn, _, _ = specificita_pri_senzitivite(y_val, knn_val_prob, 0.99)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, lw=2, label=f"vaše síť (spec={spec99:.3f})")
plt.plot(fpr_k, tpr_k, lw=2, ls="--", color="gray", label="k-NN baseline")
plt.axhline(0.99, ls=":", color="crimson", lw=1, label="požadovaná senzitivita 99 %")
plt.plot([0, 1], [0, 1], ls=":", color="lightgray")

# bod, který zvolíme prahem: nejvyšší specificita při senzitivitě >= 99 %
x_op, y_op = 1 - spec99, sens99
plt.scatter([x_op], [y_op], color="crimson", s=90, zorder=5, label="náš zvolený bod")
plt.vlines(x_op, 0, y_op, color="crimson", ls="--", lw=1)   # dolů na osu x: 1 − specificita
plt.hlines(y_op, 0, x_op, color="crimson", ls="--", lw=1)   # vlevo na osu y: senzitivita
plt.annotate(f"senzitivita = {y_op:.2f}\n1 − spec = {x_op:.2f}  (spec = {spec99:.2f})",
             (x_op, y_op), textcoords="offset points", xytext=(14, -34),
             color="crimson", fontsize=9, arrowprops=dict(arrowstyle="->", color="crimson"))

plt.xlim(0, 1); plt.ylim(0, 1.03)
plt.xlabel("1 − specificita  (falešné poplachy)"); plt.ylabel("senzitivita (záchyt nemocných)")
plt.title("ROC křivka — náš zvolený bod"); plt.legend(loc="lower right"); plt.show()

print(f"➡️  SOUTĚŽNÍ SKÓRE (na ověření): specificita @ senzitivita≥99 % = {spec99:.3f}")
print(f"    (skutečně zachyceno {sens99 * 100:.1f} % nemocných)")
"""))

student.append(md(r"""
### Vyzkoušejte si posuvník prahu

Posouvejte práh a sledujte, jak se mění matice záměn i obě metriky. Najdete práh, který drží
specificitu kolem 95 %?
"""))

student.append(code(r"""
try:
    from ipywidgets import interact, FloatSlider

    def ukaz(prah=0.5):
        (tn, fp, fn, tp), s, sp = matice_a_metriky(y_val, val_prob, prah)
        print(f"práh={prah:.2f} | senzitivita={s:.3f} | specificita={sp:.3f} "
              f"| přehlédnutí(FN)={fn} | falešné poplachy(FP)={fp}")

    interact(ukaz, prah=FloatSlider(min=0.01, max=0.99, step=0.01, value=0.5))
except Exception as e:
    print("(posuvník nedostupný, ukážu tabulku)", e)
    for prah in [0.1, 0.3, 0.5, 0.7, 0.9]:
        (tn, fp, fn, tp), s, sp = matice_a_metriky(y_val, val_prob, prah)
        print(f"práh={prah:.2f} | senzitivita={s:.3f} | specificita={sp:.3f} "
              f"| FN={fn} | FP={fp}")
"""))

student.append(md(r"""
## 10 · Ladění pro soutěž 🔬🏆

Teď zpátky nahoru a vylepšujte! Cílem je co **nejvyšší specificita při senzitivitě ≥ 99 %**
na skrytém testu (a samozřejmě překonat baseline). Páky, které máte k dispozici:

| kde | páka | co zkusit |
|---|---|---|
| sekce 6 | `K` u k-NN | jiný počet sousedů — posune baseline (zkuste, jestli k-NN vůbec dotáhnete) |
| sekce 7 | architektura hlavy | větší `hidden`, další vrstvy |
| sekce 8 | `EPOCHS` | víc epoch — ale pozor na přeučení (sledujte ověření) |
| sekce 8 | `LR` | např. 3e-4, 1e-3, 3e-3 |
| sekce 8 | `POS_WEIGHT` | > 1 zvýší senzitivitu (přísnější na nemocné) |
| sekce 4 | standardizace | už zapnutá — zkuste, jaký je rozdíl bez ní |

**Postup:** nejrychleji laďte dole v **Hřišti (sekce 13)** — upravte jednu páku, spusťte,
sledujte `spec@99 %`. Vždycky měňte **jen jednu věc**, ať víte, co pomohlo.

> 💡 Tip: musíte zachytit 99 % nemocných. Když si model není jistý, musí kvůli tomu posunout
> práh hodně nízko a označí spoustu zdravých → specificita spadne (klidně až na nulu). Hledáte
> model, který dává **vysokou pravděpodobnost právě napadeným** buňkám a nízkou těm zdravým.
"""))

student.append(md(r"""
## 11 · Odevzdání do soutěže 📤

Až budete s modelem spokojení, vygenerujte odevzdání. Spočítáme pravděpodobnosti na **skrytém
testu** (u kterého neznáte správné odpovědi) a uložíme je do souboru `predikce_<TÝM>.csv`.
Ten odevzdejte školiteli — on spočítá vaše skóre na skrytých odpovědích a vyhlásí vítěze.

> Odevzdávají se **pravděpodobnosti**, ne hotová rozhodnutí — školitel sám najde práh pro
> senzitivitu 99 %. Vy se tedy nemusíte starat o volbu prahu, jen ať jsou vaše pravděpodobnosti
> co nejlépe seřazené (jisté tam, kde mají být).
"""))

student.append(code(r"""
test = np.load("test_features.npz")
X_test_s = scaler.transform(test["X"].astype(np.float32))
Xte = torch.tensor(X_test_s, dtype=torch.float32, device=device)

model.eval()
with torch.no_grad():
    test_prob = torch.sigmoid(model(Xte)).cpu().numpy()

fname = f"predikce_{TEAM_NAME}.csv"
np.savetxt(fname, test_prob, fmt="%.6f", header="prob", comments="")
print(f"uloženo: {fname}  ({len(test_prob)} buněk)")
print("ukázka prvních 5 pravděpodobností:", np.round(test_prob[:5], 4))

# v Google Colabu rovnou nabídne stažení souboru:
try:
    from google.colab import files
    files.download(fname)
except Exception:
    print("(mimo Colab — soubor najdete vedle notebooku)")
"""))

student.append(md(r"""
## ❓ Otázky k zamyšlení

1. Proč by se model nasazený v nemocnici mohl chovat hůř než tady na datasetu? (Jiný mikroskop,
   jiné barvení, jiná populace pacientů…)
2. Měla by konečné rozhodnutí dělat AI sama, nebo jen *upozorňovat* lékaře? Kde nastavit práh?
3. Co by se stalo se senzitivitou, kdyby byla v reálu napadená jen **1 %** buněk?
"""))

student.append(md(r"""
## 12 · Chcete zkusit jiný model než MLP?

MLP není jediná možnost — nad těmihle 2048 featurami funguje spousta klasifikátorů. Klidně se
zeptejte **ChatGPT/Claude** na *logistickou regresi, SVM, random forest* nebo *gradient
boosting*. Dvě pravidla, ať to dává smysl pro soutěž:

1. Model musí vydat **pravděpodobnost** (nebo skóre) — soutěžní metrika počítá s pořadím, ne s ano/ne.
2. Pro lineární modely / SVM / k-NN nechte zapnutou **standardizaci** (`X_train_s`); stromy a boosting ji nepotřebují.

Příklad — *gradient boosting* ze `sklearn` projde úplně stejným měřením jako vaše síť:
"""))

student.append(code(r"""
from sklearn.ensemble import HistGradientBoostingClassifier

gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1).fit(X_train_s, y_train)
gb_prob = gb.predict_proba(X_val_s)[:, 1]
gb_spec, _, _ = specificita_pri_senzitivite(y_val, gb_prob, 0.99)
print(f"gradient boosting — spec@99 % = {gb_spec:.3f}")
print("(porovnejte se svou MLP a s k-NN baseline)")
"""))

student.append(md(r"""
## 13 · 🔬 Hřiště — rychlé experimenty

Tady laďte naplno a bez scrollování. Spusťte **jednou** buňku s pomocnou funkcí
`experiment(...)` (hned pod tímhle textem) — tu už pak nemusíte měnit, můžete ji sbalit.
Potom jen **upravujte úplně poslední buňku** a mačkejte **Ctrl+Enter**:

- **Hyperparametry** nahoře jsou obyčejná čísla: `epochs`, `lr`, `pos_weight`.
- **Architektura** pod nimi je seznam vrstev (`nn.Sequential`). Přidávejte a ubírejte řádky:
  `nn.Linear(z, na)` mění rozměr, `nn.ReLU()` přidá nelinearitu. Vstup musí být **2048**,
  poslední vrstva musí být **`nn.Linear(..., 1)`**.

Po každém spuštění uvidíte své `spec@99 %`, porovnání s baseline a dosavadní nejlepší. Až
budete spokojení, dejte `submit=True` a uloží se odevzdání.
"""))

student.append(code(r"""
# Pomocná funkce pro Hřiště — spusťte JEDNOU, pak ji můžete sbalit a nevšímat si jí.
# Bere hotový model (nn.Sequential) + hyperparametry, natrénuje a změří spec@99 %.
BEST = {"spec": 0.0}

def experiment(model, epochs=25, lr=1e-3, pos_weight=1.0, plot=True, submit=False):
    from sklearn.metrics import roc_curve
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]).squeeze(-1), ytr[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(Xva).squeeze(-1)).cpu().numpy()
    spec, prah, sens = specificita_pri_senzitivite(y_val, prob, 0.99)
    acc = ((prob > 0.5).astype(int) == y_val).mean()
    flag = "🏆 NOVÉ NEJLEPŠÍ!" if spec > BEST["spec"] else f"(vaše nejlepší zatím: {BEST['spec']:.3f})"
    BEST["spec"] = max(BEST["spec"], spec)
    print(f"spec@99 % = {spec:.3f}   |   přesnost = {acc:.3f}   |   {flag}")
    if plot:
        fpr, tpr, _ = roc_curve(y_val, prob)
        x_op, y_op = 1 - spec, sens
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, lw=2, label=f"váš model (spec={spec:.3f})")
        plt.plot(fpr_k, tpr_k, ls="--", color="gray", label="k-NN baseline")
        plt.axhline(0.99, ls=":", color="crimson"); plt.plot([0, 1], [0, 1], ls=":", color="lightgray")
        plt.scatter([x_op], [y_op], color="crimson", s=70, zorder=5)
        plt.vlines(x_op, 0, y_op, color="crimson", ls="--", lw=1)
        plt.hlines(y_op, 0, x_op, color="crimson", ls="--", lw=1)
        plt.annotate(f"1 − spec = {x_op:.2f}\nsenz. = {y_op:.2f}", (x_op, y_op),
                     textcoords="offset points", xytext=(8, -34), color="crimson", fontsize=8)
        plt.xlim(0, 1); plt.ylim(0, 1.03)
        plt.xlabel("1 − specificita"); plt.ylabel("senzitivita")
        plt.legend(loc="lower right"); plt.title("ROC — váš zvolený bod"); plt.show()
    if submit:
        Xte_ = torch.tensor(scaler.transform(np.load("test_features.npz")["X"].astype(np.float32)),
                            dtype=torch.float32, device=device)
        with torch.no_grad():
            tp = torch.sigmoid(model(Xte_).squeeze(-1)).cpu().numpy()
        fn_ = f"predikce_{TEAM_NAME}.csv"
        np.savetxt(fn_, tp, fmt="%.6f", header="prob", comments="")
        print(f"📤 uloženo {fn_} ({len(tp)} buněk) — odevzdejte školiteli")
    return model
"""))

student.append(code(r"""
# ===================== 🔬 HŘIŠTĚ — upravte a spusťte (Ctrl+Enter) =====================
torch.manual_seed(SEED)                  # stejný start → porovnatelné výsledky

# HYPERPARAMETRY — měňte čísla
epochs     = 25
lr         = 1e-3
pos_weight = 1.0       # > 1 = přísnější na přehlédnuté nemocné

# ARCHITEKTURA — skládejte vrstvy. Vstup 2048, poslední vrstva musí být Linear(..., 1).
model = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
    # hlubší síť? např.:  nn.Linear(2048, 512), nn.ReLU(),
    #                     nn.Linear(512, 128),  nn.ReLU(), nn.Linear(128, 1)
)

experiment(model, epochs=epochs, lr=lr, pos_weight=pos_weight, submit=False)   # submit=True → uloží predikce_<TÝM>.csv
# =====================================================================================
"""))

build(student, "malaria_classifier_STUDENT.ipynb")


# =====================================================================
#  INSTRUKTORSKÝ NOTEBOOK  (A) — příprava featur, spustit JEDNOU na GPU
# =====================================================================
inst = []

inst.append(md(r"""
# 🔧 Příprava featur — INSTRUKTORSKÝ notebook

**Spustit JEDNOU před akcí, ideálně na Google Colab s GPU**
(*Runtime → Change runtime type → GPU*).

Notebook stáhne dataset malárie, prožene všech 27 558 obrázků zmraženým **ResNet50**,
uloží výsledné **featury** (2048 čísel na buňku) a rozdělí je na `train` / `val` / `test`.
Tyhle soubory pak nahrajete na GitHub Release a studenti je jen stahují.

Výstup:
- `train.npz`, `val.npz` — featury **+ labely** (pro studenty)
- `test_features.npz` — featury **bez labelů** (pro studenty, na odevzdání)
- `test_labels.npz` — labely testu (**necháte si**, na vyhodnocení soutěže)
- `samples.zip` — pár ukázkových obrázků do studentského notebooku

Běh na Colab T4 trvá řádově **5–10 minut**.
"""))

inst.append(code(r"""
# kagglehub bývá na Colabu předinstalovaný; pro jistotu:
!pip -q install kagglehub

import os, glob, zipfile, shutil
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"
print("zařízení:", device, "| GPU:", torch.cuda.get_device_name(0) if device == "cuda" else "—")
"""))

inst.append(md(r"""
## 1 · Stažení datasetu

Stáhneme Kaggle dataset `iarunava/cell-images-for-detecting-malaria`. Sada má dvě složky
`Parasitized/` a `Uninfected/`, ale historicky obsahuje i **vnořenou složku** a zatoulané
ne-obrázkové soubory (`Thumbs.db`, `*.db`). Proto sebereme rekurzivně **jen `*.png`** a label
určíme z názvu nadřazené složky.
"""))

inst.append(code(r"""
import kagglehub
root = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
print("staženo do:", root)

png = glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
items = []
for p in png:
    parent = os.path.basename(os.path.dirname(p)).lower()
    if "parasitized" in parent:
        items.append((p, 1))
    elif "uninfected" in parent:
        items.append((p, 0))
# DETERMINISTICKÉ pořadí (nezávislé na stroji) → stejný split všude, sedí na test_labels.npz
items.sort(key=lambda t: (os.path.basename(t[0]), t[1]))
print("nalezeno obrázků:", len(items),
      "| napadených:", sum(l for _, l in items),
      "| zdravých:", sum(1 - l for _, l in items))
"""))

inst.append(md(r"""
## 2 · Předzpracování a DataLoader

Každý obrázek zmenšíme na 224×224 a znormalizujeme přesně tak, jak to očekává ResNet
předtrénovaný na ImageNetu (proto ty konkrétní průměry a odchylky).
"""))

inst.append(code(r"""
prep = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class BunkyDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label

loader = DataLoader(BunkyDataset(items, prep), batch_size=128,
                    shuffle=False, num_workers=2)
print("dávek:", len(loader))
"""))

inst.append(md(r"""
## 3 · Encoder ResNet50 → featury

Načteme ResNet50 s vahami z ImageNetu, **odřízneme poslední klasifikační vrstvu**
(`fc = Identity`) a necháme ho v režimu `eval()` bez počítání gradientů. Výstupem je pro
každou buňku **2048-rozměrný vektor**.
"""))

inst.append(code(r"""
encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
encoder.fc = nn.Identity()
encoder.eval().to(device)

feats, labels = [], []
with torch.no_grad():
    for k, (xb, yb) in enumerate(loader):
        f = encoder(xb.to(device)).cpu().numpy().astype(np.float16)
        feats.append(f); labels.append(np.asarray(yb))
        if k % 20 == 0:
            print(f"  zpracováno {k*128:6d} / {len(items)}")
X = np.concatenate(feats).astype(np.float16)
y = np.concatenate(labels).astype(np.int64)
print("hotovo. featury:", X.shape, "| labely:", y.shape, "| dtype:", X.dtype)
"""))

inst.append(md(r"""
## 4 · Stratifikovaný split 70 / 15 / 15

Rozdělíme **stratifikovaně** (zachová poměr tříd) a s **pevným seedem**, aby oba týmy
pracovaly s naprosto stejnými daty — soutěž musí být férová.
"""))

inst.append(code(r"""
from sklearn.model_selection import train_test_split

X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=42)  # 0.1765*0.85 ≈ 0.15

for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
    print(f"{name:5s}: {len(yy):6d}  (napadených {int(yy.sum())}, zdravých {int((1-yy).sum())})")

np.savez_compressed("train.npz", X=X_train.astype(np.float16), y=y_train)
np.savez_compressed("val.npz",   X=X_val.astype(np.float16),   y=y_val)
np.savez_compressed("test_features.npz", X=X_test.astype(np.float16))   # BEZ labelů
np.savez_compressed("test_labels.npz",   y=y_test)                       # jen pro vás!
for f in ["train.npz", "val.npz", "test_features.npz", "test_labels.npz"]:
    print(f, "→", round(os.path.getsize(f)/1e6, 1), "MB")
"""))

inst.append(md(r"""
## 5 · Ukázkové obrázky pro studenty

Zkopírujeme pár desítek obrázků (zvlášť napadené a zdravé), aby si je studenti mohli
prohlédnout v sekci 2 svého notebooku.
"""))

inst.append(code(r"""
os.makedirs("samples/Parasitized", exist_ok=True)
os.makedirs("samples/Uninfected", exist_ok=True)
paras = [p for p, l in items if l == 1][:15]
heal  = [p for p, l in items if l == 0][:15]
for p in paras: shutil.copy(p, "samples/Parasitized/")
for p in heal:  shutil.copy(p, "samples/Uninfected/")

with zipfile.ZipFile("samples.zip", "w") as z:
    for f in glob.glob("samples/**/*.png", recursive=True):
        z.write(f)
print("samples.zip →", round(os.path.getsize("samples.zip")/1e6, 2), "MB")
"""))

inst.append(md(r"""
## 6 · Kontrola a hosting

Rychlá kontrola, že soubory jdou znovu načíst a mají správné tvary. Pak je nahrajte:

**Doporučeno — GitHub Release:**
1. Vytvořte (klidně prázdný) repozitář, např. `malaria-veletrh`.
2. *Releases → Draft a new release*, tag `data-v1`.
3. Přetáhněte sem `train.npz`, `val.npz`, `test_features.npz`, `samples.zip`
   (`test_labels.npz` **NE** — ten si necháte!).
4. Publish. URL assetu bude `https://github.com/<vy>/<repo>/releases/download/data-v1/train.npz`.
5. Do studentského notebooku doplňte `DATA_BASE_URL` na
   `https://github.com/<vy>/<repo>/releases/download/data-v1`.

**Záloha — Google Drive:** nahrajte soubory, dejte „kdokoli s odkazem", a ve studentském
notebooku použijte `gdown <ID>` místo `wget`.
"""))

inst.append(code(r"""
for f in ["train.npz", "val.npz", "test_features.npz"]:
    d = np.load(f)
    print(f, "→ X", d["X"].shape, "| klíče:", list(d.keys()))
d = np.load("test_labels.npz")
print("test_labels.npz → y", d["y"].shape, "(zůstává jen u vás)")
print("\nVše připraveno k nahrání. 🎉")
"""))

build(inst, "prepare_features_INSTRUCTOR.ipynb")

print("\nHotovo — oba notebooky vygenerovány.")
