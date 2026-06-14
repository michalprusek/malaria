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
# 🦟 Detekuje malárii lépe člověk, nebo notebook?

### Miniprojekt — Veletrh vědy · Katedra matematiky FJFI ČVUT

Malárie každý rok zabije přibližně **600 000 lidí**. Drtivá většina obětí žije v oblastech,
kde chybí vyškolení patologové, kteří by uměli pod mikroskopem rozpoznat parazita
*Plasmodium* v krevním nátěru. Diagnóza přitom stojí na jediné otázce, opakované u tisíců
buněk: **je tahle červená krvinka napadená, nebo zdravá?**

Co kdyby tu práci zvládl obyčejný mobil s dobře natrénovaným algoritmem?

Dnes si to vyzkoušíme. Použijeme veřejný dataset amerického Národního institutu zdraví (NIH)
s **27 558 mikroskopickými snímky** jednotlivých buněk — polovina zdravých, polovina
napadených.

---

### Co je hluboké učení a *transfer learning*?

**Neuronová síť** je matematická funkce s miliony nastavitelných čísel (*vah*), kterou
„naučíme" tím, že jí ukazujeme příklady a opravujeme její chyby. **Konvoluční** neuronová síť
je typ, který se specializuje na obrázky — postupně z pixelů skládá hrany, textury, tvary.

Natrénovat takovou síť od nuly vyžaduje miliony obrázků a hodiny počítání na výkonných
grafických kartách. My na to nemáme čas ani data — a nemusíme. Použijeme **transfer learning**:

> Vezmeme síť **ResNet**, kterou už někdo natrénoval na milionech běžných fotek
> (psi, auta, židle…), a využijeme to, že se naučila *vidět* — rozpoznávat tvary a textury.
> Tuhle „zrakovinu vidění" necháme **zmraženou** a dotrénujeme jen malou **klasifikační
> hlavu**, která se rozhodne: *infikovaná, nebo zdravá?*

To je celý dnešní trik. A protože je zbytek sítě zmražený, trénink poběží i na vašem
notebooku **bez grafické karty** — během pár sekund.

---

### Co dnes uděláte

1. Podíváte se na skutečné snímky buněk.
2. Pochopíte, jak ResNet promění obrázek na **2048 čísel** (tzv. *featury*).
3. Postavíte jednoduchý **baseline (k-NN)** a pak ho překonáte **vlastní natrénovanou sítí.**
4. Naučíte se, proč v medicíně **nestačí hlásit „95 % přesnost"** — a co je *senzitivita*,
   *specificita* a *matice záměn*.
5. **Soutěž!** Dva týmy proti sobě. Vyhrává model, který nejlépe zachytí nemocné, aniž by
   příliš často zbytečně strašil zdravé.

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
## 3 · Co je *encoder* a *featury*

Lidské oko parazita pozná. Jak ho ale „uvidí" počítač, který zná jen čísla?

Tady přichází na řadu **ResNet** — náš *encoder*. Je to konvoluční síť předtrénovaná na
milionech fotek. Když jí předhodíme obrázek buňky, **promění ho na vektor 2048 čísel**.
Tahle čísla shrnují, *co na obrázku je* — jaké tam jsou tvary, textury, skvrny. Říká se jim
**featury** (příznaky).

Klíčová myšlenka:

> Encoder jsme **zmrazili** — neučí se. Jen překládá `obrázek → 2048 čísel`.
> My pak učíme jen malou hlavu, která z těch 2048 čísel rozhodne *infikovaná / zdravá*.

Načteme si připravené featury a podíváme se, jak vypadají.
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
## 5 · Lineární projekce do 2D (linear probe)

PCA jsme dělali „naslepo“ — bez znalosti správných odpovědí. Teď zkusíme opak: necháme
**jedinou lineární vrstvu** naučit se z 2048 featur **dvě čísla** (skóre pro „zdravá“ a pro
„napadená“) přímo z labelů. Tomu se říká **linear probe** — lineární klasifikátor nasazený na
zmrazené featury. Je to nejjednodušší *učený* model: featury jen zváží a sečte.

Uvidíme, že i tahle jediná vrstva třídy slušně oddělí — a dostaneme názorný 2D obrázek
s **dělící čarou** (tam, kde se obě skóre rovnají).
"""))

student.append(code(r"""
ytr_long = torch.tensor(y_train, dtype=torch.long, device=device)

probe = nn.Linear(Xtr.shape[1], 2).to(device)        # 2048 featur → 2 skóre (zdravá, napadená)
opt_p = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
ce = nn.CrossEntropyLoss()
for epoch in range(15):
    probe.train()
    perm = torch.randperm(Xtr.shape[0], device=device)
    for i in range(0, Xtr.shape[0], 256):
        idx = perm[i:i + 256]
        opt_p.zero_grad()
        ce(probe(Xtr[idx]), ytr_long[idx]).backward()
        opt_p.step()

probe.eval()
with torch.no_grad():
    z_val = probe(Xva).cpu().numpy()                 # 2D skóre pro ověřovací buňky
probe_acc = (z_val.argmax(1) == y_val).mean()
print(f"Linear probe — přesnost na ověření: {probe_acc:.4f}")
"""))

student.append(code(r"""
plt.figure(figsize=(7, 6))
for label, name, col in [(0, "zdravá", "seagreen"), (1, "napadená", "crimson")]:
    m = y_val == label
    plt.scatter(z_val[m, 0], z_val[m, 1], s=8, alpha=0.4, color=col, label=name)
lim = [float(z_val.min()), float(z_val.max())]
plt.plot(lim, lim, ls="--", color="gray", label="dělící čára (skóre stejné)")
plt.xlabel("skóre: zdravá"); plt.ylabel("skóre: napadená")
plt.legend(); plt.title(f"Linear probe — 2D projekce (přesnost {probe_acc:.3f})")
plt.show()
"""))

student.append(md(r"""
## 6 · Druhý baseline: k nejbližších sousedů (k-NN)

Vedle lineárního probu zkusíme i **model úplně bez učení** — další laťku, kterou se pak budeme
snažit překonat. Použijeme **k-NN** (*k nearest neighbors*, k nejbližších sousedů):

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
### Jak vznikne ta pravděpodobnost?

k-NN nevydává tvrdé „ano/ne", ale **pravděpodobnost**: najde k nejbližších sousedů a vezme
**vážený podíl** těch napadených, kde bližší soused váží víc (`váha = 1 / vzdálenost`). Ukažme
si to na jedné konkrétní buňce — spočítáme to ručně a porovnáme se `sklearn`.
"""))

student.append(code(r"""
i = 0  # 🔬 zkus jiný index buňky
dist, nbr = knn.kneighbors(X_val_s[i:i+1], n_neighbors=K)   # vzdálenosti a indexy K sousedů
dist, nbr = dist[0], nbr[0]
labels = y_train[nbr]                       # třídy sousedů (1 = napadená)
w = 1.0 / np.maximum(dist, 1e-12)           # vzdálenostní váhy (bližší = větší)
p_rucne = w[labels == 1].sum() / w.sum()    # vážený podíl napadených sousedů

print("vzdálenosti 5 nejbližších:", np.round(dist[:5], 3))
print("jejich třídy:            ", labels[:5], "(1 = napadená)")
print(f"napadených mezi {K} sousedy: {int(labels.sum())}/{K}")
print(f"ruční pravděpodobnost (vážená): {p_rucne:.3f}")
print(f"sklearn predict_proba:          {knn.predict_proba(X_val_s[i:i+1])[0, 1]:.3f}")
print("→ sedí: pravděpodobnost = vážený podíl napadených mezi sousedy.")
"""))

student.append(md(r"""
### A jak zvolit práh?

Z pravděpodobnosti uděláme rozhodnutí až **prahem**. Práh 0,5 není svatý — volíme ho podle
cíle. My chceme zachytit **aspoň 99 % nemocných**, takže práh *snižujeme*, dokud senzitivita
nevyleze na 99 % (na **ověřovací** sadě, nikdy na testu!). Uvidíme, že u k-NN přitom specificita
spadne skoro k nule — proto baseline na tu laťku nedosáhne.
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

Teď to nejzábavnější — **navrhnete vlastní malou neuronovou síť**, která má překonat oba
baseline. Dostane na vstupu 2048 featur a na výstupu vydá jediné číslo (*logit*), které po
převedení funkcí *sigmoid* říká **pravděpodobnost, že je buňka napadená**.

Baseline níže má jednu skrytou vrstvu. **Tady experimentujte:**
- `hidden` — kolik neuronů má skrytá vrstva (víc = větší kapacita, ale i riziko přeučení),
- `dropout` — náhodně „vypne" část neuronů při tréninku, brání přeučení (0 až ~0.5),
- můžete přidat **další vrstvy** (`nn.Linear`, `nn.ReLU`, `nn.Dropout`).
"""))

student.append(code(r"""
class Hlava(nn.Module):
    def __init__(self, in_dim=2048, hidden=128, dropout=0.3):   # 🔬 EXPERIMENT: hidden, dropout
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),       # 🔬 EXPERIMENT: zkus přidat další vrstvy nad tenhle řádek
        )

    def forward(self, x):
        return self.net(x).squeeze(1)   # vrací logit, tvar (batch,)


model = Hlava(in_dim=Xtr.shape[1], hidden=128, dropout=0.3).to(device)
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
- `WEIGHT_DECAY` — mírná penalizace velkých vah, brání přeučení,
- `POS_WEIGHT` — **jak draho stojí přehlédnutý nemocný**. Hodnota > 1 nutí model brát napadené
  buňky vážněji (zvýší senzitivitu na úkor falešných poplachů). To je přímo páka na medicínský
  kompromis, o kterém bude řeč v sekci 9!
"""))

student.append(code(r"""
# ====== 🔬 EXPERIMENT: hyperparametry ======
EPOCHS       = 25
BATCH        = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
POS_WEIGHT   = 1.0      # > 1 = přísnější na přehlédnuté nemocné
# ===========================================

loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=device))
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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
plt.plot(fpr_k, tpr_k, lw=2, ls="--", color="gray", label=f"k-NN baseline (spec={spec99_knn:.3f})")
plt.axhline(0.99, ls=":", color="crimson", label="požadovaná senzitivita 99 %")
plt.scatter([1 - spec99], [sens99], color="crimson", zorder=5)
plt.plot([0, 1], [0, 1], ls=":", color="lightgray")
plt.xlabel("1 − specificita  (falešné poplachy)"); plt.ylabel("senzitivita")
plt.title("ROC křivka — vaše síť vs. baseline"); plt.legend(loc="lower right"); plt.show()

print(f"➡️  SOUTĚŽNÍ SKÓRE (na ověření): specificita @ senzitivita≥99 % = {spec99:.3f}")
print(f"    (odpovídající práh = {prah99:.3f}, dosažená senzitivita = {sens99:.3f})")
print(f"    k-NN baseline = {spec99_knn:.3f}  →  "
      f"{'PŘEKONÁNO! 🎉' if spec99 > spec99_knn else 'zatím ne — laďte dál'}")
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
| sekce 7 | architektura hlavy | větší `hidden`, jiný `dropout`, další vrstvy |
| sekce 8 | `EPOCHS` | víc epoch — ale pozor na přeučení (sledujte ověření) |
| sekce 8 | `LR` | např. 3e-4, 1e-3, 3e-3 |
| sekce 8 | `WEIGHT_DECAY` | větší hodnota brzdí přeučení |
| sekce 8 | `POS_WEIGHT` | > 1 zvýší senzitivitu (přísnější na nemocné) |
| sekce 4 | standardizace | už zapnutá — zkuste, jaký je rozdíl bez ní |

**Postup:** změňte jednu věc → spusťte sekce 7, 8, 9 → podívejte se na soutěžní skóre →
opakujte. Vždycky měňte **jen jednu věc**, ať víte, co pomohlo.

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
## 🎓 Co jste se naučili

- **Transfer learning**: stačí dotrénovat malou hlavu nad zmraženým ResNetem — žádné
  superpočítače, žádné miliony obrázků.
- Featury jsou **2048 čísel**, do kterých předtrénovaná síť zhustila „co na obrázku je".
- **Baseline vs. učení**: k-NN si data jen pamatuje a ve 2048 rozměrech tápe; trénovaná síť se
  naučí, na čem záleží — proto baseline výrazně překoná. (Dobrý baseline ale vždy chcete mít,
  abyste věděli, co váš model vlastně přidává.)
- **Přesnost klame.** V medicíně sledujeme **senzitivitu** (nepřehlédnout nemocného) a
  **specificitu** (nestrašit zdravé), čteme **matici záměn** a **ROC křivku**.
- **Práh** je rozhodnutí o ceně chyby — a ta cena není symetrická.

### Otázky k zamyšlení
1. Proč by se model nasazený v nemocnici mohl chovat hůř než tady na datasetu? (Jiný mikroskop,
   jiné barvení, jiná populace pacientů…)
2. Měla by konečné rozhodnutí dělat AI sama, nebo jen *upozorňovat* lékaře? Kde nastavit práh?
3. Co by se stalo se senzitivitou, kdyby byla v reálu napadená jen **1 %** buněk?

**Gratulace — mezi vaším notebookem a reálným dopadem na zdravotnictví není tak daleko. 🦟➡️🩺**
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
