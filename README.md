# 🦟 Miniprojekt: Detekuje malárii lépe člověk, nebo notebook?

Výukové materiály pro středoškoláky na veletrh vědy — klasifikace mikroskopických snímků
červených krvinek (zdravá / napadená *Plasmodium*) pomocí **transfer learningu** s ResNet.
Dva týmy soutěží o nejlepší model, hodnocený **medicínsky relevantní metrikou**.

> Školitelé: Bc. Michal Průšek, Bc. Michal Bělohlávek · Katedra matematiky FJFI ČVUT
> Kontakt: prusemic@cvut.cz

---

## Co je v repozitáři

| soubor | komu | k čemu |
|---|---|---|
| `malaria_classifier_STUDENT.ipynb` | **studenti** | hlavní notebook akce — data, trénink hlavy, metriky, soutěž |
| `prepare_features_INSTRUCTOR.ipynb` | školitel | volitelně: jak featury vznikly (referenční, na Colab GPU) |
| `score_submission_INSTRUCTOR.py` | školitel | vyhodnotí odevzdání týmů a vyhlásí vítěze |
| `main.tex` / `main_en.tex` (+ `refs.bib`, `figs/`) | **studenti** | šablona reportu (LaTeX/Overleaf) v češtině i angličtině: problém, řešení ResNet50 + k-NN, vyhodnocení — k přepsání podle vlastního řešení. Bibliografie v `refs.bib` (BibTeX). |
| `prezentace.html` | školitel | úvodní prezentace (self-contained HTML, otevře se v prohlížeči): představení, problém, CNN/ResNet, PCA featur, k-NN, MLP. Navigace šipkami. |
| `tools/extract_features_local.py` | školitel | lokální generátor datových souborů (běží i na Apple MPS) |
| `tools/build_notebooks.py` | školitel | znovu vygeneruje oba `.ipynb` po úpravách |
| `README.md` | — | tenhle návod |

**Data se needistribuují v repu**, ale jako **GitHub Release `data-v1`** (viz níže).

---

## Pedagogická myšlenka ve třech větách

1. **Transfer learning**: zmrazený ResNet50 promění každý obrázek na 2048 čísel (*featury*),
   studenti učí jen malou klasifikační **hlavu** — trénink běží sekundy.
2. **Featury se počítají jen jednou**, pak studenti iterují bleskově → stihnou desítky
   experimentů za odpoledne.
3. **Didaktické těžiště** není přesnost, ale **proč „95 % přesnost" v medicíně neznamená dobrý
   model** — senzitivita, specificita, matice záměn, ROC křivka, cena přehlédnutého nemocného.

Dataset: NIH [Malaria Cell Images](https://data.lhncbc.nlm.nih.gov/public/Malaria/),
27 558 buněk, vyvážený (13 779 napadených + 13 779 zdravých).

---

## 📦 Jak studentský notebook získá data (dvě cesty, automaticky)

Notebook sám pozná, kterou cestou jít:

- **Rychlá cesta** — stáhne předpočítané featury z **GitHub Release** (pár sekund, bez GPU).
  Funguje, **pokud je repozitář veřejný** (asset Releasu musí být veřejně stažitelný).
- **Soběstačná cesta** — když rychlá data nejsou dostupná (privátní repo), notebook si
  stáhne původní obrázky z **veřejného NIH serveru** a featury si **sám spočítá** ResNetem.
  Trvá ~5 min a potřebuje zapnuté Colab GPU (*Runtime → Change runtime type → GPU*).

> ✅ **Doporučení:** pokud chcete pro studenty tu nejhladší variantu (žádné čekání, žádné GPU),
> **přepněte repozitář na veřejný**:
> ```bash
> gh repo edit michalprusek/malaria --visibility public --accept-visibility-change-consequences
> ```
> Release `data-v1` s daty už je vytvořený, takže rychlá cesta začne fungovat okamžitě.
> Když repo necháte privátní, vše funguje taky — jen se použije soběstačná cesta.

---

## ⚙️ Příprava před akcí (školitel)

Většina je hotová — data jsou vygenerovaná a nahraná v Release `data-v1`. Zbývá:

1. **Scoring klíč**: soubor `test_labels.npz` (pravé odpovědi skrytého testu) **NENÍ v repu
   ani v Release** (záměrně — je to klíč k soutěži). Vygenerujete ho lokálně:
   ```bash
   # stáhne NIH data a vyrobí všechny .npz včetně test_labels.npz
   curl -L -o _data/cell_images.zip https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip
   python3 tools/extract_features_local.py
   ```
   `test_labels.npz` si uschovejte na vyhodnocení.
2. **Notebook studentům**: dejte odkaz „Open in Colab" na `malaria_classifier_STUDENT.ipynb`
   z repa, nebo notebook nasdílejte přes Google Drive. Každý tým si v Colabu udělá vlastní
   kopii (*File → Save a copy in Drive*).

*(Volitelně: `DATA_BASE_URL` v notebooku už ukazuje na Release tohoto repa — měňte jen při
přesunu dat jinam.)*

---

## 🗓️ Modelový průběh dne (~6 h)

| čas | blok | notebook |
|---|---|---|
| 09:00–09:15 | Sraz před vrátnicí Trojanova, rozdělení na 2 týmy | — |
| 09:15–10:00 | Úvod: malárie + základy DL a transfer learningu | (výklad / sekce 0) |
| 10:00–10:30 | Setup Colabu, získání dat, prohlídka buněk | sekce 1–2 |
| 10:30–11:00 | Encoder, featury, **PCA vizualizace** (aha moment) | sekce 3 |
| 11:00–11:45 | Baseline (linear probe, k-NN) + první hlava + trénink | sekce 5–8 |
| 11:45–12:30 | Oběd | — |
| 12:30–13:15 | **Medicínské metriky**: matice záměn, senzitivita/specificita, ROC, práh | sekce 9 |
| 13:15–15:00 | 🏆 **SOUTĚŽ**: týmy ladí a odevzdávají | sekce 10–11 |
| 15:00–15:30 | Vyhodnocení na skrytém testu, vyhlášení, diskuse | `score_submission` |

---

## 🏆 Jak proběhne soutěž

1. Oba týmy mají **identická** data (deterministický split podle pevného seedu) — férové.
2. Ladí klasifikační hlavu (páky níže), sledují soutěžní skóre na **ověřovací** sadě.
3. Skutečné pořadí se počítá na **skrytém testu**, který nevidí → učí se *neoverfitovat*.
4. Každý tým odevzdá `predikce_<TÝM>.csv` (pravděpodobnosti na testu).
5. Vy soubory dáte vedle svého `test_labels.npz` a spustíte:
   ```bash
   python3 score_submission_INSTRUCTOR.py
   ```
   Skript vypíše žebříček (🥇🥈) a uloží `vysledky_soutez.png` se srovnáním ROC křivek
   — ideální na projektor.

**Soutěžní metrika = specificita při senzitivitě ≥ 99 %**: *Jak málo zdravých zbytečně
poplašíme, když musíme zachytit aspoň 99 % nemocných?* U smrtelné nemoci je „nepřehlédnout
nemocného" nepodkročitelná podmínka, proto fixujeme senzitivitu vysoko a soutěžíme ve specificitě.
(Práh úrovně se mění v `score_submission_INSTRUCTOR.py` přepínačem `--min-sens`.)

> 💡 Protože je split deterministický, `test_labels.npz` sedí na predikce z **obou** datových
> cest (rychlé i soběstačné) — scoring funguje vždy.

---

## 🔬 Páky, se kterými studenti experimentují

Notebook staví dva baseline — **linear probe (sekce 5)** a **k-NN (sekce 6)** — a pak je
studenti překonávají **vlastní neuronovou hlavou**. Páky vyznačené `# 🔬 EXPERIMENT`:

- **linear probe** (sekce 5): učená lineární projekce 2048 → 2D; názorný 2D obrázek + lineární baseline,
- **`K` u k-NN** (sekce 6): počet hlasujících sousedů — posune baseline,
- **architektura hlavy** (sekce 7): počet a šířka vrstev, `dropout`,
- **`EPOCHS`, `LR`, `WEIGHT_DECAY`** (sekce 8): délka a tempo učení, regularizace,
- **`POS_WEIGHT`** (sekce 8): jak draho stojí přehlédnutý nemocný — přímá páka na senzitivitu,
- **standardizace featur** (sekce 4): zapnutá; lze ukázat rozdíl bez ní.

Zlaté pravidlo pro studenty: **měňte vždy jen jednu věc**, ať poznáte, co pomohlo.

---

## 🧯 Řešení typických problémů

| problém | řešení |
|---|---|
| „Režim dat: sobestacna" + dlouho to trvá | zapněte Colab GPU (*Runtime → Change runtime type → GPU*) nebo zveřejněte repo (rychlá cesta) |
| rychlá cesta nestahuje | repo je privátní → buď ho zveřejněte, nebo nechte běžet soběstačnou cestu |
| Colab se odpojí | *Runtime → Run all* znovu; stažená data zůstávají na disku session |
| posuvník prahu se nezobrazí | notebook má fallback s tabulkou prahů (funguje i bez `ipywidgets`) |
| trénink je podezřele dokonalý (100 %) | nejspíš přeučení — sledujte přesnost na **ověření**, ne na tréninku |
| chci přegenerovat notebooky | upravte `tools/build_notebooks.py`, spusťte `python3 tools/build_notebooks.py` |

---

## 🛠️ Technické poznámky

- **Encoder**: ResNet50 (ImageNet, váhy `IMAGENET1K_V2`), penultimate vrstva = 2048 featur,
  ukládáno jako `float16` (komprimované `.npz`: train 54 MB + val 12 MB + test 12 MB ≈ 78 MB).
- **Split**: stratifikovaný 70 / 15 / 15 (train 19 289, val 4 135, test 4 134), pevný seed 42,
  **deterministické pořadí obrázků** (řazeno podle názvu souboru → stejný split na každém stroji).
- **Baseline = k-NN** (k=31, vážený vzdáleností, na standardizovaných featurech): AUC ≈ 0,93,
  ale soutěžní metrika **specificita @ senzitivita ≥ 99 % = 0,00** — aby chytil 99 % nemocných,
  musel by označit skoro všechny (prokletí dimenzionality). Senzitivitu 99 % tedy splní, ale
  s nulovou specificitou (prakticky bezcenně), což přesně motivuje učený model.
- **Trénovaná hlava** (malá MLP v PyTorch, `BCEWithLogitsLoss`, Adam): AUC ≈ 0,98 a
  **specificita ≈ 0,63 @ senzitivita 99 %** — výrazně překoná baseline.
- Notebooky i scoring byly otestovány spuštěním end-to-end; featury byly extrahovány a
  ověřeny na reálných datech.
- Augmentace dat zde nelze (featury jsou fixní) — vhodné téma na diskusi „co s plnými obrázky".

## Kredity

- Diagram architektury ResNet v prezentaci: **ResNet-18** z knihy *Dive into Deep Learning*
  (d2l.ai) — [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Resnet-18_architecture_(rotated).svg),
  licence **CC BY-SA 4.0** (autoři A. Zhang, Z. C. Lipton, M. Li, A. J. Smola). V prezentaci je
  použita **ležatá** varianta `figs/resnet18_d2l_rotated.svg` (portrétní originál `figs/resnet18_d2l.svg`).
