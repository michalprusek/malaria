"""Spustí studentský notebook (rychlá cesta) na syntetických datech a ověří běh bez chyby.

Podstrčí platné train/val/test_features.npz + samples.zip (>1 kB), takže REZIM zůstane
'predpocitana' a soběstačná větev (stahování 350 MB) se NEspustí.
"""
import os, shutil, tempfile, zipfile
import numpy as np
from PIL import Image
import nbformat
from nbclient import NotebookClient

RNG = np.random.default_rng(0)
DIM = 2048


def split(n):
    X = np.vstack([RNG.standard_normal((n, DIM)).astype(np.float32),
                   RNG.standard_normal((n, DIM)).astype(np.float32) + 0.9]).astype(np.float16)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int64)
    p = RNG.permutation(len(y))
    return X[p], y[p]


work = tempfile.mkdtemp(prefix="malaria_valB_")
Xtr, ytr = split(3000); Xva, yva = split(800); Xte, yte = split(800)
np.savez_compressed(os.path.join(work, "train.npz"), X=Xtr, y=ytr)
np.savez_compressed(os.path.join(work, "val.npz"), X=Xva, y=yva)
np.savez_compressed(os.path.join(work, "test_features.npz"), X=Xte)

# samples.zip s prefixem samples/ a pár PNG (>1 kB), aby REZIM zůstal 'predpocitana'
os.makedirs(os.path.join(work, "_s", "samples", "Parasitized"))
os.makedirs(os.path.join(work, "_s", "samples", "Uninfected"))
for cls in ["Parasitized", "Uninfected"]:
    for i in range(3):
        arr = (RNG.random((64, 64, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(work, "_s", "samples", cls, f"img{i}.png"))
with zipfile.ZipFile(os.path.join(work, "samples.zip"), "w") as z:
    base = os.path.join(work, "_s")
    for f in [os.path.join(dp, fn) for dp, _, fns in os.walk(base) for fn in fns]:
        z.write(f, os.path.relpath(f, base))
shutil.rmtree(os.path.join(work, "_s"))
assert os.path.getsize(os.path.join(work, "samples.zip")) > 1000

nb = nbformat.read(os.path.abspath("malaria_classifier_STUDENT.ipynb"), as_version=4)
os.environ["MPLBACKEND"] = "Agg"
NotebookClient(nb, timeout=600, kernel_name="python3",
               resources={"metadata": {"path": work}}).execute()
print("✅ Studentský notebook (rychlá cesta) proběhl bez chyby.")

# REZIM musí být 'predpocitana' (jinak by se spustilo stahování NIH)
src = "".join(c.source for c in nb.cells if c.cell_type == "code")
outs = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code"
               for o in c.get("outputs", []))
assert "Režim dat: predpocitana" in outs, "REZIM nebyl 'predpocitana'! " + outs[:500]
sub = os.path.join(work, "predikce_TYM_A.csv")
assert os.path.exists(sub) and sum(1 for _ in open(sub)) - 1 == len(yte)
print("✅ REZIM=predpocitana, odevzdání má správný počet řádků.")
shutil.rmtree(work)
