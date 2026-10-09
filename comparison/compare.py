#claude code
"""
Συγκρίνει glosses δίπλα-δίπλα: κάθε γραμμή = ένα κλιπ, κάθε στήλη = frame (ομοιόμορφη δειγματοληψία).
Βγάζει ένα PNG ανά ομάδα στον φάκελο OUT_DIR.

Χρήση:
    python compare_glosses.py --root /path/to/frames --meta metadata_with_nframes.csv
    python compare_glosses.py --root ... --meta ... --group EGO --signers 2 4 5
    python compare_glosses.py --root ... --meta ... --paths kep1_signer4_rep4_glosses/glosses0058 police3_signer2_rep2_glosses/glosses0002

ROOT: ο φάκελος που περιέχει τους φακέλους τύπου "police1_signer1_rep1_glosses/glosses0000".
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

IMG_EXT = ("jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff")

# Ομάδες προς έλεγχο
GROUPS = {
    "EGO":         ["ΕΓΩ(1)", "ΕΓΩ(2)", "ΕΓΩ(3)"],
    "GIATROS":     ["ΓΙΑΤΡΟΣ(1)", "ΓΙΑΤΡΟΣ(2)"],
    "KATO":        ["ΚΑΤΩ(1)", "ΚΑΤΩ(2)"],
    "KOKKINO":     ["ΚΟΚΚΙΝΟ(1)", "ΚΟΚΚΙΝΟ(2)"],
    "KLEVO":       ["ΚΛΕΒΩ(2)_ΕΜΕΝΑ", "ΚΛΕΒΩ(2)_ΕΣΕΝΑ"],
    "DIKO":        ["ΔΙΚΟ_ΣΟΥ", "ΔΙΚΟ_ΤΟΥ"],
    "DINO":        ["ΕΓΩ_ΔΙΝΩ_ΕΣΕΝΑ", "ΕΣΥ_ΔΙΝΩ_ΕΜΕΝΑ", "ΑΥΤΗ_ΔΙΝΩ_ΕΣΕΝΑ"],
    "ESY_EGO":     ["ΕΣΥ", "ΕΓΩ(1)"],
    "FINGERSPELL": ["ΙΚΑ(Δ.Α.)", "ΚΕΠΑ(Δ.Α.)", "ΑΜΚΑ(Δ.Α.)", "ΑΦΜ(Δ.Α.)", "ΙΜΕΙ(Δ.Α.)"],
    "NUMBERS":     ["10", "15", "18", "20", "1000", "200", "2"],
}


def list_frames(root, rel_path):
    folder = os.path.join(root, rel_path)
    if not os.path.isdir(folder):
        # δοκίμασε να βρεις τον φάκελο σε υποφακέλους του root (π.χ. root/train/...)
        hits = glob.glob(os.path.join(root, "**", rel_path), recursive=True)
        if hits:
            folder = hits[0]
        else:
            print(f"[ΛΑΘΟΣ ΦΑΚΕΛΟΣ] δεν υπάρχει: {folder}")
            return []
    files = []
    for ext in IMG_EXT:
        files += glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True)
        files += glob.glob(os.path.join(folder, "**", f"*.{ext.upper()}"), recursive=True)
    if not files:
        print(f"[ΧΩΡΙΣ ΕΙΚΟΝΕΣ] {folder} περιέχει: {sorted(os.listdir(folder))[:5]}")
    return sorted(files)


def sample_idx(n, k):
    if n <= k:
        return list(range(n))
    return np.linspace(0, n - 1, k).round().astype(int).tolist()


def pick_clips(meta, glosses, signers, split_for_extra="test"):
    """Ένα κλιπ ανά (gloss, signer) από train (διάμεσο n_frames) + ένα από test."""
    rows = []
    for g in glosses:
        sub = meta[(meta.label == g) & (meta.split == "train")]
        for s in signers:
            ss = sub[sub.signer == s].sort_values("n_frames")
            if len(ss):
                r = ss.iloc[len(ss) // 2]
                rows.append((g, f"signer {s} (train)", r.path, r.n_frames))
        te = meta[(meta.label == g) & (meta.split == split_for_extra)]
        if len(te):
            r = te.sort_values("n_frames").iloc[len(te) // 2]
            rows.append((g, f"signer {r.signer} ({split_for_extra})", r.path, r.n_frames))
        else:
            rows.append((g, f"({split_for_extra}: ΚΑΝΕΝΑ ΔΕΙΓΜΑ)", None, 0))
    return rows


def make_montage(rows, root, out_png, title, n_cols=8):
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols + 2.5, 1.7 * n_rows),
                             squeeze=False)
    for i, (gloss, who, path, nf) in enumerate(rows):
        files = list_frames(root, path) if path else []
        idx = sample_idx(len(files), n_cols)
        for j in range(n_cols):
            ax = axes[i][j]
            ax.axis("off")
            if j < len(idx):
                ax.imshow(mpimg.imread(files[idx[j]]))
        label = f"{gloss}\n{who}\n{nf} fr"
        if path and not files:
            label += "\n(ΔΕΝ βρέθηκαν εικόνες)"
        axes[i][0].text(-0.08, 0.5, label, transform=axes[i][0].transAxes,
                        ha="right", va="center", fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0.08, 0, 1, 0.97))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print("saved", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="φάκελος με τα frames")
    ap.add_argument("--meta", required=True, help="metadata.csv")
    ap.add_argument("--out", default="compare_out")
    ap.add_argument("--group", default=None, help=f"ένα από: {', '.join(GROUPS)} (default: όλα)")
    ap.add_argument("--signers", type=int, nargs="+", default=[2, 4, 5],
                    help="signers από train για κάθε gloss")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--paths", nargs="+", default=None,
                    help="συγκεκριμένοι φάκελοι κλιπ (παρακάμπτει τα groups)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    meta = pd.read_csv(a.meta)

    if a.paths:
        rows = []
        for p in a.paths:
            m = meta[meta.path == p]
            if len(m):
                r = m.iloc[0]
                rows.append((r.label, f"signer {r.signer} ({r.split})", p, r.n_frames))
            else:
                rows.append(("?", "δεν υπάρχει στο metadata", p, 0))
        make_montage(rows, a.root, os.path.join(a.out, "custom.png"), "Custom σύγκριση", a.cols)
        return

    names = [a.group] if a.group else list(GROUPS)
    for name in names:
        rows = pick_clips(meta, GROUPS[name], a.signers)
        make_montage(rows, a.root, os.path.join(a.out, f"cmp_{name}.png"),
                     f"{name}: {' vs '.join(GROUPS[name])}", a.cols)


if __name__ == "__main__":
    main()