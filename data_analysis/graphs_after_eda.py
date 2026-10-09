#claude code

"""EDA for the GSL isolated dataset.

Usage:
    python3 data_analysis/graphs_after_eda.py --csv_dir data --root data/GSL_isol --out graphs
Without --root only the CSV-based plots (part A) are produced.
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.family"] = "DejaVu Sans"  # supports Greek gloss names


# ---------------------------------------------------------------- loading
def load(csv_dir, split):
    d = pd.read_csv(Path(csv_dir) / f"{split}_greek_iso_merged.csv", sep="|",
                    header=None, names=["path", "label"])
    p = d.path.str.extract(
        r"(?P<theme>[a-z]+)(?P<tid>\d+)_signer(?P<signer>\d+)_rep(?P<rep>\d+)"
        r"_glosses/glosses(?P<g>\d+)")
    d = pd.concat([d, p], axis=1)
    d["split"] = split
    for c in ["tid", "signer", "rep", "g"]:
        d[c] = d[c].astype(int)
    return d


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- PART A: CSV only
def plot_gloss_freq(df, out):
    """01. Number of clips per gloss (train), log scale."""
    vc = df[df.split == "train"].label.value_counts()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(vc)), vc.values)
    ax.set_yscale("log")
    for t in (10, 30):
        ax.axhline(t, color="red", ls="--", lw=1)
        ax.text(len(vc) * 0.98, t * 1.1, f"{t} clips", ha="right", color="red")
    ax.set(xlabel="glosses (sorted by frequency)", ylabel="clips per gloss (log scale)",
           title=f"Gloss frequency in the train set ({len(vc)} glosses)")
    save(fig, out, "01_gloss_freq")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    vc.head(20)[::-1].plot.barh(ax=axes[0])
    axes[0].set_title("20 most frequent glosses (train)")
    vc.tail(20)[::-1].plot.barh(ax=axes[1], color="C1")
    axes[1].set_title("20 rarest glosses (train)")
    for ax in axes:
        ax.set_xlabel("clips")
        ax.set_ylabel("")
    save(fig, out, "01b_top_bottom_glosses")


def plot_split_distribution(df, out, n_top=20):
    """02. Is the gloss distribution the same in train, val and test?
    Left: share of each split taken by the 20 most frequent glosses.
    Right: share in train vs share in test for every gloss (diagonal = identical)."""
    cnt = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    share = cnt / cnt.sum() * 100                      # % of each split
    top = cnt["train"].sort_values(ascending=False).head(n_top).index

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.2, 1]})

    ax = axes[0]
    y = np.arange(n_top)
    h = 0.27
    for i, s in enumerate(["train", "val", "test"]):
        ax.barh(y + (1 - i) * h, share.loc[top, s], height=h, label=s)
    ax.set_yticks(y)
    ax.set_yticklabels(top, fontsize=11)
    ax.invert_yaxis()
    ax.set(xlabel="% of the split's clips",
           title=f"Share of the {n_top} most frequent glosses in each split")
    ax.legend()

    ax = axes[1]
    ax.scatter(share["train"], share["test"], alpha=0.6, s=25)
    m = max(share["train"].max(), share["test"].max()) * 1.05
    ax.plot([0, m], [0, m], color="red", ls="--", lw=1, label="same share")
    big = share[share["train"] > 0.3]            # skip tiny glosses: labels would overlap
    diff = (big["test"] - big["train"]).abs().sort_values(ascending=False).head(3).index
    for lab in diff:
        ax.annotate(lab, (share.loc[lab, "train"], share.loc[lab, "test"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=11)
    ax.set(xlabel="% of train clips", ylabel="% of test clips",
           title="Each point = one gloss (labelled: largest differences)")
    ax.legend()
    save(fig, out, "02_split_distribution")

    missing = cnt[(cnt["train"] < 10) & ((cnt["val"] > 0) | (cnt["test"] > 0))]
    print("\nGlosses with <10 train clips that appear in val/test:")
    print(missing if len(missing) else "none")


def plot_heatmaps(df, out):
    """03. Signer x split / theme / repetition."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["split", "theme", "rep"]):
        sns.heatmap(pd.crosstab(df.signer, df[col]), annot=True, fmt="d",
                    cmap="Blues", ax=ax)
        ax.set_title(f"signer x {col} (number of clips)")
    save(fig, out, "03_signer_heatmaps")


def plot_signer_per_gloss(df, out):
    """04. Clips per signer for the 20 most frequent / rarest glosses (train)."""
    tr = df[df.split == "train"]
    vc = tr.label.value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, labs, title in [(axes[0], vc.head(20).index, "20 most frequent glosses"),
                            (axes[1], vc.tail(20).index, "20 rarest glosses")]:
        ct = pd.crosstab(tr[tr.label.isin(labs)].label, tr.signer).loc[labs]
        sns.heatmap(ct, annot=True, fmt="d", cmap="viridis", ax=ax)
        ax.set_title(f"{title}: clips per signer")
        ax.set_ylabel("")
    save(fig, out, "04_signer_per_gloss")


def plot_rep_per_signer(df, out):
    """05. Repetitions per signer."""
    ct = pd.crosstab(df.signer, df.rep)
    fig, ax = plt.subplots(figsize=(8, 4))
    ct.plot.bar(stacked=True, ax=ax)
    ax.set(ylabel="clips", title="Clips per signer and repetition")
    ax.legend(title="repetition")
    save(fig, out, "05_rep_per_signer")

    odd = df[df.rep == 6]
    print("\nClips with repetition 6:")
    print(odd.groupby(["split", "signer", "theme", "tid"]).size())


def plot_position(df, out):
    """06. Position of a gloss inside its sentence."""
    df = df.copy()
    df["video"] = df.path.str.split("/").str[0]
    df["sent_len"] = df.groupby("video").g.transform("max") + 1
    df["pos_norm"] = df.g / (df.sent_len - 1).clip(lower=1)

    top = df.label.value_counts().head(15).index
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df[df.label.isin(top)], x="label", y="pos_norm", order=top, ax=ax)
    ax.set(ylabel="normalized position (0 = start, 1 = end of sentence)", xlabel="",
           title="Where the 15 most frequent glosses appear inside a sentence")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, out, "06_position_in_sentence")

    spread = df.groupby("label").pos_norm.agg(["mean", "std", "size"])
    print("\nGlosses that almost always appear at the same position (std<0.05, n>=30):")
    print(spread[(spread["std"] < 0.05) & (spread["size"] >= 30)]
          .sort_values("size", ascending=False).head(15))


# ---------------------------------------------------------------- PART B: images
def frame_files(root, rel):
    return sorted((Path(root) / rel).glob("frame_*.jpg"))


def image_stats(df, root, out, n_sample=300, seed=0):
    """Frames per clip, resolution, brightness/contrast per signer."""
    from PIL import Image

    rng = random.Random(seed)
    df = df.copy()

    # --- number of frames (fast: directory listing only)
    counts, missing = [], []
    for rel in df.path:
        fs = frame_files(root, rel)
        counts.append(len(fs))
        if not fs:
            missing.append(rel)
    df["n_frames"] = counts
    print(f"\nEmpty / missing folders: {len(missing)}")
    print(df.n_frames.describe())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) histogram with median / 95th percentile in the legend (always readable)
    ax = axes[0]
    sns.histplot(df.n_frames, bins=50, ax=ax, color="gray")
    for q, c in [(0.5, "tab:blue"), (0.95, "tab:red")]:
        v = df.n_frames.quantile(q)
        name = "median" if q == 0.5 else "95th percentile"
        ax.axvline(v, color=c, ls="--", lw=2, label=f"{name} = {v:.0f} frames")
    ax.set(title="(a) Clip length: all clips", xlabel="frames per clip",
           ylabel="number of clips")
    ax.legend(fontsize=11)

    # (b) one box per signer, labelled with its split (signer 3 appears as val and test)
    df["signer_split"] = df.signer.astype(str) + "\n(" + df.split + ")"
    order = (df[["signer", "split", "signer_split"]].drop_duplicates()
             .sort_values(["split", "signer"], key=lambda c: c.map({"train": 0, "val": 1, "test": 2}) if c.name == "split" else c)
             .signer_split.tolist())
    sns.boxplot(data=df, x="signer_split", y="n_frames", hue="split", order=order,
                dodge=False, legend=False, ax=axes[1])
    axes[1].set(title="(b) Clip length per signer", xlabel="signer (split)",
                ylabel="frames per clip")

    # (c) per theme
    sns.boxplot(data=df, x="theme", y="n_frames", ax=axes[2])
    axes[2].set(title="(c) Clip length per theme", xlabel="theme",
                ylabel="frames per clip")
    save(fig, out, "07_n_frames")

    short = df.nsmallest(10, "n_frames")[["path", "label", "n_frames"]]
    long_ = df.nlargest(10, "n_frames")[["path", "label", "n_frames"]]
    print("\nShortest clips:\n", short, "\nLongest clips:\n", long_)

    # --- resolution + brightness on a sample (1 middle frame per clip)
    rows = []
    valid = df[df.n_frames > 0]
    sample = pd.concat([g.sample(min(n_sample, len(g)), random_state=seed)
                        for _, g in valid.groupby("signer")])
    for _, r in sample.iterrows():
        fs = frame_files(root, r.path)
        f = fs[len(fs) // 2]
        im = Image.open(f)
        gray = np.asarray(im.convert("L"), dtype=np.float32)
        rows.append({"signer": r.signer, "split": r.split, "w": im.size[0],
                     "h": im.size[1], "brightness": gray.mean(),
                     "contrast": gray.std(), "file_kb": f.stat().st_size / 1024})
    st = pd.DataFrame(rows)
    print("\nResolutions found:\n", st.groupby(["w", "h"]).size())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, ["brightness", "contrast", "file_kb"]):
        sns.boxplot(data=st, x="signer", y=col, ax=ax)
        ax.set_title(f"{col} per signer")
    save(fig, out, "08_image_properties")
    return df


def montage(df, root, out, n_frames=8, n_glosses=4, seed=0):
    """Montage: 8 evenly spaced frames of one gloss, one row per signer."""
    from PIL import Image

    rng = random.Random(seed)
    tr = df[df.split == "train"]
    common = tr.label.value_counts().index[10:60]  # skip the most extreme glosses
    for lab in rng.sample(list(common), n_glosses):
        sub = tr[tr.label == lab].groupby("signer").head(1)
        fig, axes = plt.subplots(len(sub), n_frames, figsize=(2 * n_frames, 1.6 * len(sub)))
        axes = np.atleast_2d(axes)
        for i, (_, r) in enumerate(sub.iterrows()):
            fs = frame_files(root, r.path)
            idx = np.linspace(0, len(fs) - 1, n_frames).astype(int)
            for j, k in enumerate(idx):
                axes[i, j].imshow(Image.open(fs[k]))
                axes[i, j].axis("off")
            axes[i, 0].set_title(f"signer {r.signer}", fontsize=9, loc="left")
        fig.suptitle(f"Gloss: {lab} (8 evenly spaced frames, one row per signer)")
        save(fig, out, f"09_montage_{lab}")


def mean_image_per_signer(df, root, out, n_per_signer=100, size=(160, 120), seed=0):
    """Mean and std image per signer: framing, background and where the signer stands."""
    from PIL import Image

    signers = sorted(df.signer.unique())
    fig, axes = plt.subplots(2, len(signers), figsize=(2.4 * len(signers), 4))
    for c, s in enumerate(signers):
        sub = df[df.signer == s].sample(min(n_per_signer, (df.signer == s).sum()),
                                        random_state=seed)
        imgs = []
        for rel in sub.path:
            fs = frame_files(root, rel)
            if fs:
                imgs.append(np.asarray(Image.open(fs[len(fs) // 2]).convert("RGB")
                                       .resize(size), dtype=np.float32) / 255)
        arr = np.stack(imgs)
        axes[0, c].imshow(arr.mean(0)); axes[0, c].set_title(f"signer {s}")
        axes[1, c].imshow(arr.std(0).mean(-1), cmap="magma")
        axes[0, c].axis("off"); axes[1, c].axis("off")
    axes[0, 0].text(-0.1, 0.5, "mean image", transform=axes[0, 0].transAxes, rotation=90, va="center")
    axes[1, 0].text(-0.1, 0.5, "std image", transform=axes[1, 0].transAxes, rotation=90, va="center")
    save(fig, out, "10_mean_std_per_signer")


def motion_energy(df, root, out, n_sample=150, seed=0):
    """Mean absolute difference between consecutive frames over normalized time."""
    from PIL import Image

    sub = df[df.split == "train"].sample(n_sample, random_state=seed)
    curves = []
    for rel in sub.path:
        fs = frame_files(root, rel)
        if len(fs) < 3:
            continue
        g = [np.asarray(Image.open(f).convert("L").resize((128, 96)), dtype=np.float32)
             for f in fs]
        e = np.array([np.abs(g[i + 1] - g[i]).mean() for i in range(len(g) - 1)])
        t = np.linspace(0, 1, len(e))
        curves.append(np.interp(np.linspace(0, 1, 50), t, e))
    curves = np.array(curves)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.linspace(0, 1, 50), curves.T, color="gray", alpha=0.1)
    ax.plot(np.linspace(0, 1, 50), curves.mean(0), color="red", lw=2, label="mean over clips")
    ax.set(xlabel="normalized time (0 = first frame, 1 = last frame)",
           ylabel="mean frame difference", title="Motion energy per clip")
    ax.legend()
    save(fig, out, "11_motion_energy")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=".")
    ap.add_argument("--root", default=None, help="e.g. data/GSL_isol")
    ap.add_argument("--out", default="eda_out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df = pd.concat([load(args.csv_dir, s) for s in ["train", "val", "test"]],
                   ignore_index=True)
    print(df.groupby("split").agg(clips=("path", "size"), glosses=("label", "nunique")))

    # Part A: CSV only
    plot_gloss_freq(df, out)
    plot_split_distribution(df, out)
    plot_heatmaps(df, out)
    plot_signer_per_gloss(df, out)
    plot_rep_per_signer(df, out)
    plot_position(df, out)

    # Part B: images
    if args.root:
        df = image_stats(df, args.root, out)
        df.to_csv(out / "metadata.csv", index=False)
        montage(df, args.root, out)
        mean_image_per_signer(df, args.root, out)
        motion_energy(df, args.root, out)