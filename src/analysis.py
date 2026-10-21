"""Explore penguin morphology and evaluate species classification.
Input: data/penguins.csv. Outputs: figures/*.png and results/*.
Run from any directory: python path/to/src/analysis.py.
"""
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
COLUMNS = ["species", "island", *FEATURES, "sex", "year"]
LABELS = ["Bill length (mm)", "Bill depth (mm)", "Flipper length (mm)", "Body mass (g)"]


def main():
    start = time.perf_counter()
    np.random.seed(RANDOM_STATE)
    for folder in ["figures", "results"]:
        (ROOT / folder).mkdir(exist_ok=True)
    print("Reading and validating local data...")
    raw = pd.read_csv(ROOT / "data/penguins.csv")
    if list(raw.columns) != COLUMNS:
        raise ValueError(f"Unexpected columns: {list(raw.columns)}")
    missing = raw.isna().sum()
    data = raw.copy()
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    clean = data.dropna(subset=["species", *FEATURES]).copy()
    if set(clean.species) != set(SPECIES):
        raise ValueError("Unexpected species labels")
    print(f"Raw rows: {len(raw)}; removed: {len(raw)-len(clean)}; modeling rows: {len(clean)}")
    print("Missing counts:\n" + missing.to_string())
    print("Raw species counts:\n" + raw.species.value_counts().reindex(SPECIES).to_string())
    print("Writing descriptive statistics...")
    summary = clean.groupby("species")[FEATURES].agg(["mean", "median", "std"]).reindex(SPECIES)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.insert(0, "n", clean.groupby("species").size().reindex(SPECIES))
    summary.to_csv(ROOT / "results/summary_by_species.csv", float_format="%.3f")
    rows = []
    for x, y in [(FEATURES[0], FEATURES[2]), (FEATURES[2], FEATURES[3])]:
        for group, frame in [("overall", data), *[(s, data[data.species == s]) for s in SPECIES]]:
            pair = frame[[x, y]].dropna()
            rows.append(dict(group=group, x=x, y=y, n=len(pair), pearson_r=pair[x].corr(pair[y])))
    pd.DataFrame(rows).to_csv(ROOT / "results/correlations.csv", index=False, float_format="%.3f")
    # Descriptive separation: proportion of total variance between species.
    eta = {}
    for feature in FEATURES:
        grand = clean[feature].mean()
        between = sum(len(g) * (g[feature].mean()-grand)**2 for _, g in clean.groupby("species"))
        eta[feature] = between / ((clean[feature]-grand)**2).sum()
    sns.set_theme(style="whitegrid", context="notebook")
    palette = dict(zip(SPECIES, sns.color_palette("colorblind", 3)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, feature, label in zip(axes.flat, FEATURES, LABELS):
        sns.boxplot(data=clean, x="species", y=feature, order=SPECIES, hue="species",
                    palette=palette, ax=ax, showfliers=False)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        sns.stripplot(data=clean, x="species", y=feature, order=SPECIES,
                      color="black", alpha=.35, size=2.5, jitter=.18, ax=ax)
        ax.set(title=label.split(" (")[0], xlabel="Species", ylabel=label)
        ax.set_xticks(range(3), [f"{s}\n(n={int(summary.loc[s, 'n'])})" for s in SPECIES])
    fig.suptitle("Penguin measurements by species (complete morphology records)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/01_measurements_by_species.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    for species in SPECIES:
        group = clean[clean.species == species]
        sns.regplot(data=group, x=FEATURES[0], y=FEATURES[2], ci=None,
                    color=palette[species], label=species, ax=ax,
                    scatter_kws={"alpha": .65, "s": 28})
    ax.set(xlabel=LABELS[0], ylabel=LABELS[2], title="Bill and flipper length: species-specific linear fits")
    ax.legend(title="Species")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/02_bill_vs_flipper.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Fitting training-only pipeline and stratified cross-validation...")
    x_train, x_test, y_train, y_test = train_test_split(
        clean[FEATURES], clean.species, test_size=.25, stratify=clean.species, random_state=RANDOM_STATE)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    cv = cross_val_score(model, x_train, y_train, scoring="accuracy",
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    matrix = confusion_matrix(y_test, predicted, labels=SPECIES)
    report = classification_report(y_test, predicted, labels=SPECIES, digits=3, zero_division=0)
    lines = [
        f"Raw rows: {len(raw)}; columns: {len(raw.columns)}",
        "Original missing counts:\n" + missing.to_string(),
        "Raw species counts:\n" + raw.species.value_counts().reindex(SPECIES).to_string(),
        f"Removed rows: {len(raw)-len(clean)}; modeling rows: {len(clean)}",
        "Features: " + ", ".join(FEATURES),
        "Split: stratified 75/25 holdout; random_state=42",
        f"Training rows: {len(y_train)}; test rows: {len(y_test)}",
        "Test species counts:\n" + y_test.value_counts().reindex(SPECIES).to_string(),
        "Pipeline: StandardScaler + LogisticRegression(max_iter=2000, random_state=42)",
        f"Test accuracy: {accuracy_score(y_test, predicted):.3f}",
        f"Test macro F1: {f1_score(y_test, predicted, average='macro'):.3f}",
        "Training-only CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "CV accuracy folds: " + ", ".join(f"{v:.3f}" for v in cv),
        f"CV accuracy mean +/- SD (population, ddof=0): {cv.mean():.3f} +/- {cv.std():.3f}",
        "Confusion matrix (rows=true, columns=predicted; Adelie, Chinstrap, Gentoo):\n" + str(matrix),
        "Classification report:\n" + report,
        "Descriptive eta-squared (between-species / total variance; not model feature importance):",
        *[f"{k}: {v:.3f}" for k, v in sorted(eta.items(), key=lambda item: -item[1])],
        f"Software: Python {platform.python_version()}, pandas {pd.__version__}, numpy {np.__version__}, "
        f"matplotlib {matplotlib.__version__}, seaborn {sns.__version__}, scikit-learn {sklearn.__version__}",
    ]
    (ROOT / "results/metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=SPECIES, yticklabels=SPECIES, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="True label", title="Species classification: held-out test set")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/03_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(lines))
    print(f"Completed in {time.perf_counter()-start:.3f} seconds. Output files:")
    for folder in ["figures", "results"]:
        for path in sorted((ROOT / folder).iterdir()):
            print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
