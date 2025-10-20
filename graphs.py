import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

# Carica i risultati
results_df = pd.read_csv("risultati_classificatori.csv")

# Etichetta combinata k,l oppure testo originale
if "k" in results_df.columns and "l" in results_df.columns:
    results_df["Anonimizzazione_label"] = results_df.apply(
        lambda row: (
            f"Non anonimizzato" if row["Anonimizzato"] == "non anonimizzato"
            else f"Anon. (k={int(row['k'])}, l={int(row['l'])})"
        ),
        axis=1
    )
else:
    results_df["Anonimizzazione_label"] = results_df["Anonimizzato"]

# Cartella per i grafici
os.makedirs("grafici", exist_ok=True)

# Colormap
cmap = plt.colormaps.get_cmap("tab20")

for dataset in results_df["Dataset"].unique():
    subset = results_df[results_df["Dataset"] == dataset]

    modelli = subset["Modello"].unique()

    # Trova combinazione k,l massima per questo dataset
    if "k" in subset.columns and "l" in subset.columns:
        max_k = subset["k"].max()
        max_l = subset["l"].max()
        label_max_anon = f"Anon. (k={max_k}, l={max_l})"
    else:
        label_max_anon = None

    # Grafico principale
    if label_max_anon and label_max_anon in subset["Anonimizzazione_label"].values:
        labels_anon = ["Non anonimizzato", label_max_anon]
        color_map = {labels_anon[0]: cmap(0), labels_anon[1]: cmap(1)}
        x = np.arange(len(modelli))
        width = 0.35  # larghezza barre
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, label in enumerate(labels_anon):
            acc_vals = [
                subset[
                    (subset["Modello"] == modello) &
                    (subset["Anonimizzazione_label"] == label)
                ]["Accuracy"].mean()
                for modello in modelli
            ]
            bars = ax.bar(
                x + (i - len(labels_anon) / 2) * width + width / 2,
                acc_vals,
                width,
                label=label,
                color=color_map[label])
            for bar, val in zip(bars, acc_vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Modello")
        ax.set_title(f"Accuracy: non anonimizzato vs anonimizzazione massima - Dataset: {dataset}", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(modelli)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.linspace(0, 1.0, 6))
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f"grafici/accuracy_{dataset}_solo_estremi.png", bbox_inches="tight")
    # Grafico principale con Accuracy e F1-score
    if label_max_anon and label_max_anon in subset["Anonimizzazione_label"].values:
        metrics = ["Accuracy", "F1 Score"]  # due metriche da plottare
        labels_anon = ["Non anonimizzato", label_max_anon]
        color_map = {
            ("Accuracy", "Non anonimizzato"): "#1f77b4",  # blu scuro
            ("Accuracy", label_max_anon): "#aec7e8",  # blu chiaro
            ("F1 Score", "Non anonimizzato"): "#ff7f0e",  # arancione scuro
            ("F1 Score", label_max_anon): "#ffbb78"  # arancione chiaro
        }
        x = np.arange(len(modelli))
        width = 0.2  # larghezza barre
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, metric in enumerate(metrics):
            for j, label in enumerate(labels_anon):
                vals = [
                    subset[
                        (subset["Modello"] == modello) &
                        (subset["Anonimizzazione_label"] == label)
                        ][metric].mean()
                    for modello in modelli
                ]
                offset = (i * len(labels_anon) + j - 1.5) * width
                bars = ax.bar(
                    x + offset,
                    vals,
                    width,
                    label=f"{metric} - {label}",
                    color=color_map[(metric, label)])
                for bar, val in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8)
        ax.set_ylabel("Valore")
        ax.set_xlabel("Modello")
        ax.set_title(f"Accuracy e F1-score: non anonimizzato vs anonimizzazione massima - Dataset: {dataset}", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(modelli)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f"grafici/accuracy_f1_{dataset}_solo_estremi.png", bbox_inches="tight")
    # Grafico differenze
    if label_max_anon and label_max_anon in subset["Anonimizzazione_label"].values:
        diff_vals = []
        colors = []
        for modello in modelli:
            acc_non_anon = subset[
                (subset["Modello"] == modello) &
                (subset["Anonimizzazione_label"] == "Non anonimizzato")
            ]["Accuracy"].mean()
            acc_max_anon = subset[
                (subset["Modello"] == modello) &
                (subset["Anonimizzazione_label"] == label_max_anon)
            ]["Accuracy"].mean()
            diff = (acc_max_anon - acc_non_anon) * 100  # differenza in %
            diff_vals.append(diff)
            colors.append("green" if diff >= 0 else "red")
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(modelli, diff_vals, color=colors)
        ax.set_ylabel("Differenza di Accuracy (%)")
        ax.set_xlabel("Modello")
        ax.set_title(f"Impatto dell'anonimizzazione massima - Dataset: {dataset}")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylim(min(-5, min(diff_vals) - 1), max(5, max(diff_vals) + 1))
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        for bar, val in zip(bars, diff_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.3 if val >= 0 else -0.5),
                f"{val:+.2f}%",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=10,
                color="green" if val >= 0 else "red",
                fontweight="bold"
            )
        legend_elements = [
            Patch(facecolor='green', label='Incremento con anonimizzazione'),
            Patch(facecolor='red', label='Diminuzione con anonimizzazione')
        ]
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        plt.tight_layout()
        plt.savefig(f"grafici/differenze_{dataset}_max_vs_orig.png", bbox_inches="tight")
k_values = sorted(results_df["k"].unique())
l_values = [2, 3]
models = ["RF", "SVM", "KNN", "XGB"]
colors = {"RF": "tab:blue", "SVM": "tab:orange", "KNN": "tab:green", "XGB": "tab:red"}
for dataset in results_df["Dataset"].unique():
    subset = results_df[results_df["Dataset"] == dataset]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, l in zip(axes, l_values):
        for model in models:
            acc_vals = []
            for k in k_values:
                acc = subset[
                    (subset["Modello"] == model) &
                    (subset["k"] == k) &
                    (subset["l"] == l)
                    ]["Accuracy"].mean()
                acc_vals.append(acc)
            ax.plot(k_values, acc_vals, marker='o', label=model, color=colors[model])
        ax.set_title(f"Dataset: {dataset}, l={l}")
        ax.set_xlabel("k")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
    plt.suptitle(f"Accuratezza dei modelli al variare di k - Dataset: {dataset}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"grafici/lineplot_accuracy_{dataset}.png", bbox_inches="tight")