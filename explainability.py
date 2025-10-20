import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import pandas as pd
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

def shap_explainability(m, test_feat, k, dataset_name, mod_name, anonimizzato, ldiv, make_beeswarm=True):
    # Codifica le colonne categoriche
    for col in test_feat.columns:
        if test_feat[col].dtype == object or str(test_feat[col].dtype) == 'category':
            test_feat[col] = test_feat[col].astype(str).astype('category').cat.codes
    #explainer = shap.Explainer(m.predict, test_feat)
    explainer = shap.Explainer(m.predict_proba, test_feat)
    shap_values = explainer(test_feat)
    anon_str = "anon" if anonimizzato else "originale"
    save_dir = os.path.join("explanations", "shap", dataset_name, anon_str, mod_name)
    os.makedirs(save_dir, exist_ok=True)
    if dataset_name.lower() == "mental_health":
        print(f"Dataset multiclasse rilevato: {dataset_name}")
        print(f"Classi del modello: {m.classes_}")
        # IMPORTANZA GLOBALE
        if isinstance(shap_values.values, list):
            global_importance = np.mean(
                [np.abs(shap_values.values[i]).mean(axis=0) for i in range(len(m.classes_))],
                axis=0
            )
            shap_values_mean = np.mean(
                [np.abs(shap_values.values[i]) for i in range(len(m.classes_))],
                axis=0
            )
        elif len(shap_values.values.shape) == 3:
            # (n_samples, n_features, n_classes)
            global_importance = np.abs(shap_values.values).mean(axis=(0, 2))
            shap_values_mean = np.abs(shap_values.values).mean(axis=2)  # media tra le classi
        else:
            raise ValueError("Formato SHAP values non supportato per multiclasse.")
        shap_global = pd.DataFrame({
            "feature": test_feat.columns,
            "importance": global_importance
        }).sort_values("importance", ascending=False)
        csv_global_filename = f"shap_importance_global_k{k}_l{ldiv}.csv"
        shap_global.to_csv(os.path.join(save_dir, csv_global_filename), index=False)
        if make_beeswarm:
            shap_explanation_global = shap.Explanation(
                values=shap_values_mean,
                data=test_feat.values,
                feature_names=test_feat.columns.tolist()
            )
            shap.plots.beeswarm(shap_explanation_global, show=False, max_display=20)
            plt.title("SHAP Beeswarm - Importanza Globale (tutte le classi)")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"shap_beeswarm_global_k{k}_l{ldiv}.png"))
            plt.close()
    else:
        values = shap_values.values
        if len(values.shape) == 3 and values.shape[2] == 2:
            # prendo solo la classe positiva (colonna 1)
            class_idx = 1
            values_bin = values[:, :, class_idx]
        elif len(values.shape) == 2:
            values_bin = values
        else:
            raise ValueError("Formato SHAP values inatteso per binario.")
        shap_abs_mean = pd.DataFrame({
            "feature": test_feat.columns,
            "importance": np.abs(values_bin).mean(axis=0)
        }).sort_values("importance", ascending=False)
        csv_filename = f"shap_importance_k{k}_l{ldiv}.csv"
        shap_abs_mean.to_csv(os.path.join(save_dir, csv_filename), index=False)
        if make_beeswarm:
            shap_explanation_bin = shap.Explanation(
                values=values_bin,
                data=test_feat.values,
                feature_names=test_feat.columns.tolist()
            )
            shap.plots.beeswarm(shap_explanation_bin, show=False, max_display=20)
            plt.title("SHAP Beeswarm - Classe positiva")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"shap_beeswarm_k{k}_l{ldiv}.png"))
            plt.close()


def plot_feature_rank_evolution(dataset_name, model_name, top_n=9):
    """
    Mostra l'evoluzione del ranking delle feature più importanti
    al variare di k e l per un dataset e modello specifici.
    """
    base_dir = os.path.join("explanations", "shap", dataset_name)
    if dataset_name == 'adult':
        top_n=12
    elif dataset_name == 'mental_health':
        top_n=14
    orig_path = os.path.join(base_dir, "originale", model_name, "shap_importance_k0_l0.csv")
    if not os.path.exists(orig_path):
        print(f"File originale non trovato: {orig_path}")
        print("Contenuto cartella:", os.listdir(os.path.join(base_dir, "originale", model_name)))
        return
    df_orig = pd.read_csv(orig_path)
    df_orig["rank"] = df_orig["importance"].rank(ascending=False)
    top_features = df_orig.sort_values("rank").head(top_n)["feature"].tolist()
    # Contenitore per i dati
    rank_data = {f: [] for f in top_features}
    kl_labels = []
    # Estrazione k,l numerici
    anon_dir = os.path.join(base_dir, "anon", model_name)
    files = []
    for file in os.listdir(anon_dir):
        if file.startswith("shap_importance_k") and file.endswith(".csv"):
            parts = file.replace("shap_importance_k", "").replace(".csv", "").split("_l")
            k_val, l_val = int(parts[0]), int(parts[1])
            files.append((k_val, l_val, file))
    # Ordina prima per k, poi per l
    files = sorted(files, key=lambda x: (x[0], x[1]))
    for k_val, l_val, file in files:
        kl_label = f"k={k_val},l={l_val}"
        kl_labels.append(kl_label)
        # Leggo il file
        df = pd.read_csv(os.path.join(anon_dir, file))
        df["rank"] = df["importance"].rank(ascending=False)
        for f in top_features:
            if f in df["feature"].values:
                rank_val = df.loc[df["feature"] == f, "rank"].values[0]
            else:
                rank_val = len(df) + 1  # se la feature sparisce la metto in fondo
            rank_data[f].append(rank_val)
    # Aggiungo l'originale come prima colonna
    for f in top_features:
        orig_rank = df_orig.loc[df_orig["feature"] == f, "rank"].values[0]
        rank_data[f] = [orig_rank] + rank_data[f]
    kl_labels = ["Originale"] + kl_labels
    # Plot
    plt.figure(figsize=(12, 6))
    for f in top_features:
        plt.plot(kl_labels, rank_data[f], marker="o", label=f)
    plt.gca().invert_yaxis()  # Rank 1 in alto
    plt.xlabel("Livello di anonimizzazione")
    plt.ylabel("Rank (1 = più importante)")
    plt.title(f"Evoluzione ranking feature - {dataset_name} ({model_name})")
    plt.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(base_dir, f"rank_evolution_{model_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Grafico salvato in {save_path}")


def plot_quasi_identifier_rank_evolution(dataset_name, model_name):
    """
    Mostra l'evoluzione del ranking delle feature che sono quasi-identificatori
    al variare di k e l, per un dataset e modello specifici.
    Qui il ranking viene calcolato solo tra i QI, ignorando le altre feature.
    """

    # Definizione quasi-identificatori per dataset
    if dataset_name == 'NHANES':
        feature_columns = ["RIDAGEYR", "RIAGENDR", "BMXBMI", "LBXGLU", "LBXIN"]
    elif dataset_name == 'stroke':
        feature_columns = ["gender", "age", "Residence_type", "work_type"]
    elif dataset_name == "diabetes":
        feature_columns = ['age', 'hypertension', 'HbA1c_level', 'blood_glucose_level']
    elif dataset_name == "adult":
        feature_columns = ['age', 'education', 'marital-status', 'occupation', 'race']
    elif dataset_name == "mental_health":
        #feature_columns = ['Age', 'Course', 'Gender', 'Residence_Type']
        feature_columns = ['Age', 'Course', 'Gender', 'Relationship_Status', 'Residence_Type', 'stress_anxiety_diff', 'Mental_Severity_Score']
    else:
        raise ValueError(f"Dataset {dataset_name} non supportato")
    base_dir = os.path.join("explanations", "shap", dataset_name)
    # Carica dati originali
    orig_path = os.path.join(base_dir, "originale", model_name, "shap_importance_k0_l0.csv")
    if not os.path.exists(orig_path):
        print(f"File originale non trovato: {orig_path}")
        return
    df_orig = pd.read_csv(orig_path)
    # Filtra solo QI
    df_qi_orig = df_orig[df_orig["feature"].isin(feature_columns)].copy()
    #df_qi_orig["rank_qi"] = df_qi_orig["importance"].rank(ascending=False)
    #df_qi_orig["rank_qi"] = df_qi_orig["importance"].rank(ascending=False, method="dense")
    df_qi_orig["rank_qi"] = df_qi_orig["importance"].rank(ascending=False, method="min").astype(int)
    qis = df_qi_orig["feature"].tolist()
    if not qis:
        print(f"Nessun quasi-identificatore trovato in {dataset_name} - {model_name}")
        return
    rank_data = {f: [] for f in qis}
    kl_labels = []
    # Scandisci cartella anonimizzati ed estrai k,l
    anon_dir = os.path.join(base_dir, "anon", model_name)
    files = []
    for file in os.listdir(anon_dir):
        if file.startswith("shap_importance_k") and file.endswith(".csv"):
            parts = file.replace("shap_importance_k", "").replace(".csv", "").split("_l")
            k_val, l_val = int(parts[0]), int(parts[1])
            files.append((k_val, l_val, file))
    files = sorted(files, key=lambda x: (x[0], x[1]))
    for k_val, l_val, file in files:
        kl_label = f"k={k_val},l={l_val}"
        kl_labels.append(kl_label)
        df = pd.read_csv(os.path.join(anon_dir, file))
        df_qi = df[df["feature"].isin(feature_columns)].copy()
        #df_qi["rank_qi"] = df_qi["importance"].rank(ascending=False)
        #df_qi["rank_qi"] = df_qi["importance"].rank(ascending=False, method="dense")
        df_qi["rank_qi"] = df_qi["importance"].rank(ascending=False, method="min").astype(int)
        for f in qis:
            if f in df_qi["feature"].values:
                rank_val = df_qi.loc[df_qi["feature"] == f, "rank_qi"].values[0]
            else:
                rank_val = len(qis) + 1
            rank_data[f].append(rank_val)
    # Aggiungi l'originale
    for f in qis:
        orig_rank = df_qi_orig.loc[df_qi_orig["feature"] == f, "rank_qi"].values[0]
        rank_data[f] = [orig_rank] + rank_data[f]
    kl_labels = ["Originale"] + kl_labels
    # Plot
    plt.figure(figsize=(12, 6))
    for f in qis:
        plt.plot(kl_labels, rank_data[f], marker="o", label=f)
    plt.gca().invert_yaxis()
    plt.xlabel("Livello di anonimizzazione")
    plt.ylabel("Rank tra i quasi-identificatori (1 = più importante)")
    plt.title(f"Evoluzione ranking quasi-identificatori - {dataset_name} ({model_name})")
    plt.yticks(range(1, len(qis) + 1))
    plt.legend(title="Quasi-identificatori", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(base_dir, f"rank_evolution_QI_{model_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Grafico salvato in {save_path}")


def lime_explainability(dataframe, dataset_name, test_feat, m, is_anon, model_name, k, l, target):

    feature_cols = [c for c in dataframe.columns if c != target]
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=dataframe[feature_cols].to_numpy(),
        feature_names=feature_cols,
        mode="classification"
    )
    # Predizioni sul test set
    y_pred = m.predict(test_feat)
    selected_idx = []
    for c in np.unique(y_pred):
        idx_c = np.where(y_pred == c)[0][:5]
        selected_idx.extend(idx_c)
    # Directory per il salvataggio includendo k e l
    anon_str = "anon" if is_anon else "originale"
    save_dir = os.path.join("explanations", "lime", dataset_name, anon_str, model_name)
    os.makedirs(save_dir, exist_ok=True)
    # Genera spiegazioni per ogni istanza selezionata
    for idx in selected_idx:
        explanation = explainer.explain_instance(
            test_feat.iloc[idx].to_numpy(),
            m.predict_proba,
            #num_features=len(dataframe.columns)
            num_features=len(feature_cols)
        )
        fig = explanation.as_pyplot_figure()
        plt.title(f"LIME explanation - Pred: {y_pred[idx]}")
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"lime_instance_new_{y_pred[idx]}_k{k}_l{l}_{idx}.png")
        plt.savefig(save_path)
        plt.close(fig)
    print(f"Spiegazioni LIME salvate in: {save_dir}")


def lime_explainability_change(train_feat_orig, dataset_name,
                               test_feat_orig, test_feat_anon,
                               model_orig, model_anon,
                               y_pred_orig, y_pred_anon,
                               k, l, train_feat_anon, model_name,
                               feature_cols):
    """
    Confronta spiegazioni LIME tra modello originale e anonimizzato
    per le istanze che cambiano predizione.
    Usa feature_cols per garantire coerenza con l'addestramento.
    """

    #print(f"[lime_explainability_change] Numero di feature: {len(feature_cols)}")
    #print(f"[lime_explainability_change] Feature usate: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
    # Aggiungi questi controlli prima di creare gli explainer
    explainer_orig = lime_tabular.LimeTabularExplainer(
        #training_data=test_feat_orig[feature_cols].to_numpy(),
        training_data=train_feat_orig[feature_cols].to_numpy(),
        feature_names=feature_cols,
        mode="classification"
    )
    explainer_anon = lime_tabular.LimeTabularExplainer(
        #training_data=test_feat_anon[feature_cols].to_numpy(),
        training_data=train_feat_anon[feature_cols].to_numpy(),
        feature_names=feature_cols,
        mode="classification"
    )
    df_orig = pd.DataFrame({
        "_row_id": test_feat_orig["_row_id"].values,
        "pred_orig": y_pred_orig
    })
    df_anon = pd.DataFrame({
        "_row_id": test_feat_anon["_row_id"].values,
        "pred_anon": y_pred_anon
    })

    merged = df_orig.merge(df_anon, on="_row_id", how="inner")
    changed = merged[merged["pred_orig"] != merged["pred_anon"]]
    stable = merged[merged["pred_orig"] == merged["pred_anon"]]
    out_dir = f"lime_results/{dataset_name}/{model_name}/k{k}_l{l}"
    os.makedirs(out_dir, exist_ok=True)
    if changed.empty:
        print(f"Nessuna istanza ha cambiato predizione (k={k}, l={l}).")
        return
    print(f"Trovati {len(changed)} elementi che hanno cambiato predizione.")
    out_dir = f"lime_results/{dataset_name}/{model_name}/k{k}_l{l}"
    os.makedirs(out_dir, exist_ok=True)
    for _, row in changed.iloc[:3].iterrows():
        rid = row["_row_id"]
        inst_orig = test_feat_orig.loc[test_feat_orig["_row_id"] == rid, feature_cols].to_numpy()[0]
        inst_anon = test_feat_anon.loc[test_feat_anon["_row_id"] == rid, feature_cols].to_numpy()[0]
        pos = test_feat_orig.index.get_loc(test_feat_orig.index[test_feat_orig["_row_id"] == rid][0])
        pred_o = row["pred_orig"]
        pred_a = row["pred_anon"]

        # Spiegazione originale
        exp_orig = explainer_orig.explain_instance(
            inst_orig,
            model_orig.predict_proba,
            num_features=len(feature_cols),
        )
        # DEBUG: stampa i contributi effettivi
        #print(f"[DEBUG] LIME contributions: {exp_orig.as_list()}")
        #print(f"[DEBUG] Intercept: {exp_orig.intercept}")
        #print(f"[DEBUG] Prediction proba: {model_orig.predict_proba([inst_orig])}")
        #print(model_orig.predict_proba(inst_orig.reshape(1, -1)))
        fig_orig = exp_orig.as_pyplot_figure()
        plt.title(f"LIME explanation instance {rid} Pred: {pred_o}")
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"lime_orig_{pred_o}_k{k}_l{l}_{rid}.png")
        plt.savefig(save_path)
        plt.close(fig_orig)

        # Spiegazione anonimizzata
        exp_anon = explainer_anon.explain_instance(
            inst_anon,
            model_anon.predict_proba,
            num_features=len(feature_cols)
        )
        fig_anon = exp_anon.as_pyplot_figure()
        plt.title(f"LIME explanation instance {rid} Pred: {pred_a}")
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"lime_anon_{pred_a}_k{k}_l{l}_{rid}.png")
        plt.savefig(save_path)
        plt.close(fig_anon)

        selected_idx_typical = []
        for c in np.unique(y_pred_orig):
            idx_c = np.where(y_pred_orig == c)[0][:5]  # prime 5 istanze per classe
            selected_idx_typical.extend(idx_c)

        for idx in selected_idx_typical:
            rid = test_feat_orig.iloc[idx]["_row_id"]
            inst_orig = test_feat_orig.iloc[idx][feature_cols].to_numpy()
            inst_anon = test_feat_anon.iloc[idx][feature_cols].to_numpy()
            pred_o = y_pred_orig[idx]
            pred_a = y_pred_anon[idx]

            # Originale
            exp_orig = explainer_orig.explain_instance(inst_orig, model_orig.predict_proba,
                                                       num_features=len(feature_cols), num_samples=5000)
            fig_orig = exp_orig.as_pyplot_figure()
            plt.title(f"LIME- instance {rid} - Pred: {pred_o}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"lime_typical_orig_{pred_o}_k{k}_l{l}_{rid}.png"))
            plt.close(fig_orig)

            # Anonimizzato
            exp_anon = explainer_anon.explain_instance(inst_anon, model_anon.predict_proba,
                                                       num_features=len(feature_cols), num_samples=5000)
            fig_anon = exp_anon.as_pyplot_figure()
            plt.title(f"LIME- instance {rid} - Pred: {pred_a}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"lime_typical_anon_{pred_a}_k{k}_l{l}_{rid}.png"))
            plt.close(fig_anon)

        #print(f"Salvate spiegazioni PNG per row_id={rid}")
    # Seleziono solo istanze stabili con predizione = 1
    #stable_pos = merged[(merged["pred_orig"] == 1) & (merged["pred_anon"] == 1)]

    if not stable.empty:
        row = stable.sample(1, random_state=42).iloc[0]
        rid = row["_row_id"]

        inst_orig = test_feat_orig.loc[test_feat_orig["_row_id"] == rid, feature_cols].to_numpy()[0]
        inst_anon = test_feat_anon.loc[test_feat_anon["_row_id"] == rid, feature_cols].to_numpy()[0]

        pred_o = row["pred_orig"]
        pred_a = row["pred_anon"]

        exp_orig = explainer_orig.explain_instance(
            inst_orig,
            model_orig.predict_proba,
            num_features=len(feature_cols)
        )
        fig_orig = exp_orig.as_pyplot_figure()
        plt.title(f"LIME explanation - instance {rid} - Pred: {pred_o}")
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"lime_stable1_orig_{pred_o}_k{k}_l{l}_{rid}.png")
        plt.savefig(save_path)
        plt.close(fig_orig)

        # --- Spiegazione anonimizzata
        exp_anon = explainer_anon.explain_instance(
            inst_anon,
            model_anon.predict_proba,
            num_features=len(feature_cols)
        )
        fig_anon = exp_anon.as_pyplot_figure()
        plt.title(f"LIME explanation - instance {rid} - Pred: {pred_a}")
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"lime_stable1_anon_{pred_a}_k{k}_l{l}_{rid}.png")
        plt.savefig(save_path)
        plt.close(fig_anon)
        #print(f"Salvate spiegazioni (CASO STABILE con pred=1) per row_id={rid}")
    else:
        print(" Nessuna istanza stabile trovata con predizione = 1.")


def plot_qi_rank_all_models(dataset_name, show=False):
    """
        Crea un unico grafico con 4 sottoplot (uno per modello),
        che mostrano l'evoluzione del ranking dei quasi-identificatori.
        Legenda unica con i colori dei QI (mappatura coerente su tutti i subplot).
        """
    models = ["RF", "SVM", "KNN", "XGB"]
    # Definizione QI per dataset
    if dataset_name == 'NHANES':
        feature_columns = ["RIDAGEYR", "RIAGENDR", "BMXBMI", "LBXGLU", "LBXIN"]
    elif dataset_name == 'stroke':
        feature_columns = ["gender", "age", "Residence_type", "work_type"]
    elif dataset_name == "diabetes":
        feature_columns = ['age', 'hypertension', 'HbA1c_level', 'blood_glucose_level']
    elif dataset_name == "adult":
        feature_columns = ['age', 'education', 'marital-status', 'occupation', 'race']
    elif dataset_name == "mental_health":
        feature_columns = ['Age', 'Course', 'Gender', 'Relationship_Status', 'Residence_Type',
                           'stress_anxiety_diff', 'Mental_Severity_Score']
    else:
        raise ValueError(f"Dataset {dataset_name} non supportato")
    base_dir = os.path.join("explanations", "shap", dataset_name)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    # Palette coerente
    palette = plt.get_cmap("tab10").colors
    colors = {f: palette[i % len(palette)] for i, f in enumerate(feature_columns)}
    for ax, model_name in zip(axes, models):
        orig_path = os.path.join(base_dir, "originale", model_name, "shap_importance_k0_l0.csv")
        if not os.path.exists(orig_path):
            print(f"[WARN] File originale non trovato per {model_name}: {orig_path}")
            ax.set_title(f"{model_name} (no data)")
            continue
        df_orig = pd.read_csv(orig_path)
        df_qi_orig = df_orig[df_orig["feature"].isin(feature_columns)].copy()
        df_qi_orig = df_qi_orig.sort_values("importance", ascending=False).reset_index(drop=True)
        orig_ranks = {f: (df_qi_orig.index[df_qi_orig["feature"] == f].tolist()[0] + 1)
                      if f in df_qi_orig["feature"].values else len(feature_columns) + 1
                      for f in feature_columns}
        rank_data = {f: [] for f in feature_columns}
        kl_labels = []
        anon_dir = os.path.join(base_dir, "anon", model_name)
        if os.path.exists(anon_dir):
            files = []
            for file in os.listdir(anon_dir):
                if file.startswith("shap_importance_k") and file.endswith(".csv"):
                    try:
                        parts = file.replace("shap_importance_k", "").replace(".csv", "").split("_l")
                        k_val, l_val = int(parts[0]), int(parts[1])
                        files.append((k_val, l_val, file))
                    except Exception:
                        continue
            files = sorted(files, key=lambda x: (x[0], x[1]))
            for k_val, l_val, file in files:
                kl_labels.append(f"k={k_val},l={l_val}")
                df = pd.read_csv(os.path.join(anon_dir, file))
                df_qi = df[df["feature"].isin(feature_columns)].copy()
                df_qi = df_qi.sort_values("importance", ascending=False).reset_index(drop=True)
                temp_ranks = {f: (df_qi.index[df_qi["feature"] == f].tolist()[0] + 1)
                              if f in df_qi["feature"].values else len(feature_columns) + 1
                              for f in feature_columns}
                for f in feature_columns:
                    rank_data[f].append(temp_ranks[f])

        # Aggiungo l'originale come primo valore
        for f in feature_columns:
            rank_data[f] = [orig_ranks[f]] + rank_data[f]
        kl_labels_full = ["Originale"] + kl_labels
        for f in feature_columns:
            ax.plot(kl_labels_full, rank_data[f], marker="o", label=f, color=colors[f], linewidth=1)
        ax.set_title(f"{model_name}")
        ax.set_xlabel("Anonimizzazione")
        ax.set_ylabel("Rank tra i QI")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(len(feature_columns) + 0.5, 0.5)
        ax.set_yticks(range(1, len(feature_columns) + 1))
    # Legenda unica dai handle del primo subplot valido
    handles, labels = [], []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    fig.legend(handles, labels, loc="upper center", ncol=min(len(feature_columns), 6), title="Quasi-identificatori")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_path = os.path.join(base_dir, f"rank_evolution_QI_{dataset_name}_all_models.png")
    fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Grafico salvato in {save_path}")


#data_set = ['stroke', 'NHANES', 'diabetes', 'adult', 'mental_health']
#data_set = ['stroke', 'NHANES', 'diabetes', 'adult']
data_set = ['mental_health']
model_name = ['RF', 'SVM', 'KNN', 'XGB']
for dataset in data_set:
    for model in model_name:
        plot_feature_rank_evolution(dataset, model, top_n=9)
        #plot_quasi_identifier_rank_evolution(dataset, model)
    plot_qi_rank_all_models(dataset)


