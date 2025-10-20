import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random

def preprocess_student(filename):
    df = pd.read_csv(filename)

    # Colonne da mantenere
    keep_cols = [
        "Age", "Course", "Gender", "CGPA", "Stress_Level", "Depression_Score", "Anxiety_Score",
        "Sleep_Quality", "Physical_Activity", "Diet_Quality", "Relationship_Status",
        "Counseling_Service_Use", "Extracurricular_Involvement", "Semester_Credit_Load", "Residence_Type"
    ]

    df = df[keep_cols]


    def max_mental_issue(row):
        values = {
            1: row["Stress_Level"],
            2: row["Depression_Score"],
            3: row["Anxiety_Score"]
        }
        # Se tutti ≤ 2 allora nessun disturbo
        if all(v <= 2 for v in values.values()):
            return 0  # none
        max_value = max(values.values())
        max_keys = [k for k, v in values.items() if v == max_value]
        # Scegli casualmente una delle chiavi massime
        return random.choice(max_keys)

    # Creazione della nuova colonna
    df["mental_illness"] = df.apply(max_mental_issue, axis=1)

    df['Mental_Severity_Score'] = df['Stress_Level'] + df['Depression_Score'] + df['Anxiety_Score']
    # Gestione valori mancanti in CGPA
    df["CGPA"] = df["CGPA"].fillna(df["CGPA"].median())
    # df['has_mental_illness'] = (df['mental_illness'] > 0).astype(int)
    # df["stress_depression_diff"] = (df["Stress_Level"] - df["Depression_Score"]).abs()
    df["stress_depression_diff"] = df["Stress_Level"] - df["Depression_Score"]
    df["stress_anxiety_diff"] = (df["Stress_Level"] - df["Anxiety_Score"]).abs()
    df["stress_depression_diff"] += np.random.normal(0, 0.3, size=len(df))
    df.drop(columns=["Stress_Level", "Depression_Score", "Anxiety_Score"], inplace=True)
    # Salva il nuovo dataset
    df.to_csv("dataset/students_mental_health_processed.csv", index=False)

    return df


def load_file(data_name):
    print("Data_name", data_name)
    if data_name == 'stroke':
        filename = 'dataset/healthcare-dataset-stroke-data.csv'
        target = "stroke"
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    elif data_name == "adult":
        filename = 'dataset/adult.csv'
        categorical_columns = ['gender', 'race', 'marital-status', 'education', 'native-country', 'relationship',
                               'workclass', 'occupation', 'income']
        target = 'income'
    elif data_name == 'diabetes':
        filename = 'dataset/diabetes.csv'
        categorical_columns = ['gender', 'smoking_history']
        target = 'diabetes'
    elif data_name == 'mental_health':
        filename = 'dataset/students_mental_health.csv'
        df = preprocess_student(filename)
        # df = preprocess_students_dataset(filename)
        # df = preprocess_students_complete_dataset(filename)
        categorical_columns = ['Course', 'Gender', 'Sleep_Quality', 'Physical_Activity', 'Diet_Quality',
                               'Relationship_Status', 'Counseling_Service_Use',
                               'Extracurricular_Involvement', 'Residence_Type']
        target = 'mental_illness'
    else:
        filename = 'dataset/NHANES_age_prediction.csv'
        target = "age_group"
        categorical_columns = ['age_group']
    if data_name != 'mental_health':
        df = pd.read_csv(filename)
    if data_name == 'NHANES':
        df = df.drop('SEQN', axis=1)
    return df, target, categorical_columns


def heatmap(dataset, dataset_name):
    if dataset_name == 'stroke':
        dataset = dataset.drop('bmi', axis=1)
    id_cols = [col for col in dataset.columns if 'id' in col.lower()]
    dataset = dataset.drop(columns=id_cols)
    save_dir = os.path.join(f"info_{dataset_name}")
    os.makedirs(save_dir, exist_ok=True)
    numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = dataset[numeric_cols].corr()
    # Crea la figura
    plt.figure(figsize=(8, 6))
    # Heatmap migliorata
    sns.heatmap(
        corr_matrix,
        annot=True,  # Mostra i valori numerici
        fmt=".2f",  # Limita a due decimali
        cmap="RdBu_r",  # Scala di colori rosso-blu (rosso positivo, blu negativo)
        vmin=-1, vmax=1,  # Limiti fissi tra -1 e 1
        linewidths=0.5,  # Linee tra le celle per separarle meglio
        linecolor='white',  # Colore delle linee
        cbar_kws={"shrink": 0.75, "label": "Coefficiente di correlazione"}  # Barra colori più piccola e con etichetta
    )
    # Migliora la leggibilità dei nomi delle colonne
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    #plt.title("Heatmap di correlazione tra feature numeriche", fontsize=14, pad=20)
    # Salva immagine ad alta risoluzione
    plt.tight_layout()
    #plt.savefig(f"grafico.png", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "heatmap.png"))


def feature_distribution(dataset, dataset_name):
    # Seleziona solo le colonne numeriche
    id_cols = [col for col in dataset.columns if 'id' in col.lower()]
    dataset = dataset.drop(columns=id_cols)
    save_dir = os.path.join(f"info_{dataset_name}")
    numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
    # Fai un istogramma per ogni feature numerica
    for col in numeric_cols:
        col_dir = os.path.join(save_dir, col)
        os.makedirs(col_dir, exist_ok=True)
        plt.figure(figsize=(6,4))
        sns.histplot(dataset[col], bins=30, color="steelblue", kde=False)
        plt.title(f"Distribuzione della feature: {col}")
        plt.xlabel(col)
        plt.ylabel("Frequenza")
        #plt.savefig(f"grafico_{col}.png", dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(col_dir, "feature_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()


def target_distribution(dataset,target, dataset_name, class_labels=None):
    save_dir = os.path.join(f"info_{dataset_name}")
    counts = dataset[target].value_counts().sort_index()  # Ordina per indice
    percentages = counts / counts.sum() * 100

    plt.figure(figsize=(8, 6))

    # Colori differenziati - neutro per classe maggioritaria, evidenza per minoritaria
    colors = ["steelblue", "orange", "green", "red"]

    bars = sns.barplot(x=percentages.index, y=percentages.values,
                       hue=percentages.index, palette=colors[:len(percentages)], legend=False)

    #plt.title("Distribuzione della variabile target", fontsize=14, fontweight='bold')

    # Etichette degli assi
    if class_labels:
        # Se fornite etichette personalizzate
        tick_labels = [class_labels.get(x, str(x)) for x in percentages.index]
        plt.xticks(range(len(percentages)), tick_labels)
        plt.xlabel("Classe", fontsize=12)
    else:
        plt.xlabel("Classe", fontsize=12)

    plt.ylabel("Percentuale (%)", fontsize=12)
    plt.ylim(0, 100)

    # Aggiungi griglia orizzontale per facilitare la lettura
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Annotazioni con percentuale E frequenza assoluta
    for i, (perc, count) in enumerate(zip(percentages.values, counts.values)):
        # Posiziona il testo più in basso per evitare sovrapposizioni con il titolo
        y_pos = min(perc + 3, 85)  # Limita la posizione Y per non andare sopra il titolo
        plt.text(i, y_pos, f"{perc:.1f}%\n(n={count:,})",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    #plt.savefig("target_distribution_percentage.png", dpi=300)
    plt.savefig(os.path.join(save_dir, "target_distribution.png"))
    if class_labels:
        classe_names = [class_labels.get(x, str(x)) for x in counts.index]
    else:
        classe_names = counts.index

    table = pd.DataFrame({
        'Classe': classe_names,
        'Frequenza': counts.values,
        'Percentuale (%)': percentages.round(1).values
    })
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('tight')
    ax.axis('off')
    table_plot = ax.table(cellText=table.values, colLabels=table.columns,
                          cellLoc='center', loc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(12)
    table_plot.scale(1.2, 1.5)
    #save_path = f"./figures/{target}_{class_labels}.png"
    #plt.savefig(f"_table.png", dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "target_distribution_table.png"))
    plt.close()


def dataset_info(dataset, dataset_name):
    """Informazioni di base del dataset con tipi di dato per colonna"""
    save_dir = os.path.join(f"info_{dataset_name}")
    n_rows, n_cols = dataset.shape
    numeric_cols = len(dataset.select_dtypes(include=['number']).columns)
    categorical_cols = len(dataset.select_dtypes(include=['object']).columns)
    missing_total = dataset.isnull().sum().sum()
    binary_info = []
    for col in dataset.columns:
        unique_vals = dataset[col].nunique()
        unique_values = sorted(dataset[col].dropna().unique())

        if unique_vals == 2:
            binary_info.append('Binaria')
        elif unique_vals <= 5:
            binary_info.append(f'Categorica ({unique_vals} valori)')
        else:
            binary_info.append('Continua')
    dtypes_info = pd.DataFrame({
        'Colonna': dataset.columns,
        'Tipo': dataset.dtypes.astype(str),
        'Categoria': binary_info,
        'Valori Mancanti': dataset.isnull().sum().values
    })

    # Informazioni generali
    general_info = pd.DataFrame({
        'Informazione': ['Righe', 'Colonne', 'Var. Numeriche', 'Var. Categoriche', 'Valori Mancanti Totali'],
        'Valore': [n_rows, n_cols, numeric_cols, categorical_cols, missing_total]
    })

    # Salva info generali
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    ax.set_title('Informazioni Generali Dataset')
    ax.table(cellText=general_info.values, colLabels=general_info.columns,
                 cellLoc='center', loc='center')
    #plt.savefig(f"dataset_info.png", dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "dataset_info.png"))
    plt.close()

    # Salva tipi di dato
    fig, ax = plt.subplots(figsize=(8, max(4, len(dtypes_info) * 0.25)))
    ax.axis('off')
    #ax.set_title('Tipi di Dato per Colonna')
    ax.table(cellText=dtypes_info.values, colLabels=dtypes_info.columns,
             cellLoc='center', loc='center')
    #plt.savefig(f"columns_dtypes.png", dpi=300, bbox_inches='tight')
    #table.scale(1, 1.2)
    plt.savefig(os.path.join(save_dir, "data_info.png"), dpi=300, bbox_inches="tight",pad_inches=0.1)
    plt.close()

    #print(f"Salvato: _info.png e {}_dtypes.png")


def descriptive_stats(dataset, dataset_name):
    """Statistiche descrittive semplici"""
    save_dir = os.path.join(f"info_{dataset_name}")
    numeric_data = dataset.select_dtypes(include=['number'])
    id_cols = [col for col in numeric_data.columns if 'id' in col.lower()]
    numeric_data = numeric_data.drop(columns=id_cols)
    continuous_vars = []
    for col in numeric_data.columns:
        unique_vals = numeric_data[col].nunique()
        if unique_vals > 5:  # Solo variabili con più di 5 valori unici
            continuous_vars.append(col)
    stats = numeric_data[continuous_vars].describe().round(2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.table(cellText=stats.values, rowLabels=stats.index,
             colLabels=stats.columns, cellLoc='center', loc='center')
    #plt.savefig(f"descriptive_stats.png", dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "descriptive_stats.png"))
    plt.close()

    return stats


if __name__ == '__main__':
    #dataset = ['stroke', 'NHANES', 'adult', 'diabetes', 'mental_health']
    dataset = ['NHANES']
    for dataset_name in dataset:
        dataset, target, categorical_columns = load_file(dataset_name)
        heatmap(dataset,dataset_name)
        #target_distribution(dataset,target, dataset_name, class_labels={0: 'Nessun ictus', 1: 'Ictus'})
        target_distribution(dataset, target, dataset_name)
        descriptive_stats(dataset, dataset_name)
        dataset_info(dataset,dataset_name)
        feature_distribution(dataset,dataset_name)





