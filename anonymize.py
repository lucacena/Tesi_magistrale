import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, '/Users/lucacena/Desktop/Tesi/Prove_codice/Prova_anonypy/Anonypy_Second_trial/anonypy')
from anonypy import anonypy


#Se l'elemento è una lista, prende solo il primo elemento della lista.
def fix_sep(lst):
    if isinstance(lst, list):
        return lst[0]
    else:
        return lst


#questa è versione di prova per main_invertito
def discretize(col):
    def parse_value(x):
        # Caso 1: numerico già pronto
        if isinstance(x, (int, float, np.number)):
            return float(x)
        # Caso 2: tupla → prendo la media
        elif isinstance(x, tuple):
            return sum(x) / 2
        # Caso 3: stringa che rappresenta un intervallo tipo "10-20"
        elif isinstance(x, str):
            # se è un numero scritto come stringa (es. "70.0")
            if x.replace('.', '', 1).isdigit():
                return float(x)
            # se è un intervallo numerico tipo "0.08-42.0"
            elif "-" in x and all(part.replace('.', '', 1).isdigit() for part in x.split("-")):
                low, high = map(float, x.split("-"))
                return (low + high) / 2
            # altrimenti lascio stringa (categoria non numerica)
            else:
                return str(x)
        # Caso fallback: trasformo in stringa
        else:
            print(f"[WARNING] Valore inatteso in discretize: {x} (tipo={type(x)})")
            return str(x)
    # Applica la trasformazione
    parsed = col.apply(parse_value)
    # Se TUTTI i valori sono numerici → restituisco float
    if all(isinstance(v, (int, float, np.number)) for v in parsed):
        return parsed.astype(float)
    # Altrimenti uso il LabelEncoder solo sulle categorie residue
    le = LabelEncoder()
    return pd.Series(le.fit_transform(parsed.astype(str)), index=col.index)



def l_diversity(dataset_name, df, k, l):
    if dataset_name == 'NHANES':
        feature_columns = ["RIDAGEYR", "RIAGENDR", "BMXBMI", "LBXGLU", "LBXIN"]
        sensitive_column = "DIQ010"
    elif dataset_name == 'stroke':
        feature_columns = ["gender", "age", "Residence_type", "work_type"]
        sensitive_column = "stroke"
    elif dataset_name == "diabetes":
        feature_columns = ['age', 'hypertension', 'HbA1c_level', 'blood_glucose_level']
        sensitive_column = "diabetes"
    elif dataset_name == "adult":
        feature_columns = ['age', 'education', 'marital-status', 'occupation', 'race']
        sensitive_column = "income"
    elif dataset_name == "mental_health":
        feature_columns = ['Age', 'Course', 'Gender', 'Relationship_Status', 'Residence_Type', 'stress_anxiety_diff', 'Mental_Severity_Score']
        sensitive_column = "mental_illness"
    elif dataset_name == 'prova':
        feature_columns = ["Età", "ZIPCODE", "Sesso"]
        sensitive_column = "Malattia"
    else:
        raise ValueError('Invalid dataset name')
    df = df.copy()
    df["_row_id"] = df.index.astype(str)  #aggiunge colonna
    #print(df[sensitive_column].value_counts())
    p = anonypy.Preserver(df, feature_columns, sensitive_column)
    # Anonimizzazione
    rows = p.anonymize_l_diversity(k, l)
    dfn = pd.DataFrame(rows)
    dfn = dfn.map(lambda x: fix_sep(x))
    # Crea file csv con dataset anonimizzato prima del merge
    dfn.to_csv(f"dataset_anon/{dataset_name}_before_cat_{k}_{l}.csv", sep='\t',
               index=False)
    columns_to_discretize = feature_columns
    for col in columns_to_discretize:
        dfn[col] = discretize(dfn[col])
    dfn.to_csv(f"dataset_anon/{dataset_name}_after_discretize_{k}_{l}.csv", sep='\t',
                   index=False)
    #ricava automaticamente colonne originali NON usate nell’anonimizzazione
    excluded_cols = set(feature_columns + [sensitive_column, "_row_id"])
    original_metadata_cols = [col for col in df.columns if col not in excluded_cols]
    #merge fatto usando il _row_id
    columns_to_merge = ["_row_id"] + original_metadata_cols
    original_cols = df[columns_to_merge]
    dfn = dfn.merge(original_cols, on="_row_id", how="left")
    return dfn
