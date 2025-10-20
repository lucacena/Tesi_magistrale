import pandas as pd
from pycanon import anonymity


def control_anon(dataset_name):

    if dataset_name == "mental_health":
        l_div_file_2 = pd.read_csv("dataset_anon/mental_health_before_4.csv", sep='\t')
        QI = ['Age', 'Course', 'Gender', 'Residence_Type']
        SA = ["mental_illness"]
    elif dataset_name == "bank":
        l_div_file_2 = pd.read_csv("dataset_anon/bank_before_2.csv", sep='\t')
        l_div_file_5 = pd.read_csv("dataset_anon/bank_before_5.csv", sep='\t')
        QI = ['age', 'job', 'marital', 'housing']
        SA = ["balance"]
    elif dataset_name == "10":
        l_div_file_2 = pd.read_csv("Delete/anon_dataset/stroke_before_10.csv", sep='\t')
        QI = ["gender", "age", "smoking_status"]
        SA = ["stroke"]
    elif dataset_name == "stroke":
        l_div_file_2 = pd.read_csv("dataset_anon/stroke_k100_l2.csv", sep='\t')
        QI = ["gender", "age", "smoking_status", "Residence_type", "work_type"]
        SA = ["stroke"]
    else:
        l_div_file_2 = pd.read_csv("dataset_anon/NHANES_before_2.csv", sep='\t')
        QI = ["RIDAGEYR", "RIAGENDR"]
        SA = ["DIQ010"]

    #SA = ["SEQN", "DIQ010"]
    l_div_2 = anonymity.l_diversity(l_div_file_2, QI, SA)
    k_anon = anonymity.k_anonymity(l_div_file_2, QI)
    #l_div_5 = anonymity.l_diversity(l_div_file_5, QI, SA)
    print("l_div_2:", l_div_2)
    print("k_anon:", k_anon)
    #print("l_div_5:", l_div_5)

control_anon('stroke')
#control_anon('bank')
#control_anon('stroke')