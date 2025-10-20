from anonypy import mondrian
import pandas as pd


class Preserver:

    def __init__(self, df, feature_columns, sensitive_column):
        self.modrian = mondrian.Mondrian(df, feature_columns, sensitive_column)

    def __anonymize(self, k, l=0, p=0.0):
        partitions = self.modrian.partition(k, l, p)
        return anonymize(
            self.modrian.df,
            partitions,
            self.modrian.feature_columns,
            self.modrian.sensitive_column,
        )

    def anonymize_k_anonymity(self, k):
        return self.__anonymize(k)

    def anonymize_l_diversity(self, k, l):
        return self.__anonymize(k, l=l)

    def anonymize_t_closeness(self, k, p):
        return self.__anonymize(k, p=p)

    def __count_anonymity(self, k, l=0, p=0.0):
        partitions = self.modrian.partition(k, l, p)
        return count_anonymity(
            self.modrian.df,
            partitions,
            self.modrian.feature_columns,
            self.modrian.sensitive_column,
        )

    def count_k_anonymity(self, k):
        return self.__count_anonymity(k)

    def count_l_diversity(self, k, l):
        return self.__count_anonymity(k, l=l)

    def count_t_closeness(self, k, p):
        return self.__count_anonymity(k, p=p)


def agg_categorical_column(series):
    series.astype("category")  # workaround bug
    converted = [str(n) for n in set(series)]
    return [",".join(converted)]


def agg_numerical_column(series):
    minimum = series.min()
    maximum = series.max()
    if maximum == minimum:
        return [str(maximum)]
    else:
        return [f"{minimum}-{maximum}"]


def anonymize(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if df[column].dtype.name == "category":
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column

    rows = []
    for i, partition in enumerate(partitions): #itero su gruppi di righe già clusterizzate per l-diversity
        if max_partitions is not None and i > max_partitions:
            break

        grouped_columns = {}
        for column in feature_columns:
            grouped_columns[column] = aggregations[column](df.loc[partition, column])

        #Raccogliamo anche i _row_id (non serve che sia in feature_columns)
        grouped_columns["_row_id"] = df.loc[partition, "_row_id"].tolist()

        #Raggruppamento corretto dei row_id per valore sensibile
        sensitive_value_to_row_ids = (
            df.loc[partition]
            .groupby(sensitive_column, observed=True)["_row_id"]
            .apply(list)
            .to_dict()
        )
        """Prendi tutte le righe del gruppo corrente (partition)
            Le raggruppi per valore sensibile (es. stroke = 1 e stroke = 0)
            Per ogni valore sensibile, ottieni una lista di _row_id che avevano quel valore
            {
                  0: ['4', '5', '6'],
                  1: ['1', '2', '3']
                }"""

        for sensitive_value, row_ids in sensitive_value_to_row_ids.items():
            for rid in row_ids:
                values = {key: val for key, val in grouped_columns.items() if key != "_row_id"}
                values["_row_id"] = rid
                values[sensitive_column] = sensitive_value
                rows.append(values)
        """Costruisci una riga così:
                values[column]: contiene i valori aggregati delle feature
                values[sensitive_column]: mantiene il valore sensibile originale della riga (non aggregato!)
                values["_row_id"]: mette il vero _row_id per poter fare il merge più avanti
                lo fa per ogni combinazione valore_sensibile-row_id"""
    return rows


def count_anonymity(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if df[column].dtype.name == "category":
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column

    aggregations[sensitive_column] = "count"

    rows = []
    for i, partition in enumerate(partitions):
        if max_partitions is not None and i > max_partitions:
            break

        grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
        values = grouped_columns.apply(
            lambda x: x[0] if isinstance(x, list) else x
        ).to_dict()

        rows.append(values.copy())

    return rows
