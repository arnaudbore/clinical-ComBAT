import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score

from clinical_combat.utils.robust import find_outliers_IQR, find_outliers_MAD, reject_outliers_until_mad_equals_mean, find_outliers_VS, find_outliers_VS2, remove_top_x_percent, cheat

import matplotlib.pyplot as plt

def find_outliers(df, robust_method, args = []):
    # Find outliers
    outliers_idx = []
    if robust_method in ['IQR', 'MAD', 'MAD_MEAN', 'VS', 'VS2','TOP5', 'TOP10', 'TOP20', 'TOP30', 'TOP40', 'TOP50', 'CHEAT']:
        for metric in df['metric'].unique():
            for bundle in df['bundle'].unique():
                data = df[(df['metric'] == metric) & (df['bundle'] == bundle)]
                outliers_idx += use_robust_method(data, robust_method)
    elif robust_method == 'kmeans':
        outliers_idx = use_robust_method(data, robust_method)
    return outliers_idx

def use_robust_method(data, robust_method, args = []):
    if robust_method == 'IQR':
        return find_outliers_IQR(data)
    elif robust_method == 'MAD':
        return find_outliers_MAD(data, args)
    elif robust_method == 'MAD_MEAN':
        return reject_outliers_until_mad_equals_mean(data, args)
    elif robust_method == 'VS':
        return find_outliers_VS(data)
    elif robust_method == 'VS2':
        return find_outliers_VS2(data)
    elif robust_method in ['TOP5', 'TOP10', 'TOP20', 'TOP30', 'TOP40', 'TOP50']:
        return remove_top_x_percent(data, x=int(robust_method
        .replace('TOP', '')))
    elif robust_method == 'CHEAT':
        return cheat(data)
    else:
        raise ValueError("Invalid robust method. Choose between 'iqr' and 'mad'.")

def analyze_detection_performance(outliers_idx, mov_data):
    
    metrics_list = []
    mov_data['is_malade'] = mov_data['disease'].apply(lambda x: 0 if x == 'HC' else 1)
    mov_data['is_outlier'] = 0
    mov_data.loc[outliers_idx, 'is_outlier'] = 1
    
    bundle_column = 'metric_bundle' if 'metric_bundle' in mov_data.columns else 'bundle'
    for bundle in mov_data[bundle_column].unique():
        bundle_data = mov_data[mov_data[bundle_column] == bundle]
        #plot_outliers_data(bundle_data)

        y_true = bundle_data['is_malade'].tolist()
        y_pred = bundle_data['is_outlier'].tolist()

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        taux_faux_positifs = fp / (fp + tn) if (fp + tn) != 0 else 0
        f1 = f1_score(y_true, y_pred)

        metrics_list.append({
            'bundle': bundle,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'taux_faux_positifs': taux_faux_positifs,
            'f1_score': f1
        })

    # Overall metrics
    overall_outliers_sid = mov_data.loc[outliers_idx]['sid'].unique().tolist()
    mov_data['is_outlier'] = mov_data['sid'].apply(lambda x: 1 if x in overall_outliers_sid else 0)
    patients = mov_data.drop_duplicates(subset='sid')

    y_true = patients['is_malade'].tolist()
    y_pred = patients['is_outlier'].tolist()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    taux_faux_positifs = fp / (fp + tn) if (fp + tn) != 0 else 0
    f1 = f1_score(y_true, y_pred)

    metrics_list.append({
        'bundle': 'overall',
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'taux_faux_positifs': taux_faux_positifs,
        'f1_score': f1
    })
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.set_index('bundle', inplace=True)
    metrics_df = metrics_df.sort_index(axis=1)
    metrics_df = metrics_df.T.reset_index()
    metrics_df.rename(columns={'index': 'metric'}, inplace=True)
    metrics_df['site'] = mov_data.site.unique()[0]
    return metrics_df


def scatter_plot_with_colors(df, outliers_idx,  y_column, directory, file_path, title=None):
    df['is_malade'] = df['disease'].apply(lambda x: 0 if x == 'HC' else 1)
    df['is_outlier'] = 0
    df.loc[outliers_idx, 'is_outlier'] = 1
    colors = []
    bundle_column = 'metric_bundle' if 'metric_bundle' in df.columns else 'bundle'
    for bundle in df[bundle_column].unique():
        bundle_data = df[df[bundle_column] == bundle]
        colors = []
        for _, row in bundle_data.iterrows():
            if row['is_malade'] == 0 and row['is_outlier'] == 0:
                colors.append('blue')
            elif row['is_malade'] == 1 and row['is_outlier'] == 1:
                colors.append('green')
            elif row['is_outlier'] == 1 and row['is_malade'] == 0:
                colors.append('red')
            elif row['is_malade'] == 1 and row['is_outlier'] == 0:
                colors.append('orange')

        plt.scatter(bundle_data['age'], bundle_data[y_column], c=colors)
        plt.xlabel('age')
        plt.ylabel(y_column)
        plt.title(f"{title} - {bundle}")
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory,f"{file_path}_{bundle}.png"))
        plt.clf()


def get_matching_indexes(file_path, subset_path):
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(file_path)
    df2 = pd.read_csv(subset_path)

    # Find the matching indexes where entire rows are the same
    matching_indexes = df2["old_index"].tolist()

    # On vérifie ligne par ligne que DF2.iloc[i] == DF1.loc[DF2.iloc[i]["OldIndex"]]
    colonnes_a_verifier = ["sid", "bundle", "mean", "age"]

    # Boucle de vérification
    for i in range(len(df2)):
        index_dans_df1 = df2.iloc[i]["old_index"]
        ligne_df1 = df1.loc[index_dans_df1, colonnes_a_verifier]
        ligne_df2 = df2.iloc[i][colonnes_a_verifier]

        if not ligne_df1.equals(ligne_df2):
            print(f"Mismatch à la ligne {i} :")
            print("Dans df1 :\n", ligne_df1)
            print("Dans df2 :\n", ligne_df2)
            print("-" * 40)


    return matching_indexes
