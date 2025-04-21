from clinical_combat.harmonization import from_model_name
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import json
import os
from scipy.spatial.distance import euclidean


def remove_outliers(ref_data, mov_data, args):
    print("Removing outliers with method", args.robust)

    find_outlier = ROBUST_METHODS.get(args.robust)
    rwp = args.rwp

    # Initialize QC model
    QC = from_model_name(
        args.method.lower(),
        ignore_handedness_covariate=args.ignore_handedness,
        ignore_sex_covariate=args.ignore_sex,
        use_empirical_bayes=not args.no_empirical_bayes,
        limit_age_range=args.limit_age_range,
        degree=args.degree,
        regul_ref=args.regul_ref,
        regul_mov=args.regul_mov,
        nu=args.nu,
        tau=args.tau,
    )
    QC.fit(ref_data, mov_data, False)
    site = mov_data['site'].unique()[0]
    rwp_str = "RWP" if rwp else "NoRWP"

    # Process movement data
    design_mov, y_mov = QC.get_design_matrices(mov_data)
    y_no_cov = QC.remove_covariate_effect(design_mov, y_mov)
    y_no_cov_flat = np.array(y_no_cov).flatten()
    mov_data.insert(3, "mean_no_cova", y_no_cov_flat, True)
    mov_data = remove_covariates_effects2(mov_data)

    # Find outliers
    outliers_idx = []
    for bundle in QC.bundle_names:
        data = mov_data.query("bundle == @bundle")
        outliers_idx += find_outlier(data)


    mov_data = mov_data.drop(columns=['mean_no_cov'])

    # Save outliers
    outliers_filename = os.path.join(args.out_dir,f"outliers_{site}_{args.robust}_{rwp_str}.csv")
    outliers = mov_data.loc[outliers_idx]
    outliers.index.name = "old_index"
    outliers.to_csv(outliers_filename, index=True)

    # Remove outliers from movement data
    if rwp:
        print("RWP is applied")
        outlier_patients_ids = mov_data.loc[outliers_idx]['sid'].unique().tolist()
        if len(outlier_patients_ids) < (len(mov_data['sid'].unique().tolist())-1):
            mov_data = mov_data[~mov_data['sid'].isin(outlier_patients_ids)]
        else:
            print("All patients are outliers. RWP not applied.")
            mov_data = mov_data.drop(outliers_idx)
    else:
        mov_data = mov_data.drop(outliers_idx)
    return mov_data

def find_outliers_IQR(data):

    Q1 = data['mean_no_cov'].quantile(0.25)
    Q3 = data['mean_no_cov'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrer les valeurs aberrantes
    outliers = data[(data['mean_no_cov'] < lower_bound) | (data['mean_no_cov'] > upper_bound)]

    return outliers.index.to_list()

def find_outliers_VS(data, column='mean_no_cov'):
    """
    Équilibre les valeurs autour de la médiane en supprimant les valeurs les plus éloignées
    jusqu'à ce que la somme des écarts à droite et à gauche de la médiane soit équilibrée.

    Paramètres :
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne à analyser pour équilibrer les valeurs.

    Retourne :
        list: Une liste des indices des valeurs supprimées.
    """
    # --- Classement des métriques (comme décidé plus tôt) -------------
    METRICS_HIGH = {'md', 'mdt', 'rd', 'rdt', 'fw', 'ad', 'adt'}  # patho ↑
    METRICS_LOW  = {'fa', 'fat', 'afd'}                           # patho ↓
    # ------------------------------------------------------------------

    metric_name = data['metric'].iloc[0]

    if metric_name in METRICS_HIGH:
        side = 'right'   # on enlève les plus grosses valeurs
    elif metric_name in METRICS_LOW:
        side = 'left'    # on enlève les plus petites valeurs
    else:
        return []        # métrique non classée → on ne fait rien

    outliers_idx = []
    median = data[column].median()

    while True:
        # Moyennes des écarts de chaque côté de la médiane
        left_mean  = abs((data[data[column] < median][column] - median).mean())
        right_mean = abs((data[data[column] > median][column] - median).mean())

        # Équilibre atteint (même critère qu’avant)
        if abs(left_mean - right_mean) <= 1e-6:
            break

        if side == 'right':
            # On continue seulement si la droite domine encore
            if right_mean <= left_mean:
                break
            target_idx = data[column].idxmax()   # plus grand écart à droite
        else:  # side == 'left'
            if left_mean <= right_mean:
                break
            target_idx = data[column].idxmin()   # plus grand écart à gauche

        outliers_idx.append(target_idx)
        data = data.drop(target_idx)

    return outliers_idx


def find_outliers_VS2(data, column='mean_no_cov'):
    """
    Équilibre les valeurs autour de la médiane en supprimant les valeurs les plus éloignées
    jusqu'à ce que la somme des écarts à droite et à gauche de la médiane soit équilibrée.

    Paramètres :
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne à analyser pour équilibrer les valeurs.

    Retourne :
        list: Une liste des indices des valeurs supprimées.
    """
    # --- Classement des métriques (comme décidé plus tôt) -------------
    METRICS_HIGH = {'md', 'mdt', 'rd', 'rdt', 'fw', 'ad', 'adt'}  # patho ↑
    METRICS_LOW  = {'fa', 'fat', 'afd'}                           # patho ↓
    # ------------------------------------------------------------------

    metric_name = data['metric'].iloc[0]

    if metric_name in METRICS_HIGH:
        side = 'right'   # on enlève les plus grosses valeurs
    elif metric_name in METRICS_LOW:
        side = 'left'    # on enlève les plus petites valeurs
    else:
        return []        # métrique non classée → on ne fait rien

    outliers_idx = []
    median = data[column].median()

    while True:
        # Moyennes des écarts de chaque côté de la médiane
        median = data[column].median()
        
        left_mean  = abs((data[data[column] < median][column] - median).mean())
        right_mean = abs((data[data[column] > median][column] - median).mean())

        # Équilibre atteint (même critère qu’avant)
        if abs(left_mean - right_mean) <= 1e-6:
            break

        if side == 'right':
            # On continue seulement si la droite domine encore
            if right_mean <= left_mean:
                break
            target_idx = data[column].idxmax()   # plus grand écart à droite
        else:  # side == 'left'
            if left_mean <= right_mean:
                break
            target_idx = data[column].idxmin()   # plus grand écart à gauche

        outliers_idx.append(target_idx)
        data = data.drop(target_idx)

    return outliers_idx

def find_outliers_MAD(data, column='mean_no_cov', threshold=3.5):
    """
    Détecte les valeurs aberrantes dans un DataFrame à l'aide de la méthode MAD.

    Paramètres :
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne à analyser pour détecter les outliers.
        threshold (float): Le seuil pour considérer une valeur comme un outlier. Par défaut : 3.5.

    Retourne :
        list: Une liste des indices des valeurs aberrantes.
    """
    # Calcul de la médiane de la colonne
    median = data[column].median()

    # Calcul du MAD
    mad = np.median(np.abs(data[column] - median))

    # Calcul des scores normalisés (distance modifiée)
    if mad == 0:
        print("MAD is zero, all values will appear as non-outliers.")
        return []

    modified_z_scores = 0.6745 * (data[column] - median) / mad

    # Identification des indices des outliers
    outliers = data[np.abs(modified_z_scores) > threshold]

    return outliers.index.to_list()


def reject_outliers_until_mad_equals_mean(data,  threshold=0.001): 
    column = 'mean_no_cov'
    outliers_idx = []

    # --- Définis tes trois métriques ici ------------------------------
    METRICS_HIGH = {'md', 'mdt', 'rd', 'rdt', 'fw', 'ad', 'adt'}
    METRICS_LOW  = {'fa', 'fat', 'afd'}
    # ------------------------------------------------------------------

    metric_name = data['metric'].iloc[0]

    if metric_name in METRICS_HIGH:
        pick_idx = lambda s: s.idxmax()   # enlève la valeur maximale
    elif metric_name in METRICS_LOW:
        pick_idx = lambda s: s.idxmin()   # enlève la valeur minimale
    else:
        # Rien à faire, on retourne une liste vide
        return outliers_idx

    while True:
        median = data[column].median()
        mean   = data[column].mean()

        if abs(median - mean) / median < threshold:
            break

        target_idx = pick_idx(data[column])
        outliers_idx.append(target_idx)
        data = data.drop(target_idx)

    return outliers_idx

def remove_top_x_percent(data, column='mean_no_cov', x=5):
    """
    Supprime les x % des valeurs les plus élevées dans une colonne donnée.

    Paramètres :
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne à analyser.
        x (float): Le pourcentage des valeurs les plus élevées à supprimer.

    Retourne :
        list: Une liste des indices des valeurs supprimées.
    """
    if x <= 0 or x > 100:
        raise ValueError("x doit être un pourcentage entre 0 et 100.")

    # Calcul du nombre de valeurs à supprimer
    num_to_remove = int(len(data) * (x / 100.0))

    # Trouver les indices des x % des valeurs les plus élevées
    outliers_idx = data.nlargest(num_to_remove, column).index.to_list()
    # Afficher les 'sid' des outliers

    return outliers_idx

def top5(data):
    return remove_top_x_percent(data, column='mean_no_cov', x=5)

def top10(data):
    return remove_top_x_percent(data, column='mean_no_cov', x=10)

def top20(data):
    return remove_top_x_percent(data, column='mean_no_cov', x=20)

def top30(data):
    return remove_top_x_percent(data, column='mean_no_cov', x=30)

def top40(data):
    return remove_top_x_percent(data, column='mean_no_cov', x=40)   

def top50(data):
    return remove_top_x_percent(data, column='mean_no_cov', x=50)   

def cheat(data):
    return data[data['disease'] != 'HC'].index.to_list()

def rien(data):
    return []

    
ROBUST_METHODS = {
    "IQR": find_outliers_IQR,
    "MAD": find_outliers_MAD,
    "MMS": reject_outliers_until_mad_equals_mean,
    "VS": find_outliers_VS,
    "VS2": find_outliers_VS2,
    "TOP5": top5,
    "TOP10": top10,
    "TOP20": top20,
    "TOP30": top30,
    "TOP40": top40,
    "TOP50": top50,
    "CHEAT": cheat,
    "FLIP": rien
}
from clinical_combat.harmonization.QuickCombat import QuickCombat

def get_design_matrices(df, ignore_handedness=False):
    design = []
    Y = []
    for bundle in list(np.unique(df["bundle"])):
        data = df.query("bundle == @bundle")
        hstack_list = []
        hstack_list.append(np.ones(len(data["sid"])))  # intercept
        hstack_list.append(QuickCombat.to_category(data["sex"]))
        if not ignore_handedness:
            hstack_list.append(QuickCombat.to_category(data["handedness"]))
        ages = data["age"].to_numpy()
        hstack_list.append(ages)
        design.append(np.array(hstack_list))
        Y.append(data["mean"].to_numpy())
    return design, Y

def remove_covariates_effects2(df):
    df = df.sort_values(by=["site", "sid", "bundle"])
    ignore_handedness = False
    if df['handedness'].nunique() == 1:
        ignore_handedness = True
    design, y = get_design_matrices(df, ignore_handedness)
    alpha, beta = QuickCombat.get_alpha_beta(design, y)

    df['mean_no_cov'] = df['mean']
    for i, bundle in enumerate(list(np.unique(df["bundle"]))):
        bundle_df = df[df['bundle'] == bundle]
        covariate_effect = np.dot(design[i][1:, :].transpose(), beta[i])
        df.loc[df['bundle'] == bundle, 'mean_no_cov'] = (y[i] - covariate_effect)
    return df