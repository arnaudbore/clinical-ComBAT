import pandas as pd
import numpy as np 
import os

def calculate_precision_by_bundle(metrics_compilation_df):
    # ANALYZE BEST BUNDLES for F1, precision etc
    """
    Calcule le score de précision par bundle.

    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les données avec les colonnes 'bundle' et 'is_malade'.

    Returns:
    pd.DataFrame: Un DataFrame avec les bundles et leurs scores de précision respectifs.
    """
    total = pd.DataFrame()

    for bundle_column in metrics_compilation_df.columns:
        if bundle_column in ['site','metric','num_patients','disease_ratio','num_diseased']:
            continue # Skip non-numeric columns
        bundle_df = metrics_compilation_df[[bundle_column, 'metric']].copy()
        grouped_df = bundle_df.groupby(['metric']).mean().reset_index()
        grouped_df.set_index('metric', inplace=True)
        total = pd.concat([total, grouped_df.T])
        
    return total


# COUNT BUNDLES PER OUTLIERS
def count_bundles_per_outliers(df):
    """
    Analyze outliers in the DataFrame and calculate the percentage of SIDs with a certain number of occurrences.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'sid', 'is_outlier', and 'is_sick' columns.

    Returns:
    dict: A dictionary with the percentage of SIDs with a certain number of occurrences for sick and healthy groups.
    """
    
    # Count the number of occurrences of each SID
    # Count the number of occurrences of each combination of SID and site
    sid_counts = df.groupby(['sid', 'site', 'is_malade']).size().reset_index(name='count_bundle')
    
    # Divide the dataset into two groups: sick and healthy
    sick_sids = sid_counts[sid_counts['is_malade'] == 1]
    healthy_sids = sid_counts[sid_counts['is_malade'] == 0]
    
    # Calculate the percentage of SIDs with a certain number of occurrences for sick group
    sick_counts = sick_sids.groupby(['count_bundle']).size().reset_index(name='prct_occurence')
    sick_counts['prct_occurence'] = sick_counts['prct_occurence']/sick_counts['prct_occurence'].sum()*100
    # Calculate the percentage of SIDs with a certain number of occurrences for healthy group
    healthy_counts = healthy_sids.groupby(['count_bundle']).size().reset_index(name='prct_occurence')
    healthy_counts['prct_occurence'] = healthy_counts['prct_occurence']/healthy_counts['prct_occurence'].sum()*100

    total = pd.merge(sick_counts, healthy_counts, on=['count_bundle'], suffixes=('_sick', '_healthy'))
    
    return total

# Example usage
#bundles_per_outliers = count_bundles_per_outliers(pd.read_csv(os.path.join(MAINFOLDER, robust_method, "outliers_compilation.csv")))
#bundles_per_outliers.head(10)

def get_distribution_properties(mov_data):
    """
    Calcule plusieurs mesures de distribution pour chaque 'bundle' dans mov_data.
    mov_data doit contenir au moins deux colonnes :
        - 'bundle' (catégoriel)
        - 'mean_no_cov' (valeurs numériques)
    Retourne un DataFrame avec une ligne par mesure et une colonne par 'bundle'.
    """

    # Dictionnaires pour stocker les métriques par bundle
    skewness_per_bundle = {}
    mean_median_shift_per_bundle = {}
    kurtosis_per_bundle = {}
    
    # Mesures supplémentaires
    bowley_skewness_per_bundle = {}
    left_right_mean_diff_per_bundle = {}
    left_right_std_ratio_per_bundle = {}

    # Parcours de chaque bundle
    for bundle in mov_data['bundle'].unique():
        bundle_data = mov_data[mov_data['bundle'] == bundle]
        
        # Distribution à analyser
        x = bundle_data['mean_no_cov'].dropna()
        
        # Skewness, kurtosis, etc.
        skewness_per_bundle[bundle] = x.skew()
        kurtosis_per_bundle[bundle] = x.kurtosis()
        
        # Écart (absolu) entre la moyenne et la médiane (relatif à la médiane)
        median_val = x.median()
        mean_val = x.mean()
        mean_median_shift_per_bundle[bundle] = np.abs(mean_val - median_val) / median_val if median_val != 0 else np.nan
        
        # --- Métriques supplémentaires ---
        
        # 1. Bowley Skewness (fondée sur les quartiles)
        Q1 = x.quantile(0.25)
        Q3 = x.quantile(0.75)
        # Assurer que (Q3 - Q1) != 0
        if Q3 != Q1:
            bowley_skewness_per_bundle[bundle] = ((Q3 - median_val) - (median_val - Q1)) / (Q3 - Q1)
        else:
            bowley_skewness_per_bundle[bundle] = np.nan
        
        # 2. Différence de moyenne (côté droit vs côté gauche de la médiane), normalisée par l'écart-type global
        left_data = x[x < median_val]
        right_data = x[x > median_val]
        std_global = x.std()
        if (len(left_data) > 0) and (len(right_data) > 0) and std_global != 0:
            left_mean = left_data.mean()
            right_mean = right_data.mean()
            left_right_mean_diff_per_bundle[bundle] = (right_mean - left_mean) / std_global
        else:
            left_right_mean_diff_per_bundle[bundle] = np.nan
        
        # 3. Ratio des écarts-types (côté droit / côté gauche)
        if (len(left_data) > 1) and (len(right_data) > 1):
            left_std = left_data.std()
            right_std = right_data.std()
            # Pour éviter division par zéro
            if left_std != 0:
                left_right_std_ratio_per_bundle[bundle] = right_std / left_std
            else:
                left_right_std_ratio_per_bundle[bundle] = np.nan
        else:
            left_right_std_ratio_per_bundle[bundle] = np.nan

    # Création du DataFrame avec une ligne par propriété et une colonne par bundle
    bundles = mov_data['bundle'].unique()
    df = pd.DataFrame(
        index=[
            'skewness',
            'mean_median_shift',
            'kurtosis',
            'bowley_skewness',
            'left_right_mean_diff',
            'left_right_std_ratio'
        ],
        columns=bundles
    )

    # Remplissage du DataFrame
    for bundle in bundles:
        df.at['skewness', bundle] = skewness_per_bundle[bundle]
        df.at['mean_median_shift', bundle] = mean_median_shift_per_bundle[bundle]
        df.at['kurtosis', bundle] = kurtosis_per_bundle[bundle]
        df.at['bowley_skewness', bundle] = bowley_skewness_per_bundle[bundle]
        df.at['left_right_mean_diff', bundle] = left_right_mean_diff_per_bundle[bundle]
        df.at['left_right_std_ratio', bundle] = left_right_std_ratio_per_bundle[bundle]

    return df.reset_index().rename(columns={'index': 'property'})
