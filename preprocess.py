import sys
sys.path.append('/Users/max/Desktop/Cours/DESU/Projet/test_loading_mat')
from test_loading_mat import *

sys.path.append('/Users/max/Desktop/Cours/DESU')
from imports import *

def read_mat_txt_files_improved(folder_path):
    """
    Fonction pour lire tous les fichiers .mat.txt dans un dossier et les stocker dans des DataFrames pandas.
    """
    files = glob.glob(f"{folder_path}/*.mat.txt")
    dataframes = {}
    
    for file in files:
        data_list = read_matlab_txt_file(file)
        df = pd.DataFrame(data_list)
        
        filename = os.path.basename(file).replace('.mat.txt', '')
        dataframes[filename] = df
        print(f"Fichier {filename} lu avec read_matlab_txt_file - Shape : {df.shape}")
    
    for name, df in dataframes.items():
        globals()[f"df_{name}"] = df
        print(f"DataFrame créé : df_{name}")

    print(f"\nTotal : {len(dataframes)} fichiers chargés")
    return dataframes

def data_implementation(dataframe, columns) :
    """
    Fonction de mise en place des données où on travaille avec un seul tableau global pour certaines analyses. 

    Parametres : 
    - dataframe : dictionnaire de DataFrames
    - columns : colonnes à supprimer
    """
    df_all = pd.concat(dataframe.values(), ignore_index=True)

    df_without_columns = df_all.drop(columns=columns)

    nan_percentages = df_without_columns.isna().mean() * 100
    print("Pourcentage de valeurs manquantes par colonne : \n", nan_percentages)

    print("\nTypes de données :\n", df_without_columns.dtypes)
    print("\nValeurs manquantes :\n", df_without_columns.isnull().sum())
    print("\nDoublons : ", df_without_columns['ID_IDn'].duplicated().sum())

    df_implemented_all = df_without_columns.copy()

    return df_implemented_all

def vitesse_globale(tracks_smooth):
    """
    (position finale – position initiale) / (temps final – temps initial) sans valeur absolue.
    Paramètres :
    - tracks_smooth : chaîne de caractères représentant un tuple de deux listes (temps, positions)
    Retourne :
    - vitesse globale (float) correspondant à la formule, vitesse signée
    """
    # Conversion en liste
    if isinstance(tracks_smooth, str):
        tracks_smooth = ast.literal_eval(tracks_smooth)
    if not isinstance(tracks_smooth, (list, tuple)) or len(tracks_smooth) != 2:
        return np.nan
    temps = np.array(tracks_smooth[0])
    positions = np.array(tracks_smooth[1])
    if len(temps) < 2:
        return np.nan
    delta_pos = positions[-1] - positions[0]
    delta_temps = temps[-1] - temps[0]
    if delta_temps == 0:
        return np.nan
    return delta_pos / delta_temps

def isolation_forest(df, colonne, nouvelle_colonne=None):
    """
    Applique IsolationForest sur une colonne du DataFrame et ajoute une colonne booléenne d'outliers.
    Args:
        df: DataFrame d'entrée
        colonne: nom de la colonne à analyser
        contamination: proportion d'outliers attendue
        n_estimators: nombre d'arbres
        random_state: graine aléatoire
        nouvelle_colonne: nom de la colonne de sortie (par défaut: '{colonne}_outlier')
    Returns:
        DataFrame avec la nouvelle colonne d'outliers
    """
    from sklearn.ensemble import IsolationForest
    import numpy as np

    X = df[colonne].values.reshape(-1, 1)
    iso = IsolationForest(
        contamination=0.05,
        n_estimators=300,
        max_samples='auto',
        random_state=42,
        bootstrap=False, 
        verbose=0,
        warm_start=True
    )
    outlier_pred = iso.fit_predict(X)
    col_out = nouvelle_colonne or f"{colonne}_outlier"
    df[col_out] = (outlier_pred == -1)
    return df

def deplacement_net(tracks_smooth):
    """
    Calcule le déplacement net (position finale - position initiale).
    Retourne np.nan si la trajectoire est invalide.
    """
    if isinstance(tracks_smooth, str):
        tracks_smooth = ast.literal_eval(tracks_smooth)
    if not isinstance(tracks_smooth, (list, tuple)) or len(tracks_smooth) != 2:
        return np.nan
    positions = np.array(tracks_smooth[1])
    if len(positions) < 2:
        return np.nan
    return positions[-1] - positions[0]

def calculer_dispersion_track(tracks_smooth):
    """
    Mesure la variabilité de toutes les positions sur la trajectoire.
    Utile pour savoir si le cargo reste localisé ou explore beaucoup d’espace, même s’il fait des allers-retours.
    Si le cargo se déplace beaucoup, la dispersion est élevée.
    Paramètres :
    - tracks_smooth : chaîne de caractères représentant une liste ou un tuple de deux listes (temps, positions)
    Retourne :
    - dispersion (float), soit l'écart-type des positions
    """
    if isinstance(tracks_smooth, str):
        tracks_smooth = ast.literal_eval(tracks_smooth)
    if not isinstance(tracks_smooth, (list, tuple)) or len(tracks_smooth) != 2:
        return np.nan
    positions = np.array(tracks_smooth[1])
    if len(positions) < 2:
        return np.nan
    return np.std(positions)

def calculer_ratio_signal_dispersion(row):
    """
    Calcule le ratio entre le signal et la dispersion.
    le ratio entre le signal (signal_I) et la dispersion (signal_dispersion) pour chaque cargo.
    """
    signal = row['signal_I']
    dispersion = row['signal_dispersion'] if 'signal_dispersion' in row else np.nan
    if pd.isna(signal) or pd.isna(dispersion) or dispersion == 0:
        return np.nan
    return signal / dispersion

def calculer_duree_track(tracks_smooth):
    """
    Calcule la durée du track à partir des temps.
    track_duration : la durée du suivi (track) pour chaque cargo
    """
    if isinstance(tracks_smooth, str):
        tracks_smooth = ast.literal_eval(tracks_smooth)
    if not isinstance(tracks_smooth, (list, tuple)) or len(tracks_smooth) != 2:
        return np.nan
    temps = np.array(tracks_smooth[0])
    if len(temps) < 2:
        return np.nan
    return temps[-1] - temps[0]

def data_implementation_with_exclusion(dataframe, exclude_identifiers=None):
    """
    Fonction de mise en place des données qui filtre un DataFrame existant
    en excluant certaines lignes contenant des identifiants spécifiques.

    Parameters:
    - dataframe: DataFrame déjà concaténé (df_implemented_all)
    - exclude_identifiers: liste des identifiants partiels à exclure des données
    """
    if exclude_identifiers is None:
        exclude_identifiers = []
    
    # Travailler directement sur le DataFrame fourni
    df_without_exclusions = dataframe.copy()
    
    # Filtrer les lignes contenant les identifiants à exclure
    if exclude_identifiers and 'ID_IDn' in df_without_exclusions.columns:
        # Créer un masque pour identifier les lignes à exclure
        mask_exclude = df_without_exclusions['ID_IDn'].str.contains(
            '|'.join(exclude_identifiers), 
            case=False, 
            na=False
        )
        
        # Compter les lignes exclues
        nb_excluded = mask_exclude.sum()
        print(f"Identifiants à exclure: {exclude_identifiers}")
        print(f"Lignes exclues: {nb_excluded}")
        
        # Filtrer le DataFrame
        df_without_exclusions = df_without_exclusions[~mask_exclude]

    return df_without_exclusions