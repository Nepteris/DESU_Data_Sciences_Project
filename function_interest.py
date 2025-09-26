import sys
sys.path.append('/Users/max/Desktop/Cours/DESU')
from imports import *

def calculer_vitesse(tracks_smooth):
    # Conversion en liste
    if isinstance(tracks_smooth, str):
        tracks_smooth = ast.literal_eval(tracks_smooth)
    
    # Vérification structure
    if not isinstance(tracks_smooth, (list, tuple)) or len(tracks_smooth) != 2:
        return np.nan
    
    temps = np.array(tracks_smooth[0])
    positions = np.array(tracks_smooth[1])
            
    # Besoin d'au moins 2 points
    if len(temps) < 2:
        return np.nan
    
    # Vitesse moyenne = déplacement total / temps total
    return (positions[-1] - positions[0]) / (temps[-1] - temps[0])

def stats_vitesse_par_modalite(df, nom):
    print(f"\nStatistiques descriptives ({nom}) ---")
    for mod, label in [(1, "Front d'onde"), (2, "Fluorescence")]:
        subset = df[df['ID_modality'] == mod]['speed'].dropna()
        print(f"\n{label} (n={len(subset)})")
        print(f"  Moyenne : {subset.mean():.2f} µm/s")
        print(f"  Médiane : {subset.median():.2f} µm/s")
        print(f"  Min     : {subset.min():.2f} µm/s")
        print(f"  Max     : {subset.max():.2f} µm/s")
        print(f"  Écart-type : {subset.std():.2f} µm/s")
        print(f"  Quartiles : {subset.quantile([0.25, 0.5, 0.75]).values}")

def nettoyer_crossdetection_object(df):   
    """
    Le nettoyage pour la crossdétection est nécessaire, car les valeurs sont sous forme de string et non de boolean
    """ 
    df['crossdetection_bool_clean'] = df['crossdetection_bool'].astype(str).str.lower() == 'true'
    
    nb_true = df['crossdetection_bool_clean'].sum()
    nb_false = len(df) - nb_true
    
    print(f"Nombre de valeurs True: {nb_true}",
           f"et nombre de valeurs False : {nb_false}")
    
    return df

def particules_crossdetectees(df):
    cross_detected = df[df['crossdetection_bool_clean'] == True]
    total_particules = len(df)
    nb_cross = len(cross_detected)
    
    print(f"Nombre de particules: {total_particules}")
    print(f"Nombre de particules crossdétectées: {nb_cross} ({nb_cross/total_particules*100:.1f}%)")

    return cross_detected, total_particules, nb_cross

def métriques_crossdétectées(df_utile, cross_detected):

    # Calcul métrique
    stats_data = {
        'Toutes': [
            len(df_utile),
            df_utile['speed'].mean(),
            df_utile['speed'].median(),
            df_utile['speed'].min(),
            df_utile['speed'].max(),
            df_utile['speed'].std()
        ],
        'Crossdétectées': [
            len(cross_detected),
            cross_detected['speed'].mean(),
            cross_detected['speed'].median(),
            cross_detected['speed'].min(),
            cross_detected['speed'].max(),
            cross_detected['speed'].std()
        ]
    }
    
    # Index des métriques
    metriques = ['Nombre', 'Vitesse moyenne', 'Vitesse médiane', 
                'Vitesse min', 'Vitesse max', 'Écart-type']
    
    # Créer le DataFrame
    df_stats = pd.DataFrame(stats_data, index=metriques)
    
    return df_stats

def analyser_crossdetection_par_signal_OPD(df, seuil_signal=None):
    """
    Analyse les particules crossdétectées selon un seuil de signal_I
    UNIQUEMENT pour les données OPD (ID_modality = 1)
    
    Parameters:
    - df: DataFrame avec les données
    - seuil_signal: seuil pour distinguer OPD faible/fort (si None, utilise la médiane)
    """
    
    # Filtrer pour ne garder que les données OPD
    if 'ID_modality' not in df.columns:
        print("Colonne 'ID_modality' non trouvée dans le DataFrame")
        return
    
    df_opd = df[df['ID_modality'] == 1].copy()
    
    if len(df_opd) == 0:
        print("Aucune donnée OPD trouvée (ID_modality = 1)")
        return
    
    print(f"Analyse sur {len(df_opd)} particules OPD (sur {len(df)} total)")
    
    if 'signal_I' not in df_opd.columns:
        print("Colonne 'signal_I' non trouvée dans le DataFrame")
        return
    
    # Catégoriser selon le signal
    df_faible = df_opd[df_opd['signal_I'] <= seuil_signal]  # OPD faible
    df_fort = df_opd[df_opd['signal_I'] > seuil_signal]     # OPD fort
    
    # Particules crossdétectées dans chaque catégorie
    cross_faible = df_faible[df_faible['crossdetection_bool_clean'] == True]
    cross_fort = df_fort[df_fort['crossdetection_bool_clean'] == True]
    
    # Statistiques
    total_faible = len(df_faible)
    total_fort = len(df_fort)
    nb_cross_faible = len(cross_faible)
    nb_cross_fort = len(cross_fort)
    
    pct_cross_faible = (nb_cross_faible / total_faible * 100) if total_faible > 0 else 0
    pct_cross_fort = (nb_cross_fort / total_fort * 100) if total_fort > 0 else 0
    
    print(f"\nRésultats (OPD uniquement):")
    print(f"OPD faible (≤{seuil_signal:.2f}): {nb_cross_faible}/{total_faible} crossdétectées ({pct_cross_faible:.1f}%)")
    print(f"OPD fort (>{seuil_signal:.2f}): {nb_cross_fort}/{total_fort} crossdétectées ({pct_cross_fort:.1f}%)")
    
    # Graphiques
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Analyse crossdétection - OPD uniquement (ID_modality = 1)', fontsize=16)
    
    # 1. Distribution du signal avec seuil
    axes[0,0].hist(df_opd['signal_I'].dropna(), bins=50, alpha=0.6, color='green', label='Toutes (OPD)')
    axes[0,0].axvline(seuil_signal, color='black', linestyle='--', linewidth=2, label=f'Seuil: {seuil_signal:.2f}')
    axes[0,0].set_xlabel('Signal I (OPD)')
    axes[0,0].set_ylabel('Fréquence')
    axes[0,0].set_title('Distribution du signal OPD avec seuil')
    axes[0,0].legend()
    axes[0,0].set_yscale('log')
    
    # 2. Pourcentage de crossdétection par catégorie
    categories = ['OPD faible', 'OPD fort']
    pourcentages = [pct_cross_faible, pct_cross_fort]
    couleurs = ['lightblue', 'orange']
    
    bars = axes[0,1].bar(categories, pourcentages, color=couleurs, alpha=0.7)
    axes[0,1].set_ylabel('% Crossdétectées')
    axes[0,1].set_title('Taux de crossdétection par niveau OPD')
    
    # Ajouter les valeurs sur les barres
    for bar, pct in zip(bars, pourcentages):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                      f'{pct:.1f}%', ha='center', va='bottom')
    
    # 3. Distribution des vitesses par catégorie et crossdétection
    axes[0,2].hist(df_faible[df_faible['crossdetection_bool_clean'] == False]['speed'].dropna(), 
                   bins=30, alpha=0.5, color='green', label='OPD faible (non cross)')
    axes[0,2].hist(cross_faible['speed'].dropna(), 
                   bins=30, alpha=0.7, color='purple', label='OPD faible (cross)')
    axes[0,2].set_xlabel('Vitesse (µm/s)')
    axes[0,2].set_ylabel('Fréquence')
    axes[0,2].set_title('Vitesses - OPD faible')
    axes[0,2].set_xlim(-8, 8)
    axes[0,2].legend()
    
    # 4. Signal vs vitesse avec crossdétection
    non_cross = df_opd[df_opd['crossdetection_bool_clean'] == False]
    cross_all = df_opd[df_opd['crossdetection_bool_clean'] == True]
    
    axes[1,0].scatter(non_cross['signal_I'], non_cross['speed'], 
                     alpha=0.5, color='green', s=10, label='Non crossdétectées')
    axes[1,0].scatter(cross_all['signal_I'], cross_all['speed'], 
                     alpha=0.8, color='purple', s=20, label='Crossdétectées')
    axes[1,0].axvline(seuil_signal, color='black', linestyle='--', alpha=0.7)
    axes[1,0].set_xlabel('Signal I (OPD)')
    axes[1,0].set_ylabel('Vitesse (µm/s)')
    axes[1,0].set_title('Signal vs Vitesse (OPD)')
    axes[1,0].set_xscale('log')
    axes[1,0].set_ylim(-8, 8)
    axes[1,0].legend()
    
    # 5. Distribution des vitesses - OPD fort
    axes[1,1].hist(df_fort[df_fort['crossdetection_bool_clean'] == False]['speed'].dropna(), 
                   bins=30, alpha=0.5, color='lightgreen', label='OPD fort (non cross)')
    axes[1,1].hist(cross_fort['speed'].dropna(), 
                   bins=30, alpha=0.7, color='green', label='OPD fort (cross)')
    axes[1,1].set_xlabel('Vitesse (µm/s)')
    axes[1,1].set_ylabel('Fréquence')
    axes[1,1].set_title('Vitesses - OPD fort')
    axes[1,1].set_xlim(-8, 8)
    axes[1,1].legend()
    
    # 6. Boxplot comparatif des signaux
    data_box_signal = [
        df_faible[df_faible['crossdetection_bool_clean'] == False]['signal_I'].dropna(),
        cross_faible['signal_I'].dropna(),
        df_fort[df_fort['crossdetection_bool_clean'] == False]['signal_I'].dropna(),
        cross_fort['signal_I'].dropna()
    ]
    
    box_plot = axes[1,2].boxplot(data_box_signal, 
                                labels=['Faible\n(non cross)', 'Faible\n(cross)', 
                                       'Fort\n(non cross)', 'Fort\n(cross)'],
                                patch_artist=True)
    
    # Couleurs des boxplots
    couleurs_box = ['lightgreen', 'green', 'lightgreen', 'green']
    for patch, couleur in zip(box_plot['boxes'], couleurs_box):
        patch.set_facecolor(couleur)
    
    axes[1,2].set_ylabel('Signal I (OPD)')
    axes[1,2].set_title('Distribution des signaux OPD')
    axes[1,2].set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques détaillées
    print(f"\nStatistiques détaillées (OPD uniquement):")
    print(f"{'Catégorie':<20} {'Total':<10} {'Crossdét.':<12} {'%':<8} {'Signal moy.':<12} {'Vitesse moy.':<12}")
    print("-" * 80)
    
    # OPD faible
    signal_moy_faible = df_faible['signal_I'].mean()
    vitesse_moy_faible = df_faible['speed'].mean()
    print(f"{'OPD faible':<20} {total_faible:<10} {nb_cross_faible:<12} {pct_cross_faible:<8.1f} {signal_moy_faible:<12.2f} {vitesse_moy_faible:<12.2f}")
    
    # OPD fort
    signal_moy_fort = df_fort['signal_I'].mean()
    vitesse_moy_fort = df_fort['speed'].mean()
    print(f"{'OPD fort':<20} {total_fort:<10} {nb_cross_fort:<12} {pct_cross_fort:<8.1f} {signal_moy_fort:<12.2f} {vitesse_moy_fort:<12.2f}")
    
    # Stat Mann-Whitney U test
    vitesse_faible = df_faible['speed'].dropna()
    vitesse_fort = df_fort['speed'].dropna()

    u_stat, p_value = mannwhitneyu(vitesse_faible, vitesse_fort, alternative='two-sided')
    print(f"\nTest Mann-Whitney vitesse OPD faible vs OPD fort : U={u_stat:.2f}, p={p_value:.3f}")

    if p_value < 0.05:
        print("Différence significative entre OPD faible et OPD fort (p < 0.05)")
    else:
        print("Pas de différence significative (p ≥ 0.05)")

    return {
        'seuil': seuil_signal,
        'opd_faible': {'total': total_faible, 'cross': nb_cross_faible, 'pct': pct_cross_faible},
        'opd_fort': {'total': total_fort, 'cross': nb_cross_fort, 'pct': pct_cross_fort}
    }

def analyser_crossdetection_par_signal_Fluo(df, seuil_signal=None):
    """
    Analyse les particules crossdétectées selon un seuil de signal_I
    UNIQUEMENT pour les données Fluo (ID_modality = 2)
    
    Parameters:
    - df: DataFrame avec les données
    - seuil_signal: seuil pour distinguer Fluo faible/fort (si None, utilise la médiane)
    """

    # Filtrer pour ne garder que les données Fluo
    if 'ID_modality' not in df.columns:
        print("Colonne 'ID_modality' non trouvée dans le DataFrame")
        return

    df_fluo = df[df['ID_modality'] == 2].copy()

    if len(df_fluo) == 0:
        print("Aucune donnée Fluo trouvée (ID_modality = 2)")
        return

    print(f"Analyse sur {len(df_fluo)} particules Fluo (sur {len(df)} total)")

    if 'signal_I' not in df_fluo.columns:
        print("Colonne 'signal_I' non trouvée dans le DataFrame")
        return
    
    # Catégoriser selon le signal
    df_faible = df_fluo[df_fluo['signal_I'] <= seuil_signal]  # Fluo faible
    df_fort = df_fluo[df_fluo['signal_I'] > seuil_signal]     # Fluo fort

    # Particules crossdétectées dans chaque catégorie
    cross_faible = df_faible[df_faible['crossdetection_bool_clean'] == True]
    cross_fort = df_fort[df_fort['crossdetection_bool_clean'] == True]
    
    # Statistiques
    total_faible = len(df_faible)
    total_fort = len(df_fort)
    nb_cross_faible = len(cross_faible)
    nb_cross_fort = len(cross_fort)
    
    pct_cross_faible = (nb_cross_faible / total_faible * 100) if total_faible > 0 else 0
    pct_cross_fort = (nb_cross_fort / total_fort * 100) if total_fort > 0 else 0
    
    print(f"\nRésultats (Fluo uniquement):")
    print(f"Fluo faible (≤{seuil_signal:.2f}): {nb_cross_faible}/{total_faible} crossdétectées ({pct_cross_faible:.1f}%)")
    print(f"Fluo fort (>{seuil_signal:.2f}): {nb_cross_fort}/{total_fort} crossdétectées ({pct_cross_fort:.1f}%)")

    # Graphiques
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Analyse crossdétection - Fluo uniquement (ID_modality = 2)', fontsize=16)

    # 1. Distribution du signal avec seuil
    axes[0,0].hist(df_fluo['signal_I'].dropna(), bins=50, alpha=0.6, color='gray', label='Toutes (Fluo)')
    axes[0,0].axvline(seuil_signal, color='red', linestyle='--', linewidth=2, label=f'Seuil: {seuil_signal:.2f}')
    axes[0,0].set_xlabel('Signal I (Fluo)')
    axes[0,0].set_ylabel('Fréquence')
    axes[0,0].set_title('Distribution du signal Fluo avec seuil')
    axes[0,0].legend()
    axes[0,0].set_yscale('log')
    
    # 2. Pourcentage de crossdétection par catégorie
    categories = ['Fluo faible', 'Fluo fort']
    pourcentages = [pct_cross_faible, pct_cross_fort]
    couleurs = ['lightblue', 'orange']
    
    bars = axes[0,1].bar(categories, pourcentages, color=couleurs, alpha=0.7)
    axes[0,1].set_ylabel('% Crossdétectées')
    axes[0,1].set_title('Taux de crossdétection par niveau Fluo')
    
    # Ajouter les valeurs sur les barres
    for bar, pct in zip(bars, pourcentages):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                      f'{pct:.1f}%', ha='center', va='bottom')
    
    # 3. Distribution des vitesses par catégorie et crossdétection
    axes[0,2].hist(df_faible[df_faible['crossdetection_bool_clean'] == False]['speed'].dropna(), 
                   bins=30, alpha=0.5, color='lightcoral', label='Fluo faible (non cross)')
    axes[0,2].hist(cross_faible['speed'].dropna(), 
                   bins=30, alpha=0.7, color='red', label='Fluo faible (cross)')
    axes[0,2].set_xlabel('Vitesse (µm/s)')
    axes[0,2].set_ylabel('Fréquence')
    axes[0,2].set_title('Vitesses - Fluo faible')
    axes[0,2].set_xlim(-8, 8)
    axes[0,2].legend()
    
    # 4. Signal vs vitesse avec crossdétection
    non_cross = df_fluo[df_fluo['crossdetection_bool_clean'] == False]
    cross_all = df_fluo[df_fluo['crossdetection_bool_clean'] == True]

    axes[1,0].scatter(non_cross['signal_I'], non_cross['speed'],
                     alpha=0.5, color='orange', s=10, label='Non crossdétectées')
    axes[1,0].scatter(cross_all['signal_I'], cross_all['speed'], 
                     alpha=0.8, color='purple', s=20, label='Crossdétectées')
    axes[1,0].axvline(seuil_signal, color='black', linestyle='--', alpha=0.7)
    axes[1,0].set_xlabel('Signal I (Fluo)')
    axes[1,0].set_ylabel('Vitesse (µm/s)')
    axes[1,0].set_title('Signal vs Vitesse (Fluo)')
    axes[1,0].set_xscale('log')
    axes[1,0].set_ylim(-8, 8)
    axes[1,0].legend()
    
    # 5. Distribution des vitesses - Fluo fort
    axes[1,1].hist(df_fort[df_fort['crossdetection_bool_clean'] == False]['speed'].dropna(),
                   bins=30, alpha=0.5, color='lightcoral', label='Fluo fort (non cross)')
    axes[1,1].hist(cross_fort['speed'].dropna(),
                   bins=30, alpha=0.7, color='red', label='Fluo fort (cross)')
    axes[1,1].set_xlabel('Vitesse (µm/s)')
    axes[1,1].set_ylabel('Fréquence')
    axes[1,1].set_title('Vitesses - Fluo fort')
    axes[1,1].set_xlim(-8, 8)
    axes[1,1].legend()
    
    # 6. Boxplot comparatif des signaux
    data_box_signal = [
        df_faible[df_faible['crossdetection_bool_clean'] == False]['signal_I'].dropna(),
        cross_faible['signal_I'].dropna(),
        df_fort[df_fort['crossdetection_bool_clean'] == False]['signal_I'].dropna(),
        cross_fort['signal_I'].dropna()
    ]
    
    box_plot = axes[1,2].boxplot(data_box_signal, 
                                labels=['Faible\n(non cross)', 'Faible\n(cross)', 
                                       'Fort\n(non cross)', 'Fort\n(cross)'],
                                patch_artist=True)
    
    # Couleurs des boxplots
    couleurs_box = ['lightcoral', 'red', 'lightcoral', 'red']
    for patch, couleur in zip(box_plot['boxes'], couleurs_box):
        patch.set_facecolor(couleur)

    axes[1,2].set_ylabel('Signal I (Fluo)')
    axes[1,2].set_title('Distribution des signaux Fluo')
    axes[1,2].set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques détaillées
    print(f"\nStatistiques détaillées (Fluo uniquement):")
    print(f"{'Catégorie':<20} {'Total':<10} {'Crossdét.':<12} {'%':<8} {'Signal moy.':<12} {'Vitesse moy.':<12}")
    print("-" * 80)

    # Fluo faible
    signal_moy_faible = df_faible['signal_I'].mean()
    vitesse_moy_faible = df_faible['speed'].mean()
    print(f"{'Fluo faible':<20} {total_faible:<10} {nb_cross_faible:<12} {pct_cross_faible:<8.1f} {signal_moy_faible:<12.2f} {vitesse_moy_faible:<12.2f}")

    # Fluo fort
    signal_moy_fort = df_fort['signal_I'].mean()
    vitesse_moy_fort = df_fort['speed'].mean()
    print(f"{'Fluo fort':<20} {total_fort:<10} {nb_cross_fort:<12} {pct_cross_fort:<8.1f} {signal_moy_fort:<12.2f} {vitesse_moy_fort:<12.2f}")

    # Stat Mann-Whitney U test
    vitesse_faible = df_faible['speed'].dropna()
    vitesse_fort = df_fort['speed'].dropna()

    u_stat, p_value = mannwhitneyu(vitesse_faible, vitesse_fort, alternative='two-sided')
    print(f"\nTest Mann-Whitney vitesse Fluo faible vs Fluo fort : U={u_stat:.2f}, p={p_value:.3f}")

    if p_value < 0.05:
        print("Différence significative entre Fluo faible et Fluo fort (p < 0.05)")
    else:
        print("Pas de différence significative (p ≥ 0.05)")

    return {
        'seuil': seuil_signal,
        'fluo_faible': {'total': total_faible, 'cross': nb_cross_faible, 'pct': pct_cross_faible},
        'fluo_fort': {'total': total_fort, 'cross': nb_cross_fort, 'pct': pct_cross_fort}
    }

def seuil_signal_dispersion(df, quantile_signal=0.5, quantile_dispersion=0.5, modalite=None):
    """
    Détermine un seuil pour le signal_I en fonction de la distribution du signal et de la dispersion.
    Le seuil est calculé comme le produit des quantiles choisis pour signal_I et signal_dispersion.
    
    Parameters:
    - df: DataFrame avec les données
    - quantile_signal: quantile à utiliser pour signal_I (0.5 = médiane)
    - quantile_dispersion: quantile à utiliser pour signal_dispersion (0.5 = médiane)
    - modalite: ID_modality à filtrer (1=OPD, 2=Fluorescence, None=toutes les données)
    
    Returns:
    - float: seuil calculé (ratio signal/dispersion)
    """
    
    # Filtrage par modalité
    if modalite is not None:
        df_filtre = df[df['ID_modality'] == modalite].copy()
        modalite_nom = {1: "OPD (Front d'onde)", 2: "Fluorescence"}.get(modalite, f"Modalité {modalite}")
        print(f"Calcul du seuil pour {modalite_nom} ({len(df_filtre)} particules sur {len(df)} total)")
    
    # Supprimer les valeurs NaN pour les calculs
    df_clean = df_filtre[['signal_I', 'signal_dispersion']].dropna()
    
    if len(df_clean) == 0:
        print("Aucune donnée valide après suppression des NaN")
        return None
    
    # Calculer les seuils
    seuil_signal = df_clean['signal_I'].quantile(quantile_signal)
    seuil_dispersion = df_clean['signal_dispersion'].quantile(quantile_dispersion)
    
    # Éviter la division par zéro
    if seuil_dispersion == 0:
        print("Attention : seuil_dispersion = 0, utilisation du seuil_signal directement")
        seuil = seuil_signal
    else:
        seuil = seuil_signal / seuil_dispersion
    
    # Affichage des résultats
    print(f"Seuil signal_I (quantile {quantile_signal}): {seuil_signal:.2f}")
    print(f"Seuil dispersion (quantile {quantile_dispersion}): {seuil_dispersion:.2f}")
    print(f"Seuil ratio signal/dispersion: {seuil:.2f}")

    return seuil

def calculer_duree_suivi(tracks_smooth):
    # Convertir la chaîne en liste
    if isinstance(tracks_smooth, str):
        tracks_smooth = ast.literal_eval(tracks_smooth)
        
    if not isinstance(tracks_smooth, (list, tuple)) or len(tracks_smooth) != 2:
        return np.nan
    
    temps = np.array(tracks_smooth[0])
    if len(temps) < 2:
        return np.nan
        
    return temps[-1] - temps[0]  # Durée totale en secondes

def detecter_demi_tour(tracks_smooth, seuil_vitesse=0.1, seuil_deplacement=0.5):
    """
    Détecte si une particule fait un demi-tour pendant son suivi.
    
    Parameters:
    - tracks_smooth: trajectoire lissée [temps, positions]
    - seuil_vitesse: vitesse minimum pour considérer un mouvement significatif (µm/s)
    - seuil_deplacement: déplacement minimum requis pour valider un demi-tour (µm)
    
    Returns:
    - bool: True si demi-tour détecté, False sinon
    """
    
    # Conversion et validation
    if isinstance(tracks_smooth, str):
        tracks_smooth = ast.literal_eval(tracks_smooth)
    
    if not isinstance(tracks_smooth, (list, tuple)) or len(tracks_smooth) != 2:
        return False
    
    temps, positions = np.array(tracks_smooth[0]), np.array(tracks_smooth[1])
    
    if len(positions) < 3:
        return False
    
    # Calculer vitesses et filtrer par seuil
    vitesses = np.diff(positions) / np.diff(temps)
    vitesses_significatives = vitesses[np.abs(vitesses) > seuil_vitesse]
    
    if len(vitesses_significatives) < 2:
        return False
    
    # Détecter changement de signe (demi-tour potentiel)
    changements_signe = vitesses_significatives[:-1] * vitesses_significatives[1:] < 0
    
    if not np.any(changements_signe):
        return False
    
    # Vérifier le critère de déplacement minimum
    # Calculer la distance totale parcourue
    deplacements = np.abs(np.diff(positions))
    distance_totale = np.sum(deplacements)
    
    # Alternative: vérifier que le déplacement net n'est pas trop petit par rapport au déplacement total
    deplacement_net = abs(positions[-1] - positions[0])
    
    # Un vrai demi-tour implique que la distance totale >> déplacement net
    # ET que la distance totale dépasse le seuil minimum
    if distance_totale >= seuil_deplacement and distance_totale > 2 * deplacement_net:
        return True
    
    return False

def analyser_demi_tours_statistiques(df):
        
    df_normal = df[df['demi_tour'] == False]
    df_demi_tour = df[df['demi_tour'] == True]
    
    print(f"Répartition :")
    print(f"Normal: {len(df_normal)} ({len(df_normal)/len(df)*100:.1f}%)")
    print(f"Demi-tour : {len(df_demi_tour)} ({len(df_demi_tour)/len(df)*100:.1f}%)")
    
    # Comparaison des métriques
    metriques = ['speed', 'duree_suivi', 'nb_points']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    
    # Vitesse vs durée avec couleur selon demi-tour
    axes[0].scatter(df_normal['duree_suivi'], df_normal['speed'], 
                        alpha=0.6, label='Normal', color='grey')
    axes[0].scatter(df_demi_tour['duree_suivi'], df_demi_tour['speed'], 
                        alpha=0.8, label='Demi-tour', color='brown')
    axes[0].set_xlabel('Durée de suivi (frames)')
    axes[0].set_ylabel('Vitesse (µm/s)')
    axes[0].set_title('Vitesse vs Durée de suivi')
    axes[0].set_ylim(-8, 8)
    axes[0].legend()
    
    # Distribution des durées
    axes[1].hist(df_normal['duree_suivi'].dropna(), bins=20, alpha=0.6, 
                    label='Normal', density=True, color='grey')
    axes[1].hist(df_demi_tour['duree_suivi'].dropna(), bins=20, alpha=0.8, 
                    label='Demi-tour', density=True, color='brown')
    axes[1].set_xlabel('Durée de suivi (frames)')
    axes[1].set_ylabel('Densité')
    axes[1].set_title('Distribution des durées de suivi')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Tableau de statistiques
    print(f"\nStatistiques comparatives :")
    print(f"{'Métrique':<15} {'Normal (moy)':<12} {'Demi-tour (moy)':<15} {'Différence':<10}")
    print("-" * 60)
    for metrique in metriques:
        moy_normal = df_normal[metrique].mean()
        moy_demi_tour = df_demi_tour[metrique].mean()
        diff = ((moy_demi_tour - moy_normal) / moy_normal * 100) if moy_normal != 0 else 0
        print(f"{metrique:<15} {moy_normal:<12.2f} {moy_demi_tour:<15.2f} {diff:<10.1f}%")

def analyser_qualite_signal(df, nom_modalite):
        
    # Critères de fiabilité
    critere_1 = df['signal_I'] > 0
    critere_2 = df['signal_sigma'] <= df['signal_I']  # Sigma de l'ordre du signal
    critere_3 = df['duree_suivi'] > 1.0  # Durée suffisante

    # Combinaisons de critères
    fiable_tous = critere_1 & critere_2 & critere_3
    fiable_signal_duree = critere_1 & critere_3

    print(f"\n{nom_modalite.upper()} ({len(df)} particules):")
    print(f"Signal positif: {critere_1.sum()} ({critere_1.mean()*100:.1f}%)")
    print(f"Sigma ≤ Signal: {critere_2.sum()} ({critere_2.mean()*100:.1f}%)")
    print(f"Durée > 1s: {critere_3.sum()} ({critere_3.mean()*100:.1f}%)")
    print(f"Fiable (tous critères): {fiable_tous.sum()} ({fiable_tous.mean()*100:.1f}%)")
    print(f"Fiable (signal + durée): {fiable_signal_duree.sum()} ({fiable_signal_duree.mean()*100:.1f}%)")

    # Statistiques des signaux fiables
    if fiable_tous.sum() > 0:
        df_fiable = df[fiable_tous]
        print(f"Vitesse moyenne (fiable): {df_fiable['speed'].mean():.2f} µm/s")
        print(f"Signal moyen (fiable): {df_fiable['signal_I'].mean():.2f}")

    return fiable_tous

