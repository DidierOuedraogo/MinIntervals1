import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime
import math
from typing import Tuple, List, Dict
import re

# Configuration de la page
st.set_page_config(
    page_title="Analyseur d'Intervalles Minéralisés",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .success-box {
        background: #dcfce7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #16a34a;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    .error-box {
        background: #fef2f2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
        margin: 1rem 0;
    }
    .info-box {
        background: #dbeafe;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding-left: 24px;
        padding-right: 24px;
        background-color: #f8fafc;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ===============================================================================
# FONCTIONS DE CALCUL GÉOLOGIQUE
# ===============================================================================

@st.cache_data
def calculate_distance_to_shear(easting: float, northing: float, elevation: float) -> float:
    """
    Calcule la distance perpendiculaire d'un point à la shear zone planaire
    
    Paramètres shear zone:
    - Strike: N45°E (azimut 045°)
    - Dip: 75° vers SE
    - Centre: E450000, N5550000, Z150m
    """
    # Paramètres de la shear zone
    center_easting = 450000
    center_northing = 5550000
    center_depth = 100  # Profondeur du centre
    strike_azimuth = 45  # Degrés
    dip_angle = 75  # Degrés
    
    # Conversion en radians
    strike_rad = math.radians(strike_azimuth)
    dip_rad = math.radians(dip_angle)
    
    # Vecteur normal au plan de la shear zone
    normal_x = math.cos(strike_rad) * math.sin(dip_rad)
    normal_y = -math.sin(strike_rad) * math.sin(dip_rad)
    normal_z = math.cos(dip_rad)
    
    # Point de référence sur le plan
    ref_x = center_easting
    ref_y = center_northing
    ref_z = 250 - center_depth  # Convention: élévation = 250 - profondeur
    
    # Vecteur du point de référence au point testé
    dx = easting - ref_x
    dy = northing - ref_y
    dz = elevation - ref_z
    
    # Distance perpendiculaire au plan
    distance = abs(dx * normal_x + dy * normal_y + dz * normal_z)
    
    return distance

def generate_grade_from_distance(distance: float) -> float:
    """Génère une teneur basée sur la distance à la shear zone"""
    
    # Grade de base très faible
    base_grade = 0.01 + np.random.random() * 0.05
    
    if distance < 20:  # Zone d'influence
        proximity_factor = max(0, (20 - distance) / 20)
        
        if distance < 5:  # Très proche: hautes teneurs
            grade = 0.5 + np.random.random() * 3.0 * proximity_factor
            if np.random.random() < 0.2:  # 20% de valeurs exceptionnelles
                grade += np.random.random() * 5
        elif distance < 10:  # Proche: teneurs modérées
            grade = 0.2 + np.random.random() * 1.5 * proximity_factor
            if np.random.random() < 0.3:
                grade += np.random.random() * 2
        else:  # Zone d'influence
            grade = 0.1 + np.random.random() * 0.8 * proximity_factor
    else:
        grade = base_grade
    
    # Quelques anomalies dispersées (5%)
    if np.random.random() < 0.05:
        grade = 0.3 + np.random.random() * 1.0
    
    return grade

@st.cache_data
def generate_demo_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Génère des données de démonstration optimisées - 100 forages, 5000+ échantillons"""
    
    # Paramètres de la shear zone
    center_easting = 450000
    center_northing = 5550000
    strike_length = 800
    dip_extent = 200
    strike_azimuth = 45
    center_depth = 100
    
    # Générer 100 forages orientés
    drillholes = []
    for i in range(1, 101):
        # Position optimisée pour intercepter la shear zone
        offset_strike = (np.random.random() - 0.5) * strike_length * 0.8
        offset_dip = (np.random.random() - 0.5) * dip_extent * 0.8
        
        strike_rad = math.radians(strike_azimuth)
        
        surface_easting = center_easting + offset_strike * math.sin(strike_rad) + \
                         (np.random.random() - 0.5) * 100
        surface_northing = center_northing + offset_strike * math.cos(strike_rad) + \
                          (np.random.random() - 0.5) * 100
        
        # Azimut optimal pour croiser la shear zone
        optimal_azimuth = (strike_azimuth + 90) % 360
        azimuth_variation = (np.random.random() - 0.5) * 60
        
        drillholes.append({
            'HoleID': f'DDH-{i:03d}',
            'Easting': round(surface_easting, 2),
            'Northing': round(surface_northing, 2),
            'Elevation': round(250 + (np.random.random() - 0.5) * 50, 2),
            'Azimuth': round(optimal_azimuth + azimuth_variation, 1),
            'Dip': round(-60 + (np.random.random() - 0.5) * 20, 1),
            'Depth': round(200 + np.random.random() * 200, 2)
        })
    
    drillholes_df = pd.DataFrame(drillholes)
    
    # Générer les échantillons
    samples = []
    sample_id = 1
    
    for _, hole in drillholes_df.iterrows():
        depth = 1
        while depth < hole['Depth']:
            # Calculer position 3D de l'échantillon
            azimuth_rad = math.radians(hole['Azimuth'])
            dip_rad = math.radians(abs(hole['Dip']))
            
            sample_easting = hole['Easting'] + depth * math.sin(azimuth_rad) * math.cos(dip_rad)
            sample_northing = hole['Northing'] + depth * math.cos(azimuth_rad) * math.cos(dip_rad)
            sample_elevation = hole['Elevation'] - depth * math.sin(dip_rad)
            
            # Calculer distance à la shear zone
            distance_to_shear = calculate_distance_to_shear(
                sample_easting, sample_northing, sample_elevation
            )
            
            # Générer teneur basée sur la proximité
            grade = generate_grade_from_distance(distance_to_shear)
            
            # Déterminer la zone géologique
            if distance_to_shear < 5:
                zone = 'Shear_Zone_Core'
            elif distance_to_shear < 15:
                zone = 'Shear_Zone_Halo'
            else:
                zone = 'Host_Rock'
            
            samples.append({
                'SampleID': f'{hole["HoleID"]}-{sample_id:04d}',
                'HoleID': hole['HoleID'],
                'From': depth,
                'To': depth + 2,
                'Length': 2.0,
                'Au': round(grade, 3),
                'Easting': round(sample_easting, 2),
                'Northing': round(sample_northing, 2),
                'Elevation': round(sample_elevation, 2),
                'DistanceToShear': round(distance_to_shear, 2),
                'Zone': zone
            })
            
            depth += 2
            sample_id += 1
    
    samples_df = pd.DataFrame(samples)
    
    return samples_df, drillholes_df

# ===============================================================================
# FONCTIONS D'ANALYSE
# ===============================================================================

def identify_potential_intervals(hole_samples: pd.DataFrame, config: Dict) -> List[Dict]:
    """Identifie les intervalles potentiels dans un forage"""
    
    cutoff_grade = config['cutoff_grade']
    max_distance = config['max_distance']
    min_length = config['min_length']
    min_samples = config['min_samples']
    
    intervals = []
    current_interval = None
    
    for _, sample in hole_samples.iterrows():
        grade = sample['Au']
        distance = sample.get('DistanceToShear', 999)
        
        # Critères de sélection
        meets_grade = grade >= cutoff_grade
        meets_distance = distance <= max_distance
        
        if meets_grade and meets_distance:
            if current_interval is None:
                # Commencer un nouvel intervalle
                current_interval = {
                    'start': sample['From'],
                    'end': sample['To'],
                    'samples': [sample],
                    'length': sample['Length'],
                    'grade_sum': grade,
                    'sample_count': 1,
                    'max_grade': grade,
                    'min_grade': grade,
                    'distance_sum': distance,
                    'max_distance': distance,
                    'min_distance': distance
                }
            else:
                # Vérifier la continuité (gap max 4m)
                gap = sample['From'] - current_interval['end']
                if gap <= 4:
                    # Étendre l'intervalle
                    current_interval['end'] = sample['To']
                    current_interval['samples'].append(sample)
                    current_interval['length'] += sample['Length']
                    current_interval['grade_sum'] += grade
                    current_interval['sample_count'] += 1
                    current_interval['max_grade'] = max(current_interval['max_grade'], grade)
                    current_interval['min_grade'] = min(current_interval['min_grade'], grade)
                    current_interval['distance_sum'] += distance
                    current_interval['max_distance'] = max(current_interval['max_distance'], distance)
                    current_interval['min_distance'] = min(current_interval['min_distance'], distance)
                else:
                    # Finaliser l'intervalle précédent
                    if (current_interval['length'] >= min_length and 
                        current_interval['sample_count'] >= min_samples):
                        intervals.append(finalize_interval(current_interval))
                    
                    # Commencer un nouvel intervalle
                    current_interval = {
                        'start': sample['From'],
                        'end': sample['To'],
                        'samples': [sample],
                        'length': sample['Length'],
                        'grade_sum': grade,
                        'sample_count': 1,
                        'max_grade': grade,
                        'min_grade': grade,
                        'distance_sum': distance,
                        'max_distance': distance,
                        'min_distance': distance
                    }
        else:
            # Finaliser l'intervalle en cours si valide
            if (current_interval and 
                current_interval['length'] >= min_length and 
                current_interval['sample_count'] >= min_samples):
                intervals.append(finalize_interval(current_interval))
            current_interval = None
    
    # Traiter le dernier intervalle
    if (current_interval and 
        current_interval['length'] >= min_length and 
        current_interval['sample_count'] >= min_samples):
        intervals.append(finalize_interval(current_interval))
    
    return intervals

def finalize_interval(interval_data: Dict) -> Dict:
    """Finalise un intervalle avec calculs des moyennes"""
    
    return {
        'start': interval_data['start'],
        'end': interval_data['end'],
        'length': interval_data['length'],
        'avg_grade': interval_data['grade_sum'] / interval_data['sample_count'],
        'max_grade': interval_data['max_grade'],
        'min_grade': interval_data['min_grade'],
        'sample_count': interval_data['sample_count'],
        'avg_distance': interval_data['distance_sum'] / interval_data['sample_count'],
        'min_distance': interval_data['min_distance'],
        'max_distance': interval_data['max_distance']
    }

def calculate_mineral_intervals(samples_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Calcule les intervalles minéralisés selon les critères"""
    
    intervals = []
    interval_id = 1
    
    for hole_id, hole_samples in samples_df.groupby('HoleID'):
        hole_samples = hole_samples.sort_values('From')
        
        # Identifier les intervalles potentiels
        potential_intervals = identify_potential_intervals(hole_samples, config)
        
        for interval in potential_intervals:
            intervals.append({
                'IntervalID': interval_id,
                'HoleID': hole_id,
                'From': interval['start'],
                'To': interval['end'],
                'Length': interval['length'],
                'AvgGrade': interval['avg_grade'],
                'MaxGrade': interval['max_grade'],
                'MinGrade': interval['min_grade'],
                'SampleCount': interval['sample_count'],
                'GradeXLength': interval['avg_grade'] * interval['length'],
                'AvgDistance': interval.get('avg_distance', 0),
                'MinDistance': interval.get('min_distance', 0),
                'MaxDistance': interval.get('max_distance', 0),
                'IntervalsBefore': 1,
                'IntervalsAfter': 1,
                'DilutedLength': 0.0,
                'Note': ''
            })
            interval_id += 1
    
    return pd.DataFrame(intervals)

def apply_dilution_constraints(intervals_df: pd.DataFrame, samples_df: pd.DataFrame, 
                             max_dilution: float) -> pd.DataFrame:
    """Applique les contraintes de dilution pour regrouper les intervalles"""
    
    final_intervals = []
    
    # Grouper les intervalles par forage
    for hole_id, hole_intervals in intervals_df.groupby('HoleID'):
        hole_intervals = hole_intervals.sort_values('From')
        
        if len(hole_intervals) == 1:
            # Un seul intervalle: le garder tel quel
            interval = hole_intervals.iloc[0].copy()
            interval['IntervalsBefore'] = 1
            interval['IntervalsAfter'] = 1
            final_intervals.append(interval)
        else:
            # Multiples intervalles: appliquer la logique de regroupement
            merged_interval = apply_merging_logic(
                hole_intervals, samples_df, hole_id, max_dilution
            )
            final_intervals.append(merged_interval)
    
    return pd.DataFrame(final_intervals)

def apply_merging_logic(hole_intervals: pd.DataFrame, samples_df: pd.DataFrame, 
                       hole_id: str, max_dilution: float) -> Dict:
    """Applique la logique de regroupement pour un forage"""
    
    first_interval = hole_intervals.iloc[0]
    last_interval = hole_intervals.iloc[-1]
    
    # Calculer la dilution totale
    total_span = last_interval['To'] - first_interval['From']
    total_mineralized = hole_intervals['Length'].sum()
    dilution = total_span - total_mineralized
    
    intervals_before = len(hole_intervals)
    
    if dilution <= max_dilution:
        # Regrouper avec grade dilué
        hole_samples = samples_df[samples_df['HoleID'] == hole_id]
        span_samples = hole_samples[
            (hole_samples['From'] >= first_interval['From']) & 
            (hole_samples['To'] <= last_interval['To'])
        ]
        
        # Calcul du grade dilué
        diluted_grade = span_samples['Au'].mean()
        diluted_sample_count = len(span_samples)
        
        return {
            'IntervalID': first_interval['IntervalID'],
            'HoleID': hole_id,
            'From': first_interval['From'],
            'To': last_interval['To'],
            'Length': total_span,
            'AvgGrade': diluted_grade,
            'MaxGrade': hole_intervals['MaxGrade'].max(),
            'MinGrade': hole_intervals['MinGrade'].min(),
            'SampleCount': diluted_sample_count,
            'GradeXLength': diluted_grade * total_span,
            'AvgDistance': hole_intervals['AvgDistance'].mean(),
            'MinDistance': hole_intervals['MinDistance'].min(),
            'MaxDistance': hole_intervals['MaxDistance'].max(),
            'IntervalsBefore': intervals_before,
            'IntervalsAfter': 1,
            'DilutedLength': dilution,
            'Note': f'Regroupé ({intervals_before} intervalles)'
        }
    else:
        # Sélectionner le meilleur intervalle
        best_interval = hole_intervals.loc[
            hole_intervals['GradeXLength'].idxmax()
        ].copy()
        
        best_interval['IntervalsBefore'] = intervals_before
        best_interval['IntervalsAfter'] = 1
        best_interval['DilutedLength'] = 0.0
        best_interval['Note'] = f'Meilleur sélectionné (dilution {dilution:.1f}m > {max_dilution}m)'
        
        return best_interval.to_dict()

# ===============================================================================
# FONCTIONS UTILITAIRES
# ===============================================================================

def load_csv_file(uploaded_file):
    """Charge un fichier CSV uploadé"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        return None

def auto_detect_columns(df):
    """Détection automatique des colonnes"""
    if df is None or df.empty:
        return {}
    
    columns = df.columns.tolist()
    detected = {}
    
    patterns = {
        'HoleID': r'hole|drill|ddh|bh|sondage',
        'From': r'from|debut|start|de',
        'To': r'to|fin|end|a',
        'Au': r'au|gold|or',
        'Easting': r'east|x|utm_x',
        'Northing': r'north|y|utm_y',
        'Elevation': r'elev|z|alt|prof'
    }
    
    for field, pattern in patterns.items():
        for col in columns:
            if re.search(pattern, col, re.IGNORECASE):
                detected[field] = col
                break
    
    return detected

def validate_data(df, mapping):
    """Validation des données"""
    errors = []
    
    required_fields = ['HoleID', 'From', 'To', 'Au']
    for field in required_fields:
        if not mapping.get(field):
            errors.append(f"Champ obligatoire manquant: {field}")
    
    if mapping.get('From') and mapping.get('To'):
        try:
            invalid_intervals = df[
                pd.to_numeric(df[mapping['From']], errors='coerce') >= 
                pd.to_numeric(df[mapping['To']], errors='coerce')
            ]
            if len(invalid_intervals) > 0:
                errors.append(f"{len(invalid_intervals)} intervalles invalides (From >= To)")
        except:
            errors.append("Erreur dans les colonnes From/To")
    
    return errors

def calculate_distances_if_needed(df, mapping):
    """Calcule les distances à la shear zone si les coordonnées sont disponibles"""
    
    # Vérifier si les coordonnées sont disponibles
    coord_fields = ['Easting', 'Northing', 'Elevation']
    has_coords = all(mapping.get(field) for field in coord_fields)
    
    if has_coords:
        # Calculer les distances
        st.info("📍 Calcul des distances à la shear zone en cours...")
        
        progress_bar = st.progress(0)
        distances = []
        
        for i, row in df.iterrows():
            try:
                easting = float(row[mapping['Easting']])
                northing = float(row[mapping['Northing']])
                elevation = float(row[mapping['Elevation']])
                
                distance = calculate_distance_to_shear(easting, northing, elevation)
                distances.append(distance)
                
                # Mise à jour de la barre de progression
                if i % 100 == 0:
                    progress_bar.progress(min(i / len(df), 1.0))
                    
            except (ValueError, TypeError):
                distances.append(999)  # Distance par défaut si erreur
        
        progress_bar.progress(1.0)
        df['DistanceToShear'] = distances
        
        progress_bar.empty()
        st.success(f"✅ Distances calculées pour {len(df)} échantillons")
        
        return df, True
    else:
        # Pas de coordonnées - utiliser distance par défaut
        df['DistanceToShear'] = 999
        st.warning("⚠️ Coordonnées manquantes - distance par défaut utilisée (999m)")
        return df, False

# ===============================================================================
# FONCTIONS DE VISUALISATION
# ===============================================================================

def create_grade_distribution_plot(df, mapping):
    """Créer un graphique de distribution des teneurs"""
    
    if not mapping.get('Au'):
        return None
    
    grades = pd.to_numeric(df[mapping['Au']], errors='coerce').dropna()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Distribution des Teneurs Au (g/t)', 
            'Boîte à Moustaches', 
            'Courbe Cumulative', 
            'Teneurs vs Distance Shear Zone'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Histogramme
    fig.add_trace(
        go.Histogram(x=grades, nbinsx=50, name='Distribution', opacity=0.7),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=grades, name='Box Plot'),
        row=1, col=2
    )
    
    # Courbe cumulative
    sorted_grades = np.sort(grades)
    cumulative = np.arange(1, len(sorted_grades) + 1) / len(sorted_grades) * 100
    
    fig.add_trace(
        go.Scatter(x=sorted_grades, y=cumulative, mode='lines', name='Cumulative'),
        row=2, col=1
    )
    
    # Teneurs vs Distance (si disponible)
    if 'DistanceToShear' in df.columns:
        # Échantillonner pour performance
        sample_data = df.sample(min(1000, len(df)))
        fig.add_trace(
            go.Scatter(
                x=sample_data['DistanceToShear'],
                y=pd.to_numeric(sample_data[mapping['Au']], errors='coerce'),
                mode='markers',
                marker=dict(
                    color=pd.to_numeric(sample_data[mapping['Au']], errors='coerce'),
                    colorscale='Viridis',
                    size=4,
                    opacity=0.6
                ),
                name='Au vs Distance'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600,
        title_text="Analyse Statistique des Teneurs en Or",
        showlegend=False
    )
    
    return fig

def create_distance_analysis_plot(df):
    """Créer une analyse de la distribution des distances"""
    
    if 'DistanceToShear' not in df.columns:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Distribution des Distances à la Shear Zone',
            'Teneurs Moyennes par Zone de Distance',
            'Échantillons par Zone Géologique',
            'Corrélation Distance-Teneur'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    distances = df['DistanceToShear'].dropna()
    
    # Histogramme des distances
    fig.add_trace(
        go.Histogram(x=distances, nbinsx=30, name='Distribution Distance', opacity=0.7),
        row=1, col=1
    )
    
    # Teneurs moyennes par zones de distance
    distance_bins = pd.cut(distances, bins=10)
    grade_by_distance = df.groupby(distance_bins)['Au'].agg(['mean', 'count']).reset_index()
    grade_by_distance['distance_center'] = grade_by_distance['DistanceToShear'].apply(lambda x: x.mid)
    
    fig.add_trace(
        go.Bar(
            x=grade_by_distance['distance_center'],
            y=grade_by_distance['mean'],
            name='Teneur Moyenne',
            opacity=0.8
        ),
        row=1, col=2
    )
    
    # Échantillons par zone géologique (si disponible)
    if 'Zone' in df.columns:
        zone_counts = df['Zone'].value_counts()
        fig.add_trace(
            go.Bar(x=zone_counts.index, y=zone_counts.values, name='Échantillons par Zone'),
            row=2, col=1
        )
    
    # Corrélation distance-teneur
    sample_data = df.sample(min(1000, len(df)))
    fig.add_trace(
        go.Scatter(
            x=sample_data['DistanceToShear'],
            y=sample_data['Au'],
            mode='markers',
            marker=dict(
                size=4,
                opacity=0.6,
                color=sample_data['Au'],
                colorscale='Viridis'
            ),
            name='Corrélation'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="Analyse des Distances à la Shear Zone",
        showlegend=False
    )
    
    return fig

def create_results_analysis_plot(results_df):
    """Créer un graphique d'analyse des résultats"""
    
    if results_df is None or results_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Distribution des Longueurs',
            'Distribution des Teneurs Moyennes', 
            'Grade × Longueur (Bubble Chart)',
            'Efficacité du Regroupement'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Distribution des longueurs
    fig.add_trace(
        go.Histogram(x=results_df['Length'], nbinsx=20, name='Longueurs', opacity=0.7),
        row=1, col=1
    )
    
    # Distribution des teneurs
    fig.add_trace(
        go.Histogram(x=results_df['AvgGrade'], nbinsx=20, name='Teneurs', opacity=0.7),
        row=1, col=2
    )
    
    # Grade × Longueur
    fig.add_trace(
        go.Scatter(
            x=results_df['Length'],
            y=results_df['AvgGrade'],
            mode='markers',
            marker=dict(
                size=results_df['GradeXLength'] / 2,
                color=results_df['GradeXLength'],
                colorscale='Plasma',
                opacity=0.7,
                sizemin=4
            ),
            name='G×L',
            text=results_df['HoleID'],
            hovertemplate='Forage: %{text}<br>Longueur: %{x:.1f}m<br>Teneur: %{y:.3f}g/t<br>G×L: %{marker.color:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Efficacité du regroupement
    regrouping_data = results_df.groupby('IntervalsBefore').size().reset_index()
    regrouping_data.columns = ['IntervalsBefore', 'Count']
    
    fig.add_trace(
        go.Bar(
            x=regrouping_data['IntervalsBefore'],
            y=regrouping_data['Count'],
            name='Regroupement',
            hovertemplate='Intervalles Avant: %{x}<br>Nombre de Forages: %{y}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="Synthèse des Résultats d'Analyse",
        showlegend=False
    )
    
    return fig

# ===============================================================================
# INTERFACE PRINCIPALE
# ===============================================================================

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>⛏️ Analyseur d'Intervalles Minéralisés</h1>
        <p><strong>Optimisation géologique pour Leapfrog Geo | Développé par Didier Ouedraogo, P.Geo</strong></p>
        <p><em>Contraintes de distance ET dilution - Un intervalle par forage/shear zone</em></p>
        <p><small>📍 Calcul automatique des distances perpendiculaires à la shear zone (N45°E, 75°SE)</small></p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/3b82f6/ffffff?text=MINERAL+ANALYZER", width=200)
        
        st.markdown("---")
        st.markdown("### 🎯 Fonctionnalités")
        st.markdown("""
        - ✅ Import données CSV
        - ✅ Mapping colonnes intelligent  
        - ✅ Calcul distances automatique
        - ✅ Analyse contraintes dilution
        - ✅ Statistiques avancées
        - ✅ Export multi-format
        """)
        
        st.markdown("---")
        st.markdown("### 📐 Shear Zone")
        st.markdown("""
        - **Strike:** N45°E  
        - **Dip:** 75° vers SE
        - **Centre:** E450000, N5550000, Z150m
        - **Calcul:** Distance perpendiculaire
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍🔬 Développé par")
        st.markdown("**Didier Ouedraogo, P.Geo**")
        st.markdown("Géologue Professionnel")
        st.markdown(f"📅 {datetime.now().strftime('%d/%m/%Y')}")

    # Navigation par onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Import & Aperçu", 
        "🎛️ Mapping & Config", 
        "📈 Statistiques & Graphiques", 
        "🔬 Analyse", 
        "📋 Résultats & Export"
    ])

    # ========================================
    # TAB 1: IMPORT ET APERÇU
    # ========================================
    with tab1:
        st.header("📊 Import et Aperçu des Données")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Options d'Import")
            
            import_option = st.radio(
                "Choisissez votre option:",
                ["📁 Importer fichier CSV", "🚀 Générer données de démonstration"],
                index=0
            )
            
            if import_option == "📁 Importer fichier CSV":
                uploaded_file = st.file_uploader(
                    "Sélectionnez votre fichier CSV",
                    type=['csv'],
                    help="Colonnes requises: HoleID, From, To, Au. Optionnelles: Easting, Northing, Elevation"
                )
                
                if uploaded_file is not None:
                    samples_df = load_csv_file(uploaded_file)
                    if samples_df is not None:
                        st.session_state.samples_df = samples_df
                        st.session_state.drillholes_df = None
                        st.session_state.data_source = "imported"
                        st.success(f"✅ {len(samples_df):,} échantillons importés")
            
            else:
                if st.button("🚀 Générer Données Demo", type="primary", use_container_width=True):
                    with st.spinner("Génération des données de démonstration..."):
                        samples_df, drillholes_df = generate_demo_data()
                        st.session_state.samples_df = samples_df
                        st.session_state.drillholes_df = drillholes_df
                        st.session_state.data_source = "demo"
                        st.success("✅ Données de démonstration générées!")
        
        with col2:
            st.subheader("📋 Format CSV Attendu")
            st.code("""
HoleID,From,To,Au,Easting,Northing,Elevation
DDH-001,0,2,0.15,450100,5550200,245.5
DDH-001,2,4,1.25,450102,5550198,244.2
DDH-001,4,6,0.45,450104,5550196,243.1
            """, language="csv")
            
            st.markdown("---")
            st.markdown("### ℹ️ Informations")
            st.markdown("""
            **Colonnes obligatoires:**
            - HoleID, From, To, Au
            
            **Colonnes optionnelles:**
            - Easting, Northing, Elevation
            - Si présentes → calcul distance
            - Si absentes → distance = 999m
            """)

        # Aperçu des données
        if 'samples_df' in st.session_state:
            st.markdown("---")
            st.subheader("👀 Aperçu des Données")
            
            df = st.session_state.samples_df
            
            # Métriques générales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Échantillons", f"{len(df):,}")
            
            with col2:
                st.metric("📋 Colonnes", f"{len(df.columns)}")
            
            with col3:
                unique_holes = df[df.columns[0]].nunique() if len(df) > 0 else 0
                st.metric("🗺️ Forages", f"{unique_holes}")
            
            with col4:
                data_source = st.session_state.get('data_source', 'unknown')
                source_label = "Demo" if data_source == "demo" else "Importé"
                st.metric("📁 Source", source_label)

            # Informations détaillées
            if st.session_state.get('data_source') == 'demo':
                st.markdown("""
                <div class="info-box">
                    <h4>📊 Données de Démonstration Générées</h4>
                    <ul>
                        <li><strong>100 forages</strong> optimisés pour intercepter la shear zone</li>
                        <li><strong>5000+ échantillons</strong> avec positions 3D réalistes</li>
                        <li><strong>Distances calculées</strong> automatiquement</li>
                        <li><strong>Teneurs variables</strong> selon proximité shear zone</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Aperçu du tableau
            st.subheader("📋 Tableau des Données (50 premières lignes)")
            st.dataframe(df.head(50), use_container_width=True, height=300)
            
            # Informations sur les colonnes
            with st.expander("📊 Informations Détaillées sur les Colonnes"):
                col_info = pd.DataFrame({
                    'Colonne': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count(),
                    'Null': df.isnull().sum(),
                    'Unique': [df[col].nunique() for col in df.columns],
                    'Exemples': [', '.join(map(str, df[col].dropna().unique()[:3])) for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)

    # ========================================
    # TAB 2: MAPPING ET CONFIGURATION
    # ========================================
    with tab2:
        st.header("🎛️ Mapping des Colonnes et Configuration")
        
        if 'samples_df' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord importer des données dans l'onglet 'Import & Aperçu'")
            return
        
        df = st.session_state.samples_df
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔧 Mapping des Colonnes")
            
            # Auto-détection
            if st.button("🔍 Auto-détection des Colonnes"):
                auto_mapping = auto_detect_columns(df)
                for key, value in auto_mapping.items():
                    st.session_state[f'mapping_{key}'] = value
                st.success("Détection automatique effectuée!")
            
            # Configuration manuelle
            available_columns = [''] + df.columns.tolist()
            
            mapping = {}
            field_definitions = {
                'HoleID': ('Identifiant Forage*', 'Identifiant unique du forage'),
                'From': ('Profondeur Début*', 'Profondeur de début en mètres'),
                'To': ('Profondeur Fin*', 'Profondeur de fin en mètres'),
                'Au': ('Teneur Or*', 'Teneur en or (g/t)'),
                'Easting': ('Coordonnée Est', 'Coordonnée X pour calcul distance'),
                'Northing': ('Coordonnée Nord', 'Coordonnée Y pour calcul distance'),
                'Elevation': ('Élévation', 'Élévation Z pour calcul distance')
            }
            
            for field, (label, description) in field_definitions.items():
                mapping[field] = st.selectbox(
                    f"{label}",
                    available_columns,
                    key=f'mapping_{field}',
                    help=description
                )
            
            st.session_state.column_mapping = mapping
            
            # Validation
            errors = validate_data(df, mapping)
            
            if errors:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error("❌ Erreurs de validation:")
                for error in errors:
                    st.write(f"• {error}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("✅ Mapping validé avec succès!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Calculer les distances si nécessaire
                if st.button("📍 Calculer Distances à la Shear Zone"):
                    df_with_distances, has_coords = calculate_distances_if_needed(df, mapping)
                    st.session_state.samples_df = df_with_distances
                    st.session_state.has_coordinates = has_coords

        with col2:
            st.subheader("⚙️ Configuration des Paramètres")
            
            # Paramètres principaux
            config = {}
            config['cutoff_grade'] = st.slider(
                "💎 Teneur de Coupure (g/t Au)",
                min_value=0.1, max_value=5.0, value=0.5, step=0.1,
                help="Teneur minimum pour inclusion"
            )
            
            config['min_length'] = st.slider(
                "📏 Longueur Minimale (m)",
                min_value=0.5, max_value=20.0, value=2.0, step=0.5,
                help="Longueur minimum de l'intervalle"
            )
            
            config['min_samples'] = st.slider(
                "🔢 Échantillons Minimum",
                min_value=1, max_value=10, value=3,
                help="Nombre minimum d'échantillons"
            )
            
            st.markdown("#### 🔴 Contraintes Avancées")
            
            config['max_distance'] = st.slider(
                "⭐ Distance Max Shear Zone (m)",
                min_value=1.0, max_value=50.0, value=10.0, step=1.0,
                help="Distance maximum à la shear zone"
            )
            
            config['max_dilution'] = st.slider(
                "🔥 Dilution Maximale (m)",
                min_value=1.0, max_value=50.0, value=10.0, step=1.0,
                help="Dilution maximum permise"
            )
            
            st.session_state.analysis_config = config
            
            # Résumé de la configuration
            st.markdown("#### 📋 Résumé Configuration")
            config_display = {
                "Teneur de coupure": f"{config['cutoff_grade']} g/t",
                "Longueur minimale": f"{config['min_length']} m",
                "Échantillons minimum": f"{config['min_samples']}",
                "Distance max shear": f"{config['max_distance']} m",
                "Dilution maximale": f"{config['max_dilution']} m"
            }
            
            for key, value in config_display.items():
                st.write(f"**{key}:** {value}")

    # ========================================
    # TAB 3: STATISTIQUES ET GRAPHIQUES
    # ========================================
    with tab3:
        st.header("📈 Statistiques et Graphiques")
        
        if 'samples_df' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord importer des données")
            return
        
        if 'column_mapping' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord configurer le mapping des colonnes")
            return
        
        df = st.session_state.samples_df
        mapping = st.session_state.column_mapping
        
        # Statistiques générales
        st.subheader("📊 Statistiques Générales")
        
        if mapping.get('Au'):
            grades = pd.to_numeric(df[mapping['Au']], errors='coerce').dropna()
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("📊 Échantillons", f"{len(grades):,}")
            
            with col2:
                st.metric("💰 Teneur Min", f"{grades.min():.3f} g/t")
            
            with col3:
                st.metric("📈 Teneur Moy", f"{grades.mean():.3f} g/t")
            
            with col4:
                st.metric("📊 Médiane", f"{grades.median():.3f} g/t")
            
            with col5:
                st.metric("🔺 P95", f"{grades.quantile(0.95):.3f} g/t")
            
            with col6:
                st.metric("🎯 Teneur Max", f"{grades.max():.3f} g/t")
            
            # Statistiques par seuils
            st.markdown("---")
            st.subheader("🎯 Statistiques par Seuils de Teneur")
            
            thresholds = [0.5, 1.0, 2.0, 5.0]
            threshold_stats = []
            
            for threshold in thresholds:
                count = len(grades[grades >= threshold])
                percentage = (count / len(grades)) * 100 if len(grades) > 0 else 0
                threshold_stats.append({
                    'Seuil (g/t)': f"≥ {threshold}",
                    'Échantillons': count,
                    'Pourcentage': f"{percentage:.1f}%"
                })
            
            threshold_df = pd.DataFrame(threshold_stats)
            st.dataframe(threshold_df, use_container_width=True)
        
        # Graphiques de distribution
        st.markdown("---")
        st.subheader("📈 Analyse Graphique des Teneurs")
        
        fig_dist = create_grade_distribution_plot(df, mapping)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Analyse des distances (si disponible)
        if 'DistanceToShear' in df.columns:
            st.markdown("---")
            st.subheader("📍 Analyse des Distances à la Shear Zone")
            
            distances = df['DistanceToShear'].dropna()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📍 Distance Min", f"{distances.min():.1f} m")
            
            with col2:
                st.metric("📊 Distance Moy", f"{distances.mean():.1f} m")
            
            with col3:
                st.metric("📈 Médiane", f"{distances.median():.1f} m")
            
            with col4:
                st.metric("🎯 Distance Max", f"{distances.max():.1f} m")
            
            # Statistiques par zones de distance
            st.markdown("#### 🏷️ Répartition par Zones de Distance")
            
            zone_stats = []
            zones = [
                ("Très proche", 0, 5),
                ("Proche", 5, 15),
                ("Modéré", 15, 30),
                ("Éloigné", 30, float('inf'))
            ]
            
            for zone_name, min_dist, max_dist in zones:
                if max_dist == float('inf'):
                    zone_samples = distances[distances >= min_dist]
                else:
                    zone_samples = distances[(distances >= min_dist) & (distances < max_dist)]
                
                count = len(zone_samples)
                percentage = (count / len(distances)) * 100 if len(distances) > 0 else 0
                
                # Teneur moyenne dans cette zone
                if mapping.get('Au'):
                    zone_mask = df['DistanceToShear'].between(min_dist, max_dist if max_dist != float('inf') else df['DistanceToShear'].max())
                    zone_grades = pd.to_numeric(df[zone_mask][mapping['Au']], errors='coerce').dropna()
                    avg_grade = zone_grades.mean() if len(zone_grades) > 0 else 0
                else:
                    avg_grade = 0
                
                zone_stats.append({
                    'Zone': zone_name,
                    'Distance (m)': f"{min_dist}-{max_dist if max_dist != float('inf') else '∞'}",
                    'Échantillons': count,
                    'Pourcentage': f"{percentage:.1f}%",
                    'Teneur Moy (g/t)': f"{avg_grade:.3f}"
                })
            
            zone_df = pd.DataFrame(zone_stats)
            st.dataframe(zone_df, use_container_width=True)
            
            # Graphique des distances
            fig_distance = create_distance_analysis_plot(df)
            if fig_distance:
                st.plotly_chart(fig_distance, use_container_width=True)

    # ========================================
    # TAB 4: ANALYSE
    # ========================================
    with tab4:
        st.header("🔬 Analyse des Intervalles Minéralisés")
        
        if 'samples_df' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord importer des données")
            return
        
        if 'column_mapping' not in st.session_state or 'analysis_config' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord configurer le mapping et les paramètres")
            return
        
        df = st.session_state.samples_df
        mapping = st.session_state.column_mapping
        config = st.session_state.analysis_config
        
        # Vérifier la validation
        errors = validate_data(df, mapping)
        if errors:
            st.error("❌ Erreurs de validation. Veuillez corriger le mapping.")
            for error in errors:
                st.write(f"• {error}")
            return
        
        # Afficher la configuration active
        st.subheader("📋 Configuration Active")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("💎 Teneur Coupure", f"{config['cutoff_grade']} g/t")
        with col2:
            st.metric("📏 Long. Min", f"{config['min_length']} m")
        with col3:
            st.metric("🔢 Éch. Min", f"{config['min_samples']}")
        with col4:
            st.metric("⭐ Dist. Max", f"{config['max_distance']} m")
        with col5:
            st.metric("🔥 Dilution Max", f"{config['max_dilution']} m")
        
        # Statistiques pré-analyse
        st.subheader("📊 Statistiques Pré-Analyse")
        
        # Préparer les données d'analyse
        analysis_df = df.copy()
        
        # S'assurer que les distances sont calculées
        if 'DistanceToShear' not in analysis_df.columns:
            if all(mapping.get(field) for field in ['Easting', 'Northing', 'Elevation']):
                analysis_df, _ = calculate_distances_if_needed(analysis_df, mapping)
                st.session_state.samples_df = analysis_df
            else:
                analysis_df['DistanceToShear'] = 999
        
        # Renommer les colonnes selon le mapping
        column_rename = {v: k for k, v in mapping.items() if v}
        analysis_df = analysis_df.rename(columns=column_rename)
        
        # Convertir en numérique
        numeric_cols = ['From', 'To', 'Au']
        for col in numeric_cols:
            if col in analysis_df.columns:
                analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
        
        if 'Au' in analysis_df.columns:
            grades = analysis_df['Au'].dropna()
            above_cutoff = len(grades[grades >= config['cutoff_grade']])
            
            if 'DistanceToShear' in analysis_df.columns:
                close_to_shear = len(analysis_df[analysis_df['DistanceToShear'] <= config['max_distance']])
                valid_samples = len(analysis_df[
                    (analysis_df['Au'] >= config['cutoff_grade']) & 
                    (analysis_df['DistanceToShear'] <= config['max_distance'])
                ])
            else:
                close_to_shear = len(analysis_df)
                valid_samples = above_cutoff
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📈 Total Échantillons", f"{len(grades):,}")
            with col2:
                st.metric("💰 > Teneur Coupure", f"{above_cutoff:,}")
            with col3:
                st.metric("📍 Proches Shear Zone", f"{close_to_shear:,}")
            with col4:
                st.metric("✅ Critères Combinés", f"{valid_samples:,}")
        
        # Bouton d'analyse
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Lancer l'Analyse Complète", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    try:
                        # Barre de progression
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("📊 Préparation des données...")
                        progress_bar.progress(20)
                        
                        status_text.text("🔍 Identification des intervalles potentiels...")
                        progress_bar.progress(40)
                        
                        # Calculer les intervalles potentiels
                        potential_intervals = calculate_mineral_intervals(analysis_df, config)
                        
                        status_text.text("🔥 Application des contraintes de dilution...")
                        progress_bar.progress(60)
                        
                        # Appliquer les contraintes de dilution
                        final_intervals = apply_dilution_constraints(
                            potential_intervals, 
                            analysis_df, 
                            config['max_dilution']
                        )
                        
                        status_text.text("✅ Finalisation des résultats...")
                        progress_bar.progress(80)
                        
                        # Validation finale
                        validated_intervals = final_intervals[
                            (final_intervals['Length'] >= config['min_length']) &
                            (final_intervals['AvgGrade'] >= config['cutoff_grade']) &
                            (final_intervals['SampleCount'] >= config['min_samples'])
                        ].copy()
                        
                        # Réindexer
                        validated_intervals.reset_index(drop=True, inplace=True)
                        validated_intervals['IntervalID'] = range(1, len(validated_intervals) + 1)
                        
                        st.session_state.analysis_results = validated_intervals
                        
                        progress_bar.progress(100)
                        status_text.text("🎉 Analyse terminée!")
                        
                        # Nettoyer l'interface
                        import time
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
                        
                        if len(validated_intervals) > 0:
                            st.success(f"✅ Analyse terminée! {len(validated_intervals)} intervalles trouvés.")
                        else:
                            st.warning("⚠️ Aucun intervalle ne respecte les contraintes définies.")
                            
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()

        # Afficher les résultats si disponibles
        if 'analysis_results' in st.session_state:
            results_df = st.session_state.analysis_results
            
            if not results_df.empty:
                st.markdown("---")
                st.subheader("🎉 Résultats de l'Analyse")
                
                # Métriques de synthèse
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 Intervalles", f"{len(results_df)}")
                
                with col2:
                    avg_grade = results_df['AvgGrade'].mean()
                    st.metric("📊 Teneur Moy.", f"{avg_grade:.3f} g/t")
                
                with col3:
                    total_length = results_df['Length'].sum()
                    st.metric("📏 Long. Totale", f"{total_length:.1f} m")
                
                with col4:
                    total_metal = results_df['GradeXLength'].sum()
                    st.metric("💰 Grade×Long.", f"{total_metal:.1f}")
                
                # Tableau des résultats
                st.subheader("📋 Détail des Intervalles")
                st.dataframe(
                    results_df.round(3),
                    use_container_width=True,
                    height=300
                )
                
                # Graphique d'analyse des résultats
                fig_results = create_results_analysis_plot(results_df)
                if fig_results:
                    st.plotly_chart(fig_results, use_container_width=True)

    # ========================================
    # TAB 5: RÉSULTATS ET EXPORT
    # ========================================
    with tab5:
        st.header("📋 Résultats et Export")
        
        if 'analysis_results' not in st.session_state:
            st.warning("⚠️ Aucune analyse effectuée. Veuillez lancer l'analyse dans l'onglet précédent.")
            return
        
        results_df = st.session_state.analysis_results
        
        if results_df.empty:
            st.warning("⚠️ Aucun intervalle trouvé avec les critères actuels.")
            return
        
        # Métriques principales
        st.subheader("📊 Métriques Principales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Intervalles Totaux",
                f"{len(results_df)}",
                delta="Un par forage"
            )
        
        with col2:
            avg_grade = results_df['AvgGrade'].mean()
            max_grade = results_df['MaxGrade'].max()
            st.metric(
                "📊 Teneur Moyenne",
                f"{avg_grade:.3f} g/t",
                delta=f"Max: {max_grade:.3f}"
            )
        
        with col3:
            total_length = results_df['Length'].sum()
            avg_length = results_df['Length'].mean()
            st.metric(
                "📏 Longueur Totale",
                f"{total_length:.1f} m",
                delta=f"Moy: {avg_length:.1f}m"
            )
        
        with col4:
            total_metal = results_df['GradeXLength'].sum()
            st.metric(
                "💰 Métal Total (G×L)",
                f"{total_metal:.1f}",
                delta="Potentiel économique"
            )
        
        # Options d'export
        st.markdown("---")
        st.subheader("💾 Options d'Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export CSV standard
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📊 Télécharger CSV",
                data=csv_data,
                file_name=f"intervalles_mineralises_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export Excel avec métadonnées
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                results_df.to_excel(writer, sheet_name='Intervalles', index=False)
                
                # Feuille métadonnées
                if 'analysis_config' in st.session_state:
                    config = st.session_state.analysis_config
                    metadata = pd.DataFrame({
                        'Paramètre': [
                            'Date Analyse',
                            'Teneur Coupure (g/t)',
                            'Longueur Min (m)',
                            'Échantillons Min',
                            'Distance Max (m)',
                            'Dilution Max (m)',
                            'Intervalles Trouvés',
                            'Shear Zone Strike',
                            'Shear Zone Dip'
                        ],
                        'Valeur': [
                            datetime.now().strftime('%Y-%m-%d %H:%M'),
                            config['cutoff_grade'],
                            config['min_length'],
                            config['min_samples'],
                            config['max_distance'],
                            config['max_dilution'],
                            len(results_df),
                            'N45°E',
                            '75° SE'
                        ]
                    })
                    metadata.to_excel(writer, sheet_name='Métadonnées', index=False)
            
            st.download_button(
                label="📋 Télécharger Excel",
                data=buffer.getvalue(),
                file_name=f"intervalles_mineralises_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            # Export format Leapfrog
            leapfrog_data = results_df.copy()
            leapfrog_data['CompositeID'] = leapfrog_data['IntervalID']
            leapfrog_data['Grade'] = leapfrog_data['AvgGrade']
            leapfrog_data['Thickness'] = leapfrog_data['Length']
            leapfrog_data['Quality'] = 'High'
            leapfrog_data['AnalysisDate'] = datetime.now().strftime('%Y-%m-%d')
            leapfrog_data['Method'] = 'Dilution_Constrained'
            
            leapfrog_csv = leapfrog_data[[
                'CompositeID', 'HoleID', 'From', 'To', 'Thickness', 
                'Grade', 'Quality', 'AnalysisDate', 'Method'
            ]].to_csv(index=False)
            
            st.download_button(
                label="🗺️ Export Leapfrog",
                data=leapfrog_csv,
                file_name=f"leapfrog_intervals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Tableau détaillé final avec filtres
        st.markdown("---")
        st.subheader("📋 Tableau Détaillé avec Filtres")
        
        # Filtres
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_grade_filter = st.slider(
                "Teneur minimum:",
                min_value=0.0,
                max_value=float(results_df['AvgGrade'].max()),
                value=0.0,
                step=0.1
            )
        
        with col2:
            min_length_filter = st.slider(
                "Longueur minimum:",
                min_value=0.0,
                max_value=float(results_df['Length'].max()),
                value=0.0,
                step=0.5
            )
        
        with col3:
            holes_filter = st.multiselect(
                "Forages à afficher:",
                options=results_df['HoleID'].unique(),
                default=results_df['HoleID'].unique()
            )
        
        # Appliquer les filtres
        filtered_df = results_df[
            (results_df['AvgGrade'] >= min_grade_filter) &
            (results_df['Length'] >= min_length_filter) &
            (results_df['HoleID'].isin(holes_filter))
        ]
        
        st.dataframe(filtered_df.round(3), use_container_width=True, height=400)
        
        # Statistiques finales
        st.markdown("---")
        st.subheader("📊 Statistiques Finales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Distribution par Qualité")
            
            excellent = len(filtered_df[filtered_df['GradeXLength'] >= 10])
            bon = len(filtered_df[(filtered_df['GradeXLength'] >= 5) & (filtered_df['GradeXLength'] < 10)])
            modere = len(filtered_df[filtered_df['GradeXLength'] < 5])
            
            quality_data = pd.DataFrame({
                'Qualité': ['Excellent (G×L≥10)', 'Bon (5≤G×L<10)', 'Modéré (G×L<5)'],
                'Nombre': [excellent, bon, modere],
                'Pourcentage': [
                    f"{excellent/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%",
                    f"{bon/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%",
                    f"{modere/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%"
                ]
            })
            
            st.dataframe(quality_data, use_container_width=True)
        
        with col2:
            st.markdown("#### 📏 Distribution par Longueur")
            
            court = len(filtered_df[filtered_df['Length'] < 5])
            moyen = len(filtered_df[(filtered_df['Length'] >= 5) & (filtered_df['Length'] < 10)])
            long = len(filtered_df[filtered_df['Length'] >= 10])
            
            length_data = pd.DataFrame({
                'Longueur': ['Court (<5m)', 'Moyen (5-10m)', 'Long (≥10m)'],
                'Nombre': [court, moyen, long],
                'Pourcentage': [
                    f"{court/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%",
                    f"{moyen/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%",
                    f"{long/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%"
                ]
            })
            
            st.dataframe(length_data, use_container_width=True)
        
        # Information sur l'efficacité du regroupement
        if 'IntervalsBefore' in results_df.columns:
            st.markdown("---")
            st.subheader("🔄 Efficacité du Regroupement")
            
            regrouped = len(results_df[results_df['IntervalsBefore'] > 1])
            single = len(results_df[results_df['IntervalsBefore'] == 1])
            
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 Résumé du Regroupement</h4>
                <ul>
                    <li><strong>{single} forages</strong> avec un seul intervalle</li>
                    <li><strong>{regrouped} forages</strong> avec intervalles regroupés</li>
                    <li><strong>Taux de regroupement:</strong> {regrouped/len(results_df)*100:.1f}%</li>
                    <li><strong>Règle appliquée:</strong> Un intervalle par forage/shear zone</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()