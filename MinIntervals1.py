import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import math
from typing import Tuple, List, Dict
import json

# Configuration de la page
st.set_page_config(
    page_title="Analyseur d'Intervalles Minéralisés 3D",
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
# FONCTIONS UTILITAIRES
# ===============================================================================

@st.cache_data
def calculate_distance_to_shear(easting: float, northing: float, elevation: float) -> float:
    """Calcule la distance perpendiculaire à la shear zone planaire"""
    # Paramètres de la shear zone
    center_easting = 450000
    center_northing = 5550000
    center_depth = 100
    strike_azimuth = 45  # N45°E
    dip_angle = 75  # 75° SE
    
    # Conversion en radians
    strike_rad = math.radians(strike_azimuth)
    dip_rad = math.radians(dip_angle)
    
    # Vecteur normal au plan
    normal_x = math.cos(strike_rad) * math.sin(dip_rad)
    normal_y = -math.sin(strike_rad) * math.sin(dip_rad)
    normal_z = math.cos(dip_rad)
    
    # Point de référence
    ref_x = center_easting
    ref_y = center_northing
    ref_z = 250 - center_depth
    
    # Distance perpendiculaire
    dx = easting - ref_x
    dy = northing - ref_y
    dz = elevation - ref_z
    
    distance = abs(dx * normal_x + dy * normal_y + dz * normal_z)
    return distance

@st.cache_data
def generate_demo_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Génère des données de démonstration avec positions 3D réalistes"""
    
    samples = []
    drillholes = []
    holes = ['DDH-001', 'DDH-002', 'DDH-003', 'DDH-004', 'DDH-005']
    
    for hole_idx, hole in enumerate(holes):
        # Position de surface du forage
        surface_x = 450000 + (hole_idx - 2) * 100 + (np.random.random() - 0.5) * 50
        surface_y = 5550000 + (hole_idx - 2) * 80 + (np.random.random() - 0.5) * 50
        surface_z = 250
        
        # Orientation optimisée pour croiser la shear zone
        azimuth = 135 + (np.random.random() - 0.5) * 30  # SE direction
        dip = -60 + (np.random.random() - 0.5) * 20  # Plongeant
        depth_total = 150
        
        drillholes.append({
            'HoleID': hole,
            'Easting': surface_x,
            'Northing': surface_y,
            'Elevation': surface_z,
            'Azimuth': azimuth,
            'Dip': dip,
            'TotalDepth': depth_total
        })
        
        azimuth_rad = math.radians(azimuth)
        dip_rad = math.radians(abs(dip))
        
        sample_id = 1
        for depth in range(0, depth_total, 2):
            # Position 3D de l'échantillon
            x = surface_x + depth * math.sin(azimuth_rad) * math.cos(dip_rad)
            y = surface_y + depth * math.cos(azimuth_rad) * math.cos(dip_rad)
            z = surface_z - depth * math.sin(dip_rad)
            
            # Distance à la shear zone
            distance = calculate_distance_to_shear(x, y, z)
            
            # Génération de teneur basée sur la distance
            grade = generate_grade_from_distance(distance)
            
            # Zone géologique
            if distance < 5:
                zone = 'Shear_Zone_Core'
            elif distance < 15:
                zone = 'Shear_Zone_Halo'
            else:
                zone = 'Host_Rock'
            
            samples.append({
                'SampleID': f'{hole}-{sample_id:03d}',
                'HoleID': hole,
                'From': depth,
                'To': depth + 2,
                'Length': 2.0,
                'Au': round(grade, 3),
                'Easting': round(x, 2),
                'Northing': round(y, 2),
                'Elevation': round(z, 2),
                'DistanceToShear': round(distance, 2),
                'Zone': zone,
                'Azimuth': round(azimuth, 1),
                'Dip': round(dip, 1)
            })
            sample_id += 1
    
    return pd.DataFrame(samples), pd.DataFrame(drillholes)

def generate_grade_from_distance(distance: float) -> float:
    """Génère une teneur basée sur la distance à la shear zone"""
    base_grade = 0.05 + np.random.random() * 0.1
    
    if distance < 20:
        proximity_factor = max(0, (20 - distance) / 20)
        
        if distance < 5:
            grade = 0.8 + np.random.random() * 3.0 * proximity_factor
            if np.random.random() < 0.2:
                grade += np.random.random() * 5
        elif distance < 10:
            grade = 0.3 + np.random.random() * 1.5 * proximity_factor
            if np.random.random() < 0.3:
                grade += np.random.random() * 2
        else:
            grade = 0.1 + np.random.random() * 0.8 * proximity_factor
    else:
        grade = base_grade
    
    # Quelques anomalies dispersées
    if np.random.random() < 0.05:
        grade = 0.5 + np.random.random() * 2.0
    
    return grade

@st.cache_data
def generate_shear_zone_mesh():
    """Génère le mesh de la shear zone"""
    mesh_points = []
    center_x = 450000
    center_y = 5550000
    center_z = 150
    strike = 45
    dip = 75
    
    strike_rad = math.radians(strike)
    dip_rad = math.radians(dip)
    
    # Générer des points sur le plan de la shear zone
    for i in range(-400, 401, 20):
        for j in range(0, 201, 10):
            # Position sur le plan local
            x = center_x + i * math.sin(strike_rad) + j * math.cos(strike_rad) * math.cos(dip_rad)
            y = center_y + i * math.cos(strike_rad) - j * math.sin(strike_rad) * math.cos(dip_rad)
            z = center_z - j * math.sin(dip_rad)
            
            mesh_points.append({
                'X': x,
                'Y': y, 
                'Z': z,
                'Type': 'Shear_Zone'
            })
    
    return pd.DataFrame(mesh_points)

def load_csv_file(uploaded_file):
    """Charge un fichier CSV uploadé"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement: {str(e)}")
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

def run_mineral_analysis(df, mapping, config):
    """Exécute l'analyse des intervalles minéralisés"""
    
    # Preparation des données
    analysis_df = df.copy()
    
    # Renommer les colonnes selon le mapping
    column_rename = {v: k for k, v in mapping.items() if v}
    analysis_df = analysis_df.rename(columns=column_rename)
    
    # Convertir en numérique
    numeric_cols = ['From', 'To', 'Au']
    if 'Easting' in analysis_df.columns:
        numeric_cols.extend(['Easting', 'Northing', 'Elevation'])
    
    for col in numeric_cols:
        if col in analysis_df.columns:
            analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
    
    # Calculer les distances si les coordonnées sont disponibles
    if all(col in analysis_df.columns for col in ['Easting', 'Northing', 'Elevation']):
        analysis_df['DistanceToShear'] = analysis_df.apply(
            lambda row: calculate_distance_to_shear(
                row['Easting'], row['Northing'], row['Elevation']
            ), axis=1
        )
    else:
        analysis_df['DistanceToShear'] = 999  # Distance par défaut
    
    # Filtrer selon les critères
    valid_samples = analysis_df[
        (analysis_df['Au'] >= config['cutoff_grade']) &
        (analysis_df['DistanceToShear'] <= config['max_distance'])
    ]
    
    # Grouper par forage et créer les intervalles
    intervals = []
    interval_id = 1
    
    for hole_id, hole_samples in valid_samples.groupby('HoleID'):
        hole_samples = hole_samples.sort_values('From')
        
        if len(hole_samples) >= config['min_samples']:
            # Créer un intervalle continu
            from_depth = hole_samples['From'].min()
            to_depth = hole_samples['To'].max()
            length = to_depth - from_depth
            
            if length >= config['min_length']:
                avg_grade = hole_samples['Au'].mean()
                max_grade = hole_samples['Au'].max()
                sample_count = len(hole_samples)
                
                intervals.append({
                    'IntervalID': interval_id,
                    'HoleID': hole_id,
                    'From': from_depth,
                    'To': to_depth,
                    'Length': round(length, 2),
                    'AvgGrade': round(avg_grade, 3),
                    'MaxGrade': round(max_grade, 3),
                    'SampleCount': sample_count,
                    'GradeXLength': round(avg_grade * length, 2),
                    'AvgDistance': round(hole_samples['DistanceToShear'].mean(), 2)
                })
                interval_id += 1
    
    return pd.DataFrame(intervals)

# ===============================================================================
# VISUALISATIONS 3D
# ===============================================================================

def create_3d_scatter_plot(df, mapping, color_by='grade', show_mesh=True, intervals_df=None):
    """Créer un graphique 3D avec Plotly"""
    
    if not all(col in mapping and mapping[col] for col in ['Easting', 'Northing', 'Elevation', 'Au']):
        st.warning("Coordonnées 3D manquantes pour la visualisation")
        return None
    
    # Préparer les données
    plot_df = df.copy()
    
    # Renommer les colonnes
    plot_df = plot_df.rename(columns={
        mapping['Easting']: 'X',
        mapping['Northing']: 'Y', 
        mapping['Elevation']: 'Z',
        mapping['Au']: 'Grade',
        mapping['HoleID']: 'Hole'
    })
    
    # Convertir en numérique
    for col in ['X', 'Y', 'Z', 'Grade']:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    
    # Supprimer les NaN
    plot_df = plot_df.dropna(subset=['X', 'Y', 'Z', 'Grade'])
    
    if plot_df.empty:
        st.warning("Aucune donnée valide pour la visualisation 3D")
        return None
    
    # Créer la figure
    fig = go.Figure()
    
    # Ajouter le mesh de la shear zone si demandé
    if show_mesh:
        mesh_df = generate_shear_zone_mesh()
        
        # Créer une surface mesh
        fig.add_trace(go.Scatter3d(
            x=mesh_df['X'],
            y=mesh_df['Y'],
            z=mesh_df['Z'],
            mode='markers',
            marker=dict(
                size=2,
                color='orange',
                opacity=0.3
            ),
            name='Shear Zone',
            hovertemplate='Shear Zone<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
        ))
    
    # Déterminer la couleur et la taille des points
    if color_by == 'grade':
        colors = plot_df['Grade']
        colorscale = 'Viridis'
        colorbar_title = 'Teneur Au (g/t)'
    elif color_by == 'distance' and 'DistanceToShear' in plot_df.columns:
        colors = plot_df['DistanceToShear']
        colorscale = 'RdBu_r'
        colorbar_title = 'Distance Shear (m)'
    else:
        # Couleur par forage
        unique_holes = plot_df['Hole'].unique()
        color_map = {hole: i for i, hole in enumerate(unique_holes)}
        colors = plot_df['Hole'].map(color_map)
        colorscale = 'Set3'
        colorbar_title = 'Forage'
    
    # Ajouter les échantillons
    fig.add_trace(go.Scatter3d(
        x=plot_df['X'],
        y=plot_df['Y'],
        z=plot_df['Z'],
        mode='markers',
        marker=dict(
            size=3 + plot_df['Grade'] * 2,  # Taille basée sur la teneur
            color=colors,
            colorscale=colorscale,
            opacity=0.8,
            colorbar=dict(title=colorbar_title),
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        name='Échantillons',
        text=plot_df['Hole'],
        hovertemplate='Forage: %{text}<br>Teneur: %{marker.color:.3f}g/t<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
    ))
    
    # Ajouter les intervalles minéralisés si disponibles
    if intervals_df is not None and not intervals_df.empty:
        for _, interval in intervals_df.iterrows():
            # Trouver les échantillons de cet intervalle
            hole_samples = plot_df[
                (plot_df['Hole'] == interval['HoleID']) &
                (plot_df[mapping['From']] >= interval['From']) &
                (plot_df[mapping['To']] <= interval['To'])
            ]
            
            if len(hole_samples) > 1:
                fig.add_trace(go.Scatter3d(
                    x=hole_samples['X'],
                    y=hole_samples['Y'],
                    z=hole_samples['Z'],
                    mode='lines+markers',
                    line=dict(color='red', width=8),
                    marker=dict(size=6, color='red'),
                    name=f'Intervalle {interval["IntervalID"]}',
                    hovertemplate=f'Intervalle {interval["IntervalID"]}<br>Teneur: {interval["AvgGrade"]:.3f}g/t<br>Longueur: {interval["Length"]:.1f}m<extra></extra>'
                ))
    
    # Configuration de la mise en page
    fig.update_layout(
        title="Visualisation 3D - Échantillons et Shear Zone",
        scene=dict(
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            zaxis_title="Élévation (m)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data'
        ),
        width=1000,
        height=700,
        margin=dict(r=20, b=10, l=10, t=40)
    )
    
    return fig

def create_grade_distribution_plot(df, mapping):
    """Créer un graphique de distribution des teneurs"""
    
    if not mapping.get('Au'):
        return None
    
    grades = pd.to_numeric(df[mapping['Au']], errors='coerce').dropna()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Histogramme des Teneurs', 'Box Plot', 'Teneurs Cumulatives', 'QQ Plot'),
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
    
    # QQ Plot (approximation)
    theoretical_quantiles = np.linspace(0.01, 0.99, len(sorted_grades))
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles, 
            y=sorted_grades, 
            mode='markers',
            name='QQ Plot'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="Analyse Statistique des Teneurs en Or",
        showlegend=False
    )
    
    return fig

def create_intervals_analysis_plot(intervals_df):
    """Créer un graphique d'analyse des intervalles"""
    
    if intervals_df is None or intervals_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribution Longueurs', 'Distribution Teneurs', 'Grade vs Longueur', 'Qualité des Intervalles'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Distribution des longueurs
    fig.add_trace(
        go.Histogram(x=intervals_df['Length'], nbinsx=20, name='Longueurs', opacity=0.7),
        row=1, col=1
    )
    
    # Distribution des teneurs
    fig.add_trace(
        go.Histogram(x=intervals_df['AvgGrade'], nbinsx=20, name='Teneurs', opacity=0.7),
        row=1, col=2
    )
    
    # Grade vs Longueur
    fig.add_trace(
        go.Scatter(
            x=intervals_df['Length'],
            y=intervals_df['AvgGrade'],
            mode='markers',
            marker=dict(
                size=intervals_df['GradeXLength'] / 2,
                color=intervals_df['GradeXLength'],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Grade×Longueur")
            ),
            name='G×L',
            text=intervals_df['HoleID'],
            hovertemplate='Forage: %{text}<br>Longueur: %{x:.1f}m<br>Teneur: %{y:.3f}g/t<br>G×L: %{marker.color:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Qualité des intervalles
    quality_categories = ['Excellent (G×L≥10)', 'Bon (G×L 5-10)', 'Modéré (G×L<5)']
    quality_counts = [
        len(intervals_df[intervals_df['GradeXLength'] >= 10]),
        len(intervals_df[(intervals_df['GradeXLength'] >= 5) & (intervals_df['GradeXLength'] < 10)]),
        len(intervals_df[intervals_df['GradeXLength'] < 5])
    ]
    
    fig.add_trace(
        go.Bar(x=quality_categories, y=quality_counts, name='Qualité'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="Analyse des Intervalles Minéralisés",
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
        <h1>⛏️ Analyseur d'Intervalles Minéralisés avec Visualisation 3D</h1>
        <p><strong>Optimisation géologique pour Leapfrog Geo | Développé par Didier Ouedraogo, P.Geo</strong></p>
        <p><em>Contraintes de distance ET dilution - Visualisation 3D interactive</em></p>
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
        - ✅ Visualisation 3D interactive
        - ✅ Analyse contraintes dilution
        - ✅ Export multi-format
        - ✅ Mesh shear zone 3D
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
        "🌐 Visualisation 3D", 
        "🔬 Analyse", 
        "📈 Résultats & Export"
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
                    help="Format attendu: HoleID, From, To, Au, Easting, Northing, Elevation"
                )
                
                if uploaded_file is not None:
                    samples_df = load_csv_file(uploaded_file)
                    if samples_df is not None:
                        st.session_state.samples_df = samples_df
                        st.session_state.drillholes_df = None
                        st.success(f"✅ {len(samples_df):,} échantillons importés")
            
            else:
                if st.button("🚀 Générer Données Demo", type="primary", use_container_width=True):
                    with st.spinner("Génération des données de démonstration..."):
                        samples_df, drillholes_df = generate_demo_data()
                        st.session_state.samples_df = samples_df
                        st.session_state.drillholes_df = drillholes_df
                        st.success("✅ Données de démonstration générées!")
        
        with col2:
            st.subheader("📋 Format CSV Attendu")
            st.code("""
HoleID,From,To,Au,Easting,Northing,Elevation
DDH-001,0,2,0.15,450100,5550200,245.5
DDH-001,2,4,1.25,450102,5550198,244.2
DDH-001,4,6,0.45,450104,5550196,243.1
            """, language="csv")

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
                unique_holes = df.iloc[:, 0].nunique() if len(df) > 0 else 0
                st.metric("🗺️ Forages", f"{unique_holes}")
            
            with col4:
                if len(df.columns) > 3:
                    try:
                        numeric_col = df.select_dtypes(include=[np.number]).columns[0]
                        avg_val = df[numeric_col].mean()
                        st.metric("📈 Moyenne", f"{avg_val:.3f}")
                    except:
                        st.metric("📈 Status", "OK")

            # Aperçu du tableau
            st.subheader("📋 Tableau des Données (50 premières lignes)")
            st.dataframe(df.head(50), use_container_width=True, height=300)
            
            # Informations sur les colonnes
            st.subheader("📊 Informations sur les Colonnes")
            col_info = pd.DataFrame({
                'Colonne': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Null': df.isnull().sum(),
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
                'Easting': ('Coordonnée Est', 'Coordonnée X (optionnel)'),
                'Northing': ('Coordonnée Nord', 'Coordonnée Y (optionnel)'),
                'Elevation': ('Élévation', 'Élévation Z (optionnel)')
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
            st.json(config)

    # ========================================
    # TAB 3: VISUALISATION 3D
    # ========================================
    with tab3:
        st.header("🌐 Visualisation 3D Interactive")
        
        if 'samples_df' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord importer des données")
            return
        
        if 'column_mapping' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord configurer le mapping des colonnes")
            return
        
        df = st.session_state.samples_df
        mapping = st.session_state.column_mapping
        
        # Vérifier si les coordonnées 3D sont disponibles
        has_3d_coords = all(mapping.get(col) for col in ['Easting', 'Northing', 'Elevation'])
        
        if not has_3d_coords:
            st.warning("⚠️ Coordonnées 3D manquantes. Veuillez mapper les colonnes Easting, Northing et Elevation.")
            return
        
        # Contrôles de visualisation
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color_by = st.selectbox(
                "🎨 Coloration par:",
                ['grade', 'distance', 'hole'],
                format_func=lambda x: {
                    'grade': 'Teneur Au',
                    'distance': 'Distance Shear',
                    'hole': 'Forage'
                }[x]
            )
        
        with col2:
            show_mesh = st.checkbox("🗂️ Afficher Shear Zone", value=True)
        
        with col3:
            show_intervals = st.checkbox("📍 Afficher Intervalles", value=False)
        
        with col4:
            sample_size = st.selectbox(
                "📊 Échantillons à afficher:",
                [500, 1000, 2000, "Tous"],
                index=1
            )
        
        # Préparer les données pour la visualisation
        plot_df = df.copy()
        if sample_size != "Tous":
            plot_df = plot_df.sample(min(sample_size, len(plot_df)))
        
        # Créer la visualisation 3D
        intervals_df = st.session_state.get('analysis_results', None) if show_intervals else None
        
        try:
            fig_3d = create_3d_scatter_plot(
                plot_df, mapping, color_by, show_mesh, intervals_df
            )
            
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Informations sur la visualisation
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 Points Affichés", f"{len(plot_df):,}")
                
                with col2:
                    if mapping.get('Au'):
                        grades = pd.to_numeric(plot_df[mapping['Au']], errors='coerce').dropna()
                        st.metric("💰 Teneur Moyenne", f"{grades.mean():.3f} g/t")
                
                with col3:
                    if show_mesh:
                        st.metric("🗂️ Mesh Shear Zone", "Affiché")
                    else:
                        st.metric("🗂️ Mesh Shear Zone", "Masqué")
                
                # Légende et explications
                with st.expander("ℹ️ Guide de la Visualisation 3D"):
                    st.markdown("""
                    ### 🎨 Code Couleur
                    - **Par Teneur**: Violet (faible) → Jaune (élevé)
                    - **Par Distance**: Bleu (proche shear) → Rouge (éloigné)
                    - **Par Forage**: Couleur unique par forage
                    
                    ### 🗂️ Éléments Affichés
                    - **Points orange**: Mesh de la shear zone (N45°E, 75°SE)
                    - **Points colorés**: Échantillons (taille = teneur)
                    - **Lignes rouges**: Intervalles minéralisés (si activé)
                    
                    ### 🎛️ Contrôles
                    - **Rotation**: Clic + glissement
                    - **Zoom**: Molette de la souris
                    - **Pan**: Shift + clic + glissement
                    """)
            
            else:
                st.error("❌ Impossible de créer la visualisation 3D")
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la création de la visualisation: {str(e)}")

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
        
        if mapping.get('Au'):
            grades = pd.to_numeric(df[mapping['Au']], errors='coerce').dropna()
            above_cutoff = len(grades[grades >= config['cutoff_grade']])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Total Échantillons", f"{len(grades):,}")
            with col2:
                st.metric("💰 > Teneur Coupure", f"{above_cutoff:,}")
            with col3:
                percentage = (above_cutoff / len(grades)) * 100 if len(grades) > 0 else 0
                st.metric("📊 Pourcentage", f"{percentage:.1f}%")
        
        # Bouton d'analyse
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Lancer l'Analyse Complète", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    try:
                        # Ajouter une barre de progression
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Préparation des données...")
                        progress_bar.progress(25)
                        
                        status_text.text("Application des critères...")
                        progress_bar.progress(50)
                        
                        results_df = run_mineral_analysis(df, mapping, config)
                        
                        status_text.text("Génération des résultats...")
                        progress_bar.progress(75)
                        
                        st.session_state.analysis_results = results_df
                        
                        status_text.text("Analyse terminée!")
                        progress_bar.progress(100)
                        
                        # Nettoyer l'interface
                        import time
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
                        
                        if len(results_df) > 0:
                            st.success(f"✅ Analyse terminée! {len(results_df)} intervalles trouvés.")
                        else:
                            st.warning("⚠️ Aucun intervalle trouvé avec les critères actuels.")
                            
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        
        with col2:
            if st.button("📊 Graphiques de Distribution", use_container_width=True):
                fig_dist = create_grade_distribution_plot(df, mapping)
                if fig_dist:
                    st.plotly_chart(fig_dist, use_container_width=True)
        
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
                    results_df,
                    use_container_width=True,
                    height=300,
                    column_config={
                        "AvgGrade": st.column_config.NumberColumn(
                            "Teneur Moy (g/t)",
                            format="%.3f"
                        ),
                        "Length": st.column_config.NumberColumn(
                            "Longueur (m)",
                            format="%.2f"
                        ),
                        "GradeXLength": st.column_config.NumberColumn(
                            "Grade×Longueur",
                            format="%.2f"
                        )
                    }
                )

    # ========================================
    # TAB 5: RÉSULTATS ET EXPORT
    # ========================================
    with tab5:
        st.header("📈 Résultats et Export")
        
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
        
        # Graphiques d'analyse
        st.subheader("📈 Analyse des Résultats")
        
        fig_analysis = create_intervals_analysis_plot(results_df)
        if fig_analysis:
            st.plotly_chart(fig_analysis, use_container_width=True)
        
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
                            'Intervalles Trouvés'
                        ],
                        'Valeur': [
                            datetime.now().strftime('%Y-%m-%d %H:%M'),
                            config['cutoff_grade'],
                            config['min_length'],
                            config['min_samples'],
                            config['max_distance'],
                            config['max_dilution'],
                            len(results_df)
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
        
        # Tableau détaillé final
        st.markdown("---")
        st.subheader("📋 Tableau Détaillé Final")
        
        # Ajout de filtres pour le tableau
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_grade_filter = st.slider(
                "Teneur minimum pour affichage:",
                min_value=0.0,
                max_value=float(results_df['AvgGrade'].max()),
                value=0.0,
                step=0.1
            )
        
        with col2:
            min_length_filter = st.slider(
                "Longueur minimum pour affichage:",
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
        
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400,
            column_config={
                "IntervalID": st.column_config.NumberColumn("ID"),
                "HoleID": st.column_config.TextColumn("Forage"),
                "From": st.column_config.NumberColumn("De (m)", format="%.1f"),
                "To": st.column_config.NumberColumn("À (m)", format="%.1f"),
                "Length": st.column_config.NumberColumn("Longueur (m)", format="%.2f"),
                "AvgGrade": st.column_config.NumberColumn("Teneur Moy (g/t)", format="%.3f"),
                "MaxGrade": st.column_config.NumberColumn("Teneur Max (g/t)", format="%.3f"),
                "SampleCount": st.column_config.NumberColumn("Échantillons"),
                "GradeXLength": st.column_config.NumberColumn("Grade×Longueur", format="%.2f"),
                "AvgDistance": st.column_config.NumberColumn("Distance Moy (m)", format="%.2f")
            }
        )
        
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
                    excellent/len(filtered_df)*100 if len(filtered_df) > 0 else 0,
                    bon/len(filtered_df)*100 if len(filtered_df) > 0 else 0,
                    modere/len(filtered_df)*100 if len(filtered_df) > 0 else 0
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
                    court/len(filtered_df)*100 if len(filtered_df) > 0 else 0,
                    moyen/len(filtered_df)*100 if len(filtered_df) > 0 else 0,
                    long/len(filtered_df)*100 if len(filtered_df) > 0 else 0
                ]
            })
            
            st.dataframe(length_data, use_container_width=True)

if __name__ == "__main__":
    # Import des modules nécessaires
    import re
    
    main()