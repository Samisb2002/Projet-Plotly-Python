import json
from collections import defaultdict, Counter
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

# Charger les données JSON
def load_large_json(file_path):
    """Charger un fichier JSON volumineux."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

data_file = "merged_data.json"  # Remplacez par le chemin vers votre fichier data
metadata_file = "merged_metadata.json"  # Remplacez par le chemin vers votre fichier metadata

data = load_large_json(data_file)
metadata = load_large_json(metadata_file)

# Comptage des publications par mois
def count_publications_by_month(data):
    """Compte le nombre de publications pour chaque mois."""
    monthly_counts = defaultdict(int)
    for year, months in data.items():
        for month, days in months.items():
            for day in days:
                monthly_counts[f"{year}-{int(month):02d}"] += len(days[day])
    return dict(monthly_counts)

publication_counts = count_publications_by_month(data)

# Extraction des entités par catégorie
def get_top_entities(metadata, category='kws', min_frequency=1, top_n=20):
    """Récupère les entités les plus fréquentes pour une catégorie donnée avec un seuil minimal."""
    entities = metadata['all'][category]
    counter = Counter(entities)
    filtered_entities = [(k, v) for k, v in counter.items() if v >= min_frequency]
    return sorted(filtered_entities, key=lambda x: x[1], reverse=True)[:top_n]

# Création du tableau de bord Dash
app = dash.Dash(__name__)

# Layout du tableau de bord
app.layout = html.Div([
    html.H1("Tableau de bord des publications", style={'textAlign': 'center'}),
    
    # Section des publications par mois
    html.Div([
        html.Div([
            html.Label("Filtrer par année :"),
            dcc.Dropdown(
                id='year-filter',
                options=[{'label': year, 'value': year} for year in data.keys()],
                value=None,
                placeholder="Sélectionnez une année"
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='publications-chart')
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'marginBottom': '50px'}),
    
    # Section des entités les plus fréquentes
    html.Div([
        html.Div([
            html.Label("Catégorie d'entités :"),
            dcc.Dropdown(
                id='category-filter',
                options=[
                    {'label': 'Mots-clés', 'value': 'kws'},
                    {'label': 'Lieux', 'value': 'loc'},
                    {'label': 'Organisations', 'value': 'org'}
                ],
                value='kws',
                placeholder="Sélectionnez une catégorie"
            ),
            html.Label("Fréquence minimale :"),
            dcc.Slider(
                id='frequency-slider',
                min=1, max=100, step=1,
                marks={i: str(i) for i in range(1, 101, 10)},
                value=1
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='top-entities-chart')
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ])
])

# Callback pour mettre à jour les graphiques
@app.callback(
    [Output('publications-chart', 'figure'),
     Output('top-entities-chart', 'figure')],
    [Input('year-filter', 'value'),
     Input('category-filter', 'value'),
     Input('frequency-slider', 'value')]
)
def update_dashboard(selected_year, selected_category, min_frequency):
    # Graphique des publications par mois
    filtered_counts = publication_counts
    if selected_year:
        filtered_counts = {k: v for k, v in filtered_counts.items() if k.startswith(selected_year)}

    months = list(filtered_counts.keys())
    counts = list(filtered_counts.values())
    publications_fig = px.bar(
        x=months, y=counts,
        labels={'x': 'Mois', 'y': 'Nombre de publications'},
        title="Nombre de publications par mois"
    )
    publications_fig.update_layout(xaxis=dict(type='category'), template="plotly_white")

    # Graphique des entités les plus fréquentes
    top_entities = get_top_entities(metadata, category=selected_category, min_frequency=min_frequency)
    if top_entities:
        entities, entity_counts = zip(*top_entities)
    else:
        entities, entity_counts = [], []
    entities_fig = px.bar(
        x=entities, y=entity_counts,
        labels={'x': 'Entités', 'y': 'Fréquence'},
        title=f"Top 20 des {selected_category.capitalize()}"
    )
    entities_fig.update_layout(template="plotly_white")

    return publications_fig, entities_fig

# Lancer l'application Dash
if __name__ == '__main__':
    app.run_server(debug=True)
