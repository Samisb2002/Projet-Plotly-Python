import dash
from dash import dcc, html, Input, Output
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import json
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict, Counter

# Charger les données CSV
file_path = 'geocoded_countries_frequency.csv'
data = pd.read_csv(file_path)

# Charger les données JSON

def load_large_json(file_path):
    """Charger un fichier JSON volumineux."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

data_file = "merged_data.json"  # Remplacez par le chemin vers votre fichier data
metadata_file = "merged_metadata.json"  # Remplacez par le chemin vers votre fichier metadata

data_json = load_large_json(data_file)
metadata = load_large_json(metadata_file)

# Comptage des publications par mois
def count_publications_by_month(data_json):
    """Compte le nombre de publications pour chaque mois."""
    monthly_counts = defaultdict(int)
    for year, months in data_json.items():
        for month, days in months.items():
            for day in days:
                monthly_counts[f"{year}-{int(month):02d}"] += len(days[day])
    return dict(monthly_counts)

publication_counts = count_publications_by_month(data_json)

# Extraction des entités par catégorie
def get_top_entities(metadata, category='kws', min_frequency=1, top_n=20):
    """Récupère les entités les plus fréquentes pour une catégorie donnée avec un seuil minimal."""
    entities = metadata['all'][category]
    counter = Counter(entities)
    filtered_entities = [(k, v) for k, v in counter.items() if v >= min_frequency]
    return sorted(filtered_entities, key=lambda x: x[1], reverse=True)[:top_n]

# Charger le fichier CSV contenant les articles
articles_file = "output.csv"
articles_df = pd.read_csv(articles_file)

# Nettoyer les colonnes pour transformer les entités en listes
def parse_entities(column):
    return column.apply(lambda x: str(x).split(";") if pd.notnull(x) else [])

articles_df["Locations"] = parse_entities(articles_df["Locations"])
articles_df["Organizations"] = parse_entities(articles_df["Organizations"])
articles_df["Persons"] = parse_entities(articles_df["Persons"])

# Construire une liste de relations
relations = []
for _, row in articles_df.iterrows():
    all_entities = set(row["Locations"]) | set(row["Organizations"]) | set(row["Persons"])
    for entity_a in all_entities:
        for entity_b in all_entities:
            if entity_a != entity_b:
                relations.append((entity_a, entity_b))

# Créer un DataFrame des relations
relations_df = pd.DataFrame(relations, columns=["Entity_A", "Entity_B"])
relations_summary = relations_df.groupby(["Entity_A", "Entity_B"]).size().reset_index(name="Frequency")
cooccurrence_matrix = relations_df.groupby(["Entity_A", "Entity_B"]).size().unstack(fill_value=0)

# Charger les données pour le clustering
clustering_file_path = "test2.csv"
updated_df = pd.read_csv(clustering_file_path)
updated_df["Keywords"] = updated_df["Keywords"].fillna("").apply(lambda x: x.split(";"))

# Initialiser l'application Dash
app = dash.Dash(__name__)

# Mise en page de l'application
app.layout = html.Div([
    html.H1("Visualisation des données et des clusters", style={'textAlign': 'center'}),

    # Section de la carte
    html.Div([
        html.H2("Carte des pays les plus mentionnés"),
        dcc.RangeSlider(
            id='frequency-slider',
            min=data['frequency'].min(),
            max=data['frequency'].max(),
            step=1,
            marks={int(i): str(int(i)) for i in range(int(data['frequency'].min()), int(data['frequency'].max()) + 1, 50)},
            value=[data['frequency'].min(), data['frequency'].max()],
        ),
        html.Br(),
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': country, 'value': country} for country in data['country'].unique()],
            placeholder="Sélectionnez un ou plusieurs pays",
            multi=True
        ),
        html.Br(),
        dcc.Graph(id='choropleth-map'),
    ], style={'marginBottom': '50px'}),

    # Section des publications par mois
    html.Div([
        html.H2("Publications par mois"),
        html.Div([
            html.Label("Filtrer par année :"),
            dcc.Dropdown(
                id='year-filter',
                options=[{'label': year, 'value': year} for year in data_json.keys()],
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
        html.H2("Top des entités"),
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
                id='entity-frequency-slider',
                min=1, max=100, step=1,
                marks={i: str(i) for i in range(1, 101, 10)},
                value=1
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='top-entities-chart')
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'marginBottom': '50px'}),

    # Section du clustering
    html.Div([
        html.H2("Clustering des Mots-Clés"),
        html.Div([
            html.Label("Nombre de mots-clés les plus fréquents :"),
            dcc.Slider(
                id="top-n-slider",
                min=50,
                max=500,
                step=50,
                value=300,
                marks={i: str(i) for i in range(50, 501, 50)}
            ),
            html.Label("Nombre de clusters :"),
            dcc.Slider(
                id="num-clusters-slider",
                min=2,
                max=20,
                step=1,
                value=10,
                marks={i: str(i) for i in range(2, 21)}
            ),
        ], style={"width": "50%", "margin": "auto"}),

        html.Br(),

        dcc.Graph(id="cluster-graph")
    ], style={'marginBottom': '50px'}),

    # Section du graphe de relations
    html.Div([
        html.H2("Graphe des Relations entre Entités"),
        html.Div([
            html.Label("Fréquence minimale des relations :"),
            dcc.Slider(
                id='relation-frequency-slider',
                min=1,
                max=500,
                step=10,
                marks={i: str(i) for i in range(1, 501, 100)},
                value=50
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='node-graph')
        ])
    ], style={'marginBottom': '50px'}),

    # Section de la heatmap
    html.Div([
        html.H2("Matrice des co-occurrences entre entités"),
        html.Div([
            html.Label("Nombre maximum d'entités :"),
            dcc.Slider(
                id='heatmap-entity-slider',
                min=5,
                max=20,
                step=1,
                marks={i: str(i) for i in range(5, 21)},
                value=10
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='heatmap')
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ])
])

# Callback pour mettre à jour la carte
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('frequency-slider', 'value'),
     Input('country-dropdown', 'value')]
)
def update_map(frequency_range, selected_countries):
    filtered_data = data[
        (data['frequency'] >= frequency_range[0]) &
        (data['frequency'] <= frequency_range[1])
    ]

    if selected_countries:
        filtered_data = filtered_data[filtered_data['country'].isin(selected_countries)]

    fig = px.choropleth(
        data_frame=filtered_data,
        locations='country',
        locationmode='country names',
        color='frequency',
        color_continuous_scale='Viridis',
        range_color=[data['frequency'].min(), data['frequency'].max()],
        title='Carte des pays les plus mentionnés',
    )
    return fig

# Callback pour mettre à jour les graphiques des publications et des entités
@app.callback(
    [Output('publications-chart', 'figure'),
     Output('top-entities-chart', 'figure')],
    [Input('year-filter', 'value'),
     Input('category-filter', 'value'),
     Input('entity-frequency-slider', 'value')]
)
def update_dashboard(selected_year, selected_category, min_frequency):
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

# Callback pour le clustering
@app.callback(
    Output("cluster-graph", "figure"),
    [Input("top-n-slider", "value"),
     Input("num-clusters-slider", "value")]
)
def update_clusters(top_n, num_clusters):
    # Extraction des mots-clés les plus fréquents
    keywords = updated_df.explode("Keywords")["Keywords"].value_counts()
    top_keywords = keywords.head(top_n).index

    # Création de la matrice des mots-clés par date
    keywords_by_date = updated_df.explode("Keywords").groupby(["date", "Keywords"]).size().unstack(fill_value=0)
    keywords_by_date = keywords_by_date[top_keywords]

    # Normalisation des données
    scaler = StandardScaler()
    keywords_scaled = scaler.fit_transform(keywords_by_date.T)

    # Clustering avec K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(keywords_scaled)

    # Réduction de la dimensionnalité avec PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(keywords_scaled)

    # Créer un DataFrame pour les résultats
    keywords_cluster_df = pd.DataFrame({
        "Keyword": keywords_by_date.columns,
        "Cluster": clusters,
        "PCA1": pca_result[:, 0],
        "PCA2": pca_result[:, 1]
    })

    # Visualisation avec Plotly
    fig = px.scatter(
        keywords_cluster_df,
        x="PCA1",
        y="PCA2",
        color=keywords_cluster_df["Cluster"].astype(str),
        hover_data=["Keyword"],
        title="Clustering des Mots-Clés avec K-Means",
        labels={"Cluster": "Cluster"}
    )
    return fig

# Callback pour le graphe de relations
@app.callback(
    Output('node-graph', 'figure'),
    [Input('relation-frequency-slider', 'value')]
)
def update_node_graph(min_frequency):
    filtered_relations = relations_summary[relations_summary["Frequency"] >= min_frequency]

    G = nx.Graph()
    for _, row in filtered_relations.iterrows():
        G.add_edge(row["Entity_A"], row["Entity_B"], weight=row["Frequency"])

    pos = nx.spring_layout(G, k=0.5, seed=42)

    edge_x = []
    edge_y = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_size = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(len(list(G.neighbors(node))) * 10)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            colorbar=None
        ),
        text=node_text
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Graphe des Relations entre Entités",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    return fig

# Callback pour la heatmap
@app.callback(
    Output('heatmap', 'figure'),
    [Input('heatmap-entity-slider', 'value')]
)
def update_heatmap(max_entities):
    top_entities = relations_df["Entity_A"].value_counts().head(max_entities).index
    filtered_matrix = cooccurrence_matrix.loc[top_entities, top_entities]

    fig = px.imshow(
        filtered_matrix,
        labels=dict(x="Entité", y="Entité", color="Fréquence"),
        x=filtered_matrix.columns,
        y=filtered_matrix.index,
        title="Matrice des Co-occurrences entre Entités",
        color_continuous_scale="Viridis",
        zmax=800
    )
    fig.update_coloraxes(cmax=800)
    return fig

# Lancer l'application
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
