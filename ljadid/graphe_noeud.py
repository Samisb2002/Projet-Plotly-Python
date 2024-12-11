import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# Charger le fichier CSV contenant les articles
articles_file = "output.csv"
articles_df = pd.read_csv(articles_file)

# Fonction pour transformer les colonnes en listes
def parse_entities(column):
    return column.apply(lambda x: str(x).split(";") if pd.notnull(x) else [])

# Nettoyer les colonnes pertinentes
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

# Calculer la fréquence des relations
relations_summary = relations_df.groupby(["Entity_A", "Entity_B"]).size().reset_index(name="Frequency")

# Filtrer les relations avec une fréquence significative
min_frequency = 400  # Ajustez ce seuil selon vos besoins
filtered_relations = relations_summary[relations_summary["Frequency"] >= min_frequency]

# Construire le graphe avec NetworkX
G = nx.Graph()
for _, row in filtered_relations.iterrows():
    G.add_edge(row["Entity_A"], row["Entity_B"], weight=row["Frequency"])

# Ajuster la disposition avec une force de répulsion plus forte
pos = nx.spring_layout(G, k=1000, seed=42)  # Augmentez `k` pour une dispersion plus grande

# Construire les traces pour les arêtes (liens)
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

# Construire les traces pour les nœuds
node_x = []
node_y = []
node_text = []
node_size = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)
    node_size.append(len(list(G.neighbors(node))) * 10)  # Taille du nœud en fonction du degré

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=node_size,
        colorbar=dict(
            thickness=15,
            title="Degré du Nœud",
            xanchor='left',
            titleside='right'
        )
    ),
    text=node_text
)

# Créer la figure avec Plotly
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Graphe des Relations entre Entités",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                ))

fig.show()
