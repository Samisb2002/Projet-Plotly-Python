import pandas as pd
import plotly.express as px

# Charger le fichier CSV contenant les articles
articles_file = "output.csv"
articles_df = pd.read_csv(articles_file)

# Nettoyer les colonnes pour transformer les entités en listes
def parse_entities(column):
    return column.apply(lambda x: str(x).split(";") if pd.notnull(x) else [])

articles_df["Locations"] = parse_entities(articles_df["Locations"])
articles_df["Organizations"] = parse_entities(articles_df["Organizations"])
articles_df["Persons"] = parse_entities(articles_df["Persons"])

# Construire une liste de co-occurrences
relations = []
for _, row in articles_df.iterrows():
    all_entities = set(row["Locations"]) | set(row["Organizations"]) | set(row["Persons"])
    for entity_a in all_entities:
        for entity_b in all_entities:
            if entity_a != entity_b:
                relations.append((entity_a, entity_b))

# Créer un DataFrame des co-occurrences
relations_df = pd.DataFrame(relations, columns=["Entity_A", "Entity_B"])
cooccurrence_matrix = relations_df.groupby(["Entity_A", "Entity_B"]).size().unstack(fill_value=0)

# Limiter à Top 20 entités les plus fréquentes
top_entities = relations_df["Entity_A"].value_counts().head(20).index
filtered_matrix = cooccurrence_matrix.loc[top_entities, top_entities]

# Création de la Heatmap avec meilleure visualisation pour les valeurs < 800
fig = px.imshow(
    filtered_matrix,
    labels=dict(x="Entité", y="Entité", color="Fréquence"),
    x=filtered_matrix.columns,
    y=filtered_matrix.index,
    title="Matrice des Co-occurrences entre Entités",
    color_continuous_scale="Viridis",  # Choisir une échelle de couleurs
    zmax=800  # Maximum limité pour accentuer les petites valeurs
)

fig.update_coloraxes(cmax=800)  # Réglage pour mieux distinguer les valeurs < 800
fig.show()
