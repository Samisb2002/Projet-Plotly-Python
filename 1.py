import json
from collections import defaultdict, Counter
import plotly.express as px
import sys

# Force Python à utiliser l'encodage UTF-8 pour la sortie
sys.stdout.reconfigure(encoding='utf-8')

# 1. Charger les données volumineuses
def load_large_json(file_path):
    """Charger un fichier JSON volumineux."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 2. Explorer la structure des données
def explore_data_structure(data, level=0):
    """Affiche la structure des clés du dictionnaire pour comprendre sa hiérarchie."""
    if isinstance(data, dict):
        for key in data.keys():
            print("  " * level + f"Key: {key}")
            explore_data_structure(data[key], level + 1)
    elif isinstance(data, list):
        print("  " * level + f"List of {len(data)} items")
    else:
        print("  " * level + f"Value: {type(data)}")

# 3. Compter les publications par mois
def count_publications_by_month(data):
    """Compte le nombre de publications pour chaque mois."""
    monthly_counts = defaultdict(int)
    for year, months in data.items():
        for month, days in months.items():
            for day in days:
                monthly_counts[f"{year}-{int(month):02d}"] += len(days[day])
    return dict(monthly_counts)

# 4. Visualiser les publications par mois
def plot_publications(publication_counts):
    """Créer un graphique interactif avec Plotly."""
    dates = list(publication_counts.keys())
    counts = list(publication_counts.values())
    fig = px.bar(x=dates, y=counts, labels={'x': 'Mois', 'y': 'Nombre de publications'}, 
                 title="Nombre de publications par mois")
    fig.update_layout(xaxis=dict(type='category'), template="plotly_white")
    fig.show()

# 5. Extraire les mots-clés les plus fréquents
def get_top_keywords(metadata, top_n=20):
    """Récupère les mots-clés les plus fréquents."""
    keywords = metadata['all']['kws']
    counter = Counter(keywords)
    return counter.most_common(top_n)

# 6. Visualiser les mots-clés les plus fréquents
def plot_top_keywords(top_keywords):
    """Crée un graphique des mots-clés les plus fréquents."""
    words, counts = zip(*top_keywords)
    fig = px.bar(x=words, y=counts, labels={'x': 'Mots-clés', 'y': 'Fréquence'}, 
                 title="Top 20 des mots-clés")
    fig.update_layout(template="plotly_white")
    fig.show()

# Charger les fichiers JSON volumineux
data_file = "merged_data.json"  # Remplacez par le chemin vers votre fichier data
metadata_file = "merged_metadata.json"  # Remplacez par le chemin vers votre fichier metadata

print("Chargement des fichiers...")
data = load_large_json(data_file)
metadata = load_large_json(metadata_file)

# Explorer les structures des fichiers (facultatif)
print("Structure de DATA :")
explore_data_structure(data)

print("\nStructure de METADATA :")
explore_data_structure(metadata)

# Analyse des données : Publications par mois
print("Calcul du nombre de publications par mois...")
publication_counts = count_publications_by_month(data)

print("Visualisation des publications par mois...")
plot_publications(publication_counts)

# Analyse des données : Mots-clés les plus fréquents
print("Extraction des mots-clés les plus fréquents...")
top_keywords = get_top_keywords(metadata)

print("Visualisation des mots-clés les plus fréquents...")
plot_top_keywords(top_keywords)
