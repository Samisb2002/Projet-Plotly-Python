import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# Charger le fichier JSON
file_path = "fr.sputniknews.africa--20220630--20230630.json"

try:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"Fichier '{file_path}' chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du fichier JSON : {e}")
    exit()

# Étape 1 : Extraction des données des mots-clés par mois
monthly_data = data.get("metadata", {}).get("month", {})
if not monthly_data:
    print("Aucune donnée trouvée dans 'metadata > month'.")
    exit()

# Préparer une matrice de co-occurrence des mots-clés
all_keywords = set()
monthly_keywords = []

for year, months in monthly_data.items():
    for month, values in months.items():
        kws = values.get("kws", {})
        monthly_keywords.append(kws)
        all_keywords.update(kws.keys())

# Créer un DataFrame avec tous les mots-clés et leurs fréquences mensuelles
all_keywords = list(all_keywords)
keyword_matrix = pd.DataFrame(
    [{kw: kws.get(kw, 0) for kw in all_keywords} for kws in monthly_keywords]
)

# Étape 2 : Réduire à un sous-ensemble des mots-clés les plus fréquents
top_n = 300  # Limiter aux 300 mots-clés les plus fréquents
top_keywords = keyword_matrix.sum(axis=0).nlargest(top_n).index
keyword_matrix = keyword_matrix[top_keywords]

# Vérification : Taille du DataFrame après réduction
print("Dimensions de la matrice des mots-clés :", keyword_matrix.shape)

# Étape 3 : Transposer la matrice pour effectuer le clustering sur les mots-clés
keyword_matrix = keyword_matrix.T  # Mots-clés en lignes, périodes en colonnes

# Vérification : Taille après transposition
print("Dimensions après transposition :", keyword_matrix.shape)

# Étape 4 : Normalisation des données pour le clustering
scaler = StandardScaler()
keyword_matrix_scaled = scaler.fit_transform(keyword_matrix)

# Étape 5 : Appliquer l’algorithme K-Means
num_clusters = 5  # Nombre de clusters à former
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(keyword_matrix_scaled)

# Vérification : Longueur des clusters
print("Nombre de clusters attribués :", len(clusters))

if len(keyword_matrix) != len(clusters):
    print("Erreur : Les longueurs des mots-clés et des clusters ne correspondent pas.")
    exit()

# Étape 6 : Réduction de la dimensionnalité avec PCA pour visualisation
pca = PCA(n_components=2)
pca_result = pca.fit_transform(keyword_matrix_scaled)

# Créer un DataFrame pour les résultats
keyword_cluster_df = pd.DataFrame({
    "Keyword": keyword_matrix.index,
    "Cluster": clusters,
    "PCA1": pca_result[:, 0],
    "PCA2": pca_result[:, 1]
})

# Étape 7 : Visualisation des clusters avec Plotly
fig = px.scatter(
    keyword_cluster_df,
    x="PCA1",
    y="PCA2",
    color=keyword_cluster_df["Cluster"].astype(str),
    hover_data=["Keyword"],
    title="Clustering des Mots-Clés avec K-Means",
    labels={"Cluster": "Cluster"}
)
fig.show()
