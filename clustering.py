from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


keywords_list = df_keywords['Keyword'].tolist()

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(keywords_list)


kmeans = KMeans(n_clusters=5, random_state=42)
df_keywords['Cluster'] = kmeans.fit_predict(X)


print(df_keywords[['Keyword', 'Cluster']].head(10))
