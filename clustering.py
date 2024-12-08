from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

class Clustering:

    @staticmethod
    def cluster_keywords(df_keywords):
        keywords_list = df_keywords['Keyword'].tolist()
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(keywords_list)

        kmeans = KMeans(n_clusters=5, random_state=42)
        df_keywords['Cluster'] = kmeans.fit_predict(X)

        return df_keywords
