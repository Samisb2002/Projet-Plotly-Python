import json
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from prince import MCA
from extract_keywords import ExtractKeywords
from clustering import Clustering
from analysis_keyboard import AnalysisKeyboard
from graph_utils import GraphUtils

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def visualize_keywords(df_keywords):
    top_keywords = df_keywords.sort_values(by='Count', ascending=False).head(10)
    fig = px.bar(top_keywords, x='Keyword', y='Count',
                 title='Top 10 Keywords by Frequency',
                 labels={'Keyword': 'Keyword', 'Count': 'Frequency'})
    fig.show()

def generate_wordcloud(keywords):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keywords)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def main():
    file_path = 'fr.sputniknews.africa--20220630--20230630.json'
    data = load_data(file_path)
    keywords = ExtractKeywords.extract_keywords(file_path)

    df_keywords = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Count'])

    visualize_keywords(df_keywords)

    # Clustering
    df_keywords_clustered = Clustering.cluster_keywords(df_keywords)
    print("Palavras-chave com clusters:")
    print(df_keywords_clustered.head(10))

    generate_wordcloud(keywords)

    # Correspondence Analysis
    AnalysisKeyboard.perform_ca(df_keywords)

    # Graph Mining
    GraphUtils.generate_graph(df_keywords)

if __name__ == '__main__':
    main()
