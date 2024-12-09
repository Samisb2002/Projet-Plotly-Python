import json
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  
from extract_keywords import ExtractKeywords
from clustering import Clustering
from analysis_keyboard import AnalysisKeyboard
from graph_utils import GraphUtils
from treemap_generator import TreemapGenerator

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
    plt.draw()  
    plt.show()

def main():
    file_path = 'fr.sputniknews.africa--20221101--20221231.json'  
    data = load_data(file_path)
    keywords = ExtractKeywords.extract_keywords(file_path)

    
    df_keywords = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Count'])
    print(df_keywords.head())  

    # show keyboards 
    visualize_keywords(df_keywords)

    # Clustering 
    df_keywords_clustered = Clustering.cluster_keywords(df_keywords)
    print("Palavras-chave com clusters:")
    print(df_keywords_clustered.head(10))

    # WordCloud
    generate_wordcloud(keywords)

    # Treemap
    TreemapGenerator.generate_treemap(df_keywords)  

    # Correspondence Analysis 
    AnalysisKeyboard.perform_ca(df_keywords)

    # Graph Mining 
    GraphUtils.generate_graph(df_keywords)

if __name__ == '__main__':
    main()
