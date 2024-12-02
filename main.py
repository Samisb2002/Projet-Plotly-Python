import json
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from prince import MCA

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def extract_keywords(data):
    return data['metadata']['all']['kws']


def visualize_keywords(df_keywords):
    
    top_keywords = df_keywords.sort_values(by='Count', ascending=False).head(10)

    
    fig = px.bar(top_keywords, x='Keyword', y='Count',
                title='Top 10 Keywords by Frequency',
                labels={'Keyword': 'Keyword', 'Count': 'Frequency'})

    
    fig.show()


def cluster_keywords(df_keywords):
    keywords_list = df_keywords['Keyword'].tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(keywords_list)

    kmeans = KMeans(n_clusters=5, random_state=42)
    df_keywords['Cluster'] = kmeans.fit_predict(X)

    return df_keywords


def generate_wordcloud(keywords):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keywords)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def perform_ca(df_keywords):
    ca = MCA(n_components=2)
    ca = ca.fit(df_keywords[['Keyword', 'Count']])

    
    ca.plot_coordinates(X=df_keywords[['Keyword', 'Count']], figsize=(10, 7))
    plt.show()


def main():
  
    file_path = 'fr.sputniknews.africa--20220630--20230630.json'
    
 
    data = load_data(file_path)

    
    keywords = extract_keywords(data)

    
    df_keywords = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Count'])

    
    visualize_keywords(df_keywords)

   
    df_keywords_clustered = cluster_keywords(df_keywords)
    print("Palavras-chave com clusters:")
    print(df_keywords_clustered.head(10))

    generate_wordcloud(keywords)

    
    perform_ca(df_keywords)

if __name__ == '__main__':
    main()


#test
#import prince
#print(dir(prince))
