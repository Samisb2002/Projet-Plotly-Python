from wordcloud import WordCloud
import matplotlib.pyplot as plt

class WordCloudData:

    @staticmethod
    def generate_wordcloud(keywords):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keywords)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
