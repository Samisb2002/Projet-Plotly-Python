import plotly.express as px

class VisualizeKeyboard:

    @staticmethod
    def visualize_keywords(df_keywords):
        top_keywords = df_keywords.sort_values(by='Count', ascending=False).head(10)
        fig = px.bar(top_keywords, x='Keyword', y='Count',
                     title='Top 10 Keywords by Frequency',
                     labels={'Keyword': 'Keyword', 'Count': 'Frequency'})
        fig.show()
