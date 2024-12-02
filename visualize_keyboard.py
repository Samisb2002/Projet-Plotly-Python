import plotly.express as px


top_keywords = df_keywords.sort_values(by='Count', ascending=False).head(10)

# Create barchart
fig = px.bar(top_keywords, x='Keyword', y='Count',
            title='Top 10 Keywords by Frequency',
            labels={'Keyword': 'Keyword', 'Count': 'Frequency'})

# show
fig.show()
