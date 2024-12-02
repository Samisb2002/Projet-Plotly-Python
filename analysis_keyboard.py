from prince import CorrespondenceAnalysis


ca = CorrespondenceAnalysis(n_components=2)
ca = ca.fit(df_keywords[['Keyword', 'Count']])


ca.plot_coordinates(X=df_keywords[['Keyword', 'Count']], figsize=(10, 7))
plt.show()
