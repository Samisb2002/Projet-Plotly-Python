from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class AnalysisKeyboard:

    @staticmethod
    def perform_ca(df_keywords):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_keywords[['Count']].values)
        df_keywords['PCA1'] = pca_result[:, 0]
        df_keywords['PCA2'] = pca_result[:, 1]

        plt.figure(figsize=(10, 7))
        plt.scatter(df_keywords['PCA1'], df_keywords['PCA2'])
        for i, keyword in enumerate(df_keywords['Keyword']):
            plt.annotate(keyword, (df_keywords['PCA1'][i], df_keywords['PCA2'][i]))
        plt.title('PCA Analysis of Keywords')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
