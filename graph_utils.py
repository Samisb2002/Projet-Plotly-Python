import networkx as nx
import matplotlib.pyplot as plt

class GraphUtils:
    
    @staticmethod
    def generate_graph(df_keywords):
        G = nx.Graph()

        for index, row in df_keywords.iterrows():
            G.add_node(row['Keyword'], size=row['Count'])
            for other_index, other_row in df_keywords.iterrows():
                if row['Keyword'] != other_row['Keyword']:  # Evitar auto-conexões
                    G.add_edge(row['Keyword'], other_row['Keyword'])

        pos = nx.spring_layout(G, seed=42, iterations=50)

        nx.draw(G, pos, with_labels=True, node_size=[G.nodes[node]['size']*10 for node in G.nodes()])
        plt.show()

        return G
