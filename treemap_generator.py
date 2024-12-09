import plotly.express as px
import pandas as pd

class TreemapGenerator:
    @staticmethod
    def generate_treemap(df):
     
        print(df.head())  

        fig = px.treemap(df, 
                         path=['Keyword'],  
                         values='Count',     
                         title='Treemap of Keywords by Frequency')
        
       
        fig.show(renderer="browser")  
