import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Charger les données
file_path = 'geocoded_countries_frequency.csv'
data = pd.read_csv(file_path)

# Initialiser l'application Dash
app = dash.Dash(__name__)

# Mise en page de l'application
app.layout = html.Div([
    html.H1("Visualisation des pays les plus mentionnés"),

    # Filtre pour la plage de fréquences
    dcc.RangeSlider(
        id='frequency-slider',
        min=data['frequency'].min(),
        max=data['frequency'].max(),
        step=1,
        marks={int(i): str(int(i)) for i in range(int(data['frequency'].min()), int(data['frequency'].max()) + 1, 50)},
        value=[data['frequency'].min(), data['frequency'].max()],
    ),
    
    html.Br(),

    # Filtre pour sélectionner des pays spécifiques
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': country, 'value': country} for country in data['country'].unique()],
        placeholder="Sélectionnez un ou plusieurs pays",
        multi=True
    ),

    html.Br(),

    # Graphique de la carte choroplèthe
    dcc.Graph(id='choropleth-map'),
])

# Callback pour mettre à jour la carte en fonction des filtres
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('frequency-slider', 'value'),
     Input('country-dropdown', 'value')]
)
def update_map(frequency_range, selected_countries):
    filtered_data = data[
        (data['frequency'] >= frequency_range[0]) &
        (data['frequency'] <= frequency_range[1])
    ]

    if selected_countries:
        filtered_data = filtered_data[filtered_data['country'].isin(selected_countries)]

    # Créer la carte
    fig = px.choropleth(
        data_frame=filtered_data,
        locations='country',
        locationmode='country names',
        color='frequency',
        color_continuous_scale='Viridis',
        range_color=[data['frequency'].min(), data['frequency'].max()],
        title='Carte des pays les plus mentionnés',
    )
    return fig

# Lancer le serveur
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
