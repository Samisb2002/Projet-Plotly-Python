import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Charger les données
file_path = 'geocoded_countries_frequency.csv'
data = pd.read_csv(file_path)

# Définir la valeur limite à 300
limit = 500

# Créer une carte interactive avec Plotly
fig = px.choropleth(
    data_frame=data,
    locations='country',  # Nom des pays
    locationmode='country names',  # Mode d'identification des pays
    color='frequency',  # Valeur utilisée pour colorer les pays
    color_continuous_scale='Viridis',  # Changer l'échelle de couleurs pour un dégradé plus riche
    range_color=[data['frequency'].min(), limit],  # Étendre la plage des couleurs pour les valeurs inférieures à 300
    title='Carte des pays les plus mentionnés',
)

# Initialiser l'application Dash
app = dash.Dash(__name__)

# Mise en page de l'application
app.layout = html.Div([
    html.H1("Visualisation des pays les plus mentionnés"),
    dcc.Graph(figure=fig),
])

# Lancer le serveur
if __name__ == '__main__':
    app.run_server(debug=True)
