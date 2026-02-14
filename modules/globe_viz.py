import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def render_3d_globe():
    """Returns a fully configured Plotly 3D Orthographic Globe."""
    # Base dataset for the visualization
    df_globe = pd.DataFrame({
        'Country': ['United States', 'United Kingdom', 'UAE', 'Saudi Arabia', 'Russia', 'China'],
        'Market': ['WTI', 'Brent', 'Dubai', 'Arab Light', 'Urals', 'Daqing'],
        'Lat': [37.09, 55.37, 23.42, 23.88, 61.52, 35.86],
        'Lon': [-95.71, -3.43, 53.84, 45.07, 105.31, 104.19],
        'Risk_Score': [50, 45, 40, 48, 85, 55] 
    })
    
    fig = px.scatter_geo(
        df_globe, lat='Lat', lon='Lon', 
        hover_name='Market', hover_data={'Lat': False, 'Lon': False, 'Risk_Score': True},
        size='Risk_Score', 
        projection='orthographic',
        color='Risk_Score',
        color_continuous_scale=[config.COLORS['accent_primary'], config.COLORS['warning'], config.COLORS['danger']]
    )
    
    fig.update_layout(
        height=500, 
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False, # Hide the color bar for a cleaner UI
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            showland=True, landcolor=config.COLORS['surface'],
            showocean=True, oceancolor=config.COLORS['background'],
            showcountries=True, countrycolor=config.COLORS['border'],
            showcoastlines=True, coastlinecolor=config.COLORS['border']
        )
    )
    return fig
