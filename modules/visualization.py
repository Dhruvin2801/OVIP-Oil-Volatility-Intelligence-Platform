"""
OVIP - Visualization Module
Creates all charts and visualizations for the platform
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OVIP color scheme
COLORS = {
    'background': '#0A192F',
    'surface': '#112240',
    'accent_primary': '#64FFDA',
    'accent_secondary': '#00FF41',
    'text_primary': '#E6F1FF',
    'text_secondary': '#8892B0',
    'warning': '#FFAA00',
    'danger': '#FF073A',
    'border': '#1E3A5F',
}


def create_price_volatility_chart(
    df: pd.DataFrame,
    title: str = "Price & Volatility History"
) -> go.Figure:
    """Create dual-axis chart with price and volatility"""
    fig = go.Figure()
    
    if 'WTI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['WTI'], name='Price',
            line=dict(color=COLORS['accent_primary'], width=2),
            yaxis='y'
        ))
    
    if 'Volatility' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Volatility'], name='Volatility',
            fill='tozeroy', fillcolor='rgba(255, 7, 58, 0.2)',
            line=dict(color=COLORS['danger'], width=1),
            yaxis='y2'
        ))
    
    fig.update_layout(
        title=title, template='plotly_dark',
        plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text_primary'], family='Roboto Mono'),
        xaxis=dict(title='Date', gridcolor=COLORS['border']),
        yaxis=dict(
            title='Price ($/barrel)',
            titlefont=dict(color=COLORS['accent_primary']),
            tickfont=dict(color=COLORS['accent_primary']),
            gridcolor=COLORS['border']
        ),
        yaxis2=dict(
            title='Volatility',
            titlefont=dict(color=COLORS['danger']),
            tickfont=dict(color=COLORS['danger']),
            overlaying='y', side='right'
        ),
        hovermode='x unified', height=400,
    )
    
    return fig


def create_forecast_chart(
    df_historical: pd.DataFrame,
    df_forecast: pd.DataFrame,
    title: str = "Volatility Forecast"
) -> go.Figure:
    """Create forecast visualization with confidence intervals"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_historical['Date'], y=df_historical['Volatility'],
        name='Historical', line=dict(color=COLORS['text_primary'], width=2),
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_forecast['Date'], y=df_forecast['forecast'],
        name='Forecast', line=dict(color=COLORS['accent_primary'], width=2, dash='dash'),
        mode='lines'
    ))
    
    if 'upper_ci' in df_forecast.columns and 'lower_ci' in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=df_forecast['Date'], y=df_forecast['upper_ci'],
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_forecast['Date'], y=df_forecast['lower_ci'],
            mode='lines', line=dict(width=0),
            fillcolor='rgba(100, 255, 218, 0.2)', fill='tonexty',
            name='95% Confidence', hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=title, template='plotly_dark',
        plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text_primary'], family='Roboto Mono'),
        xaxis=dict(title='Date', gridcolor=COLORS['border']),
        yaxis=dict(title='Volatility', gridcolor=COLORS['border']),
        hovermode='x unified', height=400,
    )
    
    return fig


def create_feature_importance_chart(
    importance_dict: Dict[str, float],
    title: str = "Feature Importance"
) -> go.Figure:
    """Create horizontal bar chart of feature importance"""
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features = [f[0] for f in sorted_features]
    importance = [f[1] * 100 for f in sorted_features] 
    
    fig = go.Figure(go.Bar(
        y=features, x=importance, orientation='h',
        marker=dict(
            color=importance,
            colorscale=[[0, COLORS['border']], [1, COLORS['accent_primary']]],
            line=dict(color=COLORS['accent_primary'], width=1)
        ),
        text=[f'{v:.1f}%' for v in importance],
        textposition='outside',
    ))
    
    fig.update_layout(
        title=title, template='plotly_dark',
        plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text_primary'], family='Roboto Mono'),
        xaxis=dict(title='Importance (%)', gridcolor=COLORS['border']),
        yaxis=dict(gridcolor=COLORS['border']),
        height=max(400, len(features) * 30),
    )
    
    return fig


def create_performance_chart(
    df: pd.DataFrame,
    title: str = "Model Performance Over Time"
) -> go.Figure:
    """Create chart showing actual vs predicted values with error bands"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Volatility'],
        name='Actual', line=dict(color=COLORS['text_primary'], width=2),
    ))
    
    if 'Predicted_Vol' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Predicted_Vol'],
            name='Predicted', line=dict(color=COLORS['accent_primary'], width=2, dash='dot'),
        ))
    
    # FIXED: Drawing the error band around the Predicted value instead of Actual
    if 'Error' in df.columns and 'Predicted_Vol' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Predicted_Vol'] + abs(df['Error']),
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Predicted_Vol'] - abs(df['Error']),
            mode='lines', line=dict(width=0),
            fillcolor='rgba(255, 7, 58, 0.1)', fill='tonexty',
            name='Error Range', hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=title, template='plotly_dark',
        plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text_primary'], family='Roboto Mono'),
        xaxis=dict(title='Date', gridcolor=COLORS['border']),
        yaxis=dict(title='Volatility', gridcolor=COLORS['border']),
        hovermode='x unified', height=400,
    )
    
    return fig


def create_regime_timeline(
    df: pd.DataFrame,
    title: str = "Regime State Timeline"
) -> go.Figure:
    """Create timeline showing regime probability states"""
    regime_col = 'Crisis_Prob' if 'Crisis_Prob' in df.columns else 'L_Regime'
    
    if regime_col not in df.columns:
        logger.error(f"Regime column not found")
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df[regime_col], fill='tozeroy',
        line=dict(color=COLORS['warning'], width=2),
        fillcolor='rgba(255, 170, 0, 0.3)',
        name='Crisis Probability',
    ))
    
    fig.add_hline(y=0.3, line_dash="dash", line_color=COLORS['accent_primary'], 
                  annotation_text="Calm/Moderate", annotation_position="right")
    fig.add_hline(y=0.7, line_dash="dash", line_color=COLORS['danger'],
                  annotation_text="Moderate/Crisis", annotation_position="right")
    
    fig.update_layout(
        title=title, template='plotly_dark',
        plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text_primary'], family='Roboto Mono'),
        xaxis=dict(title='Date', gridcolor=COLORS['border']),
        yaxis=dict(title='Crisis Probability', gridcolor=COLORS['border']),
        hovermode='x unified', height=300,
    )
    
    return fig


def create_residual_plot(
    df: pd.DataFrame,
    title: str = "Residual Analysis"
) -> go.Figure:
    """Create residual diagnostic plots"""
    if 'Error' not in df.columns:
        logger.error("Error column not found for residual plot")
        return go.Figure()
    
    residuals = df['Error'].dropna()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals Over Time', 'Residual Distribution'),
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'], y=df['Error'], mode='markers',
            marker=dict(color=COLORS['accent_primary'], size=5),
            name='Residuals',
        ), row=1, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['danger'], row=1, col=1)
    
    fig.add_trace(
        go.Histogram(
            x=residuals, nbinsx=30,
            marker=dict(color=COLORS['accent_primary'], line=dict(color=COLORS['text_primary'], width=1)),
            name='Distribution',
        ), row=1, col=2
    )
    
    fig.update_layout(
        title=title, template='plotly_dark',
        plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text_primary'], family='Roboto Mono'),
        showlegend=False, height=400,
    )
    
    fig.update_xaxes(title='Date', gridcolor=COLORS['border'], row=1, col=1)
    fig.update_yaxes(title='Residual', gridcolor=COLORS['border'], row=1, col=1)
    fig.update_xaxes(title='Residual Value', gridcolor=COLORS['border'], row=1, col=2)
    fig.update_yaxes(title='Frequency', gridcolor=COLORS['border'], row=1, col=2)
    
    return fig


def create_gauge_chart(
    value: float,
    max_value: float = 1.0,
    title: str = "Metric",
    thresholds: Optional[List[Tuple[float, str]]] = None
) -> go.Figure:
    """Create gauge chart for single metric"""
    if thresholds is None:
        thresholds = [
            (0.3, COLORS['accent_primary']),  # Good
            (0.7, COLORS['warning']),         # Moderate
            (1.0, COLORS['danger']),          # High
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': COLORS['text_primary']}},
        number={'font': {'color': COLORS['accent_primary']}},
        gauge={
            'axis': {'range': [None, max_value], 'tickcolor': COLORS['text_secondary']},
            'bar': {'color': COLORS['accent_primary']},
            'bgcolor': COLORS['border'],
            'borderwidth': 2,
            'bordercolor': COLORS['text_secondary'],
            'steps': [
                {'range': [0, thresholds[0][0]], 'color': 'rgba(100, 255, 218, 0.2)'},
                {'range': [thresholds[0][0], thresholds[1][0]], 'color': 'rgba(255, 170, 0, 0.2)'},
                {'range': [thresholds[1][0], thresholds[2][0]], 'color': 'rgba(255, 7, 58, 0.2)'},
            ],
            'threshold': {
                'line': {'color': COLORS['danger'], 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text_primary'], family='Roboto Mono'),
        height=300, margin=dict(t=50, b=0, l=30, r=30)
    )
    
    return fig
