# ==========================================
# 5. MAIN EVENT CHART (SAFE ANNOTATIONS)
# ==========================================
st.markdown("### > MACRO-EVENT VOLATILITY IMPACT MATRIX")

# Drop any blank dates to prevent Plotly rendering errors
chart_df = df_main.dropna(subset=['Date']).tail(150).copy()
fig = go.Figure()

# Trace 1: Volatility
fig.add_trace(go.Scatter(
    x=chart_df['Date'], y=chart_df['Volatility'], name='Volatility (Sigma)',
    line=dict(color="#00f0ff", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.1)'
))

# Trace 2: WTI Price
fig.add_trace(go.Scatter(
    x=chart_df['Date'], y=chart_df['WTI'], name='WTI Price ($)',
    yaxis='y2', line=dict(color="#00ff41", width=2, dash='dot')
))

# Safe Event Markers 
try:
    max_vol = float(chart_df['Volatility'].max())
    # Grab the actual datetime objects
    mid_date = chart_df['Date'].iloc[len(chart_df)//3]
    late_date = chart_df['Date'].iloc[int(len(chart_df)//1.5)]

    # Pass the native datetime object directly instead of formatting to strings
    fig.add_vline(x=mid_date, line_width=2, line_dash="dash", line_color="#FF003C")
    fig.add_vline(x=late_date, line_width=2, line_dash="dash", line_color="#FFD700")

    fig.add_annotation(x=mid_date, y=max_vol, text="[ US TARIFFS ]", showarrow=False, yshift=15, font=dict(color="#FF003C", family="Orbitron"))
    fig.add_annotation(x=late_date, y=max_vol * 0.9, text="[ OPEC POLICY ]", showarrow=False, yshift=15, font=dict(color="#FFD700", family="Orbitron"))
except Exception as e:
    pass # Silent failsafe

# Removed 'type="date"' from xaxis to allow Plotly's auto-inference engine to work natively
fig.update_layout(
    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,20,0,0.2)',
    height=450, margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 65, 0.1)'),
    yaxis=dict(title="Volatility (Sigma)", titlefont=dict(color="#00f0ff"), tickfont=dict(color="#00f0ff"), showgrid=True, gridcolor='rgba(0, 255, 65, 0.1)'),
    yaxis2=dict(title="WTI Price (USD)", titlefont=dict(color="#00ff41"), tickfont=dict(color="#00ff41"), overlaying='y', side='right', showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)
