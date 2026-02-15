"""
OVIP CONFIGURATION - TACTICAL ESPIONAGE THEME
"""

# SPY THRILLER COLOR SCHEME
COLORS = {
    'bg': '#000000',               # Pure pitch black background
    'matrix': '#00FF41',           # Classic terminal hacker green
    'cyan': '#00F0FF',             # Neon radar blue/cyan for charts
    'alert': '#FF003C',            # Blood red for crisis warnings
    'warning': '#FFD700',          # Yellow for elevated risk
    'panel': 'rgba(0, 20, 0, 0.4)' # Translucent green glass for UI boxes
}

# STRATEGIC TARGET NODES (Used by the 3D Satellite Globe)
COUNTRIES = {
    'USA': {
        'lat': 37.09, 'lon': -95.71, 
        'flag': '🇺🇸', 'name': 'UNITED STATES', 'oil': 'WTI Crude'
    },
    'CHINA': {
        'lat': 35.86, 'lon': 104.20, 
        'flag': '🇨🇳', 'name': 'PEOPLES REPUBLIC OF CHINA', 'oil': 'Daqing'
    },
    'INDIA': {
        'lat': 20.59, 'lon': 78.96, 
        'flag': '🇮🇳', 'name': 'REPUBLIC OF INDIA', 'oil': 'Indian Basket'
    },
    'UAE': {
        'lat': 23.42, 'lon': 53.85, 
        'flag': '🇦🇪', 'name': 'UNITED ARAB EMIRATES', 'oil': 'Dubai Crude'
    },
    'SAUDI': {
        'lat': 23.89, 'lon': 45.08, 
        'flag': '🇸🇦', 'name': 'KINGDOM OF SAUDI ARABIA', 'oil': 'Arab Light'
    }
}
