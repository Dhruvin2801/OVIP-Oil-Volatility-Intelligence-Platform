# 🌍 OVIP: Oil Volatility Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

> **An enterprise-grade, AI-driven command center for real-time oil market volatility prediction, regime detection, and strategic hedging.**

(https://ovip-oil-volatility-intelligence-platform-vdshdbbbucscycywvezc.streamlit.app/)

## 🚀 Executive Summary

Financial markets are noisy. Predicting the exact price of oil is often a fool's errand. Instead, **OVIP focuses on Volatility Regimes**—identifying when the market is "Calm," "Transitioning," or in "Crisis." 

Built for hedge fund traders, risk managers, and policymakers, OVIP uses a dual-model machine learning architecture combined with an NLP-powered Retrieval-Augmented Generation (RAG) assistant to provide actionable intelligence.

### 🏆 Key ML Finding: "The Crisis Memory" Hypothesis
We tested whether feeding a model historical "trauma" (the 2000-2001 Dot-Com & 9/11 crashes) improves its ability to predict modern crises (like the 2022 Energy Crisis).

| Model Variant | Training Data | Test $R^2$ (2021-2025) | Insight |
| :--- | :--- | :--- | :--- |
| **Standard Model** | 2002–2025 | 17.66% | Missed early shock signals. |
| **Extended Model** | **2000–2025** | **18.59%** | **+0.93% Accuracy Gain** |

**Verdict:** Models require "Crisis Memory." By learning from the 2000 shock, the RF-11 model was ~1% better at predicting the volatility of the 2022 energy crisis out-of-sample.

---

## 🧠 Core Architecture

OVIP utilizes a multi-layered intelligence stack:
1. **NPRS-1 Binary Classifier:** Predicts volatility direction (UP/DOWN) with 68.2% accuracy.
2. **11-Pillar Regression Model:** Predicts exact volatility levels.
3. **Regime Detection System:** Uses Markov switching to classify market states.
4. **OVIP AI Assistant:** A RAG-powered chatbot (via Hugging Face) that reads live CSV data to answer policy and hedging questions without hallucinating.
5. **3D Geospatial Engine:** Interactive global supply chain mapping using Plotly.

---

## 📂 Repository Structure

The codebase follows a strict, modular, enterprise-grade Streamlit architecture:

```text
ovip/
├── app.py                          # Application Entry Point
├── requirements.txt                # Python dependencies
├── config.py                       # Global Theme & Configurations
├── README.md                       # Documentation
├── .env / .streamlit/secrets.toml  # Environment variables (Ignored)
│
├── data/                           # Data directory
│   ├── merged_final.csv            # Historical master data
│   ├── HONEST_REGIME_PROBS.csv     # Regime probabilities
│   ├── data_model_performance_2025.csv  
│   └── models/                     # Pickled trained models
│       ├── nprs1_classifier.pkl
│       ├── rf11_regressor.pkl
│       └── scaler.pkl
│
├── modules/                        # Core Logic Layer
│   ├── data_loader.py              # I/O operations
│   ├── feature_engineering.py      # Feature creation
│   ├── models.py                   # Model inference
│   ├── forecasting.py              # Temporal predictions
│   ├── nlp_sentiment.py            # Text analysis
│   ├── regime_detection.py         # Regime switching logic
│   ├── visualization.py            # Charting engine
│   ├── globe_viz.py                # 3D spatial rendering
│   └── chatbot.py                  # RAG AI integration
│
├── pages/                          # Streamlit Multipage UI
│   ├── 1_🌍_Country_Selector.py
│   ├── 2_📊_Dashboard.py           # Main Command Center
│   ├── 3_💬_AI_Assistant.py        
│   ├── 4_📈_Analytics.py
│   ├── 5_🔔_Alerts.py
│   ├── 6_📄_Reports.py
│   └── 7_⚙️_Settings.py
│
├── utils/                          # Utilities
│   ├── helpers.py                  
│   ├── api_client.py               
│   └── notifications.py            
│
└── assets/                         # Static files (CSS/Fonts)
