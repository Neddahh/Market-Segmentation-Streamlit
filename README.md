# 🎯 Market Segmentation & Revenue Impact Simulator

A deployed machine learning web app that segments customers from the iFood Marketing Analytics dataset into distinct behavioral personas and simulates revenue impact of targeted marketing campaigns.

**Live Demo:** https://market-segmentation-app-2.streamlit.app

---

## Features

### 📊 Cluster Overview
Visualizes all 4 customer segments identified by K-Means clustering. Each segment is presented as a named persona with average income, spending, age, and a recommended marketing action. Includes a spend distribution chart across all clusters.

### 🔮 Segment Predictor
Enter a customer's demographic and behavioral details — income, age, number of children, annual spend, and campaign history — and the app predicts which segment they belong to in real time. Returns the persona name, description, marketing recommendation, and a radar chart comparing the customer's profile against their cluster average.

### 💰 Revenue Impact Simulator
An interactive what-if analysis tool. Select a source and target cluster, adjust the conversion rate slider, and instantly see the projected revenue uplift in dollars and percentage. Includes a full simulation curve across all conversion rates and a KDE spend distribution chart showing the spending threshold a customer must cross to behaviorally convert to the target segment.

---

## Dataset
[iFood Marketing Analytics — Kaggle](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign)  
2,240 customers · 29 features · demographic, behavioral, and campaign response data

---

## Methods
- K-Means Clustering (K=4, validated with Elbow Method + Silhouette Score)
- Hierarchical Clustering (Dendrogram validation)
- PCA for 2D cluster visualization
- StandardScaler for feature normalization
- Revenue simulation via cluster spend delta analysis

## Tech Stack
`Python` `Streamlit` `scikit-learn` `pandas` `matplotlib` `scipy`

---

## Run Locally
```bash
git clone https://github.com/Neddahh/Market-Segmentation-Streamlit
cd Market-Segmentation-Streamlit
pip install -r requirements.txt
streamlit run streamlit_app.py
```
