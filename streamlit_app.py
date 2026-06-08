
import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Market Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
    }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        color: #00d4aa;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .persona-card {
        background: #1a1d27;
        border-left: 4px solid;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .persona-name {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .persona-desc {
        font-size: 0.85rem;
        color: #aaa;
        line-height: 1.5;
    }
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #555;
        margin-bottom: 16px;
        border-bottom: 1px solid #2a2d3a;
        padding-bottom: 8px;
    }
    .stSlider > div > div {
        background: #00d4aa !important;
    }
    div[data-testid="stSelectbox"] label {
        color: #aaa !important;
        font-size: 0.85rem !important;
    }
    .result-box {
        background: #1a1d27;
        border: 1px solid #00d4aa;
        border-radius: 8px;
        padding: 24px;
        text-align: center;
    }
    .uplift-positive {
        color: #00d4aa;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


#df = pd.read_csv('marketing_campaign.csv', sep=None, engine='python')
#st.dataframe(df.head()-86




# ─────────────────────────────────────────────
# LOAD & TRAIN MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_and_train(path='marketing_campaign.csv'):
    # Auto-detect separator — Kaggle version is tab-separated
    try:
        df = pd.read_csv(path, sep='\t')
        if df.shape[1] < 5:
            df = pd.read_csv(path, sep=',')
    except Exception:
        df = pd.read_csv(path, sep=',')
    df = df.dropna(subset=['Income'])

    #df['Age'] = 2024 - df['Year_Birth']
    df['TotalSpend'] = df[['MntWines','MntFruits','MntMeatProducts',
                            'MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)
    df['TotalChildren'] = df['Kidhome'] + df['Teenhome']
    df['TotalPurchases'] = df[['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']].sum(axis=1)
    df['CampaignAccepted'] = df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                                  'AcceptedCmp4','AcceptedCmp5','Response']].sum(axis=1)

    features = ['Income','Age','TotalSpend','TotalChildren',
                'TotalPurchases','CampaignAccepted','MntWines','MntMeatProducts']

    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    #profile = df.groupby('Cluster')[features + ['TotalSpend']].mean().round(1)
    #profile['Size'] = df.groupby('Cluster').size()
    #profile['Size_pct'] = (profile['Size'] / len(df) * 100).round(1)
    #profile = profile.reset_index(drop=True)
    
    profile = df.groupby('Cluster')[features].mean().round(1) 
    profile['TotalSpend'] = df.groupby('Cluster')['TotalSpend'].mean().round(1) 
    profile['Size'] = df.groupby('Cluster').size() 
    profile['Size_pct'] = (profile['Size'] / len(df) * 100).round(1) 
    profile = profile.reset_index(drop=True)

    return df, kmeans, scaler, profile, features


# ─────────────────────────────────────────────
# PERSONA DEFINITIONS
# ─────────────────────────────────────────────
CLUSTER_COLORS = ['#4e9af1', '#f4a261', '#00d4aa', '#e76f51']

PERSONAS = {
    0: {
        'name': 'The Comfortable Spender',
        'emoji': '💼',
        'description': 'Mid-to-high income, solid purchase frequency, moderate campaign response. Not fully maximized yet.',
        'action': 'Upsell with loyalty rewards and premium product trials.',
        'color': '#4e9af1'
    },
    1: {
        'name': 'The Budget Household',
        'emoji': '👨‍👩‍👧',
        'description': 'Multiple children, lower income, price-sensitive. Buys but spends conservatively.',
        'action': 'Target with family bundles and discount campaigns.',
        'color': '#f4a261'
    },
    2: {
        'name': 'The High-Value Loyalist',
        'emoji': '⭐',
        'description': 'High income, top spender across wine and meat, responds to multiple campaigns.',
        'action': 'Retain with exclusive perks. Focus on quality messaging, not discounts.',
        'color': '#00d4aa'
    },
    3: {
        'name': 'The Constrained Active',
        'emoji': '📌',
        'description': 'Lower income (~$31K), modest spend, mid-age. Actively buying but economically limited.',
        'action': 'Offer value-tier products and instalment-friendly promotions.',
        'color': '#e76f51'
    }
}


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Market Segmentation")
    st.markdown("<div class='section-header'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", [
        "📊 Cluster Overview",
        "🔮 Predict My Segment",
        "💰 Revenue Simulator",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div class='section-header'>Dataset</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload marketing_campaign.csv", type=['csv'])

    st.markdown("---")
    st.caption("iFood Marketing Analytics · K-Means (K=4)")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
if uploaded:
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    try:
        # Use file name + size as cache key so re-uploads work
        load_and_train.clear()
        df, kmeans, scaler, profile, features = load_and_train(tmp_path)
        data_loaded = True
    except Exception as e:
        st.error(f"Error loading file: {e}")
        data_loaded = False
elif os.path.exists('marketing_campaign.csv'):
    df, kmeans, scaler, profile, features = load_and_train()
    data_loaded = True
else:
    data_loaded = False


# ─────────────────────────────────────────────
# PAGE: CLUSTER OVERVIEW
# ─────────────────────────────────────────────
if page == "📊 Cluster Overview":
    st.markdown("# Cluster Overview")
    st.markdown("<div class='section-header'>4 distinct customer segments identified</div>", unsafe_allow_html=True)

    if not data_loaded:
        st.info("⬅️ Upload `marketing_campaign.csv` in the sidebar to load your data.")

        # Show static persona cards even without data
        st.markdown("### Customer Personas")
        for cid, p in PERSONAS.items():
            st.markdown(f"""
            <div class='persona-card' style='border-color: {p['color']}'>
                <div class='persona-name' style='color: {p['color']}'>{p['emoji']} Cluster {cid} — {p['name']}</div>
                <div class='persona-desc'>{p['description']}</div>
                <div class='persona-desc' style='margin-top:8px; color: #ccc'>
                    <strong>Marketing Action:</strong> {p['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Metric cards
        cols = st.columns(4)
        for i, col in enumerate(cols):
            p = PERSONAS[i]
           # size = int(profile.loc[i, 'Size'])
            #pct = float(profile.loc[i, 'Size_pct'])
            #spend = int(profile.loc[i, 'TotalSpend'])
            
            size = int(profile['Size'].iloc[i]) 
            pct = float(profile['Size_pct'].iloc[i]) 
            spend = float(profile['TotalSpend'].iloc[i])
            with col:
                st.markdown(f"""
                <div class='metric-card' style='border-top: 3px solid {p['color']}'>
                    <div style='font-size:1.5rem'>{p['emoji']}</div>
                    <div class='metric-value' style='color:{p['color']}'>{size}</div>
                    <div class='metric-label'>{p['name']}</div>
                    <div style='font-size:0.8rem; color:#888; margin-top:6px'>{pct}% · ${spend:,.0f} avg spend</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Cluster Profiles")
            for cid, p in PERSONAS.items():
                #inc = profile.loc[cid, 'Income']
                #spend = profile.loc[cid, 'TotalSpend']
                #age = profile.loc[cid, 'Age']
                
                inc = float(profile['Income'].iloc[cid]) 
                spend = float(profile['TotalSpend'].iloc[cid]) 
                age = float(profile['Age'].iloc[cid])
                
                
                
                st.markdown(f"""
                <div class='persona-card' style='border-color: {p['color']}'>
                    <div class='persona-name' style='color: {p['color']}'>{p['emoji']} {p['name']}</div>
                    <div class='persona-desc'>
                        Income: ${inc:,.0f} · Spend: ${spend:,.0f}/yr · Age: {age:.0f}
                    </div>
                    <div class='persona-desc' style='margin-top:6px'>{p['description']}</div>
                    <div class='persona-desc' style='margin-top:6px;color:#ccc'>
                        🎯 {p['action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Spend Distribution by Cluster")
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#0f1117')

            for cid in range(4):
                cluster_spend = df[df['Cluster'] == cid]['TotalSpend']
                ax.hist(cluster_spend, bins=30, alpha=0.6,
                        color=CLUSTER_COLORS[cid],
                        label=f"C{cid}: {PERSONAS[cid]['name']}")

            ax.set_xlabel('Total Annual Spend ($)', color='#aaa')
            ax.set_ylabel('Count', color='#aaa')
            ax.tick_params(colors='#aaa')
            ax.spines['bottom'].set_color('#333')
            ax.spines['left'].set_color('#333')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(fontsize=8, facecolor='#1a1d27', labelcolor='white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ─────────────────────────────────────────────
# PAGE: PREDICT MY SEGMENT
# ─────────────────────────────────────────────
elif page == "🔮 Predict My Segment":
    st.markdown("# Segment Predictor")
    st.markdown("<div class='section-header'>Enter customer details to predict their segment</div>", unsafe_allow_html=True)

    if not data_loaded:
        st.info("⬅️ Upload `marketing_campaign.csv` in the sidebar first.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Customer Details")
            income = st.slider("Annual Income ($)", 5000, 120000, 50000, step=1000)
            age = st.slider("Age", 18, 80, 40)
            children = st.slider("Number of Children (kids + teens)", 0, 4, 1)
            purchases = st.slider("Total Purchases per year", 0, 30, 10)
            campaigns = st.slider("Campaigns accepted (out of 6)", 0, 6, 1)
            wine_spend = st.slider("Annual Wine Spend ($)", 0, 1500, 200)
            meat_spend = st.slider("Annual Meat Spend ($)", 0, 1800, 150)

            total_spend = wine_spend + meat_spend + st.number_input(
                "Other product spend (fruits, fish, sweets, gold) $", 0, 2000, 100)

        with col2:
            st.markdown("### Prediction Result")

            input_data = np.array([[income, age, total_spend, children,
                                    purchases, campaigns, wine_spend, meat_spend]])
            input_scaled = scaler.transform(input_data)
            predicted_cluster = int(kmeans.predict(input_scaled)[0])
            persona = PERSONAS[predicted_cluster]

            st.markdown(f"""
            <div class='result-box'>
                <div style='font-size: 3rem'>{persona['emoji']}</div>
                <div style='font-family: IBM Plex Mono; font-size: 0.7rem; color: #555;
                            text-transform: uppercase; letter-spacing: 3px; margin: 8px 0'>
                    Cluster {predicted_cluster}
                </div>
                <div style='font-size: 1.4rem; font-weight: 600; color: {persona['color']};
                            font-family: IBM Plex Mono; margin-bottom: 12px'>
                    {persona['name']}
                </div>
                <div style='color: #aaa; font-size: 0.9rem; line-height: 1.6; margin-bottom: 16px'>
                    {persona['description']}
                </div>
                <div style='background: #0f1117; border-radius: 6px; padding: 12px;
                            font-size: 0.85rem; color: #ccc; text-align: left'>
                    <strong style='color: {persona['color']}'>🎯 Recommended Action</strong><br>
                    {persona['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Radar chart for this customer vs cluster average
            st.markdown("### Your Profile vs Cluster Average")
            radar_features = ['Income', 'TotalSpend', 'TotalPurchases', 'CampaignAccepted', 'TotalChildren']
            cluster_avg = profile.loc[predicted_cluster, radar_features].values

            # Normalize both using dataset min/max
            feat_min = df[radar_features].min().values
            feat_max = df[radar_features].max().values
            user_vals = np.array([income, total_spend, purchases, campaigns, children])
            user_norm = (user_vals - feat_min) / (feat_max - feat_min + 1e-9)
            cluster_norm = (cluster_avg - feat_min) / (feat_max - feat_min + 1e-9)

            N = len(radar_features)
            angles = [n / N * 2 * np.pi for n in range(N)] + [0]
            user_vals_plot = user_norm.tolist() + [user_norm[0]]
            cluster_vals_plot = cluster_norm.tolist() + [cluster_norm[0]]

            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#1a1d27')

            ax.plot(angles, user_vals_plot, color='#ffffff', linewidth=2, label='You')
            ax.fill(angles, user_vals_plot, color='#ffffff', alpha=0.15)
            ax.plot(angles, cluster_vals_plot, color=persona['color'], linewidth=2, label='Cluster avg')
            ax.fill(angles, cluster_vals_plot, color=persona['color'], alpha=0.2)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_features, color='#aaa', size=9)
            ax.set_yticklabels([])
            ax.spines['polar'].set_color('#333')
            ax.grid(color='#333')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                      facecolor='#1a1d27', labelcolor='white', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ─────────────────────────────────────────────
# PAGE: REVENUE SIMULATOR
# ─────────────────────────────────────────────
elif page == "💰 Revenue Simulator":
    st.markdown("# Revenue Impact Simulator")
    st.markdown("<div class='section-header'>What-if analysis: converting low-spend to high-spend behavior</div>",
                unsafe_allow_html=True)

    if not data_loaded:
        st.info("⬅️ Upload `marketing_campaign.csv` in the sidebar first.")
    else:
        low_cluster  = int(profile['TotalSpend'].idxmin())
        high_cluster = int(profile['TotalSpend'].idxmax())

        avg_low  = profile.loc[low_cluster,  'TotalSpend']
        avg_high = profile.loc[high_cluster, 'TotalSpend']
        n_low    = int(profile.loc[low_cluster, 'Size'])

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Simulation Controls")

            source_cluster = st.selectbox(
                "Source cluster (low-spend)",
                options=list(range(4)),
                index=low_cluster,
                format_func=lambda x: f"Cluster {x} — {PERSONAS[x]['name']}"
            )
            target_cluster = st.selectbox(
                "Target cluster (high-spend)",
                options=list(range(4)),
                index=high_cluster,
                format_func=lambda x: f"Cluster {x} — {PERSONAS[x]['name']}"
            )

            conversion_rate = st.slider(
                "Conversion Rate — what % of source customers reach target behavior?",
                min_value=1, max_value=50, value=15, step=1
            )

            # Recalculate based on selections
            s_avg   = profile.loc[source_cluster, 'TotalSpend']
            t_avg   = profile.loc[target_cluster, 'TotalSpend']
            n_source = int(profile.loc[source_cluster, 'Size'])

            n_converted    = int(n_source * conversion_rate / 100)
            n_unchanged    = n_source - n_converted
            rev_before     = n_source * s_avg
            rev_after      = (n_unchanged * s_avg) + (n_converted * t_avg)
            uplift_dollars = rev_after - rev_before
            uplift_pct     = (uplift_dollars / rev_before) * 100 if rev_before > 0 else 0

            threshold = df[df['Cluster'] == target_cluster]['TotalSpend'].quantile(0.25)

        with col2:
            st.markdown("### Results")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{n_converted}</div>
                    <div class='metric-label'>Customers Converted</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='color:#00d4aa'>${uplift_dollars:,.0f}</div>
                    <div class='metric-label'>Revenue Uplift</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='color:#00d4aa'>+{uplift_pct:.1f}%</div>
                    <div class='metric-label'>Uplift %</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:#1a1d27; border:1px solid #2a2d3a; border-radius:8px;
                        padding:16px; margin-top:16px; font-size:0.85rem; color:#aaa; line-height:1.7'>
                <strong style='color:#fff'>🎯 Conversion Threshold</strong><br>
                A customer in <strong style='color:{PERSONAS[source_cluster]['color']}'>{PERSONAS[source_cluster]['name']}</strong>
                needs to reach <strong style='color:#00d4aa'>${threshold:,.0f}/year</strong> in total spend
                to enter the behavioral range of
                <strong style='color:{PERSONAS[target_cluster]['color']}'>{PERSONAS[target_cluster]['name']}</strong>.
            </div>
            """, unsafe_allow_html=True)

        # Simulation curve
        st.markdown("---")
        st.markdown("### Uplift Across All Conversion Rates")

        rates = list(range(1, 51))
        uplifts = []
        for r in rates:
            nc = int(n_source * r / 100)
            nu = n_source - nc
            ra = (nu * s_avg) + (nc * t_avg)
            rb = n_source * s_avg
            uplifts.append(ra - rb)

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#0f1117')

        ax.fill_between(rates, uplifts, alpha=0.2, color='#00d4aa')
        ax.plot(rates, uplifts, color='#00d4aa', linewidth=2.5)
        ax.axvline(x=conversion_rate, color='white', linestyle='--', alpha=0.6,
                   label=f'Current: {conversion_rate}%')
        ax.axhline(y=uplift_dollars, color='#f4a261', linestyle=':', alpha=0.5)

        ax.set_xlabel('Conversion Rate (%)', color='#aaa')
        ax.set_ylabel('Revenue Uplift ($)', color='#aaa')
        ax.tick_params(colors='#aaa')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#333')
        ax.legend(facecolor='#1a1d27', labelcolor='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # KDE threshold plot
        st.markdown("### Spend Distributions & Conversion Threshold")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.patch.set_facecolor('#0f1117')
        ax2.set_facecolor('#0f1117')

        src_data = df[df['Cluster'] == source_cluster]['TotalSpend']
        tgt_data = df[df['Cluster'] == target_cluster]['TotalSpend']

        from scipy.stats import gaussian_kde
        for data, color, label in [
            (src_data, PERSONAS[source_cluster]['color'], PERSONAS[source_cluster]['name']),
            (tgt_data, PERSONAS[target_cluster]['color'], PERSONAS[target_cluster]['name'])
        ]:
            kde = gaussian_kde(data)
            x_range = np.linspace(0, data.max() * 1.1, 300)
            ax2.fill_between(x_range, kde(x_range), alpha=0.25, color=color)
            ax2.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)

        ax2.axvline(x=threshold, color='white', linestyle='--', linewidth=1.5,
                    label=f'Threshold: ${threshold:,.0f}')
        ax2.set_xlabel('Total Annual Spend ($)', color='#aaa')
        ax2.set_ylabel('Density', color='#aaa')
        ax2.tick_params(colors='#aaa')
        for spine in ['top', 'right']:
            ax2.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax2.spines[spine].set_color('#333')
        ax2.legend(facecolor='#1a1d27', labelcolor='white', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()