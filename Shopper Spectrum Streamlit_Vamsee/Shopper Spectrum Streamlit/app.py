"""
Shopper Spectrum: Customer Segmentation and Product Recommendations
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# 
# PAGE CONFIG
# 
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 
# CUSTOM CSS
# 
st.markdown("""
<style>
    /* ---- Fonts & base ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---- Header banner ---- */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #e0e0e0; font-size: 2.4rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { color: #a0c4ff; font-size: 1rem; margin: 0.5rem 0 0; }

    /* ---- Section headings ---- */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 4px solid #0f3460;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: white;
        border: 1px solid #e8edf2;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #0f3460; }
    .metric-card .label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }

    /* ---- Recommendation cards ---- */
    .rec-card {
        background: linear-gradient(135deg, #f8faff 0%, #eef4ff 100%);
        border: 1px solid #c7d9f8;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        transition: box-shadow 0.2s;
    }
    .rec-card:hover { box-shadow: 0 4px 16px rgba(15,52,96,0.15); }
    .rec-number {
        background: #0f3460;
        color: white;
        border-radius: 50%;
        width: 30px; height: 30px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.85rem; flex-shrink: 0;
    }
    .rec-name { color: #1a1a2e; font-weight: 500; font-size: 0.95rem; }

    /* ---- Segment badge ---- */
    .segment-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin: 0.5rem 0;
    }
    .badge-highvalue { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .badge-regular   { background: #cce5ff; color: #004085; border: 2px solid #007bff; }
    .badge-occasional{ background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .badge-atrisk    { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }

    /* ---- Segment description card ---- */
    .seg-desc {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e8edf2;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .seg-desc h4 { margin: 0 0 0.5rem; color: #1a1a2e; font-size: 1rem; }
    .seg-desc p  { margin: 0; color: #4b5563; font-size: 0.9rem; line-height: 1.6; }

    /* ---- Info box ---- */
    .info-box {
        background: #f0f7ff;
        border-left: 4px solid #0f3460;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        font-size: 0.88rem;
        color: #374151;
        margin: 0.5rem 0 1rem;
    }

    /* ---- Divider ---- */
    .divider { height: 1px; background: linear-gradient(to right, #0f3460, transparent); margin: 2rem 0; }

    /* ---- Streamlit overrides ---- */
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #1a6db5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        font-size: 0.95rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; }
    .stNumberInput > div > div > input { border-radius: 8px; }
    .stTextInput > div > div > input  { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# 
# CONSTANTS & SEGMENT CONFIG
# 
SEGMENT_CONFIG = {
    "High-Value": {
        "badge_class": "badge-highvalue",
        "icon": "💎",
        "tagline": "Your best customers",
        "description": (
            "These customers purchase frequently, recently, and spend the most. "
            "They are your brand advocates. Focus on loyalty rewards, early access, "
            "and VIP perks to retain them."
        ),
        "actions": ["Offer VIP loyalty rewards", "Invite to exclusive previews", "Personalized upsells"],
    },
    "Regular": {
        "badge_class": "badge-regular",
        "icon": "🔵",
        "tagline": "Steady, reliable purchasers",
        "description": (
            "Consistent buyers with moderate spend and frequency. They represent "
            "stable revenue. Upselling and cross-selling campaigns can convert them "
            "to High-Value customers."
        ),
        "actions": ["Cross-sell related products", "Introduce premium tiers", "Periodic engagement emails"],
    },
    "Occasional": {
        "badge_class": "badge-occasional",
        "icon": "🟡",
        "tagline": "Infrequent, opportunistic buyers",
        "description": (
            "These customers buy occasionally, likely during sales or promotions. "
            "Targeted discounts, seasonal campaigns, and nudge strategies can increase "
            "their purchase frequency."
        ),
        "actions": ["Seasonal discount campaigns", "Re-engagement emails", "Wishlist reminders"],
    },
    "At-Risk": {
        "badge_class": "badge-atrisk",
        "icon": "🔴",
        "tagline": "Churning - act fast",
        "description": (
            "Haven't purchased in a long time despite past activity. High churn risk. "
            "Win-back campaigns, exclusive comeback offers, and feedback surveys are "
            "critical to re-engage them before they're lost."
        ),
        "actions": ["Win-back discount offers", "Churn survey to gather feedback", "Re-activation email sequence"],
    },
}

# 
# MODEL LOADING
# 
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all pickled models. Falls back to on-the-fly training if pkl files are absent."""
    models = {}

    #  Try loading pre-trained pickles 
    pkl_files = {
        "kmeans":     "kmeans_model.pkl",
        "scaler":     "scaler.pkl",
        "similarity": "product_similarity.pkl",
        "label_map":  "label_map.pkl",
    }

    all_found = all(os.path.exists(p) for p in pkl_files.values())

    if all_found:
        for key, path in pkl_files.items():
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        return models, "pre-trained"

    #  Fallback: train fresh from CSV 
    csv_path = "online_retail.csv"
    if not os.path.exists(csv_path):
        return None, "no_data"

    df = pd.read_csv(csv_path, encoding="latin-1")

    # Preprocessing
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # RFM
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    # Scale & cluster
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X)

    # Label clusters
    cluster_profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    def assign_label(row):
        if row["Recency"] < 30 and row["Frequency"] >= 5:
            return "High-Value"
        elif row["Recency"] < 60 and row["Frequency"] >= 3:
            return "Regular"
        elif row["Recency"] >= 200:
            return "At-Risk"
        else:
            return "Occasional"

    label_map = cluster_profile.apply(assign_label, axis=1).to_dict()

    # Product similarity
    product_matrix = df.groupby(["CustomerID", "Description"])["Quantity"].sum().unstack(fill_value=0)
    item_matrix = product_matrix.T
    cosine_sim = cosine_similarity(item_matrix)
    product_similarity = pd.DataFrame(cosine_sim, index=item_matrix.index, columns=item_matrix.index)

    models = {
        "kmeans": kmeans,
        "scaler": scaler,
        "similarity": product_similarity,
        "label_map": label_map,
    }
    return models, "trained-on-the-fly"


@st.cache_data(show_spinner=False)
def get_dataset_stats():
    """Load CSV and return basic stats (cached)."""
    csv_path = "online_retail.csv"
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return {
        "customers": df["CustomerID"].nunique(),
        "products":  df["Description"].nunique(),
        "invoices":  df["InvoiceNo"].nunique(),
        "revenue":   df["TotalPrice"].sum(),
        "countries": df["Country"].nunique(),
    }


# 
# INFERENCE HELPERS
# 
def predict_segment(recency: float, frequency: float, monetary: float, models: dict) -> str:
    """Use KMeans + label_map to return segment string."""
    X = np.array([[recency, frequency, monetary]])
    X_scaled = models["scaler"].transform(X)
    cluster = int(models["kmeans"].predict(X_scaled)[0])
    return models["label_map"].get(cluster, "Occasional")


def get_recommendations(product_name: str, similarity: pd.DataFrame, top_n: int = 5):
    """Return top_n similar products (case-insensitive partial match)."""
    matches = [p for p in similarity.index if product_name.upper() in p.upper()]
    if not matches:
        return [], f'No product matching **"{product_name}"** found in the database.'
    product = matches[0]
    sim_scores = similarity[product].drop(index=product).sort_values(ascending=False)
    return sim_scores.head(top_n).index.tolist(), product


def get_all_products(similarity: pd.DataFrame):
    return sorted(similarity.index.tolist())


# 
# APP LAYOUT
# 
def main():
    #  Header 
    st.markdown("""
    <div class="main-header">
        <h1>🛒 Shopper Spectrum</h1>
        <p>Customer Segmentation &amp; Product Recommendations · E-Commerce Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    #  Load models 
    with st.spinner("Loading models…"):
        models, source = load_models()

    if models is None:
        st.error(
            "⚠️ Model files not found and `online_retail.csv` is also missing.\n\n"
            "Please place `kmeans_model.pkl`, `scaler.pkl`, `product_similarity.pkl`, "
            "`label_map.pkl` (from the notebook) **or** `online_retail.csv` "
            "in the same directory as `app.py`, then restart the app."
        )
        return

    if source == "trained-on-the-fly":
        st.info("ℹ️ Pre-trained `.pkl` files not found - models were trained from `online_retail.csv` automatically.")

    # ── Sidebar 
    with st.sidebar:
        st.markdown("##  Navigation")
        page = st.radio(
            "Navigation",
            [" Overview", " Product Recommendations", " Customer Segmentation"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("### ℹ About")
        st.markdown(
            "**Shopper Spectrum** analyzes an online retail dataset to:\n"
            "- Segment customers using **RFM + KMeans**\n"
            "- Recommend products via **cosine similarity**"
        )
        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.75rem;color:#9ca3af;text-align:center'>"
            "Shopper Spectrum · E-Commerce Analytics<br>Domain: BFSI &amp; Retail"
            "</div>",
            unsafe_allow_html=True,
        )

    #  Pages

    #  1. OVERVIEW 
    if page == " Overview":
        st.markdown('<div class="section-title"> Dataset Overview</div>', unsafe_allow_html=True)

        stats = get_dataset_stats()
        if stats:
            c1, c2, c3, c4, c5 = st.columns(5)
            cards = [
                (c1, f"{stats['customers']:,}",     "Unique Customers"),
                (c2, f"{stats['products']:,}",      "Unique Products"),
                (c3, f"{stats['invoices']:,}",      "Transactions"),
                (c4, f"£{stats['revenue']/1e6:.1f}M", "Total Revenue"),
                (c5, f"{stats['countries']}",        "Countries"),
            ]
            for col, val, lbl in cards:
                with col:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="value">{val}</div>'
                        f'<div class="label">{lbl}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("CSV not available for stats - place `online_retail.csv` next to `app.py`.")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title"> Project Architecture</div>', unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("####  Module 1 - Customer Segmentation")
            st.markdown("""
1. **RFM Feature Engineering** - Recency, Frequency, Monetary from raw transactions  
2. **StandardScaler** normalization  
3. **KMeans Clustering** (k=4, selected via Elbow + Silhouette)  
4. **Segment Labelling**: High-Value · Regular · Occasional · At-Risk  
""")

        with col_r:
            st.markdown("####  Module 2 - Product Recommendations")
            st.markdown("""
1. **Customer–Product Matrix** (pivot: CustomerID × Description, values = Quantity)  
2. **Item–Item Cosine Similarity** on transposed matrix  
3. **Top-5 recommendations** via partial product name match  
""")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title"> Customer Segments</div>', unsafe_allow_html=True)
        seg_cols = st.columns(4)
        segments = ["High-Value", "Regular", "Occasional", "At-Risk"]
        for col, seg in zip(seg_cols, segments):
            cfg = SEGMENT_CONFIG[seg]
            with col:
                st.markdown(
                    f'<div class="seg-desc">'
                    f'<h4>{cfg["icon"]} {seg}</h4>'
                    f'<p><em>{cfg["tagline"]}</em><br><br>{cfg["description"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    #  2. PRODUCT RECOMMENDATIONS
    elif page == " Product Recommendations":
        st.markdown('<div class="section-title"> Product Recommendation Engine</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Enter a product name (or keyword) and the system will return '
            'the 5 most similar products using <strong>Item-Based Collaborative Filtering</strong> '
            '(cosine similarity on customer purchase patterns).</div>',
            unsafe_allow_html=True,
        )

        # Two input options: text box or dropdown
        input_mode = st.radio("Input mode", [" Type a product name", " Pick from list"], horizontal=True)

        similarity = models["similarity"]
        all_products = get_all_products(similarity)

        product_input = ""
        if input_mode == " Type a product name":
            product_input = st.text_input(
                "Product Name",
                placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER",
                help="Partial matches are supported - case-insensitive.",
            )
        else:
            product_input = st.selectbox("Select a product", options=[""] + all_products)

        top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            btn_rec = st.button(" Get Recommendations")

        if btn_rec or product_input:
            if not product_input.strip():
                st.warning("Please enter or select a product name.")
            else:
                recs, matched = get_recommendations(product_input.strip(), similarity, top_n)
                if not recs:
                    st.error(matched)
                    st.markdown("**Try one of these popular products:**")
                    popular = all_products[:10]
                    for p in popular:
                        st.markdown(f"- `{p}`")
                else:
                    st.success(f"Showing top {len(recs)} recommendations for: **{matched}**")
                    st.markdown("####  Recommended Products")
                    for i, rec in enumerate(recs, 1):
                        st.markdown(
                            f'<div class="rec-card">'
                            f'<div class="rec-number">{i}</div>'
                            f'<div class="rec-name">{rec}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    st.markdown("####  Export Recommendations")
                    rec_df = pd.DataFrame({
                        "Rank": range(1, len(recs) + 1),
                        "Recommended Product": recs,
                        "Based On": matched,
                    })
                    st.dataframe(rec_df, use_container_width=True, hide_index=True)
                    csv = rec_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download as CSV",
                        data=csv,
                        file_name=f"recommendations_{matched[:20].replace(' ','_')}.csv",
                        mime="text/csv",
                    )

    # 3. CUSTOMER SEGMENTATION
    elif page == " Customer Segmentation":
        st.markdown('<div class="section-title"> Customer Segmentation Predictor</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Enter a customer\'s RFM profile to predict their segment using '
            'the trained <strong>KMeans model</strong>.</div>',
            unsafe_allow_html=True,
        )

        # ── Input form ───────
        with st.form("rfm_form"):
            st.markdown("#### Customer RFM Input")
            c1, c2, c3 = st.columns(3)

            with c1:
                recency = st.number_input(
                    " Recency (days since last purchase)",
                    min_value=1, max_value=1000, value=30,
                    help="Days elapsed since the customer last made a purchase.",
                )
            with c2:
                frequency = st.number_input(
                    " Frequency (number of orders)",
                    min_value=1, max_value=500, value=5,
                    help="Total number of unique invoices/orders.",
                )
            with c3:
                monetary = st.number_input(
                    " Monetary (total spend £)",
                    min_value=1.0, max_value=500000.0, value=1000.0, step=50.0,
                    help="Cumulative amount spent across all orders.",
                )

            submitted = st.form_submit_button(" Predict Segment")

        if submitted:
            segment = predict_segment(recency, frequency, monetary, models)
            cfg = SEGMENT_CONFIG[segment]

            st.markdown("---")
            st.markdown("####  Prediction Result")

            res_col, info_col = st.columns([1, 2])

            with res_col:
                st.markdown(
                    f'<div style="text-align:center;padding:1.5rem;">'
                    f'<div style="font-size:3rem">{cfg["icon"]}</div>'
                    f'<div class="segment-badge {cfg["badge_class"]}">{segment}</div>'
                    f'<p style="color:#6b7280;font-size:0.85rem;margin-top:0.5rem">{cfg["tagline"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with info_col:
                st.markdown(
                    f'<div class="seg-desc">'
                    f'<h4>About this segment</h4>'
                    f'<p>{cfg["description"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("**Recommended Actions:**")
                for action in cfg["actions"]:
                    st.markdown(f" {action}")

            st.markdown("---")
            # Summary card
            st.markdown("####  Input Summary")
            summary_df = pd.DataFrame({
                "Metric": ["Recency (days)", "Frequency (orders)", "Monetary (£)"],
                "Value": [recency, frequency, f"£{monetary:,.2f}"],
                "Interpretation": [
                    "Lower = more recent",
                    "Higher = more loyal",
                    "Higher = more valuable",
                ],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        #  Segment reference table
        with st.expander(" Segment Reference Table"):
            ref_data = {
                "Segment": ["High-Value 💎", "Regular 🔵", "Occasional 🟡", "At-Risk 🔴"],
                "Recency":   ["< 30 days", "< 60 days", "30–200 days", "> 200 days"],
                "Frequency": ["≥ 5 orders", "≥ 3 orders", "1–2 orders", "Any"],
                "Monetary":  ["High", "Medium", "Low–Medium", "Variable"],
                "Strategy":  [
                    "Retain with VIP rewards",
                    "Upsell & cross-sell",
                    "Promotions & nudges",
                    "Win-back campaigns",
                ],
            }
            st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)
            
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #6b7280; font-size: 0.9rem;'>Developed by Vemulapalli Vamsee Krishna</p>", unsafe_allow_html=True)

# 
# ENTRY POINT
# 
if __name__ == "__main__":
    main()
