# 🛒 Shopper Spectrum — Streamlit App

## Project: Customer Segmentation and Product Recommendations in E-Commerce

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place required files in the same directory as `app.py`

**Option A — Pre-trained models (recommended):**  
Copy these outputs from the Jupyter Notebook run:
```
kmeans_model.pkl
scaler.pkl
product_similarity.pkl
label_map.pkl
```

**Option B — Raw CSV (fallback):**  
Place `online_retail.csv` alongside `app.py`.  
The app will train and cache models automatically on first launch.

### 3. Launch the app
```bash
streamlit run app.py
```

---

## Features

### Product Recommendation Module
- Text input or dropdown to select a product
- Partial, case-insensitive product name matching
- Returns top-N similar products (3–10, slider-controlled)
- Powered by **Item-Based Collaborative Filtering (Cosine Similarity)**
- Results exportable as CSV

### Customer Segmentation Module
- Input: Recency (days), Frequency (orders), Monetary (£)
- Predicts one of four segments:
  - **High-Value** — frequent, recent, big spenders
  - **Regular** — steady mid-tier purchasers
  - **Occasional** — infrequent, opportunistic buyers
  - **At-Risk** — haven't purchased in a long time
- Displays business action recommendations per segment

### Overview Dashboard
- Live dataset stats (customers, products, revenue, countries)
- Architecture summary and segment reference guide

---

## File Structure
```
app.py                  ← Main Streamlit application
requirements.txt        ← Python dependencies
kmeans_model.pkl        ← Trained KMeans model
scaler.pkl              ← StandardScaler
product_similarity.pkl  ← Cosine similarity matrix
label_map.pkl           ← Cluster-to-label mapping
online_retail.csv       ← Raw dataset (fallback)
```
