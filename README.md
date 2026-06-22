# 🛒 Shopper Spectrum

### Customer Segmentation & Product Recommendation System for E-Commerce

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-KMeans-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Recommendation-Collaborative%20Filtering-orange?style=for-the-badge">
</p>

<p align="center">
  <b>Transforming E-Commerce Transaction Data into Actionable Business Intelligence</b>
</p>

---

# 📌 Project Overview

Shopper Spectrum is an end-to-end Machine Learning and Data Analytics project designed to analyze customer purchasing behavior in an e-commerce environment.

The project focuses on two major business problems:

### 🎯 Customer Segmentation

Identify different groups of customers based on purchasing behavior using RFM Analysis and K-Means Clustering.

### 🛍️ Product Recommendation

Recommend similar products using Item-Based Collaborative Filtering powered by Cosine Similarity.

The final solution is deployed as an interactive Streamlit web application that allows users to generate predictions and recommendations in real time.

---

# 🚀 Business Problem

Modern e-commerce platforms generate millions of transactions daily.

However, businesses often struggle to answer questions such as:

* Which customers are most valuable?
* Which customers are likely to churn?
* What products should be recommended next?
* How can marketing campaigns be personalized?

This project addresses these challenges using Data Science and Machine Learning techniques.

---

# 🎯 Project Objectives

✅ Clean and preprocess raw retail transaction data

✅ Perform Exploratory Data Analysis (EDA)

✅ Engineer RFM features

✅ Segment customers using KMeans Clustering

✅ Build Product Recommendation Engine

✅ Evaluate clustering quality

✅ Deploy models using Streamlit

---

# 📂 Dataset Information

The project utilizes an Online Retail Dataset containing customer transactions.

| Feature     | Description         |
| ----------- | ------------------- |
| InvoiceNo   | Transaction Number  |
| StockCode   | Product Identifier  |
| Description | Product Name        |
| Quantity    | Quantity Purchased  |
| InvoiceDate | Purchase Date       |
| UnitPrice   | Product Price       |
| CustomerID  | Customer Identifier |
| Country     | Customer Location   |

---

# 🔄 Complete Project Workflow

```text
Raw Retail Data
        │
        ▼
Data Cleaning & Wrangling
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering (RFM)
        │
        ▼
Data Scaling
        │
        ▼
KMeans Clustering
        │
        ▼
Customer Segmentation
        │
        ▼
Streamlit Deployment
```

---

# 🧹 Data Preprocessing

The following cleaning operations were performed:

### Missing Value Handling

* Removed records with missing CustomerID

### Cancelled Order Removal

* Removed invoices starting with "C"

### Invalid Record Removal

* Removed:

  * Negative Quantities
  * Zero Quantities
  * Negative Prices
  * Zero Prices

### Feature Creation

```python
TotalPrice = Quantity × UnitPrice
```

This feature was used to calculate customer spending behavior.

---

# 📊 Exploratory Data Analysis

The following analyses were performed:

### 🌍 Transaction Analysis by Country

Identify countries contributing the highest number of transactions.

### 🏆 Top Selling Products

Discover products generating maximum sales.

### 📈 Revenue Trends

Analyze purchasing patterns over time.

### 💰 Customer Spending Distribution

Study monetary behavior across customers.

### 📉 Correlation Analysis

Understand relationships between important features.

---

# ⚙️ Feature Engineering

## RFM Analysis

Three business-critical features were created:

### 🕒 Recency

Days since customer's last purchase.

### 🔄 Frequency

Number of purchases made by customer.

### 💵 Monetary

Total spending by customer.

```text
RFM = Customer Purchasing Behaviour
```

---

# 📏 Data Scaling

StandardScaler was used to normalize RFM features.

### Why?

Without scaling:

```text
Recency = 10
Frequency = 50
Monetary = 50000
```

Monetary values dominate clustering.

Scaling ensures fair contribution from all features.

---

# 🤖 Machine Learning Implementation

## Customer Segmentation

### Algorithm Used

```python
KMeans Clustering
```

### Cluster Selection

The optimal number of clusters was determined using:

* Elbow Method
* Silhouette Score

### Final Value

```text
K = 4
```

---

## Customer Segments

| Segment       | Description                     |
| ------------- | ------------------------------- |
| 💎 High-Value | Frequent, recent, high spenders |
| 🔵 Regular    | Consistent medium spenders      |
| 🟡 Occasional | Infrequent customers            |
| 🔴 At-Risk    | Customers likely to churn       |

---

# 🛍️ Product Recommendation Engine

### Technique Used

```python
Item-Based Collaborative Filtering
```

### Similarity Metric

```python
Cosine Similarity
```

### Workflow

```text
Customer Purchases
        │
        ▼
Customer × Product Matrix
        │
        ▼
Cosine Similarity
        │
        ▼
Top 5 Product Recommendations
```

### Example

Input:

```text
WHITE HANGING HEART T-LIGHT HOLDER
```

Output:

```text
Top 5 Similar Products
```

based on customer purchase behavior.

---

# 📈 Hypothesis Testing

Statistical tests were used to validate insights.

### Tests Performed

#### Welch T-Test

Compare customer spending between groups.

#### ANOVA

Compare spending behavior across multiple customer segments.

### Purpose

Validate that customer groups are statistically different and not randomly formed.

---

# 🌐 Streamlit Application

The project was deployed using Streamlit for real-time predictions.

---

## 🏠 Overview Dashboard

Displays:

* Total Customers
* Total Products
* Transactions
* Revenue
* Countries

---

## 🛍️ Product Recommendation Module

### Input

Product Name

### Output

Top 5 Similar Products

### Additional Features

* Product Search
* Dropdown Selection
* CSV Export

---

## 👥 Customer Segmentation Module

### Inputs

* Recency
* Frequency
* Monetary

### Output

Customer Segment

Examples:

```text
💎 High-Value
🔵 Regular
🟡 Occasional
🔴 At-Risk
```

Along with business recommendations.

---

# 💻 Technologies Used

| Category          | Technologies            |
| ----------------- | ----------------------- |
| Programming       | Python                  |
| Data Analysis     | Pandas, NumPy           |
| Visualization     | Matplotlib, Seaborn     |
| Machine Learning  | Scikit-Learn            |
| Clustering        | KMeans                  |
| Recommendation    | Collaborative Filtering |
| Similarity Metric | Cosine Similarity       |
| Deployment        | Streamlit               |

---

# 📁 Project Structure

```bash
Shopper-Spectrum/
│
├── Vamsee_Shopper_Spectrum_Submission.ipynb
├── app.py
├── online_retail.csv
│
├── kmeans_model.pkl
├── scaler.pkl
├── label_map.pkl
├── product_similarity.pkl
│
├── requirements.txt
├── README.md
│
└── assets/
```

---

# 📊 Key Outcomes

### Customer Segmentation

* Identified 4 customer groups
* Improved customer understanding
* Enabled targeted marketing strategies

### Product Recommendation

* Personalized product suggestions
* Improved customer experience
* Increased cross-selling opportunities

### Business Value

* Better customer retention
* Improved revenue generation
* Enhanced decision-making

---

# 🎓 Skills Demonstrated

✅ Data Cleaning

✅ Data Wrangling

✅ Exploratory Data Analysis

✅ Feature Engineering

✅ Statistical Analysis

✅ Data Scaling

✅ Customer Segmentation

✅ Recommendation Systems

✅ Machine Learning

✅ Streamlit Deployment

---

# 🔮 Future Improvements

* Real-Time Recommendation Engine
* User Authentication
* Cloud Deployment
* Advanced Recommendation Algorithms
* Customer Lifetime Value Prediction
* Deep Learning Based Recommendations

---

# 👨‍💻 Author

### Vemulapalli Vamsee Krishna

🎓 B.Tech CSE

---

# ⭐ If you found this project useful, consider giving it a star!
