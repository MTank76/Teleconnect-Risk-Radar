# 📡 Teleconnect-Risk-Radar: Project Documentation

This repository contains a comprehensive machine learning framework designed to predict customer churn for **Telco**, [a fictional dataset provided by IBM]. By leveraging historical customer demographics, service usage patterns, and billing data, this application provides high-accuracy churn risk analysis and actionable business insights through an interactive dashboard.

-----

## 🚀 Business Use Case: Predictive Customer Retention

The primary goal of this project is to assist telecom analysts and retention teams in identifying "at-risk" customers before they leave. Beyond simple binary predictions, the application quantifies the impact of variables like contract type, monthly charges, and technical support availability.

**Wider Industry Applications:**
The logic used in this project can be adapted for several high-value business sectors:

  * **SaaS & Subscription Services:** Identifying trial users unlikely to convert or long-term users showing signs of engagement fatigue.
  * **Finance/Banking:** Predicting credit card cancellations or account closures based on transaction frequency and fee structures.
  * **Retail Loyalty Programs:** Analyzing "purchase gaps" and "discount reliance" to identify customers likely to switch to competitors.
  * **Healthcare:** Predicting patient no-shows or plan switches based on demographic and service interaction history.
  * **Insurance:** Using ensemble models to predict policy non-renewals based on claim history and premium hikes.
  * **E-commerce:** Predicting "Customer Lifetime Value" and churn by analyzing "Recency, Frequency, and Monetary" (RFM) patterns.

-----

## 🛠️ The Approach & Methodology

The problem was approached as a **Binary Classification and Risk Estimation** challenge. The workflow followed these key phases:

1.  **Data Acquisition & Cleaning:** Telecom customer data was processed to handle categorical variables and normalize numerical fields like `Total Charges`.
2.  **Feature Importance:** Using machine learning drivers, I identified that **Total Charges**, **Monthly Charges**, and **Tenure** are the highest predictors of churn, followed by contract type and internet service (Fiber optic vs. DSL).
3.  **Model Architecture:** The system utilizes a multi-layered approach, offering a **Market Dashboard** for macro trends, an **Individual Predictor** for real-time risk assessment, and a **Batch Processing** terminal for large-scale file analysis.
4.  **Deployment:** The entire engine is wrapped in a high-fidelity **Streamlit** web application, providing a dark-themed, intuitive dashboard for executive and operational use.

-----

## 📊 Dashboard & Model Insights

The project provides three distinct viewpoints to analyze churn dynamics.

| View | Purpose | Key Metrics |
| :--- | :--- | :--- |
| **Strategic Insights** | Macro-level overview of the customer base. | Churn Rate (26.5%), Avg Tenure (32.4 mo), Avg Bill ($64.76). |
| **Individual Predictor** | Real-time "What-If" analysis for specific users. | Impacts of Contract length, Tech Support, and Payment Method. |
| **Batch Analysis** | Large-scale processing of subscriber lists. | CSV-based bulk risk scoring and data export. |

### Key Findings:

  * **Monthly Charges** and **Total Charges** account for the vast majority of predictive weight, suggesting price sensitivity is the primary churn driver.
  * **Contract Type:** Customers on Month-to-Month contracts show significantly higher churn volatility compared to those on one or two-year plans.
  * **Tech Support:** The presence of Technical Support services correlates strongly with higher customer retention.

> > **Key Insight:** Feature Importance analysis reveals that **Monthly Charges** and **Total Charges** account for \~35% of the prediction weight, suggesting price sensitivity is a major driver of churn.

-----

## 🖥️ Feature Highlights

### 📍 Individual Risk Terminal

An interactive sandbox where you can toggle demographics, service plans, and financial data. The model updates the churn risk in real-time, allowing agents to see how adding a service (like Online Security) might stabilize a customer.

### 📈 Strategic Insights Dashboard

Visualizes the correlation between tenure and charges. It includes box plots for charge dispersion and donut charts for payment method distribution (Electronic check, Mailed check, Bank transfer, Credit card).

### 📂 Batch Analysis Terminal

Designed for data teams to upload raw `.csv` files (e.g., `Telco-Customer-Churn.csv`). The engine processes thousands of rows instantly, providing a structured output ready for marketing automation.

-----

## ⚙️ Installation & Local Setup

**1. Clone the environment**

```bash
git clone https://github.com/MTank76/Teleconnect-Risk-Radar.git
cd Teleconnect-Risk-Radar
```

**2. Setup Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Fire up the Dashboard**

```bash
streamlit run app.py
```

> **Quick Start:** `pip install -r requirements.txt && streamlit run app.py`

-----

## 📂 Project Structure

```text
├── data/                # Raw and Processed Telco Datasets
├── models/              # Saved .pkl files (XGBoost, RF, Ensemble, etc.)
├── notebooks/           # Jupyter Notebooks for EDA and Model Tuning
├── src/                 # Modular Python logic for Data Preprocessing
├── app.py               # Streamlit Multi-page Web Application
├── main.py              # Pipeline execution script
└── requirements.txt     # Dependency Manifest
```

-----

## 🧪 Technologies Used

  * **Language:** Python
  * **Libraries:** Pandas, Scikit-learn, Plotly, NumPy
  * **Framework:** Streamlit 
  * **Version Control:** Git

-----

## Contributing

If you would like to improve the prediction accuracy or add new visualization modules, feel free to fork the repository and submit a pull request.

-----

## 📜 License

This project is licensed under the MIT License [![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

-----
