# 🔮 AuraSegment: Premium Customer Segmentation & Predictive Analytics
https://customer-segmentation-pbphzytgdz8nwuenbehz4r.streamlit.app/

AuraSegment is an ultra-premium, interactive machine-learning-powered web dashboard designed to segment customers based on their age, income, and purchasing habits. It categorizes shoppers into five distinct cohorts and provides real-time, cohort-specific marketing strategies for business teams.

---

## 🛠️ Tech Stack & Features

*   **Core Logic**: Python 3.12, `scikit-learn`, `numpy`, `pandas`
*   **Web Framework**: `Streamlit`
*   **Visualizations**: `matplotlib`, `seaborn`
*   **Design Paradigm**: Dark Mode, Glassmorphic Containers, and Custom HSL Gradient Cards
*   **ML Core**: Unsupervised **K-Means Clustering** & **StandardScaler** Pipeline

---

## 📂 Project Structure

```text
├── .venv/                  # Local python virtual environment (excluded from Git)
├── Mall_Customers.csv      # Historical dataset (200 retail shoppers)
├── app.py                  # Primary Streamlit application & interactive dashboard
├── cluster_model.pkl       # Trained K-Means clustering model (5 clusters)
├── scaler.pkl              # Pre-configured StandardScaler instance
├── requirements.txt        # Python dependency packages
├── .gitignore              # Configurations to prevent uploading junk to Git
└── README.md               # Project documentation (this file)
```

---

## ⚙️ Local Setup Instructions

Follow these quick commands to spin up the application on your computer:

### 1. Clone or Download the Directory
Place these files in a folder on your system (e.g., `c:\Users\m\Downloads\ml project\customer segmentation`).

### 2. Activate the Virtual Environment
Open your terminal inside the project directory and run:

**On Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
Ensure all packages are up to date:
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
Run the Streamlit app in headless or normal mode:
```bash
streamlit run app.py
```

Open your browser and navigate to **`http://localhost:8501`**.

---

## 🧠 Machine Learning Overview

AuraSegment uses a pre-trained unsupervised clustering pipeline:
1.  **Standardization (`scaler.pkl`)**: Because annual income ranges up to $150k while age tops out around 70, the data is scaled using `StandardScaler` to ensure equal feature weighting.
2.  **Clustering (`cluster_model.pkl`)**: The K-Means algorithm partitions customers into 5 groups based on spatial density.

### 👥 The 5 Target Customer Segments:
*   💎 **Regular VIPs (Cluster 1)**: High Income, High Spending. These are your ultimate brand champions.
*   🚀 **Luxury Trendsetters (Cluster 2)**: Low Income, High Spending. Highly active young buyers.
*   🎖️ **Premium Spenders (Cluster 0)**: Moderate Income, Steady loyal shoppers (mature cohort).
*   ☘️ **Budget Value-Seekers (Cluster 3)**: Moderate Income, Moderate Spending. Smart deal-oriented shoppers.
*   🛡️ **Occasional Savers (Cluster 4)**: High Income, Low Spending. Highly cautious, practical buyers.
