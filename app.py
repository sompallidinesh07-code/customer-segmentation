import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress sklearn pickle warnings for a clean user interface
warnings.filterwarnings("ignore", category=UserWarning)

# Page Configuration for Premium look
st.set_page_config(
    page_title="AuraSegment • Premium Customer Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling Injection (Modern UI with HSL Tailored Colors and Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161a24 100%);
        color: #f3f4f6;
    }
    
    /* Header Gradient Style */
    .title-gradient {
        background: linear-gradient(90deg, #a78bfa 0%, #ec4899 50%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.05em;
    }
    
    .subtitle-text {
        color: #9ca3af;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric & Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0c0e14 !important;
        border-right: 1px solid #1f2937;
    }
    
    /* Button Customization */
    .stButton>button {
        background: linear-gradient(90deg, #7c3aed 0%, #db2777 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.3) !important;
        width: 100%;
        font-size: 1.1rem !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(244, 63, 94, 0.4) !important;
        color: #ffffff !important;
    }
    
    .stButton>button:active {
        transform: translateY(1px) !important;
    }
    
    /* Result Card Styles */
    .result-card {
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-top: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Slider custom colors */
    div[data-baseweb="slider"] {
        margin-bottom: 1.5rem;
    }
    
    /* Custom Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Cache historical data load and initial clustering mapping
@st.cache_data
def load_and_cluster_historical_data():
    try:
        df = pd.read_csv("Mall_Customers.csv")
        s = joblib.load("scaler.pkl")
        m = joblib.load("cluster_model.pkl")
        
        # Scale and predict
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        X_scaled = s.transform(df[features])
        df['Cluster'] = m.predict(X_scaled)
        
        # Map cluster names
        names = {
            0: "Premium Customers",
            1: "Regular Customers",
            2: "Luxury Spenders",
            3: "Budget Shoppers",
            4: "Occasional Buyers"
        }
        df['Segment'] = df['Cluster'].map(names)
        return df
    except Exception as e:
        st.error(f"Error loading historical dataset: {e}")
        return None

# Load models safely
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load("cluster_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_ml_assets()
historical_df = load_and_cluster_historical_data()

# Metadata about the customer segments (Visual Themes and Marketing Playbooks)
SEGMENT_DETAILS = {
    0: {
        "name": "Premium Spenders",
        "description": "Mature, highly loyal customers with steady incomes and consistent spending habits.",
        "icon": "🎖️",
        "color_grad": "linear-gradient(135deg, #0f766e 0%, #0d9488 50%, #14b8a6 100%)", # Emerald / Teal
        "badge_color": "#14b8a6",
        "metrics": {"Avg Age": "55.3 yrs", "Avg Income": "$47.6k", "Avg Score": "41.7/100"},
        "strategy": [
            "🏆 Introduce premium loyalty tier with high retention rewards.",
            "✉️ Direct traditional personalized outreach and service assurance.",
            "🛡️ Focus on quality guarantees, product durability, and ease of use."
        ],
        "fact": "This group accounts for 29.0% of your customer base and forms the bedrock of steady revenue."
    },
    1: {
        "name": "Regular VIPs",
        "description": "High-income, high-spending powerhouses. They represent the highest lifetime value segment.",
        "icon": "💎",
        "color_grad": "linear-gradient(135deg, #6d28d9 0%, #7c3aed 50%, #c084fc 100%)", # Violet / Amethyst
        "badge_color": "#a855f7",
        "metrics": {"Avg Age": "32.9 yrs", "Avg Income": "$86.1k", "Avg Score": "81.5/100"},
        "strategy": [
            "🥂 Offer private event invitations, exclusive first-looks, and personal shoppers.",
            "🚀 Target with luxury collections, high-ticket items, and personalized luxury bundles.",
            "⭐ Establish a premium concierge service with dedicated support channels."
        ],
        "fact": "Your VIP powerhouses. Despite making up 20.0% of customers, they generate massive spending share."
    },
    2: {
        "name": "Luxury Trendsetters",
        "description": "Young, highly active shoppers. Low income but extremely enthusiastic spending behavior.",
        "icon": "🚀",
        "color_grad": "linear-gradient(135deg, #c2410c 0%, #ea580c 50%, #f97316 100%)", # Coral / Orange
        "badge_color": "#f97316",
        "metrics": {"Avg Age": "25.8 yrs", "Avg Income": "$26.1k", "Avg Score": "74.8/100"},
        "strategy": [
            "📱 Target heavily on social media (Instagram, TikTok) using influencer promotions.",
            "⚡ Create high-energy flash sales, trendy item drops, and gamified shopping experiences.",
            "💳 Provide flexible checkout terms, such as Buy-Now-Pay-Later (BNPL) integrations."
        ],
        "fact": "The future of the brand. Trend-sensitive and highly active in digital word-of-mouth."
    },
    3: {
        "name": "Budget Value-Seekers",
        "description": "Younger, moderate-income customers focused on getting the best price-to-value ratio.",
        "icon": "☘️",
        "color_grad": "linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #60a5fa 100%)", # Cobalt Blue
        "badge_color": "#3b82f6",
        "metrics": {"Avg Age": "26.7 yrs", "Avg Income": "$54.3k", "Avg Score": "40.9/100"},
        "strategy": [
            "📦 Design compelling value bundles, multi-buy discounts, and clear saving charts.",
            "🎟️ Launch referral-bonus campaigns to tap into their highly active peer networks.",
            "💡 Market practical, smart-choice everyday utility products."
        ],
        "fact": "A massive group of highly smart shoppers who appreciate clear value and transparency."
    },
    4: {
        "name": "Occasional Savers",
        "description": "High-income, conservative spenders. Extremely selective and deliberate about purchases.",
        "icon": "🛡️",
        "color_grad": "linear-gradient(135deg, #374151 0%, #4b5563 50%, #9ca3af 100%)", # Metallic Grey
        "badge_color": "#6b7280",
        "metrics": {"Avg Age": "44.4 yrs", "Avg Income": "$89.8k", "Avg Score": "18.5/100"},
        "strategy": [
            "📊 Run feature-comparison campaigns emphasizing utility, ROI, and technical superiority.",
            "📉 Tempt with direct, high-value cashback programs and targeted loyalty discounts.",
            "🧠 Highlight testimonials, data-driven proofs, and professional accolades."
        ],
        "fact": "Cautious high-earners. Converting them requires proof-of-value, not flashy advertisements."
    }
}

# App Layout
st.markdown('<div class="title-gradient">AuraSegment</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">🔮 Premium Customer Segmentation & Predictive Analytics Engine</div>', unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("🚨 ML Assets (scaler.pkl or cluster_model.pkl) are missing or corrupted. Please verify the project files.")
else:
    # Sidebar Setup
    with st.sidebar:
        st.markdown("### 🛠️ Interactive Simulator")
        st.markdown("Adjust customer attributes below to dynamically compute their marketing profile.")
        st.write("---")
        
        age = st.slider("🧑 Age", 18, 70, 30, help="Customer's age in years")
        income = st.slider("💰 Annual Income (k$)", 10, 150, 60, help="Customer's estimated yearly income in thousands of dollars")
        score = st.slider("🛍️ Spending Score", 1, 100, 50, help="Score assigned by mall based on customer behavior and purchasing history")
        
        st.write("---")
        st.markdown("<p style='text-align: center; color: #6b7280; font-size: 0.8rem;'>Powered by scikit-learn & Streamlit</p>", unsafe_allow_html=True)

    # Creating Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Customer Classifier",
        "📊 Cluster Analytics",
        "💡 Marketing Playbook",
        "📖 How It Works"
    ])

    # --- TAB 1: Customer Classifier ---
    with tab1:
        st.write("### 🔮 Real-Time Customer Profile Simulator")
        st.write("Modify the sidebar parameters to see the customer automatically classified into their machine-learning predicted marketing segment.")
        
        # Perform prediction
        input_data = np.array([[age, income, score]])
        
        try:
            scaled_data = scaler.transform(input_data)
            predicted_cluster = int(model.predict(scaled_data)[0])
            details = SEGMENT_DETAILS[predicted_cluster]
            
            # Beautiful prediction layout
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.markdown(f"""
                <div class="result-card" style="background: {details['color_grad']};">
                    <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">{details['icon']}</div>
                    <span class="badge">Cluster #{predicted_cluster}</span>
                    <span class="badge" style="background: rgba(0,0,0,0.3); border: none;">{details['name']}</span>
                    <h2 style="margin-top: 0.5rem; font-family: 'Outfit', sans-serif; font-weight: 700; color: #ffffff;">
                        Identified: {details['name']}
                    </h2>
                    <p style="font-size: 1.15rem; color: rgba(255, 255, 255, 0.9); margin-bottom: 1.5rem; line-height: 1.6;">
                        {details['description']}
                    </p>
                    <div style="border-top: 1px solid rgba(255,255,255,0.2); padding-top: 1rem;">
                        <strong>💡 Targeted Strategy Checklist:</strong>
                        <ul style="list-style-type: none; padding-left: 0; margin-top: 0.5rem;">
                            <li style="margin-bottom: 0.5rem;">{"</li><li style='margin-bottom: 0.5rem;'>".join(details['strategy'])}</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_right:
                st.write("### 📊 Segment Statistics")
                st.info(details['fact'])
                
                st.write("#### Cluster Averages")
                for key, val in details['metrics'].items():
                    st.metric(label=key, value=val)
                    
        except Exception as e:
            st.error(f"Prediction Error: {e}. Please ensure inputs match the expected model shape.")

    # --- TAB 2: Cluster Analytics ---
    with tab2:
        st.write("### 📊 Historical Customer Clusters & Distribution")
        st.write("Visualize the entire customer base and see exactly where the simulated customer is located relative to the historic demographics.")
        
        if historical_df is not None:
            col_plot, col_stats = st.columns([3, 2])
            
            with col_plot:
                # Custom Premium Styled Scatter Plot
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0e1117')
                ax.set_facecolor('#161a24')
                
                # Plot historical data points
                scatter = sns.scatterplot(
                    data=historical_df,
                    x='Annual Income (k$)',
                    y='Spending Score (1-100)',
                    hue='Segment',
                    palette={
                        "Premium Customers": "#14b8a6",
                        "Regular Customers": "#a855f7",
                        "Luxury Spenders": "#f97316",
                        "Budget Shoppers": "#3b82f6",
                        "Occasional Buyers": "#6b7280"
                    },
                    alpha=0.6,
                    s=80,
                    edgecolor='#0e1117',
                    linewidth=0.5,
                    ax=ax
                )
                
                # Highlight the simulated custom data point
                ax.scatter(
                    income,
                    score,
                    color='#ef4444',
                    marker='*',
                    s=400,
                    edgecolor='#ffffff',
                    linewidth=1.5,
                    zorder=10,
                    label='⚡ Simulator Customer'
                )
                
                # Visual improvements
                ax.spines['bottom'].set_color('#374151')
                ax.spines['left'].set_color('#374151')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.xaxis.label.set_color('#9ca3af')
                ax.yaxis.label.set_color('#9ca3af')
                ax.tick_params(colors='#9ca3af', which='both')
                ax.grid(color='#1f2937', linestyle='--', linewidth=0.5)
                
                ax.set_title("Customer Segmentation Map", color='#f3f4f6', fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel("Annual Income (k$)", fontsize=11, labelpad=10)
                ax.set_ylabel("Spending Score (1-100)", fontsize=11, labelpad=10)
                
                # Custom legend formatting
                legend = ax.legend(
                    facecolor='#161a24',
                    edgecolor='#374151',
                    labelcolor='#f3f4f6',
                    loc='upper right',
                    framealpha=0.9
                )
                
                st.pyplot(fig)
                
            with col_stats:
                st.write("#### 📈 Demographics Breakdown")
                st.write("This table shows the count of historical shoppers within each machine learning-identified category:")
                
                sizes = historical_df['Segment'].value_counts().reset_index()
                sizes.columns = ['Segment Name', 'Active Customers Count']
                st.dataframe(
                    sizes,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Interactive metrics
                total_cust = len(historical_df)
                st.metric("Total Analyzed Customer Profiles", f"{total_cust} Shoppers")
                
                # Simple progress bar visualization
                st.write("#### Proportional Distribution")
                for index, row in sizes.iterrows():
                    pct = float(row['Active Customers Count']) / total_cust
                    st.write(f"**{row['Segment Name']}** ({pct*100:.1f}%)")
                    st.progress(pct)
        else:
            st.info("Scatter plotting and historical statistics are unavailable because Mall_Customers.csv is not loaded.")

    # --- TAB 3: Marketing Playbook ---
    with tab3:
        st.write("### 💡 Strategic Marketing Playbooks")
        st.write("Access expert customer acquisition, retention, and engagement playbooks for all five customer segments.")
        
        for key, val in SEGMENT_DETAILS.items():
            with st.expander(f"{val['icon']} {val['name']} Playbook"):
                st.write(f"**Cohort Characterization**: {val['description']}")
                st.info(f"📊 **Demographics Info**: {val['fact']}")
                st.markdown("#### Actionable Marketing Strategies:")
                for bullet in val['strategy']:
                    st.markdown(f"- {bullet}")
                
                # Add a quick metrics table for the playbook
                st.write("---")
                col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
                col_metric_1.metric("Target Age Group", val['metrics']['Avg Age'])
                col_metric_2.metric("Income Segment", val['metrics']['Avg Income'])
                col_metric_3.metric("Spending Rating", val['metrics']['Avg Score'])

    # --- TAB 4: How It Works (Aesthetic User-Friendly Explanation) ---
    with tab4:
        st.write("### 📖 Inside the AuraSegment AI")
        st.write("Understand the machine learning engine behind the application through a simple, visual guide.")
        
        # 1. Gorgeous Interactive Workflow Diagram in HTML/CSS
        st.markdown("""
        <div style="display: flex; gap: 1rem; margin-top: 1rem; margin-bottom: 2.5rem; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 1.5rem; border-radius: 16px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🎛️</div>
                <strong style="color: #a78bfa; display: block; margin-bottom: 0.4rem; font-size: 1.05rem;">1. Capture Input</strong>
                <span style="font-size: 0.9rem; color: #9ca3af; line-height: 1.4; display: block;">You adjust Age, Annual Income, and Spending Score sliders.</span>
            </div>
            <div style="flex: 1; min-width: 220px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 1.5rem; border-radius: 16px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">⚖️</div>
                <strong style="color: #f472b6; display: block; margin-bottom: 0.4rem; font-size: 1.05rem;">2. Scale Values</strong>
                <span style="font-size: 0.9rem; color: #9ca3af; line-height: 1.4; display: block;">The app balances sizes so larger figures (income) don't drown out ages.</span>
            </div>
            <div style="flex: 1; min-width: 220px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 1.5rem; border-radius: 16px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🧠</div>
                <strong style="color: #3b82f6; display: block; margin-bottom: 0.4rem; font-size: 1.05rem;">3. Find Cluster</strong>
                <span style="font-size: 0.9rem; color: #9ca3af; line-height: 1.4; display: block;">AI patterns map the customer to their nearest cohort neighborhood.</span>
            </div>
            <div style="flex: 1; min-width: 220px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 1.5rem; border-radius: 16px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🎯</div>
                <strong style="color: #34d399; display: block; margin-bottom: 0.4rem; font-size: 1.05rem;">4. Target Smart</strong>
                <span style="font-size: 0.9rem; color: #9ca3af; line-height: 1.4; display: block;">The system delivers customized strategic checklists and analytics.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Detailed explanation grid
        col_concept, col_ingredients = st.columns([1, 1])
        
        with col_concept:
            st.write("#### 🤝 The Core Concept: Natural Groupings")
            st.markdown("""
            Think of this AI as a **smart sorting assistant**.
            
            Instead of manually grouping thousands of customers by looking at giant spreadsheets, we trained a **K-Means machine learning model**.
            
            By analyzing historical customer data, the model recognized **five distinct customer personalities**. When you input new attributes, the model calculates exactly which of these five 'neighborhoods' the customer belongs to.
            """)
            
            st.info("💡 **Fun Fact**: K-Means operates like gravity! It finds the center coordinates of each group (called centroids) and pulls similar customers toward their closest cluster center.")
            
        with col_ingredients:
            st.write("#### 📊 The Three Key Ingredients")
            st.markdown("""
            Our customer grouping engine is driven by exactly **three metrics**:
            """)
            
            st.markdown("""
            * 🧑 **Age**: Determines the buyer's generational cohort and life stage.
            * 💰 **Annual Income**: Measures the financial budget of the customer.
            * 🛍️ **Spending Score**: A custom rating (1-100) reflecting their historical buying frequency and loyalty.
            """)
            
            st.success("🤖 **Why Scaling Matters**: Annual Income ranges from $10k to $150k, while Age only ranges from 18 to 70. The model uses a **StandardScaler** to place all variables on equal footing, ensuring fair predictions.")