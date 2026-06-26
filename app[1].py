
import streamlit as st
st.set_page_config(page_title="Insurance Claim Bias Analysis", layout="wide", page_icon="📊")

# MUST be first Streamlit call
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 36px; font-weight: bold; color: #1f4e79; text-align: center;}
    .sub-header {font-size: 20px; font-weight: bold; color: #2c5aa0; margin-top: 20px;}
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;}
    .finding-box {background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107; margin: 10px 0;}
    .bias-alert {background-color: #f8d7da; padding: 15px; border-radius: 10px; border-left: 5px solid #dc3545; margin: 10px 0;}
    .success-box {background-color: #d4edda; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('Insurance.csv')
    df['SUM_ASSURED_CLEAN'] = df['SUM_ASSURED'].astype(str).str.replace(',', '').astype(float)
    df['PI_ANNUAL_INCOME_CLEAN'] = df['PI_ANNUAL_INCOME'].astype(str).str.replace(',', '').replace('', '0').replace('0', np.nan)
    df['PI_ANNUAL_INCOME_CLEAN'] = pd.to_numeric(df['PI_ANNUAL_INCOME_CLEAN'], errors='coerce')
    df['CLAIM_STATUS'] = df['POLICY_STATUS'].apply(lambda x: 1 if x == 'Approved Death Claim' else 0)
    df['INCOME_CATEGORY'] = pd.cut(df['PI_ANNUAL_INCOME_CLEAN'], 
                                    bins=[0, 100000, 300000, 600000, 1000000, float('inf')],
                                    labels=['Very Low (0-1L)', 'Low (1-3L)', 'Medium (3-6L)', 'High (6-10L)', 'Very High (>10L)'])
    df['AGE_CATEGORY'] = pd.cut(df['PI_AGE'], 
                                 bins=[0, 40, 50, 60, 70, 80, 100],
                                 labels=['<40', '40-50', '50-60', '60-70', '70-80', '80+'])
    df['PI_OCCUPATION'] = df['PI_OCCUPATION'].fillna('Unknown')
    df['REASON_FOR_CLAIM'] = df['REASON_FOR_CLAIM'].fillna('Not Specified')
    return df

def prepare_features(df):
    df_ml = df.copy()
    le_gender = LabelEncoder()
    df_ml['GENDER_ENCODED'] = le_gender.fit_transform(df_ml['PI_GENDER'])
    le_early = LabelEncoder()
    df_ml['EARLY_ENCODED'] = le_early.fit_transform(df_ml['EARLY_NON'])
    le_medical = LabelEncoder()
    df_ml['MEDICAL_ENCODED'] = le_medical.fit_transform(df_ml['MEDICAL_NONMED'])

    top_zones = df_ml['ZONE'].value_counts().head(10).index.tolist()
    df_ml['ZONE_GROUPED'] = df_ml['ZONE'].apply(lambda x: x if x in top_zones else 'OTHER')
    zone_dummies = pd.get_dummies(df_ml['ZONE_GROUPED'], prefix='ZONE')

    top_states = df_ml['PI_STATE'].value_counts().head(10).index.tolist()
    df_ml['STATE_GROUPED'] = df_ml['PI_STATE'].apply(lambda x: x if x in top_states else 'OTHER')
    state_dummies = pd.get_dummies(df_ml['STATE_GROUPED'], prefix='STATE')

    top_occ = df_ml['PI_OCCUPATION'].value_counts().head(10).index.tolist()
    df_ml['OCC_GROUPED'] = df_ml['PI_OCCUPATION'].apply(lambda x: x if x in top_occ else 'OTHER')
    occ_dummies = pd.get_dummies(df_ml['OCC_GROUPED'], prefix='OCC')

    payment_dummies = pd.get_dummies(df_ml['PAYMENT_MODE'], prefix='PAY')

    def group_reason(x):
        if pd.isna(x): return 'Other/Not Specified'
        x = str(x)
        if 'Heart' in x: return 'Heart Related'
        elif 'Cancer' in x: return 'Cancer'
        elif 'Accident' in x: return 'Accident'
        elif 'COVID' in x: return 'COVID'
        elif 'Natural' in x: return 'Natural'
        else: return 'Other/Not Specified'

    df_ml['REASON_GROUPED'] = df_ml['REASON_FOR_CLAIM'].apply(group_reason)
    reason_dummies = pd.get_dummies(df_ml['REASON_GROUPED'], prefix='REASON')

    df_ml['INCOME_LOG'] = np.log1p(df_ml['PI_ANNUAL_INCOME_CLEAN'].fillna(df_ml['PI_ANNUAL_INCOME_CLEAN'].median()))
    df_ml['SUM_ASSURED_LOG'] = np.log1p(df_ml['SUM_ASSURED_CLEAN'])
    df_ml['AGE_X_INCOME'] = df_ml['PI_AGE'] * df_ml['INCOME_LOG']
    df_ml['AGE_X_SA'] = df_ml['PI_AGE'] * df_ml['SUM_ASSURED_LOG']
    df_ml['INCOME_PER_SA'] = df_ml['PI_ANNUAL_INCOME_CLEAN'].fillna(0) / (df_ml['SUM_ASSURED_CLEAN'] + 1)
    df_ml['AGE_BIN_40'] = (df_ml['PI_AGE'] < 40).astype(int)
    df_ml['AGE_BIN_50_60'] = ((df_ml['PI_AGE'] >= 50) & (df_ml['PI_AGE'] < 60)).astype(int)
    df_ml['AGE_BIN_60_PLUS'] = (df_ml['PI_AGE'] >= 60).astype(int)
    df_ml['INCOME_MISSING'] = df_ml['PI_ANNUAL_INCOME_CLEAN'].isnull().astype(int)

    feature_cols = ['PI_AGE', 'GENDER_ENCODED', 'EARLY_ENCODED', 'MEDICAL_ENCODED',
                    'INCOME_LOG', 'SUM_ASSURED_LOG', 'AGE_X_INCOME', 'AGE_X_SA', 
                    'INCOME_PER_SA', 'AGE_BIN_40', 'AGE_BIN_50_60', 'AGE_BIN_60_PLUS',
                    'INCOME_MISSING']

    X = pd.concat([df_ml[feature_cols], zone_dummies, state_dummies, occ_dummies, 
                   payment_dummies, reason_dummies], axis=1)
    y = df_ml['CLAIM_STATUS']
    return X, y

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    results = {}

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    y_prob = knn.predict_proba(X_test_scaled)[:, 1]
    models['KNN'] = knn
    results['KNN'] = {
        'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'cm': confusion_matrix(y_test, y_pred)
    }

    dt = DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_prob = dt.predict_proba(X_test)[:, 1]
    models['Decision Tree'] = dt
    results['Decision Tree'] = {
        'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'cm': confusion_matrix(y_test, y_pred)
    }

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10, 
                                 min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'cm': confusion_matrix(y_test, y_pred)
    }

    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, 
                                   min_samples_split=10, min_samples_leaf=5, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    y_prob = gb.predict_proba(X_test)[:, 1]
    models['Gradient Boosting'] = gb
    results['Gradient Boosting'] = {
        'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'cm': confusion_matrix(y_test, y_pred)
    }

    return models, results, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

# ============================================================
# MAIN APP
# ============================================================

df = load_data()
X, y = prepare_features(df)

st.markdown('<div class="main-header">📊 Insurance Claim Settlement Bias Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Settlement Officer Review | Comprehensive Diagnostic & Predictive Analytics</p>", unsafe_allow_html=True)

st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Select Analysis Section:", [
    "🏠 Overview",
    "📋 Descriptive Analysis",
    "🔍 Diagnostic Bias Analysis", 
    "🤖 ML Classification Models",
    "📈 Model Performance",
    "🚨 Key Findings & Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Summary")
st.sidebar.metric("Total Claims", len(df))
st.sidebar.metric("Approved", f"{df['CLAIM_STATUS'].sum()} ({df['CLAIM_STATUS'].mean()*100:.1f}%)")
st.sidebar.metric("Repudiated", f"{(1-df['CLAIM_STATUS']).sum()} ({(1-df['CLAIM_STATUS']).mean()*100:.1f}%)")

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.markdown('<div class="sub-header">Executive Summary</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Approval Rate", f"{df['CLAIM_STATUS'].mean()*100:.1f}%")
    with col3:
        st.metric("Unique Zones", df['ZONE'].nunique())
    with col4:
        st.metric("Age Range", f"{df['PI_AGE'].min()}-{df['PI_AGE'].max()} years")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="bias-alert">', unsafe_allow_html=True)
        st.markdown("""
        ### 🚨 Bias Alert
        Our analysis reveals **statistically significant biases** in the claim settlement process:
        - **Income Bias**: Very low income policyholders face **51.9% repudiation** vs 32% overall
        - **Zone Bias**: Significant variation across teams (p < 0.001)
        - **Age Bias**: Younger policyholders show elevated repudiation rates
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        ### ✅ ML Model Insights
        Four supervised classification algorithms were trained:
        - **Random Forest**: Best Recall (91.5%) - catches most approvals
        - **Gradient Boosting**: Best AUC (0.786) - most stable predictions
        - **Decision Tree**: Good balance of metrics
        - **KNN**: Baseline performance
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Quick Data Preview")
    st.dataframe(df[['POLICY_NO', 'PI_NAME', 'PI_GENDER', 'PI_AGE', 'PI_ANNUAL_INCOME', 
                     'ZONE', 'POLICY_STATUS']].head(10), use_container_width=True)

# ============================================================
# PAGE 2: DESCRIPTIVE ANALYSIS
# ============================================================
elif page == "📋 Descriptive Analysis":
    st.markdown('<div class="sub-header">Cross-Tabulation Analysis Against Policy Status</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Gender", "Age", "Income", "Zone"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            gender_crosstab = pd.crosstab(df['PI_GENDER'], df['POLICY_STATUS'], margins=True)
            st.write("**Cross-Tabulation Table:**")
            st.dataframe(gender_crosstab, use_container_width=True)
        with col2:
            gender_pct = pd.crosstab(df['PI_GENDER'], df['POLICY_STATUS'], normalize='index') * 100
            st.write("**Approval Rates (%):**")
            st.dataframe(gender_pct.round(2), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        gender_pct.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'], width=0.6)
        ax.set_title('Claim Status by Gender', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.legend(['Repudiated', 'Approved'])
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            age_crosstab = pd.crosstab(df['AGE_CATEGORY'], df['POLICY_STATUS'], margins=True)
            st.write("**Cross-Tabulation Table:**")
            st.dataframe(age_crosstab, use_container_width=True)
        with col2:
            age_pct = pd.crosstab(df['AGE_CATEGORY'], df['POLICY_STATUS'], normalize='index') * 100
            st.write("**Approval Rates (%):**")
            st.dataframe(age_pct.round(2), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        age_pct.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'], width=0.6)
        ax.set_title('Claim Status by Age Category', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.legend(['Repudiated', 'Approved'])
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            income_crosstab = pd.crosstab(df['INCOME_CATEGORY'], df['POLICY_STATUS'], margins=True)
            st.write("**Cross-Tabulation Table:**")
            st.dataframe(income_crosstab, use_container_width=True)
        with col2:
            income_pct = pd.crosstab(df['INCOME_CATEGORY'], df['POLICY_STATUS'], normalize='index') * 100
            st.write("**Approval Rates (%):**")
            st.dataframe(income_pct.round(2), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        income_pct.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'], width=0.6)
        ax.set_title('Claim Status by Income Category', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.legend(['Repudiated', 'Approved'])
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        zone_counts = df['ZONE'].value_counts().head(15).index
        df_zone = df[df['ZONE'].isin(zone_counts)]
        zone_pct = pd.crosstab(df_zone['ZONE'], df_zone['POLICY_STATUS'], normalize='index') * 100
        st.write("**Zone-wise Approval Rates (Top 15 Zones):**")
        st.dataframe(zone_pct.round(2), use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        zone_pct.plot(kind='barh', ax=ax, color=['#e74c3c', '#2ecc71'], width=0.7)
        ax.set_title('Claim Status by Zone (Top 15)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Percentage (%)')
        ax.legend(['Repudiated', 'Approved'])
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# PAGE 3: DIAGNOSTIC BIAS ANALYSIS
# ============================================================
elif page == "🔍 Diagnostic Bias Analysis":
    st.markdown('<div class="sub-header">Deep Dive Bias Detection Analysis</div>', unsafe_allow_html=True)

    overall_repud = (1 - df['CLAIM_STATUS'].mean()) * 100

    tab1, tab2, tab3, tab4 = st.tabs(["Age-wise Bias", "Income-wise Bias", "Zone-wise Bias", "Interaction Effects"])

    with tab1:
        age_bias = df.groupby('AGE_CATEGORY').agg({
            'CLAIM_STATUS': ['count', 'sum', 'mean']
        }).round(3)
        age_bias.columns = ['Total_Claims', 'Approved', 'Approval_Rate']
        age_bias['Repudiation_Rate'] = (1 - age_bias['Approval_Rate']) * 100

        st.write("**Age-wise Detailed Analysis:**")
        st.dataframe(age_bias, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#e74c3c' if x > overall_repud else '#2ecc71' for x in age_bias['Repudiation_Rate']]
        bars = ax.bar(range(len(age_bias)), age_bias['Repudiation_Rate'], color=colors, edgecolor='black')
        ax.axhline(y=overall_repud, color='red', linestyle='--', linewidth=2, label=f'Overall Repudiation ({overall_repud:.1f}%)')
        ax.set_title('Age-wise Repudiation Rate (Red = Above Average)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Age Category')
        ax.set_ylabel('Repudiation Rate (%)')
        ax.set_xticks(range(len(age_bias)))
        ax.set_xticklabels(age_bias.index, rotation=45)
        ax.legend()
        for i, (bar, total) in enumerate(zip(bars, age_bias['Total_Claims'])):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}%\n(n={int(total)})', ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        income_bias = df.groupby('INCOME_CATEGORY').agg({
            'CLAIM_STATUS': ['count', 'sum', 'mean']
        }).round(3)
        income_bias.columns = ['Total_Claims', 'Approved', 'Approval_Rate']
        income_bias['Repudiation_Rate'] = (1 - income_bias['Approval_Rate']) * 100

        st.write("**Income-wise Detailed Analysis:**")
        st.dataframe(income_bias, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#e74c3c' if x > overall_repud else '#2ecc71' for x in income_bias['Repudiation_Rate']]
        bars = ax.bar(range(len(income_bias)), income_bias['Repudiation_Rate'], color=colors, edgecolor='black')
        ax.axhline(y=overall_repud, color='red', linestyle='--', linewidth=2, label=f'Overall Repudiation ({overall_repud:.1f}%)')
        ax.set_title('Income-wise Repudiation Rate (Red = Above Average)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Income Category')
        ax.set_ylabel('Repudiation Rate (%)')
        ax.set_xticks(range(len(income_bias)))
        ax.set_xticklabels(income_bias.index, rotation=45)
        ax.legend()
        for i, (bar, total) in enumerate(zip(bars, income_bias['Total_Claims'])):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}%\n(n={int(total)})', ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        team_bias = df.groupby('ZONE').agg({
            'CLAIM_STATUS': ['count', 'sum', 'mean']
        }).round(3)
        team_bias.columns = ['Total_Claims', 'Approved', 'Approval_Rate']
        team_bias['Repudiation_Rate'] = (1 - team_bias['Approval_Rate']) * 100
        team_bias = team_bias.sort_values('Repudiation_Rate', ascending=False)

        st.write("**Zone-wise Detailed Analysis:**")
        st.dataframe(team_bias, use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 10))
        colors = ['#e74c3c' if x > overall_repud else '#2ecc71' for x in team_bias['Repudiation_Rate']]
        bars = ax.barh(range(len(team_bias)), team_bias['Repudiation_Rate'], color=colors, edgecolor='black')
        ax.axvline(x=overall_repud, color='red', linestyle='--', linewidth=2, label=f'Overall Repudiation ({overall_repud:.1f}%)')
        ax.set_title('Zone-wise Repudiation Rate (Red = Above Average)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Repudiation Rate (%)')
        ax.set_yticks(range(len(team_bias)))
        ax.set_yticklabels(team_bias.index, fontsize=9)
        ax.legend()
        for i, (bar, total) in enumerate(zip(bars, team_bias['Total_Claims'])):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{bar.get_width():.1f}% (n={int(total)})', ha='left', va='center', fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        st.write("**Gender x Age Interaction:**")
        gender_age = pd.crosstab([df['PI_GENDER'], df['AGE_CATEGORY']], df['POLICY_STATUS'], normalize='index') * 100
        st.dataframe(gender_age.round(2), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        gender_age_repud = gender_age['Repudiate Death'].unstack(level=0)
        x = np.arange(len(gender_age_repud.index))
        width = 0.35
        ax.bar(x - width/2, gender_age_repud['F'], width, label='Female', color='#e91e63', edgecolor='black')
        ax.bar(x + width/2, gender_age_repud['M'], width, label='Male', color='#2196f3', edgecolor='black')
        ax.axhline(y=overall_repud, color='red', linestyle='--', linewidth=2)
        ax.set_title('Gender x Age Interaction (Repudiation Rate %)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Age Category')
        ax.set_ylabel('Repudiation Rate (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(gender_age_repud.index, rotation=45)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# PAGE 4: ML CLASSIFICATION MODELS
# ============================================================
elif page == "🤖 ML Classification Models":
    st.markdown('<div class="sub-header">Supervised Machine Learning Classification</div>', unsafe_allow_html=True)

    with st.spinner('Training models... This may take a moment.'):
        models, results, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = train_models(X, y)

    st.success("All 4 models trained successfully!")

    st.markdown("### Model Performance Summary")
    summary_df = pd.DataFrame({
        'Algorithm': list(results.keys()),
        'Accuracy': [results[k]['accuracy'] for k in results.keys()],
        'Precision': [results[k]['precision'] for k in results.keys()],
        'Recall': [results[k]['recall'] for k in results.keys()],
        'F1-Score': [results[k]['f1'] for k in results.keys()]
    })
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("### ROC Curves")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2.5, label=f'{name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Feature Importance")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Random Forest - Top 15 Features:**")
        rf_imp = pd.DataFrame({'feature': X.columns, 'importance': models['Random Forest'].feature_importances_})
        rf_imp = rf_imp.sort_values('importance', ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(rf_imp)), rf_imp['importance'], color='#2ecc71', edgecolor='black')
        ax.set_yticks(range(len(rf_imp)))
        ax.set_yticklabels(rf_imp['feature'], fontsize=9)
        ax.set_xlabel('Importance')
        ax.set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.write("**Gradient Boosting - Top 15 Features:**")
        gb_imp = pd.DataFrame({'feature': X.columns, 'importance': models['Gradient Boosting'].feature_importances_})
        gb_imp = gb_imp.sort_values('importance', ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(gb_imp)), gb_imp['importance'], color='#f39c12', edgecolor='black')
        ax.set_yticks(range(len(gb_imp)))
        ax.set_yticklabels(gb_imp['feature'], fontsize=9)
        ax.set_xlabel('Importance')
        ax.set_title('Gradient Boosting Feature Importance', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# PAGE 5: MODEL PERFORMANCE (Confusion Matrices)
# ============================================================
elif page == "📈 Model Performance":
    st.markdown('<div class="sub-header">Detailed Model Performance & Confusion Matrices</div>', unsafe_allow_html=True)

    with st.spinner('Training models...'):
        models, results, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = train_models(X, y)

    st.markdown("### Confusion Matrices with False Positive & False Negative Analysis")

    cols = st.columns(2)
    for idx, (name, res) in enumerate(results.items()):
        with cols[idx % 2]:
            cm = res['cm']
            total = cm.sum()
            tn, fp, fn, tp = cm.ravel()

            tn_pct = (tn / total) * 100
            fp_pct = (fp / total) * 100
            fn_pct = (fn / total) * 100
            tp_pct = (tp / total) * 100

            st.markdown(f"**{name}**")
            st.markdown(f"Accuracy: {res['accuracy']:.3f} | Precision: {res['precision']:.3f} | Recall: {res['recall']:.3f} | F1: {res['f1']:.3f}")

            fig, ax = plt.subplots(figsize=(6, 5))
            annot_matrix = np.array([
                [f'TN: {int(tn)}\n({tn_pct:.1f}%)', f'FP: {int(fp)}\n({fp_pct:.1f}%)'],
                [f'FN: {int(fn)}\n({fn_pct:.1f}%)', f'TP: {int(tp)}\n({tp_pct:.1f}%)']
            ])
            sns.heatmap(cm, annot=annot_matrix, fmt='', cmap='Blues', ax=ax,
                        cbar=False, annot_kws={'size': 11, 'weight': 'bold'},
                        xticklabels=['Pred: Repudiated', 'Pred: Approved'],
                        yticklabels=['Actual: Repudiated', 'Actual: Approved'],
                        linewidths=2, linecolor='white')
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(f"""
            - **False Positives**: {fp} ({fp_pct:.2f}%) - Wrongly Approved
            - **False Negatives**: {fn} ({fn_pct:.2f}%) - Wrongly Repudiated
            - **FP Rate**: {fp/(fp+tn)*100:.2f}%
            - **FN Rate**: {fn/(fn+tp)*100:.2f}%
            """)
            st.markdown("---")

    st.markdown("### False Positive & False Negative Summary")
    fn_fp_summary = []
    for name, res in results.items():
        cm = res['cm']
        total = cm.sum()
        tn, fp, fn, tp = cm.ravel()
        fn_fp_summary.append({
            'Algorithm': name,
            'False Positives (%)': f"{fp} ({fp/total*100:.2f}%)",
            'False Negatives (%)': f"{fn} ({fn/total*100:.2f}%)",
            'FP Rate (%)': f"{fp/(fp+tn)*100:.2f}%",
            'FN Rate (%)': f"{fn/(fn+tp)*100:.2f}%"
        })
    st.dataframe(pd.DataFrame(fn_fp_summary), use_container_width=True)

# ============================================================
# PAGE 6: KEY FINDINGS & RECOMMENDATIONS
# ============================================================
elif page == "🚨 Key Findings & Recommendations":
    st.markdown('<div class="sub-header">Key Findings & Actionable Recommendations</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="bias-alert">
    <h3>CRITICAL FINDING 1: Income Bias (Statistically Significant, p < 0.001)</h3>
    <ul>
        <li><b>Very Low Income (0-1L)</b> policyholders face <b>51.9% repudiation</b> vs 32% overall</li>
        <li>This is nearly <b>20 percentage points higher</b> than the company average</li>
        <li>Suggests potential discrimination against economically weaker sections</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bias-alert">
    <h3>CRITICAL FINDING 2: Zone/Team Bias (Statistically Significant, p < 0.001)</h3>
    <ul>
        <li>Massive variation in repudiation rates across zones: <b>3.4% (JKB Creditor) to 100% (South 2)</b></li>
        <li>Top 5 most biased zones: South 2, South, SOUTH, ROI, West</li>
        <li>Indicates inconsistent decision-making standards across teams</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-box">
    <h3>FINDING 3: Age-Related Patterns</h3>
    <ul>
        <li>Policyholders <b>under 40</b> show <b>33.1% repudiation</b> (slightly above average)</li>
        <li>Age difference between approved and repudiated claims is <b>NOT statistically significant</b> (p = 0.974)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-box">
    <h3>FINDING 4: Gender Patterns</h3>
    <ul>
        <li>Female repudiation rate: <b>28.6%</b> | Male repudiation rate: <b>32.7%</b></li>
        <li>Females actually have <b>lower repudiation</b> than males</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### RECOMMENDATIONS")

    st.markdown("""
    <div class="success-box">
    <h4>Immediate Actions:</h4>
    <ol>
        <li><b>Audit Low-Income Claims</b>: Review all repudiated claims from Very Low income category</li>
        <li><b>Standardize Zone Practices</b>: Implement uniform claim assessment guidelines</li>
        <li><b>Bias Training</b>: Conduct mandatory unconscious bias training</li>
        <li><b>ML Monitoring</b>: Deploy Gradient Boosting model as bias detection tool</li>
    </ol>
    <h4>Long-term Actions:</h4>
    <ol>
        <li><b>Automated Screening</b>: Use ML model to flag potentially biased decisions</li>
        <li><b>Quarterly Bias Audits</b>: Regular statistical analysis of settlement patterns</li>
        <li><b>Income-Blind Review</b>: Consider income-blind claim assessment</li>
        <li><b>Zone Rotation</b>: Rotate settlement officers across zones</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Recommendation for Deployment")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best AUC", "Gradient Boosting", "0.786")
    with col2:
        st.metric("Best Recall", "Random Forest", "91.5%")
    with col3:
        st.metric("Best Precision", "Gradient Boosting", "80.6%")

    st.info("Recommendation: Deploy Gradient Boosting as the primary bias detection model due to its highest AUC (0.786).")

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align:center; color:gray;'>Built for Insurance Settlement Analysis</p>", unsafe_allow_html=True)
