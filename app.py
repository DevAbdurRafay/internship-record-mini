import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Intern Performance Predictor", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .stAppViewContainer > section > div {
        padding-top: 0.5rem !important; 
        padding-bottom: 0.5rem !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    h1 {
        font-size: 2.2rem !important; 
        margin-top: 0px !important;
        padding-top: 0px !important;
        padding-bottom: 5px !important;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px; 
        margin-top: 5px !important;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    h2, h3 {
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    div[data-testid="stNumberInput"] {
        margin-bottom: -15px !important;
    }
    div[data-testid="stNumberInput"] label {
        padding-bottom: 0px !important;
        margin-bottom: 2px !important;
    }
    h2 {
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    .dataframe {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_and_train():
    np.random.seed(42)
    n_interns = 150
    names = ["Ahmed", "Ali", "Fatima", "Ayesha", "Bilal", "Zainab", "Hamza", "Sara", "Mustafa", "Hassan", 
             "Raza", "Sana", "Usman", "Omer", "Hira"] * 10
    intern_ids = [f"INT-{1000+i}" for i in range(n_interns)]
    attendance = np.random.randint(60, 100, n_interns)
    submissions = np.random.randint(50, 100, n_interns)
    feedback = np.random.randint(4, 10, n_interns)
    
    df = pd.DataFrame({
        'Intern_ID': intern_ids,
        'Name': names[:n_interns],
        'Attendance_Rate_%': attendance,
        'Task_Submission_Rate_%': submissions,
        'Mentor_Feedback_10': feedback
    })
    
    df['Actual_Status'] = np.where((df['Attendance_Rate_%'] >= 75) & (df['Task_Submission_Rate_%'] >= 70), 1, 0)
    
    X = df[['Attendance_Rate_%', 'Task_Submission_Rate_%', 'Mentor_Feedback_10']]
    y = df['Actual_Status']
    
    model = RandomForestClassifier(random_state=42, n_estimators=50)
    model.fit(X, y)
    
    df['Success_Probability_%'] = np.round(model.predict_proba(X)[:, 1] * 100, 1)
    
    conditions = [
        (df['Success_Probability_%'] >= 75),
        (df['Success_Probability_%'] >= 50) & (df['Success_Probability_%'] < 75),
        (df['Success_Probability_%'] < 50)
    ]
    choices = [
        "🟢 Excelent: Student is on track. Keep it up!",
        "🟡 Warning: Slight lag in metrics. Needs encouragement.",
        "🔴 Critical: High risk of dropout. Requires 1-on-1 counseling."
    ]
    df['Mentor_Insight'] = np.select(conditions, choices, default="No Data")
    
    return df, model

df_predictions, model = generate_and_train()

st.title("🎯 Intern Performance & Engagement Predictor")
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-top: -10px; margin-bottom: 15px;'>Machine Learning Dashboard for Mentors (Dark Glassmorphism Edition)</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown('<div class="glass-card"><h3>📋 Predicted Intern Performance Roster</h3>', unsafe_allow_html=True)
    search_query = st.text_input("🔍 Search Intern by Name or ID:")
    filtered_df = df_predictions.copy()
    if search_query:
        filtered_df = filtered_df[filtered_df['Name'].str.contains(search_query, case=False) | filtered_df['Intern_ID'].str.contains(search_query, case=False)]
        
    st.dataframe(
        filtered_df[['Intern_ID', 'Name', 'Attendance_Rate_%', 'Task_Submission_Rate_%', 'Mentor_Feedback_10', 'Success_Probability_%', 'Mentor_Insight']],
        use_container_width=True,
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card"><h3>🔮 Real-Time Predictor (Test Individual)</h3>', unsafe_allow_html=True)
    st.write("Enter metrics below to check instant prediction:")
    
    input_att = st.number_input("Attendance Rate (%)", min_value=0, max_value=100, value=85, step=1)
    input_sub = st.number_input("Task Submission Rate (%)", min_value=0, max_value=100, value=75, step=1)
    input_feed = st.number_input("Mentor Feedback Score (1-10)", min_value=1, max_value=10, value=7, step=1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    single_input = pd.DataFrame([[input_att, input_sub, input_feed]], columns=['Attendance_Rate_%', 'Task_Submission_Rate_%', 'Mentor_Feedback_10'])
    prob = model.predict_proba(single_input)[0][1] * 100
    
    st.markdown(f"<h2 style='text-align: center; margin-top: 5px;'>{prob:.1f}%</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #cbd5e1; margin-bottom: 10px;'>Success Probability</p>", unsafe_allow_html=True)
    
    if prob >= 75:
        st.success("Status: On Track 🟢")
    elif prob >= 50:
        st.warning("Status: At Marginal Risk 🟡")
    else:
        st.error("Status: Critical / High Risk 🔴")
        
    st.markdown('</div>', unsafe_allow_html=True)