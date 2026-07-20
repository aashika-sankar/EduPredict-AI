import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import re

from utils.predictor import predict_score
from utils.recommender import generate_recommendations
from utils.pdf_report import generate_pdf

def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Load Model and Encoders
# -----------------------------
model = joblib.load("models/best_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# Load feature importance
feature_importance = pd.read_csv("models/feature_importance.csv")
# Load cleaned dataset for analytics
df = pd.read_csv("dataset/cleaned_student_data.csv")

# -----------------------------
# Tabs for Navigation
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "🎯 Prediction",
    "📊 Dashboard",
    "ℹ️ About"
])

with tab1:
    st.markdown(
    """
    <div class="main-title">
        🎓 EduPredict AI
    </div>
    """,
    unsafe_allow_html=True,
    )

    st.markdown("""
        <div class="sub-title">
        AI-Powered Student Performance Prediction
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sub-title">
            Predict • Analyze • Recommend • Improve
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("👋 Welcome")

    st.write("""
    EduPredict AI is a Machine Learning application that predicts student academic
    performance based on academic, personal, and school-related factors.

    The application uses a trained **CatBoost Regressor** model to estimate
    exam scores and provide personalized insights that help students improve their
    academic performance.
    """)
        
    st.subheader("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.success("🤖 AI-based Score Prediction")
        st.success("📊 Interactive Dashboard")

    with col2:
        st.success("📈 Machine Learning Analytics")
        st.success("💡 Personalized Recommendations")
            
    st.subheader("⚙️ How It Works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("**1️⃣ Enter Details**")

    with col2:
        st.info("**2️⃣ AI Prediction**")

    with col3:
        st.info("**3️⃣ Analyze Report**")

    with col4:
        st.info("**4️⃣ Get Recommendations**")
            
    if st.button("🚀 Predict Student Performance",width="stretch",
        type="primary"):

        st.success("Open the 🎯 Prediction tab to begin.")

with tab2:
    with st.expander("📚 Academic Details", expanded=True):
        
        col1, col2 = st.columns(2)

        with col1:
            hours_studied = st.number_input(
                "Hours Studied (per week)",
                min_value=1,
                max_value=44,
                value=20
            )

            attendance = st.number_input(
                "Attendance (%)",
                min_value=60,
                max_value=100,
                value=80
            )

            previous_scores = st.number_input(
                "Previous Scores",
                min_value=50,
                max_value=100,
                value=75
            )

            sleep_hours = st.number_input(
                "Sleep Hours",
                min_value=4,
                max_value=10,
                value=7
            )

        with col2:
            tutoring_sessions = st.number_input(
                "Tutoring Sessions",
                min_value=0,
                max_value=8,
                value=1
            )

            physical_activity = st.number_input(
                "Physical Activity",
                min_value=0,
                max_value=6,
                value=3
            )

            gender = st.selectbox(
                "Gender",
                encoders["Gender"].classes_
            )

            motivation = st.selectbox(
                "Motivation Level",
                encoders["Motivation_Level"].classes_
            )
        
    with st.expander("🏫 School & Family Details", expanded=True):
        col3, col4 = st.columns(2)

        with col3:
            parental_involvement = st.selectbox(
                "Parental Involvement",
                encoders["Parental_Involvement"].classes_
            )

            access_to_resources = st.selectbox(
                "Access to Resources",
                encoders["Access_to_Resources"].classes_
            )

            extracurricular = st.selectbox(
                "Extracurricular Activities",
                encoders["Extracurricular_Activities"].classes_
            )

            internet_access = st.selectbox(
                "Internet Access",
                encoders["Internet_Access"].classes_
            )

            family_income = st.selectbox(
                "Family Income",
                encoders["Family_Income"].classes_
            )

        with col4:
            teacher_quality = st.selectbox(
                "Teacher Quality",
                encoders["Teacher_Quality"].classes_
            )

            school_type = st.selectbox(
                "School Type",
                encoders["School_Type"].classes_
            )

            peer_influence = st.selectbox(
                "Peer Influence",
                encoders["Peer_Influence"].classes_
            )

            learning_disabilities = st.selectbox(
                "Learning Disabilities",
                encoders["Learning_Disabilities"].classes_
            )

            parental_education = st.selectbox(
                "Parental Education Level",
                encoders["Parental_Education_Level"].classes_
            )

        distance_from_home = st.selectbox(
            "Distance From Home",
            encoders["Distance_from_Home"].classes_
        )
        
    st.divider()

    if st.button("🎯 Predict Exam Score",width="stretch"):

        input_data = {
            "Hours_Studied": hours_studied,
            "Attendance": attendance,
            "Parental_Involvement": encoders["Parental_Involvement"].transform([parental_involvement])[0],
            "Access_to_Resources": encoders["Access_to_Resources"].transform([access_to_resources])[0],
            "Extracurricular_Activities": encoders["Extracurricular_Activities"].transform([extracurricular])[0],
            "Sleep_Hours": sleep_hours,
            "Previous_Scores": previous_scores,
            "Motivation_Level": encoders["Motivation_Level"].transform([motivation])[0],
            "Internet_Access": encoders["Internet_Access"].transform([internet_access])[0],
            "Tutoring_Sessions": tutoring_sessions,
            "Family_Income": encoders["Family_Income"].transform([family_income])[0],
            "Teacher_Quality": encoders["Teacher_Quality"].transform([teacher_quality])[0],
            "School_Type": encoders["School_Type"].transform([school_type])[0],
            "Peer_Influence": encoders["Peer_Influence"].transform([peer_influence])[0],
            "Physical_Activity": physical_activity,
            "Learning_Disabilities": encoders["Learning_Disabilities"].transform([learning_disabilities])[0],
            "Parental_Education_Level": encoders["Parental_Education_Level"].transform([parental_education])[0],
            "Distance_from_Home": encoders["Distance_from_Home"].transform([distance_from_home])[0],
            "Gender": encoders["Gender"].transform([gender])[0]
        }
        
        # -----------------------------
        # Feature Engineering
        # -----------------------------

        input_data["Study_Consistency"] = (
            hours_studied * sleep_hours
        )

        input_data["Balance_Index"] = (
            sleep_hours
            + physical_activity
            + attendance
        )

        input_data["Performance_Trend"] = (
            previous_scores
            + attendance
        )

        input_data["Engagement"] = (
            hours_studied
            + attendance
            + tutoring_sessions
            + encoders["Motivation_Level"].transform([motivation])[0]
        )

        input_data["Study_Attendance"] = (
            hours_studied * attendance
        )

        input_data["Previous_Study"] = (
            previous_scores * hours_studied
        )

        input_data["Attendance_Previous"] = (
            attendance * previous_scores
        )

        input_data["Study_Efficiency"] = (
            previous_scores / (hours_studied + 1)
        )

        import pandas as pd

        input_df = pd.DataFrame([input_data])

        print("\nPrediction Features:")
        for i, col in enumerate(input_df.columns, 1):
            print(f"{i}. {col}")

        print("\nTotal Features:", len(input_df.columns))

        print("\nInput Values:")
        print(input_df.T)

        predicted_score = predict_score(model, input_data)
            
        strengths, recommendations = generate_recommendations(
            hours_studied,
            attendance,
            sleep_hours,
            tutoring_sessions,
            physical_activity,
            motivation
        )
        
        st.success("Prediction completed successfully!")

        st.subheader("📄 Student Performance Report")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🎯 Predicted Score",
                f"{predicted_score:.2f}/100"
            )

            st.progress(min(predicted_score / 100, 1.0))

            st.metric(
                "📅 Attendance",
                f"{attendance}%"
            )

        with col2:
            if predicted_score >= 90:
                performance = "Excellent"

            elif predicted_score >= 75:
                performance = "Very Good"

            elif predicted_score >= 60:
                performance = "Good"

            else:
                performance = "Needs Improvement"

            st.metric(
                "🏆 Performance",
                performance
            )
            
        if predicted_score >= 90:
            summary = (
                "The student is predicted to achieve an excellent academic performance. "
                "The input factors indicate a very high probability of scoring exceptionally well."
            )

        elif predicted_score >= 75:
            summary = (
                "The student is expected to perform very well. "
                "With consistent effort, the score can improve even further."
            )

        elif predicted_score >= 60:
            summary = (
                "The student is predicted to have a good academic performance. "
                "Improving study hours, attendance, and motivation may help increase the score."
            )

        else:
            summary = (
                "The predicted performance is below average. "
                "Increasing study time, improving attendance, and seeking additional academic support may help improve future performance."
            )
            
        st.subheader("📝 Summary")
        st.write(summary)
        
        st.subheader("🤖 AI Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Strengths")

            if strengths:
                for item in strengths:
                    st.success(item)
            else:
                st.info("No major strengths identified.")

        with col2:
            st.markdown("### 📈 Recommendations")

            if recommendations:
                for item in recommendations:
                    st.warning(item)
            else:
                st.success("Excellent! Keep up the good work.")
        
        # Remove emojis for PDF
        pdf_strengths = [
            re.sub(r'[^\x00-\x7F]+', '', item).strip()
            for item in strengths
        ]

        pdf_recommendations = [
            re.sub(r'[^\x00-\x7F]+', '', item).strip()
            for item in recommendations
        ]
               
        generate_pdf(
            "Student_Performance_Report.pdf",
            predicted_score,
            performance,
            attendance,
            summary,
            pdf_strengths,
            pdf_recommendations
        )

        with open("Student_Performance_Report.pdf", "rb") as pdf_file:
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_file,
                file_name="Student_Performance_Report.pdf",
                mime="application/pdf",
                width="stretch"
            )       
        
with tab3:
    st.header("📊 Machine Learning Dashboard")

    st.subheader("📈 Top Features Influencing Prediction")
    
    top_features = feature_importance.head(10)

    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
    )

    fig.update_layout(
        height=600,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Arial",
            size=14,
            color="black"
        ),
        yaxis=dict(
            categoryorder="total ascending",
            tickfont=dict(color="#1E293B", size=13),
            title_font=dict(color="#0F172A", size=15)
        ),
        xaxis=dict(
            tickfont=dict(color="#1E293B", size=13),
            title_font=dict(color="#0F172A", size=15)
        ),
        margin=dict(l=40, r=40, t=60, b=40),

        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    color="#1E293B",
                    width=1.5
                )
            )
        ]
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        textfont=dict(
            color="black",
            size=14
        )
    )
    
    st.plotly_chart(fig,width="stretch")
    
    st.caption(
    "The chart below shows the ten most influential features used by the model for predicting student performance.")
    
    st.subheader("📊 Model Performance Comparison (R² Score)")
    
        # Model Performance Data
    model_df = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "CatBoost",
            "Stacking"
        ],
        "R² Score": [
            0.6875,
            0.0694,
            0.6485,
            0.7343,
            0.7448,
            0.7635,
            0.7576
        ]
    })

    fig = px.bar(
        model_df,
        x="Model",
        y="R² Score",
        text="R² Score",
        title="Comparison of Machine Learning Models",
        color="Model",
        color_discrete_sequence=[
        "#7DB9DE",
        "#7DB9DE",
        "#7DB9DE",
        "#7DB9DE",
        "#7DB9DE",
        "#357CA8",
        "#7DB9DE",]
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Machine Learning Models",
        yaxis_title="R² Score",
        yaxis_range=[0,1],
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Arial",
            size=14,
            color="black"
        ),
        yaxis=dict(
            categoryorder="total ascending",
            tickfont=dict(color="#1E293B", size=13),
            title_font=dict(color="#0F172A", size=15)
        ),
        xaxis=dict(
            tickfont=dict(color="#1E293B", size=13),
            title_font=dict(color="#0F172A", size=15)
        ),
        
        showlegend=False,
        
        margin=dict(l=40, r=40, t=60, b=40),

        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    color="#1E293B",width=1.5)
            )
        ]
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
        textfont=dict(
            color="black",
            size=14
        )
    )

    st.plotly_chart(fig, width="stretch")
    
    st.caption(
    "Comparison of different regression algorithms based on their R² scores.")

    st.divider()
    
    st.subheader("📋 Model Evaluation Summary")
    
    model_metrics = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "CatBoost",
            "Stacking"
        ],
        "R² Score": [
            0.6875,
            0.0694,
            0.6485,
            0.7343,
            0.7448,
            0.7635,
            0.7576
        ],
        "MAE": [
            1.0184,
            1.7110,
            1.0912,
            0.7698,
            0.6490,
            0.5687,
            0.5770
        ],
        "RMSE": [
            2.1017,
            3.6269,
            2.2290,
            1.9381,
            1.8993,
            1.8285,
            1.8509
        ]
    })

    st.dataframe(
        model_metrics.style.highlight_max(
            subset=["R² Score"],
            color="black"
        ),
        width="stretch",
        hide_index=True
    )
    
    st.caption(
    "Performance metrics for all trained machine learning models.")

with tab4:

    st.header("ℹ️ Project Overview")

    st.markdown("""
    EduPredict AI is a machine learning-powered web application designed to
    predict student academic performance based on academic, personal, and
    environmental factors. The application provides performance predictions,
    identifies key influencing factors, and offers personalized recommendations
    to support better learning outcomes.
    """)

    st.divider()

    st.subheader("🎯 Project Objective")

    st.markdown("""
    - Predict student exam scores using Machine Learning.
    - Identify the factors influencing academic performance.
    - Provide personalized recommendations for improvement.
    - Support data-driven educational decision making.
    """)

    st.divider()

    st.subheader("🛠 Technology Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Programming**
        - Python

        **Framework**
        - Streamlit

        **Visualization**
        - Plotly
        """)

    with col2:
        st.markdown("""
        **Machine Learning**
        - Scikit-learn

        **Data Processing**
        - Pandas
        - Joblib
        """)

    st.divider()

    st.subheader("🤖 Machine Learning Model")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Algorithm", "CatBoost Regressor")
        st.metric("Problem Type", "Regression")

    with col2:
        st.metric("Dataset", "6607 Records")
        st.metric("Input Features", "26(19 i/p + 7 engineered)")

    st.divider()

    st.subheader("👩‍💻 Developer")

    st.markdown("""
    **Aashika Sankar**

    M.Sc. Computer Science

    Central University of Kerala

    AI/ML Internship – Srishti Innovative
    """)