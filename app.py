import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Retention Dashboard", layout="wide")

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3 {
    color: #00FFAA;
}
.stButton>button {
    background-color: #00FFAA;
    color: black;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("student_data.csv")

X = df.drop("drop", axis=1)
y = df["drop"]

model = RandomForestClassifier()
model.fit(X, y)

# ---------------- HEADER ----------------
st.title("🚀 AI Student Retention Intelligence Dashboard")
st.caption("Predict, Analyze, and Improve Student Retention using AI")

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("👥 Total Students", len(df))
col2.metric("⚠️ Drop-off Rate", f"{round(df['drop'].mean()*100,2)}%")
col3.metric("📊 Avg Sessions", round(df['sessions'].mean(),2))

st.divider()

# ---------------- INPUT ----------------
st.subheader("🔍 Predict Student Risk")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 15, 30, 20)
    time_spent = st.slider("Time Spent", 0, 300, 100)
    sessions = st.slider("Sessions", 0, 20, 5)

with col2:
    quiz_score = st.slider("Quiz Score", 0, 100, 60)
    last_active_days = st.slider("Last Active Days", 0, 15, 3)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Now"):

    input_data = [[age, time_spent, sessions, quiz_score, last_active_days]]
    
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Save user input
    user_data = {
        "sessions": sessions,
        "time_spent": time_spent,
        "quiz_score": quiz_score,
        "last_active_days": last_active_days
    }

    st.subheader("📊 Prediction Result")
    st.write(f"Drop-off Probability: **{round(prob*100,2)}%**")

    if prob > 0.7:
        st.error("🔴 High Risk of Drop-off")
    elif prob > 0.4:
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    # ---------------- PERSONALIZED INSIGHTS ----------------
    st.subheader("🧠 Personalized Insights")

    if prediction == 1:
        st.write("⚠️ This student is likely to disengage.")
        if sessions < 3:
            st.write("👉 Low sessions → poor engagement")
        if quiz_score < 50:
            st.write("👉 Low quiz score → learning difficulty")
        if last_active_days > 5:
            st.write("👉 Inactive → high churn risk")
    else:
        st.write("✅ Student is engaged and likely to continue.")
        if sessions > 8:
            st.write("👉 High engagement detected")
        if quiz_score > 70:
            st.write("👉 Strong academic performance")

    # ---------------- RECOMMENDATIONS ----------------
    st.subheader("💡 Smart Recommendations")

    if prediction == 1:
        st.write("👉 Send personalized reminders")
        st.write("👉 Provide easier or engaging content")
        st.write("👉 Offer mentor support")
    else:
        st.write("👉 Encourage advanced learning")
        st.write("👉 Maintain engagement with challenges")

    # ---------------- DATA INSIGHTS ----------------
    st.divider()
    st.subheader("📈 Data Insights (Dynamic)")

    fig1, ax1 = plt.subplots()

    ax1.scatter(df["sessions"], df["time_spent"], 
                c=df["drop"], cmap="coolwarm", s=80, label="Dataset")

    ax1.scatter(user_data["sessions"], user_data["time_spent"], 
                color="yellow", s=200, label="Your Input", edgecolors="black")

    ax1.set_xlabel("Sessions")
    ax1.set_ylabel("Time Spent")
    ax1.set_title("Engagement vs Time Spent")
    ax1.legend()

    st.pyplot(fig1)

    # ---------------- QUIZ DISTRIBUTION ----------------
    fig2, ax2 = plt.subplots()

    ax2.hist(df["quiz_score"], bins=6, color="skyblue", edgecolor="black")
    ax2.axvline(quiz_score, color='red', linestyle='dashed', linewidth=2)

    ax2.set_title("Quiz Score Distribution (Dataset vs Your Input)")
    ax2.set_xlabel("Quiz Score")
    ax2.set_ylabel("Number of Students")

    st.pyplot(fig2)

    st.markdown(f"""
    **📊 Insight:**  
    Your student's quiz score is **{quiz_score}**.  
    Lower scores indicate higher drop-off risk, while higher scores improve retention.
    """)

    # ---------------- DYNAMIC KEY DRIVERS ----------------
    st.divider()
    st.subheader("🧠 Key Drivers of Drop-off (Dynamic)")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig3, ax3 = plt.subplots()
    ax3.barh(importance_df["Feature"], importance_df["Importance"], color="orange")
    ax3.invert_yaxis()
    ax3.set_title("Feature Impact on Prediction")

    st.pyplot(fig3)

    st.markdown("### 🔍 What influenced this prediction:")

    for feature in importance_df.head(3)["Feature"]:
        
        if feature == "sessions":
            if sessions < 3:
                st.write("👉 Low sessions → strong drop-off signal")
            else:
                st.write("👉 High sessions → strong retention signal")

        elif feature == "quiz_score":
            if quiz_score < 50:
                st.write("👉 Low quiz score → learning difficulty → drop-off risk")
            else:
                st.write("👉 High quiz score → good performance → retention")

        elif feature == "time_spent":
            if time_spent < 60:
                st.write("👉 Low time spent → low engagement")
            else:
                st.write("👉 High time spent → strong engagement")

        elif feature == "last_active_days":
            if last_active_days > 5:
                st.write("👉 Inactive → high churn risk")
            else:
                st.write("👉 Recently active → good retention")

        elif feature == "age":
            st.write("👉 Age has minor influence compared to engagement")

    # ---------------- BUSINESS INSIGHTS ----------------
    st.divider()
    st.subheader("📊 Business Insights (Dynamic)")

    if prediction == 1:
        st.error("⚠️ High drop-off risk detected")

        if sessions < 3:
            st.write("👉 Low engagement is the main issue")

        if quiz_score < 50:
            st.write("👉 Student struggling with content")

        if last_active_days > 5:
            st.write("👉 Student inactive for long time")

    else:
        st.success("✅ Strong retention behavior detected")

        if sessions > 8:
            st.write("👉 High engagement → strong retention")

        if quiz_score > 70:
            st.write("👉 Good performance → likely to continue")

else:
    st.info("👉 Enter student details and click Predict to see dynamic insights")