# app.py — Master Study Dashboard for Ethics & HCML
import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
from datetime import datetime, timedelta

SCOPE = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
SHEET_URL = st.secrets["SHEET_URL"]
JSON_KEYFILE = "service_account.json"

@st.cache_data(ttl=60)
def load_sheet(tab):
    creds = ServiceAccountCredentials.from_json_keyfile_name(JSON_KEYFILE, SCOPE)
    gc = gspread.authorize(creds)
    sh = gc.open_by_url(SHEET_URL)
    ws = sh.worksheet(tab)
    data = ws.get_all_records()
    return pd.DataFrame(data)

def compute_kpis(course, lec, quizzes, timelog):
    avg_quiz = quizzes[quizzes["Course"]==course]["Score_%"].mean()
    knowledge = lec["Knowledge_%"].mean() if "Knowledge_%" in lec else 0
    coverage = (lec["Studied?"].sum() / len(lec))*100 if len(lec) else 0
    momentum = timelog[(timelog["Course"]==course) &
        (pd.to_datetime(timelog["Date"]) >= pd.Timestamp.today()-timedelta(days=7))]["Minutes"].sum()
    perf = 0.6*(avg_quiz or 0) + 0.4*(knowledge or 0)
    grade_prob = 100/(1+pow(2.71828, -(0.08*perf + 1.2*(coverage/100) + 0.6*min(1,momentum/600) - 7)))
    return coverage, avg_quiz, knowledge, momentum, grade_prob

def main():
    st.set_page_config("Study Dashboard", layout="wide")
    st.title("📊 Master Study Dashboard — Ethics & HCML")

    try:
        ethics = load_sheet("Lectures_Ethics")
        hcml   = load_sheet("Lectures_HCML")
        quizzes = load_sheet("Quizzes")
        timelog = load_sheet("TimeLog")
    except Exception as e:
        st.error(f"Could not load Google Sheet: {e}")
        return

    course = st.selectbox("Choose course:", ["Ethics","HCML"])
    lec = ethics if course == "Ethics" else hcml

    coverage, avg_quiz, knowledge, momentum, grade_prob = compute_kpis(course, lec, quizzes, timelog)

    cols = st.columns(5)
    cols[0].metric("Coverage %", f"{coverage:.0f}")
    cols[1].metric("Avg Quiz %", f"{avg_quiz:.1f}")
    cols[2].metric("Knowledge %", f"{knowledge:.0f}")
    cols[3].metric("Momentum (7d min)", int(momentum))
    cols[4].metric("Grade-1 Probability %", f"{grade_prob:.0f}")

    # Charts
    if not quizzes.empty:
        fig = px.line(quizzes[quizzes["Course"]==course], x="Date", y="Score_%", title="Quiz Scores")
        st.plotly_chart(fig, use_container_width=True)
    if not timelog.empty:
        df = timelog[timelog["Course"]==course].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        fig2 = px.bar(df, x="Date", y="Minutes", title="Study Time Log")
        st.plotly_chart(fig2, use_container_width=True)

    # Today plan
    st.subheader("📅 Today’s Study Plan")
    today = pd.Timestamp.today().normalize()
    if "Next_Review_Date" in lec:
        due_reviews = lec[pd.to_datetime(lec["Next_Review_Date"], errors="coerce")==today]
    else:
        due_reviews = pd.DataFrame()
    if not due_reviews.empty:
        st.write("🔔 Reviews due today:")
        st.table(due_reviews[["Lecture_ID","Title","Mastery_0_5"]])
    else:
        st.write("✅ No reviews due today.")
    if "Mastery_0_5" in lec:
        weak = lec[lec["Mastery_0_5"] <= 2].head(5)
        if not weak.empty:
            st.write("⚠️ Weak spots (low mastery):")
            st.table(weak[["Lecture_ID","Title","Mastery_0_5"]])

if __name__ == "__main__":
    main()
