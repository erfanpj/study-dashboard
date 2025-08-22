# app.py — Master Study Dashboard (no service account needed)
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

# --- CONFIG ---
SHEET_LINKS = {
    "Lectures_Ethics": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRVrMYqxBshP-H7xy3QgoDcgTtEJqmQgU9uuKR2dhkWS3-CASATJ75hXZeb3ucqGxgrPewdkySGnuE4/pub?gid=320626341&single=true&output=csv",
    "Lectures_HCML":   "https://docs.google.com/spreadsheets/d/e/2PACX-1vRVrMYqxBshP-H7xy3QgoDcgTtEJqmQgU9uuKR2dhkWS3-CASATJ75hXZeb3ucqGxgrPewdkySGnuE4/pub?gid=396133361&single=true&output=csv",
    "Quizzes":         "https://docs.google.com/spreadsheets/d/e/2PACX-1vRVrMYqxBshP-H7xy3QgoDcgTtEJqmQgU9uuKR2dhkWS3-CASATJ75hXZeb3ucqGxgrPewdkySGnuE4/pub?gid=557189726&single=true&output=csv",
    "TimeLog":         "https://docs.google.com/spreadsheets/d/e/2PACX-1vRVrMYqxBshP-H7xy3QgoDcgTtEJqmQgU9uuKR2dhkWS3-CASATJ75hXZeb3ucqGxgrPewdkySGnuE4/pub?gid=1542076677&single=true&output=csv"
}


@st.cache_data(ttl=60)
def load_csv(url):
    return pd.read_csv(url)

# --- LOAD ---
try:
    ethics  = load_csv(SHEET_LINKS["Lectures_Ethics"])
    hcml    = load_csv(SHEET_LINKS["Lectures_HCML"])
    quizzes = load_csv(SHEET_LINKS["Quizzes"])
    timelog = load_csv(SHEET_LINKS["TimeLog"])
except Exception as e:
    st.error(f"Error loading Google Sheet: {e}")
    st.stop()

# --- LAYOUT ---
st.set_page_config("Study Dashboard", layout="wide")
st.title("📊 Master Study Dashboard — Ethics & HCML")

course = st.selectbox("Choose course:", ["Ethics","HCML"])
lec = ethics if course=="Ethics" else hcml

# --- KPIs ---
def compute_kpis(course, lec, quizzes, timelog):
    avg_quiz = quizzes[quizzes["Course"]==course]["Score_%"].mean()
    knowledge = lec["Knowledge_%"].mean() if "Knowledge_%" in lec else 0
    coverage = (lec["Studied?"].sum() / len(lec))*100 if len(lec) else 0
    momentum = timelog[
        (timelog["Course"]==course) &
        (pd.to_datetime(timelog["Date"], errors="coerce") >= pd.Timestamp.today()-timedelta(days=7))
    ]["Minutes"].sum()
    perf = 0.6*(avg_quiz or 0) + 0.4*(knowledge or 0)
    grade_prob = 100/(1+pow(2.71828, -(0.08*perf + 1.2*(coverage/100) + 0.6*min(1,momentum/600) - 7)))
    return coverage, avg_quiz, knowledge, momentum, grade_prob

coverage, avg_quiz, knowledge, momentum, grade_prob = compute_kpis(course, lec, quizzes, timelog)

cols = st.columns(5)
cols[0].metric("Coverage %", f"{coverage:.0f}")
cols[1].metric("Avg Quiz %", f"{avg_quiz:.1f}")
cols[2].metric("Knowledge %", f"{knowledge:.0f}")
cols[3].metric("Momentum (7d min)", int(momentum))
cols[4].metric("Grade-1 Probability %", f"{grade_prob:.0f}")

# --- Charts ---
if not quizzes.empty:
    fig = px.line(quizzes[quizzes["Course"]==course], x="Date", y="Score_%", title="Quiz Scores")
    st.plotly_chart(fig, use_container_width=True)

if not timelog.empty:
    df = timelog[timelog["Course"]==course].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    fig2 = px.bar(df, x="Date", y="Minutes", title="Study Time Log")
    st.plotly_chart(fig2, use_container_width=True)

# --- Today’s Plan ---
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
