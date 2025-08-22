import io
from datetime import timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config("Study Dashboard", layout="wide")
st.title("📊 Master Study Dashboard — Ethics & HCML")

# ---- CSV LINKS (use /export for reliability)
LECTURES_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRVrMYqxBshP-H7xy3QgoDcgTtEJqmQgU9uuKR2dhkWS3-CASATJ75hXZeb3ucqGxgrPewdkySGnuE4/export?format=csv&gid=0"

# Optional: publish these later and fill; we’ll handle missing gracefully
QUIZZES_CSV = ""     # e.g., .../export?format=csv&gid=<gid_quizzes>
TIMELOG_CSV = ""     # e.g., .../export?format=csv&gid=<gid_daily_plan_or_timelog>

@st.cache_data(ttl=60)
def load_csv(url):
    if not url:
        return pd.DataFrame()
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    txt = r.text.strip()
    if not txt or "," not in txt:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(txt))

# ---- LOAD
lectures = load_csv(LECTURES_CSV)
quizzes  = load_csv(QUIZZES_CSV)
timelog  = load_csv(TIMELOG_CSV)

if lectures.empty:
    st.error("Lectures CSV is empty or not accessible. Make sure the Lectures tab is published as CSV and the link is correct (/export?format=csv&gid=0).")
    st.stop()

# ---- Normalize columns (be flexible with your headers)
cols = {c.lower().strip(): c for c in lectures.columns}
def get_col(*names):
    for n in names:
        key = n.lower()
        if key in cols: return cols[key]
    return None

col_course   = get_col("course")
col_lecture  = get_col("lecture_id","lecture","session","L")  # picks first that exists
col_status   = get_col("status")
col_progress = get_col("progress %","progress_%","progress")
col_know     = get_col("knowledge_%","knowledge %")
# Detect any checkbox-like columns for progress
checkbox_cols = [c for c in lectures.columns if str(c).strip().endswith("?")]

# Fill missing numeric columns if needed
if col_progress and lectures[col_progress].dtype == object:
    lectures[col_progress] = pd.to_numeric(lectures[col_progress], errors="coerce").fillna(0)

if not col_course:
    st.error("No 'Course' column found in Lectures tab. Please add a 'Course' column with values 'Ethics' or 'HCML'.")
    st.stop()

# ---- UI
course = st.selectbox("Choose course:", ["Ethics","HCML"])
lec = lectures[lectures[col_course].astype(str).str.strip().str.lower() == course.lower()].copy()

# ---- Coverage detection rules
def has_progress(row):
    # 1) any TRUE in checkbox cols
    for c in checkbox_cols:
        v = str(row.get(c, "")).strip().lower()
        if v in ("true","1","yes","y"): return True
    # 2) status not "Not started"
    if col_status and str(row.get(col_status,"")).strip().lower() not in ("", "not started"):
        return True
    # 3) progress % > 0
    if col_progress and pd.to_numeric(row.get(col_progress,0), errors="coerce") > 0:
        return True
    return False

coverage = 0.0
if not lec.empty:
    coverage = 100.0 * (lec.apply(has_progress, axis=1).sum() / len(lec))

# ---- Knowledge %
knowledge = 0.0
if col_know and col_know in lec:
    knowledge = pd.to_numeric(lec[col_know], errors="coerce").fillna(0).mean()
elif col_progress and col_progress in lec:
    # fallback: treat Progress % as proxy for knowledge (light heuristic)
    knowledge = pd.to_numeric(lec[col_progress], errors="coerce").fillna(0).mean() * 0.7

# ---- Avg Quiz %
avg_quiz = 0.0
if not quizzes.empty:
    qc = {c.lower(): c for c in quizzes.columns}
    q_course = qc.get("course")
    q_score  = qc.get("score_%") or qc.get("score %")
    if q_course and q_score:
        avg_quiz = pd.to_numeric(
            quizzes[quizzes[q_course].astype(str).str.lower()==course.lower()][q_score],
            errors="coerce"
        ).mean()

# ---- Momentum (7d)
momentum = 0.0
if not timelog.empty:
    tc = {c.lower(): c for c in timelog.columns}
    t_course = tc.get("course")
    t_date   = tc.get("date")
    t_min    = tc.get("minutes")
    if t_course and t_date and t_min:
        df = timelog[timelog[t_course].astype(str).str.lower()==course.lower()].copy()
        df[t_date] = pd.to_datetime(df[t_date], errors="coerce")
        last7 = df[df[t_date] >= pd.Timestamp.today().normalize() - timedelta(days=7)]
        momentum = pd.to_numeric(last7[t_min], errors="coerce").fillna(0).sum()

# ---- Grade-1 probability
perf = 0.6*(avg_quiz or 0) + 0.4*(knowledge or 0)
grade_prob = 100/(1+pow(2.71828, -(0.08*perf + 1.2*(coverage/100) + 0.6*min(1,momentum/600) - 7)))

# ---- KPIs
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Coverage %", f"{coverage:.0f}")
k2.metric("Avg Quiz %", f"{avg_quiz:.1f}")
k3.metric("Knowledge %", f"{knowledge:.0f}")
k4.metric("Momentum (7d min)", int(momentum))
k5.metric("Grade‑1 Probability %", f"{grade_prob:.0f}")

# ---- Table + simple charts
st.subheader("Lectures table")
show_cols = [c for c in [col_lecture, col_status, col_progress, col_know] if c]
st.dataframe(lec[show_cols] if show_cols else lec)

if not quizzes.empty and 'Date' in quizzes.columns and (('Score_%' in quizzes.columns) or ('Score %' in quizzes.columns)):
    ycol = 'Score_%' if 'Score_%' in quizzes.columns else 'Score %'
    qc = {c.lower(): c for c in quizzes.columns}
    q_course = qc.get("course")
    if q_course:
        qdf = quizzes[quizzes[q_course].astype(str).str.lower()==course.lower()].copy()
        try:
            fig = px.line(qdf, x="Date", y=ycol, title="Quiz Scores")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

if not timelog.empty:
    tc = {c.lower(): c for c in timelog.columns}
    t_course = tc.get("course")
    t_date   = tc.get("date")
    t_min    = tc.get("minutes")
    if t_course and t_date and t_min:
        df = timelog[timelog[t_course].astype(str).str.lower()==course.lower()].copy()
        df[t_date] = pd.to_datetime(df[t_date], errors="coerce")
        try:
            fig2 = px.bar(df, x=t_date, y=t_min, title="Study Time Log")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            pass
