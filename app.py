# app.py — Master Study Dashboard (multi‑sheet, no service account)
# Streamlit 1.36+ | Python 3.10+
# Reads 6 published CSV tabs from your Google Sheet:
#   Lecture (master), Lectures_Ethics, Lectures_HCML, Quizzes, TimeLog, Files_Index
# KPIs: remaining lectures, study coverage, knowledge progress, avg score, performance,
# effort to reach grade 1, probability of grade 1, and LIVE Predicted Exam Score (+ German grade).
# Charts: score trend, distribution, weekly minutes, per‑course timelines. Includes a What‑If planner.

from __future__ import annotations

import math
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Master Study Dashboard — Multi‑Sheet", page_icon="🎓", layout="wide")

# --------------------------------------------------------------------------------------
# 1) YOUR EXACT PUBLISHED CSV LINKS (as provided)
# --------------------------------------------------------------------------------------
SHEET_LINKS = {
    "Lecture":        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSQKJm3tm5rn2Y6TkwxetJOgPNC1UyqYVeFXa5uYl1wj7g-8JJEgiSiEoOn8Na-1Q/pub?gid=1660507142&single=true&output=csv",
    "Files_Index":    "https://docs.google.com/spreadsheets/d/e/2PACX-1vSQKJm3tm5rn2Y6TkwxetJOgPNC1UyqYVeFXa5uYl1wj7g-8JJEgiSiEoOn8Na-1Q/pub?gid=1060235884&single=true&output=csv",
    "Lectures_HCML":  "https://docs.google.com/spreadsheets/d/e/2PACX-1vSQKJm3tm5rn2Y6TkwxetJOgPNC1UyqYVeFXa5uYl1wj7g-8JJEgiSiEoOn8Na-1Q/pub?gid=2022662897&single=true&output=csv",
    "Lectures_Ethics":"https://docs.google.com/spreadsheets/d/e/2PACX-1vSQKJm3tm5rn2Y6TkwxetJOgPNC1UyqYVeFXa5uYl1wj7g-8JJEgiSiEoOn8Na-1Q/pub?gid=1204091179&single=true&output=csv",
    "Quizzes":        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSQKJm3tm5rn2Y6TkwxetJOgPNC1UyqYVeFXa5uYl1wj7g-8JJEgiSiEoOn8Na-1Q/pub?gid=1542821464&single=true&output=csv",
    "TimeLog":        "https://docs.google.com/spreadsheets/d/e/2PACX-1vSQKJm3tm5rn2Y6TkwxetJOgPNC1UyqYVeFXa5uYl1wj7g-8JJEgiSiEoOn8Na-1Q/pub?gid=264744847&single=true&output=csv",
}

# IMPORTANT: make these match the actual "Course" values in your sheets
COURSE_LABELS = {
    "ethics": "Ethics",
    "hcml":   "HCML",
}

# --------------------------------------------------------------------------------------
# 2) STYLES
# --------------------------------------------------------------------------------------
CSS = """
<style>
.block-container {padding-top: 1rem}
.kpi-card{background:rgba(255,255,255,.55);backdrop-filter:blur(8px);border:1px solid rgba(200,200,200,.45);
border-radius:18px;padding:16px 18px;box-shadow:0 10px 30px rgba(0,0,0,.05)}
.kpi-title{font-weight:600;opacity:.8}
.kpi-value{font-size:1.8rem;font-weight:800;margin-top:6px}
.kpi-sub{font-size:.9rem;opacity:.65}
.gradient-bar{height:8px;border-radius:999px;background:linear-gradient(90deg,#22c55e,#3b82f6,#a855f7)}
.badge{display:inline-block;padding:4px 10px;border-radius:12px;font-size:.8rem;border:1px solid rgba(200,200,200,.45);background:rgba(255,255,255,.35)}
a.lec { text-decoration: none; }
.pred-meter {height:10px;border-radius:999px;background:#eee;overflow:hidden}
.pred-meter > div {height:100%}
.small-dim{opacity:.7;font-size:.9rem}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def kpi_card(title, value, sub="", emoji=""):
    st.markdown(f"""
    <div class="kpi-card"><div class="kpi-title">{emoji} {title}</div>
    <div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>
    """, unsafe_allow_html=True)

def safe_div(a,b): return float(a)/float(b) if b else 0.0

def course_color(name:str)->str:
    n=(name or "").lower()
    if n.startswith("ethic"): return "#3b82f6"   # blue
    if "hcml" in n or "h&i" in n or "human" in n: return "#a855f7"  # purple
    return "#22c55e"

@st.cache_data(ttl=30, show_spinner=False)
def load_csv(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url)
    except Exception:
        return pd.DataFrame()

def coerce_dates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pick(df: pd.DataFrame, *names):
    for n in names:
        if n in df.columns: return n
    return None

def canon_course(s: str) -> str:
    s = (s or "").strip().lower()
    if "ethic" in s: return "Ethics"
    if "hcml" in s or "h&i" in s or "human" in s: return "HCML"
    return s.title() if s else s

# --------------------------------------------------------------------------------------
# GERMAN GRADE CONVERSION (typical TU‑D style; adjust thresholds if needed)
# Maps percentage → German 1.0(best) ... 4.0(pass) ... 5.0(fail)
# --------------------------------------------------------------------------------------
GERMAN_GRADE_MAP = [
    (95, 1.0), (90, 1.3), (85, 1.7), (80, 2.0),
    (75, 2.3), (70, 2.7), (65, 3.0), (60, 3.3),
    (55, 3.7), (50, 4.0), (0, 5.0)
]

def percent_to_german_grade(pct: float) -> float:
    try:
        p = float(pct)
    except Exception:
        return 5.0
    for threshold, grade in GERMAN_GRADE_MAP:
        if p >= threshold:
            return float(grade)
    return 5.0

# --------------------------------------------------------------------------------------
# 3) LOAD SHEETS
# --------------------------------------------------------------------------------------
st.title("🎓 Master Study Dashboard (Multi‑Sheet)")
st.caption("Live from your Google Sheets: Lecture, Lectures_Ethics, Lectures_HCML, Quizzes, TimeLog, Files_Index")
st.markdown('<div class="gradient-bar"></div>', unsafe_allow_html=True)

colA,colB,_ = st.columns([1,1,3])
with colA:
    if st.button("🔄 Refresh"):
        load_csv.clear()
        st.experimental_rerun()
with colB:
    auto = st.toggle("Auto‑refresh (30s)", value=False)
if auto:
    time.sleep(30); st.experimental_rerun()

df_master  = load_csv(SHEET_LINKS["Lecture"])
df_files   = load_csv(SHEET_LINKS["Files_Index"])
df_ethics  = load_csv(SHEET_LINKS["Lectures_Ethics"])
df_hcml    = load_csv(SHEET_LINKS["Lectures_HCML"])
df_quizzes = load_csv(SHEET_LINKS["Quizzes"])
df_time    = load_csv(SHEET_LINKS["TimeLog"])

# --------------------------------------------------------------------------------------
# 4) NORMALIZATION
# --------------------------------------------------------------------------------------
def normalize_lectures_generic(df: pd.DataFrame, default_course: str | None) -> pd.DataFrame:
    """
    Returns unified columns:
    Course, LectureID, LectureName, Status, StudyMinutes, ReviewCount, LastReviewDate, Mastery
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Course","LectureID","LectureName","Status","StudyMinutes","ReviewCount","LastReviewDate","Mastery"])

    course_c = pick(df, "Course","course","Session")
    idc      = pick(df, "Lecture_ID","LectureID","ID","LecID","Lecture Id","Session","Folder Name")
    namec    = pick(df, "Title","LectureName","Name","File Name","Session Title")
    statc    = pick(df, "Status","Studied?","State")
    minc     = pick(df, "StudyMinutes","Study_Min","Minutes","Total_Minutes","Duration")
    revc     = pick(df, "ReviewCount","Reviews","Review_Count")
    lastc    = pick(df, "Last_Review_Date","LastReviewDate","Last_Review","LastReview")
    mastc    = pick(df, "Mastery_0_5","Mastery","Knowledge_%")

    out = pd.DataFrame({
        "Course": df.get(course_c) if course_c else default_course,
        "LectureID": df.get(idc, df.get(namec, pd.Series(dtype=object))),
        "LectureName": df.get(namec, df.get(idc, pd.Series(dtype=object))),
        "Status": df.get(statc, "Not started").fillna("Not started"),
        "StudyMinutes": df.get(minc, 0),
        "ReviewCount": df.get(revc, 0),
        "LastReviewDate": df.get(lastc, pd.NaT),
        "Mastery": df.get(mastc, np.nan),
    })
    out = coerce_numeric(out, ["StudyMinutes","ReviewCount","Mastery"])
    out = coerce_dates(out, ["LastReviewDate"])
    if "Mastery" in out.columns:
        mx = out["Mastery"].dropna().max()
        if pd.notna(mx) and mx > 1.5:
            out["Mastery"] = (out["Mastery"]/5.0).clip(0,1)
    out["Course"] = out["Course"].astype(str).map(canon_course) if "Course" in out.columns else default_course
    return out

lect_master = normalize_lectures_generic(df_master, None)                         # should already include Course
lect_ethics = normalize_lectures_generic(df_ethics, COURSE_LABELS["ethics"])
lect_hcml   = normalize_lectures_generic(df_hcml,   COURSE_LABELS["hcml"])

lectures = pd.concat([lect_master, lect_ethics, lect_hcml], ignore_index=True)

def dedupe_lectures(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["_info_score"] = x["StudyMinutes"].fillna(0) + x["ReviewCount"].fillna(0) + x["Mastery"].fillna(0)
    if "LectureID" in x.columns:
        x = x.sort_values(["Course","LectureID","_info_score"], ascending=[True,True,False])
        x = x.drop_duplicates(subset=["Course","LectureID"], keep="first")
    x = x.sort_values(["Course","LectureName","_info_score"], ascending=[True,True,False])
    x = x.drop_duplicates(subset=["Course","LectureName"], keep="first")
    return x.drop(columns=["_info_score"], errors="ignore")

lectures = dedupe_lectures(lectures)

def normalize_files(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["LectureID","LectureName","FilePath"])
    idc   = pick(df, "Lecture_ID","LectureID","ID","LecID","Folder Name")
    namec = pick(df, "Title","LectureName","Name","File Name")
    pathc = pick(df, "Path","FilePath","DrivePath","URL","Link","File URL")
    out = pd.DataFrame({
        "LectureID": df.get(idc, pd.Series(dtype=object)),
        "LectureName": df.get(namec, pd.Series(dtype=object)),
        "FilePath": df.get(pathc, pd.Series(dtype=object)),
    })
    return out

files = normalize_files(df_files)

def attach_files(lectures: pd.DataFrame, files: pd.DataFrame) -> pd.DataFrame:
    L = lectures.copy()
    F = files.copy()
    if "LectureID" in L.columns and "LectureID" in F.columns:
        L = L.merge(F[["LectureID","FilePath"]], on="LectureID", how="left", suffixes=("","_byID"))
    if "LectureName" in L.columns and "LectureName" in F.columns:
        L = L.merge(F[["LectureName","FilePath"]].rename(columns={"FilePath":"FilePath_byName"}),
                    on="LectureName", how="left")
    L["FilePath"] = L["FilePath"].fillna(L.get("FilePath_byName"))
    return L.drop(columns=[c for c in ["FilePath_byName"] if c in L.columns])

lectures = attach_files(lectures, files)

def normalize_quizzes(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Course","LectureID","Date","Type","Score"])
    datec  = pick(df, "Date","Datetime")
    course = pick(df, "Course","Session")
    lec    = pick(df, "Lecture_ID","LectureID","LecID","Lecture","Folder Name")
    typc   = pick(df, "Type","Mode")
    sc     = pick(df, "Score_%","Score","Result")
    out = pd.DataFrame({
        "Course": df.get(course, pd.Series(dtype=object)),
        "LectureID": df.get(lec, pd.Series(dtype=object)),
        "Date": df.get(datec, pd.NaT),
        "Type": df.get(typc, "Quiz"),
        "Score": df.get(sc, np.nan),
    })
    out = coerce_dates(out, ["Date"])
    out = coerce_numeric(out, ["Score"])
    if "Score" in out.columns:
        max_s = out["Score"].dropna().max()
        if pd.notna(max_s) and max_s <= 1.5:
            out["Score"] = (out["Score"]*100).clip(0,100)
    out["Course"] = out["Course"].astype(str).map(canon_course)
    return out

scores = normalize_quizzes(df_quizzes)

def normalize_timelog(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Course","LectureID","Date","Minutes","Type"])
    datec = pick(df, "Date","Datetime")
    course= pick(df, "Course","Session")
    lec   = pick(df, "Lecture_ID","LectureID","LecID","Lecture","Folder Name")
    minc  = pick(df, "Minutes","Min","Duration")
    typc  = pick(df, "Type","Activity")
    out = pd.DataFrame({
        "Course": df.get(course, pd.Series(dtype=object)),
        "LectureID": df.get(lec, pd.Series(dtype=object)),
        "Date": df.get(datec, pd.NaT),
        "Minutes": df.get(minc, 0),
        "Type": df.get(typc, "Study"),
    })
    out = coerce_dates(out, ["Date"])
    out = coerce_numeric(out, ["Minutes"])
    out["Course"] = out["Course"].astype(str).map(canon_course)
    return out

sessions = normalize_timelog(df_time)

# Make sure course names are canonical everywhere
for df_ in [lectures, scores, sessions]:
    if not df_.empty and "Course" in df_.columns:
        df_["Course"] = df_["Course"].astype(str).map(canon_course)

# --------------------------------------------------------------------------------------
# 5) KPIs
# --------------------------------------------------------------------------------------
def compute_course_kpis(course_name: str,
                        lectures: pd.DataFrame,
                        scores: pd.DataFrame) -> Dict[str, float]:
    l = lectures[lectures["Course"] == course_name].copy()
    s = scores[scores["Course"] == course_name].copy()

    # total lectures: prefer LectureID, else LectureName, else row count
    if not l.empty:
        if "LectureID" in l.columns and l["LectureID"].notna().any():
            total_lectures = l["LectureID"].dropna().astype(str).nunique()
        elif "LectureName" in l.columns and l["LectureName"].notna().any():
            total_lectures = l["LectureName"].dropna().astype(str).nunique()
        else:
            total_lectures = len(l)
    else:
        total_lectures = 0

    studied_mask = (l.get("Status","").isin(["In Progress","Studied","Completed"])) | (l.get("StudyMinutes",0) > 0) if not l.empty else pd.Series([], dtype=bool)
    studied_lectures = int(l[studied_mask]["LectureID"].astype(str).nunique()) if total_lectures else 0
    coverage = safe_div(studied_lectures, total_lectures)

    # knowledge progress
    if "Mastery" in l.columns and not l["Mastery"].isna().all():
        knowledge = l["Mastery"].fillna(0).clip(0,1).mean()
    else:
        target_reviews = 3
        rc = (l.get("ReviewCount", pd.Series(dtype=float)).fillna(0)/target_reviews).clip(0,1)
        knowledge = float(rc.mean()) if len(rc) else 0.0

    avg_score = float(s["Score"].mean()) if ("Score" in s.columns and not s.empty) else 0.0

    performance = 0.35*(coverage*100) + 0.35*(knowledge*100) + 0.30*avg_score

    # effort to reach grade 1 (heuristic)
    target_cov, target_kn, target_sc = 0.95, 0.85, 90.0
    minutes_for_cov = max(0.0, target_cov-coverage)*max(1,total_lectures)*90
    minutes_for_kn  = max(0.0, target_kn-knowledge)*max(1,total_lectures)*20*3
    minutes_for_sc  = max(0.0, target_sc-avg_score)*2
    effort_minutes  = minutes_for_cov+minutes_for_kn+minutes_for_sc

    # Live probability of Grade 1
    z = (2.6*(avg_score/100.0 - 0.85) + 2.0*(knowledge - 0.80) + 2.0*(coverage - 0.90))
    p1 = float(1/(1+math.exp(-z)))
    return dict(
        total_lectures=total_lectures,
        studied_lectures=studied_lectures,
        study_coverage=coverage,
        knowledge_progress=knowledge,
        avg_score=avg_score,
        performance=performance,
        effort_minutes=effort_minutes,
        grade1_prob=p1
    )

course_list = [COURSE_LABELS["ethics"], COURSE_LABELS["hcml"]]
per_course = {}
agg = dict(total=0, studied=0, know_sum=0, scores=[], perf=[], prob=[])

for cname in course_list:
    k = compute_course_kpis(cname, lectures, scores)
    per_course[cname] = k
    agg["total"] += k["total_lectures"]
    agg["studied"] += k["studied_lectures"]
    agg["know_sum"] += k["knowledge_progress"]
    if k["avg_score"] > 0: agg["scores"].append(k["avg_score"])
    agg["perf"].append(k["performance"]); agg["prob"].append(k["grade1_prob"])

overall_cov = safe_div(agg["studied"], agg["total"]) if agg["total"] else 0.0
overall_kn  = safe_div(agg["know_sum"], len(course_list)) if course_list else 0.0
overall_sc  = float(np.mean(agg["scores"])) if agg["scores"] else 0.0
overall_pf  = float(np.mean(agg["perf"])) if agg["perf"] else 0.0
overall_pb  = float(np.mean(agg["prob"])) if agg["prob"] else 0.0

# --------------------------------------------------------------------------------------
# 5.1) LIVE PREDICTED EXAM SCORE (with German grade)
# --------------------------------------------------------------------------------------
def _bayes_mean(values, prior_mean=0.7, prior_strength=5):
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if v.empty:
        return float(prior_mean)
    n = len(v)
    m = float(v.clip(0,1).mean())
    return (prior_strength*prior_mean + n*m) / (prior_strength + n)

def compute_predicted_exam_score(course_name: str,
                                 lectures: pd.DataFrame,
                                 scores: pd.DataFrame,
                                 sessions: pd.DataFrame) -> Dict[str, float]:
    L = lectures[lectures["Course"] == course_name].copy() if not lectures.empty else pd.DataFrame()
    S = scores[scores["Course"] == course_name].copy() if not scores.empty else pd.DataFrame()
    T = sessions[sessions["Course"] == course_name].copy() if not sessions.empty else pd.DataFrame()

    # coverage
    if not L.empty:
        if "LectureID" in L.columns and L["LectureID"].notna().any():
            total_lectures = L["LectureID"].dropna().astype(str).nunique()
        elif "LectureName" in L.columns and L["LectureName"].notna().any():
            total_lectures = L["LectureName"].dropna().astype(str).nunique()
        else:
            total_lectures = len(L)
    else:
        total_lectures = 0
    studied_mask = (L.get("Status","").isin(["In Progress","Studied","Completed"])) | (L.get("StudyMinutes",0) > 0) if not L.empty else pd.Series([], dtype=bool)
    studied_lectures = int(L[studied_mask]["LectureID"].astype(str).nunique()) if total_lectures else 0
    coverage = safe_div(studied_lectures, total_lectures)
    cov_n = int(total_lectures)

    # knowledge
    if "Mastery" in L.columns and not L["Mastery"].isna().all():
        knowledge = float(L["Mastery"].fillna(0).clip(0,1).mean())
        kn_n = int(L["Mastery"].notna().sum())
    else:
        target_reviews = 3
        rc = (L.get("ReviewCount", pd.Series(dtype=float)).fillna(0)/target_reviews).clip(0,1)
        knowledge = float(rc.mean()) if len(rc) else 0.0
        kn_n = int(rc.notna().sum()) if len(rc) else 0

    # quiz (0..1 smoothed)
    quiz_pct = np.nan
    quiz_n = 0
    if not S.empty and "Score" in S.columns:
        vals = pd.to_numeric(S["Score"], errors="coerce").dropna()
        if not vals.empty:
            if (vals > 1.5).any():
                vals = vals/100.0
            quiz_pct = float(vals.clip(0,1).mean())
            quiz_n = int(len(vals))
    quiz_smoothed = _bayes_mean([] if np.isnan(quiz_pct) else [quiz_pct], prior_mean=0.7, prior_strength=5)

    # flow: minutes last 14 days vs 600‑min target
    flow = 0.5
    flow_n = 0
    if not T.empty and "Date" in T.columns and "Minutes" in T.columns:
        recent = T.dropna(subset=["Date","Minutes"]).copy()
        if not recent.empty:
            recent["Date"] = pd.to_datetime(recent["Date"], errors="coerce")
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=14)
            recent = recent[recent["Date"] >= cutoff]
            mins14 = float(pd.to_numeric(recent["Minutes"], errors="coerce").fillna(0).sum())
            target14 = 600.0
            if target14 > 0:
                flow = float(np.clip(mins14/target14, 0, 1))
            flow_n = int(len(recent))

    # dynamic weights by reliability
    w_cov = 0.35 * (1.0 if cov_n >= 8 else 0.8 if cov_n >= 3 else 0.6)
    w_kn  = 0.25 * (1.0 if kn_n  >= 8 else 0.8 if kn_n  >= 3 else 0.6)
    w_qz  = 0.35 * (1.0 if quiz_n>=20 else 0.8 if quiz_n>=10 else 0.6)
    w_fl  = 0.05 * (1.0 if flow_n>=6 else 0.7)

    comps = np.array([coverage, knowledge, quiz_smoothed, flow], dtype=float)
    weights = np.array([w_cov, w_kn, w_qz, w_fl], dtype=float)
    mask = np.isfinite(comps)
    weights[~mask] = 0.0
    if weights.sum() == 0:
        weights = np.ones_like(weights) * mask
    weights = weights / weights.sum()
    blended = float(np.clip(np.sum(weights * comps), 0, 1))
    predicted_score = round(blended * 100.0, 1)

    # confidence
    n_evidence = (cov_n/10) + (kn_n/10) + (quiz_n/25) + (flow_n/10)
    n_evidence = float(np.clip(n_evidence, 0, 1))
    dispersion = float(np.std(comps[mask])) if mask.any() else 0.35
    agreement = float(np.clip(1 - (dispersion / 0.35), 0, 1))
    confidence = round(0.5*n_evidence + 0.5*agreement, 2)

    # German grade from predicted %
    de_grade = percent_to_german_grade(predicted_score)

    return dict(
        predicted_score=predicted_score,
        predicted_de_grade=de_grade,
        confidence=confidence,
        components={
            "coverage": round(coverage*100,1),
            "knowledge": round(knowledge*100,1),
            "quiz": round(quiz_smoothed*100,1),
            "flow": round(flow*100,1)
        },
        weights={
            "coverage": round(float(weights[0]),2),
            "knowledge": round(float(weights[1]),2),
            "quiz": round(float(weights[2]),2),
            "flow": round(float(weights[3]),2),
        }
    )

def render_predicted_exam_card(course_name: str, color: str):
    res = compute_predicted_exam_score(course_name, lectures, scores, sessions)
    conf = int(round(res["confidence"]*100))
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-title">📈 Predicted Exam Score — {course_name}</div>
      <div class="kpi-value">{res['predicted_score']} %  <span class="small-dim">≈ Grade (DE): {res['predicted_de_grade']:.1f}</span></div>
      <div class="kpi-sub">Confidence: {conf}%</div>
      <div class="pred-meter" style="margin-top:8px">
        <div style="width:{res['predicted_score']}%; background:{color}"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details"):
        st.write(pd.DataFrame({
            "component":["Coverage","Knowledge","Quiz","Study flow"],
            "value %":[res["components"]["coverage"],res["components"]["knowledge"],res["components"]["quiz"],res["components"]["flow"]],
            "weight":[res["weights"]["coverage"],res["weights"]["knowledge"],res["weights"]["quiz"],res["weights"]["flow"]],
        }))

# --------------------------------------------------------------------------------------
# 6) TOP KPIs
# --------------------------------------------------------------------------------------
k = st.columns(4)
with k[0]: kpi_card("Lectures Remaining (All)", f"{max(0, agg['total']-agg['studied'])}", f"Total: {agg['total']}", "📚")
with k[1]: kpi_card("Overall Study Coverage", f"{overall_cov*100:.1f}%", "Across both courses", "🧭")
with k[2]: kpi_card("Avg Knowledge Progress", f"{overall_kn*100:.1f}%", "Mastery estimate", "🧠")
with k[3]:
    avg_grade_de = percent_to_german_grade(overall_sc) if overall_sc>0 else 5.0
    kpi_card("Avg Quiz/Test Score", f"{overall_sc:.1f}", f"≈ Avg Grade (DE): {avg_grade_de:.1f}", "🎯")

# Quick overall predicted score (mean of per-course) + DE grade
overall_preds = []
overall_de_grades = []
for c in course_list:
    r = compute_predicted_exam_score(c, lectures, scores, sessions)
    overall_preds.append(r["predicted_score"])
    overall_de_grades.append(r["predicted_de_grade"])
overall_predicted = round(float(np.mean(overall_preds)) if overall_preds else 55.0, 1)
overall_predicted_de = round(float(np.mean(overall_de_grades)) if overall_de_grades else 4.0, 1)

st.markdown("<br/>", unsafe_allow_html=True)
k2 = st.columns(1)
with k2[0]:
    kpi_card("Overall Predicted Exam Score", f"{overall_predicted} %",
             f"Average of course predictions — ≈ Grade (DE): {overall_predicted_de:.1f}", "📊")

# --------------------------------------------------------------------------------------
# 7) TABS
# --------------------------------------------------------------------------------------
tabs = st.tabs(["Overview","Course Drilldowns","Lectures","Scores","Study Sessions","Files","Schema"])

# OVERVIEW
with tabs[0]:
    a,b = st.columns(2)
    with a:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=max(0,min(overall_pf,100)),
                  number={'suffix': "%"}, title={'text':"Overall Performance"},
                  gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#10b981"}}))
        fig.update_layout(height=220, margin=dict(l=10,r=10,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig2 = go.Figure(go.Indicator(mode="gauge+number", value=overall_pb*100.0,
                  number={'suffix': "%"}, title={'text':"Live P(Grade = 1)"},
                  gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#6366f1"}}))
        fig2.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # Per‑course predicted exam score quick cards (with DE grade)
    ca, cb = st.columns(2)
    with ca:
        render_predicted_exam_card(COURSE_LABELS["ethics"], course_color(COURSE_LABELS["ethics"]))
    with cb:
        render_predicted_exam_card(COURSE_LABELS["hcml"], course_color(COURSE_LABELS["hcml"]))

    rows=[]
    for c in course_list:
        x=per_course[c]
        pred = compute_predicted_exam_score(c, lectures, scores, sessions)
        rows.append({"Course":c,
                    "Study Coverage (%)":round(x["study_coverage"]*100,1),
                    "Knowledge Progress (%)":round(x["knowledge_progress"]*100,1),
                    "Avg Score":round(x["avg_score"],1),
                    "Predicted Exam Score (%)": pred["predicted_score"],
                    "Predicted Grade (DE)": round(pred["predicted_de_grade"],1),
                    "Confidence (%)": int(round(pred["confidence"]*100)),
                    "Performance":round(x["performance"],1),
                    "Lectures Remaining":max(0,x["total_lectures"]-x["studied_lectures"]),
                    "P(Grade 1) %":round(x["grade1_prob"]*100,1),
                    "Effort (min)":int(x["effort_minutes"])})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Score trend (if any)
    if not scores.empty and {"Date","Score"}.issubset(scores.columns):
        s = scores.dropna(subset=["Date","Score"]).copy()
        if not s.empty:
            s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
            fig = px.line(s.sort_values("Date"), x="Date", y="Score", color="Course", markers=True,
                          title="Score Trend (All Assessments)")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Study minutes per week
    if not sessions.empty and {"Date","Minutes"}.issubset(sessions.columns):
        x = sessions.dropna(subset=["Date","Minutes"]).copy()
        if not x.empty:
            x["week"] = pd.to_datetime(x["Date"]).dt.to_period("W").dt.start_time
            w = x.groupby(["week","Course"], as_index=False)["Minutes"].sum()
            fig = px.bar(w, x="week", y="Minutes", color="Course", barmode="group",
                         title="Study Minutes per Week")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

# COURSE DRILLDOWNS
with tabs[1]:
    for c in course_list:
        st.markdown(f"#### {c}  <span class='badge'>Exam prep</span>", unsafe_allow_html=True)
        x = per_course[c]
        a,b,d,e,f = st.columns([1,1,1,1,1])
        with a: kpi_card("Lectures Remaining", f"{max(0,x['total_lectures']-x['studied_lectures'])}", f"Total: {x['total_lectures']}", "📚")
        with b: kpi_card("Study Coverage", f"{x['study_coverage']*100:.1f}%", "", "🧭")
        with d: kpi_card("Knowledge Progress", f"{x['knowledge_progress']*100:.1f}%", "", "🧠")
        with e:
            g_de = percent_to_german_grade(x["avg_score"]) if x["avg_score"]>0 else 5.0
            kpi_card("Avg Score", f"{x['avg_score']:.1f}", f"≈ Grade (DE): {g_de:.1f}", "🎯")
        with f: kpi_card("P(Grade 1)", f"{x['grade1_prob']*100:.1f}%", f"Effort: ~{int(x['effort_minutes'])} min", "🏆")

        # Predicted score card within each course (with DE grade)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        render_predicted_exam_card(c, course_color(c))

        w1,w2,w3 = st.columns(3)
        cov_boost = w1.slider(f"{c} — Boost coverage (%)", 0, 40, 10, 5)
        know_boost= w2.slider(f"{c} — Boost knowledge (%)", 0, 40, 10, 5)
        score_boost=w3.slider(f"{c} — Boost avg score (+)", 0, 20, 5, 1)

        new_cov  = min(1.0, x["study_coverage"] + cov_boost/100.0)
        new_kn   = min(1.0, x["knowledge_progress"] + know_boost/100.0)
        new_sc   = min(100.0, x["avg_score"] + score_boost)
        z = (2.6*(new_sc/100.0-0.85) + 2.0*(new_kn-0.80) + 2.0*(new_cov-0.90))
        p_new = 1/(1+math.exp(-z))
        st.info(f"**What‑If P(Grade 1):** {p_new*100:.1f}% → (Coverage {new_cov*100:.1f}%, Knowledge {new_kn*100:.1f}%, Avg Score {new_sc:.1f})")

        s_df = scores[scores["Course"]==c].dropna(subset=["Date","Score"]) if not scores.empty else pd.DataFrame()
        if not s_df.empty:
            s_df = s_df.sort_values("Date")
            fig = px.scatter(s_df, x="Date", y="Score", color="Type", trendline="lowess", title=f"{c} — Score Trend")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.write("---")

# LECTURES
with tabs[2]:
    st.subheader("Lectures")
    if lectures.empty:
        st.info("No lecture rows available. Check your Lecture/Lectures_Ethics/Lectures_HCML tabs.")
    else:
        show = lectures.copy()
        if "FilePath" in show.columns:
            def linkit(row):
                url = str(row["FilePath"])
                title = str(row.get("LectureName","View"))
                return f"[{title}]({url})" if url and url != "nan" else ""
            show["File"] = show.apply(linkit, axis=1)
        cols = [c for c in ["Course","LectureID","LectureName","Status","StudyMinutes","ReviewCount","Mastery","LastReviewDate","File"] if c in show.columns]
        st.dataframe(show[cols].sort_values(["Course","LectureID","LectureName"], na_position="last"),
                     use_container_width=True, hide_index=True)

# SCORES
with tabs[3]:
    st.subheader("Scores")
    if scores.empty:
        st.info("No scores yet. Fill the Quizzes tab.")
    else:
        cols = [c for c in ["Course","Date","Type","Score","LectureID"] if c in scores.columns]
        sdf = scores[cols].copy().sort_values("Date")
        # Avg grade per course (DE)
        if not sdf.empty:
            tmp = sdf.groupby("Course", as_index=False)["Score"].mean()
            tmp["Avg Grade (DE)"] = tmp["Score"].apply(percent_to_german_grade)
            st.dataframe(tmp.rename(columns={"Score":"Avg Score"}), use_container_width=True, hide_index=True)
        st.dataframe(sdf, use_container_width=True, hide_index=True)
        fig = px.histogram(sdf, x="Score", color="Course", nbins=20, title="Score Distribution")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

# STUDY SESSIONS
with tabs[4]:
    st.subheader("Study Sessions (TimeLog)")
    if sessions.empty:
        st.info("No sessions yet. Fill the TimeLog tab.")
    else:
        cols = [c for c in ["Course","Date","Minutes","Type","LectureID"] if c in sessions.columns]
        sdf = sessions[cols].copy().sort_values("Date", ascending=False)
        st.dataframe(sdf, use_container_width=True, hide_index=True)
        if "Date" in sdf.columns:
            fig = px.bar(sdf.sort_values("Date"), x="Date", y="Minutes", color="Course",
                         hover_data=["Type","LectureID"], title="Study Timeline (Minutes)")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

# FILES (raw index)
with tabs[5]:
    st.subheader("Files Index (raw)")
    if df_files.empty:
        st.info("Files_Index is empty.")
    else:
        st.dataframe(df_files, use_container_width=True, hide_index=True)

# SCHEMA
with tabs[6]:
    st.subheader("Flexible Expected Columns")
    st.markdown("""
**Lecture / Lectures_Ethics / Lectures_HCML** (any of):  
- `Course | Session` (optional in per-course tabs; filled automatically)
- `Lecture_ID | LectureID | ID | LecID | Folder Name`
- `Title | LectureName | Name | File Name`
- `Status | Studied?`
- `StudyMinutes | Study_Min | Minutes | Total_Minutes | Duration`
- `ReviewCount | Reviews`
- `Last_Review_Date | LastReviewDate | Last_Review`
- `Mastery_0_5 | Mastery | Knowledge_%` (auto-normalized to 0–1)

**Quizzes**: `Date`, `Course|Session`, `Lecture_ID|LectureID|LecID|Folder Name`, `Type`, `Score_%|Score` (auto-normalized to 0–100)

**TimeLog**: `Date`, `Course|Session`, `Lecture_ID|LectureID|LecID|Folder Name`, `Minutes`, (`Type` optional)

**Files_Index**: `Lecture_ID|LectureID|Folder Name`, `Title|LectureName|File Name`, `Path|FilePath|URL|Link|File URL`
""")
