
# app.py
# Streamlit app with Marketing → Referral Pi
# - Focus: Referral Intent Source = "Sales Generated"
# - Referred-by-enrolled-parent: ref-email == parent-email AND parent has any enrollment
# - Deal-relative window: Δdays = (Deal Create Date − Parent FIRST Enrollment Date)
#   A: 0 ≤ Δdays ≤ N;  B: otherwise (Δ<0, Δ>N, or no enrollment)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import os

st.set_page_config(page_title="Referral Pi", layout="wide")

# ---------- Utilities ----------
def find_col(df: pd.DataFrame, candidates):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    norm = {c.lower().strip(): c for c in cols}
    for c in candidates:
        k = c.lower().strip()
        if k in norm:
            return norm[k]
    # fallback: fuzzy contains
    for c in candidates:
        for col in cols:
            if c.lower().strip() in col.lower().strip():
                return col
    return None

def to_day_ts(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(pd.NaT, index=[])
    try:
        dt = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    except Exception:
        dt = pd.to_datetime(series, errors="coerce")
    # drop timezone if present
    try:
        if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
    return dt.dt.floor("D")

def norm_email(x):
    if pd.isna(x):
        return ""
    try:
        return str(x).strip().lower()
    except Exception:
        return ""

def month_bounds(d: date):
    from calendar import monthrange
    return date(d.year, d.month, 1), date(d.year, d.month, monthrange(d.year, d.month)[1])

def last_month_bounds(d: date):
    first_this = date(d.year, d.month, 1)
    last_prev = first_this - timedelta(days=1)
    return month_bounds(last_prev)

# ---------- Data load ----------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload Master_sheet-DB.csv", type=["csv"])
default_path = "/mnt/data/Master_sheet-DB.csv"

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif os.path.exists(default_path):
    df = pd.read_csv(default_path)
else:
    st.warning("Please upload your Master_sheet-DB.csv to proceed.")
    st.stop()

if df.empty:
    st.info("The uploaded file is empty.")
    st.stop()

# ---------- Sidebar: Navigation ----------
st.sidebar.header("Navigation")
master = st.sidebar.selectbox("Section", ["Marketing"], index=0)
pill = st.sidebar.selectbox("Pill", ["Referral Pi"], index=0)

# ---------- Shared column resolution ----------
CREATE_COL = find_col(df, ["Create Date","Created Date","Deal Create Date","Created On","Create_Date"])
PAY_COL    = find_col(df, ["Payment Received Date","Payment Date","Enrolment Date","Payment_Received_Date","Paid On"])
REF_INTENT = find_col(df, ["Referral Intent Source","Referral_Intent_Source","Referral Intent source","Referral intent source"])
REF_EMAIL  = find_col(df, ["Deal referred by(Email):","Deal referred by (Email)","Deal referred by Email","Deal referred by(Email)","Referred By Email","Referral Email","Referrer Email"])
PARENT_EML = find_col(df, ["Parent Email","Parent email","Email","Parent_Email"])

missing = []
if CREATE_COL is None: missing.append("Create Date")
if PAY_COL is None:    missing.append("Payment/Enrollment Date")
if REF_INTENT is None: missing.append("Referral Intent Source")
if REF_EMAIL is None:  missing.append("Deal referred by (Email)")
if PARENT_EML is None: missing.append("Parent Email")

if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

# ---------- Marketing → Referral Pi ----------
if master == "Marketing" and pill == "Referral Pi":
    st.subheader("Marketing — Referral Pi")

    # Counting mode + date presets
    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True)
    today = date.today()
    presets = ["Today","Yesterday","This Month","Last Month","Custom"]
    preset = st.radio("Date range", presets, index=2, horizontal=True)

    if preset == "Today":
        start_d, end_d = today, today
    elif preset == "Yesterday":
        start_d = today - timedelta(days=1); end_d = start_d
    elif preset == "This Month":
        start_d, end_d = month_bounds(today)
    elif preset == "Last Month":
        start_d, end_d = last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start", value=month_bounds(today)[0])
        with c2: end_d   = st.date_input("End", value=today)
        if end_d < start_d:
            start_d, end_d = end_d, start_d

    start_ts = pd.Timestamp(start_d).normalize().tz_localize(None)
    end_ts   = pd.Timestamp(end_d).normalize().tz_localize(None)

    # Base filter: Referral Intent Source = "Sales Generated"
    is_sales_gen = df[REF_INTENT].astype(str).str.strip().str.lower() == "sales generated"
    sales_df = df.loc[is_sales_gen].copy()

    # Total 'Sales Generated' in window (by deal create)
    create_dt_all = to_day_ts(sales_df[CREATE_COL]) if CREATE_COL in sales_df.columns else pd.Series(pd.NaT, index=sales_df.index)
    in_window_all = create_dt_all.between(start_ts, end_ts)
    total_sales_gen_in_window = int(in_window_all.sum())

    # Referred-by-enrolled-parent: ref==parent AND parent has any enrollment (lifetime)
    ref_norm_all    = sales_df[REF_EMAIL].map(norm_email)
    parent_norm_all = sales_df[PARENT_EML].map(norm_email)
    same_email_all  = (ref_norm_all != "") & (ref_norm_all == parent_norm_all)

    # Build first enrollment map from full df (lifetime)
    pay_dt_full = to_day_ts(df[PAY_COL]) if PAY_COL in df.columns else pd.Series(pd.NaT, index=df.index)
    parent_norm_full = df[PARENT_EML].map(norm_email)
    first_pay_any = (
        df.assign(_parent_norm=parent_norm_full, _pay_dt=pay_dt_full)
          .loc[lambda d: (d["_parent_norm"] != "") & d["_pay_dt"].notna()]
          .groupby("_parent_norm")["_pay_dt"].min()
    )

    parent_has_enrollment = parent_norm_all.map(lambda e: pd.notna(first_pay_any.get(e, pd.NaT)))
    base_mask = same_email_all & in_window_all & parent_has_enrollment
    referred_by_enrolled_in_window = int(base_mask.sum())

    # Subset for deal-relative split
    d0 = sales_df.loc[base_mask].copy()
    if d0.empty:
        k1, k2 = st.columns(2)
        with k1: st.metric("Total 'Sales Generated' (in window)", total_sales_gen_in_window)
        with k2: st.metric("Referred by enrolled parent (in window)", 0)
        st.info("No matching rows (Sales Generated + referrer == parent + enrolled).")
        st.stop()

    create_dt = create_dt_all.loc[base_mask]
    parent_first_enroll = d0[PARENT_EML].map(lambda e: first_pay_any.get(str(e).strip().lower(), pd.NaT))

    # Deal-relative window: Δdays = create − first_enroll
    N = st.number_input("Window N days relative to deal date", min_value=1, max_value=365, value=45, step=1)
    delta_days = (create_dt - parent_first_enroll).dt.days

    within_mask = delta_days.between(0, int(N))     # A: 0 ≤ Δ ≤ N
    beyond_mask = ~within_mask                      # B: Δ<0, Δ>N, or no enrollment (NaN)

    A = int(within_mask.sum())
    B = int(beyond_mask.sum())

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total 'Sales Generated' deals (in window)", total_sales_gen_in_window)
    with k2: st.metric("Referred by enrolled parent (in window)", referred_by_enrolled_in_window)
    with k3: st.metric(f"A: Within ≤{int(N)}d of deal", A)
    with k4: st.metric(f"B: Beyond {int(N)}d of deal", B)

    if (A + B) != referred_by_enrolled_in_window:
        st.warning(f"A({A}) + B({B}) != Referred-by-enrolled count ({referred_by_enrolled_in_window}). Please review date columns.")

    # Details table
    bucket_choice = st.radio("Show rows for:", [f"A: Within ≤{int(N)}d", f"B: Beyond {int(N)}d", "Both"], index=2, horizontal=True)
    if bucket_choice.startswith("A:"):
        row_mask = within_mask
    elif bucket_choice.startswith("B:"):
        row_mask = beyond_mask
    else:
        row_mask = within_mask | beyond_mask

    with st.expander("Show rows"):
        cols = []
        for c in [REF_INTENT, REF_EMAIL, PARENT_EML, CREATE_COL, "Deal Name","Record ID",
                  "Student/Academic Counsellor","Country","JetLearn Deal Source","Original Source",
                  "Payment Received Date","Payment Date","Enrolment Date","Paid On"]:
            if c in d0.columns and c not in cols:
                cols.append(c)
        detail = d0.loc[row_mask, cols].copy()
        detail["First Parent Enrollment Date (lifetime)"] = parent_first_enroll.loc[detail.index]
        detail["Δdays (deal_create − first_enroll)"] = delta_days.loc[detail.index]
        detail["Bucket"] = np.where(within_mask.loc[detail.index], f"A: ≤{int(N)}d", f"B: >{int(N)}d / after / none")
        st.dataframe(detail, use_container_width=True)
        st.download_button(
            "Download CSV — Referral Pi rows",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name="marketing_referral_pi_rows.csv",
            mime="text/csv"
        )
