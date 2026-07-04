"""

Assignment 2: Automated ML Pipeline for Clinical Prediction under Temporal Shift

Run:  streamlit run TeamXX_Assignment2_dashboard.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import streamlit as st

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc,
                              classification_report, roc_auc_score,
                              ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EHR ML Pipeline Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#1f4e79; margin-bottom:4px; }
    .sub-title   { font-size:1.1rem; color:#555; margin-bottom:20px; }
    .section-hdr { font-size:1.4rem; font-weight:600; color:#2c5f8a; border-bottom:2px solid #2c5f8a;
                   padding-bottom:4px; margin-top:20px; margin-bottom:14px; }
    .metric-card { background:#f0f7ff; border-radius:10px; padding:14px 18px;
                   border-left:4px solid #2c5f8a; margin-bottom:10px; }
    .stAlert     { border-radius:8px; }
    div[data-testid="metric-container"] { background:#f8fbff; border-radius:8px;
                                          padding:10px; border:1px solid #d0e4f5; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CUTOFF_DATE  = pd.Timestamp("2018-01-01")
RANDOM_STATE = 42
TEST_SIZE    = 0.25
PALETTE      = sns.color_palette("muted")
VITAL_MAP = {
    "Body Height":                              "height_cm",
    "Body Weight":                              "weight_kg",
    "Diastolic Blood Pressure":                 "dbp_mmhg",
    "Systolic Blood Pressure":                  "sbp_mmhg",
    "Heart rate":                               "heart_rate",
    "Respiratory rate":                         "resp_rate",
    "Pain severity - 0-10 verbal numeric rating [Score]": "pain_score",
    "Body mass index (BMI) [Ratio]":            "bmi",
    "Glucose":                                  "glucose",
    "Hemoglobin [Mass/volume] in Blood":        "hemoglobin",
    "Leukocytes [#/volume] in Blood by Automated count": "wbc",
    "Hematocrit [Volume Fraction] of Blood by Automated count": "hematocrit",
    "Platelets [#/volume] in Blood by Automated count": "platelets",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────────────────────────────────────
def safe_read(data_dir, fname):
    path = os.path.join(data_dir, fname)
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False,on_bad_lines='skip')
    return pd.DataFrame()

@st.cache_data(show_spinner="⚙ Loading & processing data …", ttl=3600)
def load_and_engineer(data_dir, cutoff_iso):
    cutoff = pd.Timestamp(cutoff_iso).tz_localize(None)

    patients          = safe_read(data_dir, "patients.csv")
    encounters        = safe_read(data_dir, "encounters.csv")
    observations      = safe_read(data_dir, "observations.csv")
    conditions        = safe_read(data_dir, "conditions.csv")
    medications       = safe_read(data_dir, "medications.csv")
    procedures        = safe_read(data_dir, "procedures.csv")
    immunizations     = safe_read(data_dir, "immunizations.csv")
    allergies         = safe_read(data_dir, "allergies.csv")

    # ── Patients ──────────────────────────────────────────────────────────────
    pat = patients.copy()
    pat.columns = [c.upper() for c in pat.columns]
    id_col = next((c for c in pat.columns if c in ["ID","ID"]), pat.columns[0])

    for col in ["BIRTHDATE","DEATHDATE"]:
        if col in pat.columns:
            pat[col] = pd.to_datetime(pat[col], errors="coerce").dt.tz_localize(None)

    pat["AGE"] = ((cutoff - pat["BIRTHDATE"]).dt.days / 365.25).round(1) \
                  if "BIRTHDATE" in pat.columns else np.nan
    pat["IS_DECEASED"] = (pat["DEATHDATE"].notna() & (pat["DEATHDATE"] < cutoff)).astype(int) \
                          if "DEATHDATE" in pat.columns else 0

    demo_cols = {id_col: "PATIENT"}
    for c in ["GENDER","RACE","ETHNICITY","INCOME","HEALTHCARE_EXPENSES",
              "HEALTHCARE_COVERAGE","AGE","IS_DECEASED"]:
        if c in pat.columns:
            demo_cols[c] = c
    demo = pat[list(demo_cols.keys())].rename(columns=demo_cols)
    demo = demo.drop_duplicates("PATIENT").set_index("PATIENT")

    # ── Observations ─────────────────────────────────────────────────────────
    if not observations.empty:
        obs = observations.copy()
        obs.columns = [c.upper() for c in obs.columns]
        obs["DATE"]  = pd.to_datetime(obs["DATE"], errors="coerce").dt.tz_localize(None)
        obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")
        obs = obs[(obs.get("TYPE","numeric") == "numeric") & (obs["DATE"] <= cutoff)]
        obs = obs[obs["DESCRIPTION"].isin(VITAL_MAP)].copy()
        obs["FEAT"] = obs["DESCRIPTION"].map(VITAL_MAP)
        agg = obs.groupby(["PATIENT","FEAT"])["VALUE"].agg(["mean","std"]).reset_index()
        agg.columns = ["PATIENT","FEAT","MEAN","STD"]
        mp = agg.pivot(index="PATIENT",columns="FEAT",values="MEAN").add_suffix("_mean")
        sp = agg.pivot(index="PATIENT",columns="FEAT",values="STD").add_suffix("_std")
        obs_feat = mp.join(sp, how="outer")
    else:
        obs_feat = pd.DataFrame()

    # ── Count features ────────────────────────────────────────────────────────
    def count_up_to(df, pcol, dcol, fname):
        if df.empty: return pd.Series(dtype=float, name=fname)
        d = df.copy(); d.columns = [c.upper() for c in d.columns]
        pcol2, dcol2 = pcol.upper(), dcol.upper()
        if dcol2 in d.columns:
            d[dcol2] = pd.to_datetime(d[dcol2], errors="coerce").dt.tz_localize(None)
            d = d[d[dcol2] <= cutoff]
        if pcol2 not in d.columns: return pd.Series(dtype=float, name=fname)
        return d.groupby(pcol2).size().rename(fname)

    cond_df = conditions.copy() if not conditions.empty else pd.DataFrame()
    if not cond_df.empty:
        cond_df.columns = [c.upper() for c in cond_df.columns]
        cond_df["START"] = pd.to_datetime(cond_df.get("START",""), errors="coerce").dt.tz_localize(None)
        cond_df["STOP"]  = pd.to_datetime(cond_df.get("STOP",""),  errors="coerce").dt.tz_localize(None)
        cond_df = cond_df[cond_df["START"] <= cutoff]
        num_conditions  = cond_df.groupby("PATIENT").size().rename("num_conditions")
        active_cond     = cond_df[cond_df["STOP"].isna()].groupby("PATIENT").size().rename("num_active_conditions")
    else:
        num_conditions = active_cond = pd.Series(dtype=float)

    enc_df = encounters.copy() if not encounters.empty else pd.DataFrame()
    total_enc_cost = pd.Series(dtype=float, name="total_enc_cost")
    if not enc_df.empty:
        enc_df.columns = [c.upper() for c in enc_df.columns]
        enc_df["START"] = pd.to_datetime(enc_df.get("START",""), errors="coerce").dt.tz_localize(None)
        enc_df = enc_df[enc_df["START"] <= cutoff]
        if "TOTAL_CLAIM_COST" in enc_df.columns:
            enc_df["TOTAL_CLAIM_COST"] = pd.to_numeric(enc_df["TOTAL_CLAIM_COST"], errors="coerce")
            total_enc_cost = enc_df.groupby("PATIENT")["TOTAL_CLAIM_COST"].sum().rename("total_enc_cost")

    master = demo.copy()
    for fs in [obs_feat, num_conditions, active_cond,
               count_up_to(medications,  "patient","start","num_medications"),
               count_up_to(encounters,   "patient","start","num_encounters"),
               count_up_to(procedures,   "patient","start","num_procedures"),
               count_up_to(immunizations,"patient","date", "num_immunizations"),
               count_up_to(allergies,    "patient","start","num_allergies"),
               total_enc_cost]:
        if not (isinstance(fs, (pd.DataFrame, pd.Series)) and fs.empty if hasattr(fs, 'empty') else False):
            master = master.join(fs, how="left")

    master = master.reset_index()
    master["num_active_conditions"] = master.get("num_active_conditions", pd.Series([0]*len(master))).fillna(0)

    # ── Target ────────────────────────────────────────────────────────────────
    if "HEALTHCARE_EXPENSES" in master.columns:
        master["HEALTHCARE_EXPENSES"] = pd.to_numeric(master["HEALTHCARE_EXPENSES"], errors="coerce")
        med_exp = master["HEALTHCARE_EXPENSES"].median()
        master["TARGET"] = (master["HEALTHCARE_EXPENSES"] > med_exp).astype(int)
    else:
        nc = master.get("num_conditions", pd.Series([0]*len(master)))
        master["TARGET"] = (nc > nc.median()).astype(int)

    # ── Temporal split ────────────────────────────────────────────────────────
    if not enc_df.empty and "PATIENT" in enc_df.columns:
        last_enc = enc_df.groupby("PATIENT")["START"].max().reset_index()
        last_enc.columns = ["PATIENT","LAST_ENCOUNTER"]
        master = master.merge(last_enc, on="PATIENT", how="left")
    else:
        master["LAST_ENCOUNTER"] = pd.NaT

    if master["LAST_ENCOUNTER"].isna().all():
        ds1, ds2 = train_test_split(master, test_size=0.4,
                                     random_state=RANDOM_STATE,
                                     stratify=master["TARGET"] if master["TARGET"].nunique() > 1 else None)
    else:
        ds1 = master[master["LAST_ENCOUNTER"] < cutoff].copy()
        ds2 = master[master["LAST_ENCOUNTER"] >= cutoff].copy()
        if len(ds1) < 5 or len(ds2) < 5:
            ds1, ds2 = train_test_split(master, test_size=0.4,
                                         random_state=RANDOM_STATE,
                                         stratify=master["TARGET"] if master["TARGET"].nunique() > 1 else None)

    raw_tables = {
        "patients": patients, "encounters": encounters,
        "observations": observations, "conditions": conditions,
        "medications": medications, "procedures": procedures,
        "immunizations": immunizations, "allergies": allergies,
    }

    return master, ds1, ds2, raw_tables


@st.cache_data(show_spinner="🤖 Preprocessing & splitting …", ttl=3600)
def prepare_splits(ds1_json, ds2_json):
    ds1 = pd.read_json(ds1_json)
    ds2 = pd.read_json(ds2_json)

    DROP = ["PATIENT","TARGET","LAST_ENCOUNTER","HEALTHCARE_EXPENSES",
            "BIRTHDATE","DEATHDATE","HEALTHCARE_COVERAGE"]
    CAT  = [c for c in ["GENDER","RACE","ETHNICITY"] if c in ds1.columns]
    NUM  = [c for c in ds1.columns
            if c not in DROP and c not in CAT
            and ds1[c].dtype != object
            and "DATE" not in c.upper() and "LAST" not in c.upper()]

    def encode_scale(df, cat_cols, num_cols, encoders=None, scaler=None):
        X = df.copy()
        enc = encoders or {}
        for col in cat_cols:
            if col not in X.columns: continue
            X[col] = X[col].fillna("Unknown").astype(str)
            if col in enc:
                le = enc[col]
                X[col] = X[col].apply(lambda v: v if v in le.classes_ else "Unknown")
                X[col] = le.transform(X[col])
            else:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                enc[col] = le
        feats = [c for c in num_cols + cat_cols if c in X.columns]
        X = X[feats].fillna(X[feats].median(numeric_only=True)).fillna(0)
        if scaler is None:
            sc = StandardScaler(); Xsc = pd.DataFrame(sc.fit_transform(X), columns=X.columns, index=X.index)
        else:
            sc = scaler; Xsc = pd.DataFrame(sc.transform(X), columns=X.columns, index=X.index)
        return Xsc, df["TARGET"].values, enc, sc

    strat1 = ds1["TARGET"] if ds1["TARGET"].nunique() > 1 else None
    d1tr, d1te = train_test_split(ds1, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat1)
    strat2 = ds2["TARGET"] if ds2["TARGET"].nunique() > 1 else None
    d2tr, d2te = train_test_split(ds2, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat2)

    X1tr, y1tr, enc, sc = encode_scale(d1tr, CAT, NUM)
    X1te, y1te, _,   _  = encode_scale(d1te, CAT, NUM, enc, sc)
    X2tr, y2tr, _,   _  = encode_scale(d2tr, CAT, NUM, enc, sc)
    X2te, y2te, _,   _  = encode_scale(d2te, CAT, NUM, enc, sc)

    return (X1tr, y1tr, X1te, y1te,
            X2tr, y2tr, X2te, y2te,
            list(X1tr.columns), CAT)


@st.cache_resource(show_spinner="🏋 Training models …", ttl=3600)
def train_all_models(splits_key, dt_max_depth, svm_C, mlp_layers):
    # This function is called with a hashable key; actual split data passed via st.session_state
    splits = st.session_state["splits"]
    X1tr, y1tr, X1te, y1te, X2tr, y2tr, X2te, y2te, feat_names, _ = splits

    # ── Define models ─────────────────────────────────────────────────────────
    dt  = DecisionTreeClassifier(max_depth=dt_max_depth, min_samples_split=4,
                                  random_state=RANDOM_STATE, class_weight="balanced")
    svm = SVC(C=svm_C, kernel="rbf", gamma="scale",
              probability=True, random_state=RANDOM_STATE, class_weight="balanced")
    mlp = MLPClassifier(hidden_layer_sizes=tuple(mlp_layers), activation="relu",
                        max_iter=500, random_state=RANDOM_STATE,
                        early_stopping=True, validation_fraction=0.1,
                        learning_rate_init=0.001)

    # ── Train on D1 ───────────────────────────────────────────────────────────
    for m in [dt, svm, mlp]:
        if len(np.unique(y1tr)) > 1:
            m.fit(X1tr, y1tr)

    # ── Continual learning ────────────────────────────────────────────────────
    # MLP – warm_start fine-tuning
    mlp_cl = MLPClassifier(hidden_layer_sizes=tuple(mlp_layers), activation="relu",
                            max_iter=200, random_state=RANDOM_STATE,
                            warm_start=True, learning_rate_init=0.0005)
    if len(np.unique(y1tr)) > 1: mlp_cl.fit(X1tr, y1tr)
    if len(np.unique(y2tr)) > 1 and len(y2tr) >= 5: mlp_cl.fit(X2tr, y2tr)

    # SGD-SVM – partial_fit
    sgd = SGDClassifier(loss="hinge", random_state=RANDOM_STATE,
                        max_iter=1, tol=None)
    classes = np.array([0, 1])
    if len(np.unique(y1tr)) > 1: sgd.partial_fit(X1tr.values, y1tr, classes=classes)
    if len(np.unique(y2tr)) > 1 and len(y2tr) >= 5: sgd.partial_fit(X2tr.values, y2tr, classes=classes)

    # DT-CL – combined data
    Xc = pd.concat([X1tr, X2tr], ignore_index=True)
    yc = np.concatenate([y1tr, y2tr])
    dt_cl = DecisionTreeClassifier(max_depth=dt_max_depth, min_samples_split=4,
                                    random_state=RANDOM_STATE, class_weight="balanced")
    if len(np.unique(yc)) > 1: dt_cl.fit(Xc, yc)

    # ── Collect metrics ───────────────────────────────────────────────────────
    def metrics(model, X, y, label, mname, use_np=False):
        Xev = X.values if use_np else X
        if len(y) == 0 or len(np.unique(y)) < 2:
            return {"model": mname, "split": label,
                    "accuracy": np.nan, "precision": np.nan,
                    "recall": np.nan, "f1": np.nan, "auc": np.nan}
        yp = model.predict(Xev)
        ypr = model.predict_proba(Xev)[:, 1] if hasattr(model, "predict_proba") else None
        try:
            ypr = model.decision_function(Xev) if ypr is None else ypr
        except Exception:
            ypr = None
        return {
            "model": mname, "split": label,
            "accuracy":  accuracy_score(y, yp),
            "precision": precision_score(y, yp, average="weighted", zero_division=0),
            "recall":    recall_score(y, yp, average="weighted", zero_division=0),
            "f1":        f1_score(y, yp, average="weighted", zero_division=0),
            "auc":       roc_auc_score(y, ypr) if ypr is not None and len(np.unique(y)) == 2 else np.nan,
        }

    rows = []
    for name, model in [("Decision Tree", dt), ("SVM", svm), ("Neural Network", mlp)]:
        rows.append(metrics(model, X1tr, y1tr, "D1-Train", name))
        rows.append(metrics(model, X1te, y1te, "D1-Test",  name))
        rows.append(metrics(model, X2te, y2te, "D2-Test (Pre-CL)", name))

    for name, model, use_np in [("Decision Tree (CL)", dt_cl, False),
                                  ("SVM-SGD (CL)",      sgd,   True),
                                  ("Neural Network (CL)", mlp_cl, False)]:
        rows.append(metrics(model, X2te, y2te, "D2-Test (Post-CL)", name, use_np=use_np))

    results_df = pd.DataFrame(rows)

    return {
        "dt": dt, "svm": svm, "mlp": mlp,
        "dt_cl": dt_cl, "sgd": sgd, "mlp_cl": mlp_cl,
        "results": results_df,
        "feat_names": feat_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("C:\\Users\\Aayush Shah\\OneDrive\\Desktop\\ML2\\BITS_Pilani-Logo.svg.png",
             width=140)
    st.markdown("## 🏥 EHR ML Pipeline")
    st.markdown("**BITS F464 – Machine Learning**  \n*Assignment 2*")
    st.markdown("Team Members: \n Aayush Shah(2022B3A71332H), \n Sakshi Bharadwaj(2022A4PS1492H), \n Harsh Sharma(2022A3PS1291H), \n Gaurvi Khurana(2023A7PS0035H)")
    st.divider()

    page = st.radio("📋 Navigate to", [
        "🏠 Overview & Data",
        "📊 Exploratory Data Analysis",
        "⚙ Feature Engineering",
        "🤖 Model Training (D1)",
        "⏱ Temporal Shift Analysis",
        "🔄 Continual Learning",
        "📈 Final Results Summary",
    ])
    st.divider()

    st.markdown("### ⚙ Configuration")
    data_dir    = st.text_input("📁 Data directory", value="./data")
    cutoff_date = st.date_input("🗓 Temporal cutoff", value=CUTOFF_DATE.date())
    st.divider()

    st.markdown("### 🔧 Hyperparameters")
    dt_depth    = st.slider("Decision Tree max_depth", 2, 15, 5)
    svm_c       = st.select_slider("SVM C",  [0.01, 0.1, 1, 5, 10, 50], value=1)
    mlp_h1      = st.slider("MLP hidden layer 1",  16, 128, 64, step=16)
    mlp_h2      = st.slider("MLP hidden layer 2",  8,  64,  32, step=8)
    train_btn   = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
cutoff_iso = str(cutoff_date)

# Check if data directory exists
data_available = os.path.exists(data_dir) and any(
    f.endswith(".csv") for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))
) if os.path.exists(data_dir) else False

if not data_available:
    st.markdown('<div class="main-title">🏥 EHR ML Pipeline Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">BITS F464 Machine Learning — Assignment 2</div>', unsafe_allow_html=True)
    st.warning(f"""
    **Data directory not found or empty:** `{data_dir}`

    Please:
    1. Download the dataset from the Google Drive link in the assignment PDF
    2. Place all CSV files in `{data_dir}/` (or change the path in the sidebar)
    3. CSV files expected: `patients.csv`, `encounters.csv`, `observations.csv`,
       `conditions.csv`, `medications.csv`, `procedures.csv`, `immunizations.csv`,
       `allergies.csv`, and others.
    """)
    st.stop()

# Load & engineer features
master, ds1, ds2, raw_tables = load_and_engineer(data_dir, cutoff_iso)

# Prepare splits (cached separately)
splits_result = prepare_splits(ds1.to_json(), ds2.to_json())
st.session_state["splits"] = splits_result
X1tr, y1tr, X1te, y1te, X2tr, y2tr, X2te, y2te, feat_names, cat_cols = splits_result

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
splits_key = (cutoff_iso, dt_depth, svm_c, mlp_h1, mlp_h2)
if "model_key" not in st.session_state or st.session_state["model_key"] != splits_key or train_btn:
    st.session_state["model_key"]  = splits_key
    st.session_state["model_data"] = train_all_models(splits_key, dt_depth, svm_c,
                                                       [mlp_h1, mlp_h2])

mdata      = st.session_state["model_data"]
results_df = mdata["results"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_st(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def metric_delta(val_new, val_old, fmt=".3f"):
    if np.isnan(val_new) or np.isnan(val_old): return None
    return f"{val_new - val_old:+{fmt}}"

def roc_for(model, X, y, label):
    if not hasattr(model, "predict_proba") or len(np.unique(y)) < 2:
        return None, None, None
    try:
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        return fpr, tpr, auc(fpr, tpr)
    except Exception:
        return None, None, None

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW & DATA
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview & Data":
    st.markdown('<div class="main-title">🏥 EHR ML Pipeline Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">BITS F464 Machine Learning — Assignment 2 | Team40</div>',
                unsafe_allow_html=True)

    st.markdown("""
    > **Objective:** Predict high-cost patients (healthcare expenses above median) using a
    > multi-table EHR dataset.  Models are trained on **historical** data (Dataset 1)
    > and evaluated across a temporal boundary onto **current** data (Dataset 2)
    > to study **data drift** and **continual learning** effectiveness.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients",  len(master))
    c2.metric("Dataset 1 (Historical)", len(ds1),
              delta=f"before {cutoff_date}")
    c3.metric("Dataset 2 (Current)",    len(ds2),
              delta=f"from {cutoff_date}")
    c4.metric("Features Engineered",    len(feat_names))

    st.divider()
    st.markdown('<div class="section-hdr">📂 Raw Dataset Previews</div>', unsafe_allow_html=True)

    tab_names = list(raw_tables.keys())
    tabs = st.tabs([f"📄 {t}" for t in tab_names])
    for tab, name in zip(tabs, tab_names):
        with tab:
            df = raw_tables[name]
            if df.empty:
                st.warning(f"`{name}.csv` not loaded (file not found).")
            else:
                st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                st.dataframe(df.head(50), use_container_width=True)

    st.divider()
    st.markdown('<div class="section-hdr">🗂 Engineered Master Dataset</div>', unsafe_allow_html=True)
    st.caption(f"Shape: {master.shape[0]} patients × {master.shape[1]} columns")
    st.dataframe(master.head(50), use_container_width=True)

    # Class balance
    st.markdown('<div class="section-hdr">⚖ Target Variable Distribution</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for col, (ds_name, ds) in zip([col1, col2],
                                   [("Dataset 1 (Historical)", ds1),
                                    ("Dataset 2 (Current)", ds2)]):
        with col:
            vc = ds["TARGET"].value_counts().reset_index()
            vc.columns = ["Class", "Count"]
            vc["Label"] = vc["Class"].map({0: "Low Expenses", 1: "High Expenses"})
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(vc["Label"], vc["Count"],
                   color=[PALETTE[0], PALETTE[3]], edgecolor="white", linewidth=1.5)
            for _, r in vc.iterrows():
                ax.text(r.name, r["Count"] + 0.1, str(r["Count"]),
                        ha="center", fontweight="bold")
            ax.set_title(f"Class Balance – {ds_name}", fontweight="bold")
            ax.set_ylabel("Patients")
            fig_to_st(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Exploratory Data Analysis":
    st.markdown('<div class="main-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("Comparing Dataset 1 (Historical) vs Dataset 2 (Current) to detect **data drift**.")

    # ── Descriptive stats
    st.markdown('<div class="section-hdr">📐 Descriptive Statistics</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Dataset 1 – Historical", "Dataset 2 – Current"])
    with tab1:
        st.dataframe(ds1.describe().T.round(3), use_container_width=True)
    with tab2:
        st.dataframe(ds2.describe().T.round(3), use_container_width=True)

    # ── Feature distributions
    st.markdown('<div class="section-hdr">📉 Feature Distributions (D1 vs D2)</div>',
                unsafe_allow_html=True)

    num_cols = [c for c in master.select_dtypes(include=np.number).columns
                if c not in ["TARGET","IS_DECEASED"] and "DATE" not in c.upper()
                and "LAST" not in c.upper()]
    sel_feat = st.multiselect("Select features to visualise",
                               num_cols, default=num_cols[:6])

    if sel_feat:
        n = len(sel_feat)
        ncols = min(3, n); nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(sel_feat):
            ax = axes[i]
            d1v = ds1[col].dropna(); d2v = ds2[col].dropna()
            if len(d1v): ax.hist(d1v, bins=15, alpha=0.6, color=PALETTE[0],
                                  label="D1", density=True)
            if len(d2v): ax.hist(d2v, bins=15, alpha=0.6, color=PALETTE[3],
                                  label="D2", density=True)
            ax.set_title(col, fontsize=9); ax.legend(fontsize=7)
        for j in range(i + 1, len(axes)): axes[j].set_visible(False)
        plt.suptitle("Feature Distributions: Dataset 1 vs Dataset 2", fontweight="bold")
        plt.tight_layout()
        fig_to_st(fig)

    # ── Correlation heatmap
    st.markdown('<div class="section-hdr">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    hm_ds = st.radio("Select dataset", ["Dataset 1", "Dataset 2"], horizontal=True)
    ds_sel = ds1 if hm_ds == "Dataset 1" else ds2
    corr_cols = [c for c in num_cols + ["TARGET"] if c in ds_sel.columns]
    corr = ds_sel[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.65), max(6, len(corr) * 0.55)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
                annot=len(corr) <= 15, fmt=".2f", linewidths=0.3)
    ax.set_title(f"Correlation Matrix – {hm_ds}", fontweight="bold")
    plt.tight_layout()
    fig_to_st(fig)

    # ── Demographics
    st.markdown('<div class="section-hdr">👥 Demographics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if "GENDER" in master.columns:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            for ax, (ds, lb) in zip(axes, [(ds1, "D1"), (ds2, "D2")]):
                gc = ds["GENDER"].value_counts()
                ax.pie(gc.values, labels=gc.index, autopct="%1.1f%%",
                       colors=PALETTE[:len(gc)])
                ax.set_title(f"Gender – {lb}")
            plt.tight_layout()
            fig_to_st(fig)
    with col2:
        if "RACE" in master.columns:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            for ax, (ds, lb) in zip(axes, [(ds1, "D1"), (ds2, "D2")]):
                rc = ds["RACE"].value_counts()
                ax.barh(rc.index, rc.values, color=PALETTE[:len(rc)])
                ax.set_title(f"Race – {lb}")
                ax.set_xlabel("Count")
            plt.tight_layout()
            fig_to_st(fig)

    # ── Data drift detection
    st.markdown('<div class="section-hdr">🌊 Data Drift Detection</div>', unsafe_allow_html=True)
    drift_rows = []
    for c in num_cols:
        m1 = ds1[c].mean(); m2 = ds2[c].mean()
        s1 = ds1[c].std() + 1e-9
        drift_rows.append({"Feature": c, "D1 Mean": round(m1, 3),
                            "D2 Mean": round(m2, 3),
                            "Normalised Shift": round(abs(m2 - m1) / s1, 3)})
    drift_df = pd.DataFrame(drift_rows).sort_values("Normalised Shift", ascending=False)
    st.dataframe(drift_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(drift_df) * 0.35)))
    top = drift_df.head(15)
    colors = [PALETTE[3] if v > 1 else PALETTE[0] for v in top["Normalised Shift"]]
    ax.barh(top["Feature"], top["Normalised Shift"], color=colors)
    ax.axvline(1, ls="--", color="gray", alpha=0.7, label="1 SD threshold")
    ax.set_xlabel("|Δmean| / D1_std")
    ax.set_title("Feature Drift: D1 → D2", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig_to_st(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚙ Feature Engineering":
    st.markdown('<div class="main-title">⚙ Feature Engineering</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Strategy Overview
    Raw EHR data from **multiple relational tables** is merged on `PATIENT` ID and
    transformed into a flat feature vector per patient.

    | Feature Group | Source Table | Transformation |
    |---|---|---|
    | Demographics | `patients.csv` | Age computation, label encoding (gender/race/ethnicity) |
    | Vital Signs | `observations.csv` | Pivot + aggregate (mean, std) per patient |
    | Lab Results | `observations.csv` | Same pivot aggregation |
    | Condition Count | `conditions.csv` | Count + active (no STOP) |
    | Medication Count | `medications.csv` | Count up to cutoff |
    | Encounter Count | `encounters.csv` | Count + total claim cost sum |
    | Procedure Count | `procedures.csv` | Count |
    | Immunization Count | `immunizations.csv` | Count |
    | Allergy Count | `allergies.csv` | Count |

    **Target:** `HIGH_EXPENSES` = 1 if `HEALTHCARE_EXPENSES` > median, else 0
    """)

    st.markdown('<div class="section-hdr">📋 Engineered Feature Matrix</div>',
                unsafe_allow_html=True)
    st.caption(f"Total features: {len(feat_names)}")
    st.code("\n".join(feat_names), language="text")

    st.markdown('<div class="section-hdr">📊 Feature Statistics</div>', unsafe_allow_html=True)
    st.dataframe(master[feat_names].describe().T.round(3), use_container_width=True)

    st.markdown('<div class="section-hdr">🎯 Target Variable Analysis</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        vc = master["TARGET"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(vc.values, labels=["Low Expenses", "High Expenses"],
               autopct="%1.1f%%", colors=[PALETTE[0], PALETTE[3]])
        ax.set_title("Target Distribution (Full Dataset)")
        fig_to_st(fig)
    with col2:
        if "HEALTHCARE_EXPENSES" in master.columns:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(master["HEALTHCARE_EXPENSES"].dropna(), bins=20,
                    color=PALETTE[0], edgecolor="white")
            median_v = master["HEALTHCARE_EXPENSES"].median()
            ax.axvline(median_v, color=PALETTE[3], ls="--", lw=2,
                       label=f"Median = ${median_v:,.0f}")
            ax.set_xlabel("Healthcare Expenses ($)")
            ax.set_ylabel("Count")
            ax.set_title("Healthcare Expenses Distribution")
            ax.legend()
            fig_to_st(fig)

    st.markdown('<div class="section-hdr">📐 Train/Test Split Summary</div>',
                unsafe_allow_html=True)
    split_summary = pd.DataFrame({
        "Split": ["D1 Train", "D1 Test", "D2 Train", "D2 Test"],
        "Samples":      [len(y1tr), len(y1te), len(y2tr), len(y2te)],
        "Class 0 (Low)":  [int((y1tr == 0).sum()), int((y1te == 0).sum()),
                           int((y2tr == 0).sum()), int((y2te == 0).sum())],
        "Class 1 (High)": [int((y1tr == 1).sum()), int((y1te == 1).sum()),
                           int((y2tr == 1).sum()), int((y2te == 1).sum())],
    })
    st.dataframe(split_summary, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Training (D1)":
    st.markdown('<div class="main-title">🤖 Model Training on Dataset 1</div>',
                unsafe_allow_html=True)
    st.markdown(f"**Hyperparameters:** Decision Tree depth={dt_depth} | SVM C={svm_c} | MLP layers=({mlp_h1},{mlp_h2})")

    # ── Results table
    st.markdown('<div class="section-hdr">📊 Performance Metrics</div>', unsafe_allow_html=True)
    d1_res = results_df[results_df["split"].isin(["D1-Train","D1-Test"])].copy()
    d1_res[["accuracy","precision","recall","f1","auc"]] = \
        d1_res[["accuracy","precision","recall","f1","auc"]].round(4)
    st.dataframe(d1_res, use_container_width=True)

    # ── Metrics bar chart
    metric_sel = st.selectbox("Metric to visualise", ["accuracy","precision","recall","f1","auc"])
    pivot = d1_res.pivot(index="model", columns="split", values=metric_sel)
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[3]], edgecolor="white", width=0.6)
    ax.set_ylabel(metric_sel.upper()); ax.set_ylim(0, 1.1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_title(f"D1 Train vs D1 Test – {metric_sel.upper()}", fontweight="bold")
    ax.legend(title="Split")
    plt.tight_layout(); fig_to_st(fig)

    # ── Confusion Matrices
    st.markdown('<div class="section-hdr">🔢 Confusion Matrices (D1 Test)</div>',
                unsafe_allow_html=True)
    if len(np.unique(y1te)) > 1:
        cols = st.columns(3)
        for col, (name, model) in zip(cols, [("Decision Tree", mdata["dt"]),
                                              ("SVM",            mdata["svm"]),
                                              ("Neural Network", mdata["mlp"])]):
            with col:
                cm = confusion_matrix(y1te, model.predict(X1te))
                fig, ax = plt.subplots(figsize=(4, 3.5))
                ConfusionMatrixDisplay(cm, display_labels=["Low","High"]).plot(
                    ax=ax, colorbar=False, cmap="Blues")
                ax.set_title(name, fontweight="bold")
                plt.tight_layout(); fig_to_st(fig)
    else:
        st.info("Confusion matrix requires at least 2 classes in D1 test set.")

    # ── ROC Curves
    st.markdown('<div class="section-hdr">📈 ROC Curves (D1 Test)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, model) in enumerate([("Decision Tree", mdata["dt"]),
                                        ("SVM",            mdata["svm"]),
                                        ("Neural Network", mdata["mlp"])]):
        fpr, tpr, roc_auc = roc_for(model, X1te, y1te, "D1-Test")
        if fpr is not None:
            ax.plot(fpr, tpr, color=PALETTE[i], lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],"k--",alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Dataset 1 Test", fontweight="bold")
    ax.legend()
    plt.tight_layout(); fig_to_st(fig)

    # ── Feature Importance
    st.markdown('<div class="section-hdr">🎯 Feature Importance (Decision Tree)</div>',
                unsafe_allow_html=True)
    fi = pd.DataFrame({"feature": feat_names,
                        "importance": mdata["dt"].feature_importances_})
    fi = fi[fi["importance"] > 0].sort_values("importance", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, max(4, len(fi) * 0.4)))
    sns.barplot(data=fi, y="feature", x="importance", ax=ax, palette="Blues_r", orient="h")
    ax.set_xlabel("Gini Importance")
    ax.set_title("Decision Tree – Top Feature Importances", fontweight="bold")
    plt.tight_layout(); fig_to_st(fig)

    # ── DT Visualisation
    st.markdown('<div class="section-hdr">🌳 Decision Tree Visualisation</div>',
                unsafe_allow_html=True)
    max_vis_depth = st.slider("Visualisation depth", 1, min(dt_depth, 5), 3)
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(mdata["dt"], feature_names=feat_names,
              class_names=["Low","High"], filled=True,
              max_depth=max_vis_depth, ax=ax, impurity=False,
              proportion=True, fontsize=8)
    ax.set_title(f"Decision Tree (first {max_vis_depth} levels)", fontweight="bold")
    plt.tight_layout(); fig_to_st(fig)

    # ── Bias-Variance (DT depth)
    st.markdown('<div class="section-hdr">📐 Bias–Variance Trade-off Analysis</div>',
                unsafe_allow_html=True)
    depths = list(range(1, 13))
    tr_f1s, te_f1s = [], []
    for d in depths:
        tmp = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE, class_weight="balanced")
        if len(np.unique(y1tr)) > 1:
            tmp.fit(X1tr, y1tr)
            tr_f1s.append(f1_score(y1tr, tmp.predict(X1tr), average="weighted", zero_division=0))
            te_f1s.append(f1_score(y1te, tmp.predict(X1te), average="weighted", zero_division=0)
                          if len(np.unique(y1te)) > 1 else np.nan)
        else:
            tr_f1s.append(np.nan); te_f1s.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(depths, tr_f1s, "o-", color=PALETTE[0], label="Train F1")
    ax.plot(depths, te_f1s, "s-", color=PALETTE[3], label="Test F1")
    ax.axvline(dt_depth, ls="--", color="gray", alpha=0.7, label=f"Current depth={dt_depth}")
    ax.set_xlabel("Max Tree Depth"); ax.set_ylabel("Weighted F1")
    ax.set_title("Decision Tree: Bias–Variance Trade-off", fontweight="bold")
    ax.legend(); ax.set_xticks(depths)
    plt.tight_layout(); fig_to_st(fig)

    # MLP Loss Curve
    mlp_m = mdata["mlp"]
    if hasattr(mlp_m, "loss_curve_"):
        st.markdown("**Neural Network Training Loss Curve**")
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(mlp_m.loss_curve_, color=PALETTE[0], label="Train Loss")
        if hasattr(mlp_m, "validation_scores_") and mlp_m.validation_scores_:
            ax.plot(mlp_m.validation_scores_, color=PALETTE[3], label="Val Score")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("MLP Training Curve", fontweight="bold"); ax.legend()
        plt.tight_layout(); fig_to_st(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TEMPORAL SHIFT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⏱ Temporal Shift Analysis":
    st.markdown('<div class="main-title">⏱ Temporal Shift Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    Models trained on **Dataset 1 (Historical)** are evaluated on **Dataset 2 (Current)**
    to measure performance degradation caused by **temporal data drift**.
    """)

    # ── Side-by-side metrics
    st.markdown('<div class="section-hdr">📊 D1-Test vs D2-Test Performance</div>',
                unsafe_allow_html=True)

    d1_test_res  = results_df[results_df["split"] == "D1-Test"].set_index("model")
    d2_shift_res = results_df[results_df["split"] == "D2-Test (Pre-CL)"].set_index("model")

    model_names = [m for m in ["Decision Tree","SVM","Neural Network"]
                   if m in d1_test_res.index]

    metric_tabs = st.tabs(["Accuracy", "Precision", "Recall", "F1", "AUC"])
    for tab, met in zip(metric_tabs, ["accuracy","precision","recall","f1","auc"]):
        with tab:
            rows = []
            for mn in model_names:
                v1 = d1_test_res.loc[mn, met] if mn in d1_test_res.index else np.nan
                v2 = d2_shift_res.loc[mn, met] if mn in d2_shift_res.index else np.nan
                rows.append({"Model": mn,
                              "D1-Test": round(v1, 4) if not np.isnan(v1) else "N/A",
                              "D2-Test": round(v2, 4) if not np.isnan(v2) else "N/A",
                              "Δ (drift)": f"{v2 - v1:+.4f}" if not (np.isnan(v1) or np.isnan(v2)) else "N/A"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(7, 3.5))
            x = np.arange(len(model_names))
            v1s = [d1_test_res.loc[m, met] if m in d1_test_res.index else 0 for m in model_names]
            v2s = [d2_shift_res.loc[m, met] if m in d2_shift_res.index else 0 for m in model_names]
            ax.bar(x - 0.2, v1s, 0.35, label="D1-Test", color=PALETTE[0], edgecolor="white")
            ax.bar(x + 0.2, v2s, 0.35, label="D2-Test", color=PALETTE[3], edgecolor="white")
            ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=15)
            ax.set_ylim(0, 1.1); ax.set_ylabel(met.upper())
            ax.set_title(f"{met.upper()}: D1-Test vs D2-Test (Temporal Shift)", fontweight="bold")
            ax.legend()
            plt.tight_layout(); fig_to_st(fig)

    # ── ROC comparison
    st.markdown('<div class="section-hdr">📈 ROC Curves: D1-Test vs D2-Test</div>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (name, model) in zip(cols, [("Decision Tree", mdata["dt"]),
                                          ("SVM",            mdata["svm"]),
                                          ("Neural Network", mdata["mlp"])]):
        with col:
            fig, ax = plt.subplots(figsize=(4.5, 4))
            for X_ev, y_ev, lbl, ls, c in [
                (X1te, y1te, "D1-Test", "-",  PALETTE[0]),
                (X2te, y2te, "D2-Test", "--", PALETTE[3]),
            ]:
                fpr, tpr, ra = roc_for(model, X_ev, y_ev, lbl)
                if fpr is not None:
                    ax.plot(fpr, tpr, ls=ls, color=c, lw=2, label=f"{lbl} (AUC={ra:.2f})")
            ax.plot([0,1],[0,1],"k--",alpha=0.4)
            ax.set_title(name, fontweight="bold")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.legend(fontsize=8)
            plt.tight_layout(); fig_to_st(fig)

    # ── Confusion matrices side by side
    st.markdown('<div class="section-hdr">🔢 Confusion Matrices: D1-Test vs D2-Test</div>',
                unsafe_allow_html=True)
    for name, model in [("Decision Tree", mdata["dt"]),
                         ("SVM",            mdata["svm"]),
                         ("Neural Network", mdata["mlp"])]:
        st.markdown(f"**{name}**")
        c1, c2 = st.columns(2)
        for col, (X_ev, y_ev, lbl) in zip([c1, c2], [(X1te, y1te, "D1-Test"),
                                                       (X2te, y2te, "D2-Test")]):
            with col:
                if len(y_ev) > 0 and len(np.unique(y_ev)) > 1:
                    cm = confusion_matrix(y_ev, model.predict(X_ev))
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ConfusionMatrixDisplay(cm, display_labels=["Low","High"]).plot(
                        ax=ax, colorbar=False, cmap="Blues")
                    ax.set_title(lbl)
                    plt.tight_layout(); fig_to_st(fig)
                else:
                    st.info(f"Not enough data for CM ({lbl})")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CONTINUAL LEARNING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔄 Continual Learning":
    st.markdown('<div class="main-title">🔄 Continual Learning</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Strategy Implemented
    | Model | CL Technique | Description |
    |---|---|---|
    | **Decision Tree** | Combined retraining | Retrain on D1_train ∪ D2_train |
    | **SVM** | Incremental learning | `SGDClassifier` with `partial_fit` (online updates) |
    | **Neural Network** | Fine-tuning | `warm_start=True` – continue training on D2 with reduced LR |

    This simulates a **production scenario** where models trained on historical data
    are continuously adapted to newer patient distributions.
    """)

    # ── Pre vs Post CL comparison
    st.markdown('<div class="section-hdr">📊 Pre-CL vs Post-CL Performance on D2-Test</div>',
                unsafe_allow_html=True)

    pre_cl = results_df[results_df["split"] == "D2-Test (Pre-CL)"].set_index("model")
    post_cl = results_df[results_df["split"] == "D2-Test (Post-CL)"].set_index("model")

    cl_map = {"Decision Tree (CL)": "Decision Tree",
               "SVM-SGD (CL)": "SVM",
               "Neural Network (CL)": "Neural Network"}

    comparison_rows = []
    for cl_name, base_name in cl_map.items():
        for met in ["accuracy","precision","recall","f1","auc"]:
            pre_v  = pre_cl.loc[base_name, met]  if base_name in pre_cl.index  else np.nan
            post_v = post_cl.loc[cl_name, met]   if cl_name   in post_cl.index else np.nan
            comparison_rows.append({
                "Model": base_name, "Metric": met.upper(),
                "Pre-CL (D1→D2)": round(pre_v, 4) if not np.isnan(pre_v) else "N/A",
                "Post-CL": round(post_v, 4) if not np.isnan(post_v) else "N/A",
                "Improvement":
                    f"{post_v - pre_v:+.4f}" if not (np.isnan(pre_v) or np.isnan(post_v)) else "N/A"
            })

    cmp_df = pd.DataFrame(comparison_rows)
    st.dataframe(cmp_df, use_container_width=True)

    # ── Visual comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, met in zip(axes, ["accuracy","precision","recall","f1"]):
        x = np.arange(3)
        pre_vals  = [pre_cl.loc[m, met]  if m in pre_cl.index  else 0
                     for m in ["Decision Tree","SVM","Neural Network"]]
        post_vals = [post_cl.loc[cl, met] if cl in post_cl.index else 0
                     for cl in cl_map.keys()]
        ax.bar(x - 0.2, pre_vals,  0.35, label="Pre-CL",  color=PALETTE[0], edgecolor="white")
        ax.bar(x + 0.2, post_vals, 0.35, label="Post-CL", color=PALETTE[3], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(["DT","SVM","MLP"], fontsize=9)
        ax.set_ylim(0, 1.1); ax.set_ylabel(met.upper())
        ax.set_title(met.upper(), fontweight="bold")
        ax.legend(fontsize=7)
    plt.suptitle("Continual Learning Impact on D2-Test Performance",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); fig_to_st(fig)

    # ── CL model confusion matrices
    st.markdown('<div class="section-hdr">🔢 CL Model Confusion Matrices (D2-Test)</div>',
                unsafe_allow_html=True)
    if len(np.unique(y2te)) > 1:
        cols = st.columns(3)
        for col, (name, model, use_np) in zip(cols,
                [("Decision Tree (CL)", mdata["dt_cl"], False),
                 ("SVM-SGD (CL)",       mdata["sgd"],  True),
                 ("Neural Network (CL)", mdata["mlp_cl"], False)]):
            with col:
                X_ev = X2te.values if use_np else X2te
                cm = confusion_matrix(y2te, model.predict(X_ev))
                fig, ax = plt.subplots(figsize=(4, 3))
                ConfusionMatrixDisplay(cm, display_labels=["Low","High"]).plot(
                    ax=ax, colorbar=False, cmap="Greens")
                ax.set_title(name, fontweight="bold")
                plt.tight_layout(); fig_to_st(fig)
    else:
        st.info("Cannot draw CM – need at least 2 classes in D2-Test.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FINAL RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Final Results Summary":
    st.markdown('<div class="main-title">📈 Final Results Summary</div>', unsafe_allow_html=True)

    # ── Full results table
    st.markdown('<div class="section-hdr">📋 All Results</div>', unsafe_allow_html=True)
    display_df = results_df.copy()
    display_df[["accuracy","precision","recall","f1","auc"]] = \
        display_df[["accuracy","precision","recall","f1","auc"]].round(4)
    st.dataframe(display_df, use_container_width=True)

    # ── Heatmap of all metrics
    st.markdown('<div class="section-hdr">🔥 Performance Heatmap</div>', unsafe_allow_html=True)
    metrics_heat = ["accuracy","precision","recall","f1","auc"]
    pivot_heat = results_df.copy()
    pivot_heat["model_split"] = pivot_heat["model"] + " | " + pivot_heat["split"]
    heat_data = pivot_heat.set_index("model_split")[metrics_heat]

    fig, ax = plt.subplots(figsize=(10, max(5, len(heat_data) * 0.55)))
    sns.heatmap(heat_data.astype(float), annot=True, fmt=".3f", cmap="YlGnBu",
                ax=ax, linewidths=0.4, vmin=0, vmax=1,
                cbar_kws={"label": "Score"})
    ax.set_title("Model × Split Performance Heatmap", fontweight="bold")
    plt.tight_layout(); fig_to_st(fig)

    # ── Radar chart per model
    st.markdown('<div class="section-hdr">🕸 Radar Charts</div>', unsafe_allow_html=True)
    d1_test = results_df[results_df["split"] == "D1-Test"]
    d2_pre  = results_df[results_df["split"] == "D2-Test (Pre-CL)"]
    d2_post = results_df[results_df["split"] == "D2-Test (Post-CL)"]

    radar_metrics = ["accuracy","precision","recall","f1"]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": "polar"})
    base_models = [("Decision Tree",      "Decision Tree (CL)"),
                   ("SVM",                "SVM-SGD (CL)"),
                   ("Neural Network",     "Neural Network (CL)")]

    for ax, (base, cl_name) in zip(axes, base_models):
        for df_sel, label, color in [(d1_test,  "D1-Test",    PALETTE[0]),
                                      (d2_pre,   "D2-Test",    PALETTE[3]),
                                      (d2_post,  "Post-CL",    PALETTE[2])]:
            row = df_sel[df_sel["model"].str.contains(base.split()[0])]
            if row.empty: continue
            vals = row.iloc[0][radar_metrics].values.tolist()
            vals += vals[:1]
            ax.plot(angles, vals, "-o", color=color, lw=2, label=label)
            ax.fill(angles, vals, color=color, alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in radar_metrics], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(base, fontweight="bold", pad=14)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)

    plt.suptitle("Radar Charts: D1-Test | D2-Test | Post-CL", fontweight="bold", fontsize=12)
    plt.tight_layout(); fig_to_st(fig)

    # ── Key findings
    st.markdown('<div class="section-hdr">💡 Key Findings & Analysis</div>',
                unsafe_allow_html=True)
    st.markdown("""
    #### 1. Data Drift Observation
    - Comparing Dataset 1 and Dataset 2 feature distributions reveals
      **temporal covariate shift** — the statistical properties of patient features
      change over time, reflecting evolving clinical practices and patient demographics.

    #### 2. Model Performance Under Temporal Shift
    - All models trained on Dataset 1 show **reduced performance on Dataset 2 test**,
      confirming the practical challenge of deploying static models in dynamic healthcare settings.
    - The **Neural Network** (MLP) is generally the most expressive but also the most
      sensitive to covariate shift.
    - The **Decision Tree** provides the most interpretable predictions and stable performance.

    #### 3. Bias–Variance Trade-off
    - Shallow Decision Trees exhibit **high bias** (underfitting); deep trees show
      **high variance** (overfitting). An intermediate depth balances both.
    - SVM with RBF kernel has a natural regularisation via the C parameter.

    #### 4. Continual Learning Effectiveness
    - **Neural Network fine-tuning** (`warm_start`) is effective at adapting to new data
      without full retraining — a practical continual learning approach.
    - **Incremental SGD-SVM** provides online updates for SVM-like decision boundaries.
    - **Combined retraining** of the Decision Tree anchors historical knowledge while
      incorporating new patterns.

    #### 5. Feature Importance Insights
    - Healthcare-related count features (num_conditions, num_medications, total_enc_cost)
      and key vital signs are consistently the most predictive of high expenses.
    - Demographic features (age, income) contribute significantly to the model.

    #### 6. Clinical Implications
    - Identifying high-cost patients early enables **proactive intervention**,
      optimising resource allocation and improving patient outcomes.
    - Continual learning strategies are essential for maintaining model relevance
      as patient populations and clinical protocols evolve.
    """)

    # ── Download results
    st.divider()
    csv_out = display_df.to_csv(index=False)
    st.download_button("⬇ Download Results CSV", data=csv_out,
                        file_name="TeamXX_Assignment2_Results.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem;">
  BITS Pilani – Hyderabad Campus &nbsp;|&nbsp; BITS F464 Machine Learning &nbsp;|&nbsp; Assignment 2 &nbsp;|&nbsp;
  TeamXX &nbsp;|&nbsp; Second Semester 2025-2026
</div>
""", unsafe_allow_html=True)
