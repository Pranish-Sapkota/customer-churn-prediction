import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📶",
    layout="centered",
    initial_sidebar_state="collapsed",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

is_dark = st.session_state.dark_mode

if is_dark:
    PAGE_BG        = "#0f172a"
    CARD_BG        = "#1e293b"
    CARD_BORDER    = "#334155"
    CARD_SHADOW    = "rgba(0,0,0,0.4)"
    TEXT_PRIMARY   = "#f1f5f9"
    TEXT_SECONDARY = "#94a3b8"
    TEXT_LABEL     = "#cbd5e1"
    INPUT_BG       = "#0f172a"
    INPUT_BORDER   = "#334155"
    DIVIDER        = "#1e293b"
    TOGGLE_BG      = "#1e293b"
    TOGGLE_BORDER  = "#334155"
    TOGGLE_COLOR   = "#f1f5f9"
    GAUGE_TRACK    = "#334155"
    GAUGE_TEXT_CLR = "#94a3b8"
    HERO_GRAD      = "linear-gradient(135deg, #020617 0%, #1e3a8a 60%, #075985 100%)"
else:
    PAGE_BG        = "#f1f5f9"
    CARD_BG        = "#ffffff"
    CARD_BORDER    = "#e2e8f0"
    CARD_SHADOW    = "rgba(15,23,42,0.07)"
    TEXT_PRIMARY   = "#0f172a"
    TEXT_SECONDARY = "#64748b"
    TEXT_LABEL     = "#374151"
    INPUT_BG       = "#ffffff"
    INPUT_BORDER   = "#cbd5e1"
    DIVIDER        = "#e2e8f0"
    TOGGLE_BG      = "#ffffff"
    TOGGLE_BORDER  = "#e2e8f0"
    TOGGLE_COLOR   = "#0f172a"
    GAUGE_TRACK    = "#e2e8f0"
    GAUGE_TEXT_CLR = "#94a3b8"
    HERO_GRAD      = "linear-gradient(135deg, #0f172a 0%, #1e40af 60%, #0369a1 100%)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

[data-testid="stAppViewContainer"] {{
    background: {PAGE_BG} !important;
    transition: background 0.3s ease;
}}
[data-testid="block-container"] {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 740px;
}}
[data-testid="stAppViewContainer"] > section > div {{
    background: {PAGE_BG};
}}

/* ── Toggle bar ── */
.top-bar {{
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1.2rem;
}}
.theme-badge {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: {TOGGLE_BG};
    border: 1px solid {TOGGLE_BORDER};
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.78rem;
    font-weight: 600;
    color: {TOGGLE_COLOR};
    box-shadow: 0 1px 4px {CARD_SHADOW};
    cursor: pointer;
    user-select: none;
    transition: all 0.2s;
}}

/* ── Hero ── */
.hero {{
    background: {HERO_GRAD};
    border-radius: 20px;
    padding: 2rem 2.4rem 1.8rem;
    color: #fff;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 24px rgba(15,23,42,0.22);
}}
.hero-tag {{
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 12px;
    color: #bae6fd;
    margin-bottom: 0.8rem;
}}
.hero h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 0.35rem;
    letter-spacing: -0.5px;
    line-height: 1.25;
}}
.hero p {{
    margin: 0;
    opacity: 0.65;
    font-size: 0.84rem;
    line-height: 1.55;
}}

/* ── Card ── */
.input-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 18px;
    padding: 1.8rem 2rem 1.6rem;
    box-shadow: 0 2px 16px {CARD_SHADOW};
    margin-bottom: 1.2rem;
    transition: background 0.3s ease, border 0.3s ease;
}}
.card-title {{
    font-size: 0.68rem;
    font-weight: 700;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1.2rem;
}}

/* ── Labels & inputs ── */
label, [data-testid="stWidgetLabel"] p {{
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: {TEXT_LABEL} !important;
}}
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {{
    background: {INPUT_BG} !important;
    border-color: {INPUT_BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 8px !important;
}}
[data-testid="stSlider"] > div > div > div {{
    color: {TEXT_PRIMARY} !important;
}}

/* ── Predict button ── */
div[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, #1d4ed8, #0369a1);
    color: #fff !important;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.75rem 2rem;
    width: 100%;
    letter-spacing: 0.01em;
    box-shadow: 0 4px 14px rgba(29,78,216,0.32);
    transition: all 0.2s;
    margin-top: 0.5rem;
}}
div[data-testid="stButton"] > button:hover {{
    box-shadow: 0 6px 20px rgba(29,78,216,0.45);
    transform: translateY(-1px);
}}

/* ── Result box ── */
.result-box {{
    border-radius: 18px;
    padding: 2rem 2.2rem;
    margin-top: 0.5rem;
    box-shadow: 0 2px 12px {CARD_SHADOW};
    transition: all 0.3s;
}}
.result-box.churn {{
    background: {'linear-gradient(135deg, #2d0a0a, #3b1010)' if is_dark else 'linear-gradient(135deg, #fff1f2, #ffe4e6)'};
    border: 1.5px solid {'#7f1d1d' if is_dark else '#fca5a5'};
}}
.result-box.safe {{
    background: {'linear-gradient(135deg, #052e16, #14532d)' if is_dark else 'linear-gradient(135deg, #f0fdf4, #dcfce7)'};
    border: 1.5px solid {'#166534' if is_dark else '#86efac'};
}}
.result-tag {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {TEXT_SECONDARY};
    margin-bottom: 0.5rem;
}}
.result-verdict {{
    font-size: 1.95rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}}
.result-sub {{
    font-size: 0.83rem;
    color: {TEXT_SECONDARY};
    margin-bottom: 1rem;
}}
.badge {{
    display: inline-block;
    border-radius: 999px;
    padding: 5px 16px;
    font-size: 0.78rem;
    font-weight: 600;
}}
.c-red   {{ color: #f87171; }}
.c-green {{ color: #4ade80; }}

/* ── Divider ── */
.divider {{
    height: 1px;
    background: {DIVIDER};
    margin: 1.6rem 0;
}}

/* ── Alert overrides ── */
[data-testid="stAlert"] {{
    border-radius: 12px !important;
    background: {'#1e293b' if is_dark else ''} !important;
    color: {TEXT_PRIMARY} !important;
}}

/* ── Footer ── */
.footer {{
    text-align: center;
    color: {TEXT_SECONDARY};
    font-size: 0.73rem;
    margin-top: 2.5rem;
    line-height: 1.9;
}}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent

@st.cache_resource(show_spinner="Initialising model…")
def load_artifacts():
    model_path    = BASE_DIR.parent / "model" / "churn_model.pkl"
    features_path = BASE_DIR.parent / "model" / "features.pkl"
    if not model_path.exists():
        st.error(f"Model not found: `{model_path}`")
        st.stop()
    if not features_path.exists():
        st.error(f"Features file not found: `{features_path}`")
        st.stop()
    return joblib.load(model_path), joblib.load(features_path)

model, feature_names = load_artifacts()

# ── Theme toggle ──
t_col1, t_col2 = st.columns([5, 1])
with t_col2:
    icon  = "☀  Light" if is_dark else "☾  Dark"
    st.button(icon, on_click=toggle_theme, use_container_width=True)

# ── Hero ──
st.markdown("""
<div class="hero">
  <div class="hero-tag">AI · Telecom Analytics</div>
  <h1>Customer Churn Predictor</h1>
  <p>Enter the customer profile below to instantly assess their likelihood of churning using a trained Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

# ── Input card ──
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Customer Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0,
                                      value=round(monthly_charges * tenure, 2), step=1.0)
    contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet        = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

with col2:
    payment         = st.selectbox("Payment Method", [
                          "Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"
                      ])
    tech_support    = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
    online_security = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
    senior          = st.selectbox("Senior Citizen",     ["No", "Yes"])

st.markdown('</div>', unsafe_allow_html=True)

predict_btn = st.button("Run Prediction")


def build_input(feature_names):
    raw = {
        "gender":                                    0,
        "SeniorCitizen":                             1 if senior == "Yes" else 0,
        "Partner":                                   0,
        "Dependents":                                0,
        "tenure":                                    tenure,
        "PhoneService":                              1,
        "PaperlessBilling":                          1 if paperless == "Yes" else 0,
        "MonthlyCharges":                            monthly_charges,
        "TotalCharges":                              total_charges,
        "MultipleLines_No phone service":            0,
        "MultipleLines_Yes":                         0,
        "InternetService_Fiber optic":               1 if internet == "Fiber optic" else 0,
        "InternetService_No":                        1 if internet == "No" else 0,
        "OnlineSecurity_No internet service":        1 if online_security == "No internet service" else 0,
        "OnlineSecurity_Yes":                        1 if online_security == "Yes" else 0,
        "OnlineBackup_No internet service":          0,
        "OnlineBackup_Yes":                          0,
        "DeviceProtection_No internet service":      0,
        "DeviceProtection_Yes":                      0,
        "TechSupport_No internet service":           1 if tech_support == "No internet service" else 0,
        "TechSupport_Yes":                           1 if tech_support == "Yes" else 0,
        "StreamingTV_No internet service":           0,
        "StreamingTV_Yes":                           0,
        "StreamingMovies_No internet service":       0,
        "StreamingMovies_Yes":                       0,
        "Contract_One year":                         1 if contract == "One year" else 0,
        "Contract_Two year":                         1 if contract == "Two year" else 0,
        "PaymentMethod_Credit card (automatic)":     1 if payment == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check":            1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check":                1 if payment == "Mailed check" else 0,
    }
    row = {f: 0 for f in feature_names}
    for k, v in raw.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([row])[feature_names]


def draw_gauge(prob):
    fig_bg = CARD_BG if not is_dark else "#1e293b"
    fig, ax = plt.subplots(figsize=(4.2, 2.5), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    theta_bg = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), lw=18, color=GAUGE_TRACK, solid_capstyle="round")

    fill_color = "#f87171" if prob >= 0.5 else "#4ade80"
    theta_fg = np.linspace(np.pi, np.pi - prob * np.pi, 300)
    ax.plot(np.cos(theta_fg), np.sin(theta_fg), lw=18, color=fill_color, solid_capstyle="round")

    angle = np.pi - prob * np.pi
    needle_color = "#f1f5f9" if is_dark else "#0f172a"
    ax.annotate("", xy=(0.67 * np.cos(angle), 0.67 * np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=needle_color, lw=2.5, mutation_scale=20))
    ax.plot(0, 0, "o", color=needle_color, ms=10)

    ax.text(0, -0.18, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=24, fontweight="700", color=fill_color)
    ax.text(0, -0.42, "Churn Probability", ha="center", va="center",
            fontsize=8.5, color=GAUGE_TEXT_CLR)

    for pct, lbl in [(0, "0%"), (0.5, "50%"), (1, "100%")]:
        a = np.pi - pct * np.pi
        ax.text(1.15 * np.cos(a), 1.15 * np.sin(a), lbl,
                ha="center", va="center", fontsize=7, color=GAUGE_TEXT_CLR)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.6, 1.2)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


if predict_btn:
    with st.spinner("Analysing customer profile…"):
        input_df   = build_input(feature_names)
        prob       = model.predict_proba(input_df)[0][1]
        prediction = "Churn" if prob >= 0.5 else "No Churn"

    if prob >= 0.75:
        risk, risk_bg, risk_color = "High Risk",     "#fee2e2" if not is_dark else "#3b0a0a", "#f87171"
    elif prob >= 0.5:
        risk, risk_bg, risk_color = "Medium Risk",   "#ffedd5" if not is_dark else "#3b1500", "#fb923c"
    elif prob >= 0.25:
        risk, risk_bg, risk_color = "Low Risk",      "#fef9c3" if not is_dark else "#2d2700", "#facc15"
    else:
        risk, risk_bg, risk_color = "Minimal Risk",  "#dcfce7" if not is_dark else "#052e16", "#4ade80"

    g_col, r_col = st.columns([1.05, 1], gap="large")

    with g_col:
        st.pyplot(draw_gauge(prob), use_container_width=True)

    with r_col:
        box_class = "churn" if prediction == "Churn" else "safe"
        val_class = "c-red"  if prediction == "Churn" else "c-green"
        verdict   = "Will Churn"  if prediction == "Churn" else "Will Retain"
        sub       = "This customer is likely to leave." if prediction == "Churn" else "This customer is likely to stay."

        st.markdown(f"""
        <div class="result-box {box_class}">
          <div class="result-tag">Prediction Output</div>
          <div class="result-verdict {val_class}">{verdict}</div>
          <div class="result-sub">{sub}</div>
          <span class="badge" style="background:{risk_bg};color:{risk_color}">{risk}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if prediction == "Churn":
        st.warning(
            "**Recommended Action:** Offer a contract upgrade or loyalty discount. "
            "A personalised retention call may significantly reduce churn risk.",
            icon="⚡",
        )
    else:
        st.success(
            "**No Intervention Needed:** This customer shows a healthy retention profile. "
            "Continue standard engagement to maintain satisfaction.",
            icon="✔",
        )

st.markdown(f"""
<div class="footer">
  Powered by Random Forest &nbsp;·&nbsp; Telco Customer Churn Dataset<br>
  Built with Streamlit<br>
  Develop by Pranish Pr Sapkota with ❤️
</div>
""", unsafe_allow_html=True)
