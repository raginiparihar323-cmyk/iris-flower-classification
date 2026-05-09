import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palettes ──────────────────────────────────────────────────────────────────
COLORS = {
    "Iris-setosa":     "#f43f5e",
    "Iris-versicolor": "#8b5cf6",
    "Iris-virginica":  "#06b6d4",
}
SPECIES_EMOJI = {
    "Iris-setosa":     "🌺",
    "Iris-versicolor": "🌿",
    "Iris-virginica":  "💜",
}
SPECIES_DESC = {
    "Iris-setosa":     "Small, compact flower with short petals. Native to Arctic and subarctic regions.",
    "Iris-versicolor": "Medium-sized bloom with violet-blue petals. Common in North America.",
    "Iris-virginica":  "Large, elegant flower with wide petals. Found in eastern North America.",
}

# ── Dark mode state ───────────────────────────────────────────────────────────
if "dark" not in st.session_state:
    st.session_state.dark = False

# ── Theme helper ──────────────────────────────────────────────────────────────
def T():
    if st.session_state.dark:
        return {
            "bg":         "#0f172a",
            "surface":    "#1e293b",
            "border":     "#334155",
            "ink":        "#f1f5f9",
            "muted":      "#94a3b8",
            "sidebar_bg": "#020617",
            "pill_bg":    "#1e293b",
            "pill_color": "#cbd5e1",
            "plot_bg":    "#1e293b",
            "plot_paper": "#1e293b",
            "plot_font":  "#f1f5f9",
            "algo_bg":    "#334155",
        }
    return {
        "bg":         "#f8f5f2",
        "surface":    "#ffffff",
        "border":     "#e2e8f0",
        "ink":        "#0f172a",
        "muted":      "#64748b",
        "sidebar_bg": "#0f172a",
        "pill_bg":    "#f1f5f9",
        "pill_color": "#334155",
        "plot_bg":    "#ffffff",
        "plot_paper": "#ffffff",
        "plot_font":  "#0f172a",
        "algo_bg":    "#0f172a",
    }

t = T()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {{
    --rose:       #f43f5e;
    --violet:     #8b5cf6;
    --cyan:       #06b6d4;
    --ink:        {t['ink']};
    --muted:      {t['muted']};
    --surface:    {t['surface']};
    --bg:         {t['bg']};
    --border:     {t['border']};
    --pill-bg:    {t['pill_bg']};
    --pill-color: {t['pill_color']};
    --radius:     18px;
}}

html, body, [class*="css"], .stApp {{
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--ink) !important;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

[data-testid="stSidebar"] {{
    background: {t['sidebar_bg']} !important;
    border-right: none;
}}
[data-testid="stSidebar"] * {{ color: #e2e8f0 !important; }}
[data-testid="stSidebar"] .stSlider > label {{
    font-weight: 600; font-size: 12px;
    letter-spacing: .06em; text-transform: uppercase;
    color: #94a3b8 !important;
}}

div.stButton > button {{
    background: linear-gradient(135deg, var(--rose), var(--violet));
    color: white !important; border: none;
    border-radius: 12px; padding: 14px 0;
    font-size: 16px; font-weight: 700;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: .03em; width: 100%;
    transition: opacity .2s, transform .15s;
    box-shadow: 0 4px 18px rgba(244,63,94,.35);
}}
div.stButton > button:hover {{
    opacity: .9; transform: translateY(-2px); color: white !important;
}}

@keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(12px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}

.metric-row {{
    display: grid; grid-template-columns: repeat(3,1fr);
    gap: 18px; margin-bottom: 28px;
}}
.metric-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 22px 20px 18px;
    display: flex; flex-direction: column; gap: 4px;
    animation: fadeUp .5s ease both;
}}
.metric-card.accent {{ border-top: 3px solid var(--rose); }}
.metric-card .label {{
    font-size: 11px; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: var(--muted);
}}
.metric-card .value {{
    font-family: 'DM Serif Display', serif;
    font-size: 42px; color: var(--ink); line-height: 1.1;
}}
.metric-card .sub {{ font-size: 13px; color: var(--muted); }}

.section-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 26px; color: var(--ink); margin: 32px 0 16px;
}}

/* Species cards */
.species-grid {{
    display: grid; grid-template-columns: repeat(3,1fr); gap: 18px; margin-top: 10px;
}}
.species-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 22px 18px 18px;
    text-align: center; transition: transform .2s, box-shadow .2s;
    animation: fadeUp .5s ease both;
}}
.species-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(0,0,0,.13);
}}
.species-card .s-emoji {{ font-size: 52px; line-height: 1; margin-bottom: 10px; }}
.species-card .s-name {{
    font-family: 'DM Serif Display', serif;
    font-size: 20px; color: var(--ink); margin-bottom: 8px;
}}
.species-card .s-badge {{
    display: inline-block; border-radius: 999px;
    padding: 3px 14px; font-size: 11px; font-weight: 700;
    letter-spacing: .06em; text-transform: uppercase;
    color: white; margin-bottom: 12px;
}}
.species-card .s-desc {{ font-size: 13px; color: var(--muted); line-height: 1.6; }}

/* Result */
.result-wrap {{
    border-radius: var(--radius); padding: 36px 28px;
    text-align: center; animation: fadeUp .4s ease both;
}}
.result-emoji {{ font-size: 72px; line-height: 1; margin-bottom: 12px; }}
.result-label {{
    font-size: 12px; font-weight: 600; letter-spacing: .14em;
    text-transform: uppercase; opacity: .75; margin-bottom: 6px;
}}
.result-species {{ font-family: 'DM Serif Display', serif; font-size: 36px; }}
.conf-bar-wrap {{
    background: var(--border); border-radius: 999px;
    height: 8px; margin: 12px auto 0; max-width: 220px; overflow: hidden;
}}
.conf-bar-fill {{ height: 100%; border-radius: 999px; }}
.result-conf {{ margin-top: 8px; font-size: 14px; color: var(--muted); }}

/* Probability bars */
.proba-row {{ display: flex; flex-direction: column; gap: 10px; margin-top: 14px; }}
.proba-item {{ display: flex; align-items: center; gap: 10px; font-size: 13px; }}
.proba-label {{ width: 100px; color: var(--muted); font-weight: 500; }}
.proba-bar-bg {{
    flex: 1; background: var(--border);
    border-radius: 999px; height: 8px; overflow: hidden;
}}
.proba-bar-fill {{ height: 100%; border-radius: 999px; }}
.proba-pct {{ width: 42px; text-align: right; color: var(--ink); font-weight: 600; }}

/* About / features */
.about-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 28px; height: 100%;
}}
.about-card h3 {{
    font-family: 'DM Serif Display', serif;
    font-size: 22px; margin-bottom: 12px; color: var(--ink);
}}
.about-card p {{ color: var(--muted); font-size: 15px; line-height: 1.7; }}
.algo-badge {{
    display: inline-block; background: {t['algo_bg']};
    color: white; border-radius: 10px;
    padding: 10px 18px; font-size: 13px;
    font-weight: 600; margin-top: 16px; letter-spacing: .04em;
}}

.pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
.pill {{
    background: var(--pill-bg); color: var(--pill-color);
    border-radius: 999px; padding: 5px 14px;
    font-size: 13px; font-weight: 500; border: 1px solid var(--border);
}}

.divider {{ border: none; border-top: 1px solid var(--border); margin: 36px 0; }}

.hero {{
    padding: 36px 0 8px; display: flex; align-items: center; gap: 18px;
}}
.hero-icon {{ font-size: 52px; line-height: 1; }}
.hero-text h1 {{
    font-family: 'DM Serif Display', serif;
    font-size: 42px; color: var(--ink); margin: 0; line-height: 1.1;
}}
.hero-text p {{ font-size: 16px; color: var(--muted); margin: 6px 0 0; }}
</style>
""", unsafe_allow_html=True)

# ── Load & train ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("Iris.csv")
    X  = df.drop(["Id", "Species"], axis=1)
    y  = df["Species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = LogisticRegression(max_iter=300)
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm_    = confusion_matrix(y_test, y_pred, labels=list(COLORS.keys()))
    return df, mdl, acc, cm_

df, model, accuracy, cm = load_and_train()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px;">
        <div style="font-size:32px;">🌸</div>
        <div style="font-family:'DM Serif Display',serif;font-size:22px;color:white;margin:4px 0 2px;">
            Iris Classifier
        </div>
        <div style="font-size:11px;color:#94a3b8;letter-spacing:.08em;text-transform:uppercase;">
            Measurement Input
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width  = st.slider("Sepal Width (cm)",  2.0, 5.0, 3.5, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width  = st.slider("Petal Width (cm)",  0.1, 2.6, 0.2, 0.1)

    st.write("")
    predict_btn = st.button("🔮 Predict Species")

    st.markdown("---")

    dark_label = "☀️ Switch to Light" if st.session_state.dark else "🌙 Switch to Dark"
    if st.button(dark_label):
        st.session_state.dark = not st.session_state.dark
        st.rerun()

    st.markdown("""
    <div style="font-size:12px;color:#64748b;line-height:1.6;margin-top:12px;">
        Adjust sliders to match your flower's measurements, then hit Predict.
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🌸</div>
    <div class="hero-text">
        <h1>Iris Flower Classification</h1>
        <p>Machine learning · 150 samples · Logistic Regression · 3 species</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metric cards (animated via CSS) ──────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card accent" style="animation-delay:.05s">
        <div class="label">Model Accuracy</div>
        <div class="value">{accuracy*100:.1f}%</div>
        <div class="sub">on held-out test set (20%)</div>
    </div>
    <div class="metric-card" style="animation-delay:.15s">
        <div class="label">Species</div>
        <div class="value">{df["Species"].nunique()}</div>
        <div class="sub">Setosa · Versicolor · Virginica</div>
    </div>
    <div class="metric-card" style="animation-delay:.25s">
        <div class="label">Total Records</div>
        <div class="value">{df.shape[0]}</div>
        <div class="sub">balanced dataset · 50 per class</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Species cards ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Meet the Species</div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="species-grid">
    <div class="species-card" style="animation-delay:.1s">
        <div class="s-emoji">🌺</div>
        <div class="s-name">Iris Setosa</div>
        <div class="s-badge" style="background:#f43f5e;">Setosa</div>
        <div class="s-desc">{SPECIES_DESC['Iris-setosa']}</div>
    </div>
    <div class="species-card" style="animation-delay:.2s">
        <div class="s-emoji">🌿</div>
        <div class="s-name">Iris Versicolor</div>
        <div class="s-badge" style="background:#8b5cf6;">Versicolor</div>
        <div class="s-desc">{SPECIES_DESC['Iris-versicolor']}</div>
    </div>
    <div class="species-card" style="animation-delay:.3s">
        <div class="s-emoji">💜</div>
        <div class="s-name">Iris Virginica</div>
        <div class="s-badge" style="background:#06b6d4;">Virginica</div>
        <div class="s-desc">{SPECIES_DESC['Iris-virginica']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Analyzing measurements…"):
        time.sleep(0.7)

    user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(user_input)[0]
    proba      = model.predict_proba(user_input)[0]
    classes    = model.classes_
    conf       = max(proba) * 100
    emoji      = SPECIES_EMOJI.get(prediction, "🌷")
    color      = COLORS.get(prediction, "#8b5cf6")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)

    res_col, radar_col = st.columns([1, 1.5])

    with res_col:
        st.markdown(f"""
        <div class="result-wrap" style="background:{color}18; border:2px solid {color}55;">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label" style="color:{color};">Predicted Species</div>
            <div class="result-species" style="color:{t['ink']};">
                {prediction.replace("Iris-", "Iris ")}
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{conf:.1f}%;background:{color};"></div>
            </div>
            <div class="result-conf">Confidence: <strong>{conf:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='margin-top:20px;font-weight:600;font-size:14px;color:{t['ink']}'>All Probabilities</div>", unsafe_allow_html=True)
        proba_html = "<div class='proba-row'>"
        for sp, p in zip(classes, proba):
            c     = COLORS.get(sp, "#8b5cf6")
            short = sp.replace("Iris-", "")
            proba_html += f"""
            <div class="proba-item">
                <div class="proba-label">{short}</div>
                <div class="proba-bar-bg">
                    <div class="proba-bar-fill" style="width:{p*100:.1f}%;background:{c};"></div>
                </div>
                <div class="proba-pct">{p*100:.1f}%</div>
            </div>"""
        proba_html += "</div>"
        st.markdown(proba_html, unsafe_allow_html=True)

    with radar_col:
        features      = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        feat_labels   = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        user_vals     = [sepal_length, sepal_width, petal_length, petal_width]
        species_means = df.groupby("Species")[features].mean()

        fig_radar = go.Figure()
        for sp, row in species_means.iterrows():
            vals = list(row) + [row.iloc[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=feat_labels + [feat_labels[0]],
                name=sp.replace("Iris-", ""),
                line=dict(color=COLORS[sp], width=2),
                fill="toself",
                opacity=0.35 if sp == prediction else 0.15,
            ))

        uvals = user_vals + [user_vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=uvals,
            theta=feat_labels + [feat_labels[0]],
            name="⭐ Your Input",
            line=dict(color="#facc15", width=3, dash="dot"),
            fill="toself",
            opacity=0.25,
        ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor=t["plot_bg"],
                radialaxis=dict(visible=True, color=t["muted"], gridcolor=t["border"]),
                angularaxis=dict(color=t["ink"], gridcolor=t["border"]),
            ),
            paper_bgcolor=t["plot_paper"],
            font=dict(family="DM Sans", color=t["plot_font"]),
            title=dict(text="Your Input vs Species Means", font=dict(size=16)),
            legend=dict(orientation="h", y=-0.18),
            margin=dict(t=60, b=50),
            height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.balloons()

# ── About + Features ──────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
col_a, col_gap, col_f = st.columns([2, 0.1, 1.8])

with col_a:
    st.markdown(f"""
    <div class="about-card">
        <h3>About This App</h3>
        <p>
            The classic Iris dataset has measurements of 150 flowers across three species.
            This app uses Logistic Regression to predict species from four simple measurements —
            sepal length, sepal width, petal length, and petal width.
        </p>
        <div class="pill-row">
            <span class="pill">🌺 Setosa</span>
            <span class="pill">🌿 Versicolor</span>
            <span class="pill">💜 Virginica</span>
        </div>
        <div class="algo-badge">⚙️ Algorithm: Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

with col_f:
    feats = [
        ("#f43f5e", "Real-time prediction"),
        ("#8b5cf6", "Confidence score + bars"),
        ("#06b6d4", "Radar chart overlay"),
        ("#f43f5e", "Dark / Light mode toggle"),
        ("#8b5cf6", "Species info cards"),
        ("#06b6d4", "Interactive charts"),
        ("#f43f5e", "Confusion matrix"),
    ]
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;font-size:14px;color:{t["ink"]}">'
        f'<span style="color:{c};font-size:16px;">✦</span>{label}</div>'
        for c, label in feats
    )
    st.markdown(f"""
    <div class="about-card">
        <h3>Features</h3>
        <div class="pill-row" style="flex-direction:column;gap:10px;">{rows}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Data Visualizations</div>", unsafe_allow_html=True)

PT = dict(
    plot_bgcolor=t["plot_bg"],
    paper_bgcolor=t["plot_paper"],
    font_family="DM Sans",
    font_color=t["plot_font"],
)

c1, c2 = st.columns(2)
with c1:
    fig1 = px.scatter(df, x="PetalLengthCm", y="PetalWidthCm",
                      color="Species", size="SepalLengthCm",
                      color_discrete_map=COLORS, title="Petal Length vs Petal Width",
                      labels={"PetalLengthCm":"Petal Length (cm)","PetalWidthCm":"Petal Width (cm)"})
    fig1.update_layout(**PT, title_font_size=15, legend=dict(orientation="h", y=-0.22))
    fig1.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color="white")))
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    sc = df["Species"].value_counts().reset_index()
    sc.columns = ["Species", "Count"]
    fig2 = px.bar(sc, x="Species", y="Count", color="Species",
                  color_discrete_map=COLORS, title="Samples per Species", text="Count")
    fig2.update_traces(textposition="outside", marker_line_width=0)
    fig2.update_layout(**PT, title_font_size=15, showlegend=False,
                       xaxis=dict(tickfont=dict(size=13)))
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    fig3 = px.box(df, x="Species", y="PetalLengthCm", color="Species",
                  color_discrete_map=COLORS, title="Petal Length Distribution",
                  labels={"PetalLengthCm":"Petal Length (cm)"}, points="all")
    fig3.update_traces(marker=dict(opacity=0.5, size=5))
    fig3.update_layout(**PT, title_font_size=15, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    slabels = list(COLORS.keys())
    fig4 = go.Figure(data=go.Heatmap(
        z=cm,
        x=[s.replace("Iris-","") for s in slabels],
        y=[s.replace("Iris-","") for s in slabels],
        colorscale=[[0, t["surface"]], [0.5,"#c084fc"],[1,"#7c3aed"]],
        showscale=False, text=cm, texttemplate="%{text}",
        textfont=dict(size=22, color=t["ink"]),
    ))
    fig4.update_layout(**PT, title="Confusion Matrix", title_font_size=15,
                       xaxis=dict(title="Predicted", side="bottom"),
                       yaxis=dict(title="Actual", autorange="reversed"))
    st.plotly_chart(fig4, use_container_width=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
with st.expander("📁 Explore the Dataset"):
    st.dataframe(df, use_container_width=True, height=320)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<br>
<div style="text-align:center;color:{t['muted']};font-size:13px;padding-bottom:24px;">
    Built with Python · Streamlit · scikit-learn · Plotly &nbsp;|&nbsp; Iris Dataset © UCI ML Repository
</div>
""", unsafe_allow_html=True)