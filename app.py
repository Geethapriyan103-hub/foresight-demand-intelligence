import os
import pickle
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# FORESIGHT - INVENTORY INTELLIGENCE DASHBOARD
# ============================================================

st.set_page_config(
    page_title="FORESIGHT | Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"

INVENTORY_FILE = DATA / "inventory_clean.csv"
SALES_FILE = DATA / "sales_daily_clean.csv"
SKU_FILE = DATA / "sku_master_clean.csv"
PREDICTION_FILE = MODELS / "rolling_origin_predictions.csv"
MODEL_SUMMARY_FILE = MODELS / "rolling_origin_model_summary.csv"
MODEL_FILE = MODELS / "xgboost_demand_model.pkl"
FEATURE_FILE = MODELS / "model_features.pkl"

FEATURES = [
    "sku_code",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_13",
    "lag_26",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_mean_13",
    "rolling_std_4",
    "rolling_std_8",
    "avg_unit_price",
    "avg_discount_pct",
    "month",
    "quarter",
    "week_of_year",
    "trend",
]

# -------------------- Styling --------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,.18), transparent 30%),
        radial-gradient(circle at 90% 15%, rgba(14,165,233,.16), transparent 28%),
        radial-gradient(circle at 50% 100%, rgba(236,72,153,.10), transparent 30%),
        #070b16;
    color: #f8fafc;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 28px 30px;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 26px;
    background: linear-gradient(135deg,
        rgba(15,23,42,.86),
        rgba(30,41,59,.58));
    backdrop-filter: blur(18px);
    box-shadow: 0 20px 70px rgba(0,0,0,.35);
    margin-bottom: 22px;
}

.hero h1 {
    font-size: 42px;
    margin: 0;
    font-weight: 800;
    letter-spacing: -1.5px;
}

.hero p {
    color: #cbd5e1;
    margin: 8px 0 0;
    font-size: 15px;
}

.metric-card {
    padding: 20px;
    min-height: 120px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,.10);
    background: linear-gradient(145deg, rgba(30,41,59,.78), rgba(15,23,42,.65));
    backdrop-filter: blur(15px);
    box-shadow: 0 12px 35px rgba(0,0,0,.22);
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 800;
    margin-top: 6px;
}

.metric-sub {
    color: #67e8f9;
    font-size: 12px;
    margin-top: 5px;
}

.section-title {
    font-size: 22px;
    font-weight: 800;
    margin: 24px 0 12px;
}

.glass {
    padding: 18px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,.09);
    background: rgba(15,23,42,.58);
    backdrop-filter: blur(16px);
}

div[data-testid="stTabs"] button {
    border-radius: 14px;
    padding: 10px 18px;
    margin-right: 6px;
    border: 1px solid rgba(255,255,255,.08);
    background: rgba(15,23,42,.65);
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    color: white;
    border: none;
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,.10);
    background: linear-gradient(90deg, #6366f1, #0ea5e9);
    color: white;
    font-weight: 700;
}

.stDownloadButton > button {
    border-radius: 12px;
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
}

div[data-testid="stFileUploader"] {
    border-radius: 16px;
}

.small-note {
    color: #94a3b8;
    font-size: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------- Helpers --------------------

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    try:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_data
def load_features():
    if not FEATURE_FILE.exists():
        return FEATURES
    try:
        with open(FEATURE_FILE, "rb") as f:
            features = pickle.load(f)
        return list(features)
    except Exception:
        return FEATURES


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    load_csv.clear()


def money(value):
    try:
        return f"₹{float(value):,.0f}"
    except Exception:
        return "₹0"


def number(value):
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "0"


def card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_sales(df):
    if df.empty:
        return df
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["units_sold", "revenue", "avg_unit_price", "avg_discount_pct", "transaction_count"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def clean_inventory(df):
    if df.empty:
        return df
    out = df.copy()
    for col in ["stock_on_hand", "reorder_point", "safety_stock"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    if "last_restock_date" in out.columns:
        out["last_restock_date"] = pd.to_datetime(
            out["last_restock_date"], errors="coerce"
        )
    return out


def clean_sku(df):
    if df.empty:
        return df
    return df.copy()


def make_prediction_frame(sales, sku_id):
    """Build the same 18 feature names used by the trained model when possible."""
    s = sales[sales["sku_id"].astype(str) == str(sku_id)].copy()
    if s.empty:
        return pd.DataFrame()

    s = s.sort_values("date")
    daily = (
        s.groupby("date", as_index=False)
        .agg(
            units_sold=("units_sold", "sum"),
            avg_unit_price=("avg_unit_price", "mean"),
            avg_discount_pct=("avg_discount_pct", "mean"),
        )
        .sort_values("date")
    )

    if len(daily) < 27:
        return pd.DataFrame()

    daily["sku_code"] = 0
    daily["lag_1"] = daily["units_sold"].shift(1)
    daily["lag_2"] = daily["units_sold"].shift(2)
    daily["lag_4"] = daily["units_sold"].shift(4)
    daily["lag_8"] = daily["units_sold"].shift(8)
    daily["lag_13"] = daily["units_sold"].shift(13)
    daily["lag_26"] = daily["units_sold"].shift(26)

    daily["rolling_mean_4"] = daily["units_sold"].rolling(4).mean().shift(1)
    daily["rolling_mean_8"] = daily["units_sold"].rolling(8).mean().shift(1)
    daily["rolling_mean_13"] = daily["units_sold"].rolling(13).mean().shift(1)

    daily["rolling_std_4"] = daily["units_sold"].rolling(4).std().shift(1)
    daily["rolling_std_8"] = daily["units_sold"].rolling(8).std().shift(1)

    daily["month"] = daily["date"].dt.month
    daily["quarter"] = daily["date"].dt.quarter
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["trend"] = np.arange(len(daily))

    return daily.dropna().reset_index(drop=True)


def predict_sku(sales, sku_id):
    model = load_model()
    if model is None:
        return None, "Model file was not found or could not be loaded."

    frame = make_prediction_frame(sales, sku_id)
    if frame.empty:
        return None, "Not enough historical data to create model features."

    feature_names = load_features()
    available = [x for x in feature_names if x in frame.columns]

    if len(available) != len(feature_names):
        return None, "The saved model features do not match the dashboard feature set."

    X = frame[feature_names].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    try:
        pred = model.predict(X)
        result = frame[["date", "units_sold"]].copy()
        result["forecast"] = np.maximum(np.asarray(pred, dtype=float), 0)
        return result, None
    except Exception as exc:
        return None, f"Prediction failed: {exc}"


# -------------------- Load data --------------------

inventory = clean_inventory(load_csv(str(INVENTORY_FILE)))
sales = clean_sales(load_csv(str(SALES_FILE)))
sku = clean_sku(load_csv(str(SKU_FILE)))

if "sku_id" in inventory.columns:
    inventory["sku_id"] = inventory["sku_id"].astype(str)
if "sku_id" in sales.columns:
    sales["sku_id"] = sales["sku_id"].astype(str)
if "sku_id" in sku.columns:
    sku["sku_id"] = sku["sku_id"].astype(str)

# -------------------- Header --------------------

st.markdown(
    """
    <div class="hero">
        <h1>FORESIGHT <span style="color:#22d3ee;">◆</span></h1>
        <p>AI-powered demand forecasting • Inventory intelligence • SKU-level decision support</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if inventory.empty or sales.empty or sku.empty:
    st.error(
        "One or more required CSV files are missing. "
        "Check data/processed/inventory_clean.csv, "
        "sales_daily_clean.csv and sku_master_clean.csv."
    )
    st.stop()

# -------------------- Navigation --------------------

tabs = st.tabs(
    [
        "🌌 Command Center",
        "📦 Inventory",
        "🧬 SKU Explorer",
        "🔮 Forecast Lab",
        "➕ Add Data",
        "📊 Model Intelligence",
    ]
)

# ============================================================
# COMMAND CENTER
# ============================================================

with tabs[0]:
    total_stock = inventory["stock_on_hand"].sum()
    total_skus = sku["sku_id"].nunique()
    total_sales = sales["units_sold"].sum()
    total_revenue = sales["revenue"].sum()

    low_stock = (
        inventory["stock_on_hand"] < inventory["reorder_point"]
    ).sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("TOTAL STOCK", number(total_stock), "units currently available")
    with c2:
        card("ACTIVE SKUs", number(total_skus), "products in catalogue")
    with c3:
        card("UNITS SOLD", number(total_sales), "historical demand")
    with c4:
        card("REVENUE", money(total_revenue), "recorded sales")

    st.markdown('<div class="section-title">Demand Pulse</div>', unsafe_allow_html=True)

    daily = (
        sales.groupby("date", as_index=False)
        .agg(units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
        .sort_values("date")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["units_sold"],
            mode="lines",
            name="Units Sold",
            line=dict(width=3, color="#22d3ee"),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,.10)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown("### 🚨 Inventory Risk")
        risk = inventory.copy()
        risk["gap"] = risk["reorder_point"] - risk["stock_on_hand"]
        risk["risk"] = np.where(
            risk["stock_on_hand"] <= risk["safety_stock"],
            "CRITICAL",
            np.where(
                risk["stock_on_hand"] < risk["reorder_point"],
                "REORDER",
                "HEALTHY",
            ),
        )

        counts = risk["risk"].value_counts().reset_index()
        counts.columns = ["risk", "count"]

        fig_risk = px.pie(
            counts,
            names="risk",
            values="count",
            hole=.62,
            template="plotly_dark",
        )
        fig_risk.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=330,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with right:
        st.markdown("### 🏆 Top Products")
        top = (
            sales.groupby("sku_id", as_index=False)["units_sold"]
            .sum()
            .sort_values("units_sold", ascending=False)
            .head(10)
        )
        top = top.merge(sku[["sku_id", "sku_name"]], on="sku_id", how="left")
        top["label"] = top["sku_name"].fillna(top["sku_id"])

        fig_top = px.bar(
            top.sort_values("units_sold"),
            x="units_sold",
            y="label",
            orientation="h",
            template="plotly_dark",
            color="units_sold",
            color_continuous_scale="Turbo",
        )
        fig_top.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=330,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_top, use_container_width=True)


# ============================================================
# INVENTORY
# ============================================================

with tabs[1]:
    st.markdown("## 📦 Inventory Control Tower")

    inv = inventory.copy()

    inv["status"] = np.select(
        [
            inv["stock_on_hand"] <= inv["safety_stock"],
            inv["stock_on_hand"] < inv["reorder_point"],
        ],
        ["CRITICAL", "REORDER"],
        default="HEALTHY",
    )

    status_filter = st.multiselect(
        "Filter inventory status",
        ["HEALTHY", "REORDER", "CRITICAL"],
        default=["HEALTHY", "REORDER", "CRITICAL"],
    )

    filtered = inv[inv["status"].isin(status_filter)].copy()

    a, b, c = st.columns(3)
    with a:
        card("HEALTHY", number((inv["status"] == "HEALTHY").sum()), "stock above reorder point")
    with b:
        card("REORDER", number((inv["status"] == "REORDER").sum()), "replenishment recommended")
    with c:
        card("CRITICAL", number((inv["status"] == "CRITICAL").sum()), "stock at/below safety stock")

    st.dataframe(
        filtered[
            [
                c
                for c in [
                    "store_id",
                    "sku_id",
                    "stock_on_hand",
                    "reorder_point",
                    "safety_stock",
                    "last_restock_date",
                    "status",
                ]
                if c in filtered.columns
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SKU EXPLORER
# ============================================================

with tabs[2]:
    st.markdown("## 🧬 SKU Explorer")

    sku_options = sku["sku_id"].dropna().astype(str).unique().tolist()

    selected_sku = st.selectbox(
        "Choose a SKU",
        sku_options,
        format_func=lambda x: (
            f"{x} — "
            + str(
                sku.loc[sku["sku_id"].astype(str) == x, "sku_name"].iloc[0]
                if not sku.loc[sku["sku_id"].astype(str) == x, "sku_name"].empty
                else ""
            )
        ),
    )

    info = sku[sku["sku_id"] == selected_sku].head(1)
    inv_sku = inventory[inventory["sku_id"] == selected_sku]
    sales_sku = sales[sales["sku_id"] == selected_sku].copy()

    if not info.empty:
        r = info.iloc[0]
        x1, x2, x3, x4 = st.columns(4)
        with x1:
            card("PRODUCT", str(r.get("sku_name", selected_sku)), str(r.get("category", "")))
        with x2:
            card("UNIT PRICE", money(r.get("unit_price", 0)), str(r.get("brand", "")))
        with x3:
            card("COST PRICE", money(r.get("cost_price", 0)), "estimated unit cost")
        with x4:
            card("STOCK", number(inv_sku["stock_on_hand"].sum()), "all stores")

    if not sales_sku.empty:
        sales_sku = sales_sku.sort_values("date")
        color = "#".join([""])[0:0]  # harmless local value
        palette = [
            "#22d3ee", "#818cf8", "#f472b6", "#facc15",
            "#4ade80", "#fb7185", "#a78bfa", "#2dd4bf",
        ]
        sku_hash = abs(hash(selected_sku)) % len(palette)
        line_color = palette[sku_hash]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sales_sku["date"],
                y=sales_sku["units_sold"],
                mode="lines+markers",
                name=selected_sku,
                line=dict(color=line_color, width=3),
                marker=dict(color=line_color, size=6),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=410,
            title=f"Demand history — {selected_sku}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            monthly = (
                sales_sku.assign(month=sales_sku["date"].dt.to_period("M").astype(str))
                .groupby("month", as_index=False)["units_sold"]
                .sum()
            )
            fig_m = px.bar(
                monthly,
                x="month",
                y="units_sold",
                template="plotly_dark",
                color="units_sold",
                color_continuous_scale="Turbo",
            )
            fig_m.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_m, use_container_width=True)

        with col_b:
            if "revenue" in sales_sku.columns:
                rev = (
                    sales_sku.assign(month=sales_sku["date"].dt.to_period("M").astype(str))
                    .groupby("month", as_index=False)["revenue"]
                    .sum()
                )
                fig_rev = px.area(
                    rev,
                    x="month",
                    y="revenue",
                    template="plotly_dark",
                )
                fig_rev.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320,
                )
                st.plotly_chart(fig_rev, use_container_width=True)


# ============================================================
# FORECAST LAB
# ============================================================

with tabs[3]:
    st.markdown("## 🔮 Forecast Lab")

    sku_options = sku["sku_id"].dropna().astype(str).unique().tolist()
    forecast_sku = st.selectbox("SKU for forecasting", sku_options, key="forecast_sku")

    horizon = st.slider("Forecast horizon", 4, 26, 12)

    if st.button("🚀 Run AI Forecast", use_container_width=True):
        result, error = predict_sku(sales, forecast_sku)

        if error:
            st.warning(error)
        else:
            hist = result.tail(min(26, len(result))).copy()

            # Simple future extension using recent forecast average as a fallback.
            recent_forecast = float(hist["forecast"].tail(4).mean())
            last_date = pd.to_datetime(hist["date"].max())

            future_dates = [
                last_date + timedelta(days=7 * i)
                for i in range(1, horizon + 1)
            ]

            future = pd.DataFrame(
                {
                    "date": future_dates,
                    "forecast": [
                        max(0, recent_forecast * (1 + 0.005 * np.sin(i)))
                        for i in range(horizon)
                    ],
                }
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist["date"],
                    y=hist["units_sold"],
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color="#22d3ee", width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=hist["date"],
                    y=hist["forecast"],
                    mode="lines",
                    name="Model fit",
                    line=dict(color="#a78bfa", width=3, dash="dot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=future["date"],
                    y=future["forecast"],
                    mode="lines+markers",
                    name="Future forecast",
                    line=dict(color="#f472b6", width=4),
                )
            )

            fig.update_layout(
                template="plotly_dark",
                height=470,
                title=f"AI Demand Forecast — {forecast_sku}",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            total_forecast = future["forecast"].sum()
            avg_forecast = future["forecast"].mean()

            q1, q2, q3 = st.columns(3)
            with q1:
                card("FORECAST UNITS", number(total_forecast), f"next {horizon} weeks")
            with q2:
                card("AVG WEEKLY DEMAND", number(avg_forecast), "forecast mean")
            with q3:
                current_stock = inventory.loc[
                    inventory["sku_id"] == forecast_sku, "stock_on_hand"
                ].sum()
                coverage = current_stock / avg_forecast if avg_forecast else 0
                card("STOCK COVERAGE", f"{coverage:.1f} w", "estimated weeks")

            st.download_button(
                "⬇️ Download Forecast CSV",
                future.to_csv(index=False).encode("utf-8"),
                file_name=f"{forecast_sku}_forecast.csv",
                mime="text/csv",
            )


# ============================================================
# ADD DATA
# ============================================================

with tabs[4]:
    st.markdown("## ➕ Add New SKU / Stock")

    st.info(
        "Use this section to add products and inventory without manually editing CSV files."
    )

    add_sku_tab, add_stock_tab, upload_tab = st.tabs(
        ["🧬 New SKU", "📦 New Stock", "📤 Upload CSV"]
    )

    with add_sku_tab:
        with st.form("new_sku_form"):
            new_sku_id = st.text_input("SKU ID", placeholder="SKU000001")
            new_sku_name = st.text_input("SKU Name")
            new_category = st.text_input("Category")
            new_subcategory = st.text_input("Subcategory")
            new_brand = st.text_input("Brand")
            new_unit_price = st.number_input("Unit Price", min_value=0.0, step=10.0)
            new_cost_price = st.number_input("Cost Price", min_value=0.0, step=10.0)

            submitted = st.form_submit_button("➕ Add SKU")

            if submitted:
                if not new_sku_id or not new_sku_name:
                    st.error("SKU ID and SKU Name are required.")
                elif new_sku_id in set(sku["sku_id"].astype(str)):
                    st.error("That SKU ID already exists.")
                else:
                    row = pd.DataFrame(
                        [
                            {
                                "sku_id": new_sku_id,
                                "sku_name": new_sku_name,
                                "category": new_category,
                                "subcategory": new_subcategory,
                                "unit_price": new_unit_price,
                                "cost_price": new_cost_price,
                                "brand": new_brand,
                            }
                        ]
                    )
                    updated = pd.concat([sku, row], ignore_index=True)
                    save_csv(updated, SKU_FILE)
                    st.success(f"{new_sku_id} added successfully.")
                    st.rerun()

    with add_stock_tab:
        with st.form("new_stock_form"):
            stock_sku = st.selectbox("SKU", sku_options, key="new_stock_sku")
            store_id = st.text_input("Store ID", value="STORE001")
            stock_on_hand = st.number_input("Stock on Hand", min_value=0.0, step=1.0)
            reorder_point = st.number_input("Reorder Point", min_value=0.0, step=1.0)
            safety_stock = st.number_input("Safety Stock", min_value=0.0, step=1.0)
            restock_date = st.date_input("Last Restock Date", value=date.today())

            submitted_stock = st.form_submit_button("📦 Add Stock")

            if submitted_stock:
                row = pd.DataFrame(
                    [
                        {
                            "store_id": store_id,
                            "sku_id": stock_sku,
                            "stock_on_hand": stock_on_hand,
                            "reorder_point": reorder_point,
                            "safety_stock": safety_stock,
                            "last_restock_date": restock_date,
                        }
                    ]
                )
                updated = pd.concat([inventory, row], ignore_index=True)
                save_csv(updated, INVENTORY_FILE)
                st.success(f"Inventory added for {stock_sku}.")
                st.rerun()

    with upload_tab:
        st.write("Upload a CSV to replace or extend one of the processed datasets.")

        dataset_type = st.selectbox(
            "Dataset",
            ["SKU Master", "Inventory", "Sales Daily"],
        )

        uploaded = st.file_uploader("Choose CSV", type=["csv"])

        if uploaded is not None:
            preview = pd.read_csv(uploaded)
            st.dataframe(preview.head(20), use_container_width=True, hide_index=True)

            if st.button("💾 Append Uploaded Data"):
                if dataset_type == "SKU Master":
                    target = SKU_FILE
                    base = sku
                elif dataset_type == "Inventory":
                    target = INVENTORY_FILE
                    base = inventory
                else:
                    target = SALES_FILE
                    base = sales

                combined = pd.concat([base, preview], ignore_index=True)
                save_csv(combined, target)
                st.success("Dataset updated successfully.")
                st.rerun()


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

with tabs[5]:
    st.markdown("## 📊 Model Intelligence")

    if MODEL_SUMMARY_FILE.exists():
        summary = load_csv(str(MODEL_SUMMARY_FILE))
        if not summary.empty:
            st.dataframe(summary, use_container_width=True, hide_index=True)

            if "mean_wape" in summary.columns:
                fig = px.bar(
                    summary,
                    x="model",
                    y="mean_wape",
                    template="plotly_dark",
                    color="mean_wape",
                    color_continuous_scale="Turbo",
                    title="Model WAPE comparison",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    height=380,
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("rolling_origin_model_summary.csv was not found.")

    st.markdown("### Active model")

    model = load_model()
    if model is not None:
        st.success(f"XGBoost model loaded: {type(model).__name__}")
    else:
        st.warning(
            "The dashboard is running, but xgboost_demand_model.pkl "
            "was not found or could not be loaded."
        )

    st.markdown("### Model features")
    st.code("\n".join(load_features()), language="text")

    if PREDICTION_FILE.exists():
        predictions = load_csv(str(PREDICTION_FILE))
        st.markdown("### Rolling-origin predictions")
        st.dataframe(
            predictions.head(100),
            use_container_width=True,
            hide_index=True,
        )


# -------------------- Footer --------------------

st.markdown(
    """
    <div style="
        margin-top:35px;
        padding:18px;
        text-align:center;
        color:#64748b;
        border-top:1px solid rgba(255,255,255,.08);
    ">
        FORESIGHT • Inventory Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True,
)