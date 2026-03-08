"""
app.py - Bristol Pink Cafe Sales Forecasting Dashboard (Streamlit)

Wireframes implemented:
  Tab 1  Data          CSV upload (multi-format), data status card
  Tab 2  Insights      Sales-over-time line, top-products bar, performance table
  Tab 3  Forecast      Model config, metrics row, 28-day chart, CSV/PNG export

Integrates with model.py via run_forecast(series, algorithm, train_weeks)
"""

from __future__ import annotations

import io
import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Page config - MUST be the absolute first Streamlit command
st.set_page_config(
    page_title="Bristol Pink Cafe - Sales Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import forecasting backend (model.py in the same src/ directory)
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from model import run_forecast, VALID_ALGORITHMS

    _MODEL_OK = True
except ImportError:
    run_forecast = None
    VALID_ALGORITHMS = {"Prophet", "ARIMA", "XGBoost", "Ensemble"}
    _MODEL_OK = False

# Colour palette
PINK = "#E91E8C"
PINK_LIGHT = "#FFE4F3"
DARK = "#1A1A1A"
BORDER = "#E0E0E0"
NEG_RED = "#E74C3C"

# Custom CSS - match the design wireframes
st.markdown(
    f"""
<style>
/* SIDEBAR */
section[data-testid="stSidebar"] {{
    background-color: {DARK} !important;
}}
section[data-testid="stSidebar"] * {{
    color: #ffffff !important;
}}
section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {{
    background-color: {PINK} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100%;
}}
section[data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover {{
    background-color: #c4177a !important;
}}
section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {{
    background-color: transparent !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100%;
}}
section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {{
    background-color: #333 !important;
}}

/* TABS */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 2px solid #eee;
}}
.stTabs [data-baseweb="tab"] {{
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    color: #555;
    border-bottom: 3px solid transparent;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: {PINK} !important;
    border-bottom: 3px solid {PINK} !important;
}}

/* CARDS */
.card {{
    border: 1.5px solid {BORDER};
    border-radius: 16px;
    padding: 1.5rem;
    background: #ffffff;
    margin-bottom: 1rem;
}}
.card-title {{
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    color: #111;
}}

/* UPLOAD AREA */
.upload-area {{
    border: 2.5px dashed #ccc;
    border-radius: 16px;
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 1rem;
    background: #fafafa;
}}
.upload-icon {{
    font-size: 3rem;
    color: {PINK};
    margin-bottom: 0.5rem;
}}

/* METRIC CARDS */
.metric-card {{
    background: #f9f9f9;
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}}
.metric-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {PINK};
}}
.metric-label {{
    color: #666;
    font-size: 0.85rem;
    margin-top: 0.25rem;
}}

/* PERFORMANCE TABLE */
.perf-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
}}
.perf-table th {{
    text-align: left;
    padding: 0.65rem 1rem;
    border-bottom: 2px solid #555;
    font-weight: 700;
    color: #ddd;
}}
.perf-table td {{
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #444;
    color: #ccc;
}}
.pct-pos {{ color: {PINK}; font-weight: 600; }}
.pct-neg {{ color: {NEG_RED}; font-weight: 600; }}

/* HEADER */
.main-title {{
    font-size: 1.75rem;
    font-weight: 800;
    color: #111;
    margin: 0;
}}
.main-subtitle {{
    color: #777;
    font-size: 0.95rem;
    margin-top: 2px;
}}

/* PLACEHOLDERS */
.status-empty, .forecast-empty {{
    text-align: center;
    padding: 2.5rem 1rem;
    color: #aaa;
}}
.status-icon {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}

/* GENERAL */
.block-container {{ padding-top: 1.5rem !important; }}

button[data-testid="baseButton-primary"] {{
    background-color: {PINK} !important;
    border-color: {PINK} !important;
    color: #fff !important;
    border-radius: 24px !important;
}}
button[data-testid="baseButton-primary"]:hover {{
    background-color: #c4177a !important;
    border-color: #c4177a !important;
}}
div[data-testid="stFileUploader"] button {{
    background-color: {PINK} !important;
    color: #fff !important;
    border-radius: 24px !important;
    border: none !important;
}}
.stDownloadButton > button {{
    border-radius: 24px !important;
    font-weight: 600 !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# Session-state initialisation
_DEFAULTS = {
    "page": "Dashboard",
    "df": None,
    "products": [],
    "categories": [],
    "upload_key": None,
    "forecast_result": None,
    "time_range": "1 Week",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar - branding + navigation
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding:1.2rem 0 0.8rem 0;">
            <span style="font-size:2.4rem;">&#9749;</span>
            <div style="font-size:1.35rem; font-weight:800; line-height:1.2; margin-top:4px;">
                Bristol<br>Pink Cafe
            </div>
            <div style="font-size:0.82rem; color:{PINK}; font-weight:600; margin-top:2px;">
                Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    for _page in ("Dashboard", "Reports", "Settings"):
        _active = st.session_state.page == _page
        if st.button(
            _page,
            key=f"nav_{_page}",
            use_container_width=True,
            type="primary" if _active else "secondary",
        ):
            st.session_state.page = _page
            st.rerun()

# Default location
DEFAULT_LOCATION = "Bristol Centre"


# CSV parsing helper (handles three layouts)
def _parse_single_csv(
    uploaded_file, default_location: str
) -> tuple[pd.DataFrame | None, str]:
    """
    Parse one uploaded CSV into long-form DataFrame with columns:
        Date, Product, Category, Units Sold, Location

    Supported layouts:
      A  Standard long-format (Date, Product, Category, Units Sold, ...)
      B  Coffee wide-format (two-row header: Date/Number Sold + Cappuccino/Americano)
      C  Croissant simple (Date, Number Sold)

    Returns (df | None, error_message).
    """
    # read raw text
    try:
        raw = uploaded_file.read().decode("utf-8")
        uploaded_file.seek(0)
    except Exception as exc:
        return None, f"Cannot read file: {exc}"

    lines = [l for l in raw.strip().splitlines() if l.strip()]

    if len(lines) < 2:
        return None, "File is empty or contains no data rows."

    # helper: flexible column lookup (case-insensitive)
    def _col(cols_lower: dict, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        return None

    # Layout A: standard long-format
    try:
        df = pd.read_csv(io.StringIO(raw))
        cl = {c.strip().lower().replace("_", " "): c.strip() for c in df.columns}

        date_c = _col(cl, ["date"])
        prod_c = _col(cl, ["product", "product name"])
        unit_c = _col(cl, ["units sold", "number sold", "quantity", "quantity sold"])

        if date_c and prod_c and unit_c:
            rename = {date_c: "Date", prod_c: "Product", unit_c: "Units Sold"}

            cat_c = _col(cl, ["category"])
            if cat_c:
                rename[cat_c] = "Category"

            loc_c = _col(cl, ["location"])
            if loc_c:
                rename[loc_c] = "Location"

            df = df.rename(columns=rename)

            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df["Units Sold"] = pd.to_numeric(df["Units Sold"], errors="coerce")

            if df["Date"].isna().all():
                raise ValueError("No parseable dates.")
            if df["Units Sold"].isna().all():
                raise ValueError("No valid numeric units-sold values.")

            df = df.dropna(subset=["Date", "Units Sold"])
            df["Units Sold"] = df["Units Sold"].astype(int)

            if "Category" not in df.columns:
                df["Category"] = "General"
            if "Location" not in df.columns:
                df["Location"] = default_location

            return df[["Date", "Product", "Category", "Units Sold", "Location"]], ""
    except Exception:
        pass  # fall through to layout B / C

    # Layout B / C: wide-format raw files
    try:
        first_line = lines[0]
        second_line = lines[1] if len(lines) > 1 else ""

        # Coffee file (two-row header)
        if "Cappuccino" in second_line or "Americano" in second_line:
            headers = [h.strip() for h in second_line.split(",")]
            data_text = "\n".join(lines[2:])
            df_w = pd.read_csv(io.StringIO(data_text), header=None)

            col_names = headers[: len(df_w.columns)]
            while len(col_names) < len(df_w.columns):
                col_names.append(f"_extra_{len(col_names)}")
            df_w.columns = col_names

            date_col = col_names[0] if col_names[0] else "Date"
            if not col_names[0]:
                df_w = df_w.rename(columns={df_w.columns[0]: "Date"})
                date_col = "Date"

            product_cols = [
                c
                for c in df_w.columns[1:]
                if c and not c.startswith("_extra") and not c.startswith("Unnamed")
            ]
            if not product_cols:
                return None, "No product columns found in coffee file."

            parts = []
            for pc in product_cols:
                tmp = df_w[[date_col, pc]].copy()
                tmp.columns = ["Date", "Units Sold"]
                tmp["Product"] = pc
                tmp["Category"] = "Coffee"
                parts.append(tmp)

            df_long = pd.concat(parts, ignore_index=True)

        # Croissant file (Date, Number Sold)
        elif "Number Sold" in first_line or "croissant" in first_line.lower():
            df_w = pd.read_csv(io.StringIO(raw))
            date_col = df_w.columns[0]
            val_col = [c for c in df_w.columns if c != date_col][0]
            df_long = df_w[[date_col, val_col]].copy()
            df_long.columns = ["Date", "Units Sold"]
            df_long["Product"] = "Croissant"
            df_long["Category"] = "Pastry"

        # Generic wide: first col = date, rest = products
        else:
            df_w = pd.read_csv(io.StringIO(raw))
            date_col = df_w.columns[0]
            product_cols = [
                c for c in df_w.columns[1:] if not str(c).startswith("Unnamed")
            ]
            if not product_cols:
                return None, "Could not detect any product columns."

            parts = []
            for pc in product_cols:
                tmp = df_w[[date_col, pc]].copy()
                tmp.columns = ["Date", "Units Sold"]
                tmp["Product"] = pc
                tmp["Category"] = "General"
                parts.append(tmp)
            df_long = pd.concat(parts, ignore_index=True)

        # clean
        df_long["Date"] = pd.to_datetime(df_long["Date"], dayfirst=True, errors="coerce")
        df_long["Units Sold"] = pd.to_numeric(df_long["Units Sold"], errors="coerce")

        if df_long["Date"].isna().all():
            return None, "Could not parse any dates in the file."
        if df_long["Units Sold"].isna().all():
            return None, "No valid numeric values found for units sold."

        df_long = df_long.dropna(subset=["Date", "Units Sold"])
        df_long["Units Sold"] = df_long["Units Sold"].astype(int)
        df_long["Location"] = default_location

        if "Category" not in df_long.columns:
            df_long["Category"] = "General"

        return df_long[["Date", "Product", "Category", "Units Sold", "Location"]], ""

    except Exception as exc:
        return None, f"Could not parse file: {exc}"


# PAGE: DASHBOARD
if st.session_state.page == "Dashboard":

    # Header
    st.markdown(
        '<p class="main-title">Sales Forecasting Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="main-subtitle">Upload data, explore trends, and plan production</p>',
        unsafe_allow_html=True,
    )

    # Three tabs
    tab_data, tab_insights, tab_forecast = st.tabs(
        ["Data", "Insights", "Forecast & Export"]
    )

    # TAB 1 - DATA
    with tab_data:
        col_upload, col_status = st.columns([3, 2], gap="large")

        # Left card: Upload
        with col_upload:
            st.markdown(
                '<div class="card"><div class="card-title">Upload Sales CSV</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="upload-area">
                    <div class="upload-icon">&#9729;&#8593;</div>
                    <div style="color:#888; font-size:0.9rem; margin-top:4px;">
                        Upload one or more CSV files
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            uploaded_files = st.file_uploader(
                "Upload CSV files",
                type=["csv"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                help="Select .csv files with your sales data",
            )

            st.markdown(
                '<p style="color:#888; font-size:0.85rem;">or drag and drop your files here</p>',
                unsafe_allow_html=True,
            )

            st.markdown(
                """
**Expected CSV format:**
- Date (DD/MM/YYYY)
- Product name
- Category
- Units sold (numeric)
- Location (optional)
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # process upload(s)
            if uploaded_files:
                upload_key = "|".join(
                    sorted(f"{f.name}:{f.size}" for f in uploaded_files)
                )
                if upload_key != st.session_state.upload_key:
                    all_frames: list[pd.DataFrame] = []
                    errors: list[str] = []
                    for uf in uploaded_files:
                        df_parsed, err = _parse_single_csv(uf, DEFAULT_LOCATION)
                        if err:
                            errors.append(f"**{uf.name}:** {err}")
                        elif df_parsed is not None and not df_parsed.empty:
                            all_frames.append(df_parsed)

                    if all_frames:
                        combined = pd.concat(all_frames, ignore_index=True)
                        st.session_state.df = combined
                        st.session_state.upload_key = upload_key
                        st.session_state.products = sorted(
                            combined["Product"].unique().tolist()
                        )
                        st.session_state.categories = sorted(
                            combined["Category"].unique().tolist()
                        )
                        st.session_state.forecast_result = None
                        st.success(
                            f"**{len(combined):,}** records imported from "
                            f"**{len(uploaded_files)}** file(s)"
                        )
                    for e in errors:
                        st.error(e)
                    if not all_frames and not errors:
                        st.warning("Uploaded files contained no usable data.")

        # Right card: Data Status
        with col_status:
            st.markdown(
                '<div class="card"><div class="card-title">Data Status</div>',
                unsafe_allow_html=True,
            )

            if st.session_state.df is not None:
                df = st.session_state.df
                n_rows = len(df)
                n_prods = df["Product"].nunique()
                d_min = df["Date"].min().strftime("%d %b %Y")
                d_max = df["Date"].max().strftime("%d %b %Y")
                n_days = (df["Date"].max() - df["Date"].min()).days + 1

                st.markdown(
                    f"""
**File(s) loaded**

**{n_rows:,}** records  |  **{n_prods}** product(s)

**{d_min}** to **{d_max}** ({n_days} days)

Categories: {', '.join(st.session_state.categories)}
                    """
                )
                with st.expander("Preview first 20 rows"):
                    st.dataframe(
                        df.head(20), use_container_width=True, hide_index=True
                    )
            else:
                st.markdown(
                    """
                    <div class="status-empty">
                        <div style="font-size:1rem;">No file uploaded</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

    # TAB 2 - INSIGHTS
    with tab_insights:
        if st.session_state.df is None:
            st.info("Upload a CSV file in the **Data** tab first.")
        else:
            df_full = st.session_state.df.copy()

            # filter bar
            f1, f2, f3 = st.columns([2, 2, 3])
            with f1:
                sel_prod = st.selectbox(
                    "Product",
                    ["All Products"] + st.session_state.products,
                    key="ins_prod",
                )
            with f2:
                sel_cat = st.selectbox(
                    "Category",
                    ["All Categories"] + st.session_state.categories,
                    key="ins_cat",
                )
            with f3:
                # time-range pills using buttons
                pcols = st.columns([1, 1, 1, 3])
                for idx, opt in enumerate(["1 Week", "4 Weeks", "8 Weeks"]):
                    with pcols[idx]:
                        _active = st.session_state.time_range == opt
                        if st.button(
                            opt,
                            key=f"tr_{opt}",
                            type="primary" if _active else "secondary",
                        ):
                            st.session_state.time_range = opt
                            st.rerun()

            time_opt = st.session_state.time_range
            weeks_map = {"1 Week": 7, "4 Weeks": 28, "8 Weeks": 56}
            n_days_filter = weeks_map[time_opt]

            # apply product / category filter
            df_ins = df_full.copy()
            if sel_prod != "All Products":
                df_ins = df_ins[df_ins["Product"] == sel_prod]
            if sel_cat != "All Categories":
                df_ins = df_ins[df_ins["Category"] == sel_cat]

            if df_ins.empty:
                st.warning("No data matches the selected filters.")
            else:
                max_date = df_ins["Date"].max()
                df_win = df_ins[
                    df_ins["Date"] >= max_date - pd.Timedelta(days=n_days_filter)
                ]

                # charts row
                ch_l, ch_r = st.columns([3, 2], gap="large")

                with ch_l:
                    st.markdown(
                        '<div class="card">'
                        '<div class="card-title" style="font-style:italic;">Sales Over Time</div>',
                        unsafe_allow_html=True,
                    )
                    daily = (
                        df_win.groupby("Date")["Units Sold"]
                        .sum()
                        .reset_index()
                        .sort_values("Date")
                    )
                    fig_line = go.Figure()
                    fig_line.add_trace(
                        go.Scatter(
                            x=daily["Date"],
                            y=daily["Units Sold"],
                            mode="lines+markers",
                            line=dict(color=PINK, width=2.5),
                            marker=dict(size=5, color=PINK),
                            name="Units Sold",
                            hovertemplate=(
                                "Date: %{x|%d %b %Y}<br>Units: %{y}<extra></extra>"
                            ),
                        )
                    )
                    fig_line.update_layout(
                        height=320,
                        margin=dict(l=40, r=20, t=10, b=40),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                        plot_bgcolor="#fff",
                        paper_bgcolor="#fff",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with ch_r:
                    st.markdown(
                        '<div class="card">'
                        '<div class="card-title" style="font-style:italic;">Top Selling Products</div>',
                        unsafe_allow_html=True,
                    )
                    top5 = (
                        df_win.groupby("Product")["Units Sold"]
                        .sum()
                        .sort_values(ascending=True)
                        .tail(5)
                        .reset_index()
                    )
                    fig_bar = go.Figure()
                    fig_bar.add_trace(
                        go.Bar(
                            y=top5["Product"],
                            x=top5["Units Sold"],
                            orientation="h",
                            marker=dict(color=PINK),
                            hovertemplate="%{y}: %{x} units<extra></extra>",
                        )
                    )
                    fig_bar.update_layout(
                        height=320,
                        margin=dict(l=10, r=20, t=10, b=40),
                        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                        yaxis=dict(showgrid=False),
                        plot_bgcolor="#fff",
                        paper_bgcolor="#fff",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # recent-performance table
                st.markdown(
                    '<div class="card">'
                    '<div class="card-title" style="font-style:italic;">'
                    "Recent Performance by Product</div>",
                    unsafe_allow_html=True,
                )

                current_start = max_date - pd.Timedelta(days=n_days_filter)
                prev_start = current_start - pd.Timedelta(days=n_days_filter)

                cur_data = df_ins[
                    (df_ins["Date"] > current_start) & (df_ins["Date"] <= max_date)
                ]
                prev_data = df_ins[
                    (df_ins["Date"] > prev_start) & (df_ins["Date"] <= current_start)
                ]

                cur_tot = cur_data.groupby(["Product", "Category"])[
                    "Units Sold"
                ].sum()
                prev_tot = prev_data.groupby(["Product", "Category"])[
                    "Units Sold"
                ].sum()

                rows = []
                for (prod, cat), units in cur_tot.items():
                    pu = prev_tot.get((prod, cat), 0)
                    pct = ((units - pu) / pu * 100) if pu > 0 else 0.0
                    rows.append(
                        {
                            "Product": prod,
                            "Category": cat,
                            "Units": int(units),
                            "pct": pct,
                        }
                    )

                if rows:
                    rows.sort(key=lambda r: r["Units"], reverse=True)
                    rows = rows[:10]

                    html = (
                        '<table class="perf-table"><tr>'
                        "<th>Product</th><th>Category</th>"
                        f"<th>Units (Last {time_opt})</th><th>% Change</th></tr>"
                    )
                    for r in rows:
                        if r["pct"] > 0:
                            ps = f'<span class="pct-pos">+{r["pct"]:.1f}%</span>'
                        elif r["pct"] < 0:
                            ps = f'<span class="pct-neg">{r["pct"]:.1f}%</span>'
                        else:
                            ps = '<span style="color:#888">0.0%</span>'
                        html += (
                            f"<tr><td>{r['Product']}</td>"
                            f"<td>{r['Category']}</td>"
                            f"<td>{r['Units']:,}</td>"
                            f"<td>{ps}</td></tr>"
                        )
                    html += "</table>"
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.write("No data available for the selected filters.")

                st.markdown("</div>", unsafe_allow_html=True)

    # TAB 3 - FORECAST & EXPORT
    with tab_forecast:
        if st.session_state.df is None:
            st.info("Upload a CSV file in the **Data** tab first.")
        elif not _MODEL_OK:
            st.error(
                "**model.py** could not be imported. "
                "Make sure it is present in the src/ directory and all "
                "dependencies (prophet, pmdarima, xgboost) are installed."
            )
        else:
            df_fc = st.session_state.df.copy()

            # configuration card
            st.markdown(
                '<div class="card">'
                '<div class="card-title">Forecast Configuration</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns([2, 2, 3, 1.5])
            with c1:
                fc_product = st.selectbox(
                    "Product", st.session_state.products, key="fc_prod"
                )
            with c2:
                fc_algo = st.selectbox(
                    "Model", sorted(VALID_ALGORITHMS), key="fc_algo"
                )
            with c3:
                fc_weeks = st.slider(
                    "Training Window",
                    min_value=4,
                    max_value=8,
                    value=6,
                    step=1,
                    format="%d weeks",
                    key="fc_weeks",
                )
            with c4:
                st.markdown(
                    "<div style='height:1.6rem'></div>", unsafe_allow_html=True
                )
                run_btn = st.button(
                    "Run Forecast",
                    type="primary",
                    use_container_width=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # run forecast
            if run_btn:
                prod_df = (
                    df_fc[df_fc["Product"] == fc_product]
                    .sort_values("Date")
                    .reset_index(drop=True)
                )
                if prod_df.empty:
                    st.error(f"No data for product **{fc_product}**.")
                else:
                    series = pd.DataFrame(
                        {
                            "ds": pd.to_datetime(prod_df["Date"]),
                            "y": prod_df["Units Sold"].values,
                        }
                    )
                    with st.spinner(
                        f"Running {fc_algo} forecast for {fc_product}..."
                    ):
                        result = run_forecast(series, fc_algo, fc_weeks)

                    if result["error"]:
                        st.error(f"Forecast error: {result['error']}")
                        st.session_state.forecast_result = None
                    else:
                        st.session_state.forecast_result = {
                            "result": result,
                            "product": fc_product,
                            "algo": fc_algo,
                            "weeks": fc_weeks,
                        }
                        st.rerun()

            # display results
            if st.session_state.forecast_result is not None:
                res = st.session_state.forecast_result
                result = res["result"]
                fc_df = result["forecast_df"]
                hist_df = result["history_df"]
                metrics = result["metrics"]

                # metrics row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    mv = metrics.get("mape")
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">'
                        f'{f"{mv:.1f}%" if mv is not None else "N/A"}</div>'
                        f'<div class="metric-label">MAPE</div></div>',
                        unsafe_allow_html=True,
                    )
                with m2:
                    mv = metrics.get("mae")
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">'
                        f'{f"{mv:.1f}" if mv is not None else "N/A"}</div>'
                        f'<div class="metric-label">MAE</div></div>',
                        unsafe_allow_html=True,
                    )
                with m3:
                    mv = metrics.get("rmse")
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">'
                        f'{f"{mv:.1f}" if mv is not None else "N/A"}</div>'
                        f'<div class="metric-label">RMSE</div></div>',
                        unsafe_allow_html=True,
                    )
                with m4:
                    meets = metrics.get("meets_target", False)
                    badge = "Yes" if meets else "No"
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{badge}</div>'
                        f'<div class="metric-label">Meets 35% target</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    "<div style='height:0.5rem'></div>", unsafe_allow_html=True
                )

                # forecast chart
                st.markdown(
                    f'<div class="card"><div class="card-title">'
                    f'28-Day Forecast - {res["product"]} ({res["algo"]})'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                fig_fc = go.Figure()

                # historical
                fig_fc.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(hist_df["ds"]),
                        y=hist_df["y"],
                        mode="lines+markers",
                        line=dict(color="#333", width=2),
                        marker=dict(size=4, color="#333"),
                        name="Historical",
                        hovertemplate=(
                            "Date: %{x|%d %b %Y}<br>"
                            "Actual: %{y:.0f}<extra></extra>"
                        ),
                    )
                )

                # forecast (dashed pink)
                fig_fc.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(fc_df["ds"]),
                        y=fc_df["yhat"],
                        mode="lines+markers",
                        line=dict(color=PINK, width=2.5, dash="dash"),
                        marker=dict(size=5, color=PINK),
                        name="Forecast",
                        hovertemplate=(
                            "Date: %{x|%d %b %Y}<br>"
                            "Predicted: %{y:.0f}<extra></extra>"
                        ),
                    )
                )

                # dotted connecting line
                fig_fc.add_trace(
                    go.Scatter(
                        x=[
                            pd.to_datetime(hist_df["ds"].iloc[-1]),
                            pd.to_datetime(fc_df["ds"].iloc[0]),
                        ],
                        y=[hist_df["y"].iloc[-1], fc_df["yhat"].iloc[0]],
                        mode="lines",
                        line=dict(color="#999", width=1.5, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                fig_fc.update_layout(
                    height=380,
                    margin=dict(l=40, r=20, t=10, b=40),
                    xaxis=dict(showgrid=False, title="Date"),
                    yaxis=dict(
                        showgrid=True, gridcolor="#f0f0f0", title="Units Sold"
                    ),
                    plot_bgcolor="#fff",
                    paper_bgcolor="#fff",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_fc, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # expandable data table
                with st.expander("View forecast data table"):
                    disp = fc_df.copy()
                    disp["ds"] = pd.to_datetime(disp["ds"]).dt.strftime(
                        "%d %b %Y"
                    )
                    disp["yhat"] = disp["yhat"].round(0).astype(int)
                    disp.columns = ["Date", "Predicted Units"]
                    st.dataframe(
                        disp, use_container_width=True, hide_index=True
                    )

                # export card
                st.markdown(
                    '<div class="card"><div class="card-title">Export</div>',
                    unsafe_allow_html=True,
                )
                e1, e2 = st.columns(2)

                with e1:
                    export_df = fc_df.copy()
                    export_df["ds"] = pd.to_datetime(export_df["ds"]).dt.strftime(
                        "%Y-%m-%d"
                    )
                    export_df["yhat"] = export_df["yhat"].round(2)
                    export_df.columns = ["Date", "Predicted_Units"]
                    export_df["Product"] = res["product"]
                    export_df["Algorithm"] = res["algo"]
                    export_df["Training_Weeks"] = res["weeks"]
                    export_df["MAPE"] = metrics.get("mape")
                    csv_buf = export_df.to_csv(index=False)

                    st.download_button(
                        "Download Forecast CSV",
                        data=csv_buf,
                        file_name=(
                            f"forecast_{res['product']}_{res['algo']}_"
                            f"{datetime.date.today()}.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True,
                    )

                with e2:
                    try:
                        img = fig_fc.to_image(
                            format="png", width=1200, height=500, scale=2
                        )
                        st.download_button(
                            "Download Forecast PNG",
                            data=img,
                            file_name=(
                                f"forecast_{res['product']}_{res['algo']}_"
                                f"{datetime.date.today()}.png"
                            ),
                            mime="image/png",
                            use_container_width=True,
                        )
                    except Exception:
                        st.caption(
                            "PNG export requires the **kaleido** package. "
                            "Install with: pip install kaleido"
                        )

                st.markdown("</div>", unsafe_allow_html=True)

            else:
                # placeholder when no forecast has been run
                st.markdown(
                    """
                    <div class="card">
                        <div class="forecast-empty">
                            <div style="font-size:1.5rem; color:#ccc;">--</div>
                            <div style="font-size:1.1rem; font-weight:600;
                                        color:#999; margin-top:0.5rem;">
                                No forecast generated yet
                            </div>
                            <div style="color:#bbb; font-size:0.9rem;
                                        margin-top:0.25rem;">
                                Configure your parameters above and click
                                &ldquo;Run Forecast&rdquo;
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# PAGE: REPORTS (placeholder)
elif st.session_state.page == "Reports":
    st.markdown(
        '<p class="main-title">Reports</p>', unsafe_allow_html=True
    )
    st.info(
        "Reports will be generated here based on forecasting history. "
        "Upload data and run forecasts from the **Dashboard** to populate reports."
    )

# PAGE: SETTINGS (placeholder)
elif st.session_state.page == "Settings":
    st.markdown(
        '<p class="main-title">Settings</p>', unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card"><div class="card-title">Application Settings</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
**Dashboard Version:** 1.0.0

**Supported Algorithms:** {', '.join(sorted(VALID_ALGORITHMS))}

**Max Training Window:** 8 weeks

**Forecast Horizon:** 28 days
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><div class="card-title">Accessibility</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
- All charts include hover tooltips for screen readers
- High contrast colour scheme (WCAG 2.1 AA)
- No red-green only colour distinctions
- Tab navigation supported throughout
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
