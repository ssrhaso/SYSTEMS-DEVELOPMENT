""" 
Main App.py Streamlit Entry Point Script

Royden Dias - 22036792
Hasaan Ahmad - 23010646
Rayyan Tahir - 24063400
"""


import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocessor import load_all, to_series, _read_coffee
from model import run_forecast, VALID_ALGORITHMS
from models import (
    Category,
    Product,
    SaleRecord,
    Prediction,
    CSVReader,
    DataManager,
    SalesAnalyzer,
    PredictionEngine,
    AlgorithmType,
    AccuracyEvaluator,
    GraphView,
    TableView,
    MainDashboard,
    ViewType,
    DashboardController,
)


st.set_page_config(
    page_title="Bristol Pink Café Analytics",
    page_icon=":coffee:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PINK = "#E8547A"
PINK_DARK = "#C0395E"
WHITE = "#FFFFFF"
TEXT_DARK = "#111827"
TEXT_MID = "#6B7280"
BORDER = "#E5E7EB"

PRODUCT_COLOURS = {
    "Cappuccino": "#000000",
    "Americano": "#E8547A",
    "Croissants": "#F59E0B",
}

PRODUCTS = ["Cappuccino", "Americano", "Croissants"]

NAV_ITEMS = [
    ("Dashboard", "fa-chart-line"),
    ("Reports", "fa-file-lines"),
    ("Settings", "fa-gear"),
]


st.markdown(
    '<link rel="stylesheet" '
    'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/'
    '6.5.0/css/all.min.css" crossorigin="anonymous"/>',
    unsafe_allow_html=True,
)

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root { color-scheme: light !important; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #111827 !important;
    font-weight: 500 !important;
    font-size: 16px !important;
}

.stApp { background-color: #F8F9FA !important; }
.main .block-container {
    background-color: #F8F9FA !important;
    padding: 1.8rem 2.2rem 2rem 2.2rem !important;
    max-width: 1400px;
}

.main p, .main span, .main label, .main td, .main th, .main li,
[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th,
[data-testid="stMarkdownContainer"] span { color: #111827 !important; }

[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span { color: #6B7280 !important; }

input[type="checkbox"] { accent-color: #E8547A !important; }
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] label span,
[data-testid="stCheckbox"] p { color: #111827 !important; }
[data-testid="stCheckbox"] label:hover span { color: #E8547A !important; }

input[type="radio"] { accent-color: #E8547A !important; }
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] p { color: #111827 !important; }
[data-testid="stRadio"] label:hover span { color: #E8547A !important; }

[data-testid="stTextInput"] label p { color: #111827 !important; }
[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border-color: #E5E7EB !important;
    border-radius: 8px !important;
}
[data-baseweb="input"] input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    background-color: #FFFFFF !important;
    opacity: 1 !important;
}
[data-baseweb="input"]:focus-within {
    border-color: #E8547A !important;
    box-shadow: 0 0 0 2px rgba(232,84,122,0.15) !important;
}
[data-testid="stTextInput"] input:disabled,
[data-baseweb="input"][aria-disabled="true"] input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    background-color: #F3F4F6 !important;
    opacity: 1 !important;
}

[data-testid="stSelectbox"] label p { color: #111827 !important; }
[data-testid="stSelectbox"] > div > div {
    background-color: #FFFFFF !important;
    border-color: #E5E7EB !important;
    border-radius: 8px !important;
    color: #111827 !important;
}
[data-testid="stSelectbox"] svg {
    color: #111827 !important;
    fill: #111827 !important;
    opacity: 1 !important;
    display: inline-block !important;
    visibility: visible !important;
}

[data-testid="stSlider"] label p { color: #111827 !important; }
[data-testid="stSlider"] [role="slider"] { background: #E8547A !important; }

[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] div { color: #111827 !important; }
[data-testid="stFileUploader"] section {
    background: #FDE8EE !important;
    border: 2px dashed #E8547A !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] section:hover {
    background: #fad4de !important;
    border-color: #C0395E !important;
}
[data-testid="stFileUploader"] button {
    background: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: background 0.15s, border-color 0.15s !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #F3F4F6 !important;
    border-color: #E8547A !important;
    color: #E8547A !important;
}

.stButton > button {
    background: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    font-size: 0.96rem !important;
    font-weight: 500 !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: #F3F4F6 !important;
    border-color: #E8547A !important;
    color: #E8547A !important;
}

button[data-testid="baseButton-primary"],
[data-testid="baseButton-primary"] {
    background: #E8547A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(232,84,122,0.30) !important;
    transition: background 0.15s, box-shadow 0.15s !important;
}
button[data-testid="baseButton-primary"]:hover {
    background: #C0395E !important;
    box-shadow: 0 4px 12px rgba(232,84,122,0.40) !important;
}

[data-testid="stDownloadButton"] > button {
    background: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.96rem !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #F3F4F6 !important;
    border-color: #E8547A !important;
    color: #E8547A !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div,
[data-testid="stSidebar"] > div > div > div,
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div:first-child,
section[data-testid="stSidebar"] > div:first-child > div {
    background-color: #111111 !important;
    background-image: none !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small { color: #FFFFFF !important; }
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
    margin: 0.5rem 0 !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: rgba(255,255,255,0.80) !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100% !important;
    text-align: left !important;
    padding: 0.65rem 1rem !important;
    min-height: 2.6rem !important;
    font-size: 1.02rem !important;
    font-weight: 500 !important;
    margin-bottom: 2px !important;
    box-shadow: none !important;
    justify-content: flex-start !important;
    transition: background 0.15s, color 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #FFFFFF !important;
    border: none !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
    background: #E8547A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    cursor: default !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 0 1.2rem !important;
    border-bottom: 2px solid #E5E7EB !important;
    gap: 0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.97rem !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
    padding: 0.85rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #E8547A !important; }
.stTabs [aria-selected="true"] {
    color: #E8547A !important;
    border-bottom-color: #E8547A !important;
    font-weight: 600 !important;
}
.stTabs [data-testid="stTabsContent"] {
    background: #FFFFFF !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1.6rem 1.6rem 1.8rem !important;
    border: 1px solid #E5E7EB !important;
    border-top: none !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border-radius: 12px !important;
    border: 1px solid #E5E7EB !important;
    padding: 0.6rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    transition: box-shadow 0.15s !important;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 12px rgba(232,84,122,0.12) !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.88rem !important;
    color: #6B7280 !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #111827 !important;
}

[data-testid="stExpander"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    background-color: #111827 !important;
    padding: 0.7rem 1rem !important;
    cursor: pointer !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
[data-testid="stExpander"] summary:hover {
    background-color: #1F2937 !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #FFFFFF !important;
    padding: 0.8rem !important;
    border-top: 1px solid #E5E7EB !important;
}

[data-testid="stDataFrame"] > div {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #E5E7EB !important;
}

[data-testid="stAlert"] { border-radius: 8px !important; }

hr { border-color: #E5E7EB !important; margin: 0.65rem 0 !important; }
</style>""", unsafe_allow_html=True)


for _k, _v in [
    ("page", "Dashboard"),
    ("forecast_res", None),
    ("forecast_cfg", {}),
    ("notifications", {
        "forecast_alerts": True,
        "low_stock": True,
        "weekly_reports": False,
        "upload_status": True,
    }),
    ("locations", [
        {"name": "Bristol Centre", "active": True},
        {"name": "Bristol Harbour", "active": True},
        {"name": "Clifton Village", "active": True},
    ]),
    ("editing_loc", None),
    ("adding_loc", False),
    ("custom_df", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_all()

try:
    df = get_data()
    data_loaded = True
    data_error = None
except Exception as exc:
    df = pd.DataFrame()
    data_loaded = False
    data_error = str(exc)

controller = DashboardController()

# If user previously uploaded a custom CSV, use it as the active dataset
if st.session_state.get("custom_df") is not None:
    df = st.session_state["custom_df"]
    data_loaded = True
    data_error = None

# Determine which products are actually present in the loaded data
available_products = [p for p in PRODUCTS if p in df.columns] if data_loaded else []

if data_loaded:
    controller.data_manager.load_from_dataframe(df)
    controller.dashboard.set_view(controller.graph_view)
    controller.dashboard.toggle_view(ViewType.GRAPH)


def html_table(data: pd.DataFrame):
    controller.table_view.render_dataframe(data)


def pink_info(msg: str):
    st.markdown(
        f"<div style='background:#FDE8EE;border:1px solid #E8547A;"
        f"border-left:4px solid #E8547A;border-radius:8px;"
        f"padding:0.7rem 1rem;margin:0.4rem 0;'>"
        f"<span style='color:#7C1D36;font-size:0.975rem;'>"
        f"<i class='fa-solid fa-circle-info' style='color:#E8547A;"
        f"margin-right:0.4rem;'></i>{msg}</span></div>",
        unsafe_allow_html=True,
    )


def section_title(fa_icon: str, text: str):
    st.markdown(
        f"<p style='font-weight:600;font-size:1.03rem;color:{TEXT_DARK};"
        f"margin:0 0 0.8rem 0;display:flex;align-items:center;gap:0.5rem;'>"
        f"<i class='fa-solid {fa_icon}' style='color:{PINK};"
        f"font-size:0.95rem;'></i> {text}</p>",
        unsafe_allow_html=True,
    )


def badge_html(text: str, green: bool = True) -> str:
    bg = "#DCFCE7" if green else "#F3F4F6"
    fg = "#15803D" if green else "#6B7280"
    return (
        f"<span style='background:{bg};color:{fg};"
        f"padding:0.2rem 0.75rem;border-radius:20px;"
        f"font-size:0.88rem;font-weight:600;'>{text}</span>"
    )


def page_header(title: str, subtitle: str):
    st.markdown(
        f"<div style='padding-bottom:1rem;margin-bottom:1.4rem;"
        f"border-bottom:1px solid {BORDER};'>"
        f"<h2 style='font-size:1.5rem;font-weight:700;"
        f"color:{TEXT_DARK};margin:0 0 0.2rem 0;'>{title}</h2>"
        f"<p style='font-size:0.97rem;color:{TEXT_MID};margin:0;'>"
        f"{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def check_row(label: str, passed):
    if passed is True:
        icon, clr = "fa-circle-check", "#22C55E"
    elif passed is False:
        icon, clr = "fa-circle-xmark", "#EF4444"
    else:
        icon, clr = "fa-circle", "#D1D5DB"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:0.5rem;"
        f"padding:0.3rem 0;font-size:0.96rem;color:{TEXT_DARK};'>"
        f"<i class='fa-solid {icon}' style='color:{clr};"
        f"font-size:0.94rem;'></i>"
        f"<span>{label}</span></div>",
        unsafe_allow_html=True,
    )


def plotly_axes(fig, height=270, top=10, y_title="Units Sold",
                x_tickformat="%d %b"):
    axis_font = dict(color="#374151", size=11)
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=top, b=0),
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(color="#374151", family="Inter, sans-serif"),
        xaxis=dict(
            showgrid=False,
            tickformat=x_tickformat,
            tickfont=axis_font,
            title=dict(font=axis_font),
        ),
        yaxis=dict(
            gridcolor="#F0F0F0",
            tickfont=axis_font,
            title=dict(text=y_title, font=axis_font),
        ),
    )


with st.sidebar:
    st.markdown(
        "<div style='padding:1.1rem 0 1rem 0;display:flex;align-items:center;gap:0.7rem;'>"
        "<div style='background:#C0395E;border-radius:10px;width:40px;height:40px;"
        "display:flex;align-items:center;justify-content:center;flex-shrink:0;'>"
        "<i class='fa-solid fa-mug-hot' style='color:#FFFFFF;font-size:1.1rem;'></i>"
        "</div>"
        "<div>"
        "<div style='font-size:1.15rem;font-weight:800;letter-spacing:-0.2px;"
        "color:#FFFFFF;line-height:1.2;'>Bristol Pink Cafe</div>"
        "<div style='font-size:0.82rem;color:rgba(255,255,255,0.45);"
        "margin-top:0.1rem;font-weight:400;'>Analytics</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='border-top:1px solid rgba(255,255,255,0.10);margin-bottom:0.6rem;'></div>",
        unsafe_allow_html=True,
    )

    for label, icon in NAV_ITEMS:
        is_active = st.session_state.page == label
        if st.button(label, key=f"nav_{label}",
                     use_container_width=True, disabled=is_active):
            st.session_state.page = label
            st.session_state.forecast_res = None
            st.rerun()

page = st.session_state.page


if page == "Dashboard":
    page_header(
        "Sales Forecasting Dashboard",
        "Upload data, explore trends, and plan 28-day production",
    )

    tab_data, tab_insights, tab_forecast = st.tabs(
        ["  Data", "  Insights", "  Forecast & Export"]
    )

    with tab_data:
        left, right = st.columns([1.5, 1], gap="large")

        with left:
            with st.container(border=True):
                section_title("fa-upload", "Upload Sales CSV")
                uploaded = st.file_uploader(
                    "Upload CSV", type=["csv"],
                    label_visibility="collapsed",
                )
                st.caption(
                    "Expected format: Date (DD/MM/YYYY)  "
                    "·  Product columns  ·  Numeric values"
                )
                if uploaded:
                    try:
                        # Use CSVReader for raw validation, then parse with pandas
                        import tempfile as _tmpfile
                        with _tmpfile.NamedTemporaryFile(
                            delete=False, suffix=".csv", mode="wb"
                        ) as _tmp:
                            _tmp.write(uploaded.getvalue())
                            _tmp_path = _tmp.name
                        raw_rows = CSVReader.read_rows(_tmp_path)
                        os.unlink(_tmp_path)
                        if not raw_rows or len(raw_rows) < 2:
                            st.error("CSV file is empty or has no data rows.")
                        else:
                            uploaded.seek(0)
                            # Detect multi-row header (coffee CSV format):
                            # the coffee file has 3+ columns
                            # (Date, Number Sold, <blank>) due to a
                            # trailing comma, whereas the croissant file
                            # only has 2 columns (Date, Number Sold).
                            first_row = pd.read_csv(uploaded, nrows=0).columns.tolist()
                            uploaded.seek(0)
                            is_coffee_format = (
                                len(first_row) >= 3
                                and any("Number Sold" in str(c) for c in first_row)
                                and any("Date" in str(c) for c in first_row)
                            )
                            if is_coffee_format:
                                # Coffee-style multi-row header — reuse
                                # the preprocessor's parsing logic via temp file
                                import tempfile as _tmpfile2
                                with _tmpfile2.NamedTemporaryFile(
                                    delete=False, suffix=".csv", mode="wb"
                                ) as _tf:
                                    _tf.write(uploaded.getvalue())
                                    _tf_path = _tf.name
                                custom = _read_coffee(_tf_path)
                                os.unlink(_tf_path)
                            else:
                                custom = pd.read_csv(uploaded)
                            # Rename common alternate headers so the
                            # rest of the app recognises the column.
                            if "Number Sold" in custom.columns:
                                custom = custom.rename(
                                    columns={"Number Sold": "Croissants"}
                                )
                            # Ensure product columns are numeric
                            for p in PRODUCTS:
                                if p in custom.columns:
                                    custom[p] = pd.to_numeric(
                                        custom[p], errors="coerce"
                                    )
                            if "Date" in custom.columns:
                                custom["Date"] = pd.to_datetime(
                                    custom["Date"],
                                    format="%d/%m/%Y",
                                    errors="coerce",
                                )
                                custom = custom.dropna(
                                    subset=["Date"]
                                ).sort_values("Date").reset_index(drop=True)
                            # Integrate uploaded data into the controller
                            st.session_state["custom_df"] = custom
                            df = custom
                            data_loaded = True
                            available_products = [p for p in PRODUCTS if p in df.columns]
                            controller.data_manager.load_from_dataframe(df)
                            st.success(
                                f"File loaded: **{uploaded.name}** — "
                                f"{len(custom):,} rows, "
                                f"{len(raw_rows) - 1} raw CSV records validated "
                                f"via CSVReader"
                            )
                            html_table(custom.head(8))
                    except Exception as e:
                        st.error(f"Could not read file: {e}")
                elif data_loaded:
                    pink_info("Using pre-loaded Pink Café data from /data/raw/")
                    disp = df.head(8).copy()
                    disp["Date"] = disp["Date"].dt.strftime("%d/%m/%Y")
                    html_table(disp)

        with right:
            with st.container(border=True):
                section_title("fa-circle-info", "Data Status")
                if data_loaded:
                    st.markdown(badge_html("Data Loaded", green=True),
                                unsafe_allow_html=True)
                    st.markdown("")
                    st.markdown(
                        f"| | |\n|:--|:--|\n"
                        f"| **Rows** | {len(df):,} |\n"
                        f"| **Products** | {', '.join(available_products)} |\n"
                        f"| **Date range** | "
                        f"{df['Date'].min().strftime('%d %b %Y')} -> "
                        f"{df['Date'].max().strftime('%d %b %Y')} |\n"
                        f"| **Missing values** | {df.isnull().sum().sum()} |\n"
                        f"| **Duplicates** | {df.duplicated().sum()} |"
                    )
                else:
                    st.markdown(badge_html("No file loaded", green=False),
                                unsafe_allow_html=True)
                    if data_error:
                        st.caption(data_error)

            st.markdown("")

            with st.container(border=True):
                section_title("fa-list-check", "Validation Rules")
                if data_loaded:
                    checks = [
                        ("Required columns present",
                         bool({"Date", "Cappuccino", "Americano", "Croissants"}
                              .issubset(df.columns))),
                        ("No invalid dates",
                         bool(df["Date"].isna().sum() == 0)),
                        ("Reasonable numeric values",
                         bool(all(
                             (df[p] >= 0).all() and (df[p] < 10_000).all()
                             for p in PRODUCTS if p in df.columns
                         ))),
                        ("No duplicate entries",
                         bool(df.duplicated().sum() == 0)),
                        ("Consistent product naming", True),
                    ]
                else:
                    checks = [(lbl, None) for lbl in [
                        "Required columns present",
                        "No invalid dates",
                        "Reasonable numeric values",
                        "No duplicate entries",
                        "Consistent product naming",
                    ]]
                for label, passed in checks:
                    check_row(label, passed)

    with tab_insights:
        if not data_loaded:
            controller.dashboard.show_error_message("No data loaded.")
        else:
            f1, f2, _ = st.columns([1.2, 2.8, 0.5])
            with f1:
                sel_product = st.selectbox(
                    "Product", ["All Products"] + available_products,
                    label_visibility="collapsed",
                )
            with f2:
                win_label = st.radio(
                    "Window", ["1 Week", "4 Weeks", "12 Weeks"],
                    horizontal=True,
                    label_visibility="collapsed",
                )

            days = {"1 Week": 7, "4 Weeks": 28, "12 Weeks": 84}[win_label]
            df_win = df.tail(days).copy()

            with st.container(border=True):
                section_title("fa-chart-line", "Sales Over Time")
                # Set dashboard to graph view for the chart section
                controller.dashboard.toggle_view(ViewType.GRAPH)
                controller.dashboard.set_view(controller.graph_view)
                plot_cols = (
                    available_products if sel_product == "All Products"
                    else [sel_product]
                )
                sales_by_product = {}
                min_date = df_win["Date"].min().date()
                for col in plot_cols:
                    records = controller.data_manager.get_sales_by_product(col)
                    sales_by_product[col] = [r for r in records if r.date >= min_date]
                fig_line = controller.graph_view.plot_multi_product_history(sales_by_product)
                plotly_axes(fig_line, height=270, top=10, y_title="Units Sold")
                fig_line.update_layout(
                    legend=dict(
                        orientation="h", y=1.18, x=1, xanchor="right",
                        font=dict(size=11, color="#374151"),
                    ),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_line, use_container_width=True)

            bc, tc = st.columns([1, 1.5], gap="large")

            with bc:
                with st.container(border=True):
                    section_title("fa-trophy", "Top Selling Products")
                    totals = dict(sorted(
                        {p: int(df_win[p].sum()) for p in available_products}.items(),
                        key=lambda x: x[1],
                    ))
                    fig_bar = go.Figure(go.Bar(
                        x=list(totals.values()),
                        y=list(totals.keys()),
                        orientation="h",
                        marker_color=[PRODUCT_COLOURS.get(k, PINK) for k in totals],
                        text=[f"{v:,}" for v in totals.values()],
                        textposition="outside",
                        textfont=dict(color="#374151", size=11),
                    ))
                    plotly_axes(fig_bar, height=190, top=5,
                                y_title="", x_tickformat=",d")
                    fig_bar.update_layout(
                        margin=dict(l=0, r=65, t=5, b=0),
                        xaxis=dict(showgrid=False, showticklabels=False),
                        yaxis=dict(showgrid=False,
                                   tickfont=dict(size=12, color="#374151")),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            with tc:
                with st.container(border=True):
                    # Switch dashboard to table view for the performance table
                    controller.dashboard.toggle_view(ViewType.TABLE)
                    controller.dashboard.set_view(controller.table_view)
                    section_title("fa-table", "Recent Performance by Product")
                    df_prev = df.tail(days * 2).head(days)
                    rows = []
                    for p in available_products:
                        product_obj = Product(p, Category.COFFEE if p != "Croissants" else Category.PASTRY)
                        product_sales = controller.data_manager.get_sales_by_product(p)
                        fluctuation = controller.sales_analyzer.get_sales_fluctuation(
                            product_obj, weeks=max(1, days // 7), sales=product_sales
                        )
                        curr = sum(fluctuation.values()) if fluctuation else int(df_win[p].sum())
                        prev = df_prev[p].sum() if len(df_prev) else curr
                        pct = ((curr - prev) / prev * 100) if prev else 0
                        rows.append({
                            "Product": p,
                            "Category": product_obj.get_type().value,
                            f"Units ({win_label})": f"{int(curr):,}",
                            "% Change": f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%",
                        })
                    html_table(pd.DataFrame(rows))

    with tab_forecast:
        if not data_loaded:
            controller.dashboard.show_error_message("No data loaded.")
        else:
            with st.container(border=True):
                section_title("fa-sliders", "Forecast Configuration")
                c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.8, 0.8])
                with c1:
                    fc_product = st.selectbox("Product", available_products, key="fc_product")
                with c2:
                    fc_algo = st.selectbox("Model", sorted(VALID_ALGORITHMS),
                                           key="fc_algo")
                with c3:
                    fc_weeks = st.slider("Training window (weeks)", 4, 8, 4,
                                         key="fc_weeks")
                with c4:
                    st.markdown("<br>", unsafe_allow_html=True)
                    run_btn = st.button("Run Forecast", type="primary",
                                        use_container_width=True)

            if run_btn:
                series = to_series(df, fc_product)
                # Train the prediction engine with historical sale records
                all_product_sales = controller.data_manager.get_sales_by_product(fc_product)
                controller.prediction_engine.train_model(all_product_sales)
                controller.prediction_engine.set_training_period(fc_weeks)
                controller.prediction_engine.algorithm = AlgorithmType(fc_algo)
                # Switch to graph view for forecast display
                controller.dashboard.toggle_view(ViewType.GRAPH)
                controller.dashboard.set_view(controller.graph_view)
                with st.spinner(
                    f"Running {fc_algo} on {fc_product} — "
                    "this may take up to 60 seconds..."
                ):
                    result = controller.prediction_engine.run_forecast_from_series(
                        series, fc_algo, fc_weeks
                    )
                controller.accuracy_evaluator.compare_from_result(result)
                st.session_state.forecast_res = result
                st.session_state.forecast_cfg = {
                    "product": fc_product,
                    "algo": fc_algo,
                    "weeks": fc_weeks,
                    "accuracy_score": controller.accuracy_evaluator.get_accuracy_score(),
                }

            res = st.session_state.forecast_res
            cfg = st.session_state.forecast_cfg

            if res is None:
                st.markdown(
                    f"<div style='text-align:center;padding:4rem 2rem;"
                    f"color:{TEXT_MID};'>"
                    f"<i class='fa-solid fa-wand-magic-sparkles'"
                    f" style='font-size:2.8rem;color:{PINK};"
                    f"margin-bottom:0.9rem;display:block;'></i>"
                    f"<p style='font-size:1.1rem;font-weight:600;"
                    f"color:{TEXT_DARK};margin:0 0 0.35rem;'>"
                    f"No forecast generated yet</p>"
                    f"<p style='font-size:0.97rem;margin:0;color:{TEXT_MID};'>"
                    f"Configure your parameters above and click "
                    f"<strong>Run Forecast</strong></p></div>",
                    unsafe_allow_html=True,
                )
            elif res["error"]:
                st.error(f"Forecast failed: {res['error']}")
            else:
                m = res["metrics"]
                hist = res["history_df"]
                fc_df = res["forecast_df"]

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("MAPE", f"{m['mape']:.2f}%" if m['mape'] else "—")
                m2.metric("MAE", f"{m['mae']:.1f}" if m['mae'] else "—")
                m3.metric("RMSE", f"{m['rmse']:.1f}" if m['rmse'] else "—")
                m4.metric("Target Met", "Yes" if m["meets_target"] else "No")
                accuracy = cfg.get("accuracy_score", 0)
                m5.metric("Accuracy", f"{accuracy:.1f}%" if accuracy else "—")

                with st.container(border=True):
                    product_obj = Product(
                        cfg.get("product", ""),
                        Category.COFFEE if cfg.get("product") != "Croissants" else Category.PASTRY,
                    )
                    history_records = [
                        SaleRecord(
                            sale_date=row["ds"].date() if hasattr(row["ds"], "date") else row["ds"],
                            bakery_location="Bristol Centre",
                            quantity_sold=int(row["y"]),
                            product=product_obj,
                        )
                        for _, row in hist.iterrows()
                    ]
                    prediction_records = [
                        Prediction(
                            product=product_obj,
                            predicted_date=row["ds"].date() if hasattr(row["ds"], "date") else row["ds"],
                            predicted_quantity=float(row["yhat"]),
                        )
                        for _, row in fc_df.iterrows()
                    ]
                    fig_fc = controller.graph_view.plot_history_and_prediction(
                        history_records, prediction_records
                    )
                    plotly_axes(fig_fc, height=340, top=65, y_title="Units Sold")
                    fig_fc.update_layout(
                        title=dict(
                            text=f"{cfg.get('product')} — {cfg.get('algo')} Forecast",
                            font=dict(size=13, color=TEXT_DARK),
                            x=0, xanchor="left", y=0.98, yanchor="top",
                        ),
                        legend=dict(
                            orientation="h", y=1.22, x=1, xanchor="right",
                            font=dict(size=11, color="#374151"),
                        ),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)

                with st.container(border=True):
                    section_title("fa-download", "Export Forecast")
                    with st.expander("Preview forecast data"):
                        prev = fc_df.copy()
                        prev["ds"] = prev["ds"].dt.strftime("%d %b %Y")
                        prev["yhat"] = prev["yhat"].round(1)
                        prev.columns = ["Date", "Forecast (units)"]
                        html_table(prev)

                    e1, e2 = st.columns(2)
                    with e1:
                        st.download_button(
                            "Download Forecast CSV",
                            data=fc_df.to_csv(index=False).encode(),
                            file_name=(
                                "forecast_"
                                + str(cfg.get("product")) + "_"
                                + str(cfg.get("algo")) + ".csv"
                            ),
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with e2:
                        summary = "\n".join([
                            f"Product:          {cfg.get('product')}",
                            f"Algorithm:        {cfg.get('algo')}",
                            f"Training window:  {cfg.get('weeks')} weeks",
                            f"MAPE:             {m['mape']}%",
                            f"MAE:              {m['mae']}",
                            f"RMSE:             {m['rmse']}",
                            f"Meets target:     {m['meets_target']}",
                            "", "Date,Forecast",
                        ] + [
                            f"{r.ds.date()},{r.yhat:.1f}"
                            for _, r in fc_df.iterrows()
                        ])
                        st.download_button(
                            "Download Full Report TXT",
                            data=summary.encode(),
                            file_name=(
                                "report_"
                                + str(cfg.get("product")) + "_"
                                + str(cfg.get("algo")) + ".txt"
                            ),
                            mime="text/plain",
                            use_container_width=True,
                        )


elif page == "Reports":
    page_header(
        "Reports",
        "Generate and download custom reports for your café",
    )

    r1, r2 = st.columns([1.3, 1], gap="large")

    with r1:
        with st.container(border=True):
            section_title("fa-file-csv", "Ready Reports")
            if data_loaded:
                for title, desc, data_bytes, fname in [
                    (
                        "Sales Summary",
                        "Last 30 days breakdown by product",
                        df.tail(30).to_csv(index=False).encode(),
                        "sales_summary.csv",
                    ),
                    (
                        "Performance Report",
                        "Full dataset performance metrics",
                        df.to_csv(index=False).encode(),
                        "performance_report.csv",
                    ),
                ]:
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(
                        f"<p style='margin:0;font-weight:600;font-size:1.0rem;"
                        f"color:{TEXT_DARK};'>{title}</p>"
                        f"<p style='margin:0;font-size:0.9rem;"
                        f"color:{TEXT_MID};'>{desc}</p>",
                        unsafe_allow_html=True,
                    )
                    c2.download_button(
                        "Download", data=data_bytes,
                        file_name=fname, mime="text/csv",
                        key=f"ready_{title}",
                        use_container_width=True,
                    )
                    st.divider()
            else:
                st.warning("No data loaded.")

        st.markdown("")

        with st.container(border=True):
            section_title("fa-screwdriver-wrench", "Custom Report Generator")
            cc1, cc2 = st.columns(2)
            with cc1:
                r_type = st.selectbox(
                    "Report type",
                    ["Sales Analysis", "Product Breakdown", "Full Export"],
                )
            with cc2:
                r_range = st.selectbox(
                    "Date range",
                    ["Last 7 days", "Last 30 days", "Last 90 days", "All data"],
                )
            if st.button("Generate Custom Report", type="primary"):
                if data_loaded:
                    n_rows = {
                        "Last 7 days": 7,
                        "Last 30 days": 30,
                        "Last 90 days": 90,
                    }.get(r_range, len(df))
                    st.download_button(
                        f"Download {r_type} ({r_range})",
                        data=df.tail(n_rows).to_csv(index=False).encode(),
                        file_name="custom_" + r_type.lower().replace(" ", "_") + ".csv",
                        mime="text/csv",
                    )
                else:
                    st.error("No data loaded.")

    with r2:
        with st.container(border=True):
            section_title("fa-clock-rotate-left", "Recent Reports")
            for name, date, fmt in [
                ("January Sales Summary",      "2026-02-01", "CSV"),
                ("Weekly Performance Report",  "2026-01-28", "CSV"),
                ("Product Analysis Q4 2025",   "2026-01-15", "CSV"),
                ("Forecast Report - December", "2025-12-28", "TXT"),
            ]:
                c1, c2 = st.columns([3, 1])
                c1.markdown(
                    f"<p style='margin:0;font-weight:600;font-size:1.0rem;"
                    f"color:{TEXT_DARK};'>{name}</p>"
                    f"<p style='margin:0;font-size:0.9rem;color:{TEXT_MID};'>"
                    f"{date}</p>",
                    unsafe_allow_html=True,
                )
                c2.markdown(
                    "<div style='padding-top:0.3rem;'>"
                    "<span style='background:#F3F4F6;color:#374151;"
                    "padding:0.2rem 0.6rem;border-radius:20px;"
                    "font-size:0.86rem;font-weight:600;'>"
                    + fmt + "</span></div>",
                    unsafe_allow_html=True,
                )
                st.divider()


elif page == "Settings":
    page_header(
        "Settings",
        "Manage your café dashboard preferences and configuration",
    )

    s1, s2 = st.columns(2, gap="large")

    with s1:
        with st.container(border=True):
            section_title("fa-user", "Profile Settings")
            st.text_input("Full Name", value="Sarah Johnson", key="s_name")
            st.text_input("Email", value="sarah@bristolpink.co.uk", key="s_email")
            st.text_input("Role", value="Café Manager", disabled=True)
            if st.button("Save Changes", type="primary"):
                st.success("Profile saved successfully.")

        st.markdown("")

        with st.container(border=True):
            section_title("fa-bell", "Notifications")
            ns = st.session_state.notifications
            ns["forecast_alerts"] = st.checkbox(
                "Forecast Alerts — get notified when a new forecast is ready",
                value=ns["forecast_alerts"],
            )
            ns["low_stock"] = st.checkbox(
                "Low Stock Warnings — alert when predicted demand may exceed supply",
                value=ns["low_stock"],
            )
            ns["weekly_reports"] = st.checkbox(
                "Weekly Reports — receive an automated summary every Monday",
                value=ns["weekly_reports"],
            )
            ns["upload_status"] = st.checkbox(
                "Upload Status — confirm when a data file has been processed",
                value=ns["upload_status"],
            )

    with s2:
        with st.container(border=True):
            section_title("fa-location-dot", "Locations")
            for idx, loc in enumerate(st.session_state.locations):
                loc_name = loc["name"]
                loc_active = loc["active"]

                if st.session_state.editing_loc == idx:
                    ec1, ec2 = st.columns([3, 1])
                    with ec1:
                        new_name = st.text_input(
                            "Location name", value=loc_name,
                            key=f"edit_name_{idx}",
                            label_visibility="collapsed",
                        )
                        new_active = st.checkbox(
                            "Active", value=loc_active,
                            key=f"edit_active_{idx}",
                        )
                    with ec2:
                        if st.button("Save", key=f"save_loc_{idx}",
                                     type="primary"):
                            new_name_stripped = new_name.strip()
                            if new_name_stripped:
                                st.session_state.locations[idx]["name"] = (
                                    new_name_stripped
                                )
                                st.session_state.locations[idx]["active"] = (
                                    new_active
                                )
                            st.session_state.editing_loc = None
                            st.rerun()
                        if st.button("Cancel", key=f"cancel_loc_{idx}"):
                            st.session_state.editing_loc = None
                            st.rerun()
                        if st.button("Delete", key=f"del_loc_{idx}"):
                            st.session_state.locations.pop(idx)
                            st.session_state.editing_loc = None
                            st.rerun()
                else:
                    lc1, lc2 = st.columns([3, 1])
                    status_badge = (
                        "<span style='background:#DCFCE7;color:#15803D;"
                        "padding:0.15rem 0.55rem;border-radius:20px;"
                        "font-size:0.84rem;font-weight:600;'>Active</span>"
                        if loc_active
                        else "<span style='background:#F3F4F6;color:#6B7280;"
                        "padding:0.15rem 0.55rem;border-radius:20px;"
                        "font-size:0.84rem;font-weight:600;'>Inactive</span>"
                    )
                    lc1.markdown(
                        f"<p style='margin:0.2rem 0;font-weight:600;"
                        f"font-size:1.0rem;color:{TEXT_DARK};'>"
                        f"{loc_name} &nbsp;{status_badge}</p>",
                        unsafe_allow_html=True,
                    )
                    if lc2.button("Edit", key=f"loc_{idx}"):
                        st.session_state.editing_loc = idx
                        st.rerun()
                st.divider()

            if st.session_state.adding_loc:
                new_loc_name = st.text_input(
                    "New location name", key="new_loc_name",
                    placeholder="Enter location name",
                )
                ab1, ab2 = st.columns(2)
                if ab1.button("Add", type="primary", key="confirm_add_loc"):
                    new_loc_stripped = new_loc_name.strip()
                    if new_loc_stripped:
                        st.session_state.locations.append(
                            {"name": new_loc_stripped, "active": True}
                        )
                    st.session_state.adding_loc = False
                    st.rerun()
                if ab2.button("Cancel", key="cancel_add_loc"):
                    st.session_state.adding_loc = False
                    st.rerun()
            else:
                if st.button("Add New Location"):
                    st.session_state.adding_loc = True
                    st.rerun()

        st.markdown("")

        with st.container(border=True):
            section_title("fa-shield-halved", "Data & Privacy")
            dp1, dp2 = st.columns(2)
            if dp1.button("Export All Data", use_container_width=True):
                if data_loaded:
                    st.download_button(
                        "Download all_data.csv",
                        data=df.to_csv(index=False).encode(),
                        file_name="all_data.csv",
                        mime="text/csv",
                    )
            dp2.button("Change Password", use_container_width=True)
            st.markdown("")
            st.selectbox(
                "Data Retention",
                [
                    "Keep data for 1 year",
                    "Keep data for 2 years",
                    "Keep indefinitely",
                ],
            )