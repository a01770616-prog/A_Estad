import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from pathlib import Path
import re
import statsmodels.api as sm
import statsmodels.api as sm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
#PALETA / PLOTLY

PALETTE = {
    "brand": "#FF385C",     
    "brand_alt": "#FF5A5F", 
    "salmon": "#FF6B6B",
    "white": "#FFFFFF",
    "gray500": "#4A4A4A",
    "gray900": "#1F1F1F",
    "bg": "#FFFFFF",
    "panel": "#F7F7F7",
}

# Paleta de competitividad Airbnb (del rojo brand a tonos más suaves)
AIRBNB_COMPETITIVENESS_SCALE = [
    "#FFF5F5",  # Muy claro (baja competitividad)
    "#FFE3E6",  # Claro
    "#FFB3BA",  # Medio claro  
    "#FF7A85",  # Medio
    "#FF5A5F",  # Medio alto (brand_alt)
    "#FF385C"   # Alto (brand principal)
]

#Config plotly por defecto
pio.templates.default = "plotly_white"
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    PALETTE["brand"], PALETTE["salmon"], "#F7A6A6",
    PALETTE["gray500"], PALETTE["gray900"], "#9CA3AF"
]
px.defaults.width = None
px.defaults.height = None
pio.templates["plotly_white"].layout.font.family = "Inter, Segoe UI, system-ui, -apple-system, sans-serif"
pio.templates["plotly_white"].layout.paper_bgcolor = PALETTE["bg"]
pio.templates["plotly_white"].layout.plot_bgcolor  = PALETTE["bg"]
pio.templates["plotly_white"].layout.margin = dict(l=10, r=10, t=40, b=10)


#CONFIG STREAMLIT

st.set_page_config(page_title="Airbnb", layout="wide")

# ESTILOS (sidebar pastel + títulos)
# =========================
AIRBNB_BRAND = "#FF385C"
AIRBNB_SOFT  = "#FF8FA3"

st.markdown(f"""
<style>
/* ===== Sidebar rosa pastel ===== */
[data-testid="stSidebar"]{{
  background: #FFD1DC !important;  /* rosa pastel */
  color: #111 !important;
}}
[data-testid="stSidebar"] *{{ color:#111 !important; }}
[data-testid="stSidebar"] > div:first-child{{ padding:18px 16px 22px 16px; }}

/* ===== Inputs en la sidebar ===== */
/* Text / Number / Select: altura fija cómoda */
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox > div > div{{
  background:#fff !important; color:#111 !important;
  border:1px solid rgba(0,0,0,.2) !important; border-radius:12px !important;
  height:42px;
}}

/* Multiselect: altura dinámica (clave para evitar superposición) */
[data-testid="stSidebar"] .stMultiSelect > div > div{{
  background:#fff !important; color:#111 !important;
  border:1px solid rgba(0,0,0,.2) !important; border-radius:12px !important;
  height:auto !important;            /* NO altura fija */
  min-height:42px;                   /* mínima estética */
  padding-top:6px; padding-bottom:6px;
  overflow:visible !important;
}}
/* Separación extra para que no pegue con el siguiente título/control */
[data-testid="stSidebar"] .stMultiSelect{{ margin-bottom:12px; }}

/* Chips (tags) del multiselect */
[data-testid="stSidebar"] [data-baseweb="tag"]{{
  background:#fff !important;
  color:#111 !important;
  border:1px solid rgba(0,0,0,.25) !important;
  border-radius:12px !important;
  box-shadow:none !important;
  margin:4px 6px 0 0;
}}
[data-testid="stSidebar"] [data-baseweb="tag"] *{{ color:#111 !important; fill:#111 !important; }}

/* Placeholders y separadores */
[data-testid="stSidebar"] input::placeholder{{ color:rgba(0,0,0,.45) !important; }}
[data-testid="stSidebar"] hr{{ display:none !important; }}

/* ===== Botones y sliders ===== */
[data-testid="stSidebar"] button[kind]{{
  background:#fff !important; color:#111 !important;
  border:1px solid rgba(0,0,0,.25) !important; border-radius:12px !important;
}}
[data-testid="stSidebar"] button[kind]:hover{{ background:#FAFAFA !important; }}

/* Slider: riel y thumb (evitar colores por defecto) */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div{{ background:rgba(0,0,0,.18) !important; }}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div{{ background:#111 !important; }}
[data-testid="stSidebar"] .stSlider [role="slider"]{{ background:#111 !important; box-shadow:none !important; }}

/* ===== Títulos del cuerpo ===== */
main .block-container h1{{ color:{AIRBNB_BRAND} !important; font-weight:800; letter-spacing:.2px; }}
main .block-container h2, main .block-container h3{{ color:{AIRBNB_SOFT} !important; font-weight:700; letter-spacing:.2px; }}
main .block-container section h1{{ color:{AIRBNB_BRAND} !important; }}
main .block-container section h2, main .block-container section h3{{ color:{AIRBNB_SOFT} !important; }}

/* Opcional: evitar que labels largos se corten/solapen en sidebar */
[data-testid="stSidebar"] label p{{ white-space:normal !important; }}
</style>
""", unsafe_allow_html=True)



#LOGO
HERE = Path(__file__).resolve().parent
LOGO_STEM = "airbnb-logo"  #Nombre imagen 
LOGO_PATH = None
for ext in (".png", ".jpg", ".jpeg", ".webp", ".svg"):
    p = HERE / f"{LOGO_STEM}{ext}"
    if p.exists():
        LOGO_PATH = str(p); break
if LOGO_PATH:
    st.sidebar.image(LOGO_PATH, width=300)
    st.sidebar.markdown("")


#DATA SOURCES 
DEFAULTS = [
    ("listingsBarcelona.csv", "Barcelona"),
    ("listingsAmsterdam.csv", "Amsterdam"),
    ("listingsMilan.csv", "Milan"),
    ("listingsGrecia.csv", "Atenas"),
    ("listingsMadrid.csv", "Madrid"),
]

paths = [ruta for ruta, _ in DEFAULTS]
names = [nombre for _, nombre in DEFAULTS]

def _to_float_price(val):
    """
    Convierte strings de precio a float de forma robusta.
    Soporta formatos: "€1.234,56", "1 234,56 €", "$1,234.56", "1 234€", "120–150", "120 - 150".
    Devuelve np.nan si no se puede parsear.
    """
    if pd.isna(val):
        return np.nan

    s = str(val).strip()
    if not s:
        return np.nan

    # Normaliza espacios "raros" y guiones
    s = (s.replace("\u00A0", " ")     # NBSP
           .replace("\u202F", " ")    # thin space
           .replace("–", "-")         # en dash
           .replace("—", "-"))        # em dash

    # Quita texto común no numérico
    s = re.sub(r"(per\s*night|/night|por\s*d[ií]a|/d[ií]a|night|noche|día|day)", "", s, flags=re.I)

    # Si es un rango (no negativo), toma el promedio
    # p.ej. "120-150", "120 - 150", "120- 150"
    m_range = re.findall(r"(?<!^)-", s)  # guiones que no son signo inicial
    if m_range:
        # extrae todos los números candidatos y promedia los dos primeros
        nums = re.findall(r"[-+]?\d[\d\s\.',]*", s)
        parsed = []
        for n in nums:
            x = _to_float_price(n)  # recursion sobre cada trozo numérico
            if pd.notna(x):
                parsed.append(x)
            if len(parsed) == 2:
                break
        if len(parsed) == 2:
            return float(np.mean(parsed))
        # si no se pudo, sigue con parsing normal de s

    # Deja solo dígitos, separadores y signo
    # (guardamos ',' '.' ' ' y apostrofe como posibles separadores de miles)
    s_clean = re.sub(r"[^0-9\-\.,'\s]", "", s).strip()
    if not s_clean:
        return np.nan

    # Heurística de separadores
    # 1) Si tiene '.' y ',', decide por el último separador como decimal cuando hay 2 dígitos detrás
    if "." in s_clean and "," in s_clean:
        last_dot = s_clean.rfind(".")
        last_com = s_clean.rfind(",")
        last = max(last_dot, last_com)
        tail = s_clean[last+1:].replace(" ", "").replace("'", "")
        if len(re.sub(r"\D", "", tail)) in (2, 1):  # 2 dígitos (típico centavos) o 1 (algunos redondeos)
            if last == last_com:
                # coma decimal => quita puntos/espacios/apóstrofes como miles y cambia coma por punto
                num = re.sub(r"[.\s']", "", s_clean).replace(",", ".")
            else:
                # punto decimal => quita comas/espacios/apóstrofes como miles
                num = re.sub(r"[, \s']", "", s_clean)
        else:
            # Si no parece decimal clásico, elimina todas las comas y espacios; trata punto como decimal si solo hay uno
            num = re.sub(r"[, \s']", "", s_clean)
    # 2) Solo comas (formato EU típico: "1 234,56" o "1234,56")
    elif "," in s_clean and "." not in s_clean:
        # si hay exactamente una coma y 1-2 dígitos al final -> decimal
        parts = s_clean.split(",")
        if len(parts) == 2 and re.fullmatch(r"\d{1,2}", re.sub(r"\D", "", parts[1] or "")):
            num = re.sub(r"[\s']", "", parts[0]) + "." + re.sub(r"\D", "", parts[1])
        else:
            # probablemente comas de miles: quítalas
            num = re.sub(r"[,\s']", "", s_clean)
    else:
        # Solo puntos o solo dígitos/espacios: quita separadores de miles (espacios/apóstrofes/comas residuales)
        num = re.sub(r"[, \s']", "", s_clean)

    # Evita casos como "-" o vacío
    if num in ("", "-", "+"):
        return np.nan

    try:
        return float(num)
    except Exception:
        # último intento: extrae primer número "claro" y reintenta
        m = re.search(r"[-+]?\d+(?:\.\d+)?", num)
        return float(m.group(0)) if m else np.nan


def _bathrooms_from_text(txt):
    if pd.isna(txt): return np.nan
    s = str(txt).lower()
    if "half" in s and not re.search(r"\d+(\.\d+)?", s): return 0.5
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan

def limpiar_estandarizar(df: pd.DataFrame, ciudad: str) -> pd.DataFrame:
    d = df.copy()
    d["ciudad"] = ciudad
    if "id" not in d.columns: d["id"] = np.arange(len(d)) + 1

    d["price"] = d.get("price", np.nan)
    d["price"] = d["price"].map(_to_float_price)

    if "neighbourhood_cleansed" in d.columns:
        d["barrio_std"] = d["neighbourhood_cleansed"]
    elif "neighbourhood" in d.columns:
        d["barrio_std"] = d["neighbourhood"]
    else:
        d["barrio_std"] = np.nan

    d["room_type"] = d.get("room_type", pd.Series(index=d.index, dtype="object")).astype(str).str.strip().replace({"nan": np.nan})
    d["accommodates"] = pd.to_numeric(d.get("accommodates", np.nan), errors="coerce")

    if "bathrooms_text" in d.columns:
        d["bathrooms_num"] = d["bathrooms_text"].map(_bathrooms_from_text)
    else:
        d["bathrooms_num"] = pd.to_numeric(d.get("bathrooms", np.nan), errors="coerce")

    d["latitude"]  = pd.to_numeric(d.get("latitude", np.nan), errors="coerce")
    d["longitude"] = pd.to_numeric(d.get("longitude", np.nan), errors="coerce")

    if "amenities" in d.columns:
        d["amenities_count"] = d["amenities"].astype(str).apply(
            lambda x: 0 if x in ("nan", "", "[]") else len([a for a in re.split(r"[,\|]", x.strip("[]")) if a.strip()])
        )
    else:
        d["amenities_count"] = np.nan

    d["price_per_person"] = np.where((d["accommodates"] >= 1) & d["price"].notna(), d["price"] / d["accommodates"], np.nan)

    # Convertir superhost a numérico (1 si es superhost, 0 si no)
    if "host_is_superhost" in d.columns:
        d["superhost_numeric"] = d["host_is_superhost"].astype(str).str.strip().str.lower().map(
            {"t": 1, "true": 1, "f": 0, "false": 0, "nan": np.nan}
        ).fillna(0).astype(float)
    else:
        d["superhost_numeric"] = 0.0

    # Agregar beds y bedrooms
    d["beds"] = pd.to_numeric(d.get("beds", np.nan), errors="coerce")
    d["bedrooms"] = pd.to_numeric(d.get("bedrooms", np.nan), errors="coerce")
    
    # Agregar minimum_nights y maximum_nights
    d["minimum_nights"] = pd.to_numeric(d.get("minimum_nights", np.nan), errors="coerce")
    d["maximum_nights"] = pd.to_numeric(d.get("maximum_nights", np.nan), errors="coerce")
    
    # Agregar availability
    d["availability_365"] = pd.to_numeric(d.get("availability_365", np.nan), errors="coerce")

    cols = ["id","ciudad","barrio_std","room_type","accommodates","bathrooms_num",
            "beds","bedrooms","minimum_nights","maximum_nights","availability_365",
            "price","price_per_person","amenities_count","latitude","longitude",
            "superhost_numeric",
            "property_type","host_is_superhost","cancellation_policy",
            "instant_bookable","review_scores_rating","number_of_reviews",
            "bed_type","neighbourhood_group_cleansed",
            "require_guest_profile_picture","require_guest_phone_verification",
            "host_response_time","host_identity_verified","has_availability","source"]
    keep = [c for c in cols if c in d.columns]
    return d[keep]

def recortar_outliers_por_ciudad(df: pd.DataFrame, col="price", p_low=0.01, p_high=0.99):
    limpio = []
    for ciudad, g in df.groupby("ciudad", dropna=False):
        if g[col].notna().sum() < 50:
            limpio.append(g); continue
        low, high = g[col].quantile(p_low), g[col].quantile(p_high)
        limpio.append(g[(g[col].isna()) | ((g[col] >= low) & (g[col] <= high))])
    return pd.concat(limpio, ignore_index=True)

@st.cache_data(show_spinner=False, hash_funcs={list: lambda x: str(x)})
def load_data(paths, names):
    here = Path(__file__).resolve().parent
    partes, warnings = [], []

    for raw_path, city in zip(paths, names):
        p = Path(raw_path)
        if not p.is_absolute(): p = here / p
        if not p.exists():
            warnings.append(f"No se encontró el archivo de **{city}**: `{raw_path}`")
            continue

        df_raw = None
        try:
            df_raw = pd.read_csv(p, low_memory=False)
        except UnicodeDecodeError:
            try:
                df_raw = pd.read_csv(p, low_memory=False, encoding="latin-1")
            except Exception as e:
                warnings.append(f"Error de lectura en **{city}**: {e}")
                continue
        except Exception as e:
            warnings.append(f"Error leyendo **{city}**: {e}")
            continue

        partes.append(limpiar_estandarizar(df_raw, city))

    if not partes:
        return pd.DataFrame(), warnings

    df_all = pd.concat(partes, ignore_index=True)
    if {"ciudad","id"}.issubset(df_all.columns):
        df_all = (df_all.sort_values(["ciudad","id"])
                        .drop_duplicates(subset=["ciudad","id"], keep="first"))

    if "price" in df_all.columns:
        df_all = recortar_outliers_por_ciudad(df_all, col="price", p_low=0.01, p_high=0.99)

    return df_all, warnings

df, warns = load_data(paths, names)
for w in warns: st.warning(w)
if df.empty:
    st.stop()

#LISTA DE CATEGÓRICAS
candidatas = [
    "room_type","barrio_std","property_type","instant_bookable","cancellation_policy",
    "host_is_superhost","host_identity_verified","host_response_time","has_availability",
    "bed_type","source","neighbourhood_group_cleansed",
    "price_range","accommodates_band","bathrooms_band","amenities_band",
]
Lista = [c for c in candidatas if c in df.columns]
if len(Lista) < 15:
    auto = [c for c in df.select_dtypes(include=["object","category"]).columns
            if c not in Lista and df[c].nunique(dropna=False) <= 50]
    Lista = (Lista + auto)[:15]
if not Lista:
    st.error("No encontré variables categóricas. Revisa que el DataFrame tenga columnas tipo object/category.")
    st.stop()


# MENÚ GENERAL

st.sidebar.title("Tipo de análisis")
View = st.sidebar.selectbox(
    label="Tipo de Análisis",
    options=["Extracción de Características", "Regresión Lineal", "Regresión No Lineal", "Regresión Logística", "ANOVA"]
)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================
# =========== EXTRACCIÓN DE CARACTERÍSTICAS ===============
# =========================================================
if View == "Extracción de Características":

    st.title("Extracción de Características")

    # --- Selección de ciudades ---
    if "ciudad" not in df.columns or df["ciudad"].dropna().empty:
        st.warning("No hay columna 'ciudad' válida en el DataFrame.")
        st.stop()

    ciudades_disp = sorted(df["ciudad"].dropna().unique().tolist())
    st.sidebar.header("Ciudades")
    selected_cities = st.sidebar.multiselect(
        "Selecciona de 1 a 5 ciudades",
        options=ciudades_disp,
        default=ciudades_disp[:min(3, len(ciudades_disp))],
        max_selections=5,
        key="cities_extraccion"
    )

    if not selected_cities:
        st.info("Selecciona al menos una ciudad para continuar.")
        st.stop()

    df_combined = df[df["ciudad"].isin(selected_cities)].copy()

    # ===================== GRÁFICAS Y ANÁLISIS =====================
    
    # Selectores comunes en sidebar
    top_k = st.sidebar.slider("Top categorías por gráfica", 5, 30, 10, key="topk_ext")
    mostrar_tabla = st.sidebar.checkbox("Mostrar tabla de frecuencias", value=False, key="mostrar_tabla_ext")
    Variable_Cat = st.sidebar.selectbox("Variables", options=Lista, key="var_cat_ext")

    # CREAR TABS PARA LOS DOS MODOS DE ANÁLISIS
    tab_ciudad, tab_comparativo = st.tabs(["Por ciudad", "Comparativo multi-ciudad"])

    # ============== TAB: POR CIUDAD ==============
    with tab_ciudad:
        # Selector de ciudad dentro del tab
        ciudad_sel = st.selectbox("Ciudad para gráficas individuales", sorted(selected_cities), key="ciudad_sel_ext")
        
        # Aquí va el código de "Por ciudad"
        df_city = df[df["ciudad"] == ciudad_sel].copy()

        Tabla_frecuencias = (
            df_city[Variable_Cat]
            .astype("object").fillna("NA").astype(str)
            .value_counts().head(top_k)
            .reset_index()
        )
        Tabla_frecuencias.columns = ['categorias', 'frecuencia']

        st.subheader('Exploración visual')
        opciones = ['Barras', 'Pastel', 'Dona', 'Área']
        graf_sel = st.selectbox('¿Qué gráfica quieres ver?', opciones, index=0, key='graf_sel_cat')

        if Tabla_frecuencias.empty:
            st.warning(f"Sin categorías para '{Variable_Cat}' en {ciudad_sel}.")
        else:
            if graf_sel == 'Barras':
                fig = px.bar(Tabla_frecuencias, x='categorias', y='frecuencia',
                             title=f'Frecuencia — {Variable_Cat} ({ciudad_sel})')
            elif graf_sel == 'Pastel':
                fig = px.pie(Tabla_frecuencias, names='categorias', values='frecuencia',
                             title=f'Frecuencia — {Variable_Cat} ({ciudad_sel})')
            elif graf_sel == 'Dona':
                fig = px.pie(Tabla_frecuencias, names='categorias', values='frecuencia', hole=0.45,
                             title=f'Dona — {Variable_Cat} ({ciudad_sel})')
            else:
                tmp = Tabla_frecuencias.sort_values('categorias')
                fig = px.area(tmp, x='categorias', y='frecuencia',
                              title=f'Área — {Variable_Cat} ({ciudad_sel})')
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        cE, cF = st.columns(2)

        # BOXPLOT
        with cE:
            if "price" in df_city.columns and df_city["price"].notna().any():
                cat_para_box = st.selectbox(
                    "Categoría para Boxplot (precio)",
                    options=[c for c in Lista if c in df_city.columns], index=0, key="boxcat"
                )
                df_box = df_city[[cat_para_box, "price"]].dropna()
                top_cats = df_box[cat_para_box].astype("object").value_counts().head(min(top_k, 15)).index
                df_box = df_box[df_box[cat_para_box].astype("object").isin(top_cats)]
                fig_box = px.box(df_box, x=cat_para_box, y="price", points=False,
                                 title=f"Boxplot de precio por {cat_para_box} — {ciudad_sel}")
                fig_box.update_layout(height=480)
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No hay columna de precio válida para el boxplot.")

        # HEATMAP: coocurrencias entre 2 categóricas
        with cF:
            cats_heat = [c for c in Lista if c in df_city.columns]
            if len(cats_heat) >= 2:
                cat_x = st.selectbox("Heatmap — Eje X", options=cats_heat, index=0, key="hx")
                cat_y = st.selectbox("Heatmap — Eje Y", options=cats_heat, index=min(1, len(cats_heat)-1), key="hy")
                t = (
                    df_city[[cat_x, cat_y]].astype("object").fillna("NA")
                    .value_counts().reset_index(name="freq")
                )
                top_x = t[cat_x].value_counts().head(min(top_k, 15)).index
                top_y = t[cat_y].value_counts().head(min(top_k, 15)).index
                t = t[t[cat_x].isin(top_x) & t[cat_y].isin(top_y)]

                fig_hm = px.density_heatmap(
                    t, x=cat_x, y=cat_y, z="freq", color_continuous_scale="Blues",
                    title=f"Heatmap de frecuencias — {cat_x} vs {cat_y} ({ciudad_sel})"
                )
                fig_hm.update_layout(height=480)
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Selecciona al menos dos variables categóricas para el heatmap.")

        # TABLA de frecuencias (toggle)
        if mostrar_tabla:
            st.markdown("### Tabla de frecuencias")
            Tabla_frecuencias = Tabla_frecuencias.reset_index(drop=True)
            Tabla_frecuencias.index = np.arange(1, len(Tabla_frecuencias) + 1)
            Tabla_frecuencias.index.name = "#"
            st.dataframe(Tabla_frecuencias, use_container_width=True)

        # === ANÁLISIS GEOESPACIAL ===
        if {"latitude", "longitude"}.issubset(df_city.columns):
            st.markdown("---")
            st.subheader("Análisis Geoespacial")

            geo_cols = []
            required_cols = ["latitude", "longitude"]
            optional_cols = ["price", "barrio_std", "host_is_superhost", "room_type"]
            
            for col in required_cols:
                if col in df_city.columns:
                    geo_cols.append(col)
            
            for col in optional_cols:
                if col in df_city.columns:
                    geo_cols.append(col)
            
            if "latitude" not in geo_cols or "longitude" not in geo_cols:
                st.warning("No hay datos de ubicación disponibles para esta ciudad.")
            else:
                df_geo = df_city[geo_cols].dropna(subset=["latitude", "longitude"])

                if len(df_geo) == 0:
                    st.warning("No hay datos válidos de ubicación para mostrar en el mapa.")
                else:
                    col_geo1, col_geo2 = st.columns([3, 1])

                    with col_geo2:
                        color_options = []
                        if "price" in df_geo.columns and df_geo["price"].notna().any():
                            color_options.append("price")
                        if "host_is_superhost" in df_geo.columns and df_geo["host_is_superhost"].notna().any():
                            color_options.append("host_is_superhost")
                        if "room_type" in df_geo.columns and df_geo["room_type"].notna().any():
                            color_options.append("room_type")
                        
                        if not color_options:
                            st.warning("No hay columnas válidas para colorear el mapa.")
                            color_by_geo_page = None
                        else:
                            color_by_geo_page = st.selectbox("Colorear por:", color_options, key="color_geo")
                        
                        map_style_page = st.selectbox(
                            "Estilo de mapa:",
                            ["open-street-map", "carto-positron", "carto-darkmatter"],
                            key="map_style"
                        )

                    with col_geo1:
                        max_points = st.slider("Máximo de puntos en el mapa", 500, 5000, 2000, step=250, key="max_points_geo")
                        show_neighborhood_labels = st.checkbox("Mostrar nombres de vecindarios", value=True, key="show_labels")
                        
                        if len(df_geo) > max_points:
                            df_geo_sample = df_geo.sample(max_points, random_state=42)
                        else:
                            df_geo_sample = df_geo.copy()

                        if color_by_geo_page is None:
                            st.info("No hay suficientes datos para renderizar el mapa con colores.")
                        elif color_by_geo_page not in df_geo_sample.columns:
                            st.error(f"La columna '{color_by_geo_page}' no está disponible en los datos.")
                        else:
                            base_hover = {}
                            if "price" in df_geo_sample.columns:
                                base_hover["price"] = ":€,.0f"
                            if "latitude" in df_geo_sample.columns:
                                base_hover["latitude"] = ":.4f"
                            if "longitude" in df_geo_sample.columns:
                                base_hover["longitude"] = ":.4f"
                            if "barrio_std" in df_geo_sample.columns:
                                base_hover["barrio_std"] = False

                            if color_by_geo_page == "price":
                                df_geo_valid = df_geo_sample.dropna(subset=["price"])
                                if len(df_geo_valid) == 0:
                                    st.warning("No hay datos válidos de precio para mostrar en el mapa.")
                                else:
                                    fig_map = px.scatter_mapbox(
                                        df_geo_valid, lat="latitude", lon="longitude",
                                        color="price", size="price",
                                        hover_name="barrio_std" if "barrio_std" in df_geo_valid.columns else None,
                                        hover_data=base_hover,
                                        mapbox_style=map_style_page,
                                        title=f"Distribución Geográfica por Precio - {ciudad_sel} ({len(df_geo_valid):,} puntos)",
                                        height=500,
                                        color_continuous_scale="Viridis",
                                        size_max=15
                                    )
                            elif color_by_geo_page == "host_is_superhost":
                                df_geo_superhost = df_geo_sample[df_geo_sample["host_is_superhost"] == 't'].copy()
                                if len(df_geo_superhost) == 0:
                                    st.warning("No hay superhosts para mostrar en el mapa.")
                                    df_geo_valid = pd.DataFrame()
                                else:
                                    hover_superhost = base_hover.copy()
                                    hover_superhost["host_is_superhost"] = True
                                    fig_map = px.scatter_mapbox(
                                        df_geo_superhost, lat="latitude", lon="longitude",
                                        color="host_is_superhost",
                                        hover_name="barrio_std" if "barrio_std" in df_geo_superhost.columns else None,
                                        hover_data=hover_superhost,
                                        mapbox_style=map_style_page,
                                        title=f"Distribución Geográfica de Superhosts - {ciudad_sel} ({len(df_geo_superhost):,} puntos)",
                                        height=500
                                    )
                                    df_geo_valid = df_geo_superhost
                            else:
                                room_hover = base_hover.copy()
                                if "room_type" in df_geo_sample.columns:
                                    room_hover["room_type"] = True
                                
                                fig_map = px.scatter_mapbox(
                                    df_geo_sample, lat="latitude", lon="longitude",
                                    color="room_type",
                                    hover_name="barrio_std" if "barrio_std" in df_geo_sample.columns else None,
                                    hover_data=room_hover,
                                    mapbox_style=map_style_page,
                                    title=f"Distribución Geográfica por Tipo de Habitación - {ciudad_sel} ({len(df_geo_sample):,} puntos)",
                                    height=500
                                )
                                df_geo_valid = df_geo_sample

                            if 'df_geo_valid' in locals() and len(df_geo_valid) > 0:
                                center_lat = df_geo_valid["latitude"].median()
                                center_lon = df_geo_valid["longitude"].median()
                                
                                lat_range = df_geo_valid["latitude"].max() - df_geo_valid["latitude"].min()
                                lon_range = df_geo_valid["longitude"].max() - df_geo_valid["longitude"].min()
                                max_range = max(lat_range, lon_range)
                                
                                if max_range < 0.1:
                                    zoom_level = 12
                                elif max_range < 0.5:
                                    zoom_level = 10
                                elif max_range < 1.0:
                                    zoom_level = 8
                                else:
                                    zoom_level = 6

                                fig_map.update_layout(
                                    mapbox=dict(
                                        center=dict(lat=center_lat, lon=center_lon), 
                                        zoom=zoom_level
                                    ),
                                    margin=dict(l=0, r=0, t=50, b=0)
                                )
                                
                                if show_neighborhood_labels and "barrio_std" in df_geo_valid.columns:
                                    neighborhood_centers = (
                                        df_geo_valid.groupby("barrio_std")
                                        .agg({
                                            "latitude": "mean",
                                            "longitude": "mean",
                                            "price": "count"
                                        })
                                        .reset_index()
                                        .rename(columns={"price": "count"})
                                    )
                                    
                                    neighborhood_centers = neighborhood_centers[neighborhood_centers["count"] >= 5]
                                    
                                    for _, row in neighborhood_centers.iterrows():
                                        fig_map.add_trace(
                                            go.Scattermapbox(
                                                lat=[row["latitude"]],
                                                lon=[row["longitude"]],
                                                mode="text",
                                                text=[f"{row['barrio_std']}<br>({row['count']} listings)"],
                                                textfont=dict(size=10, color="white"),
                                                showlegend=False,
                                                hoverinfo="skip"
                                            )
                                        )

                                st.plotly_chart(fig_map, use_container_width=True)

    # ============== TAB: COMPARATIVO MULTI-CIUDAD ==============
    with tab_comparativo:
        ciudades_disp = sorted(df["ciudad"].dropna().unique().tolist())
        ciudades_sel = st.sidebar.multiselect(
            "Ciudades a comparar",
            options=ciudades_disp,
            default=ciudades_disp,
            key="ciudades_sel"
        )
        if not ciudades_sel:
            st.warning("Selecciona al menos una ciudad para el análisis comparativo.")
            st.stop()

        df_comp = df[df["ciudad"].isin(ciudades_sel)].copy()

        # ====== Sección: Distribución categórica ======
        st.subheader(f"Distribución de '{Variable_Cat}' por ciudad (Top {top_k})")

        col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns([1.8,1.1,1.1,1.1])
        with col_cfg1:
            tipo_cat = st.selectbox(
                "Tipo de gráfica",
                [
                    "Barras (agrupadas)",
                    "Barras (apiladas)",
                    "Barras (% apiladas)",
                    "Barras por ciudad ",
                    "Pastel "
                ],
                index=0, key="tipo_cat_cmp"
            )
        with col_cfg2:
            mostrar_tabla_cmp = st.checkbox("Mostrar tabla", value=False, key="tabla_cat_cmp")
        with col_cfg3:
            normalizar_top = st.checkbox("Top K global", value=True, key="topk_global_cmp")
        with col_cfg4:
            cols_grid = st.slider("Gráficas por fila (grid)", 2, 4, 4, key="cols_grid_cat")

        if Variable_Cat in df_comp.columns and not df_comp[Variable_Cat].isna().all():
            df_comp["__cat__"] = df_comp[Variable_Cat].astype("object").fillna("NA").astype(str)

            frec = (
                df_comp.groupby(["ciudad","__cat__"]).size()
                .reset_index(name="frecuencia")
            )

            if normalizar_top:
                top_cats = (frec.groupby("__cat__")["frecuencia"].sum()
                                 .sort_values(ascending=False).head(top_k).index.tolist())
                frec = frec[frec["__cat__"].isin(top_cats)]
            else:
                frec["rk"] = frec.groupby("ciudad")["frecuencia"].rank("dense", ascending=False)
                frec = frec[frec["rk"] <= top_k].drop(columns="rk")

            # ---- Helpers de grid ----
            def render_grid(figs, cols_per_row=4, titles=None):
                if cols_per_row < 1: cols_per_row = 1
                for i in range(0, len(figs), cols_per_row):
                    cols = st.columns(min(cols_per_row, len(figs) - i))
                    for j, fig in enumerate(figs[i:i+cols_per_row]):
                        with cols[j]:
                            if titles:
                                st.markdown(f"**{titles[i+j]}**")
                            st.plotly_chart(fig, use_container_width=True)

            if tipo_cat in ["Barras (agrupadas)", "Barras (apiladas)", "Barras (% apiladas)"]:
                # Formatos de barras combinadas
                barmode = "group" if "agrupadas" in tipo_cat else "stack"

                if "% apiladas" in tipo_cat:
                    frec_pct = frec.copy()
                    frec_pct["frecuencia"] = frec_pct.groupby("__cat__")["frecuencia"].transform(
                        lambda s: s / s.sum() * 100
                    )
                    fig_comp_cat = px.bar(
                        frec_pct, x="__cat__", y="frecuencia", color="ciudad",
                        barmode="stack",
                        title=f"Participación (%) por categoría — '{Variable_Cat}'"
                    )
                    fig_comp_cat.update_yaxes(title_text="Porcentaje")
                else:
                    fig_comp_cat = px.bar(
                        frec, x="__cat__", y="frecuencia", color="ciudad",
                        barmode=barmode,
                        title=f"Top {top_k} categorías en '{Variable_Cat}' por ciudad"
                    )
                    fig_comp_cat.update_yaxes(title_text="Frecuencia")

                fig_comp_cat.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
                fig_comp_cat.update_xaxes(title_text="Categoría", automargin=True)
                st.plotly_chart(fig_comp_cat, use_container_width=True)

            elif tipo_cat == "Barras por ciudad (grid ≤4 por fila)":
                # Small multiples: una barra por ciudad
                figs, titles = [], []
                for ctz in ciudades_sel:
                    frec_ctz = frec[frec["ciudad"] == ctz].sort_values("frecuencia", ascending=False)
                    if frec_ctz.empty:
                        continue
                    fig = px.bar(
                        frec_ctz, x="__cat__", y="frecuencia",
                        title=None
                    )
                    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
                    fig.update_traces(type="bar")
                    fig.update_xaxes(title=None, tickangle=45, automargin=True)
                    fig.update_yaxes(title=None)
                    figs.append(fig); titles.append(ctz)
                if figs:
                    render_grid(figs, cols_per_row=cols_grid, titles=titles)

            else:  # "Pastel (grid ≤4 por fila)"
                figs, titles = [], []
                for ctz in ciudades_sel:
                    frec_ctz = frec[frec["ciudad"] == ctz].sort_values("frecuencia", ascending=False)
                    if frec_ctz.empty:
                        continue
                    fig_pie = px.pie(
                        frec_ctz, values="frecuencia", names="__cat__",
                        title=None
                    )
                    fig_pie.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
                    figs.append(fig_pie); titles.append(ctz)
                if figs:
                    render_grid(figs, cols_per_row=cols_grid, titles=titles)

        if mostrar_tabla_cmp:
            st.markdown("### Tabla de frecuencias")
            frec_display = frec.copy()
            frec_display.index = np.arange(1, len(frec_display) + 1)
            frec_display.index.name = "#"
            st.dataframe(frec_display, use_container_width=True)

        st.markdown("---")

        # ====== Sección: Boxplot comparativo ======
        st.subheader("Boxplot de precios por ciudad")
        if "price" in df_comp.columns and df_comp["price"].dropna().shape[0] > 0:
            fig_comp_box = px.box(
                df_comp.dropna(subset=["price"]), x="ciudad", y="price",
                points="suspectedoutliers",
                title="Boxplot de precios por ciudad"
            )
            fig_comp_box.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_comp_box, use_container_width=True)
        else:
            st.info("No hay datos de 'price' suficientes para mostrar boxplot comparativo.")

        st.markdown("---")

        # ====== Sección: Histograma comparativo ======
        st.subheader("Histograma de precios por ciudad")
        if "price" in df_comp.columns and df_comp["price"].dropna().shape[0] > 0:
            col_h1, col_h2 = st.columns([1.2, 1])
            with col_h1:
                nbins_cmp = st.slider("Número de bins (comparativo)", 10, 120, 50, step=5, key="bins_hist_cmp")
            with col_h2:
                modo_hist = st.selectbox("Modo de barras", ["overlay", "stack"], index=0, key="modo_hist_cmp")

            fig_comp_hist = px.histogram(
                df_comp, x="price", color="ciudad", nbins=nbins_cmp,
                barmode=modo_hist,
                title="Histograma comparativo de precios"
            )
            fig_comp_hist.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_comp_hist, use_container_width=True)
        else:
            st.info("No hay datos de 'price' suficientes para mostrar histograma comparativo.")

        st.markdown("---")

        # ====== Sección: Mapa comparativo ======
        st.subheader("Mapa comparativo")
        if {"latitude","longitude"}.issubset(df_comp.columns):
            col_m1, col_m2, col_m3 = st.columns([1.2,1.2,1])
            with col_m1:
                map_style_cmp = st.selectbox(
                    "Estilo de mapa",
                    ["open-street-map", "carto-positron", "carto-darkmatter"],
                    key="map_style_cmp"
                )
            with col_m2:
                # Opciones de coloración disponibles
                color_options_cmp = ["ciudad"]
                if "host_is_superhost" in df_comp.columns:
                    color_options_cmp.append("host_is_superhost")
                if "room_type" in df_comp.columns:
                    color_options_cmp.append("room_type")
                
                color_map_by = st.selectbox("Color por", color_options_cmp, index=0, key="color_map_by_cmp")
            with col_m3:
                usar_tamano_precio = st.checkbox("Tamaño por precio", value=True, key="size_price_cmp")

            # Columnas base requeridas para el mapa
            geo_cols_cmp = ["latitude","longitude","price","barrio_std","ciudad"]
            
            # Agregar columnas opcionales si existen
            if "host_is_superhost" in df_comp.columns:
                geo_cols_cmp.append("host_is_superhost")
            if "room_type" in df_comp.columns:
                geo_cols_cmp.append("room_type")
            
            df_geo_cmp = df_comp[geo_cols_cmp].dropna(subset=["latitude","longitude"]).copy()
            
            # Si se quiere usar tamaño por precio, filtrar NaN en price
            if usar_tamano_precio and "price" in df_geo_cmp.columns:
                df_geo_cmp = df_geo_cmp.dropna(subset=["price"])
            
            max_pts = st.slider("Límite de puntos por mapa", 1000, 10000, 4000, step=500, key="max_pts_cmp")
            if len(df_geo_cmp) > max_pts:
                df_geo_cmp = df_geo_cmp.sample(max_pts, random_state=42)

            # Configurar parámetros según el tipo de color y verificar que price no tenga NaN
            size_kw = dict(size="price", size_max=15) if (usar_tamano_precio and "price" in df_geo_cmp.columns and df_geo_cmp["price"].notna().any()) else {}
            
            if color_map_by == "ciudad":
                # Cuando coloreamos por ciudad (categórico)
                fig_map_cmp = px.scatter_mapbox(
                    df_geo_cmp, lat="latitude", lon="longitude",
                    color="ciudad", 
                    hover_name="ciudad",
                    hover_data={
                        "price": ":€,.0f",
                        "barrio_std": True,
                        "latitude": ":.4f",
                        "longitude": ":.4f",
                        "ciudad": False
                    },
                    mapbox_style=map_style_cmp,
                    title=f"Distribución geográfica de listados ({len(df_geo_cmp):,} puntos)",
                    height=520,
                    **size_kw
                )
            elif color_map_by == "host_is_superhost":
                # Cuando coloreamos por superhost - filtrar solo superhosts verdaderos
                df_geo_cmp_superhost = df_geo_cmp[df_geo_cmp["host_is_superhost"] == 't'].copy()
                fig_map_cmp = px.scatter_mapbox(
                    df_geo_cmp_superhost, lat="latitude", lon="longitude",
                    color="host_is_superhost", 
                    hover_name="host_is_superhost",
                    hover_data={
                        "price": ":€,.0f",
                        "barrio_std": True,
                        "ciudad": True,
                        "latitude": ":.4f",
                        "longitude": ":.4f",
                        "host_is_superhost": False
                    },
                    mapbox_style=map_style_cmp,
                    title=f"Distribución geográfica de Superhosts ({len(df_geo_cmp_superhost):,} puntos)",
                    height=520,
                    **size_kw
                )
            elif color_map_by == "room_type":
                # Cuando coloreamos por tipo de habitación
                fig_map_cmp = px.scatter_mapbox(
                    df_geo_cmp, lat="latitude", lon="longitude",
                    color="room_type", 
                    hover_name="room_type",
                    hover_data={
                        "price": ":€,.0f",
                        "barrio_std": True,
                        "ciudad": True,
                        "latitude": ":.4f",
                        "longitude": ":.4f",
                        "room_type": False
                    },
                    mapbox_style=map_style_cmp,
                    title=f"Distribución geográfica de listados ({len(df_geo_cmp):,} puntos)",
                    height=520,
                    **size_kw
                )
            else:
                # Fallback por si hay otra opción
                fig_map_cmp = px.scatter_mapbox(
                    df_geo_cmp, lat="latitude", lon="longitude",
                    color=color_map_by, 
                    hover_name=color_map_by,
                    hover_data={
                        "price": ":€,.0f",
                        "barrio_std": True,
                        "ciudad": True,
                        "latitude": ":.4f",
                        "longitude": ":.4f"
                    },
                    mapbox_style=map_style_cmp,
                    title=f"Distribución geográfica de listados ({len(df_geo_cmp):,} puntos)",
                    height=520,
                    **size_kw
                )
            
            # Usar el dataframe apropiado para calcular centro y zoom
            if color_map_by == "host_is_superhost":
                df_for_center = df_geo_cmp[df_geo_cmp["host_is_superhost"] == 't']
            else:
                df_for_center = df_geo_cmp
                
            center_lat = df_for_center["latitude"].median()
            center_lon = df_for_center["longitude"].median()
            
            # Calcular zoom dinámico para múltiples ciudades
            lat_range = df_for_center["latitude"].max() - df_for_center["latitude"].min()
            lon_range = df_for_center["longitude"].max() - df_for_center["longitude"].min()
            max_range = max(lat_range, lon_range)
            
            if len(ciudades_sel) == 1:
                zoom_level = 10
            elif max_range < 1:
                zoom_level = 8
            elif max_range < 5:
                zoom_level = 6
            elif max_range < 20:
                zoom_level = 4
            else:
                zoom_level = 2
            
            fig_map_cmp.update_layout(
                mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=zoom_level),
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            # Agregar etiquetas de ciudades en el mapa comparativo
            # Usar el dataframe apropiado según el filtro seleccionado
            if color_map_by == "host_is_superhost":
                df_for_labels = df_geo_cmp[df_geo_cmp["host_is_superhost"] == 't']
            else:
                df_for_labels = df_geo_cmp
                
            city_centers = (
                df_for_labels.groupby("ciudad")
                .agg({
                    "latitude": "mean",
                    "longitude": "mean",
                    "price": ["count", "mean"]
                })
                .round(2)
            )
            city_centers.columns = ["lat_center", "lon_center", "count", "avg_price"]
            city_centers = city_centers.reset_index()
            
            # Agregar etiquetas de ciudades
            for _, row in city_centers.iterrows():
                fig_map_cmp.add_trace(
                    go.Scattermapbox(
                        lat=[row["lat_center"]],
                        lon=[row["lon_center"]],
                        mode="text",
                        text=[f"<b>{row['ciudad']}</b><br>{row['count']} listings<br>Avg: €{row['avg_price']:,.0f}"],
                        textfont=dict(size=12, color="white"),
                        showlegend=False,
                        hoverinfo="skip"
                    )
                )
            
            st.plotly_chart(fig_map_cmp, use_container_width=True)
        else:
            st.info("No hay columnas de latitud/longitud para el mapa comparativo.")

        st.markdown("---")

        # ====== Sección: Análisis Comparativo de Competitividad ======
        st.subheader("Análisis Comparativo de Competitividad")
        
        if len(df_comp) > 0:
            # Función para calcular métricas de competitividad por ciudad
            def calculate_city_competitiveness(city_data):
                metrics = {}
                city_name = city_data['ciudad'].iloc[0] if 'ciudad' in city_data.columns else 'Unknown'
                
                # Disponibilidad (usando availability_365 - más realista)
                if 'availability_365' in city_data.columns and city_data['availability_365'].notna().any():
                    # Calcular % de disponibilidad basado en días disponibles del año
                    avg_availability_days = city_data['availability_365'].mean()
                    availability_pct = (avg_availability_days / 365) * 100
                    # Para competitividad: menos disponible = más ocupado = más competitivo
                    availability_score = 100 - availability_pct
                else:
                    # Fallback más realista para Airbnb
                    availability_pct = 65  # ~65% disponibilidad es típico
                    availability_score = 35
                
                # Profesionalismo (corregido)
                prof_components = []
                if 'host_is_superhost' in city_data.columns:
                    superhost_pct = (city_data['host_is_superhost'] == 't').mean() * 100
                    prof_components.append(superhost_pct * 0.4 / 100)
                if 'host_identity_verified' in city_data.columns:
                    verified_pct = (city_data['host_identity_verified'] == 't').mean() * 100
                    prof_components.append(verified_pct * 0.3 / 100)
                if 'host_response_time' in city_data.columns:
                    fast_response_pct = (city_data['host_response_time'] == 'within an hour').mean() * 100
                    prof_components.append(fast_response_pct * 0.3 / 100)
                
                prof_score = sum(prof_components) * 100 if prof_components else 0
                
                # Flexibilidad (corregida)
                flex_components = []
                if 'instant_bookable' in city_data.columns:
                    instant_pct = (city_data['instant_bookable'] == 't').mean() * 100
                    flex_components.append(instant_pct * 0.5 / 100)
                if 'cancellation_policy' in city_data.columns:
                    flexible_policy_pct = (city_data['cancellation_policy'] == 'flexible').mean() * 100
                    flex_components.append(flexible_policy_pct * 0.5 / 100)
                
                flex_score = sum(flex_components) * 100 if flex_components else 0
                
                # Servicios (corregido)
                if 'amenities_count' in city_data.columns and city_data['amenities_count'].notna().any():
                    amenities_score = min(city_data['amenities_count'].mean() / 15 * 100, 100)
                else:
                    amenities_score = 0
                
                # Variación de precios (corregida)
                if 'price' in city_data.columns and city_data['price'].notna().any():
                    price_cv = (city_data['price'].std() / city_data['price'].mean()) * 100 if city_data['price'].mean() > 0 else 0
                    price_score = min(price_cv / 50 * 100, 100)
                else:
                    price_score = 0
                
                # Índice compuesto (sin disponibilidad, corregido)
                competitiveness_index = (
                    prof_score * 0.40 +
                    flex_score * 0.30 +
                    amenities_score * 0.20 +
                    price_score * 0.10
                )
                
                return {
                    'Ciudad': city_name,
                    'Índice_Competitividad': competitiveness_index,
                    'Disponibilidad': availability_score,
                    'Profesionalismo': prof_score,
                    'Flexibilidad': flex_score,
                    'Servicios': amenities_score,
                    'Variación_Precios': price_score,
                    'Total_Listings': len(city_data),
                    'Precio_Promedio': city_data['price'].mean() if 'price' in city_data.columns else 0,
                    'Superhosts_Pct': (city_data['host_is_superhost'] == 't').mean() * 100 if 'host_is_superhost' in city_data.columns else 0,
                    'Disponibilidad_Pct': availability_pct
                }

            # Calcular métricas para todas las ciudades
            competitiveness_results = []
            for ciudad in ciudades_sel:
                city_data = df_comp[df_comp['ciudad'] == ciudad]
                if len(city_data) > 0:
                    competitiveness_results.append(calculate_city_competitiveness(city_data))

            if competitiveness_results:
                comp_df = pd.DataFrame(competitiveness_results)
                comp_df = comp_df.round(2)
                
                # Ranking de competitividad
                comp_df_sorted = comp_df.sort_values('Índice_Competitividad', ascending=False)
                
                # Mostrar tabla de rankings
                display_cols = ['Ciudad', 'Total_Listings', 'Superhosts_Pct', 
                               'Disponibilidad_Pct', 'Precio_Promedio']
                st.dataframe(comp_df_sorted[display_cols].reset_index(drop=True), use_container_width=True)

            else:
                st.info("No hay suficientes datos para el análisis de competitividad.")
        else:
            st.info("No hay datos disponibles para el análisis comparativo.")

        st.markdown("---")

        # ====== Comparativo Multi-Ciudad ROI ======
        st.markdown(
            "<h3 style='text-align: center;'>Comparativo Multi-Ciudad ROI</h3>", 
            unsafe_allow_html=True
        )
        # Parámetros de simulación centrados
        st.markdown(
            "<h4 style='text-align: center;'>Parámetros de simulación</h4>", 
            unsafe_allow_html=True
        )
        
        # Input centrado
        col_empty1, col_input, col_empty2 = st.columns([1, 2, 1])
        with col_input:
            gastos_mensuales_input_comp = st.number_input(
                "Gastos operativos mensuales (€)", 
                min_value=200, 
                max_value=3000, 
                value=800, 
                step=50, 
                key="gastos_comp_multi",
                help="Gastos fijos: seguros, mantenimiento, servicios, impuestos, etc. NO incluye comisiones Airbnb."
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Resultados centrados
        if len(ciudades_sel) > 0:
            st.markdown(
                "<h4 style='text-align: center;'>Ranking ROI por Ciudad</h4>", 
                unsafe_allow_html=True
            )
            
            # Calcular ROI para todas las ciudades
            roi_ciudades_comp = []
            
            for ciudad in ciudades_sel:
                    ciudad_data = df_comp[df_comp['ciudad'] == ciudad].copy()
                    
                    if len(ciudad_data) > 0:
                        # Limpiar precios
                        if 'price' in ciudad_data.columns:
                            ciudad_data['price_clean'] = ciudad_data['price'].astype(str).str.replace('$', '').str.replace(',', '')
                            ciudad_data['price_clean'] = pd.to_numeric(ciudad_data['price_clean'], errors='coerce')
                            precio_promedio_ciudad = ciudad_data['price_clean'].mean()
                        else:
                            precio_promedio_ciudad = 0
                        
                        # Ocupación realista por ciudad (basada en datos de mercado real)
                        ocupacion_rates = {
                            'Barcelona': 0.68,  # 68% ocupación
                            'Amsterdam': 0.62,  # 62% ocupación (menor por regulaciones)
                            'Milan': 0.65,      # 65% ocupación
                            'Athens': 0.70,     # 70% ocupación (destino más barato)
                            'Madrid': 0.67      # 67% ocupación
                        }
                        
                        ocupacion_pct_ciudad = ocupacion_rates.get(ciudad, 0.65) * 100
                        dias_ocupados_ciudad = ocupacion_pct_ciudad / 100 * 365
                        
                        # Revenue anual bruto
                        revenue_bruto_ciudad = precio_promedio_ciudad * dias_ocupados_ciudad
                        
                        # Comisiones y costos variables (% del revenue bruto)
                        comision_airbnb = revenue_bruto_ciudad * 0.15  # 15% comisión Airbnb
                        costos_limpieza = revenue_bruto_ciudad * 0.05   # 5% limpieza y servicios
                        
                        # Revenue neto (después de comisiones)
                        revenue_neto_ciudad = revenue_bruto_ciudad - comision_airbnb - costos_limpieza
                        
                        # Gastos operativos fijos anuales
                        gastos_operativos_ciudad = gastos_mensuales_input_comp * 12
                        
                        # Ganancia neta anual
                        ganancia_neta_ciudad = revenue_neto_ciudad - gastos_operativos_ciudad
                        
                        # Inversión inicial estimada (para ROI real)
                        # Estimación: 6-10 meses de gastos + setup inicial por ciudad
                        setup_costs = {
                            'Barcelona': 8000,   # Costos altos de setup
                            'Amsterdam': 12000,  # Muy regulado, costos altos
                            'Milan': 7000,       # Costos medios
                            'Athens': 5000,      # Costos más bajos
                            'Madrid': 6500       # Costos medios
                        }
                        
                        inversion_inicial = setup_costs.get(ciudad, 7000) + (gastos_mensuales_input_comp * 8)
                        
                        # ROI real = (Ganancia Neta Anual / Inversión Inicial) * 100
                        roi_ciudad = (ganancia_neta_ciudad / inversion_inicial * 100) if inversion_inicial > 0 else 0
                        
                        roi_ciudades_comp.append({
                            'Ciudad': ciudad,
                            'Precio_Promedio': precio_promedio_ciudad,
                            'Ocupacion_Pct': ocupacion_pct_ciudad,
                            'Dias_Ocupados': dias_ocupados_ciudad,
                            'Revenue_Bruto': revenue_bruto_ciudad,
                            'Revenue_Neto': revenue_neto_ciudad,
                            'Inversion_Inicial': inversion_inicial,
                            'Ganancia_Neta': ganancia_neta_ciudad,
                            'ROI': roi_ciudad
                        })
            
            # Ordenar por ROI
            roi_ciudades_comp.sort(key=lambda x: x['ROI'], reverse=True)
            
            # Crear tabla compacta de resultados
            roi_df = pd.DataFrame(roi_ciudades_comp)
            roi_df['Ranking'] = range(1, len(roi_df) + 1)
            
            # Formato compacto - Cards horizontales
            for idx, ciudad_roi in enumerate(roi_ciudades_comp):
                # Determinar color según ROI
                if ciudad_roi['ROI'] > 100:
                    bg_color = "#E8F5E9"
                    border_color = "#4CAF50"
                elif ciudad_roi['ROI'] > 50:
                    bg_color = "#E3F2FD"
                    border_color = "#2196F3"
                elif ciudad_roi['ROI'] > 0:
                    bg_color = "#FFF3E0"
                    border_color = "#FF9800"
                else:
                    bg_color = "#FFEBEE"
                    border_color = "#F44336"
                
                st.markdown(f"""
                <div style='background: {bg_color}; padding: 16px; border-radius: 8px; border-left: 4px solid {border_color}; margin-bottom: 12px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div style='flex: 1;'>
                            <strong style='font-size: 16px;'>#{idx+1}: {ciudad_roi['Ciudad']}</strong>
                            <div style='color: #666; font-size: 12px; margin-top: 4px;'>
                                €{ciudad_roi['Precio_Promedio']:.0f}/noche | {ciudad_roi['Ocupacion_Pct']:.0f}% ocupación ({ciudad_roi['Dias_Ocupados']:.0f} días/año)
                            </div>
                        </div>
                        <div style='flex: 1; text-align: center;'>
                            <div style='color: {border_color}; font-size: 24px; font-weight: bold;'>ROI: {ciudad_roi['ROI']:.1f}%</div>
                        </div>
                        <div style='flex: 1; text-align: right;'>
                            <div style='font-size: 14px; margin-bottom: 4px;'><strong>Ganancia Neta/Año</strong></div>
                            <div style='font-size: 18px; font-weight: bold;'>€{ciudad_roi['Ganancia_Neta']:,.0f}</div>
                        </div>
                        <div style='flex: 1; text-align: right; padding-left: 20px;'>
                            <div style='font-size: 14px; margin-bottom: 4px;'><strong>Revenue Neto</strong></div>
                            <div style='font-size: 18px; font-weight: bold;'>€{ciudad_roi['Revenue_Neto']:,.0f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("<br><br>", unsafe_allow_html=True)
            col_empty1, col_warning, col_empty2 = st.columns([1, 2, 1])
            with col_warning:
                st.warning("Selecciona al menos una ciudad para calcular ROI.")

    # === FIN DE TAB COMPARATIVO ===

# =========================================================
# ================== REGRESIÓN LINEAL =====================
# =========================================================
if View == "Regresión Lineal":

    col_title, col_help = st.columns([0.95, 0.05])
    with col_title:
        st.title("Airbnb - Regresión Lineal")
    with col_help:
        with st.popover("➕"):
            st.markdown("**¿Qué es la Regresión Lineal?**")
            st.write("Método estadístico que predice el precio de un Airbnb basándose en relaciones lineales con variables como número de huéspedes, baños, o amenidades. Ayuda a entender qué factores incrementan o disminuyen el precio de manera proporcional.")

    # Crear tabs para organizar el contenido
    tab_analisis, tab_contexto = st.tabs(["Análisis y Gráficas", "Correlaciones"])

    # --- Selección de ciudades desde el df unificado ---
    if "ciudad" not in df.columns or df["ciudad"].dropna().empty:
        st.warning("No hay columna 'ciudad' válida en el DataFrame.")
        st.stop()

    ciudades_disp = sorted(df["ciudad"].dropna().unique().tolist())
    st.sidebar.header("Ciudades")
    selected_cities = st.sidebar.multiselect(
        "Selecciona de 1 a 5 ciudades",
        options=ciudades_disp,
        default=ciudades_disp[:min(3, len(ciudades_disp))],
        max_selections=5,
        key="cities_regresion_lineal"
    )

    if not selected_cities:
        st.info("Selecciona al menos una ciudad para continuar.")
        st.stop()

    df_combined = df[df["ciudad"].isin(selected_cities)].copy()

    # --- Columnas numéricas candidatas ---
    numeric_cols = df_combined.select_dtypes(include="number").columns.tolist()
    # Excluir 'id' de las variables numéricas
    numeric_cols = [c for c in numeric_cols if c.lower() != 'id']
    default_y = "price" if "price" in numeric_cols else (numeric_cols[0] if numeric_cols else None)
    default_x = "accommodates" if "accommodates" in numeric_cols else (
        "amenities_count" if "amenities_count" in numeric_cols else (
            numeric_cols[1] if len(numeric_cols) > 1 else None
        )
    )

    if len(numeric_cols) < 2 or default_x is None or default_y is None:
        st.warning("Se requieren al menos 2 variables numéricas para la regresión.")
        st.stop()

    # --- Variables de regresión ---
    st.sidebar.header("Variables de regresión")
    x_var = st.sidebar.selectbox("Variable independiente (x)", numeric_cols, index=numeric_cols.index(default_x), key="x_var_rl")
    
    with tab_analisis:
        restantes_para_y = [c for c in numeric_cols if c != x_var]
        y_idx = restantes_para_y.index(default_y) if default_y in restantes_para_y else 0
        y_var = st.sidebar.selectbox("Variable dependiente (y)", restantes_para_y, index=y_idx, key="y_var_rl")

        remaining_vars = [col for col in numeric_cols if col not in [x_var, y_var]]
        max_predictors = min(15, len(remaining_vars))
        num_x = st.sidebar.slider(
            "¿Cuántas x adicionales (regresión múltiple)?", 
            0, max_predictors, min(2, max_predictors),
            help="Variables adicionales a la x principal",
            key="num_x_rl"
        )

        st.sidebar.header("Vista por ciudad (múltiple)")
        ciudad_focus = st.sidebar.selectbox(
            "Ciudad para el análisis detallado",
            options=["Todas"] + selected_cities,
            index=0,
            help="Elige una ciudad para la gráfica de regresión múltiple y sus métricas.",
            key="ciudad_focus_rl"
        )

        st.sidebar.markdown(f"**Total de variables independientes: {num_x + 1}** (1 principal + {num_x} adicionales)")

        x_multi_vars = []
        for i in range(num_x):
            opciones = [col for col in remaining_vars if col not in x_multi_vars]
            x_i = st.sidebar.selectbox(f"Variable x{i+2} (Total: {i+2} variables)", opciones, key=f"x{i+2}_lin")
            x_multi_vars.append(x_i)

        # --- Limpieza de datos (mantener ciudad) ---
        cols_needed = [x_var, y_var, "ciudad"]
        df_clean = df_combined[cols_needed].dropna()
        if df_clean.empty:
            st.warning("No hay datos válidos después de limpiar NaN para las variables seleccionadas.")
            st.stop()

        # --- Unidades ---
        def get_unit(var_name):
            units = {
                'price': '€','accommodates':'huéspedes','bedrooms':'habitaciones','beds':'camas',
                'bathrooms':'baños','bathrooms_num':'baños','amenities_count':'amenidades',
                'minimum_nights':'noches','maximum_nights':'noches','availability_365':'días',
                'number_of_reviews':'reseñas','reviews_per_month':'reseñas/mes',
                'review_scores_rating':'puntos','review_scores_accuracy':'puntos',
                'review_scores_cleanliness':'puntos','review_scores_checkin':'puntos',
                'review_scores_communication':'puntos','review_scores_location':'puntos',
                'review_scores_value':'puntos','calculated_host_listings_count':'propiedades',
                'latitude':'°','longitude':'°','superhost_numeric':'(0=No, 1=Sí)','price_per_person':'€/persona'
            }
            return units.get(var_name, '')

        # Función auxiliar para interpretar correlación
        def interpretar_correlacion(r):
            if r == "N/A" or not isinstance(r, (int, float)):
                return "N/A", "N/A"
            direccion = "Positiva" if r > 0 else ("Negativa" if r < 0 else "Nula")
            r_abs = abs(r)
            if r_abs >= 0.9:   fuerza = "Muy fuerte"
            elif r_abs >= 0.7: fuerza = "Fuerte"
            elif r_abs >= 0.5: fuerza = "Moderada"
            elif r_abs >= 0.3: fuerza = "Débil"
            else:              fuerza = "Muy débil"
            return direccion, fuerza

        # ===================== GRÁFICAS =====================
        st.subheader("Comparación visual")
        
        # Definir función auxiliar para regresión simple
        def _plot_reg_simple_ciudad(sub_df: pd.DataFrame, ciudad: str, x_var: str, y_var: str):
            Xc = sub_df[[x_var]].values
            yc = sub_df[y_var].values
            model_c = LinearRegression()
            model_c.fit(Xc, yc)
            yhat_c = model_c.predict(Xc)
            order = np.argsort(Xc.ravel())
            x_sorted = Xc.ravel()[order]
            y_sorted = yhat_c[order]
            r2c = r2_score(yc, yhat_c)
            rmse_c = np.sqrt(mean_squared_error(yc, yhat_c))
            mae_c = mean_absolute_error(yc, yhat_c)
            
            # Calcular correlación
            corr_c = sub_df[[x_var, y_var]].corr().iloc[0, 1]
            direccion_c, fuerza_c = interpretar_correlacion(corr_c)

            figc, axc = plt.subplots(figsize=(6, 4))
            axc.scatter(sub_df[x_var], sub_df[y_var], alpha=0.6)
            axc.plot(x_sorted, y_sorted, color='red', linewidth=2.5, linestyle='-', zorder=5, label="Línea de regresión")
            axc.set_title(f"{ciudad}: {y_var} vs {x_var}", fontsize=12)
            axc.set_xlabel(x_var); axc.set_ylabel(y_var)
            axc.legend(); axc.grid(True, alpha=0.3)
            st.pyplot(figc)
            
            # Métricas debajo de la gráfica
            st.markdown(f"• **R²:** {r2c:.4f} · **RMSE:** {rmse_c:.3f} · **MAE:** {mae_c:.3f} · **Correlación:** {corr_c:.4f} ({direccion_c}, {fuerza_c})")
        
        # Modo de visualización arriba de las columnas
        modo_grafica = st.radio(
            "Modo de visualización",
            ["Pestañas por ciudad", "Todas juntas"],
            horizontal=True,
            key="modo_reg_simple"
        )
        
        col_simple, col_multi = st.columns(2)

        # ====== REGRESIÓN SIMPLE ======
        with col_simple:
            st.markdown("**Regresión simple**")

            if modo_grafica == "Todas juntas":
                X_simple = df_clean[[x_var]].values
                y_simple = df_clean[y_var].values
                model_simple = LinearRegression()
                try:
                    model_simple.fit(X_simple, y_simple)
                    y_pred_simple = model_simple.predict(X_simple)
                    order = np.argsort(X_simple.ravel())
                    x_sorted = X_simple.ravel()[order]
                    y_sorted = y_pred_simple[order]
                    r2_value = r2_score(y_simple, y_pred_simple)

                    # métricas globales de simple (para tarjetas)
                    rmse_simple = float(np.sqrt(mean_squared_error(y_simple, y_pred_simple)))
                    mae_simple  = float(mean_absolute_error(y_simple, y_pred_simple))
                    
                    # Calcular correlación global
                    corr_simple = df_clean[[x_var, y_var]].corr().iloc[0, 1]
                    direccion_simple_all, fuerza_simple_all = interpretar_correlacion(corr_simple)

                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=df_clean[x_var], y=df_clean[y_var], hue=df_clean["ciudad"], ax=ax1, alpha=0.6)
                    ax1.plot(x_sorted, y_sorted, color='red', label="Línea de regresión", linewidth=3, linestyle='-', zorder=5)
                    ax1.set_title(f"{y_var} vs {x_var}")
                    ax1.set_xlabel(x_var); ax1.set_ylabel(y_var)
                    ax1.legend(); ax1.grid(True, alpha=0.3)
                    st.pyplot(fig1)
                    
                    # Métricas debajo de la gráfica
                    st.markdown(f"• **R²:** {r2_value:.4f} · **RMSE:** {rmse_simple:.3f} · **MAE:** {mae_simple:.3f} · **Correlación:** {corr_simple:.4f} ({direccion_simple_all}, {fuerza_simple_all})")
                except Exception as e:
                    st.error("No se pudo ajustar la regresión simple.")
                    st.exception(e)

            elif modo_grafica == "Pestañas por ciudad":
                tabs = st.tabs(selected_cities)
                for tab, ciudad in zip(tabs, selected_cities):
                    with tab:
                        sub = df_clean[df_clean["ciudad"] == ciudad][[x_var, y_var]].dropna()
                        if len(sub) < 2 or sub[x_var].nunique() < 2:
                            st.warning(f"{ciudad}: datos insuficientes para ajustar la regresión.")
                            continue
                        _plot_reg_simple_ciudad(sub, ciudad, x_var, y_var)

            else:  # "Cuadrícula por ciudad"
                n_cols = st.slider("Columnas de la cuadrícula", 2, 4, min(3, max(2, len(selected_cities))))
                cols = st.columns(n_cols)
                for i, ciudad in enumerate(selected_cities):
                    sub = df_clean[df_clean["ciudad"] == ciudad][[x_var, y_var]].dropna()
                    if len(sub) < 2 or sub[x_var].nunique() < 2:
                        with cols[i % n_cols]:
                            st.warning(f"{ciudad}: datos insuficientes para ajustar la regresión.")
                        continue
                    with cols[i % n_cols]:
                        _plot_reg_simple_ciudad(sub, ciudad, x_var, y_var)

            # Cálculo silencioso de métricas simples globales (si no las generó el modo "Todas juntas")
            if 'rmse_simple' not in locals() or 'mae_simple' not in locals():
                try:
                    X_tmp = df_clean[[x_var]].values
                    y_tmp = df_clean[y_var].values
                    mtmp = LinearRegression().fit(X_tmp, y_tmp)
                    yhat_tmp = mtmp.predict(X_tmp)
                    rmse_simple = float(np.sqrt(mean_squared_error(y_tmp, yhat_tmp)))
                    mae_simple  = float(mean_absolute_error(y_tmp, yhat_tmp))
                except Exception:
                    rmse_simple = None
                    mae_simple = None
        # ====== REGRESIÓN MÚLTIPLE (sobre eje X; por ciudad) ======
        with col_multi:
            st.markdown("**Regresión múltiple — vista sobre eje X (por ciudad)**")
            selected_predictors = [x_var] + x_multi_vars

            # Filtrado por ciudad para el análisis detallado
            if ciudad_focus != "Todas":
                df_scope = df_combined[df_combined["ciudad"] == ciudad_focus]
                titulo_ciudad = f" — {ciudad_focus}"
            else:
                df_scope = df_combined
                titulo_ciudad = " — (todas las ciudades)"

            df_multi = df_scope[[y_var] + selected_predictors].apply(pd.to_numeric, errors="coerce").dropna()

            if df_multi.empty or len(selected_predictors) < 1:
                st.info("Selecciona al menos 1 variable independiente y verifica que haya datos válidos.")
                # Inicializa KPIs para evitar N/A ruidoso
                r2_multi = None
                r2_adj_multi = None
                rmse_multi = None
                mae_multi = None
                coef_multi = None
                intercept_multi = None
            else:
                # Ajuste del modelo múltiple en el scope elegido
                X = df_multi[selected_predictors].values
                y = df_multi[y_var].values
                model = LinearRegression().fit(X, y)
                y_hat = model.predict(X)

                plot_df = df_multi.copy()
                plot_df["y_real"] = y
                plot_df["y_pred_multi"] = y_hat

                # --- Controles de visual ---
                alpha_scatter = 0.6
                colorear_resid = st.checkbox(
                    "Colorear Y real por residuo (Ŷ − Y)",
                    value=True,
                    key="color_resid_multi"
                )

                # === Gráfica principal: SOLO puntos de Y real y Ŷ (predicho) ===
                fig, ax = plt.subplots(figsize=(10, 6))

                if colorear_resid:
                    resid = plot_df["y_pred_multi"] - plot_df["y_real"]
                    sc = ax.scatter(
                        plot_df[x_var], plot_df["y_real"],
                        c=resid, s=26, alpha=alpha_scatter,
                        marker="o", label="Y real"
                    )
                    cbar = plt.colorbar(sc, ax=ax)
                    cbar.set_label("Residuo (Ŷ − Y)")
                else:
                    ax.scatter(
                        plot_df[x_var], plot_df["y_real"],
                        s=26, alpha=alpha_scatter,
                        marker="o", label="Y real"
                    )

                # Puntos de predicción múltiple (Ŷ)
                ax.scatter(
                    plot_df[x_var], plot_df["y_pred_multi"],
                    s=30, alpha=0.9,
                    marker="^", label="Ŷ (múltiple)"
                )

                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)
                ax.set_title(f"{y_var} sobre {x_var}{titulo_ciudad}: puntos de Y real y Ŷ")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
                st.pyplot(fig)

                # Métricas del modelo en el scope (ciudad/todas) y export a KPIs
                r2 = r2_score(y, y_hat)
                rmse = float(np.sqrt(mean_squared_error(y, y_hat)))
                mae = float(mean_absolute_error(y, y_hat))
                n = len(y)
                k = len(selected_predictors)
                r2_adj = 1 - ((1 - r2) * (n - 1) / (n - k - 1)) if n > k + 1 else r2

                r2_multi = r2
                r2_adj_multi = r2_adj
                rmse_multi = rmse
                mae_multi = mae
                coef_multi = np.array([model.coef_[i] for i in range(len(selected_predictors))])
                intercept_multi = float(model.intercept_)
                
                # Correlación múltiple (raíz de R²)
                r_multi = np.sqrt(r2)
                direccion_multi_graf, fuerza_multi_graf = interpretar_correlacion(r_multi)

                st.markdown(
                    f"• **R²**: {r2:.4f} · **R² ajustado**: {r2_adj:.4f} · "
                    f"**RMSE**: {rmse:.3f} · **MAE**: {mae:.3f} · **Correlación múltiple**: {r_multi:.4f} ({direccion_multi_graf}, {fuerza_multi_graf})"
                )

        # ===================== UTILIDADES DE UI =====================
        st.markdown("""
        <style>
        .card{background:#F7F7F7;border:1px solid rgba(0,0,0,.08);border-radius:16px;
            padding:14px 16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.05);}
        .card h4{margin:0 0 8px 0;font-weight:800;color:#FF385C;}
        .kpi{display:flex;gap:14px;flex-wrap:wrap}
        .kpi .metric{background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                    padding:10px 12px;min-width:160px}
        .kpi .metric .label{font-size:12px;color:#666}
        .kpi .metric .value{font-size:20px;font-weight:700;color:#666}
        </style>
        """, unsafe_allow_html=True)

        def _metric_html(label, value):
            return f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>'

        # Ecuación global simple
        def calcular_resultados(df_in: pd.DataFrame, x_col: str, y_col: str):
            dfx = df_in[[x_col, y_col]].dropna()
            if dfx[x_col].nunique() <= 1:
                return {"n": len(dfx), "r2": 0.0, "corr": 0.0, "beta0": 0.0, "beta1": 0.0}
            try:
                X = sm.add_constant(dfx[x_col].values)
                y = dfx[y_col].values
                modelo = sm.OLS(y, X).fit()
                r2 = float(modelo.rsquared) if np.isfinite(modelo.rsquared) else 0.0
                corr = float(dfx[x_col].corr(dfx[y_col])) if dfx[[x_col, y_col]].notna().all().all() else 0.0
                beta0 = float(modelo.params[0]) if len(modelo.params) > 0 else 0.0
                beta1 = float(modelo.params[1]) if len(modelo.params) > 1 else 0.0
                return {"n": int(modelo.nobs), "r2": r2, "corr": corr, "beta0": beta0, "beta1": beta1}
            except Exception:
                return {"n": len(dfx), "r2": 0.0, "corr": 0.0, "beta0": 0.0, "beta1": 0.0}

        res_ecuacion = calcular_resultados(df_combined, x_var, y_var)
        beta0_txt = f"{res_ecuacion['beta0']:.4f}"
        beta1_txt = f"{res_ecuacion['beta1']:.4f}"
        r2_simple_txt = f"{res_ecuacion['r2']:.4f}"
        corr_simple_txt = f"{res_ecuacion['corr']:.4f}"
        signo = "+" if res_ecuacion['beta0'] >= 0 else ""
        ecuacion_txt = f"{y_var} = {beta1_txt} × {x_var} {signo} {beta0_txt}"

        st.markdown(
            '<div class="card"><h4>Ecuación de Regresión Simple</h4><div class="kpi">'
            + _metric_html("β₀ (Intercepto)", beta0_txt)
            + _metric_html("β₁ (Pendiente)", beta1_txt)
            + _metric_html("R²", r2_simple_txt)
            + f'</div><div style="margin-top:12px;padding:10px;background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;font-size:16px;font-weight:600;text-align:center;">{ecuacion_txt}</div></div>',
            unsafe_allow_html=True
        )

        # Métricas de modelos (usan las variables calculadas arriba)
        rmse_simple_txt = f"{rmse_simple:.4f}" if 'rmse_simple' in locals() and rmse_simple is not None else "N/A"
        rmse_multi_txt  = f"{rmse_multi:.4f}"  if 'rmse_multi' in locals()  and rmse_multi  is not None else "N/A"
        mae_simple_txt  = f"{mae_simple:.4f}"  if 'mae_simple' in locals()  and mae_simple  is not None else "N/A"
        mae_multi_txt   = f"{mae_multi:.4f}"   if 'mae_multi' in locals()   and mae_multi   is not None else "N/A"
        r2_multi_txt    = f"{r2_multi:.4f}"    if 'r2_multi' in locals()    and r2_multi    is not None else "N/A"
        r2_adj_multi_txt= f"{r2_adj_multi:.4f}"if 'r2_adj_multi' in locals()and r2_adj_multi is not None else "N/A"

        # Si hay múltiple, mostramos ecuación y R²
        if 'coef_multi' in locals() and coef_multi is not None and 'intercept_multi' in locals() and intercept_multi is not None:
            intercept_multi_txt = f"{intercept_multi:.4f}"
            selected_predictors = [x_var] + x_multi_vars
            terminos = [f"{coef:.4f} × {var}" for var, coef in zip(selected_predictors, coef_multi)]
            signo_int = "+" if intercept_multi >= 0 else ""
            ecuacion_multi_txt = f"{y_var} = {' + '.join(terminos)} {signo_int} {intercept_multi_txt}"

            coef_metrics = _metric_html("β₀ (Intercepto)", intercept_multi_txt)
            for i, (var, coef) in enumerate(zip(selected_predictors, coef_multi)):
                coef_metrics += _metric_html(f"β{i+1} ({var})", f"{coef:.4f}")
            coef_metrics += _metric_html("R² múltiple", r2_multi_txt)
            coef_metrics += _metric_html("R² Ajustado", r2_adj_multi_txt)

            st.markdown(
                '<div class="card"><h4>Ecuación de Regresión Múltiple</h4><div class="kpi">'
                + coef_metrics
                + f'</div><div style="margin-top:12px;padding:10px;background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;font-size:14px;font-weight:600;text-align:center;">{ecuacion_multi_txt}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.info("Modelo de regresión múltiple no disponible. Asegúrate de seleccionar variables adicionales.")

        # =================== RESULTADOS POR CIUDAD (CARDS) ===================
        st.subheader("Resultados por ciudad")
        cols_cards = st.columns(3)

        for i, ciudad in enumerate(selected_cities):
            res = calcular_resultados(df_combined[df_combined["ciudad"] == ciudad], x_var, y_var)
            r2_txt = f"{res['r2']:.4f}"
            corr_txt = f"{res['corr']:.4f}"
            direccion, fuerza = interpretar_correlacion(res['corr'])
            beta0_txt = f"{res['beta0']:.4f}"
            beta1_txt = f"{res['beta1']:.4f}"
            signo = "+" if res['beta0'] >= 0 else ""
            ecuacion = f"ŷ = {beta1_txt}x {signo} {beta0_txt}"

            with cols_cards[i % 3]:
                st.markdown(
                    '<div class="card"><h4>'+ ciudad +'</h4><div class="kpi">'
                    + _metric_html("R²", r2_txt)
                    + _metric_html("Correlación", corr_txt)
                    + _metric_html("Dirección", direccion)
                    + _metric_html("Fuerza", fuerza)
                    + _metric_html("β₀", beta0_txt)
                    + _metric_html("β₁", beta1_txt)
                    + f'</div><div style="margin-top:8px;padding:8px;background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:8px;font-size:13px;font-weight:600;text-align:center;">{ecuacion}</div></div>',
                    unsafe_allow_html=True
                )

        with tab_contexto:
                    # ================== HEATMAPS DE CORRELACIÓN POR CIUDAD ==================
                    st.subheader("Matriz de Correlación por Ciudad")
                    st.markdown("**Selecciona las variables a incluir en la matriz de correlación:**")

                    all_numeric_cols = df_combined.select_dtypes(include='number').columns.tolist()
                    # Excluir 'id' de las variables numéricas
                    all_numeric_cols = [c for c in all_numeric_cols if c.lower() != 'id']
                    default_vars = [v for v in ([y_var, x_var] + x_multi_vars) if v in all_numeric_cols]

                    selected_heatmap_vars = st.multiselect(
                        "Variables para el heatmap (mínimo 2)",
                        options=all_numeric_cols,
                        default=default_vars if len(default_vars) >= 2 else all_numeric_cols[:2],
                        help="Selecciona las variables numéricas que quieres comparar en la matriz de correlación"
                    )

                    if len(selected_heatmap_vars) < 2:
                        st.warning("Selecciona al menos 2 variables para generar la matriz de correlación.")
                        st.stop()

                    for ciudad in selected_cities:
                        st.markdown(f"### {ciudad}")
                        df_ciudad = df_combined[df_combined["ciudad"] == ciudad][selected_heatmap_vars].copy()
                        df_ciudad = df_ciudad.dropna(axis=1, how='all')
                        df_ciudad = df_ciudad.loc[:, df_ciudad.nunique() > 1]

                        if df_ciudad.empty or len(df_ciudad.columns) < 2:
                            st.warning(f"No hay suficientes datos numéricos para {ciudad}")
                            continue

                        corr_matrix = df_ciudad.corr()

                        fig = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale='RdBu_r',
                            zmid=0,
                            zmin=-1,
                            zmax=1,
                            text=corr_matrix.values,
                            texttemplate='%{text:.2f}',
                            textfont={"size": 10},
                            colorbar=dict(title=dict(text="Correlación", side="right"), thickness=15, len=0.7)
                        ))
                        fig.update_layout(
                            title=f"Correlación entre Variables - {ciudad}",
                            xaxis_title="Variables",
                            yaxis_title="Variables",
                            width=900, height=800,
                            xaxis={'tickangle': 45},
                            font=dict(size=11)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                var1 = corr_matrix.columns[i]
                                var2 = corr_matrix.columns[j]
                                corr_val = corr_matrix.iloc[i, j]
                                if not np.isnan(corr_val):
                                    corr_pairs.append((var1, var2, corr_val, abs(corr_val)))
                        corr_pairs.sort(key=lambda x: x[3], reverse=True)

                        if corr_pairs:
                            with st.expander(f"Top 10 Correlaciones más Fuertes en {ciudad}", expanded=False):
                                top_10 = corr_pairs[:10]
                                for idx, (v1, v2, corr, abs_corr) in enumerate(top_10, 1):
                                    direccion = "Positiva" if corr > 0 else "Negativa"
                                    color = "#ffc107" if idx == 1 else ("#28a745" if abs_corr >= 0.7 else "#6c757d")
                                    st.markdown(
                                        f"<div style='background:{color};color:#fff;padding:8px;border-radius:6px;margin:4px 0;'>"
                                        f"<strong>#{idx}</strong> → <strong>{v1}</strong> vs <strong>{v2}</strong>: "
                                        f"<strong>{corr:.3f}</strong> ({direccion})"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                        st.markdown("---")

# ====== HELPERS REGRESIÓN NO LINEAL ======

def modelo_polinomial(x, y, grado=2):
    X = np.vander(x, N=grado+1, increasing=True)  # [1, x, x^2, ...]
    y_idx = restantes_para_y.index(default_y) if default_y in restantes_para_y else 0
    y_var = st.sidebar.selectbox("Variable dependiente (y)", restantes_para_y, index=y_idx)

    remaining_vars = [col for col in numeric_cols if col not in [x_var, y_var]]
    max_predictors = min(15, len(remaining_vars))
    num_x = st.sidebar.slider(
        "¿Cuántas x adicionales (regresión múltiple)?", 
        0, max_predictors, min(2, max_predictors),
        help="Variables adicionales a la x principal"
    )

    st.sidebar.header("Vista por ciudad (múltiple)")
    ciudad_focus = st.sidebar.selectbox(
        "Ciudad para el análisis detallado",
        options=["Todas"] + selected_cities,
        index=0,
        help="Elige una ciudad para la gráfica de regresión múltiple y sus métricas."
    )

    st.sidebar.markdown(f"**Total de variables independientes: {num_x + 1}** (1 principal + {num_x} adicionales)")

    x_multi_vars = []
    for i in range(num_x):
        opciones = [col for col in remaining_vars if col not in x_multi_vars]
        x_i = st.sidebar.selectbox(f"Variable x{i+2} (Total: {i+2} variables)", opciones, key=f"x{i+2}_lin")
        x_multi_vars.append(x_i)

    # --- Limpieza de datos (mantener ciudad) ---
    cols_needed = [x_var, y_var, "ciudad"]
    df_clean = df_combined[cols_needed].dropna()
    if df_clean.empty:
        st.warning("No hay datos válidos después de limpiar NaN para las variables seleccionadas.")
        st.stop()

    # --- Unidades ---
    def get_unit(var_name):
        units = {
            'price': '€','accommodates':'huéspedes','bedrooms':'habitaciones','beds':'camas',
            'bathrooms':'baños','bathrooms_num':'baños','amenities_count':'amenidades',
            'minimum_nights':'noches','maximum_nights':'noches','availability_365':'días',
            'number_of_reviews':'reseñas','reviews_per_month':'reseñas/mes',
            'review_scores_rating':'puntos','review_scores_accuracy':'puntos',
            'review_scores_cleanliness':'puntos','review_scores_checkin':'puntos',
            'review_scores_communication':'puntos','review_scores_location':'puntos',
            'review_scores_value':'puntos','calculated_host_listings_count':'propiedades',
            'latitude':'°','longitude':'°','superhost_numeric':'(0=No, 1=Sí)','price_per_person':'€/persona'
        }
        return units.get(var_name, '')

    # ===================== GRÁFICAS =====================
    st.subheader("Comparación visual")
    col_simple, col_multi = st.columns(2)

    # ====== REGRESIÓN SIMPLE ======
    with col_simple:
        st.markdown("**Regresión simple**")

        def _plot_reg_simple_ciudad(sub_df: pd.DataFrame, ciudad: str, x_var: str, y_var: str):
            Xc = sub_df[[x_var]].values
            yc = sub_df[y_var].values
            model_c = LinearRegression()
            model_c.fit(Xc, yc)
            yhat_c = model_c.predict(Xc)
            order = np.argsort(Xc.ravel())
            x_sorted = Xc.ravel()[order]
            y_sorted = yhat_c[order]
            r2c = r2_score(yc, yhat_c)

            figc, axc = plt.subplots(figsize=(6, 4))
            axc.scatter(sub_df[x_var], sub_df[y_var], alpha=0.6)
            axc.plot(x_sorted, y_sorted, color='red', linewidth=2.5, linestyle='-', zorder=5, label="Línea de regresión")
            axc.text(0.04, 0.96, f'R² = {r2c:.3f}', transform=axc.transAxes, fontsize=11, fontweight='bold',
                     va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1))
            axc.set_title(f"{ciudad}: {y_var} vs {x_var}", fontsize=12)
            axc.set_xlabel(x_var); axc.set_ylabel(y_var)
            axc.legend(); axc.grid(True, alpha=0.3)
            st.pyplot(figc)

        modo_grafica = st.radio(
            "Modo de visualización",
            ["Cuadrícula por ciudad", "Pestañas por ciudad", "Todas juntas"],
            horizontal=True,
            key="modo_reg_simple"
        )

        if modo_grafica == "Todas juntas":
            X_simple = df_clean[[x_var]].values
            y_simple = df_clean[y_var].values
            model_simple = LinearRegression()
            try:
                model_simple.fit(X_simple, y_simple)
                y_pred_simple = model_simple.predict(X_simple)
                order = np.argsort(X_simple.ravel())
                x_sorted = X_simple.ravel()[order]
                y_sorted = y_pred_simple[order]
                r2_value = r2_score(y_simple, y_pred_simple)

                # métricas globales de simple (para tarjetas)
                rmse_simple = float(np.sqrt(mean_squared_error(y_simple, y_pred_simple)))
                mae_simple  = float(mean_absolute_error(y_simple, y_pred_simple))

                fig1, ax1 = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=df_clean[x_var], y=df_clean[y_var], hue=df_clean["ciudad"], ax=ax1, alpha=0.6)
                ax1.plot(x_sorted, y_sorted, color='red', label="Línea de regresión", linewidth=3, linestyle='-', zorder=5)
                ax1.text(0.05, 0.95, f'R² = {r2_value:.4f}', transform=ax1.transAxes, fontsize=14, fontweight='bold',
                         va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=2))
                ax1.set_title(f"{y_var} vs {x_var}")
                ax1.set_xlabel(x_var); ax1.set_ylabel(y_var)
                ax1.legend(); ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
            except Exception as e:
                st.error("No se pudo ajustar la regresión simple.")
                st.exception(e)

        elif modo_grafica == "Pestañas por ciudad":
            tabs = st.tabs(selected_cities)
            for tab, ciudad in zip(tabs, selected_cities):
                with tab:
                    sub = df_clean[df_clean["ciudad"] == ciudad][[x_var, y_var]].dropna()
                    if len(sub) < 2 or sub[x_var].nunique() < 2:
                        st.warning(f"{ciudad}: datos insuficientes para ajustar la regresión.")
                        continue
                    _plot_reg_simple_ciudad(sub, ciudad, x_var, y_var)

        else:  # "Cuadrícula por ciudad"
            n_cols = st.slider("Columnas de la cuadrícula", 2, 4, min(3, max(2, len(selected_cities))))
            cols = st.columns(n_cols)
            for i, ciudad in enumerate(selected_cities):
                sub = df_clean[df_clean["ciudad"] == ciudad][[x_var, y_var]].dropna()
                if len(sub) < 2 or sub[x_var].nunique() < 2:
                    with cols[i % n_cols]:
                        st.warning(f"{ciudad}: datos insuficientes para ajustar la regresión.")
                    continue
                with cols[i % n_cols]:
                    _plot_reg_simple_ciudad(sub, ciudad, x_var, y_var)

        # Cálculo silencioso de métricas simples globales (si no las generó el modo "Todas juntas")
        if 'rmse_simple' not in locals() or 'mae_simple' not in locals():
            try:
                X_tmp = df_clean[[x_var]].values
                y_tmp = df_clean[y_var].values
                mtmp = LinearRegression().fit(X_tmp, y_tmp)
                yhat_tmp = mtmp.predict(X_tmp)
                rmse_simple = float(np.sqrt(mean_squared_error(y_tmp, yhat_tmp)))
                mae_simple  = float(mean_absolute_error(y_tmp, yhat_tmp))
            except Exception:
                rmse_simple = None
                mae_simple = None
    # ====== REGRESIÓN MÚLTIPLE (sobre eje X; por ciudad) ======
    with col_multi:
        st.markdown("**Regresión múltiple — vista sobre eje X (por ciudad)**")
        selected_predictors = [x_var] + x_multi_vars

        # Filtrado por ciudad para el análisis detallado
        if ciudad_focus != "Todas":
            df_scope = df_combined[df_combined["ciudad"] == ciudad_focus]
            titulo_ciudad = f" — {ciudad_focus}"
        else:
            df_scope = df_combined
            titulo_ciudad = " — (todas las ciudades)"

        df_multi = df_scope[[y_var] + selected_predictors].apply(pd.to_numeric, errors="coerce").dropna()

        if df_multi.empty or len(selected_predictors) < 1:
            st.info("Selecciona al menos 1 variable independiente y verifica que haya datos válidos.")
            # Inicializa KPIs para evitar N/A ruidoso
            r2_multi = None
            r2_adj_multi = None
            rmse_multi = None
            mae_multi = None
            coef_multi = None
            intercept_multi = None
        else:
            # Ajuste del modelo múltiple en el scope elegido
            X = df_multi[selected_predictors].values
            y = df_multi[y_var].values
            model = LinearRegression().fit(X, y)
            y_hat = model.predict(X)

            plot_df = df_multi.copy()
            plot_df["y_real"] = y
            plot_df["y_pred_multi"] = y_hat

            # --- Controles de visual ---
            alpha_scatter = 0.6
            colorear_resid = st.checkbox(
                "Colorear Y real por residuo (Ŷ − Y)",
                value=True,
                key="color_resid_multi"
            )

            # === Gráfica principal: SOLO puntos de Y real y Ŷ (predicho) ===
            fig, ax = plt.subplots(figsize=(10, 6))

            if colorear_resid:
                resid = plot_df["y_pred_multi"] - plot_df["y_real"]
                sc = ax.scatter(
                    plot_df[x_var], plot_df["y_real"],
                    c=resid, s=26, alpha=alpha_scatter,
                    marker="o", label="Y real"
                )
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label("Residuo (Ŷ − Y)")
            else:
                ax.scatter(
                    plot_df[x_var], plot_df["y_real"],
                    s=26, alpha=alpha_scatter,
                    marker="o", label="Y real"
                )

            # Puntos de predicción múltiple (Ŷ)
            ax.scatter(
                plot_df[x_var], plot_df["y_pred_multi"],
                s=30, alpha=0.9,
                marker="^", label="Ŷ (múltiple)"
            )

            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.set_title(f"{y_var} sobre {x_var}{titulo_ciudad}: puntos de Y real y Ŷ")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            st.pyplot(fig)

            # Métricas del modelo en el scope (ciudad/todas) y export a KPIs
            r2 = r2_score(y, y_hat)
            rmse = float(np.sqrt(mean_squared_error(y, y_hat)))
            mae = float(mean_absolute_error(y, y_hat))
            n = len(y)
            k = len(selected_predictors)
            r2_adj = 1 - ((1 - r2) * (n - 1) / (n - k - 1)) if n > k + 1 else r2

            r2_multi = r2
            r2_adj_multi = r2_adj
            rmse_multi = rmse
            mae_multi = mae
            coef_multi = np.array([model.coef_[i] for i in range(len(selected_predictors))])
            intercept_multi = float(model.intercept_)

            st.markdown(
                f"- **R²**: {r2:.4f} · **R² ajustado**: {r2_adj:.4f} · "
                f"**RMSE**: {rmse:.3f} · **MAE**: {mae:.3f}"
            )

    # ===================== UTILIDADES DE UI =====================
    st.markdown("""
    <style>
    .card{background:#F7F7F7;border:1px solid rgba(0,0,0,.08);border-radius:16px;
        padding:14px 16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.05);}
    .card h4{margin:0 0 8px 0;font-weight:800;color:#FF385C;}
    .kpi{display:flex;gap:14px;flex-wrap:wrap}
    .kpi .metric{background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                padding:10px 12px;min-width:160px}
    .kpi .metric .label{font-size:12px;color:#666}
    .kpi .metric .value{font-size:20px;font-weight:700;color:#666}
    </style>
    """, unsafe_allow_html=True)

    def _metric_html(label, value):
        return f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>'

    def interpretar_correlacion(r):
        if r == "N/A" or not isinstance(r, (int, float)):
            return "N/A", "N/A"
        direccion = "Positiva" if r > 0 else ("Negativa" if r < 0 else "Nula")
        r_abs = abs(r)
        if r_abs >= 0.9:   fuerza = "Muy fuerte"
        elif r_abs >= 0.7: fuerza = "Fuerte"
        elif r_abs >= 0.5: fuerza = "Moderada"
        elif r_abs >= 0.3: fuerza = "Débil"
        else:              fuerza = "Muy débil"
        return direccion, fuerza

    # Ecuación global simple
    def calcular_resultados(df_in: pd.DataFrame, x_col: str, y_col: str):
        dfx = df_in[[x_col, y_col]].dropna()
        if dfx[x_col].nunique() <= 1:
            return {"n": len(dfx), "r2": 0.0, "corr": 0.0, "beta0": 0.0, "beta1": 0.0}
        try:
            X = sm.add_constant(dfx[x_col].values)
            y = dfx[y_col].values
            modelo = sm.OLS(y, X).fit()
            r2 = float(modelo.rsquared) if np.isfinite(modelo.rsquared) else 0.0
            corr = float(dfx[x_col].corr(dfx[y_col])) if dfx[[x_col, y_col]].notna().all().all() else 0.0
            beta0 = float(modelo.params[0]) if len(modelo.params) > 0 else 0.0
            beta1 = float(modelo.params[1]) if len(modelo.params) > 1 else 0.0
            return {"n": int(modelo.nobs), "r2": r2, "corr": corr, "beta0": beta0, "beta1": beta1}
        except Exception:
            return {"n": len(dfx), "r2": 0.0, "corr": 0.0, "beta0": 0.0, "beta1": 0.0}

    res_ecuacion = calcular_resultados(df_combined, x_var, y_var)
    beta0_txt = f"{res_ecuacion['beta0']:.4f}"
    beta1_txt = f"{res_ecuacion['beta1']:.4f}"
    r2_simple_txt = f"{res_ecuacion['r2']:.4f}"
    corr_simple_txt = f"{res_ecuacion['corr']:.4f}"
    signo = "+" if res_ecuacion['beta0'] >= 0 else ""
    ecuacion_txt = f"{y_var} = {beta1_txt} × {x_var} {signo} {beta0_txt}"

    st.markdown(
        '<div class="card"><h4>Ecuación de Regresión Simple</h4><div class="kpi">'
        + _metric_html("β₀ (Intercepto)", beta0_txt)
        + _metric_html("β₁ (Pendiente)", beta1_txt)
        + _metric_html("R²", r2_simple_txt)
        + f'</div><div style="margin-top:12px;padding:10px;background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;font-size:16px;font-weight:600;text-align:center;">{ecuacion_txt}</div></div>',
        unsafe_allow_html=True
    )

    # Métricas de modelos (usan las variables calculadas arriba)
    rmse_simple_txt = f"{rmse_simple:.4f}" if 'rmse_simple' in locals() and rmse_simple is not None else "N/A"
    rmse_multi_txt  = f"{rmse_multi:.4f}"  if 'rmse_multi' in locals()  and rmse_multi  is not None else "N/A"
    mae_simple_txt  = f"{mae_simple:.4f}"  if 'mae_simple' in locals()  and mae_simple  is not None else "N/A"
    mae_multi_txt   = f"{mae_multi:.4f}"   if 'mae_multi' in locals()   and mae_multi   is not None else "N/A"
    r2_multi_txt    = f"{r2_multi:.4f}"    if 'r2_multi' in locals()    and r2_multi    is not None else "N/A"
    r2_adj_multi_txt= f"{r2_adj_multi:.4f}"if 'r2_adj_multi' in locals()and r2_adj_multi is not None else "N/A"

    # Si hay múltiple, mostramos ecuación y R²
    if 'coef_multi' in locals() and coef_multi is not None and 'intercept_multi' in locals() and intercept_multi is not None:
        intercept_multi_txt = f"{intercept_multi:.4f}"
        selected_predictors = [x_var] + x_multi_vars
        terminos = [f"{coef:.4f} × {var}" for var, coef in zip(selected_predictors, coef_multi)]
        signo_int = "+" if intercept_multi >= 0 else ""
        ecuacion_multi_txt = f"{y_var} = {' + '.join(terminos)} {signo_int} {intercept_multi_txt}"

        coef_metrics = _metric_html("β₀ (Intercepto)", intercept_multi_txt)
        for i, (var, coef) in enumerate(zip(selected_predictors, coef_multi)):
            coef_metrics += _metric_html(f"β{i+1} ({var})", f"{coef:.4f}")
        coef_metrics += _metric_html("R² múltiple", r2_multi_txt)
        coef_metrics += _metric_html("R² Ajustado", r2_adj_multi_txt)

        st.markdown(
            '<div class="card"><h4>Ecuación de Regresión Múltiple</h4><div class="kpi">'
            + coef_metrics
            + f'</div><div style="margin-top:12px;padding:10px;background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;font-size:14px;font-weight:600;text-align:center;">{ecuacion_multi_txt}</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Modelo de regresión múltiple no disponible. Asegúrate de seleccionar variables adicionales.")

    # =================== RESULTADOS POR CIUDAD (CARDS) ===================
    st.subheader("Resultados por ciudad")
    cols_cards = st.columns(3)

    for i, ciudad in enumerate(selected_cities):
        res = calcular_resultados(df_combined[df_combined["ciudad"] == ciudad], x_var, y_var)
        r2_txt = f"{res['r2']:.4f}"
        corr_txt = f"{res['corr']:.4f}"
        direccion, fuerza = interpretar_correlacion(res['corr'])
        beta0_txt = f"{res['beta0']:.4f}"
        beta1_txt = f"{res['beta1']:.4f}"
        signo = "+" if res['beta0'] >= 0 else ""
        ecuacion = f"ŷ = {beta1_txt}x {signo} {beta0_txt}"

        with cols_cards[i % 3]:
            st.markdown(
                '<div class="card"><h4>'+ ciudad +'</h4><div class="kpi">'
                + _metric_html("R²", r2_txt)
                + _metric_html("Correlación", corr_txt)
                + _metric_html("Dirección", direccion)
                + _metric_html("Fuerza", fuerza)
                + _metric_html("β₀", beta0_txt)
                + _metric_html("β₁", beta1_txt)
                + f'</div><div style="margin-top:8px;padding:8px;background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:8px;font-size:13px;font-weight:600;text-align:center;">{ecuacion}</div></div>',
                unsafe_allow_html=True
            )

    # =================== CONTEXTO DE DATOS ===================
    st.subheader("Contexto de los Datos")
    y_mean, y_min, y_max = df_clean[y_var].mean(), df_clean[y_var].min(), df_clean[y_var].max()
    x_mean, x_min, x_max = df_clean[x_var].mean(), df_clean[x_var].min(), df_clean[x_var].max()
    total_props = len(df_clean)
    y_unit, x_unit = get_unit(y_var), get_unit(x_var)

    stats_by_city = df_clean.groupby("ciudad").agg({y_var: 'mean', x_var: 'mean'}).round(2)

    st.markdown("""
    <style>
    .context-card{background:#FFF;border:2px solid #FF385C;border-radius:16px;
        padding:16px 20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(255,56,92,.15);}
    .context-card h4{margin:0 0 12px 0;font-weight:800;color:#FF385C;font-size:18px;}
    .context-kpi{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;}
    .context-metric{background:#F7F7F7;border:1px solid rgba(0,0,0,.08);border-radius:10px;
                padding:12px 14px;min-width:140px;flex:1;}
    .context-metric .label{font-size:11px;color:#666;text-transform:uppercase;font-weight:600;}
    .context-metric .value{font-size:18px;font-weight:700;color:#333;margin-top:4px;}
    .city-breakdown{background:#F7F7F7;border-radius:10px;padding:12px;margin-top:8px;}
    .city-item{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #E0E0E0;}
    .city-item:last-child{border-bottom:none;}
    .city-name{font-weight:600;color:#333;}
    .city-values{color:#666;font-size:14px;}
    </style>
    """, unsafe_allow_html=True)

    y_label = f"{y_var}" + (f" ({y_unit})" if y_unit else "")
    x_label = f"{x_var}" + (f" ({x_unit})" if x_unit else "")
    variables_list = f"<strong>{y_label}</strong> (dependiente) vs <strong>{x_label}</strong> (independiente)"
    if x_multi_vars:
        extra = []
        for v in x_multi_vars:
            v_unit = get_unit(v)
            extra.append(f"<strong>{v}{' ('+v_unit+')' if v_unit else ''}</strong>")
        variables_list += f" + {len(x_multi_vars)} variables adicionales: " + ", ".join(extra)

    st.markdown(
        f'<div class="context-card">'
        f'<h4>Variables Seleccionadas</h4>'
        f'<p style="margin:0;font-size:14px;color:#333;">{variables_list}</p>'
        f'<div class="context-kpi" style="margin-top:12px;">'
        f'<div class="context-metric"><div class="label">Total Propiedades</div><div class="value">{total_props:,}</div></div>'
        f'<div class="context-metric"><div class="label">Ciudades</div><div class="value">{len(selected_cities)}</div></div>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    city_breakdown_html = '<div class="city-breakdown"><strong>Por Ciudad:</strong>'
    for ciudad in stats_by_city.index:
        y_val = stats_by_city.loc[ciudad, y_var]
        x_val = stats_by_city.loc[ciudad, x_var]
        y_val_str = f"{y_val:.2f} {y_unit}" if y_unit else f"{y_val:.2f}"
        x_val_str = f"{x_val:.2f} {x_unit}" if x_unit else f"{x_val:.2f}"
        city_breakdown_html += f'<div class="city-item"><span class="city-name">{ciudad}</span><span class="city-values">{y_var}: {y_val_str} | {x_var}: {x_val_str}</span></div>'
    city_breakdown_html += '</div>'

    y_mean_str = f"{y_mean:.2f} {y_unit}" if y_unit else f"{y_mean:.2f}"
    y_min_str  = f"{y_min:.2f} {y_unit}" if y_unit else f"{y_min:.2f}"
    y_max_str  = f"{y_max:.2f} {y_unit}" if y_unit else f"{y_max:.2f}"
    x_mean_str = f"{x_mean:.2f} {x_unit}" if x_unit else f"{x_mean:.2f}"
    x_min_str  = f"{x_min:.2f} {x_unit}" if x_unit else f"{x_min:.2f}"
    x_max_str  = f"{x_max:.2f} {x_unit}" if x_unit else f"{x_max:.2f}"

    st.markdown(
        f'<div class="context-card">'
        f'<h4>Estadísticas de {y_label}</h4>'
        f'<div class="context-kpi">'
        f'<div class="context-metric"><div class="label">Promedio</div><div class="value">{y_mean_str}</div></div>'
        f'<div class="context-metric"><div class="label">Mínimo</div><div class="value">{y_min_str}</div></div>'
        f'<div class="context-metric"><div class="label">Máximo</div><div class="value">{y_max_str}</div></div>'
        f'</div>'
        f'{city_breakdown_html}'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="context-card">'
        f'<h4>Estadísticas de {x_label}</h4>'
        f'<div class="context-kpi">'
        f'<div class="context-metric"><div class="label">Promedio</div><div class="value">{x_mean_str}</div></div>'
        f'<div class="context-metric"><div class="label">Mínimo</div><div class="value">{x_min_str}</div></div>'
        f'<div class="context-metric"><div class="label">Máximo</div><div class="value">{x_max_str}</div></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ================== HEATMAPS DE CORRELACIÓN POR CIUDAD ==================
    st.subheader("Matriz de Correlación por Ciudad")
    st.markdown("**Selecciona las variables a incluir en la matriz de correlación:**")

    all_numeric_cols = df_combined.select_dtypes(include='number').columns.tolist()
    # Excluir 'id' de las variables numéricas
    all_numeric_cols = [c for c in all_numeric_cols if c.lower() != 'id']
    default_vars = [v for v in ([y_var, x_var] + x_multi_vars) if v in all_numeric_cols]

    selected_heatmap_vars = st.multiselect(
        "Variables para el heatmap (mínimo 2)",
        options=all_numeric_cols,
        default=default_vars if len(default_vars) >= 2 else all_numeric_cols[:2],
        help="Selecciona las variables numéricas que quieres comparar en la matriz de correlación"
    )

    if len(selected_heatmap_vars) < 2:
        st.warning("Selecciona al menos 2 variables para generar la matriz de correlación.")
        st.stop()

    for ciudad in selected_cities:
        st.markdown(f"### {ciudad}")
        df_ciudad = df_combined[df_combined["ciudad"] == ciudad][selected_heatmap_vars].copy()
        df_ciudad = df_ciudad.dropna(axis=1, how='all')
        df_ciudad = df_ciudad.loc[:, df_ciudad.nunique() > 1]

        if df_ciudad.empty or len(df_ciudad.columns) < 2:
            st.warning(f"No hay suficientes datos numéricos para {ciudad}")
            continue

        corr_matrix = df_ciudad.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title=dict(text="Correlación", side="right"), thickness=15, len=0.7)
        ))
        fig.update_layout(
            title=f"Correlación entre Variables - {ciudad}",
            xaxis_title="Variables",
            yaxis_title="Variables",
            width=900, height=800,
            xaxis={'tickangle': 45},
            font=dict(size=11)
        )
        st.plotly_chart(fig, use_container_width=True)

        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val):
                    corr_pairs.append((var1, var2, corr_val, abs(corr_val)))
        corr_pairs.sort(key=lambda x: x[3], reverse=True)

        if corr_pairs:
            with st.expander(f"Top 10 Correlaciones más Fuertes en {ciudad}", expanded=False):
                top_10 = corr_pairs[:10]
                for idx, (v1, v2, corr, abs_corr) in enumerate(top_10, 1):
                    direccion = "Positiva" if corr > 0 else "Negativa"
                    color = "#ffc107" if idx == 1 else ("#28a745" if abs_corr >= 0.7 else "#6c757d")
                    st.markdown(
                        f"<div style='background:{color};color:#fff;padding:8px;border-radius:6px;margin:4px 0;'>"
                        f"<strong>#{idx}</strong> → <strong>{v1}</strong> vs <strong>{v2}</strong>: "
                        f"<strong>{corr:.3f}</strong> ({direccion})"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        st.markdown("---")

    # =========================================================
    # ====================== HALLAZGOS ========================
    # =========================================================
    with st.expander("Hallazgos Importantes", expanded=False):
        def _sr(x, d=2):
            try: return f"{float(x):.{d}f}"
            except: return "N/A"

        # 1) Fuente para simple por ciudad
        df_simple_view = None
        try:
            rows = []
            for ciudad in selected_cities:
                sub = df_combined[df_combined["ciudad"] == ciudad][[x_var, y_var]].dropna()
                if len(sub) >= 10 and sub[x_var].nunique() > 1:
                    X = sm.add_constant(sub[x_var].values); y = sub[y_var].values
                    m = sm.OLS(y, X).fit()
                    rows.append({"ciudad":ciudad,"n":int(m.nobs),"beta0":float(m.params[0]),
                                "beta1":float(m.params[1]),"R2":float(m.rsquared),
                                "pvalue_beta1":float(m.pvalues[1])})
                else:
                    rows.append({"ciudad":ciudad,"n":len(sub),"beta0":np.nan,"beta1":np.nan,"R2":np.nan,"pvalue_beta1":np.nan})
            df_simple_view = pd.DataFrame(rows)
        except Exception:
            df_simple_view = pd.DataFrame(columns=["ciudad","n","beta0","beta1","R2","pvalue_beta1"])

        phrases = []

        # Análisis del modelo lineal
        phrases.append("**Modelo de Regresión Lineal Simple**: Captura relaciones lineales directas entre variables. La relación asume que por cada unidad de incremento en la variable independiente, la variable dependiente cambia en una cantidad constante.")
        
        # 2) Frases basadas en regresión simple (y_var ~ x_var)
        if isinstance(df_simple_view, pd.DataFrame) and len(df_simple_view):
            dfv = df_simple_view.copy()
            for c in ["R2","beta1","pvalue_beta1","n"]:
                if c in dfv.columns: dfv[c] = pd.to_numeric(dfv[c], errors="coerce")
            dfv = dfv.dropna(subset=["R2"], how="all")

            # % de ciudades con pendiente significativa
            if {"pvalue_beta1","R2"}.issubset(dfv.columns):
                mask_valid = dfv["R2"].notna()
                total_valid = int(mask_valid.sum())
                sig = int((dfv["pvalue_beta1"] < 0.05).fillna(False).sum())
                if total_valid > 0:
                    pct = 100*sig/total_valid
                    phrases.append(f"**Significancia estadística**: En {_sr(pct,1)}% de las ciudades analizadas, la pendiente es estadísticamente significativa (p < 0.05) para la relación {x_var} vs {y_var}. Esto indica que la relación observada probablemente no se debe al azar.")

            # Top 1 mejor y peor R²
            if "R2" in dfv.columns and "ciudad" in dfv.columns:
                top = dfv.dropna(subset=["R2"]).sort_values("R2", ascending=False)
                if len(top):
                    mejor_ciudad = top.iloc[0]['ciudad']
                    mejor_r2 = top.iloc[0]['R2']
                    phrases.append(f"**Mejor ajuste**: {mejor_ciudad} (R² = {_sr(mejor_r2,3)}) - El modelo explica {mejor_r2*100:.1f}% de la variabilidad en {y_var}.")
                    
                    if mejor_r2 >= 0.7:
                        phrases.append(f"En {mejor_ciudad}, existe una relación lineal fuerte entre {x_var} y {y_var}, lo que sugiere que esta variable es un buen predictor.")
                    elif mejor_r2 >= 0.4:
                        phrases.append(f"En {mejor_ciudad}, existe una relación lineal moderada. Otros factores también contribuyen significativamente a la variación de precios.")
                    else:
                        phrases.append(f"En {mejor_ciudad}, la relación lineal es débil. Se recomienda considerar transformaciones no lineales o variables adicionales.")
                
                worst = top[top["R2"]>=0].sort_values("R2", ascending=True)
                if len(worst):
                    peor_ciudad = worst.iloc[0]['ciudad']
                    peor_r2 = worst.iloc[0]['R2']
                    phrases.append(f"**Menor ajuste**: {peor_ciudad} (R² = {_sr(peor_r2,3)}) - El modelo lineal simple explica solo {peor_r2*100:.1f}% de la variabilidad.")
                    
                    if peor_r2 < 0.3:
                        phrases.append(f"En {peor_ciudad}, {x_var} tiene capacidad predictiva limitada sobre {y_var} en un modelo lineal. El mercado de Airbnb puede tener dinámicas más complejas o responder a otros factores.")

            # Sentido de la relación (positiva/negativa) entre ciudades significativas
            if {"beta1","pvalue_beta1","ciudad"}.issubset(dfv.columns):
                pos_sig = dfv[(dfv["beta1"]>0) & (dfv["pvalue_beta1"]<0.05)]["ciudad"].tolist()
                neg_sig = dfv[(dfv["beta1"]<0) & (dfv["pvalue_beta1"]<0.05)]["ciudad"].tolist()
                if len(pos_sig):
                    ciudades_pos = ', '.join(pos_sig[:5])
                    if len(pos_sig) > 5:
                        ciudades_pos += "..."
                    phrases.append(f"**Relación positiva significativa**: En {ciudades_pos}, un aumento en {x_var} se asocia con un incremento en {y_var}.")
                if len(neg_sig):
                    ciudades_neg = ', '.join(neg_sig[:5])
                    if len(neg_sig) > 5:
                        ciudades_neg += "..."
                    phrases.append(f"**Relación negativa significativa**: En {ciudades_neg}, un aumento en {x_var} se asocia con una disminución en {y_var}.")

            # Efecto típico (mediana de β1) y potencia explicativa típica (mediana R²)
            if {"beta1","R2"}.issubset(dfv.columns):
                med_b1 = dfv["beta1"].dropna().median() if "beta1" in dfv else np.nan
                med_r2 = dfv["R2"].dropna().median()
                if pd.notna(med_b1):
                    direccion = "incremento" if med_b1>=0 else "disminución"
                    phrases.append(f"**Efecto típico**: La mediana de la pendiente es {_sr(abs(med_b1),2)}, lo que implica que típicamente un {direccion} de 1 unidad en {x_var} resulta en un cambio de {_sr(abs(med_b1),2)} unidades en {y_var}.")
                if pd.notna(med_r2):
                    phrases.append(f"**Capacidad explicativa típica**: La mediana de R² es {_sr(med_r2,3)} ({med_r2*100:.1f}%), indicando la proporción promedio de variabilidad explicada por el modelo lineal simple.")

            # Análisis de consistencia
            if "R2" in dfv.columns:
                r2_std = dfv["R2"].std()
                if r2_std < 0.1:
                    phrases.append(f"**Consistencia alta**: El modelo lineal tiene un desempeño similar en todas las ciudades (desviación estándar de R² = {_sr(r2_std,3)}), sugiriendo que la relación lineal es relativamente uniforme.")
                elif r2_std < 0.2:
                    phrases.append(f"**Consistencia moderada**: Existe variación moderada en el ajuste entre ciudades (desviación estándar de R² = {_sr(r2_std,3)}).")
                else:
                    phrases.append(f"**Consistencia baja**: El ajuste varía considerablemente entre ciudades (desviación estándar de R² = {_sr(r2_std,3)}), indicando que cada mercado tiene características particulares.")

        # 3) Frases del modelo múltiple global (si existe)
        try:
            # Intenta acceder a multi_metrics si existe en el contexto
            multi_metrics_data = st.session_state.get("multi_metrics", None)
            multi_coefs_data = st.session_state.get("multi_coefs", None)
            
            if multi_metrics_data and isinstance(multi_metrics_data, dict) and len(multi_metrics_data) > 0:
                phrases.append("")
                phrases.append("**Análisis de Regresión Múltiple**:")
                r2a = multi_metrics_data.get("R2_adj", None)
                rmse = multi_metrics_data.get("RMSE", None)
                if r2a is not None:
                    phrases.append(f"El modelo múltiple alcanza un R² ajustado de {_sr(r2a,3)} ({r2a*100:.1f}%). El R² ajustado penaliza la inclusión de variables adicionales, proporcionando una medida más conservadora del ajuste.")
                if rmse is not None:
                    phrases.append(f"El error cuadrático medio (RMSE) del modelo es {_sr(rmse,2)} unidades de {y_var}, representando el error típico en las predicciones.")

            if multi_coefs_data is not None and isinstance(multi_coefs_data, pd.DataFrame) and len(multi_coefs_data) > 0:
                dfc = multi_coefs_data.copy()
                if "variable" in dfc.columns and "beta_estandarizado" in dfc.columns:
                    dfc = dfc[dfc["variable"]!="const"].dropna(subset=["beta_estandarizado"])
                    if len(dfc):
                        dfc["absb"] = dfc["beta_estandarizado"].abs()
                        top3 = dfc.sort_values("absb", ascending=False).head(3)
                        driv = []
                        for _, row in top3.iterrows():
                            var_info = f"{row['variable']} (β estandarizado = {_sr(row['beta_estandarizado'],2)}"
                            if "pvalue" in dfc.columns:
                                var_info += f", p = {_sr(row['pvalue'],3)}"
                            var_info += ")"
                            driv.append(var_info)
                        phrases.append(f"**Principales predictores**: {', '.join(driv)}. Los coeficientes estandarizados permiten comparar la importancia relativa de cada variable independientemente de sus unidades de medida.")
        except Exception:
            # Si no hay datos de regresión múltiple, continuar sin ellos
            pass

        # Interpretación contextual
        phrases.append("")
        phrases.append("**Interpretación en el contexto de Airbnb**:")
        if x_var == "accommodates":
            phrases.append("La capacidad de huéspedes muestra relación lineal con el precio. Propiedades con mayor capacidad tienden a tener precios más altos de manera proporcional.")
        elif x_var == "amenities_count":
            phrases.append("El número de amenidades se relaciona linealmente con el precio. Cada amenidad adicional contribuye de manera constante al valor percibido de la propiedad.")
        elif x_var == "number_of_reviews":
            phrases.append("El número de reseñas tiene relación lineal con los precios, posiblemente reflejando popularidad o confiabilidad de la propiedad.")
        
        # Mostrar todos los hallazgos
        for phrase in phrases:
            if phrase:
                st.markdown(phrase)

# ====== HELPERS REGRESIÓN NO LINEAL ======

def modelo_polinomial(x, y, grado=2):
    X = np.vander(x, N=grado+1, increasing=True)  # [1, x, x^2, ...]
    reg = LinearRegression().fit(X, y)
    def f_pred(x_new):
        X_new = np.vander(x_new, N=grado+1, increasing=True)
        return reg.predict(X_new)
    return reg, f_pred

def modelo_logaritmico(x, y):
    # y = b0 + b1 ln(x)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = x > 0
    x_log = np.log(x[mask])
    y_mask = y[mask]
    reg = LinearRegression().fit(x_log.reshape(-1,1), y_mask)
    def f_pred(x_new):
        x_new = np.array(x_new, dtype=float)
        x_new_log = np.log(x_new)
        return reg.predict(x_new_log.reshape(-1,1))
    return reg, f_pred, mask

def modelo_exponencial(x, y):
    # y = a * e^(b x)  -> ln(y) = ln(a) + b x
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = y > 0
    x_mask = x[mask]
    y_log = np.log(y[mask])
    reg = LinearRegression().fit(x_mask.reshape(-1,1), y_log)
    def f_pred(x_new):
        x_new = np.array(x_new, dtype=float)
        y_log_pred = reg.predict(x_new.reshape(-1,1))
        return np.exp(y_log_pred)
    return reg, f_pred, mask

def modelo_potencia(x, y):
    # y = a * x^b  -> ln(y) = ln(a) + b ln(x)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = (x > 0) & (y > 0)
    x_log = np.log(x[mask])
    y_log = np.log(y[mask])
    reg = LinearRegression().fit(x_log.reshape(-1,1), y_log)
    def f_pred(x_new):
        x_new = np.array(x_new, dtype=float)
        x_log_new = np.log(x_new)
        y_log_pred = reg.predict(x_log_new.reshape(-1,1))
        return np.exp(y_log_pred)
    return reg, f_pred, mask

def modelo_raiz(x, y):
    # y = b0 + b1 sqrt(x)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = x >= 0
    x_sqrt = np.sqrt(x[mask])
    y_mask = y[mask]
    reg = LinearRegression().fit(x_sqrt.reshape(-1,1), y_mask)
    def f_pred(x_new):
        x_new = np.array(x_new, dtype=float)
        x_sqrt_new = np.sqrt(x_new)
        return reg.predict(x_sqrt_new.reshape(-1,1))
    return reg, f_pred, mask

def modelo_exponencial_decreciente(x, y):
    # y = a * exp(-b*x) + c  ->  Ajuste no lineal con curve_fit
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    try:
        # Valores iniciales razonables
        p0 = [y.max() - y.min(), 0.01, y.min()]
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
        def f_pred(x_new):
            return func(x_new, *popt)
        return None, f_pred, np.ones(len(x), dtype=bool)
    except Exception as e:
        # Si falla, intentar con ajuste lineal simple
        return None, None, np.ones(len(x), dtype=bool)

def modelo_inverso(x, y):
    # y = 1/(a*x)  ->  y = b0 * (1/x)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = x != 0
    x_inv = 1.0 / x[mask]
    y_mask = y[mask]
    reg = LinearRegression().fit(x_inv.reshape(-1,1), y_mask)
    def f_pred(x_new):
        x_new = np.array(x_new, dtype=float)
        x_inv_new = 1.0 / x_new
        return reg.predict(x_inv_new.reshape(-1,1))
    return reg, f_pred, mask

def modelo_senoidal(x, y):
    # y = a*sin(b*x) + c  ->  Ajuste no lineal
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    def func(x, a, b, c):
        return a * np.sin(b * x) + c
    
    try:
        # Valores iniciales razonables
        a0 = (y.max() - y.min()) / 2
        b0 = 2 * np.pi / (x.max() - x.min()) if x.max() != x.min() else 1
        c0 = y.mean()
        p0 = [a0, b0, c0]
        
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
        def f_pred(x_new):
            return func(x_new, *popt)
        return None, f_pred, np.ones(len(x), dtype=bool)
    except Exception as e:
        return None, None, np.ones(len(x), dtype=bool)

def modelo_tangencial(x, y):
    # y = a*tan(x) + b  ->  Ajuste no lineal
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    def func(x, a, b):
        return a * np.tan(x) + b
    
    try:
        # Valores iniciales
        p0 = [1.0, y.mean()]
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
        def f_pred(x_new):
            return func(x_new, *popt)
        return None, f_pred, np.ones(len(x), dtype=bool)
    except Exception as e:
        return None, None, np.ones(len(x), dtype=bool)

def modelo_valor_absoluto(x, y):
    # y = a*|x| + b*x + c  ->  Ajuste no lineal
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    def func(x, a, b, c):
        return a * np.abs(x) + b * x + c
    
    try:
        # Valores iniciales
        p0 = [1.0, 0.0, y.mean()]
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
        def f_pred(x_new):
            return func(x_new, *popt)
        return None, f_pred, np.ones(len(x), dtype=bool)
    except Exception as e:
        return None, None, np.ones(len(x), dtype=bool)

def modelo_cociente_polinomial(x, y):
    # y = (a*x^2 + b) / (c*x^2)  ->  Ajuste no lineal
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = x != 0
    x_mask = x[mask]
    y_mask = y[mask]
    
    def func(x, a, b, c):
        return (a * x**2 + b) / (c * x**2)
    
    try:
        # Valores iniciales
        p0 = [1.0, y_mask.mean(), 1.0]
        popt, _ = curve_fit(func, x_mask, y_mask, p0=p0, maxfev=10000)
        def f_pred(x_new):
            return func(x_new, *popt)
        return None, f_pred, mask
    except Exception as e:
        return None, None, mask

def modelo_cuadratico_inverso(x, y):
    # y = 1/(a*x^2)  ->  y = b0 * (1/x^2)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = x != 0
    x_inv2 = 1.0 / (x[mask]**2)
    y_mask = y[mask]
    reg = LinearRegression().fit(x_inv2.reshape(-1,1), y_mask)
    def f_pred(x_new):
        x_new = np.array(x_new, dtype=float)
        x_inv2_new = 1.0 / (x_new**2)
        return reg.predict(x_inv2_new.reshape(-1,1))
    return reg, f_pred, mask

def modelo_polinomial_inverso(x, y):
    # y = (a/b)*x^2 + c*x  ->  Simplificado: y = a*x^2 + b*x
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    X = np.column_stack([x**2, x])
    reg = LinearRegression().fit(X, y)
    def f_pred(x_new):
        x_new = np.array(x_new, dtype=float)
        X_new = np.column_stack([x_new**2, x_new])
        return reg.predict(X_new)
    return reg, f_pred, np.ones(len(x), dtype=bool)

# ====== VISTA: REGRESIÓN NO LINEAL ======
if View == "Regresión No Lineal":

    col_title, col_help = st.columns([0.95, 0.05])
    with col_title:
        st.title("Airbnb – Regresión No Lineal por ciudad")
    with col_help:
        with st.popover("➕"):
            st.markdown("**¿Qué es la Regresión No Lineal?**")
            st.write("Método que modela relaciones complejas (curvas, exponenciales) entre variables. En Airbnb, captura efectos no proporcionales: por ejemplo, cómo el precio puede aumentar rápidamente al principio con más huéspedes, pero desacelerarse después. Ideal para patrones que no son líneas rectas.")

    # --- Validación de columna ciudad ---
    if "ciudad" not in df.columns or df["ciudad"].dropna().empty:
        st.warning("No hay columna 'ciudad' válida en el DataFrame.")
        st.stop()

    # --- Selección de ciudades ---
    ciudades_disp = sorted(df["ciudad"].dropna().unique().tolist())
    st.sidebar.header("Ciudades")
    selected_cities = st.sidebar.multiselect(
        "Selecciona de 1 a 5 ciudades",
        options=ciudades_disp,
        default=ciudades_disp[:min(3, len(ciudades_disp))],
        max_selections=5
    )

    if not selected_cities:
        st.info("Selecciona al menos una ciudad para continuar.")
        st.stop()

    df_combined = df[df["ciudad"].isin(selected_cities)].copy()

    # --- Columnas numéricas candidatas ---
    numeric_cols = df_combined.select_dtypes(include="number").columns.tolist()
    # Excluir 'id' de las variables numéricas
    numeric_cols = [c for c in numeric_cols if c.lower() != 'id']
    if len(numeric_cols) < 2:
        st.warning("Se requieren al menos dos columnas numéricas para hacer regresión.")
        st.stop()

    # Defaults razonables
    default_y = "precio" if "precio" in numeric_cols else numeric_cols[0]
    default_x = "num_huespedes" if "num_huespedes" in numeric_cols else numeric_cols[min(1, len(numeric_cols) - 1)]

    st.sidebar.header("Variables de regresión no lineal")
    x_var = st.sidebar.selectbox("Variable independiente (X)", numeric_cols, index=numeric_cols.index(default_x))
    restantes_para_y = [c for c in numeric_cols if c != x_var]
    y_idx = restantes_para_y.index(default_y) if default_y in restantes_para_y else 0
    y_var = st.sidebar.selectbox("Variable dependiente (Y)", restantes_para_y, index=y_idx)

    # --- Modelos disponibles (al menos 5) ---
    modelos_disponibles = {
        "Polinomial grado 2": "poly2",
        "Polinomial grado 3": "poly3",
        "Logarítmico (Y ~ ln X)": "log",
        "Exponencial (Y ~ exp(X))": "exp",
        "Potencia (Y ~ X^b)": "pow",
        "Raíz cuadrada (Y ~ sqrt(X))": "sqrt",
        "Inversa (Y ~ 1/X)": "inversa",
        "Cociente de polinomios": "cociente_poli",
        "Senoidal (Y ~ sin(X))": "senoidal",
    }

    # 👉 Solo UNA ecuación seleccionada
    nombre_modelo = st.sidebar.selectbox(
        "Ecuación no lineal",
        options=list(modelos_disponibles.keys()),
        index=0
    )
    tipo_modelo = modelos_disponibles[nombre_modelo]
    
    # Expander con definiciones de ecuaciones no lineales
    with st.sidebar.expander("Definiciones de ecuaciones no lineales"):
        st.markdown("""
        **Polinomial Grado 2 (Cuadrática)**  
        y = a·x² + b·x + c  
        Forma una U o U invertida. Captura un punto donde se alcanza un máximo o mínimo.  
        **Uso:** Buscar valores óptimos o identificar puntos de equilibrio.  
        Ejemplo: Una propiedad con 2 baños puede ser ideal, pero 5 baños no aumenta el precio proporcionalmente.
        
        **Polinomial Grado 3 (Cúbica)**  
        y = a·x³ + b·x² + c·x + d  
        Forma una S alargada. Captura tendencias que cambian de dirección dos veces.  
        **Uso:** Modelar procesos con múltiples etapas o cambios de comportamiento.  
        Ejemplo: Noches mínimas - pocas (flexibles) o muchas (rigidez) atraen menos reservas; punto medio es óptimo.
        
        **Logarítmica**  
        y = a + b·ln(x)  
        Forma una curva que sube empinada y luego se aplana. Crece cada vez más lento.  
        **Uso:** Efectos que disminuyen con el tiempo o rendimientos decrecientes.  
        Ejemplo: Las primeras 10 reviews aumentan mucho el precio, pero pasar de 100 a 110 reviews apenas suma valor.
        
        **Exponencial**  
        y = a·e^(b·x)  
        Forma una J. Crece lento al inicio y luego se dispara hacia arriba.  
        **Uso:** Crecimiento acelerado o procesos que se amplifican.  
        Ejemplo: Propiedades de lujo donde cada huésped adicional multiplica exponencialmente el precio por exclusividad.
        
        **Potencial**  
        y = a·x^b  
        Forma una curva suave. Crece de manera proporcional en toda la gráfica.  
        **Uso:** Relaciones de escala o cambios proporcionales constantes.  
        Ejemplo: Capacidad vs precio - duplicar huéspedes generalmente duplica el precio de forma consistente.
        
        **Raíz cuadrada**  
        y = a + b·√x  
        Forma una curva que sube y se va aplanando. Similar a logarítmica pero más suave.  
        **Uso:** Crecimiento que se desacelera gradualmente.  
        Ejemplo: Agregar amenidades sube el precio inicialmente, pero después de 20-30 amenidades el efecto es menor.
        
        **Inversa**  
        y = a + b/x  
        Forma una L invertida. Baja rápido al inicio y luego se estabiliza.  
        **Uso:** Efectos que disminuyen rápidamente hasta estabilizarse.  
        Ejemplo: Estancias largas reducen el precio por noche - 7 noches es mucho más barato que 2, pero 30 vs 60 noches ya no cambia tanto.
        
        **Cociente de polinomios**  
        y = (a·x + b) / (c·x + d)  
        Forma una curva que se acerca a una línea horizontal. Tiene un límite que no cruza.  
        **Uso:** Procesos con capacidad máxima o saturación.  
        Ejemplo: Calificación vs precio - mejora hasta cierto techo porque el mercado tiene un precio máximo aceptable.
        
        **Senoidal**  
        y = a·sin(b·x + c) + d  
        Forma ondas que suben y bajan repetidamente. Captura patrones cíclicos.  
        **Uso:** Datos estacionales o que se repiten en intervalos regulares.  
        Ejemplo: Precios en temporada alta (verano/diciembre) vs baja (otoño/primavera) que se repiten cada año.
        """)
    
    # Expander con métricas de evaluación
    with st.sidebar.expander("Métricas"):
        st.markdown("""
        **MAE (Error Absoluto Medio)**
        
        **En palabras:** Suma todos los errores (en valor absoluto) y divide entre cuántos datos hay.
        
        **Ejemplo:** MAE = 10€ significa que el modelo se equivoca en promedio por 10€.
        
        ---
        
        **RMSE (Raíz del Error Cuadrático Medio)**
        
        **En palabras:** Eleva cada error al cuadrado, saca el promedio y luego la raíz cuadrada.
        
        **Ejemplo:** RMSE = 15€ indica que hay algunos errores grandes que afectan el promedio.
        
        ---
        
        **MAPE (Error Porcentual Absoluto Medio)**
        
        **En palabras:** Divide cada error entre el valor real, convierte a porcentaje y saca el promedio.
        
        **Ejemplo:** MAPE = 10% significa que el modelo se equivoca un 10% del valor real.
        """)
    

    # Función helper para plotear una ciudad individual
    def _plot_nonlinear_ciudad(df_c, ciudad, x_var, y_var, tipo_modelo, nombre_modelo, mostrar_metricas=True):
        """Plotea regresión no lineal para una ciudad específica"""
        if df_c.empty:
            st.info(f"{ciudad}: Sin datos.")
            return None

        x = df_c[x_var].values.astype(float)
        y = df_c[y_var].values.astype(float)
        n = len(y)

        x_grid = np.linspace(x.min(), x.max(), 200)

        try:
            # Ajuste del modelo elegido
            if tipo_modelo == "poly2":
                reg, f_pred = modelo_polinomial(x, y, grado=2)
                y_pred = f_pred(x)
                x_plot = x_grid
                y_grid = f_pred(x_plot)

            elif tipo_modelo == "poly3":
                reg, f_pred = modelo_polinomial(x, y, grado=3)
                y_pred = f_pred(x)
                x_plot = x_grid
                y_grid = f_pred(x_plot)

            elif tipo_modelo == "log":
                reg, f_pred, mask = modelo_logaritmico(x, y)
                y_pred = np.full_like(y, np.nan, dtype=float)
                y_pred[mask] = f_pred(x[mask])
                x_plot = x_grid[x_grid > 0]
                y_grid = f_pred(x_plot)

            elif tipo_modelo == "exp":
                reg, f_pred, mask = modelo_exponencial(x, y)
                y_pred = np.full_like(y, np.nan, dtype=float)
                y_pred[mask] = f_pred(x[mask])
                x_plot = x_grid
                y_grid = f_pred(x_plot)

            elif tipo_modelo == "pow":
                reg, f_pred, mask = modelo_potencia(x, y)
                y_pred = np.full_like(y, np.nan, dtype=float)
                y_pred[mask] = f_pred(x[mask])
                x_plot = x_grid[x_grid > 0]
                y_grid = f_pred(x_plot)

            elif tipo_modelo == "sqrt":
                reg, f_pred, mask = modelo_raiz(x, y)
                y_pred = np.full_like(y, np.nan, dtype=float)
                y_pred[mask] = f_pred(x[mask])
                x_plot = x_grid[x_grid >= 0]
                y_grid = f_pred(x_plot)

            elif tipo_modelo == "inversa":
                reg, f_pred, mask = modelo_inverso(x, y)
                y_pred = np.full_like(y, np.nan, dtype=float)
                y_pred[mask] = f_pred(x[mask])
                x_plot = x_grid[x_grid != 0]
                y_grid = f_pred(x_plot)

            elif tipo_modelo == "cociente_poli":
                reg, f_pred, mask = modelo_cociente_polinomial(x, y)
                if f_pred is None:
                    st.info(f"{ciudad}: El modelo de cociente polinomial no convergió con estos datos.")
                    return None
                # Crear objeto dummy para reg si es None
                if reg is None:
                    class DummyReg:
                        coef_ = np.array([])
                        intercept_ = 0
                    reg = DummyReg()
                y_pred = np.full_like(y, np.nan, dtype=float)
                y_pred[mask] = f_pred(x[mask])
                x_plot = x_grid[x_grid != 0]
                y_grid = f_pred(x_plot)

            elif tipo_modelo == "senoidal":
                reg, f_pred, mask = modelo_senoidal(x, y)
                if f_pred is None:
                    st.info(f"{ciudad}: El modelo senoidal no convergió con estos datos.")
                    return None
                # Crear objeto dummy para reg si es None
                if reg is None:
                    class DummyReg:
                        coef_ = np.array([])
                        intercept_ = 0
                    reg = DummyReg()
                y_pred = f_pred(x)
                x_plot = x_grid
                y_grid = f_pred(x_plot)

            # Métricas
            mask_valid = ~np.isnan(y_pred)
            if mask_valid.sum() < 3:
                st.info(f"{ciudad}: Datos insuficientes para calcular métricas.")
                return None

            r2 = r2_score(y[mask_valid], y_pred[mask_valid])
            rmse = float(np.sqrt(mean_squared_error(y[mask_valid], y_pred[mask_valid])))
            mae = float(mean_absolute_error(y[mask_valid], y_pred[mask_valid]))
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y[mask_valid] - y_pred[mask_valid]) / y[mask_valid])) * 100
            mape = float(mape) if not np.isnan(mape) and not np.isinf(mape) else 0.0
            
            # Correlación de Pearson (r)
            r_pearson = np.corrcoef(y[mask_valid], y_pred[mask_valid])[0, 1]
            r_pearson = float(r_pearson) if not np.isnan(r_pearson) else 0.0
            
            # Determinar número de parámetros según el tipo de modelo
            if tipo_modelo in ["cociente_poli"]:
                p = 3  # 3 parámetros (a, b, c)
            elif tipo_modelo in ["senoidal"]:
                p = 3  # 3 parámetros (a, b, c)
            elif hasattr(reg, "coef_"):
                p = reg.coef_.size + 1
            else:
                p = 2  # default
            
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan

            # Extraer coeficientes para la ecuación
            if tipo_modelo in ["poly2", "poly3"]:
                coefs = reg.coef_ if hasattr(reg, "coef_") else []
                intercept = reg.intercept_ if hasattr(reg, "intercept_") else 0
                if tipo_modelo == "poly2":
                    ecuacion = f"{y_var} = {coefs[2]:.4f}·{x_var}² + {coefs[1]:.4f}·{x_var} + {intercept:.4f}"
                else:  # poly3
                    ecuacion = f"{y_var} = {coefs[3]:.4f}·{x_var}³ + {coefs[2]:.4f}·{x_var}² + {coefs[1]:.4f}·{x_var} + {intercept:.4f}"
            elif tipo_modelo == "log":
                b0 = reg.intercept_ if hasattr(reg, "intercept_") else 0
                b1 = reg.coef_[0] if hasattr(reg, "coef_") else 0
                ecuacion = f"{y_var} = {b1:.4f}·ln({x_var}) + {b0:.4f}"
            elif tipo_modelo == "exp":
                b = reg.coef_[0] if hasattr(reg, "coef_") else 0
                ln_a = reg.intercept_ if hasattr(reg, "intercept_") else 0
                a = np.exp(ln_a)
                ecuacion = f"{y_var} = {a:.4f}·e^({b:.4f}·{x_var})"
            elif tipo_modelo == "pow":
                b = reg.coef_[0] if hasattr(reg, "coef_") else 0
                ln_a = reg.intercept_ if hasattr(reg, "intercept_") else 0
                a = np.exp(ln_a)
                ecuacion = f"{y_var} = {a:.4f}·{x_var}^{b:.4f}"
            elif tipo_modelo == "sqrt":
                b0 = reg.intercept_ if hasattr(reg, "intercept_") else 0
                b1 = reg.coef_[0] if hasattr(reg, "coef_") else 0
                ecuacion = f"{y_var} = {b1:.4f}·√{x_var} + {b0:.4f}"
            elif tipo_modelo == "inversa":
                # El modelo es y = a*(1/x), donde a es el coeficiente
                a = reg.coef_[0] if hasattr(reg, "coef_") and len(reg.coef_) > 0 else 0
                ecuacion = f"{y_var} = {a:.4f}·(1/{x_var})"
            elif tipo_modelo == "cociente_poli":
                ecuacion = f"{y_var} = (a·{x_var}² + b)/(c·{x_var}²) [ajuste no lineal]"
            elif tipo_modelo == "senoidal":
                ecuacion = f"{y_var} = a·sin(b·{x_var}) + c [ajuste no lineal]"
            else:
                ecuacion = "Ecuación no disponible"

            # Figura
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="markers",
                name="Datos",
                opacity=0.6
            ))
            fig.add_trace(go.Scatter(
                x=x_plot, y=y_grid,
                mode="lines",
                name=nombre_modelo
            ))
            
            # Agregar anotación con R² en la esquina superior derecha
            fig.add_annotation(
                text=f"R² = {r2:.4f}",
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                xanchor="right", yanchor="top",
                showarrow=False,
                font=dict(size=16, color="white", family="Arial Black"),
                bgcolor="#FF385C",
                bordercolor="#FF385C",
                borderwidth=2,
                borderpad=8,
                opacity=0.95
            )
            
            fig.update_layout(
                title=f"{ciudad}: {y_var} vs {x_var}",
                xaxis_title=x_var,
                yaxis_title=y_var,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Mostrar métricas en formato de tarjeta
            if mostrar_metricas:
                st.markdown(f"""
                <div style="background:#F7F7F7;border:1px solid rgba(0,0,0,.08);border-radius:16px;
                            padding:14px 16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.05);">
                    <h4 style="margin:0 0 8px 0;font-weight:800;color:#FF385C;text-align:center;">Métricas del Modelo - {ciudad}</h4>
                <div style="display:flex;gap:14px;flex-wrap:wrap;justify-content:center;">
                    <div style="background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                                padding:10px 12px;min-width:160px;">
                        <div style="font-size:12px;color:#666;text-align:center;">r (Correlación)</div>
                        <div style="font-size:20px;font-weight:700;color:#666;text-align:center;">{r_pearson:.3f}</div>
                    </div>
                    <div style="background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                                padding:10px 12px;min-width:160px;">
                        <div style="font-size:12px;color:#666;text-align:center;">R²</div>
                        <div style="font-size:20px;font-weight:700;color:#666;text-align:center;">{r2:.3f}</div>
                    </div>
                    <div style="background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                                padding:10px 12px;min-width:160px;">
                        <div style="font-size:12px;color:#666;text-align:center;">R² Ajustado</div>
                        <div style="font-size:20px;font-weight:700;color:#666;text-align:center;">{r2_adj:.3f}</div>
                    </div>
                    <div style="background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                                padding:10px 12px;min-width:160px;">
                        <div style="font-size:12px;color:#666;text-align:center;">RMSE</div>
                        <div style="font-size:20px;font-weight:700;color:#666;text-align:center;">{rmse:.2f}</div>
                    </div>
                    <div style="background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                                padding:10px 12px;min-width:160px;">
                        <div style="font-size:12px;color:#666;text-align:center;">MAE</div>
                        <div style="font-size:20px;font-weight:700;color:#666;text-align:center;">{mae:.2f}</div>
                    </div>
                    <div style="background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                                padding:10px 12px;min-width:160px;">
                        <div style="font-size:12px;color:#666;text-align:center;">MAPE</div>
                        <div style="font-size:20px;font-weight:700;color:#666;text-align:center;">{mape:.2f}%</div>
                    </div>
                </div>
                <div style="margin-top:12px;padding:10px;background:#fff;border:1px solid rgba(0,0,0,.06);
                            border-radius:12px;font-size:16px;font-weight:600;text-align:center;color:#484848;">
                        {ecuacion}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            return {
                "ciudad": ciudad,
                "Modelo": nombre_modelo,
                "r (Correlación)": r_pearson,
                "R²": r2,
                "R² ajustado": r2_adj,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE (%)": mape,
                "reg_model": reg
            }

        except Exception as e:
            st.warning(f"Error en {ciudad}: {e}")
            return None
    
    # Función auxiliar para ajustar todos los modelos
    def ajustar_todos_modelos_tab3(df_ciudad_input, x_var_input, y_var_input):
        """Ajusta los 9 modelos a los datos de una ciudad y retorna métricas"""
        df_c = df_ciudad_input[[x_var_input, y_var_input]].dropna()
        if df_c.empty or len(df_c) < 10:
            return None
        
        x = df_c[x_var_input].values.astype(float)
        y = df_c[y_var_input].values.astype(float)
        n = len(y)
        
        resultados = {}
        
        for nombre, tipo in modelos_disponibles.items():
            try:
                # Ajustar modelo según tipo
                if tipo == "poly2":
                    reg, f_pred = modelo_polinomial(x, y, grado=2)
                    y_pred = f_pred(x)
                elif tipo == "poly3":
                    reg, f_pred = modelo_polinomial(x, y, grado=3)
                    y_pred = f_pred(x)
                elif tipo == "log":
                    reg, f_pred, mask = modelo_logaritmico(x, y)
                    y_pred = np.full_like(y, np.nan, dtype=float)
                    if mask.sum() > 0:
                        y_pred[mask] = f_pred(x[mask])
                elif tipo == "exp":
                    reg, f_pred, mask = modelo_exponencial(x, y)
                    y_pred = np.full_like(y, np.nan, dtype=float)
                    if mask.sum() > 0:
                        y_pred[mask] = f_pred(x[mask])
                elif tipo == "pow":
                    reg, f_pred, mask = modelo_potencia(x, y)
                    y_pred = np.full_like(y, np.nan, dtype=float)
                    if mask.sum() > 0:
                        y_pred[mask] = f_pred(x[mask])
                elif tipo == "sqrt":
                    reg, f_pred, mask = modelo_raiz(x, y)
                    y_pred = np.full_like(y, np.nan, dtype=float)
                    if mask.sum() > 0:
                        y_pred[mask] = f_pred(x[mask])
                elif tipo == "inversa":
                    reg, f_pred, mask = modelo_inverso(x, y)
                    y_pred = np.full_like(y, np.nan, dtype=float)
                    if mask.sum() > 0:
                        y_pred[mask] = f_pred(x[mask])
                elif tipo == "cociente_poli":
                    reg, f_pred, mask = modelo_cociente_polinomial(x, y)
                    if f_pred is None:
                        continue
                    y_pred = np.full_like(y, np.nan, dtype=float)
                    if mask.sum() > 0:
                        y_pred[mask] = f_pred(x[mask])
                elif tipo == "senoidal":
                    reg, f_pred, mask = modelo_senoidal(x, y)
                    if f_pred is None:
                        continue
                    y_pred = np.full_like(y, np.nan, dtype=float)
                    y_pred = f_pred(x)
                
                # Calcular métricas
                mask_valid = ~np.isnan(y_pred)
                if mask_valid.sum() < 2:
                    continue
                
                r2 = r2_score(y[mask_valid], y_pred[mask_valid])
                rmse = float(np.sqrt(mean_squared_error(y[mask_valid], y_pred[mask_valid])))
                mae = float(mean_absolute_error(y[mask_valid], y_pred[mask_valid]))
                mape = np.mean(np.abs((y[mask_valid] - y_pred[mask_valid]) / y[mask_valid])) * 100
                mape = float(mape) if not np.isnan(mape) and not np.isinf(mape) else 0.0
                
                # Correlación de Pearson (r)
                r_pearson = np.corrcoef(y[mask_valid], y_pred[mask_valid])[0, 1]
                r_pearson = float(r_pearson) if not np.isnan(r_pearson) else 0.0
                
                # Determinar número de parámetros según el tipo de modelo
                if tipo in ["cociente_poli", "senoidal"]:
                    p = 3
                elif hasattr(reg, "coef_") and reg is not None:
                    p = reg.coef_.size + 1
                else:
                    p = 2
                
                r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
                
                # Validar que las métricas sean razonables
                if np.isnan(r2) or np.isinf(r2) or r2 < -10:
                    continue
                if np.isnan(r2_adj) or np.isinf(r2_adj) or r2_adj < -10:
                    r2_adj = r2
                
                resultados[nombre] = {
                    "r (Correlación)": r_pearson,
                    "R²": r2,
                    "R² ajustado": r2_adj,
                    "RMSE": rmse,
                    "MAE": mae,
                    "MAPE (%)": mape,
                    "y_pred": y_pred,
                    "f_pred": f_pred,
                    "reg": reg
                }
            except Exception as e:
                continue
        
        return resultados if resultados else None

    tab1, tab2 = st.tabs(["Comparación visual", "Comparación Multi-Modelo"])

    # ================================================================
    # TAB 1: COMPARACIÓN VISUAL CON OPCIONES DE VISUALIZACIÓN
    # ================================================================
    with tab1:
        # Forma funcional del modelo
        forma_funcional = ""
        if tipo_modelo == "poly2":
            forma_funcional = "y = β₀ + β₁·x + β₂·x²"
        elif tipo_modelo == "poly3":
            forma_funcional = "y = β₀ + β₁·x + β₂·x² + β₃·x³"
        elif tipo_modelo == "log":
            forma_funcional = "y = β₀ + β₁·ln(x)"
        elif tipo_modelo == "exp":
            forma_funcional = "y = a·e^(b·x)"
        elif tipo_modelo == "pow":
            forma_funcional = "y = a·x^b"
        elif tipo_modelo == "sqrt":
            forma_funcional = "y = β₀ + β₁·√x"
        elif tipo_modelo == "inversa":
            forma_funcional = "y = a/x"
        elif tipo_modelo == "cociente_poli":
            forma_funcional = "y = (a·x² + b)/(c·x²)"
        elif tipo_modelo == "senoidal":
            forma_funcional = "y = a·sin(b·x) + c"
        
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, rgba(255,56,92,0.1) 0%, rgba(255,255,255,0) 100%); 
                    padding: 20px; border-left: 4px solid #FF385C; border-radius: 8px; margin-bottom: 20px;'>
            <h3 style='color: #484848; margin: 0 0 10px 0; font-weight: 700; font-size: 22px;'>
                Modelo Seleccionado: <span style='color: #FF385C;'>{nombre_modelo}</span>
            </h3>
            <div style='background: white; padding: 12px 16px; border-radius: 6px; border: 1px solid #E8E8E8;'>
                <p style='color: #767676; margin: 0 0 4px 0; font-size: 13px; font-weight: 600;'>Forma funcional</p>
                <p style='color: #484848; margin: 0; font-size: 16px; font-family: "Courier New", monospace; font-weight: 500;'>{forma_funcional}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        modo_grafica_nl = st.radio(
            "Modo de visualización",
            ["Cuadrícula por ciudad", "Pestañas por ciudad", "Todas juntas"],
            horizontal=True,
            key="modo_nonlinear"
        )

        resultados_globales = []

        if modo_grafica_nl == "Todas juntas":
            # Modo: todas las ciudades en una sola gráfica
            st.markdown("""
            <div style='background: #F7F7F7; padding: 12px 16px; border-radius: 8px; margin-bottom: 15px;'>
                <p style='color: #484848; margin: 0; font-weight: 600; font-size: 16px;'>Vista Consolidada — Todas las Ciudades</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig_all = go.Figure()
            
            for ciudad in selected_cities:
                df_c = df_combined[df_combined["ciudad"] == ciudad][[x_var, y_var]].dropna()
                if df_c.empty:
                    continue
                
                x = df_c[x_var].values.astype(float)
                y = df_c[y_var].values.astype(float)
                n = len(y)
                
                # Agregar scatter de datos
                fig_all.add_trace(go.Scatter(
                    x=x, y=y,
                    mode="markers",
                    name=ciudad,
                    opacity=0.6
                ))
                
                # Calcular curva del modelo
                try:
                    x_grid = np.linspace(x.min(), x.max(), 100)
                    
                    if tipo_modelo == "poly2":
                        reg, f_pred = modelo_polinomial(x, y, grado=2)
                        y_pred = f_pred(x)
                        x_plot = x_grid
                        y_grid = f_pred(x_plot)
                    elif tipo_modelo == "poly3":
                        reg, f_pred = modelo_polinomial(x, y, grado=3)
                        y_pred = f_pred(x)
                        x_plot = x_grid
                        y_grid = f_pred(x_plot)
                    elif tipo_modelo == "log":
                        reg, f_pred, mask = modelo_logaritmico(x, y)
                        y_pred = np.full_like(y, np.nan, dtype=float)
                        y_pred[mask] = f_pred(x[mask])
                        x_plot = x_grid[x_grid > 0]
                        y_grid = f_pred(x_plot)
                    elif tipo_modelo == "exp":
                        reg, f_pred, mask = modelo_exponencial(x, y)
                        y_pred = np.full_like(y, np.nan, dtype=float)
                        y_pred[mask] = f_pred(x[mask])
                        x_plot = x_grid
                        y_grid = f_pred(x_plot)
                    elif tipo_modelo == "pow":
                        reg, f_pred, mask = modelo_potencia(x, y)
                        y_pred = np.full_like(y, np.nan, dtype=float)
                        y_pred[mask] = f_pred(x[mask])
                        x_plot = x_grid[x_grid > 0]
                        y_grid = f_pred(x_plot)
                    elif tipo_modelo == "sqrt":
                        reg, f_pred, mask = modelo_raiz(x, y)
                        y_pred = np.full_like(y, np.nan, dtype=float)
                        y_pred[mask] = f_pred(x[mask])
                        x_plot = x_grid[x_grid >= 0]
                        y_grid = f_pred(x_plot)
                    
                    # Calcular métricas
                    mask_valid = ~np.isnan(y_pred)
                    if mask_valid.sum() >= 3:
                        r2 = r2_score(y[mask_valid], y_pred[mask_valid])
                    else:
                        r2 = 0.0
                    
                    # Agregar línea de la curva con R²
                    fig_all.add_trace(go.Scatter(
                        x=x_plot, y=y_grid,
                        mode="lines",
                        name=f"{ciudad} (R²={r2:.3f})",
                        line=dict(width=3)
                    ))
                    
                    # Continuar calculando otras métricas
                    if mask_valid.sum() >= 3:
                        rmse = float(np.sqrt(mean_squared_error(y[mask_valid], y_pred[mask_valid])))
                        mae = float(mean_absolute_error(y[mask_valid], y_pred[mask_valid]))
                        
                        # Correlación de Pearson (r)
                        r_pearson = np.corrcoef(y[mask_valid], y_pred[mask_valid])[0, 1]
                        r_pearson = float(r_pearson) if not np.isnan(r_pearson) else 0.0
                        
                        # MAPE
                        mape = np.mean(np.abs((y[mask_valid] - y_pred[mask_valid]) / y[mask_valid])) * 100
                        mape = float(mape) if not np.isnan(mape) and not np.isinf(mape) else 0.0
                        
                        # Determinar número de parámetros
                        p = reg.coef_.size + 1 if hasattr(reg, "coef_") else 2
                        
                        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
                        
                        resultados_globales.append({
                            "ciudad": ciudad,
                            "Modelo": nombre_modelo,
                            "r (Correlación)": r_pearson,
                            "R²": r2,
                            "R² ajustado": r2_adj,
                            "RMSE": rmse,
                            "MAE": mae,
                            "MAPE (%)": mape,
                            "reg_model": reg
                        })
                except Exception as e:
                    st.warning(f"Error ajustando {ciudad}: {e}")
                    continue
            
            fig_all.update_layout(
                title=f"{y_var} vs {x_var} - {nombre_modelo}",
                xaxis_title=x_var,
                yaxis_title=y_var,
                height=600
            )
            st.plotly_chart(fig_all, use_container_width=True)

        elif modo_grafica_nl == "Pestañas por ciudad":
            # Modo: pestañas por ciudad
            tabs = st.tabs(selected_cities)
            for tab, ciudad in zip(tabs, selected_cities):
                with tab:
                    df_c = df_combined[df_combined["ciudad"] == ciudad][[x_var, y_var]].dropna()
                    resultado = _plot_nonlinear_ciudad(df_c, ciudad, x_var, y_var, tipo_modelo, nombre_modelo, mostrar_metricas=False)
                    if resultado:
                        resultados_globales.append(resultado)

        else:  # "Cuadrícula por ciudad"
            # Modo: cuadrícula
            n_cols = st.slider("Columnas de la cuadrícula", 2, 4, min(3, max(2, len(selected_cities))), key="cols_nl")
            cols = st.columns(n_cols)
            
            for i, ciudad in enumerate(selected_cities):
                with cols[i % n_cols]:
                    st.markdown(f"**{ciudad}**")
                    df_c = df_combined[df_combined["ciudad"] == ciudad][[x_var, y_var]].dropna()
                    resultado = _plot_nonlinear_ciudad(df_c, ciudad, x_var, y_var, tipo_modelo, nombre_modelo, mostrar_metricas=False)
                    if resultado:
                        resultados_globales.append(resultado)

        # Guardamos resultados en sesión para Tab 2
        st.session_state["nl_results_single_model"] = resultados_globales

        # ================================================================
        # TARJETAS DE MÉTRICAS POR CIUDAD
        # ================================================================
        if resultados_globales:
            for resultado in resultados_globales:
                ciudad = resultado["ciudad"]
                r_pearson = resultado["r (Correlación)"]
                r2 = resultado["R²"]
                r2_adj = resultado["R² ajustado"]
                rmse = resultado["RMSE"]
                mae = resultado["MAE"]
                mape = resultado["MAPE (%)"]
                reg = resultado.get("reg_model", None)
                
                # Calcular ecuación con variables reales (coeficientes del modelo)
                if tipo_modelo == "poly2" and reg is not None:
                    coefs = reg.coef_ if hasattr(reg, "coef_") else []
                    intercept = reg.intercept_ if hasattr(reg, "intercept_") else 0
                    if len(coefs) >= 3:
                        ecuacion = f"{y_var} = {coefs[2]:.4f}·{x_var}² + {coefs[1]:.4f}·{x_var} + {intercept:.4f}"
                    else:
                        ecuacion = f"{y_var} = [Coeficientes no disponibles]"
                elif tipo_modelo == "poly3" and reg is not None:
                    coefs = reg.coef_ if hasattr(reg, "coef_") else []
                    intercept = reg.intercept_ if hasattr(reg, "intercept_") else 0
                    if len(coefs) >= 4:
                        ecuacion = f"{y_var} = {coefs[3]:.4f}·{x_var}³ + {coefs[2]:.4f}·{x_var}² + {coefs[1]:.4f}·{x_var} + {intercept:.4f}"
                    else:
                        ecuacion = f"{y_var} = [Coeficientes no disponibles]"
                elif tipo_modelo == "log" and reg is not None:
                    b0 = reg.intercept_ if hasattr(reg, "intercept_") else 0
                    b1 = reg.coef_[0] if hasattr(reg, "coef_") else 0
                    ecuacion = f"{y_var} = {b1:.4f}·ln({x_var}) + {b0:.4f}"
                elif tipo_modelo == "exp" and reg is not None:
                    b = reg.coef_[0] if hasattr(reg, "coef_") else 0
                    ln_a = reg.intercept_ if hasattr(reg, "intercept_") else 0
                    a = np.exp(ln_a)
                    ecuacion = f"{y_var} = {a:.4f}·e^({b:.4f}·{x_var})"
                elif tipo_modelo == "pow" and reg is not None:
                    b = reg.coef_[0] if hasattr(reg, "coef_") else 0
                    ln_a = reg.intercept_ if hasattr(reg, "intercept_") else 0
                    a = np.exp(ln_a)
                    ecuacion = f"{y_var} = {a:.4f}·{x_var}^{b:.4f}"
                elif tipo_modelo == "sqrt" and reg is not None:
                    b0 = reg.intercept_ if hasattr(reg, "intercept_") else 0
                    b1 = reg.coef_[0] if hasattr(reg, "coef_") else 0
                    ecuacion = f"{y_var} = {b1:.4f}·√{x_var} + {b0:.4f}"
                elif tipo_modelo == "inversa" and reg is not None:
                    a = reg.coef_[0] if hasattr(reg, "coef_") and len(reg.coef_) > 0 else 0
                    ecuacion = f"{y_var} = {a:.4f}·(1/{x_var})"
                else:
                    ecuacion = f"{y_var} = [Ecuación {tipo_modelo} - ajuste no lineal complejo]"
                
                st.markdown(f"""
                <div style="background:#F7F7F7;border:1px solid rgba(0,0,0,.08);border-radius:16px;
                            padding:14px 16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.05);">
                    <h4 style="margin:0 0 12px 0;font-weight:800;color:#FF385C;text-align:center;">Métricas del Modelo - {ciudad}</h4>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:14px;">
                        <div style="background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                                    padding:10px 8px;">
                            <div style="font-size:11px;color:#666;text-align:center;">r (Correlación)</div>
                            <div style="font-size:20px;font-weight:700;color:#666;text-align:center;">{r_pearson:.3f}</div>
                        </div>
                        <div style="background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                                    padding:10px 8px;">
                            <div style="font-size:11px;color:#666;text-align:center;">R²</div>
                            <div style="font-size:20px;font-weight:700;color:#666;text-align:center;">{r2:.3f}</div>
                        </div>
                        <div style="background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                                    padding:10px 8px;">
                            <div style="font-size:11px;color:#666;text-align:center;">R² Ajustado</div>
                            <div style="font-size:20px;font-weight:700;color:#666;text-align:center;">{r2_adj:.3f}</div>
                        </div>
                        <div style="background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                                    padding:10px 8px;">
                            <div style="font-size:11px;color:#666;text-align:center;">RMSE</div>
                            <div style="font-size:20px;font-weight:700;color:#666;text-align:center;">{rmse:.2f}</div>
                        </div>
                        <div style="background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                                    padding:10px 8px;">
                            <div style="font-size:11px;color:#666;text-align:center;">MAE</div>
                            <div style="font-size:20px;font-weight:700;color:#666;text-align:center;">{mae:.2f}</div>
                        </div>
                        <div style="background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:12px;
                                    padding:10px 8px;">
                            <div style="font-size:11px;color:#666;text-align:center;">MAPE</div>
                            <div style="font-size:20px;font-weight:700;color:#666;text-align:center;">{mape:.2f}%</div>
                        </div>
                    </div>
                    <div style="padding:12px;background:#fff;border:1px solid rgba(0,0,0,.06);
                                border-radius:12px;font-size:16px;font-weight:600;text-align:center;color:#484848;">
                        {ecuacion}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

        # ================================================================
        # COMPARATIVA DE AJUSTE ENTRE CIUDADES
        # ================================================================
        if resultados_globales:
            st.markdown("---")
            st.markdown("""
            <div style='padding: 15px 0 15px 0;'>
                <h3 style='color: #484848; margin: 0 0 5px 0; font-weight: 700; font-size: 22px;'>Comparativa de Desempeño</h3>
                <p style='color: #767676; margin: 0; font-size: 14px;'>Métricas del modelo {nombre_modelo} en cada ciudad</p>
            </div>
            """.format(nombre_modelo=nombre_modelo), unsafe_allow_html=True)
            df_comp = pd.DataFrame(resultados_globales)
            
            # Renombrar columna "ciudad" a "Ciudad" con mayúscula
            if 'ciudad' in df_comp.columns:
                df_comp = df_comp.rename(columns={'ciudad': 'Ciudad'})
            
            # CSS personalizado para encabezados de tabla
            st.markdown("""
                <style>
                /* Encabezados de tabla */
                [data-testid="stDataFrame"] thead tr th,
                [data-testid="stDataFrame"] thead th,
                div[data-testid="stDataFrame"] table thead tr th {
                    background-color: #FFE5E9 !important;
                    color: #FF385C !important;
                    text-align: center !important;
                    font-weight: 700 !important;
                    font-size: 14px !important;
                    padding: 12px 8px !important;
                }
                /* Celdas de datos */
                [data-testid="stDataFrame"] tbody tr td,
                [data-testid="stDataFrame"] tbody td,
                div[data-testid="stDataFrame"] table tbody tr td {
                    text-align: center !important;
                    padding: 10px 8px !important;
                }
                /* Centrar contenido en todas las celdas */
                div[data-testid="stDataFrame"] table td div,
                div[data-testid="stDataFrame"] table th div {
                    text-align: center !important;
                    justify-content: center !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(df_comp.round(4), use_container_width=True, hide_index=True)

        # ================================================================
        # HALLAZGOS IMPORTANTES DE REGRESIÓN NO LINEAL
        # ================================================================
        if resultados_globales:
            with st.expander("Análisis de resultados", expanded=False):
                df_comp = pd.DataFrame(resultados_globales)
                
                # Análisis inteligente del modelo
                df_sorted = df_comp.sort_values("R²", ascending=False)
                mejor_ciudad = df_sorted.iloc[0]["ciudad"]
                mejor_r2 = df_sorted.iloc[0]["R²"]
                mejor_r = df_sorted.iloc[0]["r (Correlación)"]
                peor_ciudad = df_sorted.iloc[-1]["ciudad"]
                peor_r2 = df_sorted.iloc[-1]["R²"]
                peor_r = df_sorted.iloc[-1]["r (Correlación)"]
                r2_promedio = df_comp["R²"].mean()
                rmse_promedio = df_comp["RMSE"].mean()
                mae_promedio = df_comp["MAE"].mean()
                if "MAPE (%)" in df_comp.columns:
                    mape_promedio = df_comp["MAPE (%)"].mean()
                else:
                    mape_promedio = None
                
                # Título contextual
                st.markdown(f"**Análisis:** `{x_var}` → `{y_var}` con modelo **{nombre_modelo}**")
                
                # Interpretación del modelo específico (conciso)
                interpretaciones = {
                    "poly2": f"Captura una relación curva con punto óptimo. Valores extremos de `{x_var}` pueden reducir `{y_var}`.",
                    "poly3": f"Captura múltiples cambios de tendencia en la relación. Útil para patrones complejos pero riesgo de sobreajuste.",
                    "log": f"Rendimientos decrecientes: las primeras unidades de `{x_var}` impactan más en `{y_var}` que las siguientes.",
                    "exp": f"Crecimiento acelerado: cada incremento en `{x_var}` aumenta `{y_var}` de forma exponencial. Típico de mercados premium.",
                    "pow": f"Elasticidad constante: cambios porcentuales en `{x_var}` producen cambios proporcionales en `{y_var}`.",
                    "sqrt": f"Rendimientos decrecientes suaves: aumentos grandes en `{x_var}` tienen impacto moderado en `{y_var}`.",
                    "inversa": f"Relación inversa: aumentar `{x_var}` disminuye `{y_var}`. Indica saturación o efectos negativos."
                }
                
                st.info(f"**Interpretación del modelo:** {interpretaciones.get(tipo_modelo, f'Describe la relación entre `{x_var}` y `{y_var}`.')}")
                
                # Análisis por ciudad (compacto)
                st.markdown("### Comparación por ciudad")
                col1, col2 = st.columns(2)
                with col1:
                    calidad_mejor = "Excelente" if mejor_r2 > 0.7 else "Buena" if mejor_r2 > 0.5 else "Moderada" if mejor_r2 > 0.3 else "Baja"
                    st.success(f"""
                    **{mejor_ciudad}** (mejor ajuste)
                    - R² = {mejor_r2:.3f} ({mejor_r2*100:.1f}%)
                    - Correlación = {mejor_r:+.3f}
                    - {calidad_mejor} capacidad predictiva
                    """)
                with col2:
                    calidad_peor = "Aceptable" if peor_r2 > 0.5 else "Limitada" if peor_r2 > 0.3 else "Baja"
                    st.warning(f"""
                    **{peor_ciudad}** (menor ajuste)
                    - R² = {peor_r2:.3f} ({peor_r2*100:.1f}%)
                    - Correlación = {peor_r:+.3f}
                    - {calidad_peor} capacidad predictiva
                    """)
                
                # Métricas de error (compacto)
                st.markdown("### Métricas de error promedio")
                cols = st.columns(3 if mape_promedio is not None else 2)
                with cols[0]:
                    st.metric("RMSE", f"{rmse_promedio:.2f}", help=f"Error cuadrático medio en unidades de {y_var}")
                with cols[1]:
                    st.metric("MAE", f"{mae_promedio:.2f}", help=f"Error absoluto medio en unidades de {y_var}")
                if mape_promedio is not None:
                    with cols[2]:
                        st.metric("MAPE", f"{mape_promedio:.1f}%", help="Error porcentual medio")
                
                # Resumen ejecutivo (inteligente)
                if r2_promedio > 0.7:
                    conclusion = f"Excelente modelo para predecir `{y_var}` desde `{x_var}`. Alta confiabilidad en {mejor_ciudad}."
                elif r2_promedio > 0.5:
                    conclusion = f"Buen modelo predictivo. `{x_var}` explica una parte significativa de `{y_var}`, especialmente en {mejor_ciudad}."
                elif r2_promedio > 0.3:
                    conclusion = f"Modelo moderado. `{x_var}` tiene relación con `{y_var}`, pero otros factores también influyen considerablemente."
                else:
                    conclusion = f"Relación débil. Considera usar otras variables predictoras o modelos alternativos."
                
                # Advertencias específicas por tipo de modelo
                advertencias = []
                if tipo_modelo == "poly3" and r2_promedio > 0.8:
                    advertencias.append("Posible sobreajuste con modelo cúbico. Valida con datos nuevos.")
                if tipo_modelo == "exp" and rmse_promedio > mae_promedio * 2:
                    advertencias.append("El modelo exponencial produce errores grandes en valores extremos.")
                if tipo_modelo in ["log", "pow"] and x_var in ["price", "precio"]:
                    advertencias.append(f"Los modelos {tipo_modelo} pueden no ser ideales cuando `{x_var}` tiene valores cercanos a cero.")
                
                st.markdown(f"**Conclusión:** {conclusion}")
                if advertencias:
                    for adv in advertencias:
                        st.warning(adv)
                
                # Contexto adicional específico solo para interpretación avanzada
                context_vars = {
                    "accommodates": f"Mayor capacidad típicamente aumenta `{y_var}`. Modelos recomendados: potencial o cuadrático.",
                    "beds": f"Rendimientos decrecientes: pasar de 1→2 camas impacta más en `{y_var}` que 10→11. Modelo recomendado: logarítmico.",
                    "bedrooms": f"Variable de alto impacto. Más habitaciones indica propiedades más grandes. Modelo recomendado: cuadrático o potencial.",
                    "bathrooms_num": f"Puede existir punto óptimo (2-3 baños). Más baños no siempre incrementa `{y_var}` proporcionalmente.",
                    "amenities_count": f"Las primeras 10-15 amenidades impactan más en `{y_var}`. Modelo recomendado: logarítmico.",
                    "number_of_reviews": f"Las primeras reseñas son críticas. Efecto marginal disminuye después. Modelo recomendado: logarítmico.",
                    "review_scores_rating": f"Diferencias pequeñas en calificación pueden tener impacto significativo en `{y_var}`.",
                    "minimum_nights": f"Requisitos altos limitan demanda pero atraen huéspedes diferentes. Trade-off: flexibilidad vs estabilidad.",
                    "availability_365": f"Alta disponibilidad puede indicar baja ocupación. Relación con `{y_var}` no es directamente causal.",
                    "price": f"Como variable independiente, puede predecir demanda, reviews o disponibilidad.",
                    "precio": f"Como variable independiente, puede predecir demanda, reviews o disponibilidad."
                }
                
                if x_var in context_vars and mejor_r2 > 0.4:
                    st.markdown(f"**Nota sobre `{x_var}`:** {context_vars[x_var]}")

    # ================================================================


    # ================================================================
    # TAB 2: COMPARACIÓN MULTI-MODELO
    # ================================================================
    with tab2:
        # ================================================================
        # COMPARACIÓN VISUAL CON TODAS LAS CURVAS
        # ================================================================
        st.markdown("""
        <div style='padding: 10px 0 10px 0;'>
            <h3 style='color: #484848; margin: 0 0 5px 0; font-weight: 700; font-size: 20px;'>Visualización Comparativa</h3>
            <p style='color: #767676; margin: 0; font-size: 14px;'>Análisis de {y_var} en función de {x_var}</p>
        </div>
        """.format(y_var=y_var, x_var=x_var), unsafe_allow_html=True)
        
        # Selector de ciudad
        ciudad_analisis = st.selectbox(
            "Selecciona la ciudad para analizar",
            selected_cities,
            key="ciudad_multi"
        )
        
        df_ciudad = df_combined[df_combined["ciudad"] == ciudad_analisis]
        resultados_ciudad = ajustar_todos_modelos_tab3(df_ciudad, x_var, y_var)
        
        if resultados_ciudad:
            # Crear gráfico con las 6 curvas superpuestas
            df_plot = df_ciudad[[x_var, y_var]].dropna()
            x_data = df_plot[x_var].values.astype(float)
            y_data = df_plot[y_var].values.astype(float)
            
            x_grid = np.linspace(x_data.min(), x_data.max(), 300)
            
            fig = go.Figure()
            
            # Agregar puntos reales
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data,
                mode="markers",
                name="Datos reales",
                marker=dict(size=6, color='lightgray', opacity=0.6)
            ))
            
            # Colores para cada modelo (9 colores para 9 modelos)
            colores = ["#FF385C", "#00A699", "#FC642D", "#767676", "#484848", "#008489", "#E91E63", "#9C27B0", "#3F51B5"]
            
            # Agregar curva de cada modelo
            for i, (nombre, metricas) in enumerate(resultados_ciudad.items()):
                try:
                    f_pred = metricas["f_pred"]
                    # Para algunos modelos, filtrar x_grid apropiadamente
                    if "Inversa" in nombre or "Cociente" in nombre:
                        x_plot = x_grid[x_grid != 0]
                    elif "Logarítmico" in nombre or "Potencia" in nombre:
                        x_plot = x_grid[x_grid > 0]
                    else:
                        x_plot = x_grid
                    
                    y_grid = f_pred(x_plot)
                    
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_grid,
                        mode="lines",
                        name=f"{nombre} (R²={metricas['R²']:.3f})",
                        line=dict(width=3, color=colores[i % len(colores)])
                    ))
                except Exception as e:
                    continue
            
            fig.update_layout(
                title=f"Comparación de Modelos - {ciudad_analisis}",
                xaxis_title=x_var,
                yaxis_title=y_var,
                height=600,
                hovermode="x unified",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(255,255,255,0.8)"
            )
        )
        
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de ranking
            st.markdown("""
            <div style='padding: 15px 0 8px 0; margin-top: 20px;'>
                <h3 style='color: #484848; margin: 0 0 5px 0; font-weight: 700; font-size: 20px;'>Ranking de Desempeño</h3>
                <p style='color: #767676; margin: 0; font-size: 14px;'>Modelos ordenados por calidad de ajuste para <strong>{ciudad_analisis}</strong></p>
            </div>
            """.format(ciudad_analisis=ciudad_analisis), unsafe_allow_html=True)
            
            df_ranking = pd.DataFrame([
                {
                    "Posición": idx + 1,
                    "Modelo": nombre,
                    "r (Correlación)": metricas["r (Correlación)"],
                    "R²": metricas["R²"],
                    "R² ajustado": metricas["R² ajustado"],
                    "RMSE": metricas["RMSE"],
                    "MAE": metricas["MAE"],
                    "MAPE (%)": metricas["MAPE (%)"]
                }
                for idx, (nombre, metricas) in enumerate(
                    sorted(resultados_ciudad.items(), key=lambda x: x[1]["R²"], reverse=True)
                )
            ])
            
            st.dataframe(df_ranking.round(4), use_container_width=True, hide_index=True)
            
            # Análisis automático
            mejor_modelo = df_ranking.iloc[0]["Modelo"]
            mejor_r2 = df_ranking.iloc[0]["R²"]
            peor_modelo = df_ranking.iloc[-1]["Modelo"]
            peor_r2 = df_ranking.iloc[-1]["R²"]
            diferencia = mejor_r2 - peor_r2
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='padding: 10px 0 5px 0;'>
                <h4 style='color: #484848; margin: 0; font-weight: 700; font-size: 18px;'>Recomendación Automática</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"""
            **Mejor modelo**: {mejor_modelo} con R² = {mejor_r2:.3f}
            
            **Peor modelo**: {peor_modelo} con R² = {peor_r2:.3f}
            
            **Diferencia de ajuste**: {diferencia:.3f} ({diferencia*100:.1f}% de mejora)
            
            **Recomendación**: {"El mejor modelo supera significativamente a los demás. Usar este modelo para predicciones." if diferencia > 0.15 else "Los modelos tienen desempeño similar. Priorizar simplicidad o interpretabilidad."}
            """)
        else:
            st.warning(f"No hay suficientes datos para ajustar modelos en {ciudad_analisis}")

# ====== VISTA: REGRESIÓN LOGÍSTICA ======
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def preparar_df_logistico(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables dicotómicas y dummies para regresión logística.
    Devuelve un df extendido con nuevas columnas binarias.
    """
    df = df_raw.copy()
    bin_map_tf = {"t": 1, "f": 0, "True": 1, "False": 0, True: 1, False: 0}

    # --- Variables binarias directas t/f (MANTENER SOLO LAS RELEVANTES) ---
    if "host_is_superhost" in df.columns:
        df["y_superhost"] = df["host_is_superhost"].map(bin_map_tf)

    if "instant_bookable" in df.columns:
        df["instant_bookable_bin"] = df["instant_bookable"].map(bin_map_tf)

    if "host_identity_verified" in df.columns:
        df["host_identity_verified_bin"] = df["host_identity_verified"].map(bin_map_tf)

    # NUEVAS VARIABLES AGREGADAS
    if "require_guest_profile_picture" in df.columns:
        df["require_guest_pic_bin"] = df["require_guest_profile_picture"].map(bin_map_tf)
    
    if "require_guest_phone_verification" in df.columns:
        df["require_guest_phone_bin"] = df["require_guest_phone_verification"].map(bin_map_tf)

    # --- Variables presencia/ausencia ---
    if "license" in df.columns:
        df["has_license"] = df["license"].notna().astype(int)

    if "host_verifications" in df.columns:
        df["has_work_email"] = df["host_verifications"].astype(str).str.contains("work_email", case=False, na=False).astype(int)
        df["has_phone_verif"] = df["host_verifications"].astype(str).str.contains("phone", case=False, na=False).astype(int)

    if "bathrooms_text" in df.columns:
        txt = df["bathrooms_text"].astype(str).str.lower()
        df["has_private_bath"] = txt.str.contains("private", na=False).astype(int)
        df["has_shared_bath"] = txt.str.contains("shared", na=False).astype(int)

    # NUEVA: Host profesional (múltiples propiedades)
    if "calculated_host_listings_count" in df.columns:
        df["is_professional_host"] = (pd.to_numeric(df["calculated_host_listings_count"], errors="coerce") > 1).astype(int)
    
    # NUEVA: Alta calificación
    if "review_scores_rating" in df.columns:
        df["high_rating"] = (pd.to_numeric(df["review_scores_rating"], errors="coerce") >= 4.5).astype(int)

    # NUEVA: Mucha experiencia (muchas reseñas)
    if "number_of_reviews" in df.columns:
        df["experienced_listing"] = (pd.to_numeric(df["number_of_reviews"], errors="coerce") >= 10).astype(int)

    # --- Dummies categóricas compactas ---
    if "room_type" in df.columns:
        room_dummies = pd.get_dummies(df["room_type"], prefix="room", drop_first=True)
        df = pd.concat([df, room_dummies], axis=1)

    if "host_response_time" in df.columns:
        resp_dummies = pd.get_dummies(df["host_response_time"], prefix="resp", drop_first=True)
        df = pd.concat([df, resp_dummies], axis=1)
    
    # NUEVA: Dummies para cancellation_policy
    if "cancellation_policy" in df.columns:
        cancel_dummies = pd.get_dummies(df["cancellation_policy"], prefix="cancel", drop_first=True)
        df = pd.concat([df, cancel_dummies], axis=1)

    return df


def get_city_cmap(ciudad: str, idx_fallback: int = 0) -> str:
    """
    Devuelve un cmap 'representativo' según la ciudad con colores característicos.
    Si no se reconoce, usa una lista de cmaps distintos como fallback.
    """
    c = str(ciudad).lower()

    # Colores característicos por ciudad:
    if "amsterdam" in c:
        return "Oranges"     # naranja (típico de Holanda) 🇳🇱
    if "atenas" in c or "athens" in c or "grecia" in c:
        return "Blues"       # azul (bandera griega + mar Egeo) 🇬🇷
    if "barcelona" in c:
        return "RdPu"        # rojo-púrpura (blaugrana del FC Barcelona) 🇪🇸
    if "madrid" in c:
        return "Reds"        # rojo (bandera española) 🇪🇸
    if "milan" in c or "milano" in c:
        return "Greens"      # verde (bandera italiana + moda) 🇮🇹
    if "paris" in c:
        return "Purples"     # púrpura (elegancia parisina) 🇫🇷
    if "berlin" in c:
        return "Greys"       # gris (arquitectura berlinesa) 🇩🇪
    if "roma" in c or "rome" in c:
        return "Oranges"     # naranja (colores romanos)
    if "lisboa" in c or "lisbon" in c:
        return "YlGn"        # amarillo-verde (azulejos portugueses)

    # Fallback: rotar entre varios
    fallback_cmaps = ["Greens", "Blues", "Reds", "Purples", "Oranges", "YlOrRd", "YlGn", "Greys"]
    return fallback_cmaps[idx_fallback % len(fallback_cmaps)]

# =========================================================
# ================== REGRESIÓN LOGÍSTICA ==================
# =========================================================
# =========================================================
# ================== REGRESIÓN LOGÍSTICA ==================
# =========================================================
if View == "Regresión Logística":

    col_title, col_help = st.columns([0.95, 0.05])
    with col_title:
        st.title("Airbnb – Regresión Logística")
    with col_help:
        with st.popover("➕"):
            st.markdown("**¿Qué es la Regresión Logística?**")
            st.write("Método de clasificación que predice categorías (sí/no, alto/bajo). En este análisis, clasifica si un listing de Airbnb está en un rango de precio alto o bajo, identificando qué características (ubicación, amenidades, tipo de propiedad) influyen en pertenecer a cada segmento.")

    # --- Validación columna ciudad ---
    if "ciudad" not in df.columns or df["ciudad"].dropna().empty:
        st.warning("No hay columna 'ciudad' válida en el DataFrame.")
        st.stop()

    # --- Preprocesamiento específico para logística (dummies y binarios) ---
    df_logit = preparar_df_logistico(df)

    # ===== Elegir variable objetivo (Y) =====
    # Candidatas: todas las columnas binarias 0/1
    posibles_targets = []
    for col in df_logit.columns:
        vals = df_logit[col].dropna().unique()
        if len(vals) == 2 and set(vals).issubset({0, 1}):
            posibles_targets.append(col)

    prioridad = [
        "y_superhost",
        "instant_bookable_bin",
        "host_identity_verified_bin",
        "host_has_profile_pic_bin",
        "has_license",
    ]
    posibles_targets = sorted(
        posibles_targets,
        key=lambda c: (prioridad.index(c) if c in prioridad else len(prioridad) + 1, c),
    )

    if not posibles_targets:
        st.error("No se encontraron variables binarias (0/1) para usar como objetivo en la regresión logística.")
        st.stop()

    st.sidebar.header("Config. – Regresión Logística")

    target_col = st.sidebar.selectbox(
        "Variable objetivo (Y)",
        options=posibles_targets,
        index=posibles_targets.index("y_superhost") if "y_superhost" in posibles_targets else 0,
        key="logit_target",
    )

    target_labels_map = {
        "y_superhost": "Superhost",
        "instant_bookable_bin": "Reservación instantánea",
        "host_identity_verified_bin": "Identidad verificada",
        "require_guest_pic_bin": "Requiere foto de huésped",
        "require_guest_phone_bin": "Requiere teléfono de huésped",
        "has_license": "Con licencia",
        "has_private_bath": "Baño privado",
        "has_shared_bath": "Baño compartido",
        "is_professional_host": "Host profesional",
        "high_rating": "Alta calificación",
        "experienced_listing": "Anuncio con experiencia",
    }
    target_name = target_labels_map.get(target_col, target_col)

    # Filtrar filas válidas para ese target
    df_logit = df_logit[df_logit[target_col].isin([0, 1])].copy()
    if df_logit.empty:
        st.error(f"No hay datos válidos para la variable objetivo seleccionada ({target_col}).")
        st.stop()

    # --- Selección de ciudades ---
    ciudades_disp = sorted(df_logit["ciudad"].dropna().unique().tolist())
    selected_cities = st.sidebar.multiselect(
        "Ciudades (1 a 5)",
        options=ciudades_disp,
        default=ciudades_disp[:min(3, len(ciudades_disp))],
        max_selections=5,
        key="logit_cities",
    )

    if not selected_cities:
        st.info("Selecciona al menos una ciudad para continuar.")
        st.stop()

    df_logit = df_logit[df_logit["ciudad"].isin(selected_cities)].copy()

    # ===== Variables predictoras (X) =====
    # Binarias base conocidas (ACTUALIZADAS - eliminadas las de baja variabilidad)
    bin_base = [
        "instant_bookable_bin",
        "host_identity_verified_bin",
        "require_guest_pic_bin",
        "require_guest_phone_bin",
        "has_license",
        "has_work_email",
        "has_phone_verif",
        "has_private_bath",
        "has_shared_bath",
        "is_professional_host",
        "high_rating",
        "experienced_listing",
    ]
    # Dummies de room_type, host_response_time y cancellation_policy
    room_cols = [c for c in df_logit.columns if c.startswith("room_")]
    resp_cols = [c for c in df_logit.columns if c.startswith("resp_")]
    cancel_cols = [c for c in df_logit.columns if c.startswith("cancel_")]

    num_candidates = [
        "price",
        "accommodates",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "availability_365",
    ]

    feature_options = []
    for c in bin_base + room_cols + resp_cols + cancel_cols + num_candidates:
        if c in df_logit.columns:
            feature_options.append(c)

    feature_options = sorted(set(feature_options))

    # No permitir que el objetivo aparezca como predictor
    if target_col in feature_options:
        feature_options.remove(target_col)

    if not feature_options:
        st.error("No se encontraron variables predictoras apropiadas para la regresión logística.")
        st.stop()

    default_feats = [c for c in [
        "instant_bookable_bin",
        "host_identity_verified_bin",
        "is_professional_host",
        "high_rating",
        "number_of_reviews",
    ] if c in feature_options]

    X_cols = st.sidebar.multiselect(
        "Variables predictoras (X)",
        options=feature_options,
        default=default_feats if default_feats else feature_options[:4],
        key="logit_features",
    )

    if len(X_cols) < 1:
        st.info("Selecciona al menos una variable predictora.")
        st.stop()

    test_size = st.sidebar.slider(
        "Proporción de test",
        0.1, 0.4, 0.3, 0.05,
        key="logit_test_size"
    )

    st.markdown(
        f"**Objetivo del modelo:** predecir `{target_name}` (1) a partir de las variables seleccionadas en X."
    )

    tab_model, tab_heatmap = st.tabs(["Modelo y matrices de confusión ", "Correlaciones por ciudad "])


    # =====================================================
    # ============ TAB 2: MODELO + CONFUSIONES S
    # =====================================================
    with tab_model:
        st.subheader("Regresión Logística por ciudad")

        metrics_rows = []
        cols_per_row = min(3, len(selected_cities)) if selected_cities else 1  # máx 3 columnas

        for idx_city, ciudad in enumerate(selected_cities):

            # Nueva fila de columnas cuando toca
            if idx_city % cols_per_row == 0:
                col_objs = st.columns(cols_per_row)

            col = col_objs[idx_city % cols_per_row]

            with col:
                st.markdown(f"**{ciudad}**")

                df_city = df_logit[df_logit["ciudad"] == ciudad].copy()
                df_city = df_city[X_cols + [target_col]].dropna()

                if df_city[target_col].nunique() < 2:
                    st.info("Solo hay una clase en esta ciudad (todo 0 o todo 1).")
                    continue

                if len(df_city) < 50:
                    st.warning(f"Pocos datos en {ciudad} ({len(df_city)} filas).")

                X = df_city[X_cols].values
                y = df_city[target_col].values

                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_size,
                        random_state=42,
                        stratify=y,
                    )
                except ValueError as e:
                    st.warning(f"No se pudo hacer train/test split: {e}")
                    continue

                logreg = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced"  # Maneja desbalance de clases
                )

                try:
                    logreg.fit(X_train, y_train)
                except Exception as e:
                    st.warning(f"No se pudo entrenar el modelo: {e}")
                    continue

                y_pred = logreg.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Calcular sensibilidad y especificidad
                cm_temp = confusion_matrix(y_test, y_pred)
                tn_temp, fp_temp, fn_temp, tp_temp = cm_temp.ravel()
                sensibilidad = tp_temp / (tp_temp + fn_temp) if (tp_temp + fn_temp) > 0 else 0
                especificidad = tn_temp / (tn_temp + fp_temp) if (tn_temp + fp_temp) > 0 else 0
                
                # Calcular precisión de la clase 0 (negativa)
                precision_clase0 = tn_temp / (tn_temp + fn_temp) if (tn_temp + fn_temp) > 0 else 0

                metrics_rows.append({
                    "ciudad": ciudad,
                    "target": target_col,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "sensibilidad": sensibilidad,
                    "especificidad": precision_clase0,
                })

                cm = confusion_matrix(y_test, y_pred)
                cmap_city = get_city_cmap(ciudad, idx_city)
                
                # Extraer valores de la matriz de confusión
                tn, fp, fn, tp = cm.ravel()

                fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                sns.heatmap(
                    cm,
                    annot=False,
                    fmt="d",
                    cmap=cmap_city,
                    cbar=False,
                    xticklabels=[f"No {target_name} (0)", f"{target_name} (1)"],
                    yticklabels=[f"No {target_name} (0)", f"{target_name} (1)"],
                    ax=ax_cm,
                )
                
                # Añadir etiquetas personalizadas con valores y nombres
                ax_cm.text(0.5, 0.5, f'TN\n{tn}', ha='center', va='center', 
                          fontsize=12, fontweight='bold', color='white' if tn > cm.max()/2 else 'black')
                ax_cm.text(1.5, 0.5, f'FP\n{fp}', ha='center', va='center', 
                          fontsize=12, fontweight='bold', color='white' if fp > cm.max()/2 else 'black')
                ax_cm.text(0.5, 1.5, f'FN\n{fn}', ha='center', va='center', 
                          fontsize=12, fontweight='bold', color='white' if fn > cm.max()/2 else 'black')
                ax_cm.text(1.5, 1.5, f'TP\n{tp}', ha='center', va='center', 
                          fontsize=12, fontweight='bold', color='white' if tp > cm.max()/2 else 'black')
                
                ax_cm.set_xlabel("Predicción")
                ax_cm.set_ylabel("Real")
                ax_cm.set_title("Matriz de confusión")
                st.pyplot(fig_cm)
                
                # Análisis de calidad de predicciones
                total_test = tn + fp + fn + tp
                positivos_reales = tp + fn
                negativos_reales = tn + fp
                balance_ratio = positivos_reales / total_test if total_test > 0 else 0
                
                # Detectar problemas
                problemas = []
                if balance_ratio < 0.1 or balance_ratio > 0.9:
                    problemas.append(f"⚠️ Datos muy desbalanceados: {balance_ratio*100:.1f}% positivos")
                if tp == 0:
                    problemas.append("❌ No detectó ningún positivo (TP=0)")
                if fn == positivos_reales and positivos_reales > 0:
                    problemas.append("❌ Predice todo como negativo")
                if sensibilidad < 0.3 and positivos_reales > 0:
                    problemas.append(f"⚠️ Sensibilidad muy baja ({sensibilidad:.2f})")
                if especificidad < 0.3 and negativos_reales > 0:
                    problemas.append(f"⚠️ Especificidad muy baja ({especificidad:.2f})")
                    
                if problemas:
                    with st.expander("⚠️ Análisis de Calidad", expanded=False):
                        for p in problemas:
                            st.markdown(f"- {p}")
                        st.markdown(f"""
                        **Distribución real en test:**
                        - Positivos: {positivos_reales} ({balance_ratio*100:.1f}%)
                        - Negativos: {negativos_reales} ({(1-balance_ratio)*100:.1f}%)
                        
                        **Recomendaciones:**
                        - Usa `class_weight='balanced'` en el modelo
                        - Considera técnicas de resampling (SMOTE)
                        - Ajusta el threshold de predicción
                        """)

                # Métricas con diseño profesional Airbnb
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background: #F7F7F7; 
                                border-radius: 8px; border: 1px solid #EBEBEB;'>
                        <div style='color: #717171; font-size: 10px; font-weight: 600; 
                                    text-transform: uppercase; margin-bottom: 4px;'>ACCURACY</div>
                        <div style='color: #FF385C; font-size: 20px; font-weight: 700;'>{acc:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background: #F7F7F7; 
                                border-radius: 8px; border: 1px solid #EBEBEB;'>
                        <div style='color: #717171; font-size: 10px; font-weight: 600; 
                                    text-transform: uppercase; margin-bottom: 4px;'>PRECISIÓN 1</div>
                        <div style='color: #FF385C; font-size: 20px; font-weight: 700;'>{prec:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background: #F7F7F7; 
                                border-radius: 8px; border: 1px solid #EBEBEB;'>
                        <div style='color: #717171; font-size: 10px; font-weight: 600; 
                                    text-transform: uppercase; margin-bottom: 4px;'>RECALL</div>
                        <div style='color: #FF385C; font-size: 20px; font-weight: 700;'>{rec:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                col4, col5, col6 = st.columns(3)
                with col4:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background: #F7F7F7; 
                                border-radius: 8px; border: 1px solid #EBEBEB;'>
                        <div style='color: #717171; font-size: 10px; font-weight: 600; 
                                    text-transform: uppercase; margin-bottom: 4px;'>F1 SCORE</div>
                        <div style='color: #FF385C; font-size: 20px; font-weight: 700;'>{f1:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col5:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background: #F7F7F7; 
                                border-radius: 8px; border: 1px solid #EBEBEB;'>
                        <div style='color: #717171; font-size: 10px; font-weight: 600; 
                                    text-transform: uppercase; margin-bottom: 4px;'>SENSIBILIDAD</div>
                        <div style='color: #FF385C; font-size: 20px; font-weight: 700;'>{sensibilidad:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col6:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background: #F7F7F7; 
                                border-radius: 8px; border: 1px solid #EBEBEB;'>
                        <div style='color: #717171; font-size: 10px; font-weight: 600; 
                                    text-transform: uppercase; margin-bottom: 4px;'>PRECISIÓN 0</div>
                        <div style='color: #FF385C; font-size: 20px; font-weight: 700;'>{especificidad:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Resumen global debajo de la cuadrícula
        if metrics_rows:
            st.markdown("---")
            st.subheader("Resumen de métricas por ciudad")
            df_metrics = pd.DataFrame(metrics_rows)
            # Eliminar columnas de tamaño de muestra y renombrar especificidad
            df_metrics_display = df_metrics.drop(columns=['n_train', 'n_test'], errors='ignore')
            df_metrics_display = df_metrics_display.rename(columns={'especificidad': 'precision_clase0'})
            st.dataframe(df_metrics_display.round(4), use_container_width=True)

        # --- Expander explicativo de variables dicotómicas ---
        st.markdown("---")
        with st.expander("Variables Dicotómicas Disponibles", expanded=False):
            # Crear diccionario de descripciones
            var_descriptions = {
                "y_superhost": ("host_is_superhost", "1 si es Superhost, 0 si no", "Target principal - Hosts de alta calidad"),
                "host_identity_verified_bin": ("host_identity_verified", "1 si identidad verificada, 0 si no", "Confiabilidad del host"),
                "is_professional_host": ("calculated_host_listings_count", "1 si tiene >1 propiedad, 0 si no", "Profesionalismo (vs casual)"),
                "instant_bookable_bin": ("instant_bookable", "1 si permite reserva instantánea, 0 si no", "Facilidad de reserva"),
                "require_guest_pic_bin": ("require_guest_profile_picture", "1 si requiere foto del huésped, 0 si no", "Selectividad del host"),
                "require_guest_phone_bin": ("require_guest_phone_verification", "1 si requiere teléfono, 0 si no", "Nivel de verificación"),
                "high_rating": ("review_scores_rating", "1 si rating ≥ 4.5, 0 si no", "Excelente calidad"),
                "experienced_listing": ("number_of_reviews", "1 si tiene ≥10 reseñas, 0 si no", "Experiencia comprobada"),
                "has_license": ("license", "1 si tiene licencia, 0 si no", "Legalidad"),
                "has_private_bath": ("bathrooms_text", "1 si tiene baño privado, 0 si no", "Privacidad"),
                "has_shared_bath": ("bathrooms_text", "1 si tiene baño compartido, 0 si no", "Tipo de alojamiento"),
                "has_work_email": ("host_verifications", "1 si tiene email laboral verificado, 0 si no", "Profesionalismo"),
                "has_phone_verif": ("host_verifications", "1 si tiene teléfono verificado, 0 si no", "Confiabilidad"),
                "price": ("price", "Precio por noche", "Variable numérica"),
                "accommodates": ("accommodates", "Número de huéspedes", "Variable numérica"),
                "minimum_nights": ("minimum_nights", "Noches mínimas de estancia", "Variable numérica"),
                "number_of_reviews": ("number_of_reviews", "Número total de reseñas", "Variable numérica"),
                "reviews_per_month": ("reviews_per_month", "Reseñas promedio por mes", "Variable numérica"),
                "availability_365": ("availability_365", "Días disponibles en el año", "Variable numérica"),
            }
            
            # Contar por tipo
            binarias_count = len([v for v in feature_options if v in var_descriptions and not v.startswith(("room_", "resp_", "cancel_")) and v not in num_candidates])
            numericas_count = len([v for v in feature_options if v in num_candidates])
            dummies_count = len([v for v in feature_options if v.startswith(("room_", "resp_", "cancel_"))])
            
            st.markdown(f"""
            **Variables seleccionadas actualmente:** {len(X_cols)}  
            **Total de variables disponibles:** {len(feature_options)} ({binarias_count} binarias + {numericas_count} numéricas + {dummies_count} categóricas)
            
            **Nota:** Las variables categóricas (dummies) son variables como "tipo de habitación" o "política de cancelación" 
            que se convierten en múltiples variables binarias (0/1). Por ejemplo, `room_type` con valores "Entire home", 
            "Private room", "Shared room" se convierte en `room_Private room` y `room_Shared room`.
            """)
            
            # Separar por tipo
            binarias_disponibles = [v for v in feature_options if v in var_descriptions and not v.startswith(("room_", "resp_", "cancel_")) and v not in num_candidates]
            numericas_disponibles = [v for v in feature_options if v in num_candidates]
            dummies_room = [v for v in feature_options if v.startswith("room_")]
            dummies_resp = [v for v in feature_options if v.startswith("resp_")]
            dummies_cancel = [v for v in feature_options if v.startswith("cancel_")]
            
            if binarias_disponibles:
                st.markdown("#### Variables Binarias (0/1)")
                tabla_bin = "| Variable | Nombre Original | Descripción |\n|----------|-----------------|-------------|\n"
                for var in binarias_disponibles:
                    if var in var_descriptions:
                        orig, desc, _ = var_descriptions[var]
                        tabla_bin += f"| `{var}` | `{orig}` | {desc} |\n"
                st.markdown(tabla_bin)
            
            if numericas_disponibles:
                st.markdown("#### Variables Numéricas")
                tabla_num = "| Variable | Nombre Original | Descripción |\n|----------|-----------------|-------------|\n"
                for var in numericas_disponibles:
                    if var in var_descriptions:
                        orig, desc, _ = var_descriptions[var]
                        tabla_num += f"| `{var}` | `{orig}` | {desc} |\n"
                st.markdown(tabla_num)
            
            if dummies_room or dummies_resp or dummies_cancel:
                st.markdown("#### Variables Categóricas (Dummies)")
                if dummies_room:
                    st.markdown(f"**Tipo de habitación:** {', '.join([f'`{v}`' for v in dummies_room])}")
                if dummies_resp:
                    st.markdown(f"**Tiempo de respuesta:** {', '.join([f'`{v}`' for v in dummies_resp])}")
                if dummies_cancel:
                    st.markdown(f"**Política de cancelación:** {', '.join([f'`{v}`' for v in dummies_cancel])}")
            
            # Mostrar targets disponibles
            st.markdown("#### Variables Objetivo (Target) Disponibles")
            targets_desc = [f"`{t}`" for t in posibles_targets if t in var_descriptions]
            if targets_desc:
                st.markdown(f"**Opciones:** {', '.join(targets_desc)}")
            st.markdown(f"**Seleccionado actualmente:** `{target_col}` ({target_name})")
    
    # =====================================================
    # =========== TAB 1: HEATMAPS POR CIUDAD (GRID) =======
    # =====================================================
    with tab_heatmap:
        st.subheader("Heatmaps de correlaciones por ciudad")

        cols_corr = X_cols + [target_col]
        cols_per_row = min(3, len(selected_cities))  # máx 3 por fila

        for idx_city, ciudad in enumerate(selected_cities):
            if idx_city % cols_per_row == 0:
                col_objs = st.columns(cols_per_row)

            col = col_objs[idx_city % cols_per_row]

            with col:
                st.markdown(f"**{ciudad}**")

                df_city = df_logit[df_logit["ciudad"] == ciudad][cols_corr].dropna()
                if df_city.empty:
                    st.info("Sin datos completos para esta ciudad.")
                    continue

                corr = df_city.corr()
                cmap_city = get_city_cmap(ciudad, idx_city)

                fig, ax = plt.subplots(
                    figsize=(0.9 * len(cols_corr), 0.9 * len(cols_corr))
                )
                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap=cmap_city,
                    square=True,
                    cbar=True,
                    ax=ax,
                    linewidths=0.4,
                    linecolor="white",
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                ax.set_title("")
                st.pyplot(fig)

                # Hallazgos dinámicos basados en correlaciones con interpretación de negocio
                target_corr = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
                
                if len(target_corr) > 0:
                    hallazgos = []
                    
                    # Diccionario de interpretaciones de negocio
                    interpretaciones = {
                        # Relaciones con Superhost
                        ("y_superhost", "host_identity_verified_bin", "+"): f"En {ciudad}: Hosts verificados tienen más probabilidad de ser Superhosts - la verificación es señal de profesionalismo",
                        ("y_superhost", "high_rating", "+"): f"En {ciudad}: Alta calificación es predictor fuerte de Superhost - la calidad del servicio es clave",
                        ("y_superhost", "number_of_reviews", "+"): f"En {ciudad}: Más reseñas correlacionan con Superhost - la experiencia acumulada importa",
                        ("y_superhost", "instant_bookable_bin", "+"): f"En {ciudad}: Superhosts tienden a ofrecer reserva instantánea - facilitan las reservas",
                        ("y_superhost", "is_professional_host", "+"): f"En {ciudad}: Hosts con múltiples propiedades tienden a ser Superhosts - profesionalización del negocio",
                        ("y_superhost", "price", "+"): f"En {ciudad}: Superhosts cobran precios más altos - su calidad justifica premium",
                        ("y_superhost", "price", "-"): f"En {ciudad}: Superhosts cobran precios más bajos - compiten por volumen con calidad",
                        
                        # Relaciones con reserva instantánea
                        ("instant_bookable_bin", "host_identity_verified_bin", "+"): f"En {ciudad}: Hosts verificados ofrecen más reserva instantánea - confían en su perfil",
                        ("instant_bookable_bin", "require_guest_pic_bin", "-"): f"En {ciudad}: Reserva instantánea va con menos requisitos al huésped - facilitan acceso",
                        ("instant_bookable_bin", "is_professional_host", "+"): f"En {ciudad}: Hosts profesionales usan más reserva instantánea - estrategia de volumen",
                        
                        # Relaciones con calificación alta
                        ("high_rating", "accommodates", "+"): f"En {ciudad}: Propiedades más grandes tienen mejores calificaciones - experiencia familiar/grupal",
                        ("high_rating", "accommodates", "-"): f"En {ciudad}: Propiedades pequeñas tienen mejores calificaciones - atención personalizada",
                        ("high_rating", "price", "+"): f"En {ciudad}: Precios altos correlacionan con mejor rating - mayor calidad percibida",
                        ("high_rating", "number_of_reviews", "+"): f"En {ciudad}: Más reseñas correlacionan con alta calificación - calidad consistente",
                        
                        # Relaciones con profesionalismo
                        ("is_professional_host", "instant_bookable_bin", "+"): f"En {ciudad}: Hosts profesionales automatizan reservas - gestión eficiente",
                        ("is_professional_host", "price", "+"): f"En {ciudad}: Hosts profesionales cobran más - mejor gestión y servicios",
                        ("is_professional_host", "price", "-"): f"En {ciudad}: Hosts profesionales cobran menos - economías de escala",
                        
                        # Relaciones de precio
                        ("price", "accommodates", "+"): f"En {ciudad}: Mayor capacidad = mayor precio - propiedades más grandes cuestan más",
                        ("price", "number_of_reviews", "+"): f"En {ciudad}: Más reseñas permiten precios más altos - reputación se monetiza",
                        ("price", "number_of_reviews", "-"): f"En {ciudad}: Propiedades con menos reseñas cobran más - posicionamiento premium desde inicio",
                    }
                    
                    # Analizar las 3 correlaciones más fuertes con el target
                    for i, (var, corr_abs) in enumerate(target_corr.head(3).items()):
                        corr_valor = corr[target_col][var]
                        signo = "+" if corr_valor > 0 else "-"
                        
                        # Buscar interpretación específica
                        key = (target_col, var, signo)
                        if key in interpretaciones:
                            hallazgos.append(f"**{interpretaciones[key]}** (r={corr_valor:.2f})")
                        else:
                            # Interpretación genérica si no hay específica
                            if abs(corr_valor) >= 0.5:
                                relacion = "aumenta significativamente" if corr_valor > 0 else "disminuye significativamente"
                                hallazgos.append(f"En {ciudad}: Cuando `{var}` aumenta, `{target_col}` {relacion} (r={corr_valor:.2f})")
                            elif abs(corr_valor) >= 0.3:
                                relacion = "tiende a aumentar" if corr_valor > 0 else "tiende a disminuir"
                                hallazgos.append(f"En {ciudad}: `{var}` {relacion} con `{target_col}` (r={corr_valor:.2f})")
                    
                    # Detectar multicolinealidad con interpretación de negocio
                    predictores = [c for c in corr.columns if c != target_col]
                    if len(predictores) > 1:
                        max_corr_pred = 0
                        pair = None
                        for i, p1 in enumerate(predictores):
                            for p2 in predictores[i+1:]:
                                corr_val = abs(corr[p1][p2])
                                if corr_val > max_corr_pred:
                                    max_corr_pred = corr_val
                                    pair = (p1, p2)
                        
                        if pair and max_corr_pred >= 0.7:
                            # Interpretaciones específicas de multicolinealidad
                            multi_interp = {
                                ("high_rating", "number_of_reviews"): "propiedades con muchas reseñas mantienen alta calidad",
                                ("is_professional_host", "instant_bookable_bin"): "hosts profesionales automatizan sus operaciones",
                                ("host_identity_verified_bin", "y_superhost"): "la verificación es paso hacia Superhost",
                                ("price", "accommodates"): "propiedades más grandes cuestan más",
                            }
                            
                            pair_key = tuple(sorted(pair))
                            if pair_key in multi_interp or tuple(reversed(pair_key)) in multi_interp:
                                key = pair_key if pair_key in multi_interp else tuple(reversed(pair_key))
                                hallazgos.append(f"⚠️ En {ciudad}: `{pair[0]}` y `{pair[1]}` están correlacionados ({max_corr_pred:.2f}) - {multi_interp[key]}")
                            else:
                                hallazgos.append(f"⚠️ En {ciudad}: `{pair[0]}` y `{pair[1]}` están muy correlacionados ({max_corr_pred:.2f}) - pueden estar midiendo lo mismo")
                    
                    # Mostrar hallazgos con formato profesional
                    if hallazgos:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #FFF5F7 0%, #FFFFFF 100%); 
                                    border-left: 3px solid #FF385C; 
                                    padding: 12px 16px; 
                                    margin-top: 12px; 
                                    border-radius: 8px;
                                    box-shadow: 0 1px 3px rgba(0,0,0,0.08);'>
                            <p style='color: #222222; font-size: 12px; font-weight: 600; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 0.5px;'>
                                Insights de Correlaciones — {ciudad}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for idx, hallazgo in enumerate(hallazgos):
                            # Detectar si es alerta (⚠️) o insight normal
                            is_alert = "⚠️" in hallazgo
                            
                            # Limpiar el hallazgo
                            hallazgo_clean = hallazgo.replace("⚠️", "").replace("**", "").strip()
                            
                            # Extraer el valor de correlación si existe
                            corr_match = re.search(r'\(r=(-?\d+\.\d+)\)', hallazgo_clean)
                            if corr_match:
                                corr_val = float(corr_match.group(1))
                                hallazgo_text = hallazgo_clean.split('(r=')[0].strip()
                                corr_badge = f"<span style='background: {'#FFE8EC' if is_alert else '#E7F5EC'}; color: {'#C13515' if is_alert else '#0A7B3E'}; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 600; margin-left: 6px;'>r = {corr_val:.2f}</span>"
                            else:
                                hallazgo_text = hallazgo_clean
                                corr_badge = ""
                            
                            # Icono según tipo
                            icon = "⚠️" if is_alert else "→"
                            icon_color = "#C13515" if is_alert else "#FF385C"
                            bg_color = "#FFF5F5" if is_alert else "#FAFAFA"
                            
                            st.markdown(f"""
                            <div style='background: {bg_color}; 
                                        padding: 10px 14px; 
                                        margin: 6px 0; 
                                        border-radius: 6px;
                                        border: 1px solid {"#FFE0E0" if is_alert else "#F0F0F0"};'>
                                <p style='margin: 0; color: #484848; font-size: 13px; line-height: 1.5;'>
                                    <span style='color: {icon_color}; font-weight: 700; margin-right: 6px;'>{icon}</span>
                                    {hallazgo_text}
                                    {corr_badge}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background: #F7F7F7; 
                                    border-left: 3px solid #DDDDDD; 
                                    padding: 10px 14px; 
                                    margin-top: 12px; 
                                    border-radius: 6px;'>
                            <p style='margin: 0; color: #717171; font-size: 13px; font-style: italic;'>
                                Correlaciones débiles entre las variables seleccionadas en {ciudad}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)


#######################################ANOVA#############################################################
# ============================
#            ANOVA
# ============================
if View == "ANOVA":

    col_title, col_help = st.columns([0.95, 0.05])
    with col_title:
        st.title("ANOVA (Análisis de Varianza)")
    with col_help:
        with st.popover("➕"):
            st.markdown("**¿Qué es ANOVA?**")
            st.write("Análisis de Varianza que compara si existen diferencias significativas en los precios promedio entre diferentes grupos (ciudades, tipos de alojamiento, barrios). Te dice si las variaciones de precio observadas son estadísticamente reales o solo casualidad, ayudándote a identificar factores clave que impactan el mercado de Airbnb.")
    st.sidebar.subheader("Parámetros ANOVA")

    # -------------------------
    # Selección de ciudades
    # -------------------------
    ciudades_disp = sorted(df["ciudad"].dropna().unique().tolist())
    ciudades_sel = st.sidebar.multiselect(
        "Ciudades incluidas en el análisis",
        options=ciudades_disp,
        default=ciudades_disp
    )

    df_anova = df[df["ciudad"].isin(ciudades_sel)].copy()

    # -------------------------
    # Variable dependiente
    # -------------------------
    num_candidatas = [
        c for c in df_anova.select_dtypes(include=["number"]).columns
        if c not in ["latitude", "longitude", "id"]
    ]

    if "price" in num_candidatas:
        num_candidatas = ["price"] + [c for c in num_candidatas if c != "price"]

    y_var = st.sidebar.selectbox("Variable dependiente", num_candidatas)

    # -------------------------
    # Factores categóricos
    # -------------------------
    factores_candidatos = []
    for c in Lista:
        if c in df_anova.columns:
            n_cat = df_anova[c].nunique(dropna=True)
            if 2 <= n_cat <= 30:
                factores_candidatos.append(c)

    # Tabs
    tab1, tab2 = st.tabs(["ANOVA de 1 Factor", "ANOVA de 2 Factores"])

    # =====================================================
    #                ANOVA DE 1 FACTOR
    # =====================================================
    with tab1:

        factor_1 = st.sidebar.selectbox(
            "Factor categórico",
            factores_candidatos,
            key="anova1_factor"
        )

        st.header("ANOVA de 1 Factor")

        datos = df_anova[[y_var, factor_1, "ciudad"]].dropna().copy()
        datos[factor_1] = datos[factor_1].astype("category")

        st.subheader("Hipótesis")
        st.markdown(
            f"""
            **H₀:** Las medias de `{y_var}` son iguales entre los niveles de `{factor_1}`.  
            **H₁:** Al menos un nivel tiene una media diferente.
            """
        )

        # -------------------------
        # BOXPLOTS POR CIUDAD
        # -------------------------
        st.subheader("Distribución por ciudad (Boxplots)")

        ciudades_graph = datos["ciudad"].unique().tolist()
        columnas = st.slider("Columnas", 2, 4, 3, key="cols1")

        for i in range(0, len(ciudades_graph), columnas):
            row_cities = ciudades_graph[i:i + columnas]
            cols = st.columns(len(row_cities))

            for col, city in zip(cols, row_cities):
                with col:
                    st.markdown(f"### {city}")

                    df_city = datos[datos["ciudad"] == city]

                    fig = px.box(
                        df_city,
                        x=factor_1,
                        y=y_var,
                        points="suspectedoutliers",
                        title=f"{city}: {y_var} vs {factor_1}"
                    )
                    fig.update_layout(height=350)

                    st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # TABLA ANOVA 1F
        # -------------------------
        st.subheader("Tabla ANOVA 1 factor")

        resultados_1f = []
        for city in ciudades_graph:
            df_city = datos[datos["ciudad"] == city]

            modelo = smf.ols(f"{y_var} ~ C({factor_1})", data=df_city).fit()
            tabla = anova_lm(modelo, typ=3)
            
            # Filtrar filas no deseadas
            tabla = tabla[~tabla.index.isin(["Residual", "Intercept"])]

            row = tabla.loc[f"C({factor_1})"]

            resultados_1f.append({
                "Ciudad": city,
                "F": row["F"],
                "p-value": row["PR(>F)"]
            })

        tabla_1f = pd.DataFrame(resultados_1f)
        # Formatear con notación científica para p-values muy pequeños
        tabla_1f_display = tabla_1f.copy()
        tabla_1f_display['p-value'] = tabla_1f_display['p-value'].apply(
            lambda x: "< 1e-100" if x == 0.0 else (f"{x:.4e}" if x < 0.0001 else f"{x:.8f}")
        )
        tabla_1f_display['F'] = tabla_1f_display['F'].apply(lambda x: f"{x:.4f}")
        st.dataframe(tabla_1f_display, use_container_width=True)

        # -------------------------
        # INTERPRETACIÓN 1F
        # -------------------------
        with st.expander("Interpretación de Resultados", expanded=False):
            for idx, (_, row) in enumerate(tabla_1f.iterrows()):
                city = row["Ciudad"]
                f_val = row["F"]
                pval = row["p-value"]

                p_str = f"{pval:.4e}" if pval < 0.0001 else f"{pval:.8f}"
                f_str = f"{f_val:.4f}"
                
                if pval < 0.05:
                    st.markdown(f"""<div style='background-color: #FFE8E8; padding: 15px; border-radius: 5px; border-left: 4px solid #FF5A5F; margin-bottom: 10px;'>
<strong style='color: #FF5A5F; font-size: 16px;'>{city}</strong><br>
<strong>F = {f_str}, p = {p_str}</strong><br>
<strong>Se rechaza H₀:</strong> El factor '{factor_1}' tiene un efecto significativo sobre '{y_var}'. Las diferencias entre grupos NO son debidas al azar.
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style='background-color: #F0F8F7; padding: 15px; border-radius: 5px; border-left: 4px solid #00A699; margin-bottom: 10px;'>
<strong style='color: #00A699; font-size: 16px;'>{city}</strong><br>
<strong>F = {f_str}, p = {p_str}</strong><br>
<strong>No se rechaza H₀:</strong> El factor '{factor_1}' NO tiene efecto significativo sobre '{y_var}'. Las diferencias observadas pueden ser debidas al azar.
</div>""", unsafe_allow_html=True)



    # =====================================================
    #                ANOVA DE 2 FACTORES
    # =====================================================
    with tab2:

        colx, coly = st.sidebar.columns(2)
        with colx:
            factor_A = st.selectbox("Factor A", factores_candidatos, key="A")
        with coly:
            factor_B = st.selectbox("Factor B", [f for f in factores_candidatos if f != factor_A], key="B")

        incluir_interacción = st.sidebar.checkbox("Incluir interacción A×B", True)

        st.header("ANOVA de 2 Factores")

        datos = df_anova[[y_var, factor_A, factor_B, "ciudad"]].dropna().copy()
        datos[factor_A] = datos[factor_A].astype("category")
        datos[factor_B] = datos[factor_B].astype("category")

        # -------------------------
        # BOXPLOTS
        # -------------------------
        st.subheader("Distribución por ciudad (Boxplots)")

        ciudades_graph = datos["ciudad"].unique().tolist()
        columnas = st.slider("Columnas", 2, 4, 3, key="cols2")

        for i in range(0, len(ciudades_graph), columnas):
            row_cities = ciudades_graph[i:i + columnas]
            cols = st.columns(len(row_cities))

            for col, city in zip(cols, row_cities):

                df_city = datos[datos["ciudad"] == city]
                niveles = sorted(df_city[factor_B].unique())

                color_map = {
                    niveles[0]: "#EF553B",   # izquierda (rojo)
                    niveles[1]: "#00A699"    # derecha (verde)
                }

                with col:
                    st.markdown(f"### {city}")

                    fig = px.box(
                        df_city,
                        x=factor_A,
                        y=y_var,
                        color=factor_B,
                        color_discrete_map=color_map,
                        points="suspectedoutliers"
                    )
                    fig.update_layout(height=350)

                    st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # GRÁFICO DE INTERACCIÓN
        # -------------------------
        st.subheader("Gráfico de interacción")

        medias = datos.groupby([factor_A, factor_B])[y_var].mean().reset_index()
        niveles = sorted(datos[factor_B].unique())

        color_map_global = {
            niveles[0]: "#EF553B",
            niveles[1]: "#00A699"
        }

        fig_inter = px.line(
            medias,
            x=factor_A,
            y=y_var,
            color=factor_B,
            markers=True,
            title=f"Interacción entre {factor_A} y {factor_B}",
            color_discrete_map=color_map_global
        )

        fig_inter.update_layout(height=400)
        st.plotly_chart(fig_inter, use_container_width=True)

        # -------------------------
        # TABLA ANOVA 2F
        # -------------------------
        st.subheader("Tabla ANOVA 2 factores")

        resultados_2f = []

        for city in ciudades_graph:
            df_city = datos[datos["ciudad"] == city]

            formula = f"{y_var} ~ C({factor_A}) * C({factor_B})" if incluir_interacción \
                      else f"{y_var} ~ C({factor_A}) + C({factor_B})"

            modelo = smf.ols(formula, data=df_city).fit()
            tabla = anova_lm(modelo, typ=3)
            
            # Filtrar filas no deseadas
            tabla = tabla[~tabla.index.isin(["Residual", "Intercept"])]

            for idx in tabla.index:
                resultados_2f.append({
                    "Ciudad": city,
                    "Efecto": idx,
                    "F": tabla.loc[idx]["F"],
                    "p-value": tabla.loc[idx]["PR(>F)"]
                })

        tabla_2f = pd.DataFrame(resultados_2f)
        # Formatear con notación científica para p-values muy pequeños
        tabla_2f_display = tabla_2f.copy()
        tabla_2f_display['p-value'] = tabla_2f_display['p-value'].apply(
            lambda x: "< 1e-100" if x == 0.0 else (f"{x:.4e}" if x < 0.0001 else f"{x:.8f}")
        )
        tabla_2f_display['F'] = tabla_2f_display['F'].apply(lambda x: f"{x:.4f}")
        st.dataframe(tabla_2f_display, use_container_width=True)

        # -------------------------
        # INTERPRETACIÓN 2F
        # -------------------------
        with st.expander("Interpretación de Resultados", expanded=False):
            for idx, (_, row) in enumerate(tabla_2f.iterrows()):
                city = row["Ciudad"]
                efecto = row["Efecto"]
                f_val = row["F"]
                pval = row["p-value"]

                p_str = f"{pval:.4e}" if pval < 0.0001 else f"{pval:.8f}"
                f_str = f"{f_val:.4f}"
                
                # Determinar tipo de efecto
                if ":" in efecto:
                    tipo_efecto = "interacción"
                    explicacion = "Los factores se influencian mutuamente"
                else:
                    tipo_efecto = "efecto principal"
                    explicacion = "Este factor afecta la variable dependiente"
                
                if pval < 0.05:
                    st.markdown(f"""<div style='background-color: #FFE8E8; padding: 15px; border-radius: 5px; border-left: 4px solid #FF5A5F; margin-bottom: 10px;'>
<strong style='color: #FF5A5F; font-size: 16px;'>{city} — {efecto}</strong><br>
<strong>F = {f_str}, p = {p_str}</strong><br>
<strong>Se rechaza H₀:</strong> {tipo_efecto.capitalize()} significativo. {explicacion}.
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style='background-color: #F0F8F7; padding: 15px; border-radius: 5px; border-left: 4px solid #00A699; margin-bottom: 10px;'>
<strong style='color: #00A699; font-size: 16px;'>{city} — {efecto}</strong><br>
<strong>F = {f_str}, p = {p_str}</strong><br>
<strong>No se rechaza H₀:</strong> {tipo_efecto.capitalize()} NO significativo. Las diferencias pueden ser debidas al azar.
</div>""", unsafe_allow_html=True)
