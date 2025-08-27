
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import io
from sqlalchemy import create_engine, text
import datetime

# Prophet/cmdstanpy logging seviyesini düşür (terminal spam'ini engelle)
import logging
try:
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    logging.getLogger("prophet").setLevel(logging.CRITICAL)
except Exception:
    pass

# ---- CSV fallback (yerel test için) ----
CSV_CFG = getattr(st, "secrets", {}).get("csv", {}) if hasattr(st, "secrets") else {}
CSV_DIR = CSV_CFG.get("dir", ".")  # default: current folder
CSV_ENABLE = bool(CSV_CFG.get("enable", False))

@st.cache_data(show_spinner=False)
def _csv_path(name: str) -> str:
    import os
    # beklenen dosya adları: woshit.csv, woshitattributes.csv, wosauthor.csv, yoksisbirim.csv, wosaddress.csv, cuauthor.csv, cuauthorrid.csv
    return os.path.join(CSV_DIR, name)

@st.cache_data(show_spinner=False)
def _csv_exists(name: str) -> bool:
    import os
    return False if not CSV_ENABLE else os.path.exists(_csv_path(name))

@st.cache_data(show_spinner=False)
def _csv_load(name: str) -> pd.DataFrame:
    try:
        p = _csv_path(name)
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

# Opsiyonel: Kümeleme için scikit-learn
try:
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    SKLEARN_OK = True
except Exception:
    KMeans = None  # type: ignore
    StandardScaler = None  # type: ignore
    SKLEARN_OK = False

# Prophet opsiyonel (varsa kullanılacak) — sembol import etmeyelim ki linter şikayet etmesin
try:
    import importlib
    importlib.import_module("prophet")
    PROPHET_OK = True
except Exception:
    PROPHET_OK = False

st.set_page_config(page_title="Üniversite Analizi (DB + CSV)", layout="wide")
st.title("🎓 Üniversite Analizi — SQL Server / CSV Hibrit")

# Tek tıkta çalış modu: yıllık seri üretimi için UI'yı gizle ve otomatik çalıştır
ONE_CLICK = False

# Geliştirici modu: son kullanıcıdan gizli diagnostik panelleri göstermeyin
DEBUG_MODE = bool(getattr(st, "secrets", {}).get("debug_mode", False))

# ----------------- Yardımcılar -----------------


@st.cache_data(show_spinner=False)
def read_csv(uploaded, sep=",", decimal="."):
    return pd.read_csv(uploaded, sep=sep, decimal=decimal)

# --- Yıl aralığı keşif yardımcıları ---


@st.cache_data(show_spinner=False)
def db_year_bounds() -> tuple[int, int]:
    """Veritabanından mevcut en eski ve en yeni yılı getirir (WOSHit.SourcePublishYear)."""
    cur_year = datetime.date.today().year
    try:
        df = sql_to_df(
            "SELECT MIN(SourcePublishYear) AS MinY, MAX(SourcePublishYear) AS MaxY "
            "FROM dbo.WOSHit WHERE SourcePublishYear IS NOT NULL"
        )
        miny = int(df["MinY"].iloc[0]) if not df.empty and pd.notna(df["MinY"].iloc[0]) else cur_year
        maxy = int(df["MaxY"].iloc[0]) if not df.empty and pd.notna(df["MaxY"].iloc[0]) else cur_year
    except Exception:
        miny, maxy = cur_year, cur_year
    maxy = min(maxy, cur_year)
    return miny, maxy

def csv_year_bounds(df: pd.DataFrame | None) -> tuple[int, int]:
    """CSV içindeki yıl kolonunu bulup min/max döndürür."""
    cur_year=datetime.date.today().year
    if df is None or df.empty:
        return cur_year, cur_year
    candidates=["Yil", "YIL", "yil", "year", "Year",
        "SOURCEPUBLISHYEAR", "SourcePublishYear", "sourcepublishyear"]
    year_col=None
    for c in df.columns:
        if str(c) in candidates:
            year_col=c
            break
    if year_col is None:
        # normalize etmeyi dene
        df_tmp=normalize_columns(df.copy(), "alan_yillik")
        if "Yil" in df_tmp.columns:
            year_col="Yil"
    if year_col is None:
        return cur_year, cur_year
    s=pd.to_numeric(df[year_col], errors="coerce").dropna()
    if s.empty:
        return cur_year, cur_year
    miny, maxy=int(s.min()), int(s.max())
    maxy=min(maxy, cur_year)
    return miny, maxy

# --- Atıf kolonu var mı kontrolü ve otomatik seçim ---
@st.cache_data(show_spinner=False)
def db_has_column(schema: str, table: str, column: str) -> bool:
    try:
        q=(
            "SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA='{_q(schema)}' AND TABLE_NAME='{_q(table)}' "
            f"AND COLUMN_NAME='{_q(column)}'"
        )
        df=sql_to_df(q, tag=f"colchk:{schema}.{table}.{column}")
        return not df.empty
    except Exception:
        return False

# --- Table existence helper ---
@st.cache_data(show_spinner=False)
def db_has_table(schema: str, table: str) -> bool:
    try:
        q = (
            "SELECT 1 FROM INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA='{_q(schema)}' AND TABLE_NAME='{_q(table)}'"
        )
        df = sql_to_df(q, tag=f"tblchk:{schema}.{table}")
        return not df.empty
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def detect_citation_column() -> str | None:
    # Common possibilities we saw in different dumps
    for cand in ("TimesCited", "CitiationCount", "times_cited"):
        if db_has_column("dbo", "WOSHit", cand):
            return cand
    return None

# --- Auto-detection helper for CUAUTHOR columns ---
@st.cache_data(show_spinner=False)
def detect_cuauthor_mapping() -> tuple[str | None, str | None]:
    """Try to detect CUAUTHOR table and its Hit/Author id column names."""
    schema = ORG2_SCHEMA
    tbl = CUAUTHOR_TBL
    if not db_has_table(schema, tbl):
        return None, None
    # Common candidates
    hit_cands = ["HitId", "hit_id", "WosHitId", "HITID"]
    auth_cands = ["AuthorId", "author_id", "AUTHORID"]
    found_hit = None
    found_auth = None
    for c in hit_cands:
        if db_has_column(schema, tbl, c):
            found_hit = c
            break
    for c in auth_cands:
        if db_has_column(schema, tbl, c):
            found_auth = c
            break
    return found_hit, found_auth

def build_engine():
    """SQLAlchemy engine oluşturur (pyodbc + secrets.toml).
    - .streamlit/secrets.toml içinde [db] altında opsiyonel:
        driver_path = "/opt/homebrew/lib/libmsodbcsql.18.dylib"
      varsa önce onu kullanır.
    - Yoksa 'driver' (örn: "ODBC Driver 18 for SQL Server") adını kullanır.
    - Bağlantı kurulamazsa ve 18 deneniyorsa otomatik 17'ye düşer.
    """
    # --- Secrets doğrulama (kullanıcıya net hata mesajı) ---
    missing_msg = None
    try:
        s = st.secrets["db"]  # type: ignore[index]
    except Exception:
        missing_msg = "`.streamlit/secrets.toml` içinde `[db]` bölümü bulunamadı."
        s = {}
    required = ["server", "username", "password", "database"]
    missing = [k for k in required if not str(s.get(k, "")).strip()]
    if missing_msg or missing:
        try:
            st.error(
                "🔐 Veritabanı ayarları eksik: " +
                (missing_msg or "") +
                (" Eksik alanlar: " + ", ".join(missing) if missing else "")
            )
        except Exception:
            pass
        raise RuntimeError("Eksik veritabanı secrets ayarları")

    server = s["server"]
    user = s["username"]
    pwd = s["password"]
    db = s["database"]
    driver = s.get("driver", "ODBC Driver 18 for SQL Server")
    driver_path = s.get("driver_path")  # opsiyonel, tam yol
    encrypt = s.get("encrypt", False)
    trust_cert = s.get("trust_server_certificate", True)

    # DRIVER token'ını oluştur (tam yol öncelikli)
    if driver_path:
        driver_token=f"DRIVER={{{driver_path}}};"
    else:
        driver_token=f"DRIVER={{{driver}}};"

    base_conn=(
        f"{driver_token}"
        f"SERVER={server};"
        f"DATABASE={db};"
        f"UID={user};PWD={pwd};"
        f"Encrypt={'yes' if bool(encrypt) else 'no'};"
        f"TrustServerCertificate={'yes' if bool(trust_cert) else 'no'};"
    )

    # Önce birincil ayarla dene
    params=quote_plus(base_conn)
    try:
        engine = create_engine(
            f"mssql+pyodbc:///?odbc_connect={params}",
            pool_pre_ping=True, pool_recycle=1800, pool_size=5, max_overflow=10
        )
        # Bağlantı sağlığı: basit SELECT 1
        try:
            from sqlalchemy import text as _sa_text  # local alias
            with engine.connect() as conn:
                conn.execute(_sa_text("SELECT 1"))
        except Exception as ex:
            try:
                st.error(f"🔌 Veritabanına bağlanılamadı (PRIMARY): {ex}")
            except Exception:
                pass
            raise
        return engine
    except Exception:
        # Eğer tam yol kullanılmadıysa ve 18 deneniyorsa 17'ye fallback dene
        if (not driver_path) and ("18" in driver):
            fallback_driver="ODBC Driver 17 for SQL Server"
            fallback_token=f"DRIVER={{{fallback_driver}}};"
            conn_fallback=base_conn.replace(driver_token, fallback_token)
            params_fb=quote_plus(conn_fallback)
            engine = create_engine(
                f"mssql+pyodbc:///?odbc_connect={params_fb}",
                pool_pre_ping=True, pool_recycle=1800, pool_size=5, max_overflow=10
            )
            try:
                from sqlalchemy import text as _sa_text  # local alias
                with engine.connect() as conn:
                    conn.execute(_sa_text("SELECT 1"))
            except Exception as ex:
                try:
                    st.error(f"🔌 Veritabanına bağlanılamadı (FALLBACK): {ex}")
                except Exception:
                    pass
                raise
            return engine
        raise

@st.cache_data(show_spinner=True)
def sql_to_df(sql: str, tag: str = "", params: dict | None = None) -> pd.DataFrame:
    """Run SQL and return DataFrame. tag is an extra cache key to bust cache
    when parameters (e.g., year_min/year_max) change.
    """
    engine=build_engine()
    return pd.read_sql(text(sql), engine, params=params)

# Helper to ensure year range changes invalidate cached DB calls used by UI
# (so Alan dağılımı / Yazar tabları gerçekten yıl aralığına göre yenilenir)


# Build a cache-busting tag that includes year range and current org selections
def _cache_tag_from_state() -> str:
    try:
        y1, y2 = int(year_min), int(year_max)
    except Exception:
        y1, y2 = 0, 0
    def _norm(v):
        if v is None:
            return "-"
        s = str(v)
        # Avoid exploding cache key with long strings
        s = s.strip()
        s = s.replace("|", "/")[:80]
        return s
    uni = _norm(st.session_state.get("y_uni"))
    fac = _norm(st.session_state.get("y_fac"))
    dep = _norm(st.session_state.get("y_dep"))
    sub = _norm(st.session_state.get("y_sub"))
    return f"{y1}-{y2}|{uni}|{fac}|{dep}|{sub}"

def run_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    try:
        tag = _cache_tag_from_state()
    except Exception:
        tag = ""
    return sql_to_df(sql, tag, params=params)

def plot_barh(labels, values, title, xlabel):
    fig, ax=plt.subplots(figsize=(10, 6))
    ax.barh(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig

def make_chart(df, x_col, y_col, kind, title, xlabel=None, ylabel=None):
    fig, ax=plt.subplots(figsize=(10, 6))
    if kind == "Yatay çubuk":
        ax.barh(df[x_col], df[y_col])
    elif kind == "Dikey çubuk":
        ax.bar(df[x_col], df[y_col])
    elif kind == "Pasta":
        ax.pie(df[y_col], labels=df[x_col], autopct="%1.1f%%")
    elif kind == "Çizgi":
        ax.plot(df[x_col], df[y_col], marker="o")
    elif kind == "Alan":
        ax.plot(df[x_col], df[y_col], marker="o")
        ax.fill_between(df[x_col], df[y_col], alpha=0.2)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig

# Kolon adlarını normalize etme
COLMAPS={
    "alan_yillik": {
        "Alan": {"alan", "field", "category", "subject", "value"},
        "Yil": {"yil", "yıl", "year", "sourcepublishyear"},
        "YayinSayisi": {"yayin", "count", "n", "tot", "toplam", "yayinsayisi"},
    },
    "yazar_yayin": {
        "Yazar": {"yazar", "author", "wosstandard"},
        "YayinSayisi": {"yayin", "count", "n", "tot", "yayinsayisi"},
    },
    "yazar_atif": {
        "Yazar": {"yazar", "author", "wosstandard"},
        "ToplamAtif": {"timescited", "citiationcount", "citations", "toplamatif"},
    },
    "alan_dagilim": {
        "Alan": {"alan", "field", "category", "subject", "value"},
        "YayinSayisi": {"yayin", "count", "n", "tot", "yayinsayisi"},
    },
}

def normalize_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    maps=COLMAPS.get(kind, {})
    if not maps or df is None or df.empty:
        return df
    lower={c.lower(): c for c in df.columns}
    rename={}
    for target, candidates in maps.items():
        for cand in candidates:
            if cand in lower:
                rename[lower[cand]]=target
                break
    if rename:
        df=df.rename(columns=rename)
    return df

def simple_lm_forecast(years, y, target_year: int) -> float:
    """Basit doğrusal regresyon (numpy polyfit). years ve y NumPy array veya liste olabilir."""
    years=np.asarray(years, dtype=float)
    y=np.asarray(y, dtype=float)
    coef=np.polyfit(years, y, 1)  # type: ignore[call-overload]
    slope=float(coef[0])
    intercept=float(coef[1])
    y_pred=slope * float(target_year) + intercept
    return float(max(0.0, y_pred))

# --- Alan adlarını tutarlılaştırma ve güven aralığı hesap yardımcıları ---

def _canon_area_key(s: str) -> str:
    """Alan etiketlerini daha tutarlı hale getirmek için sade bir anahtar üretir."""
    if not isinstance(s, str):
        return ""
    t=_strip_tr(s).lower()
    # & ve 'and' eşitle
    t=t.replace("&", " and ")
    t=t.replace("/", " ")
    t=" ".join(t.split())
    # çok temel eş anlamlı kümeler
    SYN={
        "science and technology": "science-technology",
        "technology": "science-technology",
        "engineering": "engineering",
        "computer science": "computer-science",
        "computer sciences": "computer-science",
        "life sciences": "life-sciences",
        "life sciences & biomedicine": "life-sciences",
        "life sciences and biomedicine": "life-sciences",
        "physical sciences": "physical-sciences",
    }
    return SYN.get(t, t)

def _most_frequent(iterable):
    from collections import Counter
    c=Counter([x for x in iterable if pd.notna(x)])
    return c.most_common(1)[0][0] if c else None

def _blend_forecast(y_lm: float | None, y_pr: float | None) -> float:
    """İki model varsa ortalamasını; tek model varsa onu döndür."""
    vals=[v for v in [y_lm, y_pr] if v is not None and pd.notna(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))

def _volatility_ci(y: np.ndarray, point: float) -> tuple[float, float]:
    """
    Yıllık serinin yüzde değişimlerinin standart sapmasından (σ) basit bir CI üret.
    CI ~ point * (1 ± σ). Çok küçük veride σ=0 kabul edilir.
    """
    if y is None or len(y) < 2:
        return max(0.0, float(point)), max(0.0, float(point))
    y=np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct=np.diff(y) / np.where(y[:-1] == 0, np.nan, y[:-1])
    sigma=float(np.nanstd(pct))
    low=max(0.0, float(point) * (1.0 - sigma))
    high=max(0.0, float(point) * (1.0 + sigma))
    return low, high

# ---- Optional Prophet forecaster (isolated to avoid type-checker noise) ----
from typing import Optional

def _prophet_forecast_safe(years: pd.Series | np.ndarray | list, values: pd.Series | np.ndarray | list, target_year: int) -> Optional[float]:
    """Return Prophet forecast for target_year or None if Prophet unavailable or fails.
    This function imports Prophet locally and keeps types loose to avoid Pylance complaints.
    """
    if not PROPHET_OK:
        return None
    try:
        # Local import keeps global type as unknown for static checker
        from prophet import Prophet as _Prophet  # type: ignore
        import pandas as _pd  # local alias to avoid name shadowing
        years_arr = np.asarray(years)
        vals_arr = np.asarray(values, dtype=float)
        if len(years_arr) < 3:
            return None
        dfp = _pd.DataFrame({"ds": _pd.to_datetime(years_arr, format="%Y"), "y": vals_arr})
        m = _Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)  # type: ignore[arg-type]
        m.fit(dfp)
        fut = _pd.DataFrame({"ds": [_pd.to_datetime(f"{int(target_year)}-01-01")]})
        yhat = float(m.predict(fut)["yhat"].iloc[0])
        return yhat
    except Exception:
        return None

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("Veri Kaynağı")
    st.caption("Bu uygulama veriyi doğrudan **SQL Server**’dan çeker.")
    source="SQL Server"

    st.divider()
    st.header("Parametreler")

    # Prophet'i isteğe bağlı yap
    _use_prophet = st.checkbox("Prophet'i kullan", value=False,
                                help="Seçiliyse Prophet modeli çalıştırılır. Seçilmezse yalnızca doğrusal model (LM) çalışır.")
    st.session_state["use_prophet"] = bool(_use_prophet)
    if not _use_prophet:
        PROPHET_OK = False  # Prophet import edilmiş olsa bile hesaplamayı devre dışı bırak

    # Mevcut veri aralığını otomatik keşfet (her zaman SQL Server)
    cur_year=datetime.date.today().year
    min_bound, max_bound=db_year_bounds()

    # Kullanıcı arayüzü: sınırlar veri aralığına sabitlenir
    st.markdown(
        "<small style='color: #666;'>Verilerin hangi yıldan başlayarak gösterileceğini seçin. En eski yıl, elimizdeki ilk veri yılıdır.</small>",
        unsafe_allow_html=True
    )
    # Varsayılan başlangıç: 2019 (eğer aralık dışında kalıyorsa sınırlar içinde tutulur)
    default_start=int(min(max(2019, int(min_bound)), int(max_bound)))
    year_min=st.number_input(
        "Başlangıç yılı",
        int(min_bound),
        int(max_bound),
        default_start,
        1
    )
    # En eski yıl bilgisini kullanıcıya not olarak göster
    st.caption(
        f"📌 Bu veritabanında görülen en eski yıl: **{int(min_bound)}**. Daha geriye gidilemez; istersen {int(min_bound)}'e kadar çekebilirsin.")
    st.markdown(
        "<small style='color: #666;'>Verilerin hangi yıla kadar gösterileceğini seçin. Gelecekteki yıllar seçilemez.</small>",
        unsafe_allow_html=True
    )
    year_max=st.number_input("Bitiş yılı", int(
        min_bound), int(max_bound), int(max_bound), 1)

    # --- Yıl aralığı değiştiyse cache'i temizle ve çıktıları güncelle ---
    prev_key_min="_prev_year_min"
    prev_key_max="_prev_year_max"
    prev_min=st.session_state.get(prev_key_min)
    prev_max=st.session_state.get(prev_key_max)
    if prev_min is None or prev_max is None:
        st.session_state[prev_key_min]=int(year_min)
        st.session_state[prev_key_max]=int(year_max)
    else:
        if int(year_min) != int(prev_min) or int(year_max) != int(prev_max):
            st.session_state[prev_key_min]=int(year_min)
            st.session_state[prev_key_max]=int(year_max)
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.session_state["ran"]=True

    # Kurumsal seçimler değişince de önbelleği temizleyelim (dropdown anahtarları)
    for key in ("y_uni", "y_fac", "y_dep", "y_sub"):
        if key in st.session_state:
            # Her yeniden çizimde bu değerler değiştiyse cache'i sıfırlamak güvenli
            st.session_state.setdefault(f"_prev_{key}", st.session_state[key])
            if st.session_state[f"_prev_{key}"] != st.session_state[key]:
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                st.session_state[f"_prev_{key}"]=st.session_state[key]
                st.session_state["ran"]=True

    cur_year = datetime.date.today().year
    # Kullanıcı geçmiş yıllar için de tahmin yapabilsin: alt sınır veri kümesindeki en eski yıl
    min_ty = int(min_bound)
    max_ty = int(cur_year + 10)  # en fazla bugünden +10 yıl
    # Varsayılan: bugünün yılından bir sonraki yıl (aralık dışına taşarsa sınırlar içinde tut)
    default_ty = int(cur_year + 1)
    if default_ty < min_ty:
        default_ty = min_ty
    if default_ty > max_ty:
        default_ty = max_ty
    target_year = st.number_input(
        "Tahmin yılı",
        min_ty,
        max_ty,
        default_ty,
        1
    )
    st.caption(
        "📅 Geçmiş veya gelecek bir yıl seçebilirsiniz. Varsayılan olarak bugünün yılından bir sonraki yıl gelir. Bu bir öngörüdür; kesin sonuç değildir."
    )

    # Model seçimi kaldırıldı: her iki model de otomatik çalışır
    st.caption("Tahminler iki modelle hesaplanır: **Doğrusal (LM)** ve (varsa) **Prophet**. En düşük MAE ‘önerilen’ olarak işaretlenir.")

    top_n=st.slider(
        "Otomatik seçim ve grafikler için alan sayısı", 5, 20, 10, 1)
    run_btn=st.button("🚀 Çalıştır")

    # Run / Sıfırla durumu
    reset_btn=st.button("🧹 Sıfırla")
    if reset_btn:
        st.session_state["ran"]=False
    if run_btn:
        st.session_state["ran"]=True
    ran=st.session_state.get("ran", False)

    if not ONE_CLICK:
        st.caption("🔧 Takıldıysa: yıllık seriyi zorla üret")
        if st.button("🧩 Yıllık seriyi yeniden oluştur", key="force_series_sidebar"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.session_state["ran"]=True
            st.rerun()

    # ONE_CLICK: ilk açılışta otomatik çalıştır
    if ONE_CLICK and not ran:
        st.session_state["ran"]=True
        ran=True

# --- Konu/Alan attribute adı veritabanına göre değişebiliyor. Esnek çözüm:
# Varsayılan (en yaygın) konu/alan attribute adları — fallback
SUBJECT_ATTRS_DEFAULT=[
    "category_info.subject",
    "category_info.subheading",
    "category_info.heading",
]
SUBJECT_CANDIDATES=[
    "category_info.subject",
    "category_info.subheading",
    "category_info.heading",
    "category_info.enhanced-subject",      # bazı şemalarda sonu 's' olmadan
    "category_info.enhanced-subjects",
]

@st.cache_data(show_spinner=False)
def resolve_subject_attrs(y1: int, y2: int) -> list[str]:
    """
    Veritabanında gerçekten veri döndüren konu/alan attribute adlarını bul.
    Önce seçili yıl aralığı için dener; hiçbiri çıkmazsa yıl filtresi olmadan dener;
    yine yoksa varsayılan listeye düşer.
    """
    working: list[str]=[]

    # 1) Seçili yıl aralığı için hızlı yoklama
    for cand in SUBJECT_CANDIDATES:
        try:
            q = (
                """
                SELECT TOP 1 1 AS ok
                FROM dbo.WosHitAttributes wa
                JOIN dbo.WOSHit wh ON wh.HitId = wa.HitId
                WHERE wa.Name = '""" + _q(cand) + """'
                  AND wh.SourcePublishYear BETWEEN :y1 AND :y2
                """
            )
            df = run_sql(q, params={"y1": int(y1), "y2": int(y2)})
            if not df.empty:
                working.append(cand)
        except Exception:
            pass
    if working:
        return working

    # 2) Yıl filtresi olmadan dene (daha eski kayıtlar için)
    for cand in SUBJECT_CANDIDATES:
        try:
            q=f"SELECT TOP 1 1 AS ok FROM dbo.WosHitAttributes WHERE Name = '{_q(cand)}'"
            df=run_sql(q)
            if not df.empty:
                working.append(cand)
        except Exception:
            pass
    if working:
        return working

    # 3) Son çare: en yaygın öntanımlı adlar
    return SUBJECT_ATTRS_DEFAULT

# Basit SQL tırnak kaçışı (tek tırnakları iki tek tırnak yapar)

def _q(v: str) -> str:
    return v.replace("'", "''") if isinstance(v, str) else v

def _in_list_sql(strs: list[str]) -> str:
    return ", ".join(f"'{_q(s)}'" for s in strs)

# ---- Kurumsal filtreleme için yardımcılar (varsayılan attribute adları) ----
ORG_CFG=st.secrets.get("org", {}) if hasattr(st, "secrets") else {}
UNI_ATTR=ORG_CFG.get("uni_attr", "affiliation.organization")
FAC_ATTR=ORG_CFG.get("faculty_attr", "affiliation.suborganization")
DEPT_ATTR=ORG_CFG.get("department_attr", "affiliation.suborganization_2")
SUB_ATTR=ORG_CFG.get("subunit_attr", "affiliation.suborganization_3")

_DEF_ALL="— Tümü —"

# ---- YOKSIS / CUAUTHOR tabanlı kurumsal filtreleme yardımcıları ----
ORG2_CFG=st.secrets.get("org2", {}) if hasattr(st, "secrets") else {}
ORG2_SCHEMA=ORG2_CFG.get("schema", "dbo")

def _qual(tbl: str) -> str:
    """Şema adını ekle (örn. dbo.tablo). Zaten noktalıysa dokunma."""
    return tbl if "." in tbl else f"{ORG2_SCHEMA}.{tbl}"

YOKSIS_TBL=ORG2_CFG.get("yoksis_table", "yoksisbirim")
Y_ID=ORG2_CFG.get("id_col", "BirimID")
Y_PARENT=ORG2_CFG.get("parent_col", "UstBirimID")
Y_NAME=ORG2_CFG.get("name_col", "BirimAdi")
Y_LEVEL=ORG2_CFG.get("level_col", "Duzey")

Y_L_UNI=int(ORG2_CFG.get("level_uni", 1))
Y_L_FAC=int(ORG2_CFG.get("level_fac", 2))
Y_L_DEP=int(ORG2_CFG.get("level_dept", 3))
Y_L_SUB=int(ORG2_CFG.get("level_sub", 4))

# Varsayılan birleşik düzey kümeleri (yaygın kullanım)
UNI_LEVELS_DEF=[Y_L_UNI]
FAC_LEVELS_DEF=list(dict.fromkeys([Y_L_FAC, 9]))
DEPT_LEVELS_DEF=list(dict.fromkeys([Y_L_DEP, 10, 13]))
SUB_LEVELS_DEF=list(dict.fromkeys([Y_L_SUB, 10, 13]))

# secrets.toml içinde [org2] altında fac_levels / dept_levels / sub_levels / uni_levels listeleri tanımlanabilir

def _levels_from_secret(key: str, default: list[int]) -> list[int]:
    vals=ORG2_CFG.get(key, default)
    try:
        out=[int(x) for x in vals] if isinstance(
            vals, (list, tuple)) else default
    except Exception:
        out=default
    # benzersiz ve sıralı tut
    return list(dict.fromkeys(out))

UNI_LEVELS=_levels_from_secret("uni_levels", UNI_LEVELS_DEF)
FAC_LEVELS=_levels_from_secret("fac_levels", FAC_LEVELS_DEF)
DEPT_LEVELS=_levels_from_secret("dept_levels", DEPT_LEVELS_DEF)
SUB_LEVELS=_levels_from_secret("sub_levels", SUB_LEVELS_DEF)


CUAUTHOR_TBL=ORG2_CFG.get("cuauthor_table", "cuauthor")
CA_HIT=ORG2_CFG.get("hitid_col")  # opsiyonel
CA_UNIT=ORG2_CFG.get("unit_id_col", "BirimID")
CA_AUTHOR=ORG2_CFG.get("author_id_col")  # opsiyonel

# Otomatik tespit: Eğer CA_HIT / CA_AUTHOR secrets'ta yoksa, tablodan bulmayı dene
if not CA_HIT or not CA_AUTHOR:
    _det_hit, _det_auth = detect_cuauthor_mapping()
    if _det_hit and not CA_HIT:
        CA_HIT = _det_hit
    if _det_auth and not CA_AUTHOR:
        CA_AUTHOR = _det_auth

WOSAUTHOR_TBL=ORG2_CFG.get("wos_author_table", "WosAuthor")
WA_AUTHOR=ORG2_CFG.get("wa_author_id_col", "AuthorId")
WA_NAME=ORG2_CFG.get("wa_name_col", "wosStandard")
# Researcher ID kolonu her veritabanında bulunmayabilir → opsiyonel yap
WA_RID=ORG2_CFG.get("wa_researcher_col")
WA_RID_EXPR=f"wa.{WA_RID}" if WA_RID else "NULL"

@st.cache_data(show_spinner=False)
def load_yoksis_df() -> pd.DataFrame:
    # CSV fallback
    if _csv_exists("yoksisbirim.csv"):
        df = _csv_load("yoksisbirim.csv").rename(columns={
            "BirimID": "id", "UstBirimID": "parent_id", "BirimAdi": "name", "Duzey": "lvl",
            "Tur": "tur"  # allow mapping from text levels
        })
    else:
        try:
            df = sql_to_df(
                f"SELECT * FROM {_qual(YOKSIS_TBL)}"
            )
        except Exception as ex:
            try:
                st.warning(f"YOKSIS sorgusu başarısız: {_qual(YOKSIS_TBL)} — {ex}")
            except Exception:
                pass
            return pd.DataFrame(columns=["id", "parent_id", "name", "lvl"])  # empty

    if df is None or df.empty:
        return pd.DataFrame(columns=["id", "parent_id", "name", "lvl"])  # empty

    # Standardize likely column names from your schema (YoksisId, UstYoksisId, Ad, Tur, AdIng ...)
    rename_map = {
        Y_ID: "id",
        Y_PARENT: "parent_id",
        Y_NAME: "name",
        Y_LEVEL: "lvl",
        "YoksisId": "id",
        "UstYoksisId": "parent_id",
        "BirimID": "id",
        "UstBirimID": "parent_id",
        "Ad": "name",
        "BirimAdi": "name",
        "Tur": "tur",
    }
    for k, v in list(rename_map.items()):
        if k in df.columns:
            df = df.rename(columns={k: v})

    # Coerce numeric types where appropriate
    for c in ("id", "parent_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If a numeric `lvl` is not present, derive it from textual `tur`
    if "lvl" not in df.columns or df["lvl"].isna().all():
        lvl_map = {
            # Turkish variants
            "universite": 1, "üniversite": 1, "üniversite": 1,
            "fakulte": 2, "fakülte": 2, "fakülte": 2,
            "bölüm": 3, "bölüm": 3, "bolum": 3, "department": 3,
            "anabilim dali": 4, "anabilim dalı": 4, "abd": 4, "division": 4,
            # English
            "university": 1, "faculty": 2, "school": 2, "college": 2,
        }
        def _to_lvl(x: object) -> float:
            if not isinstance(x, str):
                return float("nan")
            key = _strip_tr(x).lower()
            return float(lvl_map.get(key, float("nan")))
        if "tur" in df.columns:
            df["lvl"] = df["tur"].map(_to_lvl)

    # Keep only required columns
    keep = [c for c in ["id", "parent_id", "name", "lvl"] if c in df.columns]
    df = df[keep].dropna(subset=["id"]).copy()
    return df

@st.cache_data(show_spinner=False)
def yoksis_options(level: int, parent_id: int | None=None) -> pd.DataFrame:
    df=load_yoksis_df()
    if df.empty:
        return df
    out=df[df["lvl"] == level]
    if parent_id is not None:
        out=out[out["parent_id"] == int(parent_id)]
    return out.sort_values(by="name", kind="stable")

# Birden fazla düzey için (birleştirilmiş), parent_id opsiyonel
@st.cache_data(show_spinner=False)
def yoksis_options_any(levels: list[int], parent_id: int | None=None) -> pd.DataFrame:
    df=load_yoksis_df()
    if df.empty:
        return df
    out=df[df["lvl"].isin(levels)]
    if parent_id is not None:
        out=out[out["parent_id"] == int(parent_id)]
    return out.sort_values(by="name", kind="stable")

@st.cache_data(show_spinner=False)
def yoksis_descendants(root_ids: list[int]) -> list[int]:
    df=load_yoksis_df()
    if df.empty:
        return []
    ids=set(int(x) for x in root_ids)
    added=True
    while added:
        added=False
        children=df[df["parent_id"].isin(ids)]["id"].astype(int).tolist()
        for c in children:
            if c not in ids:
                ids.add(c)
                added=True
    return sorted(ids)

# WHERE parçası üret (cuauthor ile)

def org_where_cuauthor(selected_ids: list[int]) -> str:
    if not selected_ids:
        return "1=1"  # filtre yok
    ids_csv=",".join(str(int(x)) for x in selected_ids)
    return f"ca.{CA_UNIT} IN ({ids_csv})"

# --- Dinamik kurum alanı çözümleyici ve keşif ---

@st.cache_data(show_spinner=False)
def list_attribute_names(limit: int=1000) -> pd.DataFrame:
    """WosHitAttributes.Name için en sık görülen isimleri getirir."""
    # CSV fallback
    if _csv_exists("woshitattributes.csv"):
        df = _csv_load("woshitattributes.csv")
        if "Name" in df.columns:
            out = (
                df["Name"].dropna()
                  .value_counts()  # type: ignore[attr-defined]
                  .head(int(limit))
                  .rename_axis("Name").reset_index(name="Cnt")
            )
            return out
    try:
        sql=f"""
            SELECT TOP {limit} Name, COUNT(*) AS Cnt
            FROM dbo.WosHitAttributes
            GROUP BY Name
            ORDER BY Cnt DESC
        """
        return sql_to_df(sql)
    except Exception:
        return pd.DataFrame(columns=["Name", "Cnt"])  # Boş dönüş

# Mevcut seçim haritasını al (öncelik: session_state -> secrets -> defaults)

def get_org_names() -> dict:
    m=st.session_state.get("org_map")
    if m and all(k in m for k in ("UNI", "FAC", "DEPT", "SUB")):
        return m
    return {
        "UNI": ORG_CFG.get("uni_attr", UNI_ATTR),
        "FAC": ORG_CFG.get("faculty_attr", FAC_ATTR),
        "DEPT": ORG_CFG.get("department_attr", DEPT_ATTR),
        "SUB": ORG_CFG.get("subunit_attr", SUB_ATTR),
    }

# Session'a yaz

def set_org_names(uni: str, fac: str, dept: str, sub: str) -> None:
    st.session_state["org_map"]={"UNI": uni,
        "FAC": fac, "DEPT": dept, "SUB": sub}

# Attribute adları için olası varyasyonlar (örn. "affiliation.organization" vs "affiliation.organization-enhanced")
def _attr_aliases(attr: str) -> list[str]:
    alts=[attr]
    if attr.startswith("affiliation.") and not attr.endswith("-enhanced"):
        alts.append(attr + "-enhanced")
    return alts

# ---- Kurumsal isim eşleşmesini sağlamlaştırmak için yardımcılar ----
_TR_MAP=str.maketrans({
    "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U",
    "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
})

def _strip_tr(s: str) -> str:
    """Türkçe karakterleri sadeleştir, boşlukları tek boşluğa indir, kırp."""
    if not isinstance(s, str):
        return ""
    return " ".join(s.translate(_TR_MAP).split()).strip()

# Sık görülen fakülte/birim eş anlamlıları (TR→EN)
_TR_EN_SYNONYMS={
    # Institution / generic
    "FAKULTE": ["FACULTY"],
    "FAKULTESI": ["FACULTY"],
    "ENSTITU": ["INSTITUTE"],
    "ENSTITUSU": ["INSTITUTE"],
    "YUKSEKOKUL": ["SCHOOL", "COLLEGE"],
    "YUKSEKOKULU": ["SCHOOL", "COLLEGE"],
    "UNIVERSITESI": ["UNIVERSITY", "UNIV"],
    "UNIVERSITE": ["UNIVERSITY", "UNIV"],
    "UNIV": ["UNIVERSITY"],
    # Broad academic
    "FEN": ["SCIENCE", "SCIENCES"],
    "EDEBIYAT": ["LETTERS", "LITERATURE"],
    "FEN EDEBIYAT": ["SCIENCE AND LETTERS", "SCIENCE & LETTERS"],
    "MUHENDISLIK": ["ENGINEERING"],
    "EGITIM": ["EDUCATION"],
    "TIP": ["MEDICINE", "MEDICAL"],
    "ECZACILIK": ["PHARMACY"],
    "DIS HEKIMLIGI": ["DENTISTRY", "DENTAL"],
    "ZIRAAT": ["AGRICULTURE", "AGRICULTURAL"],
    "IKTISAT": ["ECONOMICS"],
    "IDARI BILIMLER": ["ADMINISTRATIVE SCIENCES", "ADMINISTRATIVE"],
    "ILETISIM": ["COMMUNICATION"],
    "HUKUK": ["LAW"],
    "MIMARLIK": ["ARCHITECTURE", "ARCH"],
    # Common engineering departments
    "INS AAT": ["CIVIL"],     # tokenisation may split as ["INS", "AAT"]
    "INSAAT": ["CIVIL"],
    "BILGISAYAR": ["COMPUTER"],
    "YAZILIM": ["SOFTWARE"],
    "ELEKTRIK": ["ELECTRICAL", "ELECTRIC"],
    "ELEKTRONIK": ["ELECTRONICS"],
    "ELEKTRIK ELEKTRONIK": ["ELECTRICAL AND ELECTRONICS", "ELECTRICAL & ELECTRONICS"],
    "MAKINE": ["MECHANICAL"],
    "ENDUSTRI": ["INDUSTRIAL"],
    "CEVRE": ["ENVIRONMENTAL"],
    "GIDA": ["FOOD"],
    "JEOMATIK": ["GEOMATICS"],
    "HARITA": ["SURVEYING", "GEODESY"],
    "MATERIALS": ["MATERIALS"],  # allow English token to persist
    # Health/other common departments
    "HEM S IREL IK": ["NURSING"],
    "HEMSIRELIK": ["NURSING"],
    "VETERINER": ["VETERINARY"],
    # Sciences
    "KIMYA": ["CHEMISTRY"],
    "FIZIK": ["PHYSICS"],
    "MATEMATIK": ["MATHEMATICS"],
    "BIYOLOJI": ["BIOLOGY", "LIFE SCIENCES"],
}

def _tokens(s: str) -> list[str]:
    """Kurum adını sadeleştirip majuskül tokenlere ayırır (harf/rakam)."""
    import re
    s2=_strip_tr(s).upper()
    toks=re.findall(r"[A-Z0-9]+", s2)
    # Drop generic structural words that rarely appear in English address strings
    STOPWORDS={
        "BOLUM", "BOLUMU", "ANABILIM", "ANABILIMDALI", "ANA", "BILIM", "DALI",
        "PROGRAM", "PROGRAMI", "ENSTITU", "ENSTITUSU", "YUKSEKOKUL", "YUKSEKOKULU",
        "FAKULTE", "FAKULTESI"
    }
    toks=[t for t in toks if t not in STOPWORDS]
    return toks

def _wildcard_from_tokens(toks: list[str]) -> str:
    """Sıralı tokenlerden %A%B%C% deseni üretir."""
    if not toks:
        return ""
    return "%" + "%".join(toks) + "%"

def _synonym_tokens(toks: list[str]) -> list[list[str]]:
    """TR→EN eş anlamlılarla alternatif token dizileri oluştur."""
    # Basit strateji: her token için eş anlamlıları varsa yerlerine de koy
    outs=[toks]
    # birleştirilmiş stringlerde TR→EN map denemeleri
    for tr, ens in _TR_EN_SYNONYMS.items():
        tr_toks=tr.split()
        # alt dizi olarak geçiyorsa varyasyon ekle
        for i in range(0, max(1, len(toks) - len(tr_toks) + 1)):
            if toks[i:i+len(tr_toks)] == tr_toks:
                for en in ens:
                    new=toks[:i] + en.split() + toks[i+len(tr_toks):]
                    outs.append(new)
    # Tekil eş anlamlı yer değişimleri
    for i, t in enumerate(toks):
        if t in _TR_EN_SYNONYMS:
            for en in _TR_EN_SYNONYMS[t]:
                new=toks.copy(); new[i:i+1]=en.split()
                outs.append(new)
    # Yinelenenleri kaldır
    uniq=[]
    seen=set()
    for arr in outs:
        key=" ".join(arr)
        if key not in seen:
            seen.add(key); uniq.append(arr)
    return uniq

def _org_like_patterns(raw: str) -> list[str]:
    """
    Kurum/fakülte/birim ismi için birden fazla LIKE deseni üret.
    Daha kapsayıcı olması için:
      - Türkçe karakterler sadeleştirilir,
      - TR→EN eş anlamlılarla varyasyonlar üretilir,
      - her varyant için hem tam-token hem de **3–5 harflik prefix** eşleşmeleri eklenir,
      - yaygın **kısaltmalar** da (ENG, EDU, AGR, PHARM, ARCH, LAW, MED, COMM vb.) denenir.
    """
    toks=_tokens(raw)
    if not toks:
        return []

    variants=_synonym_tokens(toks)
    patterns: list[str]=[]

    # Yardımcı: bir token için prefixler üret (3–5 harf)
    def _prefixes(t: str) -> list[str]:
        t=t.strip()
        out: list[str]=[]
        for k in (5, 4, 3):
            if len(t) >= k:
                out.append(t[:k])
        return out

    # Sektörel/kurumsal bilinen kısaltmalar (TR→EN kısa kodlar)
    ABBR={
        "EGITIM": ["EDUC", "EDU", "EGIT", "EGT"],
        "MUHENDISLIK": ["ENG", "MUH"],
        "MIMARLIK": ["ARCH"],
        "ZIRAAT": ["AGR", "AGRI"],
        "FEN": ["SCI", "SC"],
        "EDEBIYAT": ["LIT", "LET"],
        "TIP": ["MED"],
        "ECZACILIK": ["PHARM"],
        "HUKUK": ["LAW"],
        "ILETISIM": ["COMM"],
        "IDARI": ["ADM", "ADMIN"],
        "BILGISAYAR": ["COMP", "CS"],
        "YAZILIM": ["SOFT"],
        "ELEKTRIK": ["ELEC"],
        "ELEKTRONIK": ["ELEC", "ELECTRON"],
        "MAKINE": ["MECH"],
        "ENDUSTRI": ["IND", "IE"],
        "CEVRE": ["ENV", "ENVIR"],
        "KIMYA": ["CHEM"],
        "FIZIK": ["PHYS"],
        "MATEMATIK": ["MATH"],
        "BIYOLOJI": ["BIOL", "LIFE"],
        "VETERINER": ["VET"],
        "DIS": ["DENT", "DENTAL"],
    }

    # 1) Sıralı tüm tokenler (katı eşleşme): %A%B%C%
    patterns.extend(["%" + "%".join(v) + "%" for v in variants if v])

    # 2) Tek token ve prefix tabanlı esneklik
    all_single_tokens={t for v in variants for t in v}
    for t in all_single_tokens:
        if len(t) >= 3:
            # tam token
            patterns.append(f"%{t}%")
            # prefixler
            for px in _prefixes(t):
                patterns.append(f"%{px}%")
            # bilinen kısaltmalar
            if t in ABBR:
                for ab in ABBR[t]:
                    patterns.append(f"%{ab}%")

    # 3) Varyantların ilk token’i için de looser tek-token deseni
    for v in variants:
        if v:
            patterns.append(f"%{v[0]}%")
            for px in _prefixes(v[0]):
                patterns.append(f"%{px}%")

    # 4) Ham metnin sadeleştirilmiş hali (tam parça)
    raw_simpl=_strip_tr(raw)
    if raw_simpl:
        patterns.append(f"%{raw_simpl}%")

    # Yinelenenleri kaldır
    out: list[str]=[]
    seen: set[str]=set()
    for p in patterns:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out[:80]  # çok fazla LIKE paterni performansı düşürmesin diye sınırla



def _addr_like_clause(alias: str, raw: str) -> str:
    """WosAddress.full_address üzerinde TR/EN varyasyonlarına göre LIKE koşulları üretir."""
    pats = _org_like_patterns(raw)
    if not pats:
        return "1=1"
    ors = " OR ".join(
        f"{alias}.full_address COLLATE Turkish_CI_AI LIKE N'{_q(p)}'" for p in pats
    )
    return f"({ors})"

# Strict Çukurova-only address clause
def _cukurova_addr_clause(alias: str) -> str:
    """Strict filter: only addresses that clearly belong to Cukurova University.
    Matches common variants in TR/EN to avoid hospital or unrelated hits.
    """
    return (
        f"("
        f"{alias}.full_address COLLATE Turkish_CI_AI LIKE N'%Cukurova Univ%' OR "
        f"{alias}.full_address COLLATE Turkish_CI_AI LIKE N'%Cukurova University%' OR "
        f"{alias}.full_address COLLATE Turkish_CI_AI LIKE N'%Çukurova Üniversitesi%'"
        f")"
    )

def _like_join(alias: str, hit_table_alias: str, attr: str, val: str) -> str:
    """
    Kurumsal filtre JOIN'i: attribute isimleri için aliasları kullanır ve
    değer eşleşmesini birden fazla LIKE paterniyle (TR→EN varyasyonlar) yapar.
    """
    names_csv=", ".join(f"'{_q(n)}'" for n in _attr_aliases(attr))
    pats=_org_like_patterns(val)
    if not pats:
        # en azından ham arama kalsın
        pats=[f"%{_strip_tr(val)}%"]
    ors=" OR ".join(
        f"{alias}.Value COLLATE Turkish_CI_AI LIKE N'{_q(p)}'" for p in pats)
    return (
        f"JOIN dbo.WosHitAttributes {alias} ON {alias}.HitId = {hit_table_alias}.HitId "
        f"AND {alias}.Name IN ({names_csv}) "
        f"AND ({ors})"
    )

# ---- Dinamik attribute value keşfi: seçilen ad için gerçek değerleri bul ----

@st.cache_data(show_spinner=False)
def discover_attr_values(attr_name: str, raw: str, limit: int=50) -> list[str]:
    """
    WosHitAttributes içinde, verilen attribute (ör. affiliation.suborganization)
    ve kullanıcının seçtiği ad (raw) için yıl aralığında geçen gerçek Value'ları keşfeder.
    Dönen liste doğrudan IN (...) içinde kullanılabilir.
    """
    try:
        if not raw or raw == _DEF_ALL:
            return []
        pats=_org_like_patterns(raw)
        if not pats:
            return []

        names_csv=", ".join(f"'{_q(n)}'" for n in _attr_aliases(attr_name))
        ors=" OR ".join(
            f"a.Value COLLATE Turkish_CI_AI LIKE N'{_q(p)}'" for p in pats)
        sql = (
            f"""
            SELECT DISTINCT TOP {int(limit)} a.Value AS v
            FROM dbo.WosHitAttributes a
            JOIN dbo.WOSHit wh ON wh.HitId = a.HitId
            WHERE a.Name IN ({names_csv})
              AND ({ors})
              AND wh.SourcePublishYear BETWEEN :y1 AND :y2
            ORDER BY v
            """
        )
        df = run_sql(sql, params={"y1": int(year_min), "y2": int(year_max)})
        vals=[str(x) for x in df.get(
            "v", pd.Series(dtype=object)).dropna().tolist()]
        return vals
    except Exception:
        return []
    pats=_org_like_patterns(raw)
    if not pats:
        pats=[f"%{_strip_tr(raw)}%"]
    ors=" OR ".join(
        f"{alias}.full_address COLLATE Turkish_CI_AI LIKE N'{_q(p)}'" for p in pats)
    return f"({ors})"

# Üst seçimlere göre attribute değerlerini getir (kademeli)

@st.cache_data(show_spinner=False)
def distinct_values_for(attr_name: str,
                        sel_uni: str | None=None,
                        sel_fac: str | None=None,
                        sel_dept: str | None=None) -> list[str]:
    try:
        # CSV fallback
        if _csv_exists("woshit.csv") and _csv_exists("woshitattributes.csv"):
            wh = _csv_load("woshit.csv")
            wa = _csv_load("woshitattributes.csv")
            # yıl filtresi
            if "SourcePublishYear" in wh.columns:
                y1, y2 = int(year_min), int(year_max)
                wh = wh[(pd.to_numeric(wh["SourcePublishYear"], errors="coerce") >= y1) & (pd.to_numeric(wh["SourcePublishYear"], errors="coerce") <= y2)]
            names = get_org_names()
            # hedef attr belirle
            if attr_name == UNI_ATTR:
                target = names["UNI"]
            elif attr_name == FAC_ATTR:
                target = names["FAC"]
            elif attr_name == DEPT_ATTR:
                target = names["DEPT"]
            else:
                target = names["SUB"]
            # alias'lı isim varyasyonları
            candidates = _attr_aliases(target)
            cur = wa[wa["Name"].astype(str).isin(candidates)]
            # JOIN wh
            if "HitId" in wh.columns and "HitId" in cur.columns:
                cur = cur.merge(wh[["HitId"] + (["SourcePublishYear"] if "SourcePublishYear" in wh.columns else [])], on="HitId", how="inner")
            vals = sorted(cur["Value"].dropna().astype(str).unique().tolist()) if "Value" in cur.columns else []
            return vals
        names=get_org_names()
        # Hedef attr'ı, mevcut haritaya göre belirle
        if attr_name == UNI_ATTR:
            target=names["UNI"]
        elif attr_name == FAC_ATTR:
            target=names["FAC"]
        elif attr_name == DEPT_ATTR:
            target=names["DEPT"]
        else:
            target=names["SUB"]
        # JOIN parçaları da güncel isimlere göre kurulsun
        joins=[]
        if sel_uni and sel_uni != _DEF_ALL:
            joins.append(_like_join("orgu", "wh", names["UNI"], sel_uni))
        if sel_fac and sel_fac != _DEF_ALL:
            joins.append(_like_join("orgf", "wh", names["FAC"], sel_fac))
        if sel_dept and sel_dept != _DEF_ALL:
            joins.append(_like_join("orgd", "wh", names["DEPT"], sel_dept))
        joins_sql="\n".join(joins)
        names_csv=", ".join(f"'{_q(n)}'" for n in _attr_aliases(target))
        sql = (
            """
            SELECT DISTINCT cur.Value AS Val
            FROM dbo.WOSHit wh
            JOIN dbo.WosHitAttributes cur ON cur.HitId = wh.HitId AND cur.Name IN (""" + names_csv + """)
            """ + joins_sql + """
            WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
            ORDER BY Val
            """
        )
        dfv = run_sql(sql, params={"y1": int(year_min), "y2": int(year_max)})
        vals=[str(x) for x in dfv.get("Val", pd.Series(dtype=object)).dropna().tolist()]
        if vals:
            return vals
        # Fallback: bazı kurulumlarda Name sütunu `affiliation.organizationX` gibi varyantlara sahip olabiliyor.
        # Bu durumda Name için LIKE prefix eşleşmesi dene (case-insensitive).
        names_like = " OR ".join(
            [f"LOWER(cur.Name) LIKE LOWER('{_q(n)}%')" for n in _attr_aliases(target)]
        )
        sql_like = (
            """
            SELECT DISTINCT cur.Value AS Val
            FROM dbo.WOSHit wh
            JOIN dbo.WosHitAttributes cur ON cur.HitId = wh.HitId
            """ + ("\n" + joins_sql if joins_sql else "") + "\n" +
            """
            WHERE (""" + names_like + ")\n"
            "  AND wh.SourcePublishYear BETWEEN :y1 AND :y2\n"
            "ORDER BY Val\n"
        )
        dfv2 = run_sql(sql_like, params={"y1": int(year_min), "y2": int(year_max)})
        vals2=[str(x) for x in dfv2.get("Val", pd.Series(dtype=object)).dropna().tolist()]
        return vals2
    except Exception:
        return []

# Seçimlere göre JOIN parçalarını üret
def build_org_joins(selected_uni: str | None,
                    selected_fac: str | None,
                    selected_dept: str | None,
                    selected_sub: str | None,
                    addr_alias: str="wa") -> str:
    clauses: list[str]=[]
    # Her durumda: sadece Çukurova Üniversitesi adresleri (hastane gibi kurumları dışla)
    clauses.append(_cukurova_addr_clause(addr_alias))
    # full_address içinde hiyerarşik tüm bilgiler bulunduğu için, her seçili düzeyi
    # ayrı bir LIKE grubu olarak AND ile bağlarız. Her grubun içinde TR→EN varyasyonları OR ile birleşir.
    if selected_uni and selected_uni != _DEF_ALL:
        clauses.append(_addr_like_clause(addr_alias, selected_uni))
    if selected_fac and selected_fac != _DEF_ALL:
        clauses.append(_addr_like_clause(addr_alias, selected_fac))
    if selected_dept and selected_dept != _DEF_ALL:
        clauses.append(_addr_like_clause(addr_alias, selected_dept))
    if selected_sub and selected_sub != _DEF_ALL:
        clauses.append(_addr_like_clause(addr_alias, selected_sub))

    if not clauses:
        return ""  # kurumsal filtre yok

    where_part=" AND ".join(clauses)
    # Tek bir JOIN ile tüm adres filtrelerini uygula
    return f"\nJOIN dbo.WosAddress {addr_alias} ON {addr_alias}.HitId = wh.HitId AND {where_part}"

# Yeni: WosHitAttributes tabanlı kurumsal JOIN'ler (attribute-based)

def build_org_joins_attrs(selected_uni, selected_fac, selected_dept, selected_sub) -> str:
    """
    WosHitAttributes tabanlı kurumsal JOIN'ler.
    - Önce seçime karşılık gelen gerçek Value'ları dinamik olarak keşfeder (discover_attr_values).
    - Değer listesi bulunamazsa LIKE varyasyonlarıyla eşleşir.
    """
    names=get_org_names()
    joins: list[str]=[]

    # Üniversite
    if selected_uni and selected_uni != _DEF_ALL:
        vals=discover_attr_values(names["UNI"], selected_uni)
        if vals:
            joins.append(
                f"JOIN dbo.WosHitAttributes orgu ON orgu.HitId = wh.HitId "
                f"AND orgu.Name IN ({_in_list_sql(_attr_aliases(names['UNI']))}) "
                f"AND orgu.Value IN ({_in_list_sql(vals)})"
            )
        else:
            joins.append(_like_join("orgu", "wh", names["UNI"], selected_uni))

    # Fakülte / birim
    if selected_fac and selected_fac != _DEF_ALL:
        vals=discover_attr_values(names["FAC"], selected_fac)
        if vals:
            joins.append(
                f"JOIN dbo.WosHitAttributes orgf ON orgf.HitId = wh.HitId "
                f"AND orgf.Name IN ({_in_list_sql(_attr_aliases(names['FAC']))}) "
                f"AND orgf.Value IN ({_in_list_sql(vals)})"
            )
        else:
            joins.append(_like_join("orgf", "wh", names["FAC"], selected_fac))

    # Bölüm / ABD
    if selected_dept and selected_dept != _DEF_ALL:
        vals=discover_attr_values(names["DEPT"], selected_dept)
        if vals:
            joins.append(
                f"JOIN dbo.WosHitAttributes orgd ON orgd.HitId = wh.HitId "
                f"AND orgd.Name IN ({_in_list_sql(_attr_aliases(names['DEPT']))}) "
                f"AND orgd.Value IN ({_in_list_sql(vals)})"
            )
        else:
            joins.append(_like_join(
                "orgd", "wh", names["DEPT"], selected_dept))

    # Alt birim
    if selected_sub and selected_sub != _DEF_ALL:
        vals=discover_attr_values(names["SUB"], selected_sub)
        if vals:
            joins.append(
                f"JOIN dbo.WosHitAttributes orgs ON orgs.HitId = wh.HitId "
                f"AND orgs.Name IN ({_in_list_sql(_attr_aliases(names['SUB']))}) "
                f"AND orgs.Value IN ({_in_list_sql(vals)})"
            )
        else:
            joins.append(_like_join("orgs", "wh", names["SUB"], selected_sub))

    return "\n".join(joins)

# ----------- YENİ: Kurumsal filtreli SQL query yardımcıları -----------

def build_org_series_sql(
    year_min: int,
    year_max: int,
    selected_uni,
    selected_fac,
    selected_dept,
    selected_sub,
    ids: list[int],
    join_mode: str="addr"
) -> str:
    """
    Alan-yıllık seri için kurumsal filtreli SQL üretir.
    join_mode: "addr" (adres tabanlı, varsayılan) veya "attr" (attribute tabanlı).
    """
    if CA_HIT and CA_AUTHOR:
        # CUAUTHOR tabanlı (değişmeden)
        return f"""
            SELECT sub.Value AS Alan, wh.SourcePublishYear AS Yil, COUNT(*) AS YayinSayisi
            FROM dbo.WOSHit wh
            JOIN dbo.WosHitAttributes sub ON sub.HitId = wh.HitId AND sub.Name IN ({_in_list_sql(resolve_subject_attrs(int(year_min), int(year_max)))})
            JOIN {_qual(CUAUTHOR_TBL)} ca ON ca.{CA_HIT} = wh.HitId
            WHERE {org_where_cuauthor(ids)}
              AND wh.SourcePublishYear BETWEEN :y1 AND :y2
            GROUP BY sub.Value, wh.SourcePublishYear
            ORDER BY sub.Value, wh.SourcePublishYear;
        """
    else:
        # JOIN seçenekleri
        joins_addr=build_org_joins(
            selected_uni, selected_fac, selected_dept, selected_sub)
        joins_attr=build_org_joins_attrs(
            selected_uni, selected_fac, selected_dept, selected_sub)
        joins_sql=joins_addr if join_mode == "addr" else joins_attr
        return f"""
            SELECT sub.Value AS Alan, wh.SourcePublishYear AS Yil, COUNT(*) AS YayinSayisi
            FROM dbo.WOSHit wh
            JOIN dbo.WosHitAttributes sub ON sub.HitId = wh.HitId AND sub.Name IN ({_in_list_sql(resolve_subject_attrs(int(year_min), int(year_max)))})
            {joins_sql}
            WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
            GROUP BY sub.Value, wh.SourcePublishYear
            ORDER BY sub.Value, wh.SourcePublishYear;
        """


def build_org_researchers_sql(
    year_min: int,
    year_max: int,
    selected_uni,
    selected_fac,
    selected_dept,
    selected_sub,
    ids: list[int],
    join_mode: str="addr"
) -> str:
    """
    Kurumsal filtreli araştırmacı listesi SQL.
    join_mode: "addr" (adres tabanlı, varsayılan) veya "attr" (attribute tabanlı).
    """
    if CA_HIT and CA_AUTHOR:
        # CUAUTHOR tabanlı (değişmeden)
        return f"""
            SELECT DISTINCT wa.{WA_NAME} AS Yazar, {WA_RID_EXPR} AS ResearcherID
            FROM {_qual(CUAUTHOR_TBL)} ca
            JOIN {_qual(WOSAUTHOR_TBL)} wa ON wa.{WA_AUTHOR} = ca.{CA_AUTHOR}
            JOIN dbo.WOSHit wh ON wh.HitId = ca.{CA_HIT}
            WHERE {org_where_cuauthor(ids)}
              AND wh.SourcePublishYear BETWEEN :y1 AND :y2
            ORDER BY Yazar;
        """
    else:
        if join_mode == "addr":
            # Use a different alias for address table to avoid clashing with WosAuthor alias 'wa'
            joins_sql=build_org_joins(
                selected_uni, selected_fac, selected_dept, selected_sub, addr_alias="ad")
            return f"""
                SELECT DISTINCT wa.{WA_NAME} AS Yazar, NULL AS ResearcherID
                FROM dbo.WOSHit wh
                JOIN dbo.WosAuthor wa ON wa.HitId = wh.HitId
                {joins_sql}
                WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
                ORDER BY Yazar;
            """
        else:
            # Attribute-based: WosHit JOIN WosAuthor, ardından attribute joinler
            joins_sql=build_org_joins_attrs(
                selected_uni, selected_fac, selected_dept, selected_sub)
            return f"""
                SELECT DISTINCT wa.{WA_NAME} AS Yazar, NULL AS ResearcherID
                FROM dbo.WOSHit wh
                JOIN dbo.WosAuthor wa ON wa.HitId = wh.HitId
                {joins_sql}
                WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
                ORDER BY Yazar;
            """

# -------- YÖKSİS-ID tabanlı seri SQL (CuAuthor -> CuAuthorRID -> WosAuthor -> WOSHit) --------

def build_org_series_sql_yoksis(
    year_min: int,
    year_max: int,
    yoksis_ids: list[int],
) -> str:
    """
    Seçilen YÖKSİS düğümlerinin (veya torunlarının) ID listesiyle
    alan-yıllık seri SQL'ini üretir.
    Zincir: CuAuthor (YoksisId) -> CuAuthorRID (ResearcherID) ->
            WosAuthor (researcherId) -> WOSHit/Attributes.
    """
    if not yoksis_ids:
        # Güvenli tarafta kal: asla boş IN (...) üretme
        return ""
    ids_csv = ",".join(str(int(x)) for x in yoksis_ids)
    subjects_csv = _in_list_sql(
        resolve_subject_attrs(int(year_min), int(year_max))
    )
    return f"""
        SELECT sub.Value AS Alan, wh.SourcePublishYear AS Yil, COUNT(*) AS YayinSayisi
        FROM (
            SELECT DISTINCT wa.HitId
            FROM {_qual(CUAUTHOR_TBL)} c
            JOIN {_qual('CuAuthorRID')} cr ON cr.CuAuthorID = c.ID
            JOIN dbo.WosAuthor wa ON wa.researcherId = cr.ResearcherID
            JOIN dbo.WOSHit wh ON wh.HitId = wa.HitId
            WHERE c.YoksisId IN ({ids_csv})
              AND wh.SourcePublishYear BETWEEN :y1 AND :y2
        ) h
        JOIN dbo.WOSHit wh ON wh.HitId = h.HitId
        JOIN dbo.WosHitAttributes sub
          ON sub.HitId = h.HitId AND sub.Name IN ({subjects_csv})
        GROUP BY sub.Value, wh.SourcePublishYear
        ORDER BY sub.Value, wh.SourcePublishYear;
    """

# --------- Araştırmacı tablosu gösterici yardımcı ---------

def _render_researchers_table(df: pd.DataFrame) -> None:
    """
    Kullanıcı dostu araştırmacı tablosu:
    - 'ResearcherID' sütununu 'Araştırmacı ID' olarak yeniden adlandırır.
    - Eğer sütun tamamen boş/None ise sütunu gizler ve küçük bir not gösterir.
    """
    if df is None or df.empty:
        st.info("Seçilen filtre için araştırmacı bulunamadı.")
        return
    df2=df.copy()
    if "ResearcherID" in df2.columns:
        # Normalize: None / "None" / boş string → NaN
        df2["ResearcherID"]=df2["ResearcherID"].replace(
            {None: pd.NA, "None": pd.NA, "": pd.NA})
        if df2["ResearcherID"].isna().all():
            df2=df2.drop(columns=["ResearcherID"])
            st.caption(
                "ℹ️ Bu veri kümesinde **Araştırmacı ID** bilgisi bulunamadı; bu sütun gizlendi.")
        else:
            df2=df2.rename(columns={"ResearcherID": "Araştırmacı ID"})
    st.dataframe(df2, use_container_width=True)

# -------- Kurumsal eşleşme doğrulama yardımcıları (addr vs attr) --------

def validation_subjects_and_samples(year_min: int, year_max: int,
                                    selected_uni, selected_fac, selected_dept, selected_sub,
                                    join_mode: str="addr",
                                    top_k: int=25, sample_n: int=40) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Seçilen kurumsal filtreyi kullanarak:
      1) En çok görülen alan/konu listesini (top_k) ve
      2) Örnek satırları (sample_n; yıl, alan, full_address) döndürür.
    join_mode: "addr" veya "attr" (build_org_joins/build_org_joins_attrs kullanır).
    """
    subjects_csv=_in_list_sql(
        resolve_subject_attrs(int(year_min), int(year_max)))
    if join_mode == "attr":
        joins=build_org_joins_attrs(
            selected_uni, selected_fac, selected_dept, selected_sub)
    else:
        # Adres modunda: adres JOIN'i zaten bu fonksiyon içinde yapılır; alias çakışmaması için 'ad' kullan
        joins=build_org_joins(
            selected_uni, selected_fac, selected_dept, selected_sub, addr_alias="ad")

    # 1) Alan dağılımı (top_k)
    sql_top = f"""
        SELECT TOP {int(top_k)} sub.Value AS Alan, COUNT(*) AS Kayit
        FROM dbo.WOSHit wh
        JOIN dbo.WosHitAttributes sub
          ON sub.HitId = wh.HitId AND sub.Name IN ({subjects_csv})
        {joins}
        WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
        GROUP BY sub.Value
        ORDER BY Kayit DESC;
    """
    # Örnek satırlar (yıl/alan/full_address)
    # - addr modunda: adres JOIN'i zaten 'joins' içinde var ve alias 'ad' olarak geldi → ayrı LEFT JOIN eklemeyiz.
    # - attr modunda: adres JOIN'i yok → burada LEFT JOIN ile 'wa' alias'ını ekleriz.
    if join_mode == "attr":
        full_addr_select="ISNULL(wa.full_address, '(adres yok)') AS full_address"
        left_join_addr="LEFT JOIN dbo.WosAddress wa ON wa.HitId = wh.HitId"
    else:
        full_addr_select="ISNULL(ad.full_address, '(adres yok)') AS full_address"
        left_join_addr=""  # adres already joined in joins with alias 'ad'

    sql_samples=f"""
        SELECT TOP {int(sample_n)}
            wh.SourcePublishYear AS Yil,
            sub.Value AS Alan,
            {full_addr_select}
        FROM dbo.WOSHit wh
        JOIN dbo.WosHitAttributes sub
          ON sub.HitId = wh.HitId AND sub.Name IN ({subjects_csv})
        {joins}
        {left_join_addr}
        WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
        ORDER BY wh.SourcePublishYear DESC;
    """
    try:
        params = {"y1": int(year_min), "y2": int(year_max)}
        df_top = run_sql(sql_top, params=params)
        df_smp = run_sql(sql_samples, params=params)
    except Exception:
        df_top, df_smp=pd.DataFrame(columns=["Alan", "Kayit"]), pd.DataFrame(
            columns=["Yil", "Alan", "full_address"])

    return normalize_columns(df_top, "alan_dagilim"), normalize_columns(df_smp, "alan_yillik")

# -------- Basitleştirilmiş mod bazlı SQL üretici (tüm/veri yok durumları için) --------
def build_org_series_sql_by_mode(
    year_min: int,
    year_max: int,
    selected_uni,
    selected_fac,
    selected_dept,
    selected_sub,
    ids: list[int],
    mode: str = "auto"  # auto | addr | attr | uni_only | none
) -> tuple[str, str]:
    """
    mode:
      - auto: sadece işaret; SQL dışarıda oluşturulur (bu fonksiyon "" döndürür)
      - addr: WosAddress.full_address ile tüm seçili düzeyler
      - attr: WosHitAttributes ile tüm seçili düzeyler
      - uni_only: sadece üniversite düzeyi
      - none: kurumsal filtre yok (tüm veriler)
    Dönüş: (sql, human_label)
    """
    subjects_csv = _in_list_sql(
        resolve_subject_attrs(int(year_min), int(year_max))
    )

    # 0) Otomatik modda bu fonksiyon SQL üretmez
    if mode == "auto":
        return "", "auto"

    # 1) Kurumsal filtre yok → tüm veri
    if mode == "none":
        sql = f"""
            WITH Y AS (
                SELECT :y1 AS Yil
                UNION ALL
                SELECT Yil + 1 FROM Y WHERE Yil < :y2
            ),
            D AS (
                SELECT sub.Value AS Alan, wh.SourcePublishYear AS Yil, COUNT(*) AS YayinSayisi
                FROM dbo.WOSHit wh
                JOIN dbo.WosHitAttributes sub
                  ON sub.HitId = wh.HitId AND sub.Name IN ({subjects_csv})
                WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
                GROUP BY sub.Value, wh.SourcePublishYear
            ),
            A AS (SELECT DISTINCT Alan FROM D)
            SELECT a.Alan, y.Yil, COALESCE(d.YayinSayisi, 0) AS YayinSayisi
            FROM A a
            CROSS JOIN Y y
            LEFT JOIN D d ON d.Alan = a.Alan AND d.Yil = y.Yil
            ORDER BY a.Alan, y.Yil
            OPTION (MAXRECURSION 0);
        """
        return sql, "tümü"

    # 2) Sadece üniversite düzeyi (adres tabanlı hızlı yol)
    if mode == "uni_only":
        where_addr = _addr_like_clause("wa", selected_uni) if selected_uni and selected_uni != _DEF_ALL else "1=1"
        join_addr = "" if where_addr == "1=1" else f"JOIN dbo.WosAddress wa ON wa.HitId = wh.HitId AND {where_addr}"
        sql = f"""
            SELECT sub.Value AS Alan, wh.SourcePublishYear AS Yil, COUNT(*) AS YayinSayisi
            FROM dbo.WOSHit wh
            JOIN dbo.WosHitAttributes sub
              ON sub.HitId = wh.HitId AND sub.Name IN ({subjects_csv})
            {join_addr}
            WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
            GROUP BY sub.Value, wh.SourcePublishYear
            ORDER BY sub.Value, wh.SourcePublishYear;
        """
        return sql, f"sadece üniversite: {selected_uni or '—'}"

    # 3) Tüm seçili düzeylerle — adres veya attribute
    if mode in ("addr", "attr"):
        joins = build_org_joins(selected_uni, selected_fac, selected_dept, selected_sub) if mode == "addr" \
                else build_org_joins_attrs(selected_uni, selected_fac, selected_dept, selected_sub)
        sql = f"""
            SELECT sub.Value AS Alan, wh.SourcePublishYear AS Yil, COUNT(*) AS YayinSayisi
            FROM dbo.WOSHit wh
            JOIN dbo.WosHitAttributes sub
              ON sub.HitId = wh.HitId AND sub.Name IN ({subjects_csv})
            {joins}
            WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
            GROUP BY sub.Value, wh.SourcePublishYear
            ORDER BY sub.Value, wh.SourcePublishYear;
        """
        return sql, ("adres" if mode == "addr" else "attribute")

    # Varsayılan: bilinmeyen mod
    return "", mode
@st.cache_data(show_spinner=False)
def quick_count_for_mode(year_min: int, year_max: int,
                         selected_uni, selected_fac, selected_dept, selected_sub,
                         ids: list[int], mode: str) -> int:
    """Verinin olup olmadığını hızlı kontrol etmek için COUNT(*) döndürür."""
    sql, _ = build_org_series_sql_by_mode(year_min, year_max,
                                          selected_uni, selected_fac, selected_dept, selected_sub,
                                          ids, mode)
    if not sql:
        return 0

    # MAXRECURSION ipucunu çıkarıp sarmal sayım yapalım
    sql_no_opt = sql.replace("OPTION (MAXRECURSION 0);", "")
    q = f"SELECT COUNT(*) AS Cnt FROM ({sql_no_opt}) t"
    try:
        df = run_sql(q, params={"y1": int(year_min), "y2": int(year_max)})
        return int(df['Cnt'].iloc[0]) if not df.empty else 0
    except Exception:
        return 0
# ------ Esnek deneme/gevşetme: kurumsal filtre boş dönerse aşamalı gevşet ------RSION 0)", "")
def try_fetch_org_series(
    year_min: int,
    year_max: int,
    selected_uni,
    selected_fac,
    selected_dept,
    selected_sub,
    ids: list[int],
) -> tuple[pd.DataFrame, str]:
    """
    Kurumsal filtreli yıllık seriyi getirmeyi dener.
    Eğer kullanıcı fakülte/bölüm/alt birim düzeylerinden **herhangi birini** seçtiyse,
    **katı (strict)** modda sadece tam eşleşmeyi getirir; hiçbir gevşetme yapmaz.
    Seçim yapılmadıysa önce attr/addr dener, sonra kademeli gevşetir.
    """
    # 0) Eğer YÖKSİS ID listesi geldiyse, önce doğrudan bu yolla dene
    if ids and isinstance(ids, (list, tuple)) and len(ids) > 0:
        try:
            sql_yx = build_org_series_sql_yoksis(int(year_min), int(year_max), [int(x) for x in ids])
            if sql_yx:
                params = {"y1": int(year_min), "y2": int(year_max)}
                df_yx = run_sql(sql_yx, params=params)
                df_yx = normalize_columns(df_yx, "alan_yillik")
                if df_yx is not None and not df_yx.empty:
                    return df_yx, "yoksis-id"
        except Exception:
            pass

    def _lab(u, f, d, s):
        parts = [p for p in [u, f, d, s] if p and p != _DEF_ALL]
        return " > ".join(parts) if parts else "—"

    # "— Tümü —" → None normalizasyonu
    u0 = None if (selected_uni == _DEF_ALL) else selected_uni
    f0 = None if (selected_fac == _DEF_ALL) else selected_fac
    d0 = None if (selected_dept == _DEF_ALL) else selected_dept
    s0 = None if (selected_sub == _DEF_ALL) else selected_sub

    # Eğer kullanıcı fakülte/bölüm/alt birimden herhangi birini seçtiyse
    # katı mod: sadece tam eşleşmeyi deneriz (önce attr sonra addr)
    strict = bool(f0 or d0 or s0)

    attempts: list[tuple[str, str, str | None, str | None, str | None, str | None]] = []
    if strict:
        attempts.append(("attr", "tam", u0, f0, d0, s0))
        attempts.append(("addr", "tam", u0, f0, d0, s0))
    else:
        # 1) Tam filtre (önce attr, sonra addr)
        attempts.append(("attr", "tam", u0, f0, d0, s0))
        attempts.append(("addr", "tam", u0, f0, d0, s0))
        # 2) Alt birimi at
        attempts.append(("attr", "alt_yok", u0, f0, d0, None))
        attempts.append(("addr", "alt_yok", u0, f0, d0, None))
        # 3) Bölümü de at (sadece fakülte + opsiyonel üniversite)
        attempts.append(("attr", "bolum_yok", u0, f0, None, None))
        attempts.append(("addr", "bolum_yok", u0, f0, None, None))
        # 4) Sadece üniversite
        attempts.append(("attr", "sadece_uni", u0, None, None, None))
        attempts.append(("addr", "sadece_uni", u0, None, None, None))

    for mode, tag, u, f, d, s in attempts:
        try:
            sql_try = build_org_series_sql(
                int(year_min), int(year_max),
                u, f, d, s,
                ids,
                join_mode=mode
            )
            params = {"y1": int(year_min), "y2": int(year_max)}
            df_try = run_sql(sql_try, params=params)
            df_try = normalize_columns(df_try, "alan_yillik")
            if df_try is not None and not df_try.empty:
                return df_try, f"{mode}/{tag} ({_lab(u, f, d, s)})"
        except Exception:
            continue

    return pd.DataFrame(columns=["Alan", "Yil", "YayinSayisi"]), ("strict" if strict else "none")

def get_df_from_source(kind: str) -> pd.DataFrame:
    # Her zaman SQL Server'dan çek
    top_n_ext = int(top_n) * 5 if 'top_n' in globals() else 1000
    params = None
    try:
        if kind == "alan_dagilim":
            subjects_csv = _in_list_sql(
                resolve_subject_attrs(int(year_min), int(year_max))
            )
            sql_text = (
                """
                SELECT
                    t.Value AS Alan,
                    COUNT(*) AS YayinSayisi
                FROM (
                    SELECT DISTINCT wa.HitId, wa.Value
                    FROM dbo.WosHitAttributes wa
                    JOIN dbo.WOSHit wh ON wa.HitId = wh.HitId
                    WHERE wa.Name IN (""" + subjects_csv + """)
                      AND wh.SourcePublishYear BETWEEN :y1 AND :y2
                ) AS t
                GROUP BY t.Value
                ORDER BY YayinSayisi DESC;
                """
            )
            params = {"y1": int(year_min), "y2": int(year_max)}
        elif kind == "yazar_yayin":
            sql_text = f"""
                SELECT TOP {top_n_ext} wa.wosStandard AS Yazar, COUNT(*) AS YayinSayisi
                FROM dbo.WosAuthor wa
                JOIN dbo.WOSHit wh ON wa.HitId = wh.HitId
                WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
                GROUP BY wa.wosStandard
                ORDER BY YayinSayisi DESC;
            """
            params = {"y1": int(year_min), "y2": int(year_max)}
        elif kind == "yazar_atif":
            params = {"y1": int(year_min), "y2": int(year_max)}
            cit_col = detect_citation_column()
            if cit_col:
                sql_text = f"""
                    SELECT TOP {top_n_ext}
                        wa.wosStandard AS Yazar,
                        SUM(COALESCE(TRY_CONVERT(int, wh.{cit_col}), 0)) AS ToplamAtif
                    FROM dbo.WosAuthor wa
                    JOIN dbo.WOSHit wh ON wa.HitId = wh.HitId
                    WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
                    GROUP BY wa.wosStandard
                    ORDER BY ToplamAtif DESC;
                """
            else:
                sql_text = f"""
                    SELECT TOP {top_n_ext}
                        wa.wosStandard AS Yazar,
                        CAST(0 AS int) AS ToplamAtif
                    FROM dbo.WosAuthor wa
                    JOIN dbo.WOSHit wh ON wa.HitId = wh.HitId
                    WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
                    GROUP BY wa.wosStandard
                    ORDER BY COUNT(*) DESC;
                """
        elif kind == "alan_yillik":
            subjects_csv = _in_list_sql(
                resolve_subject_attrs(int(year_min), int(year_max))
            )
            sql_text = (
                """
                WITH Y AS (
                    SELECT :y1 AS Yil
                    UNION ALL
                    SELECT Yil + 1 FROM Y WHERE Yil < :y2
                ),
                D AS (
                    SELECT sub.Value AS Alan, wh.SourcePublishYear AS Yil, COUNT(*) AS YayinSayisi
                    FROM dbo.WOSHit wh
                    JOIN dbo.WosHitAttributes sub ON sub.HitId = wh.HitId AND sub.Name IN (""" + subjects_csv + """)
                    WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
                    GROUP BY sub.Value, wh.SourcePublishYear
                ),
                A AS (
                    SELECT DISTINCT Alan FROM D
                )
                SELECT a.Alan, y.Yil, COALESCE(d.YayinSayisi, 0) AS YayinSayisi
                FROM A a
                CROSS JOIN Y y
                LEFT JOIN D d ON d.Alan = a.Alan AND d.Yil = y.Yil
                ORDER BY a.Alan, y.Yil
                OPTION (MAXRECURSION 0);
                """
            )
            params = {"y1": int(year_min), "y2": int(year_max)}
        else:
            sql_text = ""
    except Exception:
        sql_text = ""
    df = run_sql(sql_text, params=params) if sql_text else pd.DataFrame()
    return normalize_columns(df, kind)

# ---- Address-based helpers (WosAddress.full_address) ----
import re

ADDR_SPLIT_RE = re.compile(r"\s*,\s*")

def _addr_parts(addr: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Parse a 'full_address' like 'Cukurova Univ, Faculty of Medicine, Dept X, ...'
    Returns (uni, fac, dept, sub) best-effort.
    """
    if not isinstance(addr, str) or not addr:
        return (None, None, None, None)
    parts = ADDR_SPLIT_RE.split(addr)
    parts = [p.strip() for p in parts if p and p.strip()]
    if not parts:
        return (None, None, None, None)
    uni = parts[0]
    fac = None
    dep = None
    sub = None
    # Heuristics
    for p in parts[1:]:
        pl = p.lower()
        if fac is None and ("fac" in pl or "faculty" in pl or "fak" in pl):
            fac = p
            continue
        if dep is None and ("dept" in pl or "department" in pl or "böl" in pl or "bol" in pl or "abd" in pl):
            dep = p
            continue
        if sub is None:
            sub = p
    return (uni, fac, dep, sub)

@st.cache_data(show_spinner=False)
def addr_distinct(level: str,
                  y1: int, y2: int,
                  sel_uni: str | None = None,
                  sel_fac: str | None = None,
                  sel_dept: str | None = None,
                  limit_uni_raw: str | None = None) -> list[str]:
    """Return distinct address components by parsing WosAddress.full_address for hits in year range.
    level: 'uni' | 'fac' | 'dept' | 'sub'
    Parents may be provided to narrow results.
    limit_uni_raw: Eğer set edilirse, tüm düzeyleri bu üniversite ile sınırla.
    """
    sql = (
        """
        SELECT DISTINCT wa.full_address AS A
        FROM dbo.WosAddress wa
        JOIN dbo.WOSHit wh ON wh.HitId = wa.HitId
        WHERE wh.SourcePublishYear BETWEEN :y1 AND :y2
        """
    )
    # Eğer belirli bir üniversiteye odaklanmak istiyorsak (örn. Çukurova), adres WHERE koşuluna ekleyelim
    if limit_uni_raw and isinstance(limit_uni_raw, str) and limit_uni_raw.strip():
        # Eski: sql = sql + "\n  AND " + _addr_like_clause("wa", limit_uni_raw.strip())
        # Yeni: Sadece belirli varyasyonlar için LIKE ile filtrele
        sql += f"""
          AND (
              wa.full_address COLLATE Turkish_CI_AI LIKE N'%Cukurova Univ%'
              OR wa.full_address COLLATE Turkish_CI_AI LIKE N'%Cukurova University%'
              OR wa.full_address COLLATE Turkish_CI_AI LIKE N'%Çukurova Üniversitesi%'
          )
        """
    df = run_sql(sql, params={"y1": int(y1), "y2": int(y2)})
    if df is None or df.empty or "A" not in df.columns:
        return []
    rows = df["A"].dropna().astype(str).tolist()
    out: list[str] = []
    for a in rows:
        u, f, d, s = _addr_parts(a)
        # Eğer belirli bir üniversiteye odaklanıldıysa, yalnızca ilk parçası (üniversite) bu filtreyle uyuşan adresleri kabul et
        in_scope = True
        if limit_uni_raw and isinstance(limit_uni_raw, str) and limit_uni_raw.strip():
            lim = _strip_tr(limit_uni_raw).lower()
            uu = _strip_tr(u or "").lower()
            in_scope = (lim in uu) if uu else False
        if not in_scope:
            continue
        if level == "uni" and u:
            out.append(u)
        elif level == "fac" and f and (sel_uni is None or (u and u == sel_uni)):
            out.append(f)
        elif level == "dept" and d and (sel_uni is None or (u and u == sel_uni)) and (sel_fac is None or (f and f == sel_fac)):
            out.append(d)
        elif level == "sub" and s and (sel_uni is None or (u and u == sel_uni)) and (sel_fac is None or (f and f == sel_fac)) and (sel_dept is None or (d and d == sel_dept)):
            out.append(s)
    # unique + sorted
    uniq = sorted({x.strip(): None for x in out}.keys())
    return uniq

# ----------------- Sekmeler -----------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Alan dağılımı", "Yazar (yayın)", "Yazar (atıf)", "Alan yıllık seri", "Tahmin"
])

if ran:
    try:
        # 1) Alan dağılımı
        df_alan = get_df_from_source("alan_dagilim")
        with tab1:
            st.subheader("Alanlara göre yayın sayısı")
            if df_alan is not None and not df_alan.empty and {"Alan", "YayinSayisi"}.issubset(df_alan.columns):
                st.dataframe(df_alan.head(200), use_container_width=True)
                df_top = df_alan.head(int(top_n))
                chart_kind1 = st.selectbox(
                    "Grafik türü", ["Yatay çubuk", "Dikey çubuk", "Pasta"], index=0, key="chart_t1")
                fig = make_chart(df_top, "Alan", "YayinSayisi", chart_kind1,
                                 "En Fazla Yayın Yapılan Alanlar", "Alan", "Yayın Sayısı")
                st.pyplot(fig, use_container_width=True)
                buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
                st.download_button("📥 Grafiği PNG olarak indir", buf,
                                   "alan_dagilimi.png", "image/png", key="alan_png")
                st.download_button("📥 Sonuç (CSV)", df_alan.to_csv(index=False),
                                   "alan_dagilimi.csv", "text/csv", key="alan_csv")
            else:
                st.info("Beklenen kolonlar bulunamadı veya veri boş.")

        # 2) En çok yayın yapan yazarlar
        df_yazar_yayin = get_df_from_source("yazar_yayin")
        with tab2:
            st.subheader("En çok yayın yapan yazarlar")
            if df_yazar_yayin is not None and not df_yazar_yayin.empty and {"Yazar", "YayinSayisi"}.issubset(df_yazar_yayin.columns):
                st.dataframe(df_yazar_yayin.head(200), use_container_width=True)
                df_top = df_yazar_yayin.head(int(top_n))
                chart_kind2 = st.selectbox(
                    "Grafik türü", ["Yatay çubuk", "Dikey çubuk", "Pasta"], index=0, key="chart_t2")
                fig2 = make_chart(df_top, "Yazar", "YayinSayisi", chart_kind2,
                                   "En Çok Yayın Yapan Yazarlar", "Yazar", "Yayın Sayısı")
                st.pyplot(fig2, use_container_width=True)
                buf = io.BytesIO(); fig2.savefig(buf, format="png"); buf.seek(0)
                st.download_button("📥 Grafiği PNG olarak indir", buf,
                                   "yazar_yayin.png", "image/png", key="yazar_yayin_png")
                st.download_button("📥 Sonuç (CSV)", df_yazar_yayin.to_csv(index=False),
                                   "yazar_yayin.csv", "text/csv", key="yazar_yayin_csv")
            else:
                st.info("Beklenen kolonlar bulunamadı veya veri boş.")

        # 3) En çok atıf alan yazarlar
        df_yazar_atif = get_df_from_source("yazar_atif")
        with tab3:
            st.subheader("En çok atıf alan yazarlar")
            _cit = detect_citation_column()
            if _cit is None:
                st.caption("ℹ️ Bu veritabanında **atıf (TimesCited)** kolonu bulunamadı; bu bölüm yayın sayısına göre 0 atıfla gösterilir.")
            if df_yazar_atif is not None and not df_yazar_atif.empty and {"Yazar", "ToplamAtif"}.issubset(df_yazar_atif.columns):
                st.dataframe(df_yazar_atif.head(200), use_container_width=True)
                df_top = df_yazar_atif.head(int(top_n))
                chart_kind3 = st.selectbox(
                    "Grafik türü", ["Yatay çubuk", "Dikey çubuk", "Pasta"], index=0, key="chart_t3")
                fig3 = make_chart(df_top, "Yazar", "ToplamAtif", chart_kind3,
                                   "En Çok Atıf Alan Yazarlar", "Yazar", "Toplam Atıf")
                st.pyplot(fig3, use_container_width=True)
                buf = io.BytesIO(); fig3.savefig(buf, format="png"); buf.seek(0)
                st.download_button("📥 Grafiği PNG olarak indir", buf,
                                   "yazar_atif.png", "image/png", key="yazar_atif_png")
                st.download_button("📥 Sonuç (CSV)", df_yazar_atif.to_csv(index=False),
                                   "yazar_atif.csv", "text/csv", key="yazar_atif_csv")
            else:
                st.info("Beklenen kolonlar bulunamadı veya veri boş.")

        # 4) Alan bazlı yıllık seri
        df_alan_yillik = get_df_from_source("alan_yillik")
        # Boşsa bir kez önbellek temizleyip yeniden dene
        if (df_alan_yillik is None) or df_alan_yillik.empty:
            if not st.session_state.get("_auto_rebuilt_series", False):
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                st.session_state["_auto_rebuilt_series"] = True
                df_alan_yillik = get_df_from_source("alan_yillik")
                if (df_alan_yillik is None) or df_alan_yillik.empty:
                    st.caption("ℹ️ Yıllık seri verisi henüz gelmedi; seçili filtrelerde sonuç olmayabilir.")

        with tab4:
            st.subheader("Alan bazlı yıllık yayın sayısı")
            if df_alan_yillik is not None and not df_alan_yillik.empty and {"Alan", "Yil", "YayinSayisi"}.issubset(df_alan_yillik.columns):
                st.dataframe(df_alan_yillik.head(200), use_container_width=True)
                alanlar = sorted(df_alan_yillik["Alan"].dropna().unique().tolist())
                secilen_alan = st.selectbox("Grafiğini çiz:", alanlar)
                if secilen_alan:
                    tmp = df_alan_yillik[df_alan_yillik["Alan"] == secilen_alan]
                    df_one = tmp.sort_values(by="Yil")  # type: ignore[arg-type]
                    chart_kind4 = st.selectbox(
                        "Grafik türü", ["Çizgi", "Dikey çubuk", "Alan"], index=0, key="chart_t4")
                    fig4 = make_chart(df_one, "Yil", "YayinSayisi", chart_kind4,
                                      f"{secilen_alan} — Yıllık Yayın Sayısı", "Yıl", "Yayın")
                    st.pyplot(fig4, use_container_width=True)
                    buf = io.BytesIO(); fig4.savefig(buf, format="png"); buf.seek(0)
                    st.download_button("📥 Grafiği PNG olarak indir", buf,
                                       f"alan_yillik_{secilen_alan}.png", "image/png", key="alan_yillik_png")
            else:
                st.info("Beklenen kolonlar bulunamadı veya veri boş.")

        # 5) Tahmin (yeni: sadece alan seçimi, kurumsal filtre yok)
        with tab5:
            # --- Yeni Tahmin: Bölüm/Fakülte/Birim seçimleri yok ---
            st.subheader(f"📈 Alan bazlı tahmin — hedef yıl: {int(target_year)}")
            st.caption("Kullanıcı tüm alanlar listesinden bir alan seçer; sadece o alan için tahmin yapılır.")

            # 1) Tüm alanların listesi (df_alan_yillik'ten)
            if df_alan_yillik is None or df_alan_yillik.empty or not {"Alan","Yil","YayinSayisi"}.issubset(df_alan_yillik.columns):
                st.warning("Yıllık seri verisi yok veya beklenen kolonlar bulunamadı. Önce 'Alan yıllık seri' verisi üretildiğinden emin olun.")
            else:
                alan_list = (
                    sorted(df_alan_yillik["Alan"].dropna().astype(str).unique().tolist())
                )
                if not alan_list:
                    st.info("Listelenecek alan bulunamadı.")
                else:
                    sec_alan = st.selectbox(
                        "Alan seçin veya yazmaya başlayın",
                        options=alan_list,
                        index=0,
                        key="tahmin_alan_select",
                        help="Listenin içinde yazdıkça otomatik filtreleme yapılır.",
                        placeholder="Örn: phys, comp, bio..."
                    )

                    # 2) Seçilen alan için tek seri
                    ser = (
                        df_alan_yillik[df_alan_yillik["Alan"].astype(str) == str(sec_alan)]
                        .copy()
                        .sort_values(by="Yil")
                    )
                    if ser.empty:
                        st.info("Seçilen alan için seri bulunamadı.")
                    else:
                        # Temel istatistikler ve hedef yıl doğrulaması
                        years_series = pd.to_numeric(ser["Yil"], errors="coerce").dropna().astype(int)
                        vals_series  = pd.to_numeric(ser["YayinSayisi"], errors="coerce").fillna(0).astype(float)
                        if years_series.empty:
                            st.info("Bu alan için yıl bilgisi bulunamadı.")
                            st.stop()

                        # NumPy dizilerine dönüştür ve aralık sınırlarını al
                        years = years_series.to_numpy()
                        vals  = vals_series.to_numpy()
                        min_year = int(years.min())
                        max_year = int(years.max())

                        # Eğer hedef yıl eğitim aralığındaysa (in-sample), gerçek değeri al
                        actual_value = None
                        try:
                            mask_in_sample = (years_series == int(target_year))
                            if bool(mask_in_sample.any()):
                                actual_value = float(vals_series[mask_in_sample].iloc[0])
                                st.caption(f"📌 Seçtiğiniz yıl **{int(target_year)}** mevcut veri aralığında; bu bir **in‑sample** tahmindir (gerçekle aynı olabilir).")
                        except Exception:
                            actual_value = None

                        if len(years) < 2:
                            st.info("Bu alan için en az 2 yıllık veri gerekli (LM için). Daha uzun seri ile daha sağlıklı tahmin yapılır.")
                            st.stop()

                        # 3) Tahminler — LM zorunlu, Prophet isteğe bağlı
                        y_lm = float("nan")
                        if len(years) >= 2:
                            try:
                                y_lm = simple_lm_forecast(years, vals, int(target_year))
                            except Exception:
                                y_lm = float("nan")

                        y_prop = None
                        if st.session_state.get("use_prophet", False):
                            y_prop = _prophet_forecast_safe(years, vals, int(target_year))

                        y_suggested = _blend_forecast(y_lm, y_prop)
                        base_for_ci = (
                            y_suggested if (y_suggested is not None and not pd.isna(y_suggested)) else
                            (y_lm if not pd.isna(y_lm) else 0.0)
                        )
                        lo, hi = _volatility_ci(vals, float(base_for_ci))

                        # 4) Özet kartlar
                        c1, c2, c3 = st.columns(3)
                        c1.metric(f"LM @ {int(target_year)}", ("—" if pd.isna(y_lm) else f"{y_lm:.0f}"))
                        c2.metric(f"Prophet @ {int(target_year)}", ("None" if (y_prop is None or pd.isna(y_prop)) else f"{y_prop:.0f}"))
                        c3.metric("Önerilen", ("—" if (y_suggested is None or pd.isna(y_suggested)) else f"{y_suggested:.0f}"))
                        st.caption(f"Yaklaşık güven aralığı: {lo:.0f} – {hi:.0f}")
                        # Uyarı: veri seyrekliği / yüksek oynaklık
                        try:
                            total_years = len(years)
                            zero_years = int((vals == 0).sum()) if isinstance(vals, np.ndarray) else 0
                            coverage_ratio = total_years / max(1, (max_year - min_year + 1))  # 0–1
                            zero_ratio = zero_years / max(1, total_years)
                            # Değişkenlik: bir önceki yıla göre yüzde değişimlerin std'si
                            with np.errstate(divide="ignore", invalid="ignore"):
                                pct_changes = np.diff(vals) / np.where(vals[:-1] == 0, np.nan, vals[:-1]) if total_years >= 2 else np.array([])
                            sigma = float(np.nanstd(pct_changes)) if pct_changes.size else 0.0

                            sparse_flag = (total_years < 6) or (coverage_ratio < 0.6) or (zero_ratio >= 0.4)
                            volatile_flag = sigma >= 0.6  # oldukça oynak seri

                            if sparse_flag or volatile_flag:
                                note_parts = []
                                if sparse_flag:
                                    note_parts.append("veri seyrek (uzun boşluklar/çok az yıl)")
                                if volatile_flag:
                                    note_parts.append("seri çok oynak")
                                st.warning("⚠️ Bu alan için " + " ve ".join(note_parts) + "; tahminler **yüksek belirsizlik** içerir.")
                        except Exception:
                            pass

                        # 5) Grafik (tek alan) — tahmin noktası + CI hata çubuğu ile
                        # Not: make_chart yerine özel çizim, çünkü hedef yıl noktasına CI ekliyoruz
                        try:
                            # Temel seri grafiği
                            fig_sel, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(ser["Yil"], ser["YayinSayisi"], marker="o")
                            ax.set_title(f"{sec_alan} — Yıllık Yayın Sayısı")
                            ax.set_xlabel("Yıl")
                            ax.set_ylabel("Yayın Sayısı")

                            # Y ekseni: negatifleri engelle, tam sayı tick'leri kullan
                            ax.set_ylim(bottom=0)
                            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                            # Son gözlem yılını işaretle
                            try:
                                last_year = int(ser["Yil"].iloc[-1])
                                last_val = float(ser["YayinSayisi"].iloc[-1])
                                ax.scatter([last_year], [last_val], s=35, color="blue", zorder=3)
                                ax.annotate("Son veri", xy=(last_year, last_val),
                                            xytext=(5, -12), textcoords="offset points")
                            except Exception:
                                pass

                            # LM / Prophet / Önerilen noktalarını ayrı işaretle
                            try:
                                if not pd.isna(y_lm):
                                    ax.scatter([int(target_year)], [float(y_lm)], marker="s", s=45, label="LM", zorder=3)
                                if (y_prop is not None) and (not pd.isna(y_prop)):
                                    ax.scatter([int(target_year)], [float(y_prop)], marker="^", s=55, label="Prophet", zorder=3)
                                if (y_suggested is not None) and (not pd.isna(y_suggested)):
                                    ax.scatter([int(target_year)], [float(y_suggested)], s=65, label="Önerilen", zorder=4)
                                if ax.get_legend_handles_labels()[0]:
                                    ax.legend()
                            except Exception:
                                pass

                            # X eksenini hedef yılı kapsayacak şekilde genişlet
                            try:
                                xmin = int(min(min_year, int(target_year)))
                                xmax = int(max(max_year, int(target_year)))
                                ax.set_xlim(xmin, xmax)
                            except Exception:
                                pass

                            # Tahmin noktası ve CI (varsa) — tek bir hata çubuğu
                            if (y_suggested is not None) and (not pd.isna(y_suggested)):
                                y0 = float(y_suggested)
                                # errorbar için simetrik olmayan yerr kullanıyoruz
                                yerr_low  = max(0.0, y0 - float(lo))
                                yerr_high = max(0.0, float(hi) - y0)
                                ax.errorbar(
                                    [int(target_year)], [y0],
                                    yerr=[[yerr_low], [yerr_high]],
                                    fmt='o', capsize=6, linewidth=1.5
                                )
                                # Noktanın üstüne küçük bir etiket
                                ax.annotate(
                                    f"Tahmin: {y0:.0f}",
                                    xy=(int(target_year), y0),
                                    xytext=(5, 8), textcoords='offset points'
                                )

                            fig_sel.tight_layout()
                            st.pyplot(fig_sel, use_container_width=True)
                        except Exception:
                            # Her ihtimale karşı eski basit grafiğe geri dön
                            fig_sel = make_chart(ser, "Yil", "YayinSayisi", "Çizgi",
                                                 f"{sec_alan} — Yıllık Yayın Sayısı", "Yıl", "Yayın Sayısı")
                            st.pyplot(fig_sel, use_container_width=True)

                        # Eğer seçilen yıl geçmişteyse ve gerçek değer varsa, hatayı göster
                        if actual_value is not None:
                            err_txt = ""
                            if (y_suggested is not None) and (not pd.isna(y_suggested)):
                                try:
                                    err_val = abs(float(y_suggested) - float(actual_value))
                                    err_txt = f" — Tahmin hatası: ±{err_val:.0f}"
                                except Exception:
                                    err_txt = ""
                            st.caption(f"📌 Gerçek {int(target_year)}: {actual_value:.0f}{err_txt}")
                        elif actual_value is None and int(target_year) <= max_year:
                            # Geçmiş yıl seçildi ama seride o yıl yoksa bilgilendir
                            st.caption("📌 Seçtiğiniz yıl veri aralığında olabilir; bu çıktı geri-tahmin (hindcast) amaçlıdır.")

                        # 6) İndirme: sadece seçilen alanın serisi ve tek satır tahmin
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                "📥 Seri (CSV)", ser.to_csv(index=False), f"seri_{_strip_tr(str(sec_alan))}.csv", "text/csv", key="seri_csv"
                            )
                        with col_dl2:
                            df_pred_row = pd.DataFrame([
                                {
                                    "Alan": sec_alan,
                                    "Yil": int(target_year),
                                    "LM": (None if pd.isna(y_lm) else float(y_lm)),
                                    "Prophet": (None if (y_prop is None or pd.isna(y_prop)) else float(y_prop)),
                                    "Onerilen": (None if (y_suggested is None or pd.isna(y_suggested)) else float(y_suggested)),
                                    "CI_low": float(lo),
                                    "CI_high": float(hi),
                                }
                            ])
                            st.download_button(
                                "📥 Tahmin (CSV)", df_pred_row.to_csv(index=False), f"tahmin_{_strip_tr(str(sec_alan))}_{int(target_year)}.csv", "text/csv", key="tahmin_csv_single"
                            )

        # 6) Kümeleme (opsiyonel)

    except Exception as ex:
        st.error(f"Beklenmeyen hata: {ex}")
else:
    st.info("Çalıştırmak için ‘Çalıştır’ düğmesine basın veya One-Click modu aktifse ekran otomatik yenilenir.")