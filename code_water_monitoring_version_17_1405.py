"""
Water Quality Monitoring Application — Version 18
=====================================
Sentinel-2 based automatic calculation of:
  - Water Turbidity (NDTI)
  - Chlorophyll-a Concentration
  - Colored Dissolved Organic Matter (CDOM)

All three parameters share the same preprocessing pipeline (cloud filtering,
snow masking, waterbody extraction) and are computed automatically and
sequentially after the user selects an area of interest. The interface is
designed for managers and non-technical decision-makers: no remote-sensing
jargon, no parameter pickers, no processing logs.

Version 18 repairs interrupted downloads, cache recovery, authentication retries
and resume controls. Embedded Persian typography, parameter-specific display
thresholds, the coverage view and validated atomic cache writes are retained.
Scientific equations, geospatial events and exports are preserved. Expert chat
streams responses with a longer model timeout and bounded weather research.
Launch in the existing environment with:
    streamlit run code_water_monitoring_version_18_1405.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Polygon
import rasterio
import datetime
import math
import ee
import tempfile
import requests
import time
import warnings
import base64
import json
import inspect
import hashlib
import shutil
from datetime import date
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="سامانه پایش کیفیت آب | نسخه ۱۸",
    page_icon="🌊"
)

import folium
from folium import plugins
from streamlit_folium import st_folium

# branca + jinja2 are folium's own dependencies; they are used to inject the
# small piece of JavaScript that shows the live area while a region is being
# drawn. If they are ever unavailable the app simply runs without that label.
try:
    from branca.element import MacroElement as _BrancaMacroElement
    from jinja2 import Template as _JinjaTemplate
    _LIVE_AREA_SUPPORTED = True
except Exception:
    _LIVE_AREA_SUPPORTED = False

# =============================================================================
# CONSTANTS
# =============================================================================
# --- Fixed processing thresholds (no longer user-configurable) -------------
CLOUD_THRESHOLD = 10          # % — fixed cloud coverage threshold (CLOUDY_PIXEL_PERCENTAGE)
CLOUD_PROB_THRESHOLD = 15      # per-pixel cloud probability cutoff

# Water body detection threshold
# AWEIsh (Automated Water Extraction Index, shadow variant) — shared by both
# indices so NDTI and NDCI are computed on the exact same water mask.
# AWEIsh = Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2
AWEI_THRESHOLD = 0.05

# Snow detection thresholds (preprocessing only — excludes snow from water)
NDSI_THRESHOLD = 0.39
SNOW_B11_THRESHOLD = 0.1  # kept for reference; no longer used by is_snow (see below)

# MODIS-heritage water/snow discrimination test (Hall et al., 1995; Riggs et al.),
# translated to Sentinel-2 bands. Water absorbs NIR almost completely regardless
# of turbidity, while snow reflects strongly there, so this is a more robust way
# to keep turbid/sediment-laden water from being flagged as snow than the SWIR
# threshold alone.
NIR_SNOW_THRESHOLD = 0.11    # B8 (NIR) — MODIS band 2 analogue
GREEN_SNOW_THRESHOLD = 0.1   # B3 (Green) — MODIS band 4 analogue

# Parameter identifiers (internal use only — never shown as a user choice)
PARAM_TURBIDITY = "Turbidity (NDTI)"
PARAM_CHLOROPHYLL = "Chlorophyll Index"
PARAM_CDOM = "CDOM"

# Every parameter the app monitors, in the order they are processed and shown.
ALL_PARAMETERS = [PARAM_TURBIDITY, PARAM_CHLOROPHYLL, PARAM_CDOM]

# Chlorophyll visualization range
CHL_VMIN = -1.0
CHL_VMAX = 0.9

# CDOM (Colored Dissolved Organic Matter) visualization range — fallback only.
# Unlike NDTI/NDCI, CDOM is not a normalized-difference index bounded to
# [-1, 1]: it is an absorption coefficient whose plausible range differs a lot
# between water bodies. The app therefore derives the display range from the
# actual data of the run (2nd-98th percentile across all months, so colours stay
# comparable month to month) and only falls back to these constants when that is
# not possible.
CDOM_VMIN = 0.0
CDOM_VMAX = 30.0


def empty_param_dict(factory=dict):
    """A fresh {parameter: empty container} map for every monitored parameter."""
    return {p: factory() for p in ALL_PARAMETERS}


def param_short_name(parameter_type):
    """Short technical label. Internal use only (file names, Excel export)."""
    if parameter_type == PARAM_TURBIDITY:
        return "NDTI"
    if parameter_type == PARAM_CHLOROPHYLL:
        return "Chl-a"
    return "CDOM"


# Persian names of the GREGORIAN months — the app's date range is Gregorian,
# these are only the Persian spellings of the Gregorian month names.
GREGORIAN_MONTHS_FA = [
    "ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن",
    "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر",
]


def param_persian_name(parameter_type):
    """
    Persian name shown to the user. Everything the end user reads on a
    parameter page — statistics summary, monthly table, image captions and
    chart labels — uses this instead of the English abbreviations.
    """
    if parameter_type == PARAM_TURBIDITY:
        return "کدورت"
    if parameter_type == PARAM_CHLOROPHYLL:
        return "کلروفیل"
    return "مواد آلی محلول"


def param_decimals(parameter_type):
    """How many decimals to show for this parameter's values."""
    return 4 if parameter_type == PARAM_TURBIDITY else 2


def format_param_value(parameter_type, value, empty="—"):
    """Format one mean value for display, honouring the parameter's precision."""
    try:
        if value is None or np.isnan(value):
            return empty
    except TypeError:
        return empty
    return f"{value:.{param_decimals(parameter_type)}f}"


# Display-only condition rules. These never enter the satellite pipeline,
# raster colour maps, exports, statistical expert analysis or AI prompts.
WQ_DISPLAY_CONDITIONS = {
    'normal': {'label': 'نرمال', 'accent': '#15803D', 'background': '#F0FDF4',
               'border': '#BBDFC8', 'symbol': '●'},
    'warning': {'label': 'هشدار', 'accent': '#A16207', 'background': '#FEFCE8',
                'border': '#E9D88A', 'symbol': '■'},
    'dangerous': {'label': 'خطرناک', 'accent': '#C2410C', 'background': '#FFF7ED',
                  'border': '#F3C8A7', 'symbol': '▲'},
    'critical': {'label': 'بحرانی', 'accent': '#B91C1C', 'background': '#FEF2F2',
                 'border': '#F0C4C4', 'symbol': '◆'},
    'unavailable': {'label': 'داده‌ای در دسترس نیست', 'accent': '#52647B',
                    'background': '#F5F7FA', 'border': '#DCE3EB', 'symbol': '—'},
}

# Classification is a UI rule for the selected parameter, never a raster/model
# transformation. CDOM intentionally has no classification thresholds.
WQ_DISPLAY_THRESHOLDS = {
    PARAM_TURBIDITY: (-0.3, 0.0, 0.3),
    PARAM_CHLOROPHYLL: (-0.3, 0.0, 0.3),
}
WQ_CHART_FONT = 'WQ Vazirmatn'


# Unmodified Vazirmatn v33.003 variable font, embedded for offline/Posit use.
# Source: https://github.com/rastikerdar/vazirmatn/tree/v33.003
# The complete copyright and redistribution license accompany the font below.
WQ_VAZIRMATN_OFL = 'Copyright 2015 The Vazirmatn Project Authors (https://github.com/rastikerdar/vazirmatn)\n\nThis Font Software is licensed under the SIL Open Font License, Version 1.1.\nThis license is copied below, and is also available with a FAQ at:\nhttp://scripts.sil.org/OFL\n\n\n-----------------------------------------------------------\nSIL OPEN FONT LICENSE Version 1.1 - 26 February 2007\n-----------------------------------------------------------\n\nPREAMBLE\nThe goals of the Open Font License (OFL) are to stimulate worldwide\ndevelopment of collaborative font projects, to support the font creation\nefforts of academic and linguistic communities, and to provide a free and\nopen framework in which fonts may be shared and improved in partnership\nwith others.\n\nThe OFL allows the licensed fonts to be used, studied, modified and\nredistributed freely as long as they are not sold by themselves. The\nfonts, including any derivative works, can be bundled, embedded, \nredistributed and/or sold with any software provided that any reserved\nnames are not used by derivative works. The fonts and derivatives,\nhowever, cannot be released under any other type of license. The\nrequirement for fonts to remain under this license does not apply\nto any document created using the fonts or their derivatives.\n\nDEFINITIONS\n"Font Software" refers to the set of files released by the Copyright\nHolder(s) under this license and clearly marked as such. This may\ninclude source files, build scripts and documentation.\n\n"Reserved Font Name" refers to any names specified as such after the\ncopyright statement(s).\n\n"Original Version" refers to the collection of Font Software components as\ndistributed by the Copyright Holder(s).\n\n"Modified Version" refers to any derivative made by adding to, deleting,\nor substituting -- in part or in whole -- any of the components of the\nOriginal Version, by changing formats or by porting the Font Software to a\nnew environment.\n\n"Author" refers to any designer, engineer, programmer, technical\nwriter or other person who contributed to the Font Software.\n\nPERMISSION & CONDITIONS\nPermission is hereby granted, free of charge, to any person obtaining\na copy of the Font Software, to use, study, copy, merge, embed, modify,\nredistribute, and sell modified and unmodified copies of the Font\nSoftware, subject to the following conditions:\n\n1) Neither the Font Software nor any of its individual components,\nin Original or Modified Versions, may be sold by itself.\n\n2) Original or Modified Versions of the Font Software may be bundled,\nredistributed and/or sold with any software, provided that each copy\ncontains the above copyright notice and this license. These can be\nincluded either as stand-alone text files, human-readable headers or\nin the appropriate machine-readable metadata fields within text or\nbinary files as long as those fields can be easily viewed by the user.\n\n3) No Modified Version of the Font Software may use the Reserved Font\nName(s) unless explicit written permission is granted by the corresponding\nCopyright Holder. This restriction only applies to the primary font name as\npresented to the users.\n\n4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font\nSoftware shall not be used to promote, endorse or advertise any\nModified Version, except to acknowledge the contribution(s) of the\nCopyright Holder(s) and the Author(s) or with their explicit written\npermission.\n\n5) The Font Software, modified or unmodified, in part or in whole,\nmust be distributed entirely under this license, and must not be\ndistributed under any other license. The requirement for fonts to\nremain under this license does not apply to any document created\nusing the Font Software.\n\nTERMINATION\nThis license becomes null and void if any of the above conditions are\nnot met.\n\nDISCLAIMER\nTHE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,\nEXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF\nMERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT\nOF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE\nCOPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,\nINCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL\nDAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING\nFROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM\nOTHER DEALINGS IN THE FONT SOFTWARE.\n'
WQ_VAZIRMATN_WOFF2_BASE64 = (
    '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'
    'AAA='
)


def _embedded_persian_font_css():
    """A self-contained Persian webfont; no browser/CDN font request needed."""
    return (
        "@font-face {font-family:'WQ Vazirmatn';"
        "src:url(data:font/woff2;base64," + WQ_VAZIRMATN_WOFF2_BASE64 + ") format('woff2');"
        "font-weight:100 900;font-style:normal;font-display:swap;}"
    )


def _quality_display_thresholds(parameter_type):
    """Return its normal/warning/danger upper bounds, or None for CDOM."""
    return WQ_DISPLAY_THRESHOLDS.get(parameter_type)


def _finite_monthly_value(value):
    """Read an existing observation for display; missing is never zero."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _water_quality_condition(value, parameter_type=PARAM_TURBIDITY):
    """Four display stages; each boundary belongs to the lower-severity stage."""
    thresholds = _quality_display_thresholds(parameter_type)
    number = _finite_monthly_value(value)
    if number is None or thresholds is None:
        return 'unavailable'
    normal_max, warning_max, dangerous_max = thresholds
    if number <= normal_max:
        return 'normal'
    if number <= warning_max:
        return 'warning'
    if number <= dangerous_max:
        return 'dangerous'
    return 'critical'


def _overall_water_quality_condition(results, parameter_type):
    """Unweighted arithmetic mean of finite monthly observations, including 0."""
    if parameter_type not in (PARAM_TURBIDITY, PARAM_CHLOROPHYLL):
        return None, 'unavailable'
    from decimal import Decimal, localcontext

    values = [_finite_monthly_value(r.get('mean_value')) for r in results]
    values = [value for value in values if value is not None]
    if not values:
        return None, 'unavailable'
    # Arithmetic on round-trip decimal representations avoids summation drift
    # at display boundaries (e.g. the mean of -0.4 and -0.2 is -0.3).
    # No observation is rounded, normalized, weighted or written back to data.
    with localcontext() as context:
        context.prec = 40
        overall_mean = float(sum(Decimal(str(value)) for value in values) / len(values))
    return overall_mean, _water_quality_condition(overall_mean, parameter_type)


def _monthly_quality_records(results, parameter_type):
    """JSON-safe, quality-only view of the existing rows in their original order."""
    classified = parameter_type in (PARAM_TURBIDITY, PARAM_CHLOROPHYLL)
    records = []
    segment = 0
    for sequence, result in enumerate(results):
        month = str(result.get('month_name', '—'))
        try:
            month_date = datetime.datetime.strptime(month, '%Y-%m').strftime('%Y-%m-01')
        except ValueError:
            month_date = None
        value = _finite_monthly_value(result.get('mean_value'))
        if value is None:
            segment += 1
        condition = _water_quality_condition(value, parameter_type) if classified else None
        records.append({
            'month': month, 'date': month_date, 'sequence': sequence, 'segment': segment,
            'value': value, 'condition': condition,
            'condition_label': WQ_DISPLAY_CONDITIONS[condition]['label'] if classified else '',
            'display_value': format_param_value(parameter_type, value, empty='بدون داده'),
        })
    return records

# -----------------------------------------------------------------------------
# Persian UI font (B Nazanin)
# -----------------------------------------------------------------------------
# "B Nazanin" is not a free web font, so it only renders if it happens to be
# installed on the viewer's machine. To make the app look identical for every
# user, drop a copy of the font file next to the app (any ONE of the paths
# below) and it will be embedded directly into the page as a base64 @font-face
# rule — no web server configuration needed.
#
#   Recommended:  assets/BNazanin.woff2      (smallest / fastest)
#   Also fine:    assets/BNazanin.ttf  |  BNazanin.ttf  |  fonts/BNazanin.ttf
#
# If no file is found, the app silently falls back to Vazirmatn (loaded from
# Google Fonts), exactly like the previous version — nothing breaks.
BNAZANIN_FONT_CANDIDATES = [
    "assets/BNazanin.woff2",
    "assets/BNazanin.woff",
    "assets/BNazanin.ttf",
    "assets/B Nazanin.ttf",
    "assets/B-Nazanin.ttf",
    "fonts/BNazanin.woff2",
    "fonts/BNazanin.ttf",
    "BNazanin.woff2",
    "BNazanin.ttf",
    "B Nazanin.ttf",
]

# Optional: if you prefer to host the font on Google Drive (like the app logo),
# put the shareable file ID here and leave the local files out. The file must be
# shared as "Anyone with the link -> Viewer".
BNAZANIN_FONT_DRIVE_ID = ""

# Download settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = (10, 30)  # connect/read seconds; EE RPCs have their own deadline
EE_RPC_TIMEOUT_MS = 30000
EE_RPC_MAX_RETRIES = 0
CHUNK_SIZE = 8192
MIN_FILE_SIZE = 10000

# --- Connection-loss handling -------------------------------------------------
# Retry transient failures with bounded requests, then pause at the failed
# month. Deterministic data/size/permission errors retain their actual cause.
NETWORK_FAILURE_STREAK = 2

# --- Session-independent download cache ---------------------------------------
# The downloaded GeoTIFFs live in a folder whose name is derived from the run
# itself (region + dates + settings), not from a random tempfile name. That way
# the exact same run always maps to the same folder and can be resumed even if
# the Streamlit session was lost entirely (dropped websocket, page reload).
CACHE_ROOT_NAME = "wq_monitor_cache"
RUN_CONFIG_FILE = "run_config.json"
RUN_STATE_FILE = "run_state.json"
CACHE_MAX_AGE_DAYS = 14

# Status constants
STATUS_NO_DATA = "no_data"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# =============================================================================
# Session State Initialization
# =============================================================================
if 'drawn_polygons' not in st.session_state:
    st.session_state.drawn_polygons = []
if 'last_drawn_polygon' not in st.session_state:
    st.session_state.last_drawn_polygon = None
if 'ee_initialized' not in st.session_state:
    st.session_state.ee_initialized = False
if 'current_temp_dir' not in st.session_state:
    st.session_state.current_temp_dir = None
if 'downloaded_months' not in st.session_state:
    # nested by parameter: {PARAM_TURBIDITY: {...}, PARAM_CHLOROPHYLL: {...}}
    st.session_state.downloaded_months = empty_param_dict(dict)
if 'month_statuses' not in st.session_state:
    st.session_state.month_statuses = empty_param_dict(dict)
if 'results' not in st.session_state:
    # nested by parameter
    st.session_state.results = empty_param_dict(list)
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'selected_region_index' not in st.session_state:
    st.session_state.selected_region_index = 0
if 'processing_in_progress' not in st.session_state:
    st.session_state.processing_in_progress = False
if 'processing_config' not in st.session_state:
    st.session_state.processing_config = None
if 'mean_data' not in st.session_state:
    st.session_state.mean_data = empty_param_dict(dict)
if 'download_summary' not in st.session_state:
    # simple end-user facing summary: {PARAM_TURBIDITY: (downloaded, available), ...}
    st.session_state.download_summary = {}
if 'resume_after_interruption' not in st.session_state:
    # True when a previous run was interrupted and can be resumed
    st.session_state.resume_after_interruption = False
if 'connection_interrupted' not in st.session_state:
    # Set by the processing loop when the link to Earth Engine drops, so the
    # run stops at that point instead of failing every remaining month
    st.session_state.connection_interrupted = False
if 'cdom_display_range' not in st.session_state:
    # (vmin, vmax) colour range derived from the CDOM data of the current run
    st.session_state.cdom_display_range = None
if 'map_version' not in st.session_state:
    # Bumped whenever a region is saved or deleted. It is part of the map
    # widget's key, so the widget is remounted and drops the shape still held
    # by its own drawing toolbar — otherwise a deleted region would be sent
    # back by the widget and reappear on the map.
    st.session_state.map_version = 0
if 'pending_run' not in st.session_state:
    # 'start' | 'resume' | None — a click arms the run, the next script run
    # executes it (so the page is fully rendered before processing begins)
    st.session_state.pending_run = None
if 'last_run_error' not in st.session_state:
    st.session_state.last_run_error = None
if 'active_page' not in st.session_state:
    # Which of the four top-navigation pages is currently shown:
    # 'setup' | 'turbidity' | 'chlorophyll' | 'chat'
    st.session_state.active_page = 'setup'


# =============================================================================
# Earth Engine Authentication
# =============================================================================
def initialize_earth_engine():
    """Initialize a bounded EE transport; failed attempts must remain retryable."""
    try:
        import google.auth
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google.oauth2.service_account import Credentials as ServiceCredentials
        import httplib2

        timeout_seconds = EE_RPC_TIMEOUT_MS / 1000.0

        class BoundedAuthRequest(Request):
            def __call__(self, url, method='GET', body=None, headers=None,
                         timeout=None, **kwargs):
                # EE's setDeadline does not cover OAuth token refresh. Bound
                # that separate transport too, including an expired saved token.
                limit = min(timeout, timeout_seconds) if isinstance(timeout, (int, float)) and timeout > 0 else timeout_seconds
                return super().__call__(url, method=method, body=body,
                                        headers=headers, timeout=limit, **kwargs)

        # setDeadline rebuilds EE resources and cannot precede the first
        # Initialize in current EE releases. A bounded public HTTP transport
        # covers that first request; setDeadline is applied after initialization.
        ee.data.setMaxRetries(EE_RPC_MAX_RETRIES)
        with requests.Session() as auth_session:
            auth_request = BoundedAuthRequest(session=auth_session)
            base64_key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
            project = None
            if base64_key:
                key_data = json.loads(base64.b64decode(base64_key).decode())
                credentials = ServiceCredentials.from_service_account_info(
                    key_data, scopes=ee.oauth.SCOPES)
            else:
                # Read saved credentials without EE's unbounded token check.
                # Ambient Google credentials remain supported when no saved
                # Earth Engine refresh token exists; no interactive login runs.
                try:
                    saved_credentials = ee.oauth.get_credentials_arguments()
                except OSError:
                    saved_credentials = {}
                if saved_credentials.get('refresh_token'):
                    credentials = Credentials(token=None, **saved_credentials)
                    project = (saved_credentials.get('quota_project_id')
                               or ee.oauth.get_appdefault_project())
                    if project and project != credentials.quota_project_id:
                        credentials = credentials.with_quota_project(project)
                else:
                    credentials, project = google.auth.default(
                        scopes=ee.oauth.SCOPES, request=auth_request)
                    project = getattr(credentials, 'quota_project_id', None) or project
            credentials.refresh(auth_request)

        transport = httplib2.Http(timeout=timeout_seconds)
        ee.Initialize(credentials, http_transport=transport, project=project)
        ee.data.setDeadline(EE_RPC_TIMEOUT_MS)
        return True, "Earth Engine initialized"
    except Exception as auth_error:
        return False, f"Earth Engine initialization failed: {auth_error}"


# =============================================================================
# Helper Functions
# =============================================================================
def get_utm_zone(longitude):
    return math.floor((longitude + 180) / 6) + 1


# -----------------------------------------------------------------------------
# Persistent download cache
# -----------------------------------------------------------------------------
def _cache_root():
    """First writable location for the cache folder (never raises)."""
    candidates = [
        os.path.join(tempfile.gettempdir(), CACHE_ROOT_NAME),
        os.path.join(os.path.expanduser("~"), "." + CACHE_ROOT_NAME),
    ]
    try:
        candidates.append(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "." + CACHE_ROOT_NAME)
        )
    except Exception:
        pass

    for folder in candidates:
        try:
            os.makedirs(folder, exist_ok=True)
            probe = os.path.join(folder, ".write_test")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            return folder
        except Exception:
            continue
    return tempfile.mkdtemp()


def get_cache_dir_for_config(config):
    """
    Deterministic cache folder for one monitoring run: the same region + date
    range + settings always resolve to the same folder, so a resume finds the
    months that were already downloaded instead of starting over.
    """
    try:
        payload = {
            'coords': [[round(float(x), 6), round(float(y), 6)] for x, y in config['polygon_coords']],
            'start': config['start_date'],
            'end': config['end_date'],
            'scale': config.get('scale', 10),
            'cloud': config.get('cloudy_pixel_percentage', CLOUD_THRESHOLD),
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    except Exception:
        key = "default"

    folder = os.path.join(_cache_root(), key)
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception:
        return tempfile.mkdtemp()
    return folder


def save_run_config(cache_dir, config):
    """
    Store run metadata next to its downloaded files for inspection/recovery.

    Resume currently uses processing_config from the same Streamlit session.
    A new session does not automatically load this file or adopt another run;
    recovery also requires the server's cached files to remain available.
    """
    _write_cache_json(os.path.join(cache_dir, RUN_CONFIG_FILE), config)


def _write_cache_json(path, payload):
    """Commit a checkpoint only after a complete write in the same directory."""
    pending_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=os.path.dirname(path),
                                         prefix='.checkpoint_', suffix='.tmp', delete=False) as stream:
            pending_path = stream.name
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending_path, path)
        return True
    except Exception:
        return False
    finally:
        if pending_path and os.path.exists(pending_path):
            try:
                os.remove(pending_path)
            except OSError:
                pass


def _load_month_checkpoint(cache_dir, parameter_type):
    """Restore only the selected run's month ledger, never a server-wide latest run."""
    try:
        with open(os.path.join(cache_dir, RUN_STATE_FILE), encoding='utf-8') as stream:
            state = json.load(stream)
        if state.get('version') != 1:
            return {}
        statuses = state.get('parameters', {}).get(parameter_type, {})
        if not isinstance(statuses, dict):
            return {}
        return {month: entry for month, entry in statuses.items()
                if isinstance(month, str) and isinstance(entry, dict)
                and entry.get('status') in (STATUS_COMPLETE, STATUS_NO_DATA, STATUS_FAILED, STATUS_SKIPPED)}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def _save_month_checkpoint(cache_dir, parameter_type, statuses):
    """Persist progress immediately, matching the reference app's per-month ledger."""
    parameters = {parameter: _load_month_checkpoint(cache_dir, parameter) for parameter in ALL_PARAMETERS}
    parameters[parameter_type] = statuses
    return _write_cache_json(os.path.join(cache_dir, RUN_STATE_FILE),
                             {'version': 1, 'parameters': parameters})


def _cached_water_thumbnail(wq_path, rgb_path, month_name, parameter_type, value_range=None):
    """Reuse completed processing like change_detection_app's classification cache.

    Store lossless PNGs and plain JSON instead of loading executable pickle files.
    CDOM's common display range is part of the identity, so added months cannot
    leave its older thumbnails on a different colour scale.
    """
    try:
        sources = []
        for path in (wq_path, rgb_path):
            stat = os.stat(path)
            sources.append([os.path.abspath(path), stat.st_size, stat.st_mtime_ns])
        identity = {'version': 1, 'sources': sources, 'parameter': parameter_type,
                    'month': month_name, 'range': list(value_range) if value_range else None}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        folder = os.path.join(os.path.dirname(wq_path), 'processed_cache')
        prefix = os.path.join(folder, key)
        metadata_path = prefix + '.json'
        try:
            with open(metadata_path, encoding='utf-8') as stream:
                metadata = json.load(stream)
            if metadata['signature'] != key:
                raise ValueError('Cache identity mismatch')
            with Image.open(prefix + '_rgb.png') as source:
                rgb_image = source.copy()
            with Image.open(prefix + '_wq.png') as source:
                wq_image = source.copy()
            return {'rgb_image': rgb_image, 'wq_image': wq_image, 'month_name': month_name,
                    'mean_value': float(metadata['mean']) if metadata['mean'] is not None else np.nan,
                    'water_coverage': float(metadata['coverage']),
                    'valid_pixels': int(metadata['valid_pixels']), 'parameter_type': parameter_type}
        except (OSError, ValueError, TypeError, KeyError):
            pass
    except (OSError, ValueError, TypeError):
        prefix = None

    thumb = generate_thumbnails(wq_path, rgb_path, month_name, parameter_type, value_range=value_range)
    if thumb is not None and prefix is not None:
        try:
            os.makedirs(folder, exist_ok=True)
            # Metadata is committed last and is the completion marker.
            for suffix, image in (('_rgb.png', thumb['rgb_image']), ('_wq.png', thumb['wq_image'])):
                with tempfile.NamedTemporaryFile(dir=folder, suffix='.png', delete=False) as stream:
                    image_path = stream.name
                try:
                    image.save(image_path, format='PNG')
                    os.replace(image_path, prefix + suffix)
                finally:
                    if os.path.exists(image_path):
                        os.remove(image_path)
            _write_cache_json(metadata_path, {
                'signature': key, 'mean': _finite_monthly_value(thumb['mean_value']),
                'coverage': float(thumb['water_coverage']), 'valid_pixels': int(thumb['valid_pixels']),
            })
        except (OSError, ValueError, TypeError, KeyError):
            pass  # Cache failure never discards a scientifically computed result.
    return thumb


def _prepare_monitoring_run(config):
    """Start the selected run while retaining its downloaded files, as in the reference."""
    st.session_state.processing_config = config
    st.session_state.current_temp_dir = _cache_dir_for_active_run(config)
    save_run_config(st.session_state.current_temp_dir, config)
    st.session_state.downloaded_months = empty_param_dict(dict)
    st.session_state.month_statuses = empty_param_dict(dict)
    st.session_state.results = empty_param_dict(list)
    st.session_state.mean_data = empty_param_dict(dict)
    st.session_state.download_summary = {}
    st.session_state.cdom_display_range = None
    st.session_state.processing_complete = False
    st.session_state.processing_in_progress = True
    st.session_state.resume_after_interruption = False
    st.session_state.last_run_error = None
    st.session_state.pending_run = 'start'


def _cache_dir_for_active_run(config):
    """Keep the reference app's session directory when it belongs to this run."""
    current_dir = st.session_state.get('current_temp_dir')
    if current_dir and os.path.isdir(current_dir):
        try:
            with open(os.path.join(current_dir, RUN_CONFIG_FILE), encoding='utf-8') as stream:
                saved = json.load(stream)
            if json.dumps(saved, sort_keys=True) == json.dumps(config, sort_keys=True):
                return current_dir
        except (OSError, ValueError, TypeError):
            pass
    return get_cache_dir_for_config(config)


def _failure_message(error):
    """Keep the cause useful without displaying signed URL query credentials."""
    import re
    message = str(error).strip() or type(error).__name__
    return re.sub(r'(https?://[^?\s]+)\?[^\s]+', r'\1?[redacted]', message)[:1500]


def _is_transient_error(error):
    message = _failure_message(error).lower()
    return isinstance(error, (ConnectionError, TimeoutError, requests.exceptions.Timeout,
                              requests.exceptions.ConnectionError)) or any(
        token in message for token in (
            'timeout', 'timed out', 'connection', 'temporar', 'network', 'dns',
            'remote end closed', 'reset by peer', 'deadline exceeded', 'rate limit',
            'too many requests', '429', '500', '502', '503', '504', 'service unavailable',
        )
    )


def _retry_ee_request(request):
    """Retry a transient metadata error without losing the current month."""
    for attempt in range(MAX_RETRIES):
        try:
            return request()
        except Exception as error:
            if not _is_transient_error(error) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY_BASE ** (attempt + 1))


def _recover_interrupted_run():
    """A new script execution cannot inherit a lock from an abandoned one."""
    if st.session_state.processing_config is None:
        st.session_state.pending_run = None
        st.session_state.processing_in_progress = False
        return
    if st.session_state.processing_in_progress and not st.session_state.get('pending_run'):
        st.session_state.processing_in_progress = False
        st.session_state.resume_after_interruption = bool(st.session_state.processing_config)


def _execute_monitoring_run(config, resume):
    """Always release controls; a failed run resumes only on an explicit click."""
    st.session_state.pending_run = None
    st.session_state.processing_in_progress = True
    st.session_state.resume_after_interruption = False
    st.session_state.last_run_error = None
    try:
        aoi = None

        def resolve_aoi():
            nonlocal aoi
            if aoi is None:
                ee_ok, ee_msg = initialize_earth_engine()
                st.session_state.ee_initialized = ee_ok
                if not ee_ok:
                    raise ConnectionError(ee_msg)
                aoi = ee.Geometry.Polygon([config['polygon_coords']])
            return aoi

        # Restoring complete TIFFs/thumbnails (including RGB assembly) is local.
        # Authenticate only if a missing month actually requires a server call.
        success = run_full_analysis(resolve_aoi, config['start_date'], config['end_date'],
                                    config.get('cloudy_pixel_percentage', CLOUD_THRESHOLD),
                                    config.get('scale', 10), resume=resume)
        st.session_state.processing_complete = bool(success or _has_any_results())
        if not success and not _has_any_results() and not st.session_state.resume_after_interruption:
            st.warning('داده‌ای برای این منطقه و بازه زمانی یافت نشد.')
    except Exception as error:
        st.session_state.resume_after_interruption = True
        st.session_state.last_run_error = _failure_message(error)
        st.error('پایش کامل نشد. فایل‌های دریافت‌شده حفظ شده‌اند؛ دکمه «ادامه از محل قطع» را بزنید.')
    except BaseException:
        # Streamlit rerun/stop signals must propagate, but never leave the UI locked.
        st.session_state.resume_after_interruption = True
        raise
    finally:
        st.session_state.processing_in_progress = False


def reset_run_cache(cache_dir):
    """Empty a run's cache folder — used when the user starts a FRESH run."""
    try:
        if not cache_dir or not os.path.isdir(cache_dir):
            return
        if _cache_root() not in os.path.abspath(cache_dir):
            return  # never touch anything outside our own cache root
        for name in os.listdir(cache_dir):
            path = os.path.join(cache_dir, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
            except Exception:
                continue
    except Exception:
        pass


def prune_old_cache_dirs(max_age_days=CACHE_MAX_AGE_DAYS):
    """Delete run folders nobody has touched for a fortnight."""
    root = _cache_root()
    cutoff = time.time() - max_age_days * 86400
    try:
        for name in os.listdir(root):
            folder = os.path.join(root, name)
            try:
                if os.path.isdir(folder) and os.path.getmtime(folder) < cutoff:
                    shutil.rmtree(folder, ignore_errors=True)
            except Exception:
                continue
    except Exception:
        pass


def validate_geotiff_file(file_path, expected_bands=1):
    """Validate that a GeoTIFF file is complete and readable."""
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"

        if os.path.getsize(file_path) == 0:
            return False, "File is empty"

        with rasterio.open(file_path) as src:
            if src.count < expected_bands:
                return False, f"Wrong band count ({src.count}, expected {expected_bands})"
            # A readable TIFF header does not prove its compressed pixel blocks
            # survived a dropped transfer. Read block-by-block to bound memory.
            for band_index in range(1, src.count + 1):
                for _, window in src.block_windows(band_index):
                    src.read(band_index, window=window)

        return True, "File is valid"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


# =============================================================================
# Water Quality Calculation (GEE Server-Side)
# =============================================================================
def create_water_quality_collection(aoi, start_date, end_date, parameter_type, cloudy_pixel_percentage=CLOUD_THRESHOLD):
    """
    Create water quality collection for Turbidity, Chlorophyll or CDOM.

    Snow detection is used as a PREPROCESSING step to exclude snow/ice pixels
    from water detection. Snow mask is never downloaded or shown to the user.

    For TURBIDITY (NDTI):
    1. Link S2_SR with S2_CLOUD_PROBABILITY
    2. Apply cloud mask (probability < 15)
    3. Calculate NDSI for snow detection: (B3 - B11) / (B3 + B11)
    4. Create snow mask (MODIS-heritage water/snow test): NDSI > 0.42 AND B8 (NIR) > 0.11 AND B3 (Green) > 0.1
    5. Calculate AWEIsh for water body detection: B2 + 2.5*B3 - 1.5*(B8+B11) - 0.25*B12 > 0.05, excluding snow
    6. Calculate NDTI (turbidity index): (B4 - B3) / (B4 + B3)

    For CHLOROPHYLL:
    1. Link S2_SR with S2_CLOUD_PROBABILITY
    2. Apply cloud mask (probability < 15)
    3. Calculate NDSI for snow detection: (B3 - B11) / (B3 + B11)
    4. Create snow mask (MODIS-heritage water/snow test): NDSI > 0.42 AND B8 (NIR) > 0.11 AND B3 (Green) > 0.1
    5. Calculate AWEIsh for water body detection: B2 + 2.5*B3 - 1.5*(B8+B11) - 0.25*B12 > 0.05, excluding snow
    6. Calculate Chlorophyll Index (NDCI): (B5 - B4) / (B5 + B4)

    For CDOM (Colored Dissolved Organic Matter):
    1-5. Exactly the same preprocessing chain as above (cloud mask, snow mask,
         AWEIsh water body) so all three parameters are computed on the very
         same water pixels and are directly comparable.
    6. Calculate CDOM from the green/red band ratio:
           CDOM = 537 * exp(-2.93 * (B3 / B4))
       This is the Landsat-heritage band-ratio model of Brezonik, Menken &
       Bauer (2005), applied to the equivalent Sentinel-2 bands. Its output
       approximates the CDOM absorption coefficient at 440 nm (a440), in
       units of 1/m — higher values mean more dissolved organic ("humic")
       matter and darker, tea-coloured water.
    """
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(start_date, end_date)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage)))

    s2_cloud_prob = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date))

    join_filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    joined = ee.Join.saveFirst('cloud_probability').apply(
        primary=s2_sr, secondary=s2_cloud_prob, condition=join_filter
    )

    def add_cloud_band(feature):
        img = ee.Image(feature)
        cloud_prob_img = ee.Image(img.get('cloud_probability'))
        return img.addBands(cloud_prob_img.select('probability'))

    s2_joined = ee.ImageCollection(joined.map(add_cloud_band))

    if parameter_type == PARAM_TURBIDITY:
        def calculate_turbidity(img):
            cloud = img.select('probability')
            cloud_free = cloud.lt(CLOUD_PROB_THRESHOLD)

            sr = img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).multiply(0.0001)

            ndsi = sr.normalizedDifference(['B3', 'B11']).rename('ndsi')
            is_snow = (
                ndsi.gt(NDSI_THRESHOLD)                          # NDSI > 0.42 — spectral snow signature
                .And(sr.select('B8').gt(NIR_SNOW_THRESHOLD))     # NIR ~0.11 — excludes water, snow reflects strongly here
                .And(sr.select('B3').gt(GREEN_SNOW_THRESHOLD))   # Green ~0.1 — excludes dark shadow/non-snow surfaces
            )

            awei = sr.expression(
                'BLUE + 2.5 * GREEN - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2',
                {
                    'BLUE': sr.select('B2'),
                    'GREEN': sr.select('B3'),
                    'NIR': sr.select('B8'),
                    'SWIR1': sr.select('B11'),
                    'SWIR2': sr.select('B12'),
                }
            ).rename('awei')
            water_body = awei.gt(AWEI_THRESHOLD).And(is_snow.Not())

            ndti = sr.normalizedDifference(['B4', 'B3']).rename('wq_index')

            wq_masked = ndti.updateMask(cloud_free).updateMask(water_body)

            rgb = sr.select(['B4', 'B3', 'B2'])

            combined = (wq_masked
                       .addBands(rgb)
                       .addBands(water_body.rename('water_mask')))

            return combined.clip(aoi).copyProperties(img, ['system:time_start'])

        return s2_joined.map(calculate_turbidity)

    elif parameter_type == PARAM_CDOM:
        def calculate_cdom(img):
            cloud = img.select('probability')
            cloud_free = cloud.lt(CLOUD_PROB_THRESHOLD)

            sr = img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).multiply(0.0001)

            ndsi = sr.normalizedDifference(['B3', 'B11']).rename('ndsi')
            is_snow = (
                ndsi.gt(NDSI_THRESHOLD)                          # NDSI > 0.42 — spectral snow signature
                .And(sr.select('B8').gt(NIR_SNOW_THRESHOLD))     # NIR ~0.11 — excludes water, snow reflects strongly here
                .And(sr.select('B3').gt(GREEN_SNOW_THRESHOLD))   # Green ~0.1 — excludes dark shadow/non-snow surfaces
            )

            awei = sr.expression(
                'BLUE + 2.5 * GREEN - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2',
                {
                    'BLUE': sr.select('B2'),
                    'GREEN': sr.select('B3'),
                    'NIR': sr.select('B8'),
                    'SWIR1': sr.select('B11'),
                    'SWIR2': sr.select('B12'),
                }
            ).rename('awei')
            water_body = awei.gt(AWEI_THRESHOLD).And(is_snow.Not())

            # CDOM from the green/red band ratio (Brezonik et al., 2005):
            #     CDOM = 537 * exp(-2.93 * (B3 / B4))
            # Red reflectance is the denominator, so pixels where it is zero or
            # negative (deep shadow, residual masking artefacts) are excluded
            # rather than producing an infinite ratio.
            red_ok = sr.select('B4').gt(0)

            cdom = sr.expression(
                '537 * exp(-2.93 * (B03 / B04))',
                {
                    'B03': sr.select('B3'),
                    'B04': sr.select('B4'),
                }
            ).rename('wq_index')

            wq_masked = (cdom
                         .updateMask(cloud_free)
                         .updateMask(water_body)
                         .updateMask(red_ok))

            rgb = sr.select(['B4', 'B3', 'B2'])

            combined = (wq_masked
                       .addBands(rgb)
                       .addBands(water_body.rename('water_mask')))

            return combined.clip(aoi).copyProperties(img, ['system:time_start'])

        return s2_joined.map(calculate_cdom)

    else:  # CHLOROPHYLL
        def calculate_chlorophyll(img):
            cloud = img.select('probability')
            cloud_free = cloud.lt(CLOUD_PROB_THRESHOLD)

            sr = img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12']).multiply(0.0001)

            ndsi = sr.normalizedDifference(['B3', 'B11']).rename('ndsi')
            is_snow = (
                ndsi.gt(NDSI_THRESHOLD)                          # NDSI > 0.42 — spectral snow signature
                .And(sr.select('B8').gt(NIR_SNOW_THRESHOLD))     # NIR ~0.11 — excludes water, snow reflects strongly here
                .And(sr.select('B3').gt(GREEN_SNOW_THRESHOLD))   # Green ~0.1 — excludes dark shadow/non-snow surfaces
            )

            awei = sr.expression(
                'BLUE + 2.5 * GREEN - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2',
                {
                    'BLUE': sr.select('B2'),
                    'GREEN': sr.select('B3'),
                    'NIR': sr.select('B8'),
                    'SWIR1': sr.select('B11'),
                    'SWIR2': sr.select('B12'),
                }
            ).rename('awei')
            water_body = awei.gt(AWEI_THRESHOLD).And(is_snow.Not())

            # NDCI (Normalized Difference Chlorophyll Index): (B5 - B4) / (B5 + B4)
            # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndci/
            chl_index = sr.normalizedDifference(['B5', 'B4']).rename('wq_index')

            wq_masked = chl_index.updateMask(cloud_free).updateMask(water_body)

            rgb = sr.select(['B4', 'B3', 'B2'])

            combined = (wq_masked
                       .addBands(rgb)
                       .addBands(water_body.rename('water_mask')))

            return combined.clip(aoi).copyProperties(img, ['system:time_start'])

        return s2_joined.map(calculate_chlorophyll)


def get_monthly_composite(wq_collection, aoi, year, month):
    """Create monthly composite from water quality collection."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')

    monthly = wq_collection.filterDate(start, end)
    count = monthly.size().getInfo()

    if count == 0:
        return None, 0, "No images"

    composite = monthly.median()

    stats = composite.select('wq_index').reduceRegion(
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.count(), sharedInputs=True
        ).combine(
            ee.Reducer.minMax(), sharedInputs=True
        ),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )

    return composite, count, stats


# =============================================================================
# Download Functions
# =============================================================================
def download_band_with_retry(image, band, aoi, output_path, scale=10):
    """Download a single band with retry mechanism."""
    # --- Cache check FIRST, before any network call ---------------------------
    # A band already on disk must cost nothing. The old order asked Earth Engine
    # for the AOI bounds before looking at the cache, so every resume paid for a
    # round trip per band — and, with no connection, that call failed and the
    # month was discarded even though its files were sitting right there.
    if os.path.exists(output_path):
        is_valid, msg = validate_geotiff_file(output_path, expected_bands=1)
        if is_valid:
            return True, "cached"
        try:
            os.remove(output_path)
        except Exception:
            pass

    temp_path = output_path + '.tmp'
    if os.path.exists(temp_path):
        os.remove(temp_path)

    last_error = None

    for attempt in range(MAX_RETRIES):
        response = None
        try:
            # Bounds and URL generation are network calls too. Retry them with
            # the band, rather than failing permanently before the retry loop.
            region = aoi.bounds().getInfo()['coordinates']
            url = image.select(band).getDownloadURL({
                'scale': scale, 'region': region, 'format': 'GEO_TIFF', 'bands': [band]
            })

            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)

            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    last_error = "GEE rate limit"
                    raise Exception(last_error)

                downloaded_size = 0
                started = time.monotonic()
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if time.monotonic() - started > 180:
                            raise requests.exceptions.Timeout('Band transfer exceeded 180 seconds')
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                is_valid, msg = validate_geotiff_file(temp_path, expected_bands=1)
                if is_valid:
                    os.replace(temp_path, output_path)
                    return True, "success"
                else:
                    last_error = f"Validation failed: {msg}"
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise Exception(last_error)
            else:
                last_error = f"HTTP {response.status_code}"
                raise Exception(last_error)

        except requests.exceptions.Timeout:
            last_error = "Timeout"
        except requests.exceptions.ConnectionError:
            last_error = "Connection error"
        except Exception as e:
            last_error = _failure_message(e)
        finally:
            if response is not None:
                response.close()

        for f in [output_path, temp_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

        if attempt < MAX_RETRIES - 1:
            wait_time = RETRY_DELAY_BASE ** (attempt + 1)
            time.sleep(wait_time)

    return False, last_error


def download_monthly_data(composite, aoi, temp_dir, month_name, param_short, scale=10):
    """
    Download monthly composite (Water Quality Index + RGB bands).
    Snow mask is NOT downloaded — used only server-side in GEE.
    No UI status is written here; progress is summarized at a higher level.
    """
    wq_path = os.path.join(temp_dir, f"wq_index_{param_short}_{month_name}.tif")
    rgb_path = os.path.join(temp_dir, f"rgb_{param_short}_{month_name}.tif")

    wq_valid, _ = validate_geotiff_file(wq_path, expected_bands=1)
    rgb_valid, _ = validate_geotiff_file(rgb_path, expected_bands=3)

    if wq_valid and rgb_valid:
        return wq_path, rgb_path, STATUS_COMPLETE, "Cached"

    try:
        success, msg = download_band_with_retry(composite, 'wq_index', aoi, wq_path, scale)
        if not success:
            return None, None, STATUS_FAILED, f"WQ Index download failed: {msg}"

        bands_dir = os.path.join(temp_dir, f"bands_{param_short}_{month_name}")
        os.makedirs(bands_dir, exist_ok=True)

        rgb_bands = ['B4', 'B3', 'B2']
        band_files = []

        for band in rgb_bands:
            band_file = os.path.join(bands_dir, f"{band}.tif")
            success, msg = download_band_with_retry(composite, band, aoi, band_file, scale)

            if not success:
                return None, None, STATUS_FAILED, f"RGB {band} download failed: {msg}"

            band_files.append(band_file)

        with rasterio.open(band_files[0]) as src:
            meta = src.meta.copy()
        meta.update(count=3)

        # Only a fully written, readable composite may occupy the final cache
        # path. Interrupted assembly leaves a disposable .tmp, never a cache hit.
        rgb_temp_path = rgb_path + '.tmp'
        try:
            with rasterio.open(rgb_temp_path, 'w', **meta) as dst:
                for i, band_file in enumerate(band_files):
                    with rasterio.open(band_file) as src:
                        dst.write(src.read(1), i+1)
            rgb_valid, rgb_message = validate_geotiff_file(rgb_temp_path, expected_bands=3)
            if not rgb_valid:
                raise ValueError(f"RGB composite validation failed: {rgb_message}")
            os.replace(rgb_temp_path, rgb_path)
        finally:
            if os.path.exists(rgb_temp_path):
                try:
                    os.remove(rgb_temp_path)
                except OSError:
                    pass

        return wq_path, rgb_path, STATUS_COMPLETE, "Downloaded"

    except Exception as e:
        return None, None, STATUS_FAILED, f"Error: {str(e)}"


# =============================================================================
# Visualization Functions
# =============================================================================
def create_turbidity_colormap():
    colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
    return LinearSegmentedColormap.from_list('turbidity', colors, N=256)


def create_chlorophyll_colormap():
    colors = ['#9400D3', '#4B0082', '#0000FF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000']
    return LinearSegmentedColormap.from_list('chlorophyll', colors, N=256)


def create_cdom_colormap():
    """Clear blue water -> yellow -> amber -> dark brown (tea-coloured, humic)."""
    colors = ['#08306B', '#2171B5', '#6BAED6', '#EDF8B1', '#FEE391', '#EC7014', '#662506']
    return LinearSegmentedColormap.from_list('cdom', colors, N=256)


def compute_cdom_display_range(wq_paths, low=2, high=98):
    """
    Derive one common colour range for every CDOM month of a run.

    CDOM is an absorption coefficient, not a bounded index, so a fixed scale
    either saturates or washes out depending on the water body. The 2nd-98th
    percentile of all months pooled together keeps the colours meaningful AND
    comparable between months (which a per-image stretch would not).
    """
    samples = []
    for path in wq_paths:
        try:
            with rasterio.open(path) as src:
                data = src.read(1)[::4, ::4]  # subsample: plenty for a percentile
            finite = data[np.isfinite(data) & (data != 0)]
            if finite.size:
                samples.append(finite)
        except Exception:
            continue

    if not samples:
        return CDOM_VMIN, CDOM_VMAX

    pooled = np.concatenate(samples)
    if pooled.size == 0:
        return CDOM_VMIN, CDOM_VMAX

    vmin = float(np.percentile(pooled, low))
    vmax = float(np.percentile(pooled, high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return CDOM_VMIN, CDOM_VMAX
    return vmin, vmax


def generate_thumbnails(wq_path, rgb_path, month_name, parameter_type, max_size=300,
                        value_range=None):
    """Generate RGB and water quality index thumbnails."""
    try:
        with rasterio.open(wq_path) as src:
            wq_data = src.read(1)

        with rasterio.open(rgb_path) as src:
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)

        rgb = np.stack([red, green, blue], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0)

        def percentile_stretch(band, lower=2, upper=98):
            valid = band[band > 0]
            if len(valid) == 0:
                return np.zeros_like(band, dtype=np.uint8)
            p_low = np.percentile(valid, lower)
            p_high = np.percentile(valid, upper)
            if p_high <= p_low:
                p_high = p_low + 0.001
            stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
            return (stretched * 255).astype(np.uint8)

        rgb_uint8 = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            rgb_uint8[:, :, i] = percentile_stretch(rgb[:, :, i])

        wq_valid = np.nan_to_num(wq_data, nan=np.nan)

        valid_wq = wq_valid[~np.isnan(wq_valid) & (wq_valid != 0)]
        mean_value = np.nanmean(valid_wq) if len(valid_wq) > 0 else np.nan
        valid_pixel_count = len(valid_wq)
        total_pixels = wq_data.size
        water_coverage = (valid_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0

        if parameter_type == PARAM_TURBIDITY:
            cmap = create_turbidity_colormap()
            wq_normalized = np.clip((wq_valid + 0.3) / 0.6, 0, 1)
        elif parameter_type == PARAM_CDOM:
            cmap = create_cdom_colormap()
            vmin, vmax = value_range if value_range else (CDOM_VMIN, CDOM_VMAX)
            if vmax <= vmin:
                vmax = vmin + 1e-6
            wq_normalized = np.clip((wq_valid - vmin) / (vmax - vmin), 0, 1)
        else:
            cmap = create_chlorophyll_colormap()
            wq_normalized = np.clip((wq_valid - CHL_VMIN) / (CHL_VMAX - CHL_VMIN), 0, 1)

        wq_normalized = np.nan_to_num(wq_normalized, nan=0)

        wq_colored = cmap(wq_normalized)[:, :, :3]
        wq_uint8 = (wq_colored * 255).astype(np.uint8)

        water_mask = (~np.isnan(wq_valid)) & (wq_valid != 0)
        for i in range(3):
            wq_uint8[:, :, i] = np.where(water_mask, wq_uint8[:, :, i], 50)

        pil_rgb = Image.fromarray(rgb_uint8, mode='RGB')
        pil_wq = Image.fromarray(wq_uint8, mode='RGB')

        h, w = pil_rgb.size[1], pil_rgb.size[0]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_rgb = pil_rgb.resize((new_w, new_h), Image.LANCZOS)
            pil_wq = pil_wq.resize((new_w, new_h), Image.LANCZOS)

        return {
            'rgb_image': pil_rgb,
            'wq_image': pil_wq,
            'month_name': month_name,
            'mean_value': mean_value,
            'water_coverage': water_coverage,
            'valid_pixels': valid_pixel_count,
            'parameter_type': parameter_type
        }

    except Exception:
        # Quietly skip a month that fails to render rather than surfacing
        # remote-sensing error internals to a non-technical user.
        return None


# =============================================================================
# Main Processing Pipeline (silent — no remote-sensing internals shown)
# =============================================================================
def process_single_parameter(aoi, start_date, end_date, parameter_type, temp_dir,
                              cloudy_pixel_percentage=CLOUD_THRESHOLD, scale=10,
                              resume=False, progress_callback=None):
    """
    Run the full pipeline for one parameter (NDTI or Chlorophyll-a):
    cloud filtering -> snow masking -> waterbody extraction -> index calculation
    -> download -> thumbnail generation.

    Key resilience behaviours (adapted from old version):
    - Per-month session state writes: progress is preserved after every month so
      that a connection drop never discards completed work.
    - Resume logic: when resume=True, months already present in
      st.session_state.downloaded_months[parameter_type] (and whose files are
      still valid on disk) are skipped entirely.
    - File-level cache: download_monthly_data() validates existing GeoTIFFs on
      disk before attempting a new download — so cached files survive page
      reloads even without session state.
    - EE server calls (.size().getInfo(), get_monthly_composite) are wrapped in
      try/except so a transient network error on one month does not crash the
      whole pipeline; the month is marked STATUS_FAILED and processing continues.

    Returns: (results_list, mean_data_dict, downloaded_count, available_count)
    No technical logs are written to the UI; this function is silent.

    progress_callback(done, total, month_name), if given, is invoked once
    before the download loop starts (to reflect months already recovered
    from cache/resume) and once after every month is handled, so the caller
    can drive a progress bar / stage label.
    """
    param_short = {
        PARAM_TURBIDITY: "turbidity",
        PARAM_CHLOROPHYLL: "chlorophyll",
        PARAM_CDOM: "cdom",
    }[parameter_type]

    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)

    # ------------------------------------------------------------------
    # Defer the collection until a missing month actually needs Earth Engine.
    # ------------------------------------------------------------------
    wq_collection = None

    month_infos = []
    for month_index in range(total_months):
        year = start_dt.year + (start_dt.month - 1 + month_index) // 12
        month = (start_dt.month - 1 + month_index) % 12 + 1
        month_infos.append({'month_name': f"{year}-{month:02d}", 'year': year, 'month': month})
    current_months = {month['month_name'] for month in month_infos}

    # ------------------------------------------------------------------
    # FIX A: Restore already-downloaded months from session state (resume)
    # ------------------------------------------------------------------
    # Ensure the nested dict for this parameter exists in session state
    if not isinstance(st.session_state.downloaded_months.get(parameter_type), dict):
        st.session_state.downloaded_months[parameter_type] = {}
    if not isinstance(st.session_state.month_statuses.get(parameter_type), dict):
        st.session_state.month_statuses[parameter_type] = {}

    # Like the reference month_analysis_results ledger, restore progress before
    # building the remaining-month list. Scope it to this exact run and period.
    statuses = _load_month_checkpoint(temp_dir, parameter_type)
    if resume:
        statuses.update(st.session_state.month_statuses[parameter_type])
    st.session_state.month_statuses[parameter_type] = {
        month: entry for month, entry in statuses.items() if month in current_months
    }

    downloaded_months = {}

    if resume and st.session_state.downloaded_months.get(parameter_type):
        for month_name, paths in st.session_state.downloaded_months[parameter_type].items():
            if month_name not in current_months:
                continue
            if paths.get('wq_index') and paths.get('rgb'):
                if any(os.path.dirname(os.path.abspath(paths[key])) != os.path.abspath(temp_dir)
                       for key in ('wq_index', 'rgb')):
                    continue
                wq_valid, _ = validate_geotiff_file(paths['wq_index'], expected_bands=1)
                rgb_valid, _ = validate_geotiff_file(paths['rgb'], expected_bands=3)
                if wq_valid and rgb_valid:
                    downloaded_months[month_name] = paths
                    # Keep the cached status entry as well
                    if month_name not in st.session_state.month_statuses[parameter_type]:
                        st.session_state.month_statuses[parameter_type][month_name] = {
                            'status': STATUS_COMPLETE, 'message': 'Cached'
                        }

    # Also check disk for any months whose files exist but session state was lost
    # (e.g. after a page reload without resume — file-level cache recovery)
    for month_info in month_infos:
        month_name = month_info['month_name']
        if month_name in downloaded_months:
            continue
        wq_path = os.path.join(temp_dir, f"wq_index_{param_short}_{month_name}.tif")
        rgb_path = os.path.join(temp_dir, f"rgb_{param_short}_{month_name}.tif")
        wq_valid, _ = validate_geotiff_file(wq_path, expected_bands=1)
        rgb_valid, _ = validate_geotiff_file(rgb_path, expected_bands=3)
        # A disconnect/rerun can arrive after the last source band was saved
        # but before RGB assembly. Finish that local work before querying EE.
        if wq_valid and not rgb_valid:
            bands_dir = os.path.join(temp_dir, f"bands_{param_short}_{month_name}")
            if all(validate_geotiff_file(os.path.join(bands_dir, f"{band}.tif"), 1)[0]
                   for band in ('B4', 'B3', 'B2')):
                _, _, recovered_status, _ = download_monthly_data(
                    None, aoi, temp_dir, month_name, param_short, scale
                )
                rgb_valid = recovered_status == STATUS_COMPLETE
        if wq_valid and rgb_valid:
            downloaded_months[month_name] = {'wq_index': wq_path, 'rgb': rgb_path}
            st.session_state.downloaded_months[parameter_type][month_name] = downloaded_months[month_name]
            st.session_state.month_statuses[parameter_type][month_name] = {
                'status': STATUS_COMPLETE, 'message': 'Cached (disk)'
            }

    st.session_state.downloaded_months[parameter_type] = dict(downloaded_months)
    _save_month_checkpoint(temp_dir, parameter_type, st.session_state.month_statuses[parameter_type])

    # Months not yet downloaded. Only STATUS_NO_DATA counts as settled from the
    # status alone; a month marked STATUS_COMPLETE is skipped because its files
    # were just validated into downloaded_months above — relying on the status
    # by itself meant a month whose files had gone missing was skipped forever.
    already_statused = {
        m for m, s in st.session_state.month_statuses[parameter_type].items()
        if s.get('status') == STATUS_NO_DATA
    }
    months_to_process = [
        m for m in month_infos
        if m['month_name'] not in downloaded_months
        and m['month_name'] not in already_statused
    ]

    available_count = len(downloaded_months)  # start with already-recovered months

    # Progress bookkeeping: months already resolved before the loop (from the
    # resume/disk-cache recovery above) count as "done" immediately so the
    # progress bar reflects real state instead of restarting from zero.
    already_done_count = total_months - len(months_to_process)
    processed_count = already_done_count
    if progress_callback:
        progress_callback(processed_count, total_months, None)

    # ------------------------------------------------------------------
    # Per-month EE + download loop with bounded retries and immediate checkpoints.
    # ------------------------------------------------------------------
    consecutive_failures = 0

    for month_info in months_to_process:
        month_name = month_info['month_name']

        # FIX C: Wrap every EE server call so a transient error skips the month
        try:
            if callable(aoi):
                aoi = aoi()
            if wq_collection is None:
                wq_collection = create_water_quality_collection(
                    aoi, start_date, end_date, parameter_type, cloudy_pixel_percentage
                )
            composite, count, stats = _retry_ee_request(
                lambda: get_monthly_composite(
                    wq_collection, aoi, month_info['year'], month_info['month']
                )
            )
        except Exception as error:
            message = _failure_message(error)
            st.session_state.month_statuses[parameter_type][month_name] = {
                'status': STATUS_FAILED, 'message': message, 'stage': 'monthly_composite'
            }
            st.session_state.last_run_error = f"{parameter_type} / {month_name}: {message}"
            _save_month_checkpoint(temp_dir, parameter_type, st.session_state.month_statuses[parameter_type])
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count, total_months, month_name)

            consecutive_failures = consecutive_failures + 1 if _is_transient_error(error) else 0
            if consecutive_failures >= NETWORK_FAILURE_STREAK:
                st.session_state.connection_interrupted = True
                break
            continue

        if composite is None or count == 0:
            st.session_state.month_statuses[parameter_type][month_name] = {
                'status': STATUS_NO_DATA, 'message': 'No images'
            }
            _save_month_checkpoint(temp_dir, parameter_type, st.session_state.month_statuses[parameter_type])
            consecutive_failures = 0   # the server answered: the link is fine
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count, total_months, month_name)
            continue

        available_count += 1

        wq_path, rgb_path, status, message = download_monthly_data(
            composite, aoi, temp_dir, month_name, param_short, scale
        )

        # FIX D: Write to session state immediately after each month — not at end
        st.session_state.month_statuses[parameter_type][month_name] = {
            'status': status, 'message': message, 'stage': 'download'
        }

        if status == STATUS_COMPLETE:
            downloaded_months[month_name] = {'wq_index': wq_path, 'rgb': rgb_path}
            # Persist to session state right away so a crash/reload can recover
            st.session_state.downloaded_months[parameter_type][month_name] = {
                'wq_index': wq_path, 'rgb': rgb_path
            }
            consecutive_failures = 0
        else:
            st.session_state.last_run_error = f"{parameter_type} / {month_name}: {message}"
            consecutive_failures = consecutive_failures + 1 if _is_transient_error(message) else 0

        _save_month_checkpoint(temp_dir, parameter_type, st.session_state.month_statuses[parameter_type])

        processed_count += 1
        if progress_callback:
            progress_callback(processed_count, total_months, month_name)

        if consecutive_failures >= NETWORK_FAILURE_STREAK:
            st.session_state.connection_interrupted = True
            break

    # ------------------------------------------------------------------
    # Thumbnail generation (uses only successfully downloaded months)
    # ------------------------------------------------------------------
    results = []
    mean_data = {}
    if 'results' not in st.session_state:
        st.session_state.results = empty_param_dict(list)
    if 'mean_data' not in st.session_state:
        st.session_state.mean_data = empty_param_dict(dict)

    # CDOM gets one colour range shared by every month of the run (see
    # compute_cdom_display_range); the two normalized-difference indices keep
    # their fixed scales exactly as before.
    value_range = None
    if parameter_type == PARAM_CDOM and downloaded_months:
        value_range = compute_cdom_display_range(
            [downloaded_months[m]['wq_index'] for m in sorted(downloaded_months.keys())]
        )
        st.session_state.cdom_display_range = value_range

    for month_name in sorted(downloaded_months.keys()):
        paths = downloaded_months[month_name]
        thumb = _cached_water_thumbnail(paths['wq_index'], paths['rgb'], month_name, parameter_type,
                                         value_range=value_range)
        if thumb:
            results.append(thumb)
            mean_data[month_name] = {'mean': thumb['mean_value'], 'coverage': thumb['water_coverage']}
            st.session_state.results[parameter_type] = list(results)
            st.session_state.mean_data[parameter_type] = dict(mean_data)

    return results, mean_data, len(downloaded_months), available_count


def run_full_analysis(aoi, start_date, end_date, cloudy_pixel_percentage=CLOUD_THRESHOLD,
                       scale=10, resume=False):
    """
    Automatically runs preprocessing + all three index calculations (NDTI, then
    Chlorophyll-a, then CDOM) in sequence. Displays only a simple,
    user-friendly summary.

    Resilience additions vs. original:
    - resume=True is forwarded to process_single_parameter so cached months are
      skipped rather than re-downloaded.
    - Each parameter block is wrapped in try/except so a hard failure on one
      index does not prevent the others from running.
    - processing_config is written to session state here so the main() button
      handler can pass it to a resume run later.
    """
    st.session_state.connection_interrupted = False

    # Cache folder for this exact run (see get_cache_dir_for_config): the same
    # region + date range always maps to the same folder, so months already on
    # disk are reused instead of downloaded twice — even after the session was
    # lost. The run's configuration is stored beside the images so the app can
    # offer to resume it on a completely fresh session.
    config = st.session_state.get('processing_config')
    if config and config.get('polygon_coords'):
        temp_dir = _cache_dir_for_active_run(config)
        save_run_config(temp_dir, config)
    elif st.session_state.current_temp_dir and os.path.exists(st.session_state.current_temp_dir):
        temp_dir = st.session_state.current_temp_dir
    else:
        temp_dir = tempfile.mkdtemp()
    st.session_state.current_temp_dir = temp_dir
    # Atomic metadata replacement touches this run before old-run pruning.
    prune_old_cache_dirs()

    summary_placeholder = st.empty()
    download_summary = dict(st.session_state.download_summary)  # preserve any prior summary

    # ------------------------------------------------------------------
    # Progress bar + stage label — shows overall level (%) and which stage
    # (parameter + month) is currently being processed. Total work units are
    # "months × 3 parameters"; months already recovered via resume/disk cache
    # count as already-done so the bar starts from the right place on Resume.
    # ------------------------------------------------------------------
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    total_units = max(total_months * len(ALL_PARAMETERS), 1)

    progress_bar = st.progress(0)
    stage_text = st.empty()

    def make_progress_callback(stage_label, unit_offset):
        def _callback(done_in_stage, total_in_stage, month_name):
            done_units = unit_offset + done_in_stage
            percent = int(min(1.0, done_units / total_units) * 100)
            if month_name:
                stage_text.markdown(
                    f"**{stage_label}** — در حال پردازش ماه «{month_name}» "
                    f"({done_in_stage} از {total_in_stage}) — {percent}٪"
                )
            else:
                stage_text.markdown(f"**{stage_label}** — در حال آماده‌سازی... — {percent}٪")
            progress_bar.progress(min(1.0, done_units / total_units))
        return _callback

    with st.spinner("در حال پایش کیفیت آب... این فرآیند ممکن است چند دقیقه طول بکشد"):
        # --- Turbidity (NDTI) ---
        turb_results, turb_mean, turb_downloaded, turb_available = [], {}, 0, 0
        try:
            turb_results, turb_mean, turb_downloaded, turb_available = process_single_parameter(
                aoi, start_date, end_date, PARAM_TURBIDITY, temp_dir,
                cloudy_pixel_percentage, scale, resume=resume,
                progress_callback=make_progress_callback("🌊 مرحله ۱ از ۳ — شاخص کدورت آب", 0)
            )
        except Exception as error:
            st.session_state.last_run_error = f"{PARAM_TURBIDITY}: {_failure_message(error)}"
            st.session_state.connection_interrupted = _is_transient_error(error)

        # Merge with any results already in session state (resume case)
        if turb_results:
            st.session_state.results[PARAM_TURBIDITY] = turb_results
            st.session_state.mean_data[PARAM_TURBIDITY] = turb_mean
        download_summary[PARAM_TURBIDITY] = (turb_downloaded, turb_available)

        summary_placeholder.info(
            f"🌊 شاخص کدورت: {turb_downloaded} تصویر از {turb_available} تصویر موجود دریافت شد."
        )

        # --- Chlorophyll-a ---
        # If the link went down during turbidity there is no point hammering
        # Earth Engine for the next two indices: stop and let the user resume.
        chl_results, chl_mean, chl_downloaded, chl_available = [], {}, 0, 0
        try:
            if st.session_state.connection_interrupted:
                raise InterruptedError("connection lost")
            chl_results, chl_mean, chl_downloaded, chl_available = process_single_parameter(
                aoi, start_date, end_date, PARAM_CHLOROPHYLL, temp_dir,
                cloudy_pixel_percentage, scale, resume=resume,
                progress_callback=make_progress_callback("🌿 مرحله ۲ از ۳ — شاخص کلروفیل", total_months)
            )
        except Exception as error:
            if not isinstance(error, InterruptedError):
                st.session_state.last_run_error = f"{PARAM_CHLOROPHYLL}: {_failure_message(error)}"
                st.session_state.connection_interrupted = _is_transient_error(error)

        if chl_results:
            st.session_state.results[PARAM_CHLOROPHYLL] = chl_results
            st.session_state.mean_data[PARAM_CHLOROPHYLL] = chl_mean
        download_summary[PARAM_CHLOROPHYLL] = (chl_downloaded, chl_available)

        summary_placeholder.info(
            f"🌊 شاخص کدورت: {turb_downloaded} تصویر از {turb_available} تصویر موجود دریافت شد.\n\n"
            f"🌿 شاخص کلروفیل: {chl_downloaded} تصویر از {chl_available} تصویر موجود دریافت شد."
        )

        # --- CDOM (Colored Dissolved Organic Matter) ---
        cdom_results, cdom_mean, cdom_downloaded, cdom_available = [], {}, 0, 0
        try:
            if st.session_state.connection_interrupted:
                raise InterruptedError("connection lost")
            cdom_results, cdom_mean, cdom_downloaded, cdom_available = process_single_parameter(
                aoi, start_date, end_date, PARAM_CDOM, temp_dir,
                cloudy_pixel_percentage, scale, resume=resume,
                progress_callback=make_progress_callback(
                    "🍂 مرحله ۳ از ۳ — شاخص مواد آلی محلول", total_months * 2
                )
            )
        except Exception as error:
            if not isinstance(error, InterruptedError):
                st.session_state.last_run_error = f"{PARAM_CDOM}: {_failure_message(error)}"
                st.session_state.connection_interrupted = _is_transient_error(error)

        if cdom_results:
            st.session_state.results[PARAM_CDOM] = cdom_results
            st.session_state.mean_data[PARAM_CDOM] = cdom_mean
        download_summary[PARAM_CDOM] = (cdom_downloaded, cdom_available)

    if st.session_state.connection_interrupted:
        # Stopped on purpose the moment the link went down. Everything already
        # downloaded is on disk and in session state; the months that were never
        # attempted carry no status at all, so «ادامه از محل قطع» continues from
        # exactly this point rather than repeating earlier months.
        stage_text.markdown(
            "⛔ ارتباط با سرور قطع شد و پایش در همان نقطه متوقف شد — "
            "ماه‌های دریافت‌شده حفظ شده‌اند."
        )
        st.session_state.resume_after_interruption = True
    elif all(
        st.session_state.month_statuses.get(parameter, {}).get(
            f"{start_dt.year + (start_dt.month - 1 + index) // 12}-"
            f"{(start_dt.month - 1 + index) % 12 + 1:02d}", {}
        ).get('status') in (STATUS_COMPLETE, STATUS_NO_DATA)
        for parameter in ALL_PARAMETERS for index in range(total_months)
    ):
        progress_bar.progress(1.0)
        stage_text.markdown("✅ پردازش هر سه شاخص به پایان رسید — ۱۰۰٪")
    else:
        st.session_state.resume_after_interruption = True
        stage_text.warning('بخشی از پایش هنوز تکمیل نشده است؛ فایل‌ها حفظ شده‌اند و می‌توانید ادامه دهید.')

    st.session_state.download_summary = download_summary

    has_any_results = any(
        bool(st.session_state.results.get(p)) for p in ALL_PARAMETERS
    )
    return has_any_results


# =============================================================================
# Legend + Management Guidance (Persian) — always visible, no jargon
# =============================================================================
def render_turbidity_guidance_panel():
    """Permanently visible legend + management guidance for Turbidity (NDTI)."""
    st.markdown("### 🎨 راهنمای رنگ و تفسیر مدیریتی — شاخص کدورت آب")

    col_legend, col_text = st.columns([1, 2])

    with col_legend:
        fig, ax = plt.subplots(figsize=(5, 0.45))
        cmap = create_turbidity_colormap()
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_xticks([0, 128, 255])
        ax.set_xticklabels(['آب شفاف', 'متوسط', 'بسیار کدر'])
        ax.set_yticks([])
        st.pyplot(fig)
        plt.close(fig)

    with col_text:
        st.markdown(
            """
**افزایش کدورت آب:**
- کاهش کیفیت آب قابل استفاده برای کشاورزی و مصارف شهری
- افزایش هزینه‌های تصفیه آب
- کاهش نفوذ نور به آب و آسیب به اکوسیستم و آبزیان
- نشانه احتمالی فرسایش خاک، رسوب‌گذاری یا آلودگی در حوضه آبریز

**کاهش کدورت آب:**
- بهبود کیفیت آب و کاهش هزینه‌های تصفیه
- شرایط مطلوب‌تر برای زیست‌بوم آبی و ماهی‌پروری
- نشانه کنترل مؤثر فرسایش و مدیریت بهتر حوضه آبریز

**چرا پایش این شاخص مهم است؟**
پایش روند کدورت به مدیران امکان می‌دهد قبل از وقوع بحران (مانند رسوب‌گذاری در سدها یا
افزایش هزینه تصفیه) اقدام کنند. تغییرات ناگهانی معمولاً نشانه رویدادهایی مانند بارش‌های
شدید، فعالیت‌های عمرانی در بالادست یا تخلیه پساب است و نیازمند بررسی سریع است.
            """
        )


def render_chlorophyll_guidance_panel():
    """Permanently visible legend + management guidance for Chlorophyll-a."""
    st.markdown("### 🎨 راهنمای رنگ و تفسیر مدیریتی — شاخص کلروفیل")

    col_legend, col_text = st.columns([1, 2])

    with col_legend:
        fig, ax = plt.subplots(figsize=(5, 0.45))
        cmap = create_chlorophyll_colormap()
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_xticks([0, 128, 255])
        ax.set_xticklabels(['کم', 'متوسط', 'بالا (شکوفایی جلبکی)'])
        ax.set_yticks([])
        st.pyplot(fig)
        plt.close(fig)

    with col_text:
        st.markdown(
            """
**افزایش غلظت کلروفیل:**
- احتمال شکوفایی جلبکی و کاهش کیفیت آب آشامیدنی
- افزایش هزینه‌های تصفیه و خطر مسدود شدن فیلترها
- کاهش اکسیژن محلول در آب و خطر برای آبزیان و ماهی‌پروری
- در موارد شدید، احتمال سمیت آب و توقف موقت برداشت آب

**کاهش غلظت کلروفیل:**
- بهبود کیفیت آب و کاهش ریسک‌های بهداشتی
- کاهش هزینه‌های عملیاتی تصفیه‌خانه
- شرایط پایدارتر برای اکوسیستم آبی

**چرا پایش این شاخص مهم است؟**
افزایش ناگهانی کلروفیل معمولاً پیش‌نشانگر شکوفایی جلبکی است که در صورت عدم اقدام به‌موقع
می‌تواند منجر به توقف تأمین آب، هزینه‌های اضطراری تصفیه یا آسیب به صنعت ماهی‌پروری شود.
پایش منظم این شاخص امکان برنامه‌ریزی پیشگیرانه و کاهش ریسک اقتصادی را فراهم می‌کند.
            """
        )


def render_cdom_guidance_panel():
    """Permanently visible legend + management guidance for CDOM."""
    st.markdown("### 🎨 راهنمای رنگ و تفسیر مدیریتی — شاخص مواد آلی محلول رنگی")

    col_legend, col_text = st.columns([1, 2])

    with col_legend:
        fig, ax = plt.subplots(figsize=(5, 0.45))
        cmap = create_cdom_colormap()
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_xticks([0, 128, 255])
        ax.set_xticklabels(['کم (آب زلال)', 'متوسط', 'زیاد (آب چای‌رنگ)'])
        ax.set_yticks([])
        st.pyplot(fig)
        plt.close(fig)

        value_range = st.session_state.get('cdom_display_range')
        if value_range:
            st.caption(
                f"محدوده رنگی این پایش: از {value_range[0]:.1f} تا {value_range[1]:.1f} "
                "(بر متر) — بر پایه‌ی داده‌های همین منطقه و بازه زمانی تعیین شده است."
            )

    with col_text:
        st.markdown(
            """
**افزایش مواد آلی محلول رنگی:**
- تیره و چای‌رنگ شدن آب و کاهش مطلوبیت آب شرب (رنگ، بو و مزه)
- افزایش مصرف مواد ضدعفونی‌کننده و خطر تشکیل فرآورده‌های جانبی گندزدایی در تصفیه‌خانه
- کاهش نفوذ نور به اعماق و تغییر شرایط زیستی بدنه آبی
- نشانه احتمالی ورود رواناب حوضه آبریز، زهکشی اراضی آلی/تالابی، یا تخلیه فاضلاب

**کاهش مواد آلی محلول رنگی:**
- شفاف‌تر شدن آب و کاهش نیاز به مواد شیمیایی در تصفیه
- کاهش ریسک بهداشتی مرتبط با فرآورده‌های جانبی گندزدایی
- نشانه کاهش ورود بار آلی از حوضه آبریز

**چرا پایش این شاخص مهم است؟**
مواد آلی محلول رنگی، برخلاف کدورت، ذرات معلق نیستند و با ته‌نشینی ساده حذف نمی‌شوند؛
بنابراین افزایش آن مستقیماً بر هزینه و پیچیدگی فرایند تصفیه آب شرب اثر می‌گذارد. این شاخص
معمولاً پس از بارش‌های سنگین و ذوب برف (ورود رواناب از خاک و پوشش گیاهی حوضه) افزایش
می‌یابد و یکی از بهترین نشانگرهای ورود بار آلی از سطح حوضه به بدنه آبی است.

*این شاخص از نسبت باند سبز به قرمز سنتینل-۲ و رابطه‌ی Brezonik و همکاران (۲۰۰۵) محاسبه
می‌شود و تقریبی از ضریب جذب مواد آلی محلول در طول موج ۴۴۰ نانومتر (بر متر) است.*
            """
        )


# =============================================================================
# Display: imagery, time-series, and statistics for one parameter
# =============================================================================
def display_side_by_side_imagery(results, parameter_type):
    """Side-by-side processed index image and corresponding RGB image."""
    if not results:
        st.info("داده‌ای برای نمایش در این بازه زمانی وجود ندارد.")
        return

    param_fa = param_persian_name(parameter_type)

    for r in results:
        mean_str = format_param_value(parameter_type, r['mean_value'], empty="بدون داده")

        cols = st.columns(2)
        cols[0].image(r['wq_image'], caption=f"{r['month_name']} — {param_fa}: {mean_str}", use_container_width=True)
        cols[1].image(r['rgb_image'], caption=f"{r['month_name']} — تصویر طبیعی (RGB)", use_container_width=True)


def _render_chart_heading(title):
    """Explicit webfont heading unaffected by Streamlit's heading theme."""
    import html as _html

    st.markdown(f'<h4 class="wq-chart-heading" dir="rtl">{_html.escape(title)}</h4>',
                unsafe_allow_html=True)


def _monthly_coverage_records(results):
    """Read the existing per-parameter coverage percentages; never infer area."""
    rows = []
    segment = 0
    for sequence, result in enumerate(results):
        month = str(result.get('month_name', '—'))
        try:
            parsed = datetime.datetime.strptime(month, '%Y-%m')
            month_date, year, month_number = parsed.strftime('%Y-%m-01'), str(parsed.year), parsed.month
        except ValueError:
            month_date, year, month_number = None, None, None
        coverage = _finite_monthly_value(result.get('water_coverage'))
        if coverage is None:
            segment += 1
        rows.append({'month': month, 'date': month_date, 'year': year,
                     'month_number': month_number, 'sequence': sequence,
                     'segment': segment, 'coverage': coverage})
    return rows


def _coverage_chart_config():
    """Shared typography for the separate coverage views."""
    return {
        'font': WQ_CHART_FONT, 'background': '#FFFFFF', 'view': {'stroke': None},
        'axis': {'labelFont': WQ_CHART_FONT, 'titleFont': WQ_CHART_FONT,
                 'labelFontSize': 12, 'titleFontSize': 14, 'titleFontWeight': 600,
                 'titlePadding': 16, 'labelColor': '#52647B', 'titleColor': '#334155',
                 'gridColor': '#E7EDF3', 'domain': False, 'tickColor': '#CED8E3'},
        'legend': {'labelFont': WQ_CHART_FONT, 'titleFont': WQ_CHART_FONT,
                   'labelFontSize': 12, 'titleFontSize': 13},
        'text': {'font': WQ_CHART_FONT},
    }


def _build_water_coverage_chart(results, parameter_type):
    """A separate presentation-only percentage chart, not a quality index."""
    rows = _monthly_coverage_records(results)
    values = [row['coverage'] for row in rows if row['coverage'] is not None]
    if not values:
        return None
    temporal = all(row['date'] is not None for row in rows)
    x = {'field': 'date' if temporal else 'month',
         'type': 'temporal' if temporal else 'ordinal', 'title': 'ماه (میلادی)',
         'axis': {'labelAngle': -35, 'grid': False, 'labelOverlap': 'greedy'}}
    if temporal:
        x['scale'] = {'type': 'utc'}
        x['axis'].update({'format': '%Y-%m', 'tickCount': min(len(rows), 12)})
    else:
        x['sort'] = None
    y = {'field': 'coverage', 'type': 'quantitative', 'title': 'پوشش آب (درصد)',
         'scale': {'domain': [min(0.0, min(values)), max(100.0, max(values))], 'nice': False},
         'axis': {'format': '.0f', 'tickCount': 6}}
    shared = {'x': x, 'y': y, 'detail': {'field': 'segment', 'type': 'nominal'},
              'order': {'field': 'sequence', 'type': 'quantitative'}}
    return {
        '$schema': 'https://vega.github.io/schema/vega-lite/v5.json',
        'description': f'پوشش آب در داده‌های {param_persian_name(parameter_type)}',
        'data': {'values': rows}, 'height': 250,
        'usermeta': {'embedOptions': {'renderer': 'svg'}},
        'padding': {'left': 12, 'right': 18, 'top': 20, 'bottom': 12},
        'layer': [
            {'transform': [{'filter': 'isValid(datum.coverage)'}],
             'mark': {'type': 'area', 'color': '#087F8C', 'opacity': 0.09,
                      'interpolate': 'linear'}, 'encoding': shared},
            {'transform': [{'filter': 'isValid(datum.coverage)'}],
             'mark': {'type': 'line', 'color': '#087F8C', 'strokeWidth': 2.5,
                      'interpolate': 'linear'}, 'encoding': shared},
            {'transform': [{'filter': 'isValid(datum.coverage)'}],
             'mark': {'type': 'point', 'filled': True, 'size': 75, 'color': '#087F8C',
                      'stroke': '#FFFFFF', 'strokeWidth': 1.2},
             'encoding': {'x': x, 'y': y, 'tooltip': [
                 {'field': 'month', 'type': 'nominal', 'title': 'ماه (میلادی)'},
                 {'field': 'coverage', 'type': 'quantitative', 'title': 'پوشش آب (درصد)', 'format': '.2f'},
             ]}},
        ],
        'config': _coverage_chart_config(),
    }


def _build_water_coverage_heatmap(results, parameter_type):
    """Calendar cells of existing observations, with blank cells for no data."""
    rows = [row for row in _monthly_coverage_records(results)
            if row['date'] is not None and row['coverage'] is not None]
    if not rows:
        return None
    values = [row['coverage'] for row in rows]
    years = list(dict.fromkeys(row['year'] for row in rows))
    month_names = json.dumps([''] + GREGORIAN_MONTHS_FA, ensure_ascii=False)
    return {
        '$schema': 'https://vega.github.io/schema/vega-lite/v5.json',
        'description': f'تقویم پوشش آب — {param_persian_name(parameter_type)}',
        # Native Streamlit fit sizing counts the axes and legend in this height.
        # Keep enough room for real calendar cells even for a single year.
        'data': {'values': rows}, 'height': max(220, 150 + len(years) * 44),
        'usermeta': {'embedOptions': {'renderer': 'svg'}},
        'mark': {'type': 'rect', 'cornerRadius': 4, 'stroke': '#FFFFFF', 'strokeWidth': 3},
        'encoding': {
            'x': {'field': 'month_number', 'type': 'ordinal', 'title': 'ماه (میلادی)',
                  'scale': {'domain': list(range(1, 13))},
                  'axis': {'labelExpr': f'{month_names}[datum.value]', 'labelAngle': -30,
                           'grid': False, 'labelLimit': 75}},
            'y': {'field': 'year', 'type': 'ordinal', 'title': 'سال (میلادی)',
                  'sort': years, 'axis': {'labelAngle': 0, 'grid': False}},
            'color': {'field': 'coverage', 'type': 'quantitative',
                      'scale': {'domain': [min(0.0, min(values)), max(100.0, max(values))],
                                'range': ['#E8F2F7', '#0B7185']},
                      'legend': {'title': 'پوشش آب (درصد)', 'orient': 'bottom', 'gradientLength': 200}},
            'tooltip': [
                {'field': 'month', 'type': 'nominal', 'title': 'ماه (میلادی)'},
                {'field': 'coverage', 'type': 'quantitative', 'title': 'پوشش آب (درصد)', 'format': '.2f'},
            ],
        },
        'config': _coverage_chart_config(),
    }


def _render_water_coverage_panel(results, parameter_type):
    """Optional coverage view; keeps the main water-quality chart quality-only."""
    with st.expander('💧 پوشش آب و تغییرات ماهانه', expanded=False):
        _render_chart_heading('روند ماهانه پوشش آب')
        st.caption(f'درصد پوشش آب در داده‌های {param_persian_name(parameter_type)} همین پایش')
        spec = _build_water_coverage_chart(results, parameter_type)
        if spec is None:
            st.info('داده معتبر پوشش آب برای این بازه زمانی در دسترس نیست.')
            return
        st.vega_lite_chart(spec=spec, use_container_width=True, theme=None)
        heatmap = _build_water_coverage_heatmap(results, parameter_type)
        if heatmap is not None:
            _render_chart_heading('تقویم پوشش آب')
            st.vega_lite_chart(spec=heatmap, use_container_width=True, theme=None)
        st.markdown(
            '<div class="wq-chart-note" dir="rtl">'
            'این درصدها همان مقادیر محاسبه‌شده پایش و سهم پیکسل‌های معتبر آب در تصاویر این شاخص هستند؛ '
            'بیانگر مساحت جغرافیایی آب یا حجم مخزن نیستند. '
            'در تقویم، خانه خالی به معنی نبود مشاهده معتبر است؛ مقدار صفر یک مشاهده واقعی است. '
            'رنگ آبی فقط میزان پوشش را نشان می‌دهد و طبقه‌بندی کیفیت آب نیست.'
            '</div>', unsafe_allow_html=True,
        )


def _build_water_quality_chart(results, parameter_type):
    """Native Streamlit/Vega-Lite presentation of unchanged monthly values.

    No satellite processing, interpolation, smoothing, normalization or area
    series is performed here. Null observations break the line instead of
    being plotted as zero. CDOM deliberately has no condition encoding.
    """
    records = _monthly_quality_records(results, parameter_type)
    values = [row['value'] for row in records if row['value'] is not None]
    if not values:
        return None

    thresholds = _quality_display_thresholds(parameter_type)
    classified = thresholds is not None
    temporal = all(row['date'] is not None for row in records)
    x_encoding = {
        'field': 'date' if temporal else 'month',
        'type': 'temporal' if temporal else 'ordinal',
        'title': 'ماه (میلادی)',
        'axis': {'labelAngle': -35, 'labelOverlap': 'greedy',
                 'labelPadding': 10, 'grid': False},
    }
    if temporal:
        x_encoding['axis'].update({'format': '%Y-%m', 'tickCount': min(len(records), 12)})
        x_encoding['scale'] = {'type': 'utc'}
    else:
        x_encoding['sort'] = None

    low, high = min(values), max(values)
    if classified:
        lower_threshold, middle_threshold, upper_threshold = thresholds
        low, high = min(low, lower_threshold), max(high, upper_threshold)
    padding = max((high - low) * 0.14, 0.08 if classified else 0.05)
    domain = [low - padding, high + padding]
    if parameter_type == PARAM_CDOM and low >= 0:
        domain[0] = max(0.0, domain[0])

    unit = 'بر متر' if parameter_type == PARAM_CDOM else 'بی‌بُعد'
    y_encoding = {
        'field': 'value', 'type': 'quantitative',
        'title': f'میانگین {param_persian_name(parameter_type)} ({unit})',
        'scale': {'domain': domain, 'zero': False, 'nice': False},
        'axis': {'format': '.3~f', 'tickCount': 6, 'labelPadding': 8},
    }
    tooltip = [
        {'field': 'month', 'type': 'nominal', 'title': 'ماه (میلادی)'},
        {'field': 'value', 'type': 'quantitative', 'title': f'مقدار ({unit})', 'format': '.6f'},
    ]
    if classified:
        tooltip.append({'field': 'condition_label', 'type': 'nominal', 'title': 'وضعیت ماه'})

    layers = []
    if classified:
        conditions = ['normal', 'warning', 'dangerous', 'critical']
        colors = [WQ_DISPLAY_CONDITIONS[key]['accent'] for key in conditions]
        # Finite visible extents are chart bounds only; readings remain untouched.
        bands = [
            {'lower': domain[0], 'upper': lower_threshold, 'condition': 'normal'},
            {'lower': lower_threshold, 'upper': middle_threshold, 'condition': 'warning'},
            {'lower': middle_threshold, 'upper': upper_threshold, 'condition': 'dangerous'},
            {'lower': upper_threshold, 'upper': domain[1], 'condition': 'critical'},
        ]
        layers.append({
            'data': {'values': bands},
            'mark': {'type': 'rect', 'opacity': 0.055, 'tooltip': None},
            'encoding': {
                'y': {'field': 'lower', 'type': 'quantitative',
                      'scale': {'domain': domain, 'zero': False, 'nice': False},
                      'axis': y_encoding['axis'], 'title': y_encoding['title']},
                'y2': {'field': 'upper'},
                'color': {'field': 'condition', 'type': 'nominal',
                          'scale': {'domain': conditions, 'range': colors},
                          'legend': None},
            },
        })
        references = [
            {'threshold': lower_threshold, 'label': 'مرز نرمال و هشدار'},
            {'threshold': middle_threshold, 'label': 'مرز هشدار و خطرناک'},
            {'threshold': upper_threshold, 'label': 'مرز خطرناک و بحرانی'},
        ]
        layers.append({
            'data': {'values': references},
            'mark': {'type': 'rule', 'stroke': '#8392A5', 'strokeWidth': 1,
                     'strokeDash': [5, 5], 'opacity': 0.7},
            'encoding': {
                'y': {'field': 'threshold', 'type': 'quantitative',
                      'axis': y_encoding['axis'], 'title': y_encoding['title']},
                'tooltip': [
                    {'field': 'label', 'type': 'nominal', 'title': 'مرز وضعیت'},
                    {'field': 'threshold', 'type': 'quantitative', 'title': 'مقدار', 'format': '.1f'},
                ],
            },
        })
        layers.append({
            'data': {'values': references},
            'mark': {'type': 'text', 'align': 'right', 'baseline': 'bottom',
                     'dx': -8, 'dy': -7, 'font': WQ_CHART_FONT,
                     'fontSize': 12, 'fontWeight': 500, 'color': '#52647B'},
            'encoding': {
                'x': {'value': 'width'},
                'y': {'field': 'threshold', 'type': 'quantitative',
                      'axis': y_encoding['axis'], 'title': y_encoding['title']},
                'text': {'field': 'label', 'type': 'nominal'},
            },
        })

    line_color = '#334E6C' if classified else '#946020'
    layers.append({
        'transform': [{'filter': 'isValid(datum.value)'}],
        'mark': {'type': 'line', 'color': line_color, 'strokeWidth': 2.5,
                 'interpolate': 'linear'},
        'encoding': {'x': x_encoding, 'y': y_encoding,
                     'detail': {'field': 'segment', 'type': 'nominal'},
                     'order': {'field': 'sequence', 'type': 'quantitative'}},
    })
    point_encoding = {'x': x_encoding, 'y': y_encoding, 'tooltip': tooltip}
    point_mark = {'type': 'point', 'filled': True, 'size': 105,
                  'stroke': '#FFFFFF', 'strokeWidth': 1.5, 'opacity': 1}
    if classified:
        point_encoding['color'] = {
            'field': 'condition', 'type': 'nominal',
            'scale': {'domain': conditions, 'range': colors},
            'legend': None,
        }
        point_encoding['shape'] = {
            'field': 'condition', 'type': 'nominal',
            'scale': {'domain': conditions,
                      'range': ['circle', 'square', 'triangle-up', 'diamond']}, 'legend': None,
        }
    else:
        point_mark['color'] = line_color
    layers.append({
        'transform': [{'filter': 'isValid(datum.value)'}],
        'mark': point_mark, 'encoding': point_encoding,
    })
    return {
        '$schema': 'https://vega.github.io/schema/vega-lite/v5.json',
        'description': f'روند ماهانه {param_persian_name(parameter_type)}',
        'data': {'values': records}, 'layer': layers,
        'height': 370, 'padding': {'left': 12, 'right': 18, 'top': 24, 'bottom': 12},
        'usermeta': {'embedOptions': {'renderer': 'svg'}},
        'autosize': {'type': 'fit', 'contains': 'padding'},
        'config': {
            'font': WQ_CHART_FONT,
            'background': '#FFFFFF', 'view': {'stroke': None},
            'title': {'font': WQ_CHART_FONT, 'fontSize': 17, 'fontWeight': 700},
            'legend': {'labelFont': WQ_CHART_FONT, 'titleFont': WQ_CHART_FONT},
            'text': {'font': WQ_CHART_FONT, 'fontSize': 13},
            'axis': {'labelFont': WQ_CHART_FONT, 'titleFont': WQ_CHART_FONT,
                     'labelFontSize': 13, 'titleFontSize': 15, 'titleFontWeight': 600,
                     'titlePadding': 20, 'labelColor': '#52647B', 'titleColor': '#334155',
                     'gridColor': '#E7EDF3', 'gridOpacity': 0.85, 'domain': False,
                     'tickColor': '#CED8E3', 'labelLimit': 160},
        },
    }


def _render_monthly_quality_table(results, parameter_type):
    """Display table only. The original Excel export (including area) is unchanged."""
    import pandas as pd

    records = _monthly_quality_records(results, parameter_type)
    columns = {
        'ماه (میلادی)': [row['month'] for row in records],
        f'میانگین {param_persian_name(parameter_type)}': [row['display_value'] for row in records],
    }
    if parameter_type in (PARAM_TURBIDITY, PARAM_CHLOROPHYLL):
        columns['وضعیت ماه'] = [row['condition_label'] for row in records]
    with st.expander('📋 جدول داده‌های ماهانه'):
        _render_persian_dataframe_html(pd.DataFrame(columns))


def _render_quality_statistics(results, parameter_type):
    """Five coordinated cards for turbidity/chlorophyll, derived on every rerun."""
    import html as _html

    overall_mean, condition = _overall_water_quality_condition(results, parameter_type)
    theme = WQ_DISPLAY_CONDITIONS[condition]
    values = [_finite_monthly_value(row.get('mean_value')) for row in results]
    values = [value for value in values if value is not None]
    coverage = [_finite_monthly_value(row.get('water_coverage')) for row in results]
    coverage = [value for value in coverage if value is not None]
    param_fa = param_persian_name(parameter_type)
    cards = [
        ('وضعیت کلی', theme['label'] if values else 'بدون داده',
         f'بر پایه {len(values)} ماه دارای داده' if values else 'داده معتبر در دسترس نیست', False),
        (f'میانگین {param_fa}', format_param_value(parameter_type, overall_mean), 'میانگین ماه‌های معتبر', True),
        (f'حداکثر {param_fa}', format_param_value(parameter_type, max(values) if values else None),
         'بیشترین میانگین ماهانه', True),
        (f'حداقل {param_fa}', format_param_value(parameter_type, min(values) if values else None),
         'کمترین میانگین ماهانه', True),
        ('میانگین پوشش آب', f'{np.mean(coverage):.1f}%' if coverage else '—', 'پوشش منطقه انتخاب‌شده', True),
    ]
    card_html = []
    for label, value, detail, numeric in cards:
        value_html = _html.escape(str(value))
        if numeric:
            value_html = f'<bdi dir="ltr">{value_html}</bdi>'
        card_html.append(
            '<article class="wq-stat-card">'
            f'<div class="wq-stat-label">{_html.escape(label)}</div>'
            f'<div class="wq-stat-value">{value_html}</div>'
            f'<div class="wq-stat-detail">{_html.escape(detail)}</div></article>'
        )
    st.markdown('#### خلاصه آماری')
    st.markdown(
        f'<div class="wq-stat-grid" dir="rtl" '
        f'style="--wq-state-accent:{theme["accent"]};--wq-state-bg:{theme["background"]};'
        f'--wq-state-border:{theme["border"]};">{"".join(card_html)}</div>',
        unsafe_allow_html=True,
    )
    if not values:
        st.info('داده معتبر کیفیت آب در این بازه زمانی در دسترس نیست.')
    if results:
        _render_monthly_quality_table(results, parameter_type)


def display_time_series_chart(results, parameter_type):
    """Interactive, Persian, quality-only monthly chart; CDOM is unclassified."""
    _render_chart_heading(f'روند زمانی {param_persian_name(parameter_type)}')
    thresholds = _quality_display_thresholds(parameter_type)
    classified = thresholds is not None
    if classified:
        lower_text, middle_text, upper_text = (f'{value:g}'.replace('-', '−') for value in thresholds)
        intervals = [
            ('normal', f'≤ {lower_text}'),
            ('warning', f'({lower_text}, {middle_text}]'),
            ('dangerous', f'({middle_text}, {upper_text}]'),
            ('critical', f'&gt; {upper_text}'),
        ]
        items = []
        for condition, interval in intervals:
            theme = WQ_DISPLAY_CONDITIONS[condition]
            items.append(
                '<span class="wq-condition-item">'
                f'<span class="wq-condition-dot" style="color:{theme["accent"]}">{theme["symbol"]}</span>'
                f'<span>{theme["label"]}</span> <bdi dir="ltr">{interval}</bdi></span>'
            )
        st.markdown(
            '<div class="wq-condition-legend" dir="rtl" aria-label="راهنمای وضعیت ماهانه">'
            + ''.join(items) + '</div>',
            unsafe_allow_html=True,
        )
    spec = _build_water_quality_chart(results, parameter_type)
    if spec is None:
        st.info('در این بازه زمانی، داده معتبر کیفیت آب برای نمایش نمودار موجود نیست.')
        return
    st.vega_lite_chart(spec=spec, use_container_width=True, theme=None)
    st.caption('برای مشاهده ماه، مقدار دقیق و وضعیت ماهانه، نشانگر را روی هر نقطه قرار دهید.'
               if classified else 'برای مشاهده ماه و مقدار دقیق مواد آلی محلول، نشانگر را روی هر نقطه قرار دهید.')
    if classified:
        st.markdown(
            '<div class="wq-chart-note" dir="rtl">'
            'رنگ نشانگرها وضعیت هر ماه را نشان می‌دهد؛ وضعیت کلی بر پایه میانگین ماه‌های معتبر است. '
            'طبقه‌بندی با مقادیر کامل و پیش از گرد کردن انجام می‌شود. ماه‌های بدون داده با صفر جایگزین نمی‌شوند. '
            'این راهنما مربوط به نمودار ماهانه است؛ رنگ تصاویر ماهواره‌ای مطابق راهنمای تصاویر باقی می‌ماند.'
            '</div>', unsafe_allow_html=True,
        )


def display_statistics_summary(results, parameter_type):
    """Statistics summary, contained within the same parameter page."""
    if parameter_type in (PARAM_TURBIDITY, PARAM_CHLOROPHYLL):
        _render_quality_statistics(results, parameter_type)
        return

    # CDOM retains the four original cards and their original calculations.
    if not results:
        return

    param_fa = param_persian_name(parameter_type)

    months = [r['month_name'] for r in results]
    mean_values = [r['mean_value'] if not np.isnan(r['mean_value']) else 0 for r in results]
    coverage_values = [r['water_coverage'] for r in results]
    valid_values = [m for m in mean_values if m != 0]

    st.markdown("#### 📈 خلاصه آماری")

    col1, col2, col3, col4 = st.columns(4)

    if valid_values:
        col1.metric(f"میانگین {param_fa}", format_param_value(parameter_type, np.mean(valid_values)))
        col2.metric(f"حداکثر {param_fa}", format_param_value(parameter_type, np.max(valid_values)))
        col3.metric(f"حداقل {param_fa}", format_param_value(parameter_type, np.min(valid_values)))
    else:
        col1.metric(f"میانگین {param_fa}", "—")
        col2.metric(f"حداکثر {param_fa}", "—")
        col3.metric(f"حداقل {param_fa}", "—")

    col4.metric("میانگین پوشش آب", f"{np.mean(coverage_values):.1f}%")

    _render_monthly_quality_table(results, parameter_type)


def generate_combined_timeseries_excel():
    """
    Build a single Excel (.xlsx) workbook containing the monthly time-series
    values for ALL monitored parameters — Turbidity (NDTI), Chlorophyll (NDCI)
    and CDOM — as one sheet each. Returns workbook bytes for st.download_button.
    """
    import io
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)  # drop the default empty sheet

    header_font = Font(name='Arial', bold=True, color='FFFFFF')
    header_fill = PatternFill(start_color='1F77B4', end_color='1F77B4', fill_type='solid')
    body_font = Font(name='Arial')
    center = Alignment(horizontal='center')

    center_lat, center_lon = _get_roi_center_coordinates()
    lat_out = round(float(center_lat), 6) if center_lat is not None else "—"
    lon_out = round(float(center_lon), 6) if center_lon is not None else "—"

    sections = [
        (PARAM_TURBIDITY, "کدورت (NDTI)", "میانگین NDTI"),
        (PARAM_CHLOROPHYLL, "کلروفیل (NDCI)", "میانگین NDCI"),
        (PARAM_CDOM, "مواد آلی محلول (CDOM)", "میانگین CDOM"),
    ]

    for parameter_type, sheet_name, value_header in sections:
        ws = wb.create_sheet(title=sheet_name)
        ws.sheet_view.rightToLeft = True

        headers = ["ماه", value_header, "پوشش آب (%)", "عرض جغرافیایی مرکز", "طول جغرافیایی مرکز"]
        ws.append(headers)
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center

        results = st.session_state.results.get(parameter_type, [])
        for r in sorted(results, key=lambda x: x['month_name']):
            mean_val = r['mean_value']
            mean_out = round(float(mean_val), 4) if not np.isnan(mean_val) else "بدون داده"
            ws.append([r['month_name'], mean_out, round(float(r['water_coverage']), 1), lat_out, lon_out])

        if not results:
            ws.cell(row=2, column=1, value="داده‌ای موجود نیست.").font = body_font
        else:
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.font = body_font
                    cell.alignment = center

        column_widths = [14, 18, 16, 20, 20]
        for i, w in enumerate(column_widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w

        ws.freeze_panes = "A2"

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def _get_roi_center_coordinates():
    """
    Return (lat, lon) of the center (centroid) of the region of interest used
    for the current/most recent monitoring run, based on the polygon stored
    in st.session_state.processing_config. Returns (None, None) if no run has
    been configured yet.
    """
    config = st.session_state.get('processing_config')
    if not config or not config.get('polygon_coords'):
        return None, None
    try:
        centroid = Polygon(config['polygon_coords']).centroid
        return centroid.y, centroid.x
    except Exception:
        return None, None


def render_parameter_page(parameter_type):
    """
    Full page for one parameter, in the required order:
    1. Statistics Summary (خلاصه آماری)
    2. Legend + Management Guidance Panel
    3. Time-series chart
    4. Side-by-side imagery (collapsible)
    """
    if parameter_type == PARAM_TURBIDITY:
        _render_active_section_badge("🌊", "کدورت آب", "#0B6E76", "#2FC2CE")
    elif parameter_type == PARAM_CDOM:
        _render_active_section_badge("🍂", "مواد آلی محلول رنگی", "#7A4A12", "#D9A05B")
    else:
        _render_active_section_badge("🌿", "شاخص کلروفیل", "#1B7A3D", "#4CC26B")

    results = st.session_state.results.get(parameter_type, [])

    # Expert brief for THIS parameter, at the very top of the page.
    # Generated once per parameter per monitoring run and then cached.
    if results:
        _render_param_brief(parameter_type)

    if results or parameter_type in (PARAM_TURBIDITY, PARAM_CHLOROPHYLL):
        display_statistics_summary(results, parameter_type)
        st.divider()

    if parameter_type == PARAM_TURBIDITY:
        render_turbidity_guidance_panel()
    elif parameter_type == PARAM_CDOM:
        render_cdom_guidance_panel()
    else:
        render_chlorophyll_guidance_panel()

    st.divider()

    if not results:
        st.info("برای مشاهده نتایج، ابتدا یک منطقه را انتخاب و پایش را اجرا کنید.")
        return

    # Time-series chart first (right after the legend), processed imagery after
    display_time_series_chart(results, parameter_type)

    _render_water_coverage_panel(results, parameter_type)

    st.divider()

    with st.expander("🖼️ تصاویر پردازش‌شده (برای نمایش/پنهان‌سازی کلیک کنید)", expanded=False):
        display_side_by_side_imagery(results, parameter_type)


# =============================================================================
# نظر متخصص آب — Statistical Analysis Pipeline
# (ported as-is from the separate analysis notebook: Mann-Kendall trend test,
#  MAD-based anomaly detection, seasonal climatology, cross-parameter
#  correlation — only the input source changes: in-memory Excel bytes
#  instead of a file path, so nothing needs to be written to disk on Posit.)
# =============================================================================
def _expert_compute_trend(values, n):
    """Mann-Kendall trend test: standard (always) + seasonal (once >=24 months exist)."""
    import pymannkendall as mk

    result = {}

    if n < 4:
        result["standard"] = {"available": False, "reason": f"Mann-Kendall needs at least ~4 points; only {n} available."}
        result["seasonal"] = {"available": False, "reason": "Not enough data."}
        return result

    mk_res = mk.original_test(values)
    result["standard"] = {
        "method": "Mann-Kendall (non-seasonal)",
        "trend": mk_res.trend,
        "significant": bool(mk_res.h),
        "p_value": float(mk_res.p),
        "tau": float(mk_res.Tau),
        "sen_slope_per_month": float(mk_res.slope),
        "note": ("Does not separate the seasonal cycle from the trend. "
                 "If a strong seasonal pattern exists, prefer the seasonal result below when available.")
    }

    if n >= 24:
        try:
            smk_res = mk.seasonal_test(values, period=12)
            result["seasonal"] = {
                "method": "Seasonal Mann-Kendall (Hirsch-Slack), period=12 months",
                "trend": smk_res.trend,
                "significant": bool(smk_res.h),
                "p_value": float(smk_res.p),
                "sen_slope_per_month": float(smk_res.slope),
                "replicates_per_season": round(n / 12, 1),
                "power_caution": (f"Only ~{n/12:.1f} years of data per calendar month are available. "
                                   "Seasonal Mann-Kendall has low statistical power below ~3-4 years; "
                                   "treat this result as directional, not conclusive.")
            }
        except Exception as e:
            result["seasonal"] = {"available": False, "reason": f"Could not compute: {e}"}
    else:
        result["seasonal"] = {
            "available": False,
            "reason": f"Needs at least 24 months (2 full years) to run; only {n} months available."
        }

    return result


def _expert_detect_anomalies_mad(df, date_col, value_column, threshold=3.5):
    """Median Absolute Deviation based anomaly detection."""
    values = df[value_column]
    median = values.median()
    mad = np.median(np.abs(values - median))

    if mad == 0:
        return []

    modified_z = 0.6745 * (values - median) / mad

    anomalies = []
    for idx, row in df.iterrows():
        if abs(modified_z[idx]) > threshold:
            anomalies.append({
                "date": row[date_col].strftime("%Y-%m"),
                "value": float(row[value_column]),
                "modified_z_score": round(float(modified_z[idx]), 2)
            })
    return anomalies


def _expert_analyze_sheet(df, value_column, date_col=None, water_col=None):
    import pandas as pd

    date_col = date_col or df.columns[0]
    water_col = water_col or df.columns[-1]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    values = df[value_column]
    n = len(values)

    raw_observations = [
        {
            "date": row[date_col].strftime("%Y-%m"),
            "value": float(row[value_column]),
            "water_coverage_pct": float(row[water_col])
        }
        for _, row in df.iterrows()
    ]

    trend = _expert_compute_trend(values.values, n)
    anomalies = _expert_detect_anomalies_mad(df, date_col, value_column)

    df["MonthNumber"] = df[date_col].dt.month
    seasonal_climatology = df.groupby("MonthNumber")[value_column].mean().round(4).to_dict()

    summary = {
        "period": {
            "start": df[date_col].min().strftime("%Y-%m"),
            "end": df[date_col].max().strftime("%Y-%m"),
            "months": int(n)
        },
        "raw_observations": raw_observations,
        "statistics": {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "variance": float(values.var()),
            "min": float(values.min()),
            "max": float(values.max()),
            "range": float(values.max() - values.min())
        },
        "trend": trend,
        "extremes": {
            "minimum": {"date": df.loc[values.idxmin(), date_col].strftime("%Y-%m"), "value": float(values.min())},
            "maximum": {"date": df.loc[values.idxmax(), date_col].strftime("%Y-%m"), "value": float(values.max())}
        },
        "seasonal_climatology": seasonal_climatology,
        "water_coverage": {
            "mean_percent": float(df[water_col].mean()),
            "minimum_percent": float(df[water_col].min()),
            "maximum_percent": float(df[water_col].max())
        },
        "missing_values": int(values.isna().sum()),
        "anomalies": anomalies
    }
    return summary


def _expert_compute_correlation_by_date(df1, value_col1, df2, value_col2, date_col1=None, date_col2=None):
    """Correlate two parameter series matched by actual date, not row position."""
    import pandas as pd

    date_col1 = date_col1 or df1.columns[0]
    date_col2 = date_col2 or df2.columns[0]

    d1 = df1[[date_col1, value_col1]].copy()
    d1[date_col1] = pd.to_datetime(d1[date_col1])
    d1 = d1.rename(columns={date_col1: "date", value_col1: "v1"})

    d2 = df2[[date_col2, value_col2]].copy()
    d2[date_col2] = pd.to_datetime(d2[date_col2])
    d2 = d2.rename(columns={date_col2: "date", value_col2: "v2"})

    merged = pd.merge(d1, d2, on="date", how="inner")
    unmatched_1 = set(d1["date"]) - set(merged["date"])
    unmatched_2 = set(d2["date"]) - set(merged["date"])

    return {
        "pearson_correlation": float(merged["v1"].corr(merged["v2"])),
        "n_matched_dates": int(len(merged)),
        "unmatched_dates_first_only": sorted(d.strftime("%Y-%m") for d in unmatched_1),
        "unmatched_dates_second_only": sorted(d.strftime("%Y-%m") for d in unmatched_2),
    }


def analyze_water_quality_from_bytes(excel_bytes):
    """
    Same logic as the standalone analyze_water_quality(excel_file) from the
    analysis notebook, adapted to read the in-memory Excel bytes produced by
    generate_combined_timeseries_excel() instead of a file on disk.

    Robustness addition (not present in the original notebook): a month with
    no data is written into the Excel export as the text "بدون داده" rather
    than a number, which would otherwise turn the whole column non-numeric
    and silently drop that parameter from the analysis. Those cells are
    coerced to NaN and excluded here instead.

    Also mirrors the notebook's latest addition: the center coordinates of
    the region of interest (columns "عرض جغرافیایی مرکز" / "طول جغرافیایی
    مرکز" in the exported Excel) are lifted into top-level
    "center_latitude" / "center_longitude" keys of the returned summary, so
    the location-aware chat agent can use them directly (e.g. to fetch
    historical weather for that exact point) without re-parsing the sheet.
    """
    import io
    import pandas as pd

    excel_buffer = io.BytesIO(excel_bytes)
    xls = pd.ExcelFile(excel_buffer)
    sheet_names = xls.sheet_names
    results = {}
    cleaned_frames = {}

    for sheet in sheet_names:
        df = pd.read_excel(excel_buffer, sheet_name=sheet)
        numeric_like_cols = [c for c in df.columns if c != df.columns[0]]
        for c in numeric_like_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=numeric_like_cols[:1])  # drop rows with no value for the main indicator

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2 or df.empty:
            continue
        value_col = numeric_cols[0]
        water_col = "پوشش آب (%)" if "پوشش آب (%)" in df.columns else None
        results[sheet] = _expert_analyze_sheet(df, value_col, water_col=water_col)
        cleaned_frames[sheet] = df

    # Correlations between every pair of parameters (turbidity ↔ chlorophyll,
    # turbidity ↔ CDOM, chlorophyll ↔ CDOM), each matched by calendar month.
    if len(cleaned_frames) >= 2:
        sheets = list(cleaned_frames.keys())
        relationships = {}
        for i in range(len(sheets)):
            for j in range(i + 1, len(sheets)):
                sheet_a, sheet_b = sheets[i], sheets[j]
                df1, df2 = cleaned_frames[sheet_a], cleaned_frames[sheet_b]
                try:
                    col1 = df1.select_dtypes(include=np.number).columns[0]
                    col2 = df2.select_dtypes(include=np.number).columns[0]
                    relationships[f"{sheet_a} ↔ {sheet_b}"] = \
                        _expert_compute_correlation_by_date(df1, col1, df2, col2)
                except Exception:
                    continue
        if relationships:
            results["relationships"] = relationships

    # --- Lift region center coordinates to the top level (for the chat agent) ---
    if cleaned_frames:
        first_sheet_df = list(cleaned_frames.values())[0]
        lat_col = "عرض جغرافیایی مرکز"
        lon_col = "طول جغرافیایی مرکز"
        if lat_col in first_sheet_df.columns and lon_col in first_sheet_df.columns:
            lat_vals = pd.to_numeric(first_sheet_df[lat_col], errors='coerce').dropna()
            lon_vals = pd.to_numeric(first_sheet_df[lon_col], errors='coerce').dropna()
            if not lat_vals.empty and not lon_vals.empty:
                results["center_latitude"] = float(lat_vals.iloc[0])
                results["center_longitude"] = float(lon_vals.iloc[0])

    return results


def _expert_results_signature():
    """Cheap signature used to detect when monitoring results changed, so the
    JSON summary + chat history for نظر متخصص آب can be refreshed automatically."""
    sig = []
    for p in ALL_PARAMETERS:
        results = st.session_state.results.get(p, [])
        sig.append(tuple(sorted(
            (r['month_name'], None if np.isnan(r['mean_value']) else round(float(r['mean_value']), 6))
            for r in results
        )))
    return tuple(sig)


# =============================================================================
# نظر متخصص آب — LLM Agent Chat (LangGraph ReAct agent, OpenAI-compatible)
# =============================================================================
# NOTE ON CREDENTIALS: this app is hosted on a Posit server that cannot read a
# local .env file, so the API base URL / keys (OpenAI-compatible LLM, Tavily,
# OpenWeatherMap) are all hardcoded below and read directly from this file
# instead of being loaded through python-dotenv or environment variables, per
# explicit request. If this project's git repo is ever shared or made public,
# consider moving these values to Posit Connect's own "Environment Variables"
# panel (Settings → Vars) instead of leaving live keys in source — that still
# avoids the .env problem without exposing the keys in version control.
OPENAI_BASE_URL = "https://api.avalai.ir/v1"
OPENAI_API_KEY = "aa-xLsSw3ad4txKKuwvHY7cPGy1StemeS3xtuChVf9utHKOd3Cr"
EXPERT_CHAT_MODEL = "gpt-5.2"  # change to whichever model your endpoint provides

# Tavily web search key, used by the agent's general-purpose web search tool
# (qualitative/general climate context). Hardcoded directly here — read from
# the code, not from a .env file — because the Posit server this app is
# deployed on cannot read a local .env file.
TAVILY_API_KEY = "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"

# OpenWeatherMap Geocoding API key, used by the agent's reverse-geocoding
# tool (lat/lon -> city/region/country name). Also hardcoded directly here
# for the same Posit-server reason as above.
OPENWEATHER_API_KEY = "5fce9bd0bcda8e2cd43468bf50755c82"


# =============================================================================
# خلاصه مدیریتی — the fixed question asked once, to the SAME expert agent
# =============================================================================
# The «خلاصه مدیریتی» page reuses the expert agent exactly as the chat page does
# — identical system prompt, identical analysis JSON, identical tools. The only
# difference is that instead of the user typing a question, this one fixed
# question is sent once and its answer is cached.
#
# ---------------------------------------------------------------------------
# THIS TEXT IS THE ONE THING TO EDIT if you want a different style of summary.
# Everything else on that page is just plumbing and presentation.
# ---------------------------------------------------------------------------
EXECUTIVE_SUMMARY_PROMPT = """یک «خلاصه مدیریتی» از وضعیت کیفیت آب این پهنه آبی بنویس.

قالب و محدودیت‌ها:
- دقیقاً بین ۵ تا ۶ خط بنویس. هر خط یک جمله کوتاه و مستقل باشد (حداکثر حدود ۲۰ کلمه).
- بدون عنوان، بدون تیتر، بدون فهرست شماره‌دار، بدون جدول، و بدون هیچ مقدمه‌ای مانند «بر اساس داده‌های ارائه‌شده».
- هر خط را با یک ایموجی مرتبط شروع کن.
- زبان فارسی ساده و روان برای یک مدیر غیرمتخصص باشد؛ هیچ اصطلاح فنی، مخفف انگلیسی یا نام شاخص
  (NDTI، NDCI، CDOM، من-کندال، انحراف مطلق از میانه و مانند آن) به کار نبر. به‌جای آن‌ها بنویس
  «گل‌آلودگی آب»، «رشد جلبک» و «مواد آلی محلول در آب».
- لحن حرفه‌ای اما جذاب و خواندنی باشد؛ طوری که خواننده در همین چند خط، تصویری روشن و در عین حال
  جالب از وضعیت این پهنه آبی به دست بیاورد و احساس کند چیز تازه‌ای یاد گرفته است.

محتوای اجباری، به همین ترتیب:
۱. نام منطقه (آن را با ابزار reverse_geocode از مختصات مرکز به دست بیاور) به‌همراه یک قضاوت کلی از
   کیفیت آب در کل بازه پایش: خوب، قابل قبول، یا نگران‌کننده.
۲. مهم‌ترین یا جالب‌ترین یافته‌ی این پایش؛ چیزی که برای خواننده تازگی دارد یا انتظارش را ندارد.
۳. بدترین ماه یا فصل دوره و اینکه در آن زمان کدام جنبه‌ی کیفیت آب به اوج رسیده است؛ نام ماه را حتماً
   به تقویم شمسی بنویس.
۴. روند بلندمدت کیفیت آب در کل دوره با در نظر گرفتن هر سه جنبه: رو به بهبود، بدون تغییر معنادار، یا
   رو به بدتر شدن.
۵. یک توصیه‌ی عملی و مشخص برای مدیر؛ مثلاً بهترین زمان پایش میدانی، آمادگی تصفیه‌خانه، یا بررسی
   منابع ورودی در حوضه آبریز.

قواعد:
- فقط بر پایه‌ی داده‌های تحلیل موجود بنویس؛ هیچ عدد، ماه یا رویدادی را از خودت نساز.
- اگر داده‌ی یکی از جنبه‌ها برای نتیجه‌گیری کافی نیست، آن را کوتاه و صادقانه بگو، اما خلاصه را از ۶ خط
  بلندتر نکن.
- عدد کم بیاور؛ فقط جایی که واقعاً به درک مدیر کمک می‌کند."""


# =============================================================================
# تحلیل کارشناسی بالای هر صفحه پارامتر — one brief per parameter page
# =============================================================================
# Same expert agent, same system prompt, same analysis JSON as the chat page —
# only the question differs. One brief per parameter, generated the first time
# that page is opened after the data or its prompt changes and then cached,
# so switching between pages reuses the current brief.
#
# Edit these three prompts to change the short, plain-language Persian summary
# shown on each parameter page. Keep the same length and style on all three.
# ---------------------------------------------------------------------------

TURBIDITY_PAGE_SUMMARY_PROMPT = """خلاصه‌ای کوتاه از وضعیت «گل‌آلودگی آب» در این پهنه آبی برای یک کاربر غیرمتخصص بنویس.

قالب و زبان:
- فقط ۲ تا ۳ خط کوتاه فارسی بنویس؛ هر خط یک جمله کامل با حداکثر حدود ۲۰ واژه باشد و با شکست خط جدا شود.
- لحن رسمی، ساده و روان باشد؛ از واژه‌های روزمره استفاده کن تا بدون دانش تخصصی کیفیت آب قابل فهم باشد.
- بدون عنوان، مقدمه، فهرست، شماره‌گذاری، جدول یا ایموجی بنویس؛ اصطلاح آماری، نام روش و مخفف انگلیسی به کار نبر.

محتوا و قواعد:
- وضعیت کلی گل‌آلودگی آب و تغییر آن در دوره پایش را با زبان توصیفی بیان کن؛ اگر روند روشنی وجود ندارد، آن را ساده بگو.
- اگر الگوی فصلی یا دوره قابل توجهی در داده‌ها روشن است، فقط مهم‌ترین نکته را کوتاه بیان کن؛ نام ماه را شمسی بنویس.
- در همین ۲ تا ۳ خط، در صورت نیاز یک پیشنهاد عملی ساده و متناسب با شواهد برای پایش گل‌آلودگی آب بیاور.
- میانگین، میانه، کمینه، بیشینه، دامنه اعداد و نتایج آزمون‌های آماری را گزارش نکن. معمولاً هیچ عددی نیاور؛ فقط اگر برای فهم وضعیت ضروری است، حداکثر یک عدد ذکر کن.
- فقط از داده‌های تحلیل و نتایج ابزارها استفاده کن؛ علت، رویداد، نوع یا کاربرد پهنه آبی را حدس نزن و از این شاخص به‌تنهایی درباره سلامت یا آشامیدنی بودن آب حکم نده.
- اگر داده کافی نیست، محدودیت را در همان ۲ تا ۳ خط ساده و صریح بگو؛ متن را طولانی نکن و درباره دو پارامتر دیگر توضیح جداگانه نده."""


CHLOROPHYLL_PAGE_SUMMARY_PROMPT = """خلاصه‌ای کوتاه از وضعیت «رشد جلبک» در این پهنه آبی برای یک کاربر غیرمتخصص بنویس.

قالب و زبان:
- فقط ۲ تا ۳ خط کوتاه فارسی بنویس؛ هر خط یک جمله کامل با حداکثر حدود ۲۰ واژه باشد و با شکست خط جدا شود.
- لحن رسمی، ساده و روان باشد؛ از واژه‌های روزمره استفاده کن تا بدون دانش تخصصی کیفیت آب قابل فهم باشد.
- بدون عنوان، مقدمه، فهرست، شماره‌گذاری، جدول یا ایموجی بنویس؛ اصطلاح آماری، نام روش و مخفف انگلیسی به کار نبر.

محتوا و قواعد:
- وضعیت کلی رشد جلبک و تغییر آن در دوره پایش را با زبان توصیفی بیان کن؛ اگر روند روشنی وجود ندارد، آن را ساده بگو.
- اگر فصل یا دوره افزایش رشد جلبک در داده‌ها روشن است، فقط مهم‌ترین نکته را کوتاه بیان کن؛ نام ماه را شمسی بنویس.
- در همین ۲ تا ۳ خط، در صورت نیاز یک پیشنهاد عملی ساده و متناسب با شواهد برای پایش رشد جلبک بیاور.
- میانگین، میانه، کمینه، بیشینه، دامنه اعداد و نتایج آزمون‌های آماری را گزارش نکن. معمولاً هیچ عددی نیاور؛ فقط اگر برای فهم وضعیت ضروری است، حداکثر یک عدد ذکر کن.
- فقط از داده‌های تحلیل و نتایج ابزارها استفاده کن؛ علت، رویداد، نوع یا کاربرد پهنه آبی را حدس نزن و از این شاخص به‌تنهایی درباره سمی بودن جلبک یا آشامیدنی بودن آب حکم نده.
- اگر داده کافی نیست، محدودیت را در همان ۲ تا ۳ خط ساده و صریح بگو؛ متن را طولانی نکن و درباره دو پارامتر دیگر توضیح جداگانه نده."""


CDOM_PAGE_SUMMARY_PROMPT = """خلاصه‌ای کوتاه از وضعیت «مواد آلی محلول در آب» در این پهنه آبی برای یک کاربر غیرمتخصص بنویس.

قالب و زبان:
- فقط ۲ تا ۳ خط کوتاه فارسی بنویس؛ هر خط یک جمله کامل با حداکثر حدود ۲۰ واژه باشد و با شکست خط جدا شود.
- لحن رسمی، ساده و روان باشد؛ از واژه‌های روزمره استفاده کن تا بدون دانش تخصصی کیفیت آب قابل فهم باشد.
- بدون عنوان، مقدمه، فهرست، شماره‌گذاری، جدول یا ایموجی بنویس؛ اصطلاح آماری، نام روش و مخفف انگلیسی به کار نبر.

محتوا و قواعد:
- وضعیت کلی مواد آلی محلول در آب و تغییر آن در دوره پایش را با زبان توصیفی بیان کن؛ اگر روند روشنی وجود ندارد، آن را ساده بگو.
- اگر دوره افزایش این مواد در داده‌ها روشن است، فقط مهم‌ترین نکته و اثر احتمالی آن بر رنگ آب را کوتاه بیان کن؛ نام ماه را شمسی بنویس.
- در همین ۲ تا ۳ خط، در صورت نیاز یک پیشنهاد عملی ساده و متناسب با شواهد برای پایش این مواد بیاور.
- میانگین، میانه، کمینه، بیشینه، دامنه اعداد و نتایج آزمون‌های آماری را گزارش نکن. معمولاً هیچ عددی نیاور؛ فقط اگر برای فهم وضعیت ضروری است، حداکثر یک عدد ذکر کن.
- فقط از داده‌های تحلیل و نتایج ابزارها استفاده کن؛ علت، رویداد، نوع یا کاربرد پهنه آبی را حدس نزن و از این شاخص به‌تنهایی درباره آلودگی قطعی یا آشامیدنی بودن آب حکم نده.
- اگر داده کافی نیست، محدودیت را در همان ۲ تا ۳ خط ساده و صریح بگو؛ متن را طولانی نکن و درباره دو پارامتر دیگر توضیح جداگانه نده."""


PARAM_PAGE_SUMMARY_PROMPTS = {
    PARAM_TURBIDITY:   TURBIDITY_PAGE_SUMMARY_PROMPT,
    PARAM_CHLOROPHYLL: CHLOROPHYLL_PAGE_SUMMARY_PROMPT,
    PARAM_CDOM:        CDOM_PAGE_SUMMARY_PROMPT,
}

# Card theme (CSS class) + icon + title for the brief on each parameter page.
PARAM_BRIEF_THEME = {
    PARAM_TURBIDITY:   ("wq-brief-turbidity",   "🧠",
                        "تحلیل کارشناسی هوش مصنوعی — گل‌آلودگی آب"),
    PARAM_CHLOROPHYLL: ("wq-brief-chlorophyll", "🧠",
                        "تحلیل کارشناسی هوش مصنوعی — رشد جلبک"),
    PARAM_CDOM:        ("wq-brief-cdom",        "🧠",
                        "تحلیل کارشناسی هوش مصنوعی — مواد آلی محلول در آب"),
}


# =============================================================================
# دکمه‌های آماده صفحه چت — fixed prompts behind the quick-action buttons
# =============================================================================
# Pressing one of these sends the prompt to exactly the same expert agent the
# typed questions go to. Edit the text here to change what each button asks.

QUICK_BRIEF_PROMPT = """در حداکثر ۳ تا ۴ خط و با زبانی کاملاً ساده و غیرتخصصی، وضعیت کلی کیفیت آب این منطقه را توضیح بده.

- بدون عنوان، بدون فهرست شماره‌دار و بدون هیچ اصطلاح فنی یا مخفف انگلیسی بنویس؛ طوری که یک فرد غیرمتخصص
  به‌راحتی متوجه شود.
- فقط تصویر کلی را بگو: آب این پهنه در مجموع در چه وضعیتی است، مهم‌ترین نکته‌ای که باید بدانیم چیست، و
  آیا جای نگرانی هست یا نه.
- اگر ماه یا فصل خاصی وضعیت متفاوت یا بدتری داشته، در همین چند خط کوتاه به آن اشاره کن.
- عدد کم بیاور و هیچ عدد، ماه یا رویدادی را از خودت نساز."""


QUICK_DEEP_PROMPT = """یک تحلیل کامل، دقیق و حرفه‌ای از کیفیت آب این منطقه ارائه بده و هر آنچه را که از داده‌های موجود قابل استخراج است پوشش بده.

پاسخ را با عنوان‌های کوتاه و در این ساختار بنویس:
۱. معرفی منطقه و بازه پایش — نام منطقه را با ابزار reverse_geocode از مختصات مرکز به دست بیاور.
۲. وضعیت هر سه جنبه کیفیت آب به‌صورت جداگانه: میانگین، دامنه تغییرات، بیشینه و کمینه و ماهی که در آن رخ داده است.
۳. روند بلندمدت هر جنبه بر پایه آزمون روند: جهت روند، معنادار بودن یا نبودن آن، و شیب تغییر.
۴. الگوی فصلی: کدام ماه‌ها یا فصل‌ها به‌طور سیستماتیک بدترین و کدام بهترین وضعیت را دارند.
۵. ناهنجاری‌ها: ماه‌های غیرعادی را فهرست کن و برای حداکثر چهار ماه مهم و متمایز با ابزار داده‌های هواشناسی، علت محتمل
   (بارش شدید، ذوب برف، باد، دمای بالا) را بررسی و بیان کن.
۶. رابطه میان جنبه‌های کیفیت آب: همبستگی‌های موجود را تفسیر کن و توضیح بده چه فرایند محیطی می‌تواند آن را توجیه کند.
۷. کیفیت و محدودیت داده: درصد پوشش آب، ماه‌های بدون داده، و هر محدودیتی که نتیجه‌گیری را محتاطانه می‌کند.
۸. جمع‌بندی و توصیه‌های عملی برای بهره‌بردار و مدیر، همراه با اولویت‌بندی.

همه چیز را فقط بر پایه داده‌های تحلیل موجود و نتایج ابزارها بنویس و هیچ عدد، ماه یا رویدادی را از خودت نساز.
پاسخ باید کاملاً فارسی و بدون مخفف یا واژه انگلیسی باشد."""


def _build_agent_system_prompt(analysis_json):
    """
    Persian system prompt for the water-quality expert agent. Combines the
    original evidence-based / no-hallucination rules with instructions for
    when to use each of the three available tools (reverse geocoding, web
    search, historical weather). The statistical analysis JSON (including,
    when available, center_latitude / center_longitude of the monitored
    region) is embedded directly in the prompt, exactly as in the previous
    non-agent version — it is NOT exposed as a separate callable tool.
    """
    return f"""شما یک متخصص باتجربه در زمینه کیفیت آب، سنجش‌ازدور ماهواره‌ای (سنتینل-۲)، و اقلیم‌شناسی هستید.

در ادامه، خلاصه‌ی تحلیل آماری سری زمانی **سه شاخص** کیفیت آب یک بدنه‌ی آبی، به‌صورت JSON در اختیار
شما قرار گرفته است. این خلاصه شامل مختصات مرکز منطقه (کلیدهای center_latitude و center_longitude،
در صورت وجود)، نتیجه‌ی آزمون روند من-کندال، ناهنجاری‌های شناسایی‌شده (بر پایه‌ی انحراف مطلق از
میانه)، الگوی فصلی چندساله، آمار توصیفی، و همبستگی دوبه‌دوی شاخص‌ها (کلید relationships) است.

سه شاخص موجود در داده‌ها (هر سه روی دقیقاً یک ماسک پهنه‌ی آبی و یک زنجیره‌ی پیش‌پردازش یکسان —
حذف ابر، حذف برف، و استخراج پهنه آب با AWEIsh — محاسبه شده‌اند و بنابراین مستقیماً با هم قابل
مقایسه‌اند):

۱. «شاخص کدورت آب» (برگه‌ی «کدورت (NDTI)» در داده‌ها) — اختلاف نرمال‌شده‌ی باند قرمز و سبز؛ بی‌بعد و
   در بازه‌ی ۱- تا ۱+. افزایش آن یعنی آب گل‌آلودتر و ذرات معلق بیشتر (فرسایش خاک، رسوب، رواناب،
   فعالیت عمرانی بالادست).

۲. «شاخص کلروفیل» (برگه‌ی «کلروفیل (NDCI)») — اختلاف نرمال‌شده‌ی باند لبه‌ی قرمز و قرمز؛ بی‌بعد و در
   بازه‌ی ۱- تا ۱+. افزایش آن یعنی زیست‌توده‌ی جلبکی/فیتوپلانکتونی بیشتر و احتمال شکوفایی جلبکی.

۳. «شاخص مواد آلی محلول رنگی» (برگه‌ی «مواد آلی محلول (CDOM)») — از نسبت باند سبز به قرمز سنتینل-۲ و
   با رابطه‌ی CDOM = 537 × exp(−2.93 × (B3/B4)) محاسبه می‌شود (رابطه‌ی باندی Brezonik و همکاران،
   ۲۰۰۵) و تقریبی از ضریب جذب مواد آلی محلول در طول موج ۴۴۰ نانومتر، با واحد «بر متر»، است. این
   شاخص برخلاف دو شاخص قبلی نرمال‌شده نیست، همیشه مثبت است و خط خنثی در صفر ندارد؛ بنابراین مقادیر
   آن را هرگز مانند NDTI/NDCI در بازه‌ی ۱- تا ۱+ تفسیر نکنید.

نکات تفسیری مهم درباره‌ی شاخص مواد آلی محلول رنگی:
- این شاخص «ماده‌ی محلول» را می‌سنجد، نه ذرات معلق. بنابراین افزایش هم‌زمان کدورت و مواد آلی محلول
  معمولاً نشانه‌ی ورود رواناب از سطح حوضه (بارش سنگین یا ذوب برف) است، در حالی که افزایش مواد آلی
  محلول بدون افزایش کدورت بیشتر به زهکشی اراضی آلی/تالابی، تخلیه‌ی فاضلاب یا تجزیه‌ی مواد آلی درون
  خود بدنه‌ی آبی اشاره دارد.
- افزایش آن برای بهره‌بردار آب شرب مهم است: رنگ، بو و مزه‌ی آب را تغییر می‌دهد، مصرف مواد
  گندزدا را بالا می‌برد و خطر تشکیل فرآورده‌های جانبی گندزدایی را افزایش می‌دهد. این مواد با
  ته‌نشینی ساده حذف نمی‌شوند.
- مواد آلی محلول رنگی نور را جذب می‌کند و می‌تواند بر برآورد کلروفیل اثر بگذارد؛ اگر کلروفیل و مواد
  آلی محلول هم‌زمان و به‌شدت همبسته بودند، این احتمال را در تفسیر خود صریحاً ذکر کنید.

داده‌های تحلیل:
{analysis_json}

شما به سه ابزار دسترسی دارید:
۱. reverse_geocode: یافتن دقیق نام شهر/منطقه/کشور بر اساس یک مختصات جغرافیایی (عرض و طول جغرافیایی)،
   با استفاده از سرویس Geocoding شرکت OpenWeatherMap. هر زمان که نیاز به شناسایی نام منطقه‌ی مورد
   مطالعه بر اساس center_latitude / center_longitude موجود در داده‌های تحلیل بالا داشتید، ابتدا از
   همین ابزار استفاده کنید — نه حدس زدن و نه جست‌وجوی وب — چون این ابزار مستقیماً و با دقت بالا نام
   مکان را از روی مختصات برمی‌گرداند.
۲. tavily_search (جست‌وجوی وب عمومی): برای اطلاعات کیفی و کلی اقلیمی (نوع اقلیم، طبقه‌بندی کوپن،
   الگوهای فصلی بارش و دما، رویدادهای خاص مانند سیل یا خشک‌سالی در آن منطقه) که به‌صورت عددی در دسترس
   نیست و باید از منابع وب یافت شود. همچنین برای هر پرسش عمومی دیگری که نیاز به اطلاعات به‌روز از وب
   دارد.
۳. get_monthly_weather_stats: ابزار اصلیِ «تحلیل علّی» شما — داده‌های دقیق هواشناسی تاریخی (بایگانی
   Open-Meteo) را برای یک مختصات جغرافیایی و یک سال/ماه مشخص برمی‌گرداند: دمای بیشینه/کمینه روزانه،
   بارش، برف، سرعت باد و درصد روزهای برفی آن ماه. **نقش اصلی این ابزار پاسخ به سؤال مستقیم کاربر درباره‌ی
   وضعیت هوا نیست، بلکه یافتن دلیل محتمل روند یا ناهنجاری‌های شاخص‌های کیفیت آب است** — یعنی از این ابزار
   باید به‌عنوان بخشی از فرایند عادی و خودکار تحلیل خودتان استفاده کنید، دقیقاً مانند یک متخصص انسانی که
   قبل از تفسیر یک نوسان غیرعادی، بی‌درنگ و بدون نیاز به درخواست کاربر، داده‌ی هواشناسی همان ماه را
   بررسی می‌کند.

دستورالعمل‌های استفاده از ابزارها:
- **استفاده‌ی خودکار و پیش‌کنشانه از get_monthly_weather_stats (نه صرفاً واکنشی):** هر زمان که در
  داده‌های تحلیل بالا با یکی از موارد زیر مواجه شدید، پیش از پاسخ نهایی، به‌طور خودکار get_monthly_weather_stats
  را برای ماه‌(های) مرتبط فراخوانی کنید — حتی اگر کاربر صریحاً کلمه‌ی «هوا» یا «آب‌وهوا» را به کار نبرده
  باشد:
    الف) هر آیتم در فهرست anomalies (ناهنجاری‌های شناسایی‌شده با انحراف مطلق از میانه) — بررسی کنید آیا
         بارش شدید، ذوب برف (برف قابل‌توجه در ماه‌های قبل یا همان ماه)، یا باد شدید می‌تواند توضیح‌دهنده‌ی
         آن جهش/افت ناگهانی باشد.
    ب) ماه‌های extremes.minimum و extremes.maximum — همین بررسی برای مقادیر حداکثر و حداقل کل بازه.
    پ) وقتی trend.standard یا trend.seasonal یک روند معنادار (significant: true) نشان می‌دهد — چند ماه
       نمونه (برای مثال ابتدای، میانه، و انتهای بازه‌ی دارای روند) را از نظر بارش/دما/باد بررسی کنید تا
       ببینید آیا روند مشاهده‌شده با یک روند هواشناسی هم‌راستا (مثلاً کاهش تدریجی بارش، افزایش دما) قابل
       توضیح است یا خیر.
    ت) وقتی کاربر درباره‌ی «چرا» یک تغییر رخ داده، یا درباره‌ی مقایسه‌ی دو ماه/دو فصل سؤال می‌پرسد.
  خلاصه: get_monthly_weather_stats بخشی جدایی‌ناپذیر از تفسیر روند/ناهنجاری شماست، نه یک ابزار جانبی که
  فقط با درخواست مستقیم کاربر برای «وضعیت هوا» فعال می‌شود.
- برای مختصات این ابزار و reverse_geocode، همیشه از center_latitude / center_longitude موجود در داده‌های
  تحلیل بالا استفاده کنید (در صورت نبودن این مقادیر در داده‌ها، صریحاً به کاربر بگویید که مختصات منطقه
  در دسترس نیست). هرگز از کاربر نخواهید مختصات را برایتان ارسال کند.
- سایر ابزارها را فقط زمانی فراخوانی کنید که واقعاً لازم است: reverse_geocode هر زمان که نام منطقه هنوز
  شناسایی نشده؛ tavily_search برای زمینه‌ی کیفی/کلی اقلیمی یا رویدادهای خاص (سیل، خشک‌سالی) که در داده‌ی
  عددی Open-Meteo دیده نمی‌شود.
- هرگز مختصات یا نام منطقه را حدس نزنید. برای شناسایی نام منطقه، همیشه ابتدا از reverse_geocode استفاده
  کنید؛ فقط اگر reverse_geocode نتیجه‌ای نداد یا کاربر اطلاعات کیفی/توصیفی بیشتری خواست، سراغ
  tavily_search بروید.
- ترتیب پیشنهادی هنگام تفسیر یک نوسان یا ناهنجاری: ابتدا در صورت نامشخص بودن نام منطقه، آن را با
  reverse_geocode شناسایی کنید؛ سپس بلافاصله و بدون نیاز به تأیید کاربر، داده‌ی عددی دقیق ماه(های)
  موردنظر را با get_monthly_weather_stats بگیرید؛ و در صورت نیاز، زمینه‌ی کلی اقلیمی یا رویدادهای خاص
  را نیز با tavily_search تکمیل کنید.

محدودیت بررسی در هر پاسخ:
- از میان ماه‌های ناهنجاری، بیشینه/کمینه و روند، حداکثر چهار ماه متمایز و مهم را برای بررسی هواشناسی
  انتخاب کنید؛ لازم نیست برای تک‌تک موارد ابزار فراخوانی شود. ماه مشترک بین چند شاخص را تکرار نکنید.
- فراخوانی‌های مستقل را در یک مرحله انجام دهید و حداکثر در سه مرحله از ابزارها استفاده کنید؛ سپس
  پاسخ را با شواهد موجود تکمیل کنید. ابزار ناموفق را در همان پاسخ دوباره فراخوانی نکنید.
- اگر ابزار در دسترس نبود یا سقف بررسی تکمیل شد، همه بخش‌های تحلیل آب را از داده‌های موجود بنویسید
  و محدودیت بررسی علت هواشناسی را صریح بگویید. رابطه آماری به‌تنهایی اثبات علت نیست.

دستورالعمل‌های مربوط به تاریخ (تقویم شمسی):
- کاربر با تقویم شمسی (جلالی) صحبت می‌کند و تاریخ‌ها را در پاسخ‌های خود باید به همین صورت (مثلاً «مرداد
  ۱۴۰۳») بیان کنید، مگر آنکه خود کاربر از تقویم میلادی استفاده کند.
- داده‌های تحلیل بالا (raw_observations، anomalies، extremes، seasonal_climatology) و ورودی‌های ابزار
  get_monthly_weather_stats بر پایه‌ی تاریخ میلادی (year, month) هستند. تبدیل بین تقویم شمسی و میلادی را
  همیشه خودتان، به‌صورت داخلی و بی‌صدا انجام دهید.
- هرگز از کاربر معادل میلادی یک تاریخ شمسی، یا مختصات جغرافیایی منطقه را نپرسید؛ این اطلاعات یا در
  داده‌های تحلیل بالا موجود است یا باید خودتان با ابزارها به دست آورید.

دستورالعمل‌های پاسخ‌گویی:
۱. پاسخ خود را در درجه‌ی اول بر پایه‌ی داده‌های JSON بالا و در صورت لزوم نتایج ابزارها بنا کنید و از
   دانش عمومی خود درباره‌ی کیفیت آب، سنجش‌ازدور و علوم محیط‌زیست برای تفسیر و تکمیل پاسخ استفاده کنید.
۲. تفسیر را مبتنی بر شواهد ارائه دهید؛ به‌جای نسبت‌دادن هر نوسان یا ناهنجاری به‌طور پیش‌فرض به «خطای
   حسگر»، ابتدا توضیح‌های محیطی، هیدرولوژیکی و هواشناسی محتمل را با کمک get_monthly_weather_stats در
   نظر بگیرید (رواناب فصلی، بارندگی، ذوب برف، سیل، رسوب‌گذاری، شکوفایی جلبکی و مانند آن).
۳. اگر داده‌ی کافی برای نتیجه‌گیری قطعی وجود ندارد (برای نمونه کمتر از ۲۴ ماه برای آزمون فصلی، یا
   نتایج ابزارها ناکافی/متناقض بود)، این محدودیت را صریح بیان کنید؛ حدس قطعی نزنید و هیچ واقعیتی را از
   خود نسازید.
۴. در پاسخ نهایی، در صورت استفاده از ابزارها، بین «شناسایی منطقه»، «داده‌ی عددی دقیق هواشناسی مرتبط با
   روند/ناهنجاری» و «زمینه‌ی کلی اقلیمی» تمایز قائل شوید.
۵. هر سه شاخص را با هم و در کنار یکدیگر تفسیر کنید، نه جدا از هم. به همبستگی‌های موجود در کلید
   relationships توجه کنید و هر جا الگوی مشترک یا واگرایی معناداری بین شاخص‌ها دیدید (برای نمونه
   افزایش هم‌زمان کدورت و مواد آلی محلول پس از یک ماه پربارش، یا افزایش کلروفیل بدون تغییر دو شاخص
   دیگر)، آن را صریح بیان کرده و توضیح فرایندی آن را ارائه دهید.
۶. اگر پرسش کاربر به زبان فارسی باشد، پاسخ باید کاملاً و فقط به زبان فارسی نوشته شود و از هیچ مخفف یا
   واژه‌ی انگلیسی استفاده نشود (برای نمونه به‌جای NDTI بنویسید «شاخص کدورت آب»، به‌جای NDCI بنویسید
   «شاخص کلروفیل»، و به‌جای CDOM بنویسید «مواد آلی محلول رنگی»؛ به‌جای MAD بنویسید «انحراف مطلق از
   میانه»).
۷. اگر پرسش کاربر به زبان دیگری باشد، به همان زبان پاسخ دهید.

قاعده مخصوص خلاصه کوتاه یک پارامتر:
اگر درخواست، خلاصه ۲ تا ۳ خطی یک پارامتر است، پاسخ نهایی را فقط درباره همان پارامتر و در همان تعداد
خط، با زبان رسمی و ساده برای کاربر غیرمتخصص بنویسید. بررسی سایر شاخص‌ها و جزئیات ابزارها را در تحلیل
داخلی انجام دهید و در خروجی توضیح جداگانه نیاورید. از گزارش آمار و اعداد غیرضروری خودداری کنید و
قالب، طول و واژه‌های ساده خواسته‌شده در پرسش را رعایت کنید؛ محدودیت داده را نیز در همان چند خط بگویید.
"""


# =============================================================================
# نظر متخصص آب — Cached network helpers (reverse geocoding + historical
# weather)
# =============================================================================
# Disk caching avoids repeated location/weather requests. It does not change
# the LLM client's own read timeout: a caught "Request timed out." can still
# originate there. Streaming and the model timeout are configured separately
# below. These per-request timeouts are not a whole-turn wall-clock deadline.
# =============================================================================
@st.cache_data(ttl=None, persist="disk", show_spinner=False)
def _reverse_geocode_cached(lat: float, lon: float) -> list:
    """Cached OpenWeatherMap reverse-geocoding lookup. Coordinates are rounded
    to ~100m precision so nearby points within the same monitored waterbody
    reuse the same cache entry instead of each triggering a fresh request."""
    params = {
        "lat": round(lat, 3),
        "lon": round(lon, 3),
        "limit": 5,
        "appid": OPENWEATHER_API_KEY,
    }
    r = requests.get(
        "https://api.openweathermap.org/geo/1.0/reverse",
        params=params,
        timeout=(5, 15),  # (connect, read) — fail fast instead of hanging up to 30s
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=None, persist="disk", show_spinner=False)
def _fetch_monthly_weather_cached(lat: float, lon: float, year: int, month: int) -> dict:
    """Cached Open-Meteo historical-archive lookup for one lat/lon/year/month.
    Historical weather for a past month never changes, so once fetched it is
    reused for every future question about that exact month — by any user,
    in any session — instead of hitting the API again.

    This helper backs the agent's get_monthly_weather_stats tool, whose main
    purpose is diagnostic: explaining *why* a turbidity/chlorophyll trend or
    anomaly occurred (heavy rainfall, snowmelt, wind-driven mixing, etc.),
    not just answering a direct question about the weather itself."""
    import calendar

    last_day = calendar.monthrange(year, month)[1]
    params = {
        "latitude": round(lat, 3),
        "longitude": round(lon, 3),
        "start_date": f"{year}-{month:02d}-01",
        "end_date": f"{year}-{month:02d}-{last_day}",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,wind_speed_10m_max",
        "timezone": "auto",
    }
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=(5, 15),  # (connect, read) — fail fast instead of hanging up to 30s
    )
    r.raise_for_status()
    return r.json()


EXPERT_REQUEST_TIMEOUT_SECONDS = 180
EXPERT_MAX_WEATHER_MONTHS = 4
EXPERT_RECURSION_LIMIT = 16


def _get_expert_model():
    """Allow slow report generation without silently replaying a paid request.

    Streaming limits each read's wait for new bytes; this is an inactivity
    timeout, not a total generation deadline. The gateway may impose its own
    shorter limit, which must be diagnosed on the deployment if it persists.
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=EXPERT_CHAT_MODEL,
        model_provider="openai",
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        temperature=0.3,
        timeout=EXPERT_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        streaming=True,
    )


def _summarize_expert_weather(daily):
    """Summarize observed values; missing weather must not become zero rain/snow."""
    import math

    def observed(key):
        return [float(value) for value in (daily.get(key) or [])
                if isinstance(value, (int, float)) and math.isfinite(value)]

    high = observed("temperature_2m_max")
    low = observed("temperature_2m_min")
    rain = observed("precipitation_sum")
    snow = observed("snowfall_sum")
    wind = observed("wind_speed_10m_max")
    snow_days = sum(value > 0 for value in snow) if snow else None
    return {
        "total_days": len(daily.get("time") or []),
        "snowfall_observed_days": len(snow),
        "snow_days": snow_days,
        "snow_day_percentage": round(100 * snow_days / len(snow), 1) if snow else None,
        "temperature_max_monthly": max(high) if high else None,
        "temperature_min_monthly": min(low) if low else None,
        "precipitation_monthly": sum(rain) if rain else None,
        "precipitation_observed_days": len(rain),
        "snowfall_monthly": sum(snow) if snow else None,
        "wind_speed_max_monthly": max(wind) if wind else None,
    }


def _get_expert_agent(analysis_json):
    """
    Build a fresh LangGraph ReAct agent with the three tools: OpenWeatherMap
    reverse geocoding, Tavily web search, and Open-Meteo historical weather.
    Rebuilt on every call since the system prompt embeds the current
    analysis JSON, which changes whenever new monitoring results are
    generated (see _expert_results_signature()). Building a ReAct agent is
    cheap (no network calls happen until a tool is actually invoked), so
    recreating it per question keeps the code simple and avoids
    stale-prompt bugs.

    API keys (TAVILY_API_KEY, OPENWEATHER_API_KEY) are hardcoded constants
    read directly from this file rather than from a .env file, since the
    Posit server this app runs on cannot read a local .env.

    The reverse_geocode and get_monthly_weather_stats tools below call the
    disk-cached helpers defined just above this function
    (_reverse_geocode_cached / _fetch_monthly_weather_cached) instead of
    hitting their APIs directly — see the comment on those helpers for why.

    Requires: langchain, langchain-openai, langgraph, langchain-tavily
    (pip install langchain langchain-openai langgraph langchain-tavily)
    """
    from threading import Lock
    from langchain_core.tools import tool
    from langchain_tavily import TavilySearch
    from langgraph.prebuilt import create_react_agent

    # TavilySearch reads its key from the environment — set it directly from
    # the hardcoded constant above (no .env involved).
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

    model = _get_expert_model()
    weather_months = set()
    weather_lock = Lock()  # a single model turn can request tools concurrently

    tavily_tool = TavilySearch(max_results=5, topic="general")

    @tool
    def reverse_geocode(lat: float, lon: float) -> str:
        """Reverse-geocode a latitude/longitude pair into a place name
        (city/town/village, state/province, and country) using the
        OpenWeatherMap Geocoding API. Use this whenever you need to identify
        the name of the region/city/country for a given set of coordinates
        (e.g. the center_latitude / center_longitude of the monitored water
        body) — it is faster and more precise than a general web search for
        this specific purpose."""
        try:
            results = _reverse_geocode_cached(lat, lon)
        except Exception as e:
            return str({
                "lat": lat, "lon": lon, "results": [],
                "error": f"geocoding service unavailable or timed out: {e}",
            })

        if not results:
            return str({"lat": lat, "lon": lon, "results": [], "note": "No place name found for these coordinates."})

        return str({
            "lat": lat,
            "lon": lon,
            "results": [
                {
                    "name": item.get("name"),
                    "state": item.get("state"),
                    "country": item.get("country"),
                    "local_names": item.get("local_names"),
                }
                for item in results
            ],
        })

    @tool
    def get_monthly_weather_stats(lat: float, lon: float, year: int, month: int) -> str:
        """Get daily historical weather data (max/min temperature, precipitation,
        snowfall, wind speed) for a given latitude/longitude and a specific
        year/month, using the Open-Meteo historical archive API. Also returns
        the number of days with snowfall and the snow-day percentage for that
        month.

        This is your primary DIAGNOSTIC tool, not just a way to answer direct
        weather questions. Its main purpose is to help you explain the likely
        physical cause of a turbidity/chlorophyll trend or anomaly found in
        the statistical analysis JSON (e.g. a spike coinciding with heavy
        precipitation or rapid snowmelt, or a chlorophyll rise coinciding with
        calm, warm, low-wind conditions favorable to algal blooms).

        Proactively select up to four important distinct months from anomalies,
        extremes or significant trends. Reuse a month shared by several water
        indices. Do not retry failed lookups in this answer. Once the budget is
        reached, finish the water-quality report with available evidence and
        explicitly state the limits on weather-based explanations.

        Always pass the center_latitude / center_longitude already present in
        the analysis JSON, and the Gregorian year/month equivalent of the
        month you are investigating (convert internally from Shamsi/Jalali if
        that is how the user phrased the date) — never ask the user for
        coordinates or for the Gregorian date."""
        weather_key = (round(lat, 3), round(lon, 3), year, month)
        with weather_lock:
            if weather_key in weather_months:
                return str({"error": "This month was already checked in this answer; use its earlier result, or state that it was unavailable."})
            if len(weather_months) >= EXPERT_MAX_WEATHER_MONTHS:
                return str({"error": "Weather lookup limit reached. Complete the report using available evidence and disclose unverified weather causes."})
            weather_months.add(weather_key)
        try:
            data = _fetch_monthly_weather_cached(lat, lon, year, month)
        except Exception as e:
            return str({"error": f"weather service unavailable or timed out: {e}"})

        daily = data.get("daily", {})
        # Preserve daily evidence once; do not duplicate the arrays as "summary".
        data = dict(data, summary=_summarize_expert_weather(daily))
        return str(data)

    tools = [reverse_geocode, tavily_tool, get_monthly_weather_stats]
    try:
        analysis_json = json.dumps(json.loads(analysis_json), ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        pass
    system_prompt = _build_agent_system_prompt(analysis_json)
    return create_react_agent(model, tools, prompt=system_prompt)


def _expert_history_messages(chat_history):
    """Keep at most six recent complete, successful exchanges within 24k chars."""
    pairs = []
    pending = None
    for message in chat_history:
        if message.get("role") == "user":
            pending = message
        elif message.get("role") == "assistant":
            if (pending is not None and not message.get("error")
                    and not message.get("content", "").startswith("خطا در ارتباط با عامل هوش مصنوعی:")):
                pairs.append((pending, message))
            pending = None
    recent = []
    size = 0
    for pair in reversed(pairs[-6:]):
        pair_size = sum(len(message.get("content", "")) for message in pair)
        if size + pair_size > 24000:
            break
        recent.insert(0, pair)
        size += pair_size
    return [message for pair in recent for message in pair]


def _expert_message_text(content):
    """Display text only, excluding reasoning and tool-call content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content
                       if isinstance(block, dict) and block.get("type") == "text"
                       and isinstance(block.get("text"), str))
    return ""


def _expert_error_message(error):
    # Keep the original dependency/configuration error visible if the chatbot
    # packages themselves are missing; error rendering must not import-fail.
    try:
        from openai import APITimeoutError
        timeout_errors = (TimeoutError, APITimeoutError)
    except ImportError:
        timeout_errors = (TimeoutError,)

    if isinstance(error, timeout_errors):
        return ("پاسخ سرویس هوش مصنوعی در زمان مجاز دریافت نشد. لطفاً دوباره تلاش کنید. "
                "اگر این خطا تکرار شد، اتصال سرور برنامه به سرویس هوش مصنوعی و محدودیت زمان سرویس بررسی شود.")
    if any(cls.__name__ == "GraphRecursionError" for cls in type(error).__mro__):
        return "تعداد مراحل بررسی به سقف مجاز رسید. لطفاً پرسش را به یک موضوع یا بازه کوتاه‌تر محدود کنید."
    return f"خطا در ارتباط با عامل هوش مصنوعی: {error}"


def ask_water_quality_expert(question, analysis_json, chat_history, on_text=None, on_progress=None):
    """
    Send `question` to the water-quality expert agent, giving it the running
    chat history for context. chat_history is a list of
    {"role": "user"/"assistant", "content": ...} dicts — the same shape
    already used elsewhere in the app for st.session_state.expert_chat_history,
    so no other call site needs to change.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    agent = _get_expert_agent(analysis_json)

    messages = []
    for msg in _expert_history_messages(chat_history):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    answer = ""
    partial = ""
    message_id = None
    tool_labels = {
        "reverse_geocode": "در حال شناسایی منطقه…",
        "get_monthly_weather_stats": "در حال بررسی داده‌های هواشناسی ماه‌های منتخب…",
        "tavily_search": "در حال بررسی منابع تکمیلی…",
    }
    if on_progress:
        on_progress("در حال تحلیل داده‌های آب؛ تهیه گزارش جامع ممکن است چند دقیقه طول بکشد…")
    for mode, event in agent.stream(
        {"messages": messages},
        config={"recursion_limit": EXPERT_RECURSION_LIMIT},
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            chunk, metadata = event
            if metadata.get("langgraph_node") != "agent":
                continue
            chunk_id = getattr(chunk, "id", None)
            if chunk_id != message_id:
                partial = ""
                message_id = chunk_id
            partial += _expert_message_text(chunk.content)
            if on_text and partial and not getattr(chunk, "tool_call_chunks", None):
                on_text(partial)
        elif mode == "updates":
            for node, update in event.items():
                if node != "agent" or not isinstance(update, dict):
                    continue
                for message in update.get("messages", []):
                    calls = getattr(message, "tool_calls", None)
                    if calls:
                        partial = ""
                        message_id = None
                        if on_text:
                            on_text("")
                        if on_progress:
                            on_progress(tool_labels.get(calls[0].get("name"), "در حال بررسی شواهد تکمیلی…"))
                    else:
                        answer = _expert_message_text(message.content)
                        # LangGraph can end with this sentinel instead of raising
                        # GraphRecursionError when its remaining steps run low.
                        if answer.strip() == "Sorry, need more steps to process this request.":
                            raise RuntimeError("تعداد مراحل بررسی به سقف مجاز رسید؛ لطفاً پرسش را به یک موضوع یا بازه کوتاه‌تر محدود کنید.")
                        if on_text:
                            on_text(answer)
    if not answer.strip():
        raise RuntimeError("سرویس هوش مصنوعی پاسخ متنی کامل برنگرداند؛ لطفاً دوباره تلاش کنید.")
    return answer


def _inject_persian_chat_css():
    """Readable Persian chat chrome, independent of the agent or its prompts."""
    st.markdown(
        """
        <style>
        [data-testid="stChatMessage"] {
            direction: rtl; text-align: right; background: #FFFFFF;
            border: 1px solid #DCE5EC; border-radius: 13px;
            padding: 1rem 1.2rem; margin-bottom: .8rem;
            box-shadow: 0 3px 12px rgba(20,44,64,.035);
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
            direction: rtl; text-align: right;
            font-family: 'Vazirmatn', Tahoma, 'B Nazanin', 'BNazanin', sans-serif;
            font-size: .98rem; line-height: 2;
        }
        [data-testid="stChatMessage"] pre, [data-testid="stChatMessage"] code {
            direction: ltr; text-align: left; unicode-bidi: isolate;
        }
        [data-testid="stChatInput"] textarea {
            direction: rtl; text-align: right; font-size: .96rem;
            font-family: 'Vazirmatn', Tahoma, 'B Nazanin', 'BNazanin', sans-serif;
        }
        [data-testid="stChatInput"] { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _force_quick_button_style():
    """Style the existing quick actions without altering the DOM or event handlers."""
    st.markdown(
        """
        <style>
        [class*="st-key-wqquick_"] button {
            min-height: 3.6rem; border-radius: 11px; padding: .7rem 1rem;
            font-family: 'Vazirmatn', Tahoma, 'B Nazanin', 'BNazanin', sans-serif;
            direction: rtl; font-size: 1rem; font-weight: 600; line-height: 1.8;
            box-shadow: 0 3px 12px rgba(20,44,64,.04); text-shadow: none;
        }
        [class*="st-key-wqquick_"] button p { font-size: 1rem; font-weight: 600; line-height: 1.8; }
        [class*="st-key-wqquick_brief"] button { background: #FFF7E9; color: #804B0F; border: 1px solid #E6D3AF; }
        [class*="st-key-wqquick_brief"] button:hover:not(:disabled) { background: #FAEFD9; border-color: #BA8E48; color: #6F400B; }
        [class*="st-key-wqquick_deep"] button { background: #EDF7F7; color: #086873; border: 1px solid #BFDDE0; }
        [class*="st-key-wqquick_deep"] button:hover:not(:disabled) { background: #DFEFF0; border-color: #589DA5; color: #07545D; }
        @supports selector(:has(*)) {
            [data-testid="stElementContainer"]:has(.wq-quick-anchor) + [data-testid="stHorizontalBlock"] button {
                min-height: 3.6rem; border-radius: 11px; font-size: 1rem;
                background: #EDF7F7; color: #086873; border: 1px solid #BFDDE0;
            }
            [data-testid="stElementContainer"]:has(.wq-nav-anchor) + [data-testid="stHorizontalBlock"] button[kind="primary"] {
                background: #142C40; color: #FFFFFF; border-color: #142C40;
                box-shadow: inset 0 -3px 0 #57B9BF;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_expert_analysis():
    """
    Make sure the statistical-analysis JSON for the current monitoring results
    exists in session state, recomputing it when the results have changed.

    Shared by «خلاصه مدیریتی» and «چت بات» so both always talk to the agent
    about exactly the same analysis. Returns True when the JSON is ready.
    """
    if 'expert_analysis_json' not in st.session_state:
        st.session_state.expert_analysis_json = None
    if 'expert_analysis_signature' not in st.session_state:
        st.session_state.expert_analysis_signature = None

    if not any(st.session_state.results.get(p) for p in ALL_PARAMETERS):
        return False

    signature = _expert_results_signature()
    if (st.session_state.expert_analysis_json is not None
            and st.session_state.expert_analysis_signature == signature):
        return True

    with st.spinner("در حال تحلیل آماری داده‌های سری زمانی..."):
        try:
            excel_bytes = generate_combined_timeseries_excel()
            analysis = analyze_water_quality_from_bytes(excel_bytes)
            st.session_state.expert_analysis_json = json.dumps(analysis, ensure_ascii=False, indent=2)
            st.session_state.expert_analysis_signature = signature
            # The data changed -> anything derived from it is stale
            st.session_state.expert_chat_history = []
            st.session_state.executive_summary = None
            st.session_state.executive_summary_signature = None
            st.session_state.param_briefs = {}
            st.session_state.param_brief_signatures = {}
        except Exception as e:
            st.error(f"خطا در تحلیل داده‌ها: {e}")
            return False

    return True


def _render_summary_card(summary_text):
    """Render the 5-6 line summary as one readable RTL card, a line per row."""
    import html as _html

    lines = [ln.strip() for ln in str(summary_text).splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        lines = [str(summary_text).strip()]

    rows = "".join(
        f'<div class="wq-summary-line">{_html.escape(ln)}</div>' for ln in lines
    )
    st.markdown(f'<div class="wq-summary-card">{rows}</div>', unsafe_allow_html=True)


def _render_param_brief_card(parameter_type, brief_text):
    """Render the per-parameter expert brief as one styled RTL card, themed to
    match the colour of the page it sits on. Purely presentational."""
    import html as _html

    css_class, icon, title = PARAM_BRIEF_THEME.get(
        parameter_type,
        ("wq-brief-turbidity", "\U0001f9e0", "تحلیل کارشناسی هوش مصنوعی"),
    )

    # Preserve the requested short lines while normalising whitespace within
    # each line. Escape each line before inserting HTML line breaks.
    lines = [" ".join(line.split()) for line in str(brief_text).splitlines()]
    body = "<br>".join(_html.escape(line) for line in lines if line)

    st.markdown(
        f"""
        <div class="wq-param-brief {css_class}">
            <div class="wq-param-brief-head">
                <span class="wq-param-brief-icon">{icon}</span>
                <span class="wq-param-brief-title">{_html.escape(title)}</span>
            </div>
            <div class="wq-param-brief-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_param_brief(parameter_type):
    """
    تحلیل کارشناسی کوتاه بالای صفحه هر پارامتر: همان عامل متخصص و همان دستور
    سیستمی صفحه چت، اما با یک پرسش ثابتِ مخصوص همان پارامتر.

    برای هر پارامتر فقط یک بار (در نخستین باز شدن آن صفحه پس از هر اجرای پایش)
    تولید و سپس ذخیره می‌شود؛ بنابراین جابه‌جایی میان صفحه‌ها هزینه‌ای ندارد و
    با تغییر نتایج پایش یا متن پرسش، به‌طور خودکار بازتولید می‌شود.
    """
    if 'param_briefs' not in st.session_state:
        st.session_state.param_briefs = {}
    if 'param_brief_signatures' not in st.session_state:
        st.session_state.param_brief_signatures = {}

    prompt = PARAM_PAGE_SUMMARY_PROMPTS.get(parameter_type)
    if not prompt:
        return

    if not _ensure_expert_analysis():
        return

    signature = (_expert_results_signature(), prompt)
    is_stale = (
        not st.session_state.param_briefs.get(parameter_type)
        or st.session_state.param_brief_signatures.get(parameter_type) != signature
    )

    if is_stale:
        with st.spinner("در حال تهیه تحلیل کارشناسی این شاخص توسط متخصص هوش مصنوعی..."):
            try:
                brief = ask_water_quality_expert(
                    prompt,
                    st.session_state.expert_analysis_json,
                    []   # a standalone question: no chat history
                )
                st.session_state.param_briefs[parameter_type] = brief
                st.session_state.param_brief_signatures[parameter_type] = signature
            except Exception as e:
                st.warning(
                    "تحلیل کارشناسی این شاخص تهیه نشد (خطا در ارتباط با متخصص هوش مصنوعی): "
                    f"{e}"
                )
                return

    brief = st.session_state.param_briefs.get(parameter_type)
    if brief:
        _render_param_brief_card(parameter_type, brief)


def render_summary_page():
    """
    صفحه «خلاصه مدیریتی»: همان عامل متخصص و همان دستور سیستمی صفحه چت، اما با یک
    پرسش ثابت (EXECUTIVE_SUMMARY_PROMPT) که فقط یک بار اجرا و نتیجه‌اش ذخیره
    می‌شود. خروجی، یک خلاصه ۵ تا ۶ خطی و ساده برای مدیر است که هر سه شاخص را با
    هم می‌بیند.
    """
    _render_active_section_badge("🧭", "خلاصه مدیریتی کیفیت آب", "#14507A", "#4A9BD4")

    if 'executive_summary' not in st.session_state:
        st.session_state.executive_summary = None
    if 'executive_summary_signature' not in st.session_state:
        st.session_state.executive_summary_signature = None

    if not any(st.session_state.results.get(p) for p in ALL_PARAMETERS):
        st.info("برای مشاهده خلاصه مدیریتی، ابتدا پایش را اجرا کنید.")
        return

    if not _ensure_expert_analysis():
        return

    signature = _expert_results_signature()
    is_stale = (
        st.session_state.executive_summary is None
        or st.session_state.executive_summary_signature != signature
    )

    # Generated once per monitoring run and then cached, so simply switching
    # between pages never costs another call to the language model. It is
    # refreshed automatically whenever the monitoring results change.
    if is_stale:
        with st.spinner("در حال تهیه خلاصه مدیریتی توسط متخصص هوش مصنوعی..."):
            try:
                summary = ask_water_quality_expert(
                    EXECUTIVE_SUMMARY_PROMPT,
                    st.session_state.expert_analysis_json,
                    []   # a standalone question: no chat history
                )
                st.session_state.executive_summary = summary
                st.session_state.executive_summary_signature = signature
            except Exception as e:
                st.error(
                    "خلاصه مدیریتی تهیه نشد (خطا در ارتباط با متخصص هوش مصنوعی): "
                    f"{e}"
                )
                return

    if st.session_state.executive_summary:
        _render_summary_card(st.session_state.executive_summary)
        st.caption(
            "این خلاصه به‌طور خودکار از تحلیل آماری هر سه شاخص کدورت، کلروفیل و مواد آلی محلول "
            "تهیه شده است. برای پرسش‌های دقیق‌تر، از صفحه «چت بات» استفاده کنید."
        )


def render_expert_chat_tab():
    """
    صفحه «نظر متخصص آب»: به‌صورت خودکار خروجی اکسل پایش را می‌گیرد، پایپ‌لاین
    تحلیل آماری موجود را روی آن اجرا می‌کند، خلاصه JSON تولید می‌کند، و یک
    رابط گفتگو با یک عامل هوشمند (LangGraph ReAct agent) در اختیار کاربر
    قرار می‌دهد. این عامل علاوه بر خلاصه JSON، به سه ابزار نیز دسترسی دارد:
    شناسایی نام منطقه از روی مختصات (reverse geocoding با OpenWeatherMap،
    نتایج آن روی دیسک کش می‌شوند)، جست‌وجوی وب برای زمینه‌ی کلی اقلیمی
    (Tavily)، و دریافت داده‌های دقیق هواشناسی تاریخی (Open-Meteo، نتایج آن
    نیز روی دیسک کش می‌شوند) برای مختصات مرکز منطقه — این ابزار هواشناسی
    به‌طور خودکار در تحلیل علّی روندها و ناهنجاری‌ها به کار می‌رود، نه فقط در
    پاسخ به سؤال مستقیم درباره‌ی وضعیت هوا.
    """
    _inject_persian_chat_css()
    # Styles the two ready-made question buttons regardless of the Streamlit
    # version's DOM/class naming (see _force_quick_button_style). Invisible.
    _force_quick_button_style()

    _render_active_section_badge("💬", "چت با متخصص آب", "#E08E0B", "#F5A524")

    if 'expert_chat_history' not in st.session_state:
        st.session_state.expert_chat_history = []

    if not any(st.session_state.results.get(p) for p in ALL_PARAMETERS):
        st.info("برای استفاده از این بخش، ابتدا پایش را اجرا کنید تا داده‌ای برای تحلیل وجود داشته باشد.")
        return

    # Same analysis JSON the «خلاصه مدیریتی» page uses
    if not _ensure_expert_analysis():
        return

    if st.button("🗑️ شروع گفتگوی جدید", key="expert_chat_reset"):
        st.session_state.expert_chat_history = []
        st.rerun()

    # ------------------------------------------------------------------
    # Ready-made questions. They go to exactly the same expert agent a typed
    # question goes to; only the wording is fixed (see QUICK_*_PROMPT).
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="wq-quick-hint">می‌توانید سؤال خود را بنویسید، یا یکی از بررسی‌های آماده زیر را اجرا کنید:</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="wq-quick-anchor"></div>', unsafe_allow_html=True)

    quick_question = None
    quick_label = None

    qcol_left, qcol_right = st.columns(2)
    # Right-to-left: the quick review sits on the right, the deep one on the left
    if qcol_right.button("⚡ بررسی خلاصه و سریع", key="wqquick_brief",
                         use_container_width=True,
                         help="توضیح کوتاه و ساده از وضعیت کلی کیفیت آب، در ۳ تا ۴ خط"):
        quick_question = QUICK_BRIEF_PROMPT
        quick_label = "⚡ بررسی خلاصه و سریع"

    if qcol_left.button("🔬 بررسی جامع و دقیق", key="wqquick_deep",
                        use_container_width=True,
                        help="تحلیل کامل هر سه شاخص: روند، الگوی فصلی، ناهنجاری‌ها، همبستگی‌ها و توصیه‌ها"):
        quick_question = QUICK_DEEP_PROMPT
        quick_label = "🔬 بررسی جامع و دقیق"

    st.divider()

    for msg in st.session_state.expert_chat_history:
        with st.chat_message(msg["role"]):
            # A ready-made question is shown by its button title, while the full
            # prompt behind it stays in the history as the agent's context.
            st.markdown(msg.get("display") or msg["content"])

    typed_question = st.chat_input("سؤال خود را درباره کیفیت آب این منطقه بپرسید...")

    user_question = typed_question or quick_question
    display_text = None if typed_question else quick_label

    if user_question:
        st.session_state.expert_chat_history.append(
            {"role": "user", "content": user_question, "display": display_text}
        )
        with st.chat_message("user"):
            st.markdown(display_text or user_question)

        with st.chat_message("assistant"):
            progress = st.empty()
            response = st.empty()
            failed = False
            with st.spinner("در حال بررسی توسط متخصص هوش مصنوعی..."):
                try:
                    answer = ask_water_quality_expert(
                        user_question,
                        st.session_state.expert_analysis_json,
                        st.session_state.expert_chat_history[:-1],
                        on_text=response.markdown,
                        on_progress=progress.caption,
                    )
                except Exception as e:
                    failed = True
                    answer = _expert_error_message(e)
                finally:
                    progress.empty()
                response.markdown(answer)

        st.session_state.expert_chat_history.append({"role": "assistant", "content": answer, "error": failed})


# =============================================================================
# Global App Styling — Persian (B Nazanin) font + RTL text + professional
# color palette for general UI chrome only (buttons, headers, inputs,
# sidebar, tabs, metrics, alerts, dataframes, progress bar). This is purely
# presentational: it does not touch the turbidity/chlorophyll colormaps
# (create_turbidity_colormap / create_chlorophyll_colormap) or any of the
# matplotlib figures used to render scientific results, and it does not
# alter any data-processing, calculation, or workflow logic.
# =============================================================================
@st.cache_data(show_spinner=False)
def _load_bnazanin_font():
    """
    Look for a local B Nazanin font file (see BNAZANIN_FONT_CANDIDATES) and, if
    found, return (base64_data, css_format, absolute_path) so it can be embedded
    straight into the page as an @font-face rule. Falls back to the optional
    Google Drive copy, and finally to (None, None, None) — in which case the app
    keeps using the existing "locally installed B Nazanin, else Vazirmatn"
    behaviour and nothing breaks.
    """
    ext_to_format = {
        ".woff2": "woff2",
        ".woff": "woff",
        ".ttf": "truetype",
        ".otf": "opentype",
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for rel_path in BNAZANIN_FONT_CANDIDATES:
        for candidate in (os.path.join(base_dir, rel_path), rel_path):
            if os.path.isfile(candidate):
                ext = os.path.splitext(candidate)[1].lower()
                fmt = ext_to_format.get(ext)
                if not fmt:
                    continue
                try:
                    with open(candidate, "rb") as f:
                        return base64.b64encode(f.read()).decode(), fmt, os.path.abspath(candidate)
                except Exception:
                    continue

    if BNAZANIN_FONT_DRIVE_ID:
        try:
            url = f"https://drive.google.com/uc?export=download&id={BNAZANIN_FONT_DRIVE_ID}"
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return base64.b64encode(response.content).decode(), "truetype", None
        except Exception:
            pass

    return None, None, None


def _bnazanin_font_face_css():
    """Return the @font-face block for B Nazanin (empty string if unavailable)."""
    font_b64, fmt, font_path = _load_bnazanin_font()
    if not font_b64:
        return ""

    # Make the same font available to matplotlib as well, so the legends and the
    # time-series charts use B Nazanin too instead of the generic fallback.
    if font_path and os.path.splitext(font_path)[1].lower() in (".ttf", ".otf"):
        try:
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(font_path)
        except Exception:
            pass

    return f"""
        @font-face {{
            font-family: 'B Nazanin';
            src: url(data:font/{fmt};charset=utf-8;base64,{font_b64}) format('{fmt}');
            font-weight: normal;
            font-style: normal;
            font-display: swap;
        }}
        @font-face {{
            font-family: 'B Nazanin';
            src: url(data:font/{fmt};charset=utf-8;base64,{font_b64}) format('{fmt}');
            font-weight: bold;
            font-style: normal;
            font-display: swap;
        }}
    """


def _inject_global_app_css():
    """One presentation-only design system; map geometry and widget events stay native."""
    font_face = _embedded_persian_font_css() + _bnazanin_font_face_css()
    if font_face:
        st.markdown(f"<style>{font_face}</style>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        :root {
            --wq-font: 'WQ Vazirmatn', Tahoma, Arial, sans-serif;
            --wq-navy: #142C40;
            --wq-teal-dark: #086873;
            --wq-teal: #087F8C;
            --wq-teal-light: #57B9BF;
            --wq-amber: #B36B15;
            --wq-amber-dark: #8D5110;
            --wq-bg-1: #F3F6F9;
            --wq-bg-2: #F8FAFC;
            --wq-card: #FFFFFF;
            --wq-border: #DCE5EC;
            --wq-muted: #536879;
            --wq-shadow: 0 4px 18px rgba(20,44,64,.045);
        }
        html, body, .stApp { font-family: var(--wq-font); color: var(--wq-navy); }
        .stApp { background: var(--wq-bg-1); }
        .stMarkdown, .stText, .stCaption, label, h1, h2, h3, h4, h5, h6,
        .stButton button, .stDownloadButton button, .stTextInput input,
        .stTextArea textarea, .stNumberInput input, .stDateInput input,
        .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"],
        [data-testid="stMetric"], [data-testid="stAlert"], [data-testid="stCaptionContainer"] {
            font-family: var(--wq-font);
        }
        .stMarkdown, .stText, .stCaption, [data-testid="stCaptionContainer"],
        [data-testid="stWidgetLabel"], [data-testid="stAlert"],
        h1, h2, h3, h4, h5, h6 {
            direction: rtl;
            text-align: right;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] ul,
        [data-testid="stMarkdownContainer"] ol {
            direction: rtl; text-align: right; font-family: var(--wq-font);
        }
        [data-testid="stMarkdownContainer"] ul,
        [data-testid="stMarkdownContainer"] ol {
            padding-inline-start: 1.3rem; padding-inline-end: 0;
        }
        /* Scope RTL to text and controls: columns already have an explicit
           Persian reading order in Python; Leaflet remains left-to-right. */
        .stTextInput input, .stTextArea textarea, [data-testid="stChatInput"] textarea,
        .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"],
        [role="listbox"], [role="option"] { direction: rtl; text-align: right; }
        .stNumberInput input, .stDateInput input {
            direction: ltr; text-align: right; font-variant-numeric: tabular-nums;
        }
        .stMarkdown p, .stMarkdown li { font-size: .98rem; line-height: 1.95; }
        [data-testid="stCaptionContainer"], .stCaption { color: var(--wq-muted); font-size: .85rem; line-height: 1.85; }
        .block-container {
            max-width: 1440px;
            padding-top: 2.5rem;
            padding-bottom: 3.5rem;
            padding-left: 2.25rem;
            padding-right: 2.25rem;
        }
        [data-testid="stHorizontalBlock"] { gap: 1rem; }
        hr { border-color: var(--wq-border); margin: 1.6rem 0 !important; opacity: .8; }
        h1 { color: var(--wq-navy); font-size: 1.95rem; font-weight: 800; line-height: 1.7; }
        h2 { color: var(--wq-navy); font-size: 1.5rem; font-weight: 800; line-height: 1.7; }
        h3 { color: var(--wq-navy); font-size: 1.2rem; font-weight: 700; line-height: 1.8; }
        h4 { color: var(--wq-navy); font-size: 1.05rem; font-weight: 700; margin: 1.15rem 0 .7rem; }
        .wq-app-header {
            direction: rtl; display: flex; align-items: center; gap: 1rem;
            padding: 1.35rem 1.6rem .55rem;
            border-radius: 18px 18px 0 0;
            background: linear-gradient(120deg, #142C40, #173F50);
            color: #FFFFFF;
        }
        .wq-app-header img { width: 3.2rem; height: 3.2rem; object-fit: contain; border-radius: 10px; }
        .wq-app-emblem {
            width: 3.1rem; height: 3.1rem; display: inline-flex; align-items: center;
            justify-content: center; background: rgba(255,255,255,.1); border-radius: 12px; font-size: 1.8rem;
        }
        .wq-app-eyebrow { display: block; font-size: .72rem; font-weight: 500; color: #B9D7DF; margin-bottom: .3rem; }
        .wq-app-title { font-size: clamp(1.35rem, 2.2vw, 1.9rem); font-weight: 800; line-height: 1.7; color: #FFFFFF; }
        .wq-app-subtitle {
            direction: rtl; text-align: right; background: linear-gradient(120deg, #142C40, #173F50);
            color: #DAE8ED; padding: .15rem 1.6rem 1.25rem; margin-bottom: 1.15rem;
            border-radius: 0 0 18px 18px; font-size: .87rem; line-height: 1.9;
            box-shadow: 0 6px 20px rgba(20,44,64,.09);
        }
        .wq-app-subtitle b { color: #FFFFFF; font-weight: 600; }
        .stButton button, .stDownloadButton button {
            direction: rtl; min-height: 2.9rem; padding: .52rem 1rem; border-radius: 10px;
            background: #FFFFFF; border: 1px solid #CCD9E2; color: var(--wq-navy);
            font-weight: 600; box-shadow: 0 1px 3px rgba(20,44,64,.025);
            transition: background .16s ease, border-color .16s ease, box-shadow .16s ease;
        }
        .stButton button p, .stDownloadButton button p { font-size: .91rem; font-weight: 600; line-height: 1.7; }
        .stButton button:hover:not(:disabled), .stDownloadButton button:hover:not(:disabled) {
            background: #F0F7F8; border-color: var(--wq-teal); color: var(--wq-teal-dark);
            box-shadow: 0 3px 9px rgba(8,127,140,.08);
        }
        .stButton button[kind="primary"], .stButton button[kind="primaryFormSubmit"],
        [data-testid="baseButton-primary"], .stDownloadButton button {
            background: var(--wq-teal); border-color: var(--wq-teal); color: #FFFFFF;
        }
        .stButton button[kind="primary"]:hover:not(:disabled),
        .stDownloadButton button:hover:not(:disabled) {
            background: var(--wq-teal-dark); border-color: var(--wq-teal-dark); color: #FFFFFF;
        }
        .stButton button:disabled, .stDownloadButton button:disabled {
            background: #EDF1F5; border-color: #DFE6EC; color: #687C8C; box-shadow: none; opacity: 1;
        }
        button:focus-visible, input:focus-visible, textarea:focus-visible,
        summary:focus-visible, a:focus-visible {
            outline: 3px solid #4C9DB3 !important; outline-offset: 3px;
        }
        .wq-nav-anchor, .wq-quick-anchor { display: none; }
        [class*="st-key-wqnav_"] button {
            min-height: 3.1rem; background: #FFFFFF; border: 1px solid var(--wq-border);
            color: #344F63; border-radius: 11px; box-shadow: none;
        }
        [class*="st-key-wqnav_"] button[kind="primary"] {
            background: var(--wq-navy); border-color: var(--wq-navy); color: #FFFFFF;
            box-shadow: inset 0 -3px 0 #57B9BF;
        }
        [class*="st-key-wqnav_"] button[kind="primary"]:hover:not(:disabled) {
            background: #214558; border-color: #214558; color: #FFFFFF;
        }
        [class*="st-key-wqnav_"] button:disabled {
            background: #EDF1F5; border: 1px dashed #CBD6DF; color: #687C8C;
        }
        .wq-nav-hint { direction: rtl; text-align: right; color: var(--wq-muted); font-size: .8rem; margin: .2rem 0 .9rem; }
        [data-testid="stWidgetLabel"] p { font-size: .91rem; font-weight: 600; color: var(--wq-navy); }
        .stTextInput input, .stTextArea textarea, .stNumberInput input,
        .stDateInput input, .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {
            font-size: .95rem; border-radius: 9px; background: #FFFFFF; color: var(--wq-navy);
        }
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div { border-color: #CCD9E2; min-height: 2.8rem; }
        .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
            border-color: var(--wq-teal); box-shadow: 0 0 0 2px rgba(8,127,140,.12);
        }
        .stTabs [data-baseweb="tab-list"] { direction: rtl; gap: .5rem; border-bottom: 1px solid var(--wq-border); }
        .stTabs [data-baseweb="tab"] { padding: .7rem 1.1rem; font-weight: 600; color: var(--wq-muted); border-radius: 8px 8px 0 0; }
        .stTabs [aria-selected="true"] { color: var(--wq-teal-dark); background: #E8F4F5; }
        .stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }
        .wq-step-header {
            direction: rtl; display: flex; align-items: center; gap: .7rem;
            margin: 1.65rem 0 1rem; padding: .9rem 1.1rem; background: #FFFFFF;
            border: 1px solid var(--wq-border); border-right: 3px solid var(--wq-teal);
            border-radius: 12px; box-shadow: var(--wq-shadow);
        }
        .wq-step-number {
            display: inline-flex; align-items: center; justify-content: center;
            min-width: 1.9rem; height: 1.9rem; border-radius: 8px;
            background: #E8F4F5; color: var(--wq-teal-dark); font-weight: 700; font-size: .9rem;
        }
        .wq-step-icon { font-size: 1.2rem; }
        .wq-step-title { color: var(--wq-navy); font-size: 1.12rem; font-weight: 700; }
        .wq-page-badge {
            direction: rtl; display: flex; align-items: center; gap: .65rem;
            padding: 1rem 1.15rem; margin: .8rem 0 1.2rem; background: #FFFFFF;
            border: 1px solid var(--wq-border); border-right: 4px solid var(--wq-badge-start, var(--wq-teal));
            border-radius: 12px; box-shadow: var(--wq-shadow);
        }
        .wq-page-badge-icon { font-size: 1.35rem; }
        .wq-page-badge-text { color: var(--wq-navy); font-size: 1.2rem; font-weight: 700; }
        [data-testid="stMetric"] {
            direction: rtl; text-align: right; background: #FFFFFF; border: 1px solid var(--wq-border);
            border-top: 3px solid #89B7C1; border-radius: 12px; padding: 1rem; box-shadow: var(--wq-shadow);
        }
        [data-testid="stMetricLabel"] p { font-size: .84rem; font-weight: 600; color: var(--wq-muted); }
        [data-testid="stMetricValue"] { direction: ltr; text-align: right; font-size: 1.8rem; font-weight: 700; font-variant-numeric: tabular-nums; }
        .wq-stat-grid { direction: rtl; display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: .8rem; margin: .8rem 0 1rem; }
        .wq-stat-card {
            direction: rtl; text-align: right; position: relative; min-width: 0; min-height: 150px;
            padding: 1rem 1.05rem; border-radius: 13px; background: var(--wq-state-bg, #FFFFFF);
            border: 1px solid var(--wq-state-border, var(--wq-border));
            border-top: 3px solid var(--wq-state-accent, #687C8C); box-shadow: var(--wq-shadow);
            transition: border-color .2s ease, background-color .2s ease;
        }
        .wq-stat-label { font-size: .85rem; color: #344C5F; line-height: 1.8; font-weight: 600; }
        .wq-stat-symbol { display: inline-block; margin-left: .25rem; color: var(--wq-state-accent, #536879); font-size: 1rem; }
        .wq-stat-value { display: block; font-size: clamp(1.35rem, 1.9vw, 1.85rem); line-height: 1.8; font-weight: 800; color: var(--wq-state-accent, var(--wq-navy)); margin-top: .25rem; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
        .wq-stat-value bdi { direction: ltr; unicode-bidi: isolate; }
        .wq-stat-detail { color: #536879; font-size: .78rem; line-height: 1.9; margin-top: .3rem; }
        .wq-condition-legend { direction: rtl; display: flex; flex-wrap: wrap; gap: .55rem 1.2rem; align-items: center; margin: .65rem 0 1.8rem; }
        .wq-condition-item { display: inline-flex; align-items: center; gap: .4rem; font-size: .8rem; color: #344C5F; line-height: 1.9; }
        .wq-condition-item bdi { direction: ltr; unicode-bidi: isolate; }
        .wq-condition-dot { display: inline-flex; align-items: center; justify-content: center; width: 1rem; font-size: .85rem; flex-shrink: 0; }
        .wq-chart-note { direction: rtl; text-align: right; color: var(--wq-muted); font-size: .79rem; line-height: 1.9; margin: .4rem 0 .7rem; }
        .wq-chart-heading {
            font-family: var(--wq-font) !important; font-size: 1.22rem !important;
            font-weight: 700 !important; line-height: 1.9; color: var(--wq-navy);
            direction: rtl; text-align: right; margin: 1.2rem 0 2rem !important;
        }
        [data-testid="stVegaLiteChart"] svg, .vega-embed svg { direction: ltr; }
        .vega-embed svg text { font-family: var(--wq-font) !important; }
        [data-testid="stVegaLiteChart"], [data-testid="stAltairChart"] {
            background: #FFFFFF; padding: .6rem; border: 1px solid var(--wq-border); border-radius: 14px; box-shadow: var(--wq-shadow);
        }
        #vg-tooltip-element { direction: rtl; text-align: right; font-family: var(--wq-font); border: 1px solid var(--wq-border); border-radius: 9px; box-shadow: 0 4px 18px rgba(20,44,64,.12); }
        #vg-tooltip-element td.value { direction: ltr; unicode-bidi: isolate; text-align: left; }
        .wq-table-wrapper { direction: rtl; width: 100%; overflow-x: auto; border: 1px solid var(--wq-border); border-radius: 11px; background: #FFFFFF; margin: .5rem 0; }
        .wq-styled-table { width: 100%; border-collapse: collapse; font-family: var(--wq-font); font-size: .85rem; }
        .wq-styled-table thead th { background: #EDF3F7; color: var(--wq-navy); font-weight: 700; padding: .8rem 1rem; text-align: right; border: 0; white-space: nowrap; }
        .wq-styled-table tbody td { border: 0; border-bottom: 1px solid #E8EDF2; padding: .72rem 1rem; text-align: right; color: #344C5F; unicode-bidi: plaintext; }
        .wq-styled-table tbody td:nth-child(2) { direction: ltr; font-variant-numeric: tabular-nums; }
        .wq-styled-table tbody tr:nth-child(even) { background: #F8FAFC; }
        .wq-styled-table tbody tr:hover { background: #EFF7F8; }
        .wq-styled-table tbody tr:last-child td { border-bottom: 0; }
        [data-testid="stDataFrame"] { border: 1px solid var(--wq-border); border-radius: 11px; overflow: hidden; }
        [data-testid="stExpander"] details { background: #FFFFFF; border: 1px solid var(--wq-border); border-radius: 11px; box-shadow: none; }
        [data-testid="stExpander"] summary { direction: rtl; color: var(--wq-navy); padding: .8rem 1rem; }
        [data-testid="stExpander"] summary p { font-family: var(--wq-font); font-size: .92rem; font-weight: 600; }
        [data-testid="stAlert"], .stAlert { border-radius: 10px; border: 1px solid rgba(106,133,153,.16); box-shadow: none; }
        [data-testid="stAlert"] p, .stAlert p { font-size: .91rem; line-height: 1.9; }
        .stProgress [role="progressbar"] > div { background: var(--wq-teal); }
        .wq-status-line { direction: rtl; display: flex; align-items: center; min-height: 2.8rem; }
        .wq-status-chip { direction: rtl; display: inline-flex; align-items: center; gap: .4rem; border: 1px solid; border-radius: 999px; padding: .4rem .85rem; font-size: .82rem; font-weight: 600; }
        .wq-status-ready { background: #EDF8F1; color: #17603E; border-color: #C9E7D5; }
        .wq-status-idle { background: #EDF2F6; color: #4B6477; border-color: #D6E1E9; }
        .wq-status-running { background: #FFF5E6; color: #89500D; border-color: #F0DCB9; }
        .wq-status-strip-spacer { height: .5rem; }
        .wq-ready-banner { direction: rtl; text-align: right; background: #EDF8F7; border: 1px solid #C6E4DF; border-right: 4px solid #218575; border-radius: 12px; color: #205B52; padding: 1rem 1.2rem; font-size: .97rem; line-height: 2; margin: .8rem 0; }
        .wq-param-brief { direction: rtl; text-align: right; background: #FFFFFF; border: 1px solid var(--wq-brief-border, var(--wq-border)); border-right: 4px solid var(--wq-brief-accent, var(--wq-teal)); border-radius: 13px; padding: 1.15rem 1.3rem; margin: .4rem 0 1.3rem; box-shadow: var(--wq-shadow); }
        .wq-param-brief-head { display: flex; align-items: center; gap: .5rem; padding-bottom: .7rem; margin-bottom: .65rem; border-bottom: 1px solid #E8EDF2; }
        .wq-param-brief-icon { font-size: 1.15rem; }
        .wq-param-brief-title { font-size: .96rem; font-weight: 700; color: var(--wq-brief-accent, var(--wq-teal-dark)); }
        .wq-param-brief-body { font-size: .99rem; font-weight: 400; line-height: 2.1; color: #344C5F; }
        .wq-brief-turbidity { --wq-brief-accent: #086873; --wq-brief-border: #D1E3E6; }
        .wq-brief-chlorophyll { --wq-brief-accent: #28724C; --wq-brief-border: #D5E5DA; }
        .wq-brief-cdom { --wq-brief-accent: #8F602A; --wq-brief-border: #E8DECF; }
        .wq-summary-card { direction: rtl; text-align: right; background: #FFFFFF; border: 1px solid var(--wq-border); border-right: 4px solid var(--wq-teal); border-radius: 13px; padding: 1rem 1.3rem; box-shadow: var(--wq-shadow); }
        .wq-summary-line { font-size: 1rem; line-height: 2; color: #344C5F; padding: .7rem 0; border-bottom: 1px solid #E8EDF2; }
        .wq-summary-line:last-child { border-bottom: 0; }
        .wq-quick-hint { direction: rtl; text-align: right; font-size: .91rem; color: var(--wq-muted); margin: 1rem 0 .7rem; line-height: 1.9; }
        [data-testid="stImageCaption"] { direction: rtl; text-align: center; font-family: var(--wq-font); font-size: .82rem; color: var(--wq-muted); line-height: 1.9; }
        /* Only the ROI map receives a frame; charts, chat components and
           other embedded tools retain their own dimensions and event surfaces. */
        [class*="st-key-roi_map_"] iframe,
        [class*="st-key-roi_map_locked_"] iframe,
        iframe[title="streamlit_folium.st_folium"],
        iframe[src*="streamlit_folium.st_folium"] {
            display: block; margin: .2rem auto .6rem; border: 1px solid #CBD9E2 !important;
            border-radius: 15px; box-shadow: 0 7px 24px rgba(20,44,64,.10);
        }
        section[data-testid="stSidebar"] { background: #142C40; direction: rtl; text-align: right; }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #E6F0F4; }
        @media (max-width: 1100px) {
            .block-container { padding-left: 1.3rem; padding-right: 1.3rem; }
            .wq-stat-grid { grid-template-columns: repeat(3, minmax(0,1fr)); }
            .wq-stat-value { font-size: 1.65rem; }
            [class*="st-key-wqnav_"] button { padding: .45rem .45rem; }
            [class*="st-key-wqnav_"] button p { font-size: .8rem; }
        }
        @media (max-width: 700px) {
            .block-container { padding-top: 1.3rem; padding-left: .85rem; padding-right: .85rem; }
            .wq-app-header { padding: 1.1rem 1rem .5rem; gap: .7rem; }
            .wq-app-subtitle { padding: .1rem 1rem 1rem; font-size: .78rem; }
            .wq-app-title { font-size: 1.35rem; }
            .wq-app-eyebrow { font-size: .66rem; }
            .wq-stat-grid { grid-template-columns: repeat(2, minmax(0,1fr)); gap: .65rem; }
            .wq-stat-card { min-height: 135px; padding: .85rem; }
            .wq-stat-label { font-size: .76rem; }
            .wq-step-title { font-size: .98rem; }
            .wq-param-brief { padding: 1rem; }
            .wq-condition-legend { gap: .4rem .8rem; }
        }
        @media (max-width: 380px) { .wq-stat-grid { grid-template-columns: 1fr; } }
        @media (prefers-reduced-motion: reduce) {
            .stButton button, .stDownloadButton button, .wq-stat-card { transition: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _inject_persian_chat_css()
    _force_quick_button_style()


def _render_step_header(step_number, icon, title):
    """Colored, icon-led replacement for the old plain st.header() numbered
    workflow titles (1️⃣/2️⃣/3️⃣ …). Purely presentational."""
    st.markdown(
        f"""
        <div class="wq-step-header">
            <div class="wq-step-number">{step_number}</div>
            <div class="wq-step-icon">{icon}</div>
            <div class="wq-step-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_active_section_badge(icon, title, start_hex, end_hex):
    """Colored 'you are here' badge shown at the top of each of the three
    result tabs (Turbidity / Chlorophyll / Expert chat) so the currently
    active page is always visually obvious. Purely presentational."""
    st.markdown(
        f"""
        <div class="wq-page-badge" style="--wq-badge-start:{start_hex}; --wq-badge-end:{end_hex};">
            <span class="wq-page-badge-icon">{icon}</span>
            <span class="wq-page-badge-text">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_persian_dataframe_html(df):
    """Render a DataFrame as a custom-styled, RTL Persian HTML table. This is
    a pure presentation swap for st.dataframe: identical columns and values
    are shown, only the typography/spacing/colors differ (st.dataframe's
    canvas-based grid cannot be restyled with CSS, which is why a plain HTML
    table is used here instead)."""
    import html as _html

    # Isolate dates/numbers from the surrounding Persian reading direction.
    # Escape cell contents before adding markup; the original frame is untouched.
    def _display_cell(value):
        text = str(value)
        escaped = _html.escape(text)
        if text and all(char in '0123456789.,+−-% /:—' for char in text):
            return f'<bdi dir="ltr">{escaped}</bdi>'
        return escaped

    formatters = {column: _display_cell for column in df.columns}
    html_table = df.to_html(index=False, escape=False, border=0, classes="wq-styled-table",
                            formatters=formatters)
    st.markdown(f'<div class="wq-table-wrapper">{html_table}</div>', unsafe_allow_html=True)


# =============================================================================
# App logo (replaces the 🌊 wave emoji at the top of the UI)
# =============================================================================
@st.cache_data(show_spinner=False)
def _get_app_logo_base64():
    """Load the app icon from Google Drive and return it as a base64 string
    for inline embedding next to the page title.

    The local assets/app_logo.png file is not present on this deployment
    (Posit Cloud), so the logo is fetched from Google Drive instead using
    the file's shareable ID. The Drive file must be shared as "Anyone with
    the link -> Viewer" for this request to succeed.

    Cached so the file is only downloaded once per session. On any failure
    (network issue, permissions, bad ID) this returns an empty string
    instead of raising, so a missing/unreachable logo never crashes the app
    — the <img> tag will just render nothing instead.
    """
    GOOGLE_DRIVE_LOGO_FILE_ID = "1sX8iMkqYvYDJ-XocOa8Hmr0e6-4R-n6i"
    url = f"https://drive.google.com/thumbnail?id={GOOGLE_DRIVE_LOGO_FILE_ID}&sz=w1000"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return base64.b64encode(response.content).decode()
    except Exception:
        return ""


# =============================================================================
# Top navigation (four pages) — replaces the old single scrolling page
# =============================================================================
# key, icon, Persian title
NAV_PAGES = [
    ("setup",       "🛰️", "تعریف پایش"),
    ("turbidity",   "🌊", "کدورت"),
    ("chlorophyll", "🌿", "کلروفیل"),
    ("cdom",        "🍂", "مواد آلی"),
    ("summary",     "🧭", "خلاصه"),
    ("chat",        "🤖", "چت بات"),
]


def _has_any_results():
    """True once at least one parameter has monitoring results in memory."""
    return any(bool(st.session_state.results.get(p)) for p in ALL_PARAMETERS)



def _render_app_header():
    """Persian product header with the existing cached logo and an offline fallback."""
    logo_b64 = _get_app_logo_base64()
    logo_tag = (
        f'<img src="data:image/png;base64,{logo_b64}" alt="نشان سامانه پایش کیفیت آب">'
        if logo_b64 else '<span class="wq-app-emblem" aria-hidden="true">🌊</span>'
    )
    st.markdown(
        f"""
        <div class="wq-app-header" dir="rtl">
            {logo_tag}
            <div>
                <span class="wq-app-eyebrow">پایش ماهواره‌ای و تحلیل محیط‌زیست</span>
                <span class="wq-app-title">سامانه پایش کیفیت آب</span>
            </div>
        </div>
        <div class="wq-app-subtitle" dir="rtl">
            پایش <b>کدورت</b>، <b>کلروفیل</b> و <b>مواد آلی محلول رنگی</b>
            با تصاویر ماهواره‌ای <bdi dir="ltr">Sentinel-2</bdi>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_top_nav():
    """
    The four-page tab bar at the top of the app. The three result pages stay
    visible at all times (so the structure of the app is obvious from the first
    moment) but remain locked until a monitoring run has produced results.
    """
    results_ready = _has_any_results()
    busy = st.session_state.processing_in_progress

    # Marker used by the CSS to find this row on Streamlit versions that do not
    # expose per-key element classes.
    st.markdown('<div class="wq-nav-anchor"></div>', unsafe_allow_html=True)

    cols = st.columns(len(NAV_PAGES), gap="small")

    # Persian reading order: the first page («تعریف پایش») goes in the RIGHTMOST
    # column, the last one («چت بات») on the left. Filling the columns in
    # reverse is what produces that, independently of any CSS.
    for col, (page_key, icon, title) in zip(list(cols)[::-1], NAV_PAGES):
        locked = (page_key != "setup") and (not results_ready or busy)
        is_active = (st.session_state.active_page == page_key)
        label = f"{icon}  {title}" + ("  🔒" if locked else "")

        clicked = col.button(
            label,
            key=f"wqnav_{page_key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            disabled=locked,
        )
        if clicked and not is_active:
            st.session_state.active_page = page_key
            st.rerun()

    if not results_ready and not busy:
        st.markdown(
            '<div class="wq-nav-hint">🔒 صفحه‌های نتایج پس از اجرای پایش فعال می‌شوند.</div>',
            unsafe_allow_html=True,
        )


def _render_status_strip():
    """
    Run status + 'clear results' action. Shown on the first page directly below
    the «شروع پایش» / «ادامه از محل قطع» buttons (it used to live in the
    sidebar, which has been removed in favour of the top navigation).
    """
    st.markdown('<div class="wq-status-strip-spacer"></div>', unsafe_allow_html=True)

    col_clear, col_status = st.columns([1, 3])

    with col_status:
        if st.session_state.processing_in_progress:
            chip_class, chip_text = "wq-status-running", "⏳ در حال پردازش..."
        elif _has_any_results():
            chip_class, chip_text = "wq-status-ready", "✅ نتایج پایش موجود است"
        else:
            chip_class, chip_text = "wq-status-idle", "ℹ️ هنوز پایشی انجام نشده است"
        st.markdown(
            f'<div class="wq-status-line">'
            f'<span class="wq-status-chip {chip_class}">{chip_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_clear:
        if st.button(
            "🗑️ پاک کردن نتایج",
            use_container_width=True,
            disabled=st.session_state.processing_in_progress,
        ):
            st.session_state.downloaded_months = empty_param_dict(dict)
            st.session_state.month_statuses = empty_param_dict(dict)
            st.session_state.results = empty_param_dict(list)
            st.session_state.mean_data = empty_param_dict(dict)
            st.session_state.download_summary = {}
            st.session_state.cdom_display_range = None
            st.session_state.current_temp_dir = None
            st.session_state.processing_config = None
            st.session_state.processing_complete = False
            st.session_state.processing_in_progress = False
            st.session_state.pending_run = None
            st.session_state.resume_after_interruption = False
            st.session_state.connection_interrupted = False
            st.session_state.last_run_error = None
            st.session_state.expert_analysis_json = None
            st.session_state.expert_analysis_signature = None
            st.session_state.executive_summary = None
            st.session_state.executive_summary_signature = None
            st.session_state.expert_chat_history = []
            st.session_state.active_page = 'setup'
            st.rerun()


# =============================================================================
# Region-of-interest map
# =============================================================================
# The drawings made with the Leaflet Draw toolbar only live inside the browser
# widget, so they vanished on the very next Streamlit rerun (for example right
# after «ذخیره منطقه»). Every saved region — and the freshly drawn, not yet
# saved one — is therefore re-added to the map as a real folium layer on every
# run, which keeps the region permanently visible and lets the map be shown in
# a read-only form while a monitoring run is in progress.
# =============================================================================
ROI_COLOR_SAVED = "#0E8E99"      # teal — a saved region
ROI_COLOR_SELECTED = "#F5A524"   # amber — the region the run will use
ROI_COLOR_DRAFT = "#E04B2F"      # red   — drawn but not saved yet


def _st_folium_compat(fmap, **kwargs):
    """
    Call st_folium while staying compatible with older streamlit-folium
    releases that do not support `key` / `returned_objects`. Limiting the
    returned objects (when available) is what stops harmless panning and
    zooming from triggering a full app rerun.
    """
    try:
        supported = inspect.signature(st_folium).parameters
    except Exception:
        supported = {}
    safe_kwargs = {k: v for k, v in kwargs.items() if not supported or k in supported}
    return st_folium(fmap, **safe_kwargs)


def _polygon_to_latlon(polygon):
    """Shapely stores (lon, lat); folium wants (lat, lon)."""
    return [(y, x) for x, y in polygon.exterior.coords]


def polygon_area_km2(polygon):
    """
    Approximate area of a lat/lon polygon in square kilometres.

    One degree of latitude is ~110.57 km everywhere, but one degree of
    longitude shrinks towards the poles, so the polygon's own mid-latitude is
    used to scale it. (The previous `area * 111 * 111` ignored that and
    overestimated by roughly 20% at Iran's latitudes.)
    """
    try:
        min_lat = polygon.bounds[1]
        max_lat = polygon.bounds[3]
        mean_lat = math.radians((min_lat + max_lat) / 2.0)
        km_per_deg_lat = 110.574
        km_per_deg_lon = 111.320 * math.cos(mean_lat)
        return abs(polygon.area) * km_per_deg_lat * km_per_deg_lon
    except Exception:
        return 0.0


def _format_area_km2(area_km2):
    """Readable area text — more decimals for small regions."""
    if area_km2 >= 100:
        return f"{area_km2:,.0f} کیلومتر مربع"
    if area_km2 >= 10:
        return f"{area_km2:.1f} کیلومتر مربع"
    return f"{area_km2:.2f} کیلومتر مربع"


# =============================================================================
# Live area readout while the user is drawing
# =============================================================================
# Leaflet.draw knows the shape being drawn long before the user finishes it, so
# a small badge is pinned under the bounding box of the in-progress shape and
# updated on every mouse move. The area itself comes from Leaflet.draw's own
# L.GeometryUtil.geodesicArea(), i.e. a proper geodesic area on the ellipsoid —
# not a degree-square approximation.
#
# Everything is wrapped in try/catch: if a future Leaflet.draw release renames
# an internal, the badge silently disappears and drawing keeps working.
_LIVE_AREA_JS = """
{% macro script(this, kwargs) %}
(function () {
  var MAP = {{ this._parent.get_name() }};

  function setup() {
    try {
      if (typeof L === 'undefined' || !MAP || !MAP.getContainer) { return false; }
      var container = MAP.getContainer();
      if (!container) { return false; }

      var label = L.DomUtil.create('div', '', container);
      label.style.cssText = [
        'position:absolute',
        'z-index:1000',
        'display:none',
        'pointer-events:none',
        'white-space:nowrap',
        'direction:rtl',
        'transform:translateX(-50%)',
        'background:rgba(255,255,255,0.97)',
        'color:#0A3F4A',
        'border:2px solid #E04B2F',
        'border-radius:999px',
        'padding:4px 14px',
        "font-family:'B Nazanin','BNazanin','Vazirmatn',Tahoma,sans-serif",
        'font-size:14px',
        'font-weight:700',
        'box-shadow:0 3px 10px rgba(10,63,74,0.40)'
      ].join(';');

      var state = { active: false, type: null, anchor: null, known: null, pts: null };

      // Same spherical-excess formula Leaflet.draw uses, kept local so the
      // badge does not depend on L.GeometryUtil being present.
      function geodesicArea(ring) {
        var n = ring.length, area = 0.0, d2r = Math.PI / 180, i, p1, p2;
        if (n < 3) { return 0; }
        for (i = 0; i < n; i++) {
          p1 = ring[i];
          p2 = ring[(i + 1) % n];
          area += ((p2.lng - p1.lng) * d2r) *
                  (2 + Math.sin(p1.lat * d2r) + Math.sin(p2.lat * d2r));
        }
        return Math.abs(area * 6378137.0 * 6378137.0 / 2.0);
      }

      function formatArea(m2) {
        var km2 = m2 / 1000000.0;
        if (km2 >= 100) { return km2.toFixed(0) + ' کیلومتر مربع'; }
        if (km2 >= 10) { return km2.toFixed(1) + ' کیلومتر مربع'; }
        if (km2 >= 0.01) { return km2.toFixed(2) + ' کیلومتر مربع'; }
        return Math.round(m2) + ' متر مربع';
      }

      function hide() {
        label.style.display = 'none';
      }

      function reset() {
        state.active = false;
        state.type = null;
        state.anchor = null;
        state.known = null;
        state.pts = null;
        hide();
      }

      function draw(ring) {
        if (!ring || ring.length < 3) { hide(); return; }
        var area = geodesicArea(ring);
        if (!isFinite(area) || area <= 0) { hide(); return; }
        var south = ring[0].lat, west = ring[0].lng, east = ring[0].lng, i;
        for (i = 1; i < ring.length; i++) {
          if (ring[i].lat < south) { south = ring[i].lat; }
          if (ring[i].lng < west) { west = ring[i].lng; }
          if (ring[i].lng > east) { east = ring[i].lng; }
        }
        var pt = MAP.latLngToContainerPoint(L.latLng(south, (west + east) / 2));
        label.innerHTML = 'مساحت: ' + formatArea(area);
        label.style.left = pt.x + 'px';
        label.style.top = (pt.y + 10) + 'px';
        label.style.display = 'block';
        state.pts = ring;
      }

      function flatten(latlngs) {
        while (latlngs && latlngs.length && latlngs[0] && latlngs[0].length !== undefined) {
          latlngs = latlngs[0];
        }
        return latlngs || [];
      }

      // Any polygon/polyline layer that appeared after drawing started is the
      // shape currently being drawn. No private Leaflet.draw fields involved.
      function newShapeLayer() {
        var found = null;
        try {
          MAP.eachLayer(function (layer) {
            if (found || !layer || typeof layer.getLatLngs !== 'function') { return; }
            if (state.known && state.known[L.Util.stamp(layer)]) { return; }
            found = layer;
          });
        } catch (err) {
          return null;
        }
        return found;
      }

      function ringFor(cursor) {
        // Rectangle: build it from the anchor corner and the cursor, so the
        // area is live from the very first pixel of the drag.
        if (state.type === 'rectangle' && state.anchor && cursor) {
          var a = state.anchor;
          if (Math.abs(a.lat - cursor.lat) > 1e-12 && Math.abs(a.lng - cursor.lng) > 1e-12) {
            return [
              L.latLng(a.lat, a.lng),
              L.latLng(a.lat, cursor.lng),
              L.latLng(cursor.lat, cursor.lng),
              L.latLng(cursor.lat, a.lng)
            ];
          }
          return null;
        }

        var layer = newShapeLayer();
        if (layer) {
          var ring = flatten(layer.getLatLngs()).slice();
          var isRect = !!(L.Rectangle && layer instanceof L.Rectangle);
          if (!isRect && cursor) { ring.push(cursor); }
          return ring;
        }
        return null;
      }

      function onMove(cursor) {
        if (!state.active || !cursor) { return; }
        draw(ringFor(cursor));
      }

      MAP.on('draw:drawstart', function (e) {
        state.active = true;
        state.type = (e && e.layerType) ? e.layerType : null;
        state.anchor = null;
        state.pts = null;
        state.known = {};
        try {
          MAP.eachLayer(function (layer) { state.known[L.Util.stamp(layer)] = true; });
        } catch (err) {
          state.known = null;
        }
        hide();
      });

      MAP.on('draw:drawstop', reset);
      MAP.on('draw:created', reset);
      MAP.on('draw:canceled', reset);

      MAP.on('mousedown', function (e) {
        if (state.active && !state.anchor && e && e.latlng) { state.anchor = e.latlng; }
      });
      MAP.on('mousemove', function (e) {
        if (e && e.latlng) { onMove(e.latlng); }
      });

      // DOM-level backstop: fires even if Leaflet's own map events are
      // swallowed by the active draw handler.
      container.addEventListener('mousedown', function (ev) {
        if (!state.active || state.anchor) { return; }
        try { state.anchor = MAP.mouseEventToLatLng(ev); } catch (err) { return; }
      }, true);
      container.addEventListener('mousemove', function (ev) {
        if (!state.active) { return; }
        try { onMove(MAP.mouseEventToLatLng(ev)); } catch (err) { return; }
      }, true);

      MAP.on('move', function () { if (state.active && state.pts) { draw(state.pts); } });
      MAP.on('zoom', function () { if (state.active && state.pts) { draw(state.pts); } });

      return true;
    } catch (err) {
      return true;
    }
  }

  // The map variable may not be assigned yet depending on script order, so
  // keep trying briefly instead of giving up on the first pass.
  var tries = 0;
  function attempt() {
    tries += 1;
    var ok = false;
    try { ok = setup(); } catch (err) { ok = true; }
    if (!ok && tries < 40) { setTimeout(attempt, 100); }
  }
  attempt();
})();
{% endmacro %}
"""


if _LIVE_AREA_SUPPORTED:
    class LiveAreaLabel(_BrancaMacroElement):
        """Folium element that injects the live-area script into the map."""
        _template = _JinjaTemplate(_LIVE_AREA_JS)

        def __init__(self):
            super().__init__()
            self._name = "LiveAreaLabel"
else:
    LiveAreaLabel = None


def _add_area_label(layer, polygon, text, color):
    """
    Small floating badge placed just below the polygon showing its area, so the
    user gets an immediate sense of how large the selected region is.

    The badge lives inside the map's own iframe, where the app's stylesheet does
    not reach — hence the inline styling.
    """
    try:
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        anchor_lat = min_lat
        anchor_lon = (min_lon + max_lon) / 2.0
    except Exception:
        return

    html = (
        f'<div style="'
        f'display:inline-block; white-space:nowrap; direction:rtl;'
        f'background:rgba(255,255,255,0.94); color:#0A3F4A;'
        f'border:2px solid {color}; border-left:none; border-right:none;'
        f'border-top:3px solid {color}; border-bottom:3px solid {color};'
        f'border-radius:999px; padding:3px 10px;'
        f'font-family:\'B Nazanin\',\'BNazanin\',\'Vazirmatn\',Tahoma,sans-serif;'
        f'font-size:13px; font-weight:700; line-height:1.6;'
        f'box-shadow:0 2px 6px rgba(10,63,74,0.35);'
        f'transform:translateX(-50%);'
        f'">{text}</div>'
    )

    folium.Marker(
        location=[anchor_lat, anchor_lon],
        icon=folium.DivIcon(html=html, icon_size=(0, 0), icon_anchor=(0, -6)),
    ).add_to(layer)


def _build_roi_map(interactive=True, highlight_index=None):
    """Build the ROI map with every saved region drawn as a permanent layer."""
    saved = list(st.session_state.drawn_polygons)
    draft = st.session_state.last_drawn_polygon
    draft_is_saved = draft is not None and any(p.equals(draft) for p in saved)

    shapes = saved + ([draft] if (draft is not None and not draft_is_saved) else [])

    if shapes:
        xs_min = min(s.bounds[0] for s in shapes)
        ys_min = min(s.bounds[1] for s in shapes)
        xs_max = max(s.bounds[2] for s in shapes)
        ys_max = max(s.bounds[3] for s in shapes)
        center = [(ys_min + ys_max) / 2, (xs_min + xs_max) / 2]
    else:
        center = [35.6892, 51.3890]

    fmap = folium.Map(location=center, zoom_start=8, control_scale=True)

    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google', name='تصویر ماهواره‌ای'
    ).add_to(fmap)

    if interactive:
        # Only the two drawing tools are shown. The plugin's own edit/delete
        # toolbar and its Export button are switched off: they act on the
        # browser-side layer only, so they never matched what the app actually
        # had stored and were a source of confusion. Deleting a region is done
        # with the 🗑️ button in «مناطق ذخیره‌شده» (saved regions) and with
        # «حذف منطقه رسم‌شده» (a region drawn but not yet saved).
        plugins.Draw(
            export=False,
            position='topleft',
            draw_options={
                'polyline': False, 'rectangle': True, 'polygon': True,
                'circle': False, 'marker': False, 'circlemarker': False,
            },
            edit_options={'edit': False, 'remove': False},
        ).add_to(fmap)

        # Live area readout that follows the shape as it is being drawn
        if LiveAreaLabel is not None:
            try:
                fmap.add_child(LiveAreaLabel())
            except Exception:
                pass

    roi_layer = folium.FeatureGroup(name='مناطق انتخاب‌شده', show=True)

    for i, polygon in enumerate(saved):
        is_selected = (highlight_index is not None and i == highlight_index)
        color = ROI_COLOR_SELECTED if is_selected else ROI_COLOR_SAVED
        area_text = _format_area_km2(polygon_area_km2(polygon))
        label = f"منطقه {i + 1}" + (" — انتخاب‌شده برای پایش" if is_selected else "")
        folium.Polygon(
            locations=_polygon_to_latlon(polygon),
            color=color,
            weight=5 if is_selected else 3,
            opacity=0.95,
            fill=True,
            fill_color=color,
            fill_opacity=0.22 if is_selected else 0.12,
            tooltip=f"{label} — {area_text}",
        ).add_to(roi_layer)
        _add_area_label(roi_layer, polygon, f"منطقه {i + 1} • {area_text}", color)

    if draft is not None and not draft_is_saved:
        draft_area_text = _format_area_km2(polygon_area_km2(draft))
        folium.Polygon(
            locations=_polygon_to_latlon(draft),
            color=ROI_COLOR_DRAFT,
            weight=3,
            dash_array='8,6',
            fill=True,
            fill_color=ROI_COLOR_DRAFT,
            fill_opacity=0.10,
            tooltip=f"منطقه رسم‌شده (ذخیره‌نشده) — {draft_area_text}",
        ).add_to(roi_layer)
        _add_area_label(
            roi_layer, draft, f"منطقه جدید • {draft_area_text}", ROI_COLOR_DRAFT
        )

    roi_layer.add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)

    if shapes:
        fmap.fit_bounds([[ys_min, xs_min], [ys_max, xs_max]], padding=(25, 25))

    return fmap


def _render_roi_map():
    """
    Render the map. During a monitoring run it stays on screen as a read-only
    view (drawing tools removed, no interaction sent back to the server) instead
    of being replaced by a placeholder message.
    """
    busy = st.session_state.processing_in_progress
    highlight = st.session_state.selected_region_index if st.session_state.drawn_polygons else None

    map_col_l, map_col_c, map_col_r = st.columns([0.35, 6.3, 0.35])

    with map_col_c:
        fmap = _build_roi_map(interactive=not busy, highlight_index=highlight)

        # The version suffix forces a fresh widget after a save or a delete, so
        # the drawing toolbar cannot hand back a region the user just removed.
        version = st.session_state.map_version

        if busy:
            # returned_objects=[] -> the component never sends data back, so the
            # map cannot trigger a rerun in the middle of the processing run.
            _st_folium_compat(
                fmap, key=f"roi_map_locked_{version}", width=1000, height=560,
                use_container_width=True,
                returned_objects=[],
            )
            st.caption("🔒 نقشه در حین پردازش فقط برای نمایش است؛ منطقه انتخاب‌شده با رنگ نارنجی مشخص شده است.")
            return

        map_data = _st_folium_compat(
            fmap, key=f"roi_map_{version}", width=1000, height=560,
            use_container_width=True,
            returned_objects=["last_active_drawing"],
        )

        if map_data and map_data.get('last_active_drawing'):
            geom = (map_data['last_active_drawing'] or {}).get('geometry', {}) or {}
            if geom.get('type') == 'Polygon':
                try:
                    new_polygon = Polygon(geom['coordinates'][0])
                except Exception:
                    new_polygon = None
                if new_polygon is not None and (
                    st.session_state.last_drawn_polygon is None
                    or not st.session_state.last_drawn_polygon.equals(new_polygon)
                ):
                    st.session_state.last_drawn_polygon = new_polygon
                    st.rerun()   # redraw immediately as a permanent layer

        has_unsaved_draft = (
            st.session_state.last_drawn_polygon is not None
            and not any(p.equals(st.session_state.last_drawn_polygon)
                        for p in st.session_state.drawn_polygons)
        )

        if st.session_state.last_drawn_polygon is not None:
            draft_area = _format_area_km2(
                polygon_area_km2(st.session_state.last_drawn_polygon)
            )
            if has_unsaved_draft:
                st.info(
                    f"✅ منطقه رسم شد (خط‌چین قرمز) — مساحت تقریبی: **{draft_area}**. "
                    "برای نگه‌داشتن آن، «ذخیره منطقه» را بزنید."
                )
            else:
                st.success("✅ منطقه انتخاب‌شده روی نقشه نمایش داده می‌شود.")

        # The discard button only appears while there is something to discard,
        # so there is never an inactive delete control on screen.
        if has_unsaved_draft:
            save_col, discard_col = st.columns([3, 1])
        else:
            save_col, discard_col = st.container(), None

        if save_col.button("💾 ذخیره منطقه", use_container_width=True):
            if st.session_state.last_drawn_polygon:
                is_duplicate = any(
                    existing.equals(st.session_state.last_drawn_polygon)
                    for existing in st.session_state.drawn_polygons
                )
                if not is_duplicate:
                    st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                    st.session_state.selected_region_index = len(st.session_state.drawn_polygons) - 1
                    st.session_state.map_version += 1
                    st.success("✅ منطقه ذخیره شد!")
                    st.rerun()
                else:
                    st.warning("⚠️ این منطقه قبلاً ذخیره شده است")
            else:
                st.warning("⚠️ ابتدا یک منطقه را روی نقشه رسم کنید")

        # Removes a region that was drawn by mistake and never saved.
        if discard_col is not None and discard_col.button(
            "✖️ حذف منطقه رسم‌شده",
            use_container_width=True,
            help="منطقه‌ای که رسم شده اما ذخیره نشده است را از نقشه بردارید",
        ):
            st.session_state.last_drawn_polygon = None
            st.session_state.map_version += 1
            st.rerun()


# =============================================================================
# Page 1 — area of interest, time period, and "start monitoring"
# =============================================================================
def render_setup_page():
    # ==========================================================================
    # 1. Region selection
    # ==========================================================================
    _render_step_header(1, "🗺️", "انتخاب منطقه مورد نظر (بدنه آبی)")

    _render_roi_map()

    if st.session_state.drawn_polygons:
        st.subheader("📍 مناطق ذخیره‌شده")
        for i, p in enumerate(st.session_state.drawn_polygons):
            c1, c2, c3 = st.columns([3, 1, 1])
            centroid = p.centroid
            c1.write(f"**منطقه {i+1}**: ~{_format_area_km2(polygon_area_km2(p))}")
            c2.write(f"مرکز: ({centroid.y:.4f}, {centroid.x:.4f})")
            if c3.button("🗑️", key=f"del_{i}", disabled=st.session_state.processing_in_progress):
                removed = st.session_state.drawn_polygons.pop(i)

                # The deleted region must also leave the map. Two things keep a
                # copy of it: the "last drawn" polygon in session state, and the
                # map widget's own drawing toolbar — clear the first, and bump
                # map_version so the widget is remounted and forgets the second.
                if st.session_state.last_drawn_polygon is not None and \
                        st.session_state.last_drawn_polygon.equals(removed):
                    st.session_state.last_drawn_polygon = None
                st.session_state.map_version += 1

                if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
                    st.session_state.selected_region_index = max(0, len(st.session_state.drawn_polygons) - 1)
                st.rerun()

    # ==========================================================================
    # 2. Time period
    # ==========================================================================
    _render_step_header(2, "📅", "بازه زمانی (تقویم میلادی)")

    # The month/year drop-downs below replace the previous calendar widget on
    # purpose: that widget is rendered by the browser and, on a Persian-locale
    # browser, it showed a Jalali (solar) calendar. Picking the Gregorian year
    # and month explicitly removes any dependence on the browser's locale — and
    # the pipeline works month by month anyway, so the day was never used.
    _today = date.today()
    _default_end_year, _default_end_month = _today.year, _today.month
    _default_start_year, _default_start_month = _today.year - 1, _today.month

    year_options = list(range(2017, _today.year + 1))   # Sentinel-2 L2A starts 2017

    def _month_label(m):
        return f"{m:02d} — {GREGORIAN_MONTHS_FA[m - 1]}"

    dc1, dc2, dc3, dc4 = st.columns(4)
    # Right-to-left reading order: «از» on the right, «تا» on the left
    col_start_year, col_start_month, col_end_year, col_end_month = dc4, dc3, dc2, dc1

    start_year = col_start_year.selectbox(
        "از سال (میلادی)", year_options,
        index=year_options.index(_default_start_year) if _default_start_year in year_options else 0,
        disabled=st.session_state.processing_in_progress, key="start_year")
    start_month = col_start_month.selectbox(
        "از ماه", list(range(1, 13)), index=_default_start_month - 1,
        format_func=_month_label,
        disabled=st.session_state.processing_in_progress, key="start_month")
    end_year = col_end_year.selectbox(
        "تا سال (میلادی)", year_options,
        index=year_options.index(_default_end_year) if _default_end_year in year_options else len(year_options) - 1,
        disabled=st.session_state.processing_in_progress, key="end_year")
    end_month = col_end_month.selectbox(
        "تا ماه (غیرشامل)", list(range(1, 13)), index=_default_end_month - 1,
        format_func=_month_label,
        disabled=st.session_state.processing_in_progress, key="end_month")

    start = date(start_year, start_month, 1)
    end = date(end_year, end_month, 1)

    if start >= end:
        st.error("بازه تاریخ نامعتبر است — تاریخ پایان باید بعد از تاریخ شروع باشد.")
        st.stop()

    months = (end.year - start.year) * 12 + (end.month - start.month)
    st.info(f"📅 بازه انتخابی: **{months} ماه**")

    # ==========================================================================
    # 3. Run analysis — fully automatic (preprocessing + both indices)
    # ==========================================================================
    _render_step_header(3, "🚀", "اجرای پایش")

    selected_polygon = None

    if st.session_state.drawn_polygons:
        region_options = []
        for i, p in enumerate(st.session_state.drawn_polygons):
            region_options.append(f"منطقه {i+1} (~{_format_area_km2(polygon_area_km2(p))})")

        if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
            st.session_state.selected_region_index = 0

        selected_idx = st.selectbox(
            "🎯 انتخاب منطقه",
            range(len(region_options)),
            format_func=lambda i: region_options[i],
            index=st.session_state.selected_region_index,
            disabled=st.session_state.processing_in_progress
        )

        st.session_state.selected_region_index = selected_idx
        selected_polygon = st.session_state.drawn_polygons[selected_idx]

    elif st.session_state.last_drawn_polygon is not None:
        selected_polygon = st.session_state.last_drawn_polygon
        st.info("ℹ️ استفاده از منطقه رسم‌شده (ذخیره‌نشده)")
    else:
        st.warning("⚠️ ابتدا یک منطقه را روی نقشه رسم کنید")

    st.caption("پس از اجرا، پیش‌پردازش (حذف ابر، حذف برف، استخراج بدنه آب)، سپس شاخص کدورت، شاخص کلروفیل و شاخص مواد آلی محلول به‌طور خودکار محاسبه می‌شوند.")

    if st.session_state.processing_in_progress:
        if st.button('⏸️ توقف و حفظ پیشرفت', key='pause_monitoring'):
            st.session_state.pending_run = None
            st.session_state.processing_in_progress = False
            st.session_state.resume_after_interruption = True
            st.rerun()

    selected_config = {
        'polygon_coords': list(selected_polygon.exterior.coords),
        'start_date': start.strftime('%Y-%m-%d'),
        'end_date': end.strftime('%Y-%m-%d'),
        'cloudy_pixel_percentage': CLOUD_THRESHOLD,
        'scale': 10,
    } if selected_polygon is not None else None

    # Recover only the explicitly selected region/period after session loss.
    # Never adopt the latest folder from another server user.
    resume_config = st.session_state.processing_config
    if resume_config is None and selected_config is not None:
        candidate_dir = get_cache_dir_for_config(selected_config)
        try:
            if any(name.endswith('.tif') or name == RUN_STATE_FILE or name.startswith('bands_')
                   for name in os.listdir(candidate_dir)):
                resume_config = selected_config
        except OSError:
            pass  # A concurrently evicted folder must not hide the controls.

    btn_col1, btn_col2, btn_col3 = st.columns(3)

    start_btn = btn_col1.button(
        "🚀 شروع پایش",
        type="primary",
        disabled=st.session_state.processing_in_progress or selected_polygon is None
    )

    # Even an initialization failure before the first download must be retryable.
    resume_btn = btn_col2.button(
        "🔄 ادامه از محل قطع",
        disabled=(
            resume_config is None or
            st.session_state.processing_in_progress
        ),
        help="اگر اتصال اینترنت قطع شد، پس از اتصال مجدد این دکمه را فشار دهید تا دانلود از همانجا ادامه یابد."
    )
    has_failed = any(entry.get('status') == STATUS_FAILED
                     for statuses in st.session_state.month_statuses.values()
                     for entry in statuses.values())
    retry_btn = btn_col3.button(
        '🔁 تلاش دوباره برای ماه‌های ناموفق',
        disabled=(not has_failed or st.session_state.processing_config is None
                  or st.session_state.processing_in_progress),
        help='فقط ماه‌های ناموفق یا باقی‌مانده بررسی می‌شوند؛ تصاویر کامل دوباره دریافت نمی‌شوند.',
    )
    st.caption('تصاویر و نتایج کامل این محدوده و بازه زمانی از حافظه ذخیره‌شده استفاده می‌شوند؛ '
               'شروع پایش فایل‌های دریافت‌شده را پاک نمی‌کند.')

    # --- A click only ARMS the run; the work happens on the next script run ---
    # Doing it this way means the whole page (map included) is already rendered
    # in its locked, read-only state before the long download loop begins — the
    # map therefore stays visible for the entire run and can no longer trigger a
    # rerun that would abort the processing halfway through.
    if start_btn:
        _prepare_monitoring_run(selected_config)
        st.rerun()

    if (resume_btn or retry_btn) and resume_config is not None:
        st.session_state.processing_config = resume_config
        st.session_state.processing_in_progress = True
        st.session_state.resume_after_interruption = False
        st.session_state.pending_run = 'resume'
        st.rerun()

    # Execute only a deliberate Start/Resume click. An abandoned busy flag must
    # not repeatedly start the same failing request and lock every button again.
    pending = st.session_state.get('pending_run')

    if pending and st.session_state.processing_config is not None:
        config = st.session_state.processing_config

        # Start and Resume both retain validated disk cache.
        is_fresh = (pending == 'start')

        _execute_monitoring_run(config, resume=not is_fresh)
        st.rerun()

    # Hint when a partial run can be resumed
    if st.session_state.resume_after_interruption and not st.session_state.processing_in_progress:
        st.warning(
            "⚠️ پایش کامل نشده است؛ فایل‌های دریافت‌شده حفظ شده‌اند. "
            "برای تلاش دوباره، دکمه «ادامه از محل قطع» را فشار دهید."
        )
        with st.expander('جزئیات آخرین توقف و ماه‌های ناموفق'):
            if st.session_state.get('last_run_error'):
                st.text(st.session_state.last_run_error)
            for parameter, statuses in st.session_state.month_statuses.items():
                for month_name, entry in sorted(statuses.items()):
                    if entry.get('status') == STATUS_FAILED:
                        st.text(f"{parameter} / {month_name}: {entry.get('message', '')}")

    # --- Run status + clear-results action, directly under the run buttons ---
    _render_status_strip()

    # Simple, user-friendly download summary (persists after run)
    if st.session_state.download_summary:
        st.divider()
        turb_d, turb_a = st.session_state.download_summary.get(PARAM_TURBIDITY, (0, 0))
        chl_d, chl_a = st.session_state.download_summary.get(PARAM_CHLOROPHYLL, (0, 0))
        cdom_d, cdom_a = st.session_state.download_summary.get(PARAM_CDOM, (0, 0))
        st.info(
            f"🌊 شاخص کدورت: {turb_d} تصویر از {turb_a} تصویر موجود دریافت شد.\n\n"
            f"🌿 شاخص کلروفیل: {chl_d} تصویر از {chl_a} تصویر موجود دریافت شد.\n\n"
            f"🍂 شاخص مواد آلی محلول: {cdom_d} تصویر از {cdom_a} تصویر موجود دریافت شد."
        )

    # ==========================================================================
    # Results are ready — point the user at the three result pages above
    # ==========================================================================
    if st.session_state.processing_complete and _has_any_results():
        st.divider()
        _render_step_header(4, "📊", "نتایج پایش")
        ready_message = (
            'نتایج ماه‌های کامل در دسترس است؛ برای تکمیل بازه، پایش را ادامه دهید.'
            if st.session_state.resume_after_interruption
            else '✅ پایش با موفقیت انجام شد.'
        )

        st.markdown(
            f"""
            <div class="wq-ready-banner">
                {ready_message}<br>
                برای مشاهده نتایج، از نوار بالای صفحه یکی از صفحه‌های
                <b>🌊 کدورت</b>، <b>🌿 کلروفیل</b>، <b>🍂 مواد آلی</b>،
                <b>🧭 خلاصه</b> یا <b>🤖 چت بات</b> را انتخاب کنید.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Download combined time-series (all three indices) as one .xlsx ---
        st.download_button(
            label="⬇️ دانلود سری زمانی کدورت آب، کلروفیل و مواد آلی محلول — یک فایل اکسل",
            data=generate_combined_timeseries_excel(),
            file_name="water_quality_timeseries.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


# =============================================================================
# Main Application — header, top navigation, then the active page
# =============================================================================
def main():
    _recover_interrupted_run()
    _inject_global_app_css()
    _render_app_header()

    # Render controls and cached results even while Earth Engine is unavailable.
    # Initialization is attempted only when the user starts/resumes a run.
    # While a run is in progress the app always stays on the first page, so the
    # progress bar and the resume logic remain visible to the user.
    if st.session_state.processing_in_progress:
        st.session_state.active_page = 'setup'

    # If results were cleared while a result page was open, fall back to page 1.
    if st.session_state.active_page != 'setup' and not _has_any_results():
        st.session_state.active_page = 'setup'

    _render_top_nav()

    page = st.session_state.active_page

    if page == 'turbidity':
        render_parameter_page(PARAM_TURBIDITY)
    elif page == 'chlorophyll':
        render_parameter_page(PARAM_CHLOROPHYLL)
    elif page == 'cdom':
        render_parameter_page(PARAM_CDOM)
    elif page == 'summary':
        render_summary_page()
    elif page == 'chat':
        render_expert_chat_tab()
    else:
        render_setup_page()


if __name__ == "__main__":
    main()
