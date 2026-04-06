#!/usr/bin/env python3
"""
Image Gallery Viewer with Descriptions
A Streamlit app to view images and their corresponding descriptions from a Parquet database.

Usage:
    streamlit run viewer.py
    streamlit run viewer.py -- --database /path/to/database.parquet
    streamlit run viewer.py -- --db ./vision_ai.parquet
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import pyperclip
import io
import pandas as pd
import sys
import argparse
from datetime import datetime
from math import gcd

# Page configuration
st.set_page_config(
    page_title="Image Gallery Viewer",
    page_icon="🖼️",
    layout="wide"
)

# ── Nord Dark Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font import ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global base ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* ── Main content area background ── */
.stApp {
    background-color: #2E3440;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #3B4252;
    border-right: 1px solid #4C566A;
}
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #88C0D0;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-top: 1.2rem;
    margin-bottom: 0.3rem;
}
[data-testid="stSidebar"] .stCaption {
    color: #8FBCBB;
    font-size: 0.78rem;
}

/* ── Sidebar section dividers ── */
[data-testid="stSidebar"] hr {
    border-color: #4C566A;
    margin: 0.6rem 0;
}

/* ── Prompt / Description box ── */
.description-box {
    background-color: #3B4252;
    color: #E5E9F0;
    padding: 18px 20px;
    border-radius: 10px;
    border: 1px solid #4C566A;
    max-height: 420px;
    overflow-y: auto;
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.92rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-wrap: break-word;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
    scrollbar-width: thin;
    scrollbar-color: #4C566A #2E3440;
}
.description-box::-webkit-scrollbar { width: 6px; }
.description-box::-webkit-scrollbar-track { background: #2E3440; }
.description-box::-webkit-scrollbar-thumb { background-color: #4C566A; border-radius: 3px; }

/* ── Filename bar ── */
.filename-box {
    background-color: #434C5E;
    color: #ECEFF4;
    padding: 8px 14px;
    border-radius: 8px 8px 0 0;
    font-size: 0.88rem;
    font-weight: 600;
    border: 1px solid #4C566A;
    border-bottom: none;
    margin-bottom: 0;
}

/* ── Timestamp bar ── */
.timestamp-box {
    background-color: #3B4252;
    color: #8FBCBB;
    padding: 6px 12px;
    border-left: 3px solid #88C0D0;
    border: 1px solid #4C566A;
    border-top: none;
    margin-bottom: 0px;               /* zero gap if followed by meta */
    font-size: 0.78rem;
    letter-spacing: 0.02em;
}
/* When timestamp is the last in the stack */
.timestamp-box.last {
    border-radius: 0 0 8px 8px;
    margin-bottom: 14px;
}

/* ── Image metadata bar (size / aspect ratio / megapixels) ── */
.image-meta-box {
    background-color: #3B4252;
    color: #D8DEE9;
    padding: 5px 12px;
    border-left: 3px solid #81A1C1;   /* slightly different accent vs timestamp's #88C0D0 */
    border: 1px solid #4C566A;
    border-top: none;
    margin-bottom: 0px;
    font-size: 0.78rem;
    letter-spacing: 0.02em;
}
/* Last meta row before description needs bottom rounding + margin */
.image-meta-box.last {
    border-radius: 0 0 8px 8px;
    margin-bottom: 14px;
}

/* ── Image shadow ── */
[data-testid="stImage"] img {
    border-radius: 10px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45);
    border: 1px solid #4C566A;
    transition: box-shadow 0.2s ease;
}
[data-testid="stImage"] img:hover {
    box-shadow: 0 6px 24px rgba(136, 192, 208, 0.25);
}

/* ── Captions under images ── */
.stCaption {
    color: #81A1C1;
    font-size: 0.78rem;
}

/* ── Section headings in main area ── */
h1 {
    color: #ECEFF4;
    font-weight: 700;
}
h2, h3 {
    color: #88C0D0;
}

/* ── st.metric cards ── */
[data-testid="stMetric"] {
    background-color: #3B4252;
    border: 1px solid #4C566A;
    border-radius: 8px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] {
    color: #81A1C1 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #ECEFF4 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* ── Divider lines ── */
hr {
    border-color: #4C566A !important;
}

/* ── st.info / st.warning / st.success / st.error ── */
[data-testid="stAlert"] {
    border-radius: 8px;
    font-size: 0.88rem;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 7px;
    font-size: 0.84rem;
    font-weight: 500;
    transition: background-color 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    box-shadow: 0 0 0 2px #88C0D0;
}
/* Primary button specifically */
.stButton > button[kind="primary"] {
    background-color: #81A1C1;
    color: #2E3440;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
    background-color: #88C0D0;
}

/* ── Download button ── */
.stDownloadButton > button {
    border-radius: 7px;
    font-size: 0.84rem;
}

/* ── Text inputs & text areas ── */
.stTextInput input, .stTextArea textarea {
    background-color: #3B4252 !important;
    color: #E5E9F0 !important;
    border-color: #4C566A !important;
    border-radius: 7px !important;
    font-size: 0.88rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #88C0D0 !important;
    box-shadow: 0 0 0 2px rgba(136,192,208,0.25) !important;
}

/* ── Select boxes & multiselect ── */
.stSelectbox > div, .stMultiSelect > div {
    border-radius: 7px !important;
}

/* ── Slider ── */
.stSlider [data-testid="stSlider"] > div {
    border-radius: 999px;
}

/* ── Radio buttons ── */
.stRadio label {
    color: #D8DEE9;
    font-size: 0.88rem;
}

/* ── Footer override ── */
.footer-bar {
    text-align: center;
    color: #4C566A;
    font-size: 0.78rem;
    padding-top: 8px;
    letter-spacing: 0.03em;
}

/* ── Hide pagination anchor ── */
[id^="top-anchor-"] {
    display: block;
    height: 0;
    margin: 0;
    padding: 0;
    overflow: hidden;
}
/* ── Entry card wrapper (adds subtle north-card feel) ── */
.entry-card {
    background-color: #3B4252;
    border: 1px solid #434C5E;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    transition: box-shadow 0.2s ease;
}
.entry-card:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
}
</style>
""", unsafe_allow_html=True)


def parse_cli_args():
    """
    Parse command line arguments.
    Streamlit passes arguments after '--' to the script.
    
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Image Gallery Viewer with Parquet database support"
    )
    parser.add_argument(
        '--database',
        '--db',
        dest='database',
        type=str,
        default=None,
        help='Path to Parquet database file'
    )
    
    args_to_parse = []
    
    if 'streamlit' in sys.argv[0].lower() or any('streamlit' in arg.lower() for arg in sys.argv):
        script_found = False
        for arg in sys.argv:
            if script_found:
                args_to_parse.append(arg)
            elif arg.endswith('.py'):
                script_found = True
    else:
        args_to_parse = sys.argv[1:]
    
    if not args_to_parse:
        return parser.parse_args([])
    
    return parser.parse_args(args_to_parse)


def load_parquet_db(parquet_path: Path) -> pd.DataFrame:
    """Load Parquet database."""
    try:
        df = pd.read_parquet(parquet_path)
        
        # Ensure datetime columns are properly typed
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        if 'modified_at' in df.columns:
            df['modified_at'] = pd.to_datetime(df['modified_at'])
        
        return df
    except Exception as e:
        st.error(f"Error loading Parquet database: {str(e)}")
        return pd.DataFrame(columns=['image_path', 'prompt', 'description', 'created_at', 'modified_at'])


def save_parquet_db(df: pd.DataFrame, parquet_path: Path) -> bool:
    """Save DataFrame to Parquet file."""
    try:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        return True
    except Exception as e:
        st.error(f"Error saving Parquet database: {str(e)}")
        return False


def create_thumbnail(image_path: Path, max_size: int = 300):
    """Create a thumbnail of the image."""
    try:
        image = Image.open(image_path)
        thumbnail = image.copy()
        thumbnail.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return thumbnail
    except Exception as e:
        st.error(f"Error creating thumbnail: {str(e)}")
        return None


def format_datetime(dt) -> str:
    """Format datetime for display."""
    if pd.isna(dt):
        return "N/A"
    
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "N/A"


def compute_aspect_ratio(width: int, height: int) -> str:
    """Return a simplified aspect ratio string, snapping to standard ratios if close."""
    if width <= 0 or height <= 0:
        return "N/A"
    
    actual_ratio = width / height
    
    # Define standard ratios (name, value)
    standard_ratios = [
        ("1:1", 1.0),
        ("4:3", 4/3), ("3:4", 3/4),
        ("3:2", 3/2), ("2:3", 2/3),
        ("16:9", 16/9), ("9:16", 9/16),
        ("21:9", 21/9), ("9:21", 9/21),
        ("5:4", 5/4), ("4:5", 4/5),
        ("9:7", 9/7), ("7:9", 7/9),
        ("16:10", 16/10), ("10:16", 10/16)
    ]
    
    # Find the closest standard ratio
    best_match = None
    min_diff = float('inf')
    
    for name, ratio in standard_ratios:
        diff = abs(actual_ratio - ratio)
        if diff < min_diff:
            min_diff = diff
            best_match = name
            
    # If the match is very close (within 2% error), use the standard name
    if min_diff < 0.02:
        return best_match
        
    # Fallback to simplified exact ratio or decimal
    divisor = gcd(width, height)
    r_w = width // divisor
    r_h = height // divisor
    
    if max(r_w, r_h) > 99:
        return f"{actual_ratio:.2f}:1"
    return f"{r_w}:{r_h}"


def display_image_with_description(row: pd.Series, index: int, thumbnail_size: int = 300, df_key: str = "main_df"):
    """Display an image with its description."""
    image_path = Path(row['image_path'])
    description = row['description']
    created_at = row.get('created_at', None)
    modified_at = row.get('modified_at', None)
    
    edit_key = f"edit_mode_{index}"
    if edit_key not in st.session_state:
        st.session_state[edit_key] = False
    
    # ── Retrieve dimensions (hoisted for use in both columns) ──
    width, height = None, None
    if image_path.exists():
        try:
            with Image.open(image_path) as _img:
                width, height = _img.size
        except Exception:
            pass

    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            if not image_path.exists():
                st.error(f"❌ Image not found: {image_path.name}")
                st.caption(f"📁 {image_path}")
            else:
                # Use pre-computed width/height
                file_size = image_path.stat().st_size / 1024
                
                thumbnail = create_thumbnail(image_path, thumbnail_size)
                if thumbnail:
                    st.image(thumbnail, caption=None, width=thumbnail_size)
                    st.caption(f"📁 {image_path.name}")
                    if width is not None and height is not None:
                        st.caption(f"📐 {width}×{height} | {file_size:.1f} KB")
                    else:
                        st.caption(f"⚖️ {file_size:.1f} KB")
                else:
                    st.error("Could not load image")
                
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            st.caption(f"📁 {image_path}")
    
    with col2:
        # 1. Filename box (Top)
        st.markdown(
            f'<div class="filename-box">🖼️ {image_path.name}</div>',
            unsafe_allow_html=True
        )
        
        has_meta = width is not None and height is not None
        
        # 2. Timestamps bar (if available)
        if created_at is not None or modified_at is not None:
            timestamp_parts = []
            if created_at is not None and not pd.isna(created_at):
                timestamp_parts.append(f"📅 Created: {format_datetime(created_at)}")
            if modified_at is not None and not pd.isna(modified_at):
                timestamp_parts.append(f"✏️ Modified: {format_datetime(modified_at)}")
            
            if timestamp_parts:
                timestamp_text = " &nbsp;·&nbsp; ".join(timestamp_parts)
                ts_class = "timestamp-box" if has_meta else "timestamp-box last"
                st.markdown(
                    f'<div class="{ts_class}">{timestamp_text}</div>',
                    unsafe_allow_html=True
                )
        
        # 3. Image size, megapixels & aspect ratio bar (if image was readable)
        if has_meta:
            aspect = compute_aspect_ratio(width, height)
            mp = (width * height) / 1_000_000
            meta_text = (
                f"📐 {width}×{height} ({mp:.2f} MP) &nbsp;·&nbsp; "
                f"⬛ Ratio: {aspect}"
            )
            st.markdown(
                f'<div class="image-meta-box last">{meta_text}</div>',
                unsafe_allow_html=True
            )
        
        st.markdown(
            "<p style='font-size:0.72rem; color:#81A1C1; text-transform:uppercase;"
            " letter-spacing:0.09em; font-weight:600; margin-bottom:4px;'>Prompt</p>",
            unsafe_allow_html=True
        )
        
        if st.session_state[edit_key]:
            edited_description = st.text_area(
                "Edit prompt:",
                value=description,
                height=300,
                key=f"edit_textarea_{index}",
                label_visibility="collapsed"
            )
            
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
            
            with btn_col1:
                if st.button("💾 Save", key=f"save_{index}", use_container_width=True, type="primary"):
                    df = st.session_state[df_key]
                    mask = df['image_path'] == str(image_path)
                    df.loc[mask, 'description'] = edited_description
                    
                    # Update modified_at timestamp
                    if 'modified_at' in df.columns:
                        df.loc[mask, 'modified_at'] = pd.Timestamp.now()
                    
                    st.session_state[df_key] = df
                    
                    if save_parquet_db(df, st.session_state.parquet_path):
                        st.session_state[edit_key] = False
                        st.success("✓ Prompt saved!", icon="✅")
                        st.rerun()
                    else:
                        st.error("Failed to save changes")
            
            with btn_col2:
                if st.button("❌ Cancel", key=f"cancel_{index}", use_container_width=True):
                    st.session_state[edit_key] = False
                    st.rerun()
            
            st.caption(f"📝 {len(edited_description)} characters")
        
        else:
            st.markdown(
                f'<div class="description-box">{description}</div>',
                unsafe_allow_html=True
            )
            
            st.markdown("")
            
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1, 1])
            
            with btn_col1:
                if st.button(f"✏️ Edit", key=f"edit_{index}", use_container_width=True):
                    st.session_state[edit_key] = True
                    st.rerun()
            
            with btn_col2:
                if st.button(f"📋 Copy", key=f"copy_desc_{index}", use_container_width=True):
                    try:
                        pyperclip.copy(description)
                        st.toast("✓ Copied!", icon="✅")
                    except:
                        st.session_state[f'show_copy_{index}'] = True
            
            with btn_col3:
                if st.button(f"📁 Path", key=f"copy_path_{index}", use_container_width=True):
                    try:
                        pyperclip.copy(str(image_path))
                        st.toast("✓ Path copied!", icon="✅")
                    except:
                        st.session_state[f'show_copy_path_{index}'] = True
            
            with btn_col4:
                st.download_button(
                    label="💾 Text",
                    data=description,
                    file_name=f"{image_path.stem}.txt",
                    mime="text/plain",
                    key=f"download_{index}",
                    use_container_width=True
                )
            
            if st.session_state.get(f'show_copy_{index}', False):
                st.text_area("Select and copy:", value=description, height=100, key=f"manual_copy_{index}")
            
            if st.session_state.get(f'show_copy_path_{index}', False):
                st.text_area("Select and copy path:", value=str(image_path), height=50, key=f"manual_copy_path_{index}")

            
            st.caption(f"📝 {len(description)} characters | Full path: {image_path}")
    
    st.divider()


def render_pagination(current_page: int, total_pages: int):
    """Render pagination controls with slider."""
    if total_pages <= 1:
        return current_page
    
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    
    # Navigation buttons row
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("⏮️ First", use_container_width=True, disabled=(current_page == 1)):
            st.session_state.current_page = 1
            st.rerun()
    
    with col2:
        if st.button("◀️ Previous", use_container_width=True, disabled=(current_page == 1)):
            st.session_state.current_page = current_page - 1
            st.rerun()
    
    with col3:
        if st.button("Next ▶️", use_container_width=True, disabled=(current_page == total_pages)):
            st.session_state.current_page = current_page + 1
            st.rerun()
    
    with col4:
        if st.button("Last ⏭️", use_container_width=True, disabled=(current_page == total_pages)):
            st.session_state.current_page = total_pages
            st.rerun()
    
    # Slider for page navigation
    st.markdown("")  # Small spacing
    new_page = st.slider(
        f"Page {current_page} of {total_pages}",
        min_value=1,
        max_value=total_pages,
        value=current_page,
        step=1
    )
    
    if new_page != current_page:
        st.session_state.current_page = new_page
        st.rerun()
    
    return st.session_state.get('current_page', current_page)


def apply_search_filter(df: pd.DataFrame, search_query: str, search_in: str) -> pd.DataFrame:
    """Apply search filter to the dataframe."""
    if not search_query:
        return df
    
    search_lower = search_query.lower()
    # Normalize slashes for search query if it looks like a path
    search_norm = search_lower.replace('\\', '/')
    
    if search_in == "Filename OR Prompt":
        df_copy = df.copy()
        df_copy['filename'] = df_copy['image_path'].apply(lambda x: Path(x).name.lower())
        mask = (
            df_copy['filename'].str.contains(search_lower, na=False) |
            df_copy['description'].fillna('').str.lower().str.contains(search_lower, na=False)
        )
        return df[mask]
    
    elif search_in == "Prompt":
        mask = df['description'].fillna('').str.lower().str.contains(search_lower, na=False)
        return df[mask]
    
    elif search_in == "Filename":
        df_copy = df.copy()
        df_copy['filename'] = df_copy['image_path'].apply(lambda x: Path(x).name.lower())
        mask = df_copy['filename'].str.contains(search_lower, na=False)
        return df[mask]
    
    elif search_in == "Full Path":
        # Normalize paths to use forward slashes for consistent searching
        path_series = df['image_path'].str.lower().str.replace('\\', '/', regex=False)
        mask = path_series.str.contains(search_norm, na=False)
        return df[mask]
    
    else:  # All
        df_copy = df.copy()
        df_copy['filename'] = df_copy['image_path'].apply(lambda x: Path(x).name.lower())
        path_series = df_copy['image_path'].str.lower().str.replace('\\', '/', regex=False)
        mask = (
            df_copy['description'].fillna('').str.lower().str.contains(search_lower, na=False) |
            path_series.str.contains(search_norm, na=False)
        )
        return df[mask]


def apply_sorting(df: pd.DataFrame, sort_option: str) -> pd.DataFrame:
    """Apply sorting to the dataframe."""
    if sort_option == "Image Name (A-Z)":
        return df.sort_values('image_path')
    elif sort_option == "Image Name (Z-A)":
        return df.sort_values('image_path', ascending=False)
    elif sort_option == "Prompt (A-Z)":
        return df.sort_values('description')
    elif sort_option == "Prompt (Z-A)":
        return df.sort_values('description', ascending=False)
    elif sort_option == "Created Date (Newest First)":
        if 'created_at' in df.columns:
            # Put NaT values at the end
            return df.sort_values('created_at', ascending=False, na_position='last')
        else:
            st.warning("⚠️ 'created_at' column not found in database")
            return df
    elif sort_option == "Created Date (Oldest First)":
        if 'created_at' in df.columns:
            # Put NaT values at the end
            return df.sort_values('created_at', ascending=True, na_position='last')
        else:
            st.warning("⚠️ 'created_at' column not found in database")
            return df
    elif sort_option == "Modified Date (Newest First)":
        if 'modified_at' in df.columns:
            # Put NaT values at the end
            return df.sort_values('modified_at', ascending=False, na_position='last')
        else:
            st.warning("⚠️ 'modified_at' column not found in database")
            return df
    elif sort_option == "Modified Date (Oldest First)":
        if 'modified_at' in df.columns:
            # Put NaT values at the end
            return df.sort_values('modified_at', ascending=True, na_position='last')
        else:
            st.warning("⚠️ 'modified_at' column not found in database")
            return df
    else:
        return df


def main():
    """Main function for the Streamlit app."""
    
    cli_args = parse_cli_args()
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    if 'cli_database_path' not in st.session_state:
        st.session_state.cli_database_path = cli_args.database
    
    st.title("Image Gallery Viewer")
    st.markdown(
        "<p style='color:#81A1C1; font-size:0.97rem; margin-top:-10px;'>"
        "Browse · Search · Edit · Export &nbsp;— AI prompt database viewer"
        "</p>",
        unsafe_allow_html=True
    )
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if st.session_state.cli_database_path:
            default_db_path = st.session_state.cli_database_path
            st.info(f"📌 Database set via CLI: {Path(default_db_path).name}")
        else:
            default_db_path = "./vision_ai.parquet"
        
        db_path = st.text_input(
            "Parquet Database Path",
            value=default_db_path,
            help="Enter the path to the Parquet database file"
        )
        
        st.caption("💡 Tip: Enter the full or relative path to your Parquet database")
        if st.session_state.cli_database_path:
            st.caption("🔧 Database path was provided via CLI argument")
        
        if db_path:
            parquet_path = Path(db_path)
            
            if not parquet_path.exists():
                st.error("❌ Database file does not exist")
                return
            
            if not parquet_path.is_file():
                st.error("❌ Path is not a file")
                return
            
            if parquet_path.suffix != '.parquet':
                st.warning("⚠️ File does not have .parquet extension")
            
            st.success("✓ Valid database file")
            
            st.session_state.parquet_path = parquet_path
            
            with st.spinner("Loading database..."):
                df = load_parquet_db(parquet_path)
            
            if df.empty:
                st.warning("⚠️ Database is empty or could not be loaded")
                return
            
            required_columns = {'image_path', 'prompt', 'description'}
            if not required_columns.issubset(df.columns):
                st.error(f"❌ Database missing required columns. Found: {list(df.columns)}")
                return
            
            if 'main_df' not in st.session_state or st.session_state.get('last_db_path') != str(parquet_path):
                st.session_state.main_df = df.copy()
                st.session_state.last_db_path = str(parquet_path)
            else:
                df = st.session_state.main_df.copy()
            
            df['exists'] = df['image_path'].apply(lambda x: Path(x).exists())
            images_exist = df['exists'].sum()
            images_missing = len(df) - images_exist
            
            # Check for datetime columns
            has_created_at = 'created_at' in df.columns
            has_modified_at = 'modified_at' in df.columns
            
            st.markdown("---")
            st.markdown("### Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Entries", len(df))
                st.metric("Images Found", images_exist)
            with col2:
                st.metric("Images Missing", images_missing)
                st.metric("Unique Prompts", df['prompt'].nunique())
            
            # Show datetime statistics if available
            if has_created_at:
                valid_created = df['created_at'].notna().sum()
                if valid_created > 0:
                    oldest = df['created_at'].min()
                    newest = df['created_at'].max()
                    st.caption(f"📅 Date range: {format_datetime(oldest)} to {format_datetime(newest)}")
            
            file_size = parquet_path.stat().st_size / 1024
            if file_size > 1024:
                file_size_str = f"{file_size/1024:.2f} MB"
            else:
                file_size_str = f"{file_size:.2f} KB"
            
            st.caption(f"💾 Database size: {file_size_str}")
            
            st.markdown("---")
            st.info("✏️ Click 'Edit' on any description to modify it. Changes are saved to the database.")
            
            st.markdown("---")
            st.markdown("### Display")
            thumbnail_size = st.slider(
                "Thumbnail Size",
                min_value=150,
                max_value=500,
                value=300,
                step=50,
                help="Adjust the maximum size of image thumbnails"
            )
            
            items_per_page = st.selectbox(
                "Items per page",
                [5, 10, 20, 50, 100],
                index=1,
                key="items_per_page"
            )
            
            st.markdown("---")
            st.markdown("### Filters")

            # Extract subdirectories from image_path for a dropdown
            # We want to find common date-like or folder-like components
            def get_all_subdirs(df):
                subdirs = set()
                for p in df['image_path'].dropna():
                    parts = Path(p).parts
                    # Add all parts except the filename and the root
                    for part in parts[:-1]:
                        if part and part != '/' and part != '\\':
                            subdirs.add(part)
                return sorted(list(subdirs), reverse=True) # Usually want newest dates first

            available_subdirs = get_all_subdirs(df)
            
            selected_subdirs = st.multiselect(
                "Limit to Subdirectory (Select)",
                options=available_subdirs,
                default=[],
                help="Select one or more subdirectories to filter by."
            )

            subdir_query = st.text_input(
                "Limit to Subdirectory (Manual Search)",
                value="",
                placeholder="e.g. 2026-03-12",
                help="Only show images whose path contains this subdirectory name."
            )
            
            existence_filter = st.radio(
                "Show",
                ["All Entries", "Images Found Only", "Images Missing Only"],
                index=0
            )
            
            if existence_filter == "Images Found Only":
                filtered_df = df[df['exists']].copy()
            elif existence_filter == "Images Missing Only":
                filtered_df = df[~df['exists']].copy()
            else:
                filtered_df = df.copy()

            # Apply subdirectory filters
            if selected_subdirs or subdir_query:
                path_series = filtered_df['image_path'].str.lower().str.replace('\\', '/', regex=False)
                
                mask = pd.Series(True, index=filtered_df.index)
                
                if selected_subdirs:
                    # Match any of the selected subdirs
                    subdir_mask = pd.Series(False, index=filtered_df.index)
                    for sd in selected_subdirs:
                        subdir_mask |= path_series.str.contains(f"/{sd.lower()}/", na=False) | \
                                      path_series.str.endswith(f"/{sd.lower()}", na=False) | \
                                      path_series.str.startswith(f"{sd.lower()}/", na=False) | \
                                      (path_series == sd.lower())
                    mask &= subdir_mask
                
                if subdir_query:
                    subdir_norm = subdir_query.lower().replace('\\', '/')
                    mask &= path_series.str.contains(subdir_norm, na=False)
                
                filtered_df = filtered_df[mask]
            
            unique_prompts = sorted(df['prompt'].unique())
            if len(unique_prompts) > 1:
                selected_prompts = st.multiselect(
                    "Filter by Prompt",
                    options=["All"] + unique_prompts,
                    default=["All"]
                )
                
                if "All" not in selected_prompts:
                    filtered_df = filtered_df[filtered_df['prompt'].isin(selected_prompts)]
            
            st.markdown("---")
            st.markdown("### Sorting")
            
            # Build sort options dynamically based on available columns
            sort_options = []

            # Add created date options first if available
            if has_created_at:
                sort_options.extend([
                    "Created Date (Newest First)",
                    "Created Date (Oldest First)"
                ])

            # Then add other options
            sort_options.extend([
                "Image Name (A-Z)",
                "Image Name (Z-A)",
                "Prompt (A-Z)",
                "Prompt (Z-A)"
            ])

            if has_modified_at:
                sort_options.extend([
                    "Modified Date (Newest First)",
                    "Modified Date (Oldest First)"
                ])

            sort_option = st.selectbox(
                "Sort by",
                sort_options,
                index=0  # Will default to "Created Date (Newest First)" if has_created_at is True
            )

            
            filtered_df = apply_sorting(filtered_df, sort_option)
            
            st.markdown("---")
            st.markdown("### Search")
            
            search_in = st.selectbox(
                "Search in",
                ["Filename OR Prompt", "Prompt", "Filename", "Full Path", "All"],
                index=0,
                help="Choose where to search."
            )
            
            search_query = st.text_input("Search", "", placeholder="Enter search term...")
            
            if search_in == "Filename OR Prompt":
                st.caption("🔍 Will match if found in filename OR prompt")
            elif search_in == "All":
                st.caption("🔍 Will search across all fields")
            
            filtered_df = apply_search_filter(filtered_df, search_query, search_in)
            
            if search_query:
                st.info(f"Found {len(filtered_df)} matching result(s)")
            
            # Reset page only when filters actually change
            current_filter = (existence_filter, sort_option, search_query, search_in, subdir_query, tuple(selected_subdirs))
            
            if 'last_filter' not in st.session_state:
                st.session_state.last_filter = current_filter
            elif st.session_state.last_filter != current_filter:
                st.session_state.current_page = 1
                st.session_state.last_filter = current_filter
            else:
                st.session_state.last_filter = current_filter
        else:
            st.info("👈 Enter a database path to get started")
            return
    
    if filtered_df.empty:
        st.info("No entries match the current filter criteria")
        return
    
    total_pages = (len(filtered_df) - 1) // items_per_page + 1
    
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages
    if st.session_state.current_page < 1:
        st.session_state.current_page = 1
    
    current_page = st.session_state.current_page
    
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_df))
    
    # ZERO-JAVASCRIPT SCROLL-TO-TOP HACK
    # By rendering a new, invisible text input with 'autofocus' whenever the page number changes,
    # the browser natively scrolls the window to make this input visible, defeating Streamlit's 
    # scroll-restoration behavior without using any unreliable JavaScript.
    st.markdown(
        f'<input type="text" autofocus style="width: 1px; height: 1px; border: none; padding: 0; margin: 0; opacity: 0;" id="focus_pg_{current_page}">',
        unsafe_allow_html=True
    )
    
    st.caption(f"Showing items {start_idx + 1}-{end_idx} of {len(filtered_df)}")
    
    page_df = filtered_df.iloc[start_idx:end_idx]
    for idx, row in page_df.iterrows():
        display_image_with_description(row, idx, thumbnail_size)
    
    # Pagination at the bottom
    render_pagination(current_page, total_pages)
    
    st.markdown("---")
    st.markdown(
        '<div class="footer-bar">'
        'Image Gallery Viewer &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Parquet'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
