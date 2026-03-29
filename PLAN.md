# UI Improvement Plan — `streamlit_viewer.py`

> **Target file:** `streamlit_viewer.py` (758 lines, single file)
> **Goal:** Apply a Nord dark colour palette, improve readability and general usability.
> **Constraint:** Do NOT change any Python logic, data loading, filtering, sorting, or pagination behaviour.
> Only the visual layer (CSS, labels, layout tweaks, markdown strings) may change.

---

## 1. Nord Colour Palette Reference

Use these exact hex values everywhere. Never use other colours unless justified below.

| Role | Token | Hex |
|---|---|---|
| Background (deepest) | `nord0` | `#2E3440` |
| Background (panels/sidebar) | `nord1` | `#3B4252` |
| Background (elevated cards) | `nord2` | `#434C5E` |
| Background (highlights/hover) | `nord3` | `#4C566A` |
| Snow (off-white text) | `nord4` | `#D8DEE9` |
| Snow (slightly brighter text) | `nord5` | `#E5E9F0` |
| Snow (pure/headings) | `nord6` | `#ECEFF4` |
| Frost (accent blue-teal) | `nord7` | `#8FBCBB` |
| Frost (primary accent) | `nord8` | `#88C0D0` |
| Frost (link / interactive) | `nord9` | `#81A1C1` |
| Frost (muted blue) | `nord10` | `#5E81AC` |
| Aurora (red / error) | `nord11` | `#BF616A` |
| Aurora (orange / warning) | `nord12` | `#D08770` |
| Aurora (yellow / caution) | `nord13` | `#EBCB8B` |
| Aurora (green / success) | `nord14` | `#A3BE8C` |
| Aurora (purple / accent2) | `nord15` | `#B48EAD` |

---

## 2. Streamlit Theme Configuration (`.streamlit/config.toml`)

Create the file `.streamlit/config.toml` next to `streamlit_viewer.py` if it does not already exist.

```toml
[theme]
base = "dark"
primaryColor       = "#88C0D0"   # nord8  — buttons, sliders, focus rings
backgroundColor    = "#2E3440"   # nord0  — main page background
secondaryBackgroundColor = "#3B4252"  # nord1  — sidebar, stExpander bg
textColor          = "#D8DEE9"   # nord4  — default body text
font               = "sans serif"
```

This gives every native Streamlit widget (buttons, sliders, selectboxes, text-inputs, metrics) the Nord palette for free without touching Python code.

---

## 3. Custom CSS Block Replacement

Replace the entire `st.markdown("""…""", unsafe_allow_html=True)` CSS block that is currently injected near line 30 with the following.
Keep `unsafe_allow_html=True`. Place it in exactly the same location in the file (after `st.set_page_config(…)`).

```python
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

/* ── Timestamp bar ── */
.timestamp-box {
    background-color: #434C5E;
    color: #8FBCBB;
    padding: 6px 12px;
    border-radius: 6px;
    border-left: 3px solid #88C0D0;
    margin-bottom: 10px;
    font-size: 0.78rem;
    letter-spacing: 0.02em;
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
```

---

## 4. Per-Entry Card Wrapper

In `display_image_with_description()` (around line 182), wrap the `with st.container():` block using an HTML card div.

### Change A — Open the card before `col1, col2`

Find this code (approximately line 182):
```python
    with st.container():
        col1, col2 = st.columns([1, 2])
```

Replace with:
```python
    st.markdown('<div class="entry-card">', unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns([1, 2])
```

### Change B — Close the card after `st.divider()`

Find this code (approximately line 313):
```python
        st.divider()
```

Replace with:
```python
        st.divider()
    st.markdown('</div>', unsafe_allow_html=True)
```

> **Note:** `st.divider()` inside the card will render a nord-coloured `<hr>` thanks to the CSS.
> The outer `</div>` closes `entry-card`.
> Remove the original bare `st.divider()` if it appears again outside the card (there should only be one).

---

## 5. Timestamp Box: Keep the HTML, update the class

No Python change needed. The new CSS for `.timestamp-box` (defined in §3) already replaces the light-grey look with a Nord-accented left-bordered strip. Verify the existing code still outputs:

```python
st.markdown(
    f'<div class="timestamp-box">{timestamp_text}</div>',
    unsafe_allow_html=True
)
```

No change required here — it's already correct.

---

## 6. Section Heading Labels (Sidebar Readability)

The sidebar uses `st.markdown("### 📊 Statistics")` etc. These are purely cosmetic strings; improve them by slightly rewriting the labels so they feel consistent and uncluttered. Apply these label replacements:

| Find (exact string) | Replace with |
|---|---|
| `"### 📊 Statistics"` | `"### Statistics"` |
| `"### 🖼️ Display"` | `"### Display"` |
| `"### 🔍 Filters"` | `"### Filters"` |
| `"### 📑 Sorting"` | `"### Sorting"` |
| `"### 🔎 Search"` | `"### Search"` |

> The emoji in headings creates visual clutter on dark backgrounds and clashes slightly with the Nord style. The CSS (§3) already styles `h3` in the sidebar with uppercase tracking and the frost blue colour — this is enough visual hierarchy.
> Keep the emoji in `st.caption()` and button labels; they work fine there.

---

## 7. Footer

Replace the current footer block (approximately lines 748–753):

```python
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Image Gallery Viewer | Built with Streamlit | Powered by Parquet"
        "</div>",
        unsafe_allow_html=True
    )
```

with:

```python
    st.markdown(
        '<div class="footer-bar">'
        'Image Gallery Viewer &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Parquet'
        '</div>',
        unsafe_allow_html=True
    )
```

> Uses the `.footer-bar` CSS class (§3) for a subtle, consistent nord-grey tone instead of an inline `gray`.

---

## 8. Page Title & Subtitle

Replace the title block (approximately line 461):

```python
    st.title("🖼️ Image Gallery Viewer")
    st.markdown("View images and their AI-generated descriptions from Parquet database")
```

with:

```python
    st.title("Image Gallery Viewer")
    st.markdown(
        "<p style='color:#81A1C1; font-size:0.97rem; margin-top:-10px;'>"
        "Browse · Search · Edit · Export &nbsp;— AI prompt database viewer"
        "</p>",
        unsafe_allow_html=True
    )
```

> The emoji in the page title `st.title()` renders inconsistently across platforms. The subtitle now doubles as a quick feature summary in the Frost accent colour.

---

## 9. Pagination: Visual Polish

In `render_pagination()`, replace the plain `st.markdown("---")` line (approximately line 321) with a more explicit spacer:

```python
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
```

Also update the slider label from the current dynamic f-string:

```python
        f"📄 Page {current_page} of {total_pages}",
```

to:

```python
        f"Page {current_page} of {total_pages}",
```

> The `📄` glyph can misalign the slider label on some OS/font combos. Plain text is cleaner here.

---

## 10. Prompt Section Heading

In `display_image_with_description()`, replace:

```python
            st.markdown("### Prompt")
```

with:

```python
            st.markdown(
                "<p style='font-size:0.72rem; color:#81A1C1; text-transform:uppercase;"
                " letter-spacing:0.09em; font-weight:600; margin-bottom:4px;'>Prompt</p>",
                unsafe_allow_html=True
            )
```

> This stops it rendering as a large `<h3>` and instead uses a small label style consistent with the Nord sidebar headings.

---

## 11. Stat Captions Inline Characters

For the stats section in the sidebar, the stat caption (approximately line 543):

```python
                    st.caption(f"📅 Date range: {format_datetime(oldest)} to {format_datetime(newest)}")
```

Keep as-is — the `📅` is fine in captions.

---

## 12. Items-per-Page & Sidebar Metrics — Compact Columns

No layout change is needed for the two-column metrics block — it's already compact. The `.stMetric` CSS applied in §3 gives it the card treatment automatically.

---

## 13. Summary of Files to Create / Modify

| Action | File | What to do |
|---|---|---|
| **CREATE** | `.streamlit/config.toml` | Full Nord theme (§2) |
| **MODIFY** | `streamlit_viewer.py` | §3 CSS block, §4 card wrappers, §6 heading labels, §7 footer, §8 title, §9 pagination, §10 prompt heading |

---

## 14. Testing Checklist

After applying all changes, verify:

- [ ] `streamlit run streamlit_viewer.py -- --db ./prompti.parquet` starts without errors
- [ ] Sidebar background is `#3B4252` (Nord1), not white/grey
- [ ] Main background is `#2E3440` (Nord0)
- [ ] Image cards show with rounded corners and subtle shadow
- [ ] Description box scrollbar is thin and Nord-coloured
- [ ] Timestamps render with left frost-blue border
- [ ] Metric tiles show with Nord card background
- [ ] Primary buttons (Save) use `#81A1C1` background with dark text
- [ ] Secondary buttons (Edit, Copy, Path) have Nord hover ring
- [ ] Footer is small, subtle, dark frost text
- [ ] Title has no emoji, subtitle is accent-coloured
- [ ] All filtering, sorting, pagination, save, and copy functions still work exactly as before
