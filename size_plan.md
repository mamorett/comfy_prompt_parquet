# Plan: Image Size & Aspect Ratio Display

## Goal

Add two new pieces of information per image — **pixel dimensions** (e.g. `1280×720`) and
**aspect ratio** (e.g. `16:9`) — rendered immediately after the **Created / Modified** timestamp
bar and visually consistent with the existing Nord Dark card style.

---

## Current UI Structure (per card, right column `col2`)

```
┌─────────────────────────────────────────────┐  ← .filename-box
│ 🖼️  filename.png                            │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐  ← .timestamp-box
│ 📅 Created: 2026-03-12 14:05:00  ·  ✏️ Mod │
└─────────────────────────────────────────────┘
   ← NEW ROWS GO HERE
┌─────────────────────────────────────────────┐  ← .description-box
│  <prompt text>                              │
└─────────────────────────────────────────────┘
```

---

## New UI Rows (after the timestamp bar)

Two new rows will be inserted between `.timestamp-box` and the `Prompt` label:

```
┌─────────────────────────────────────────────┐  ← .image-meta-box  (size row)
│ 📐 Size: 1280×720                           │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐  ← .image-meta-box  (ratio row)
│ ⬛ Aspect Ratio: 16:9                       │
└─────────────────────────────────────────────┘
```

Both rows reuse the Nord Dark palette and border styling already established by `.timestamp-box`,
keeping the visual language consistent.

---

## CSS Addition — `.image-meta-box`

Add a new class in the existing `<style>` block (right after the `.timestamp-box` rule, ~line 118):

```css
/* ── Image metadata bar (size / aspect ratio) ── */
.image-meta-box {
    background-color: #3B4252;
    color: #D8DEE9;
    padding: 5px 12px;
    border-left: 3px solid #81A1C1;   /* slightly different accent vs timestamp's #88C0D0 */
    border: 1px solid #4C566A;
    border-top: none;
    margin-bottom: 0px;               /* zero gap so rows stack flush */
    font-size: 0.78rem;
    letter-spacing: 0.02em;
}
/* Last meta row before description needs bottom rounding + margin */
.image-meta-box.last {
    border-radius: 0 0 8px 8px;
    margin-bottom: 14px;
}
```

> **Note on border continuity:** `.filename-box` has `border-radius: 8px 8px 0 0`.
> `.timestamp-box` has `border-radius: 0 0 8px 8px` **only when it is the last bar**.
> Once these new meta rows are added, `.timestamp-box` should lose its bottom rounding
> (remove `border-radius: 0 0 8px 8px` from it), and the last `.image-meta-box` gets
> the bottom rounding via the `.last` modifier.

---

## Helper Function — `compute_aspect_ratio`

Add a utility function near the top of the file (after `format_datetime`, ~line 356):

```python
from math import gcd

def compute_aspect_ratio(width: int, height: int) -> str:
    """Return a simplified aspect ratio string like '16:9' or '4:3'."""
    if width <= 0 or height <= 0:
        return "N/A"
    divisor = gcd(width, height)
    r_w = width // divisor
    r_h = height // divisor
    # Cap absurdly large ratios (e.g. 1:1 crops stored at odd pixel counts)
    # by capping the simplified ratio to a max of 99 on either side.
    if max(r_w, r_h) > 99:
        # Fall back to rounding to 2 decimal places
        return f"{width/height:.2f}:1"
    return f"{r_w}:{r_h}"
```

---

## Changes to `display_image_with_description`

### 1. Retrieve dimensions (already available)

The existing code already computes `width` and `height` in `col1` (lines 378–379).
These values need to be **hoisted above the `col1/col2` split** so they are accessible
in `col2` as well.

```python
# Before col1, col2 = st.columns([1, 2])
width, height = None, None
if image_path.exists():
    try:
        with Image.open(image_path) as _img:
            width, height = _img.size
    except Exception:
        pass
```

The `col1` block then uses the already-computed `width`/`height` (no re-opening).

### 2. Render the new rows in `col2`

Immediately after the timestamp bar block (after the closing `</div>` of `.timestamp-box`,
~line 413), insert:

```python
# 3. Image size & aspect ratio bars (if image was readable)
if width is not None and height is not None:
    aspect = compute_aspect_ratio(width, height)
    st.markdown(
        f'<div class="image-meta-box">📐 Size: {width}×{height} px</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="image-meta-box last">⬛ Aspect Ratio: {aspect}</div>',
        unsafe_allow_html=True
    )
```

> When the image file is **missing** (already handled by the existing `st.error` path),
> `width` and `height` remain `None`, so the rows are simply skipped — no broken output.

---

## CSS Adjustment to `.timestamp-box`

Remove `border-radius: 0 0 8px 8px` from `.timestamp-box` (it will always be followed by
the new meta rows when an image exists):

```css
/* Before */
.timestamp-box {
    ...
    border-radius: 0 0 8px 8px;
    ...
}

/* After */
.timestamp-box {
    /* border-radius removed – bottom rounding now on .image-meta-box.last */
    ...
}
```

If no timestamp data exists (the bar is skipped), the `.filename-box` already provides
top rounding; the very first `.image-meta-box` will lack a top border via `border-top: none`
and will still flow cleanly.

---

## Summary of File Edits

| Location | Change |
|---|---|
| CSS block (~line 107–118) | Remove `border-radius` from `.timestamp-box`; add `.image-meta-box` rule |
| After `format_datetime` (~line 356) | Add `compute_aspect_ratio(width, height)` function |
| `display_image_with_description`, before column split | Hoist `width, height` computation |
| `col1` block (~line 378) | Use pre-computed `width`/`height` instead of recomputing |
| `col2` block, after timestamp bar (~line 413) | Insert size and aspect ratio `<div>` rows |

---

## Edge Cases

| Scenario | Behaviour |
|---|---|
| Image file missing | `width`/`height` stay `None`; meta rows are skipped silently |
| 1×1 image (or identical W/H) | Ratio = `1:1` |
| Very unusual pixel counts (e.g. 1920×1017) | `gcd` may give oversized numbers; fallback to decimal form `1.89:1` |
| No timestamp row | Meta rows follow directly after `.filename-box` with no gap |
