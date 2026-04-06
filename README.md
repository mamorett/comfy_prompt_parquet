# 🖼️ Comfy Prompt Parquet

> **The ultimate toolkit for ComfyUI prompt management.**  
> Extract, browse, edit, and embed your generative AI metadata with ease.

---

## ✨ Overview

Comfy Prompt Parquet is a high-performance suite of tools designed for creators using ComfyUI. It bridges the gap between raw image files and a searchable, editable database.

- 🛠️ **Extractor**: Scans your images and builds a blazing-fast Parquet database.
- 🎨 **Viewer**: A beautiful, Nord-themed Streamlit app to browse and edit prompts.
- 📦 **Embedder**: Create portable databases with embedded, resized thumbnails.

---

## 🚀 Key Features

### 🌈 Elegant UI (Streamlit)
- **Nord Dark Theme**: A premium, easy-on-the-eyes aesthetic.
- **Unified Controls**: Consistent, glowing interactive buttons for Edit, Copy, Path, and Download.
- **Smart Metadata**: Automatic calculation of megapixels and "snapping" aspect ratios (e.g., 16:9, 3:2).
- **In-place Editing**: Modify prompts directly in the browser; changes save instantly to the database.

### 🔍 Powerful Discovery
- **Full-Text Search**: Instant filtering across filenames, prompts, and paths.
- **Advanced Filtering**: Filter by subdirectory, existence on disk, or specific prompt tags.
- **Deep Sorting**: Sort by creation date, modification date, or alphabetically.

### 📦 Portable Data
- **Image Embedding**: Convert path-based databases into standalone files.
- **Smart Resizing**: Embed images at **25% scale** (or custom) to keep database size manageable.
- **Safety First**: Embedding and resizing are performed **entirely in memory**. Your original image files are never touched.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/comfy-prompt-parquet.git
cd comfy-prompt-parquet

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 CLI Reference

### 🛠️ Extractor (`comfyprompt_extractor.py`)
Builds or updates your Parquet database from image metadata.

| Option | Shorthand | Description |
| :--- | :--- | :--- |
| `--input` | `-i` | **Required.** Directory, single file, or glob pattern (e.g., `*.png`). |
| `--database` | `--db` | **Required.** Path to the output Parquet database file. |
| `--recursive` | `-r` | Recursively search subdirectories for images. |
| `--override` | | Replace existing entries in the database for the same files. |
| `--file-list` | `-f` | Provide a text file containing a list of image paths to process. |
| `--use-parameters`| | Use A1111/parameters extraction instead of ComfyUI workflow JSON. |
| `--help` | `-h` | Show all available options and examples. |

**Example:**
```bash
python comfyprompt_extractor.py -i ./images -r --db gallery.parquet --override
```

---

### 🎨 Viewer (`streamlit_viewer.py`)
Launch the interactive web gallery.

| Option | Shorthand | Description |
| :--- | :--- | :--- |
| `--database` | `--db` | Path to the Parquet database to load. |
| `--help` | `-h` | Show all available options. |

> **Note**: To pass arguments to the script via Streamlit, use the `--` separator:  
> `streamlit run streamlit_viewer.py -- --db gallery.parquet`

---

### 📦 Embedder (`embed_images.py`)
Embeds resized images directly into the Parquet file for portability.

| Option | Shorthand | Description |
| :--- | :--- | :--- |
| `--input` | `-i` | **Required.** Path to the input Parquet file. |
| `--output` | `-o` | **Required.** Path to the output Parquet file. |
| `--scale` | `-s` | Resizing scale factor (default: `0.25` for 25% size). |
| `--dry-run` | `-d` | Verify paths and estimate final size without writing files. |
| `--col` | | Column name containing image paths (default: `image_path`). |
| `--target` | | Column name for embedded bytes (default: `image_bytes`). |

**Example:**
```bash
python embed_images.py -i gallery.parquet -o portable.parquet -s 0.2 -d
```

---

## 🐳 Docker Deployment

The viewer is ready for containerized environments.

```bash
# Build the image
docker build -t comfy-viewer .

# Run the container
docker run -p 8501:8501 \
  -v /path/to/images:/data/images \
  -v /path/to/collection.parquet:/app/collection.parquet \
  comfy-viewer -- --db collection.parquet
```

---

## 📜 License
Licensed under the [MIT License](LICENSE). 
Created with ❤️ for the AI Art community.
