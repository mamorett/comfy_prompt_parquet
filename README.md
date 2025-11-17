# Comfy Prompt Parquet

This project provides a set of tools to extract prompts from ComfyUI-generated images and view them in a user-friendly web interface. It consists of two main components:

1.  A Python script (`comfyprompt_extractor.py`) to extract positive prompts from PNG metadata and store them in a Parquet database.
2.  A Streamlit application (`streamlit_viewer.py`) to browse, search, and edit the extracted prompts and their corresponding images.

## Features

### Extractor (`comfyprompt_extractor.py`)

*   **Prompt Extraction**: Extracts positive prompts from ComfyUI PNG metadata.
*   **Parquet Database**: Saves extracted data to an efficient Parquet file.
*   **Multiple Input Sources**: Process images from a directory, a single file, a glob pattern, or a list of files.
*   **Idempotency**: Skips already processed images unless the `--override` flag is used.
*   **Graceful Shutdown**: Saves progress on `Ctrl-C`.

### Viewer (`streamlit_viewer.py`)

*   **Image Gallery**: Displays images and their prompts in a gallery format.
*   **Search and Filter**: Full-text search for prompts, descriptions, and filenames. Filter by prompt, and image status (found or missing).
*   **Sorting**: Sort images by creation date, modification date, image name, or prompt.
*   **In-place Editing**: Edit descriptions directly in the web interface and save them back to the database.
*   **Copy to Clipboard**: Easily copy prompts, descriptions, and file paths.
*   **Pagination**: Navigate through large collections of images.

## Requirements

The required Python libraries are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Extract Prompts

Run the `comfyprompt_extractor.py` script to populate your Parquet database.

**Process all PNGs in a directory:**

```bash
python comfyprompt_extractor.py -i /path/to/your/images --database prompts.parquet
```

**Process a single file:**

```bash
python comfyprompt_extractor.py -i image.png --database prompts.parquet
```

**Process files using a glob pattern:**

```bash
python comfyprompt_extractor.py -i "images/*.png" --database prompts.parquet
```

**Override existing entries:**

```bash
python comfyprompt_extractor.py -i /path/to/your/images --database prompts.parquet --override
```

### 2. View the Gallery

Run the `streamlit_viewer.py` script to launch the web interface.

```bash
streamlit run streamlit_viewer.py -- --database prompts.parquet
```

If you omit the `--database` argument, it will look for a `vision_ai.parquet` file in the current directory.

You can then access the gallery in your web browser at the URL provided by Streamlit (usually `http://localhost:8501`).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.