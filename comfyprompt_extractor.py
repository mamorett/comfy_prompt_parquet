#!/usr/bin/env python3
"""
ComfyUI Prompt Extractor to Parquet
Extracts positive prompts from ComfyUI-generated PNG files and saves to Parquet database.
"""

import sys
import os
import glob
import json
import argparse
import signal
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from PIL import Image
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Global variables for signal handling
new_entries = []
db_df = None
parquet_path = None
override_mode = False
use_parameters_mode = False


def extract_positive_prompts_only(file_path: str) -> Dict[str, Any]:
    """
    Extract only positive prompts from ComfyUI PNG metadata
    
    Args:
        file_path: Path to the PNG file
        
    Returns:
        Dictionary containing extracted positive prompts only
    """
    try:
        with Image.open(file_path) as img:
            if img.format != 'PNG':
                raise ValueError(f"File is not a PNG: {img.format}")
            
            metadata = img.info
            result = {
                'positive_prompts': []
            }
            
            # Track processed node IDs to avoid duplicates
            processed_nodes = set()
            
            # Try workflow first (usually more detailed)
            if 'workflow' in metadata:
                try:
                    workflow_data = json.loads(metadata['workflow'])
                    prompts = extract_positive_from_workflow(workflow_data, processed_nodes)
                    result['positive_prompts'].extend(prompts)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse workflow JSON: {e}")
            
            # Only check prompt data if we didn't find anything in workflow
            if not result['positive_prompts'] and 'prompt' in metadata:
                try:
                    prompt_data = json.loads(metadata['prompt'])
                    
                    prompts = extract_positive_from_prompt_data(prompt_data, processed_nodes)
                    result['positive_prompts'].extend(prompts)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse prompt JSON: {e}")
            
            return result
            
    except Exception as e:
        raise Exception(f"Error reading PNG file: {e}")

def extract_positive_prompts_parameters(file_path: str) -> Dict[str, Any]:
    """Extract positive prompt using Parameters metadata and direct PNG properties."""
    try:
        with Image.open(file_path) as img:
            if img.format != 'PNG':
                raise ValueError(f"File is not a PNG: {img.format}")

            metadata = img.info
            result = {
                'file_info': {
                    'filename': os.path.basename(file_path),
                    'size': img.size,
                    'mode': img.mode
                },
                'positive_prompts': [],
                'extraction_method': 'parameters'
            }

            # First, try the ORIGINAL parameters extraction
            prompt_text = extract_positive_from_parameters_strict(metadata)
            if prompt_text:
                result['positive_prompts'].append({
                    'text': prompt_text,
                    'node_id': 'parameters',
                    'node_type': 'parameters',
                    'title': 'Parameters',
                    'source': 'parameters'
                })
            else:
                # If original method fails, try PNG properties as fallback
                prompt_text = extract_positive_from_png_properties(metadata)
                if prompt_text:
                    result['positive_prompts'].append({
                        'text': prompt_text,
                        'node_id': 'png_properties',
                        'node_type': 'png_properties',
                        'title': 'PNG Properties',
                        'source': 'png_properties'
                    })

            return result

    except Exception as e:
        raise Exception(f"Error reading PNG file: {e}")


def extract_positive_from_parameters_strict(metadata: Dict[str, Any]) -> str:
    """
    Extract ONLY the positive prompt from an Automatic1111/SD WebUI-style
    'parameters' metadata field.

    Requirements:
    - Do NOT include negative prompt text.
    - Do NOT include leading 'Positive prompt:' label, if present.

    Handles formats like:

        Positive prompt: a cat in a hat
        Negative prompt: lowres, bad anatomy
        Steps: 20, Sampler: Euler, ...

    or:

        a cat in a hat
        Negative prompt: ...
        Steps: ...

    or inline:

        Positive prompt: a cat in a hat, cozy room, Steps: 20, ...
    """
    params_text = metadata.get('parameters')
    if not params_text or not isinstance(params_text, str):
        return ''

    lines = params_text.splitlines()
    positive_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # Stop at negative prompt or settings lines (we don't want them)
        if lower.startswith('negative prompt:'):
            break
        if lower.startswith('steps:'):
            break

        # Handle inline 'Steps:' on the same line as the prompt
        # e.g. "Positive prompt: text, text, Steps: 20, Sampler: Euler"
        if ' steps:' in lower:
            # Keep only before 'Steps:'
            before_steps, _ = stripped.split('Steps:', 1)
            stripped = before_steps.strip()
            lower = stripped.lower()
            if not stripped:
                break  # nothing left

        # Strip leading "Positive prompt:" label if present
        if lower.startswith('positive prompt:'):
            stripped = stripped[len('Positive prompt:'):].strip()
            # Recompute lower for further checks if needed
            lower = stripped.lower()

        if stripped:
            positive_lines.append(stripped)

    prompt_text = ' '.join(positive_lines).strip()
    return prompt_text if is_valid_prompt_text(prompt_text) else ''


def extract_positive_from_png_properties(metadata: Dict[str, Any]) -> str:
    """
    Fallback: extract a prompt-like text from common PNG text fields.

    We intentionally DO NOT touch 'workflow' or JSON-like blobs here,
    to avoid pulling entire workflows, and we avoid pure negative prompts.
    """
    candidate_keys = [
        'parameters',         # sometimes positive prompt can hide here in non-standard cases
        'prompt',
        'positive',
        'positive_prompt',
        'description',
        'comment',
        'Comment',
        'Description',
        'Prompt',
        'PositivePrompt',
    ]

    for key in candidate_keys:
        if key in metadata:
            value = metadata.get(key)
            if isinstance(value, str):
                text = value.strip()

                # Avoid huge blobs / obvious JSON that look like full workflows
                if len(text) > 2000:
                    continue
                if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
                    continue

                # Ignore text that starts with negative prompt label
                if text.lower().startswith('negative prompt:'):
                    continue

                # Strip 'Positive prompt:' label if present
                if text.lower().startswith('positive prompt:'):
                    text = text[len('Positive prompt:'):].strip()

                if is_valid_prompt_text(text):
                    return text

    return ''




def is_valid_prompt_text(text: str) -> bool:
    """
    Check if extracted text is a valid prompt (not just a number or node reference).
    
    Args:
        text: Text to validate
        
    Returns:
        True if text appears to be a valid prompt
    """
    if not text or not text.strip():
        return False
    
    text = text.strip()
    
    # Reject if it's just a pure number (node ID reference)
    if text.isdigit():
        return False
    
    # Reject if it's a very short number-like string (likely a node reference)
    if len(text) <= 5 and text.replace('.', '').replace('-', '').isdigit():
        return False
    
    # Accept everything else (including short prompts)
    return True


def extract_positive_from_workflow(workflow_data: Dict, processed_nodes: set) -> List[Dict]:
    """Extract positive prompts from workflow nodes"""
    positive_prompts = []
    
    nodes = workflow_data.get('nodes', [])
    
    for node in nodes:
        node_id = node.get('id')
        node_type = node.get('type', '')
        title = node.get('title', '').lower()
        
        # Skip if already processed
        if node_id in processed_nodes:
            continue
        
        # Look for CLIPTextEncode nodes
        if (node_type == 'CLIPTextEncode' or 
            'cliptext' in node_type.lower() or 
            node.get('properties', {}).get('Node name for S&R') == 'CLIPTextEncode'):
            
            prompt_text = None
            
            # First try widgets_values
            widgets_values = node.get('widgets_values', [])
            if widgets_values and len(widgets_values) > 0:
                prompt_value = widgets_values[0]
                
                # Handle both string and list types
                if isinstance(prompt_value, list):
                    prompt_text = ' '.join(str(item) for item in prompt_value if item)
                else:
                    prompt_text = str(prompt_value) if prompt_value else None
            
            # If widgets_values is empty, try inputs field
            if not prompt_text or not prompt_text.strip():
                inputs = node.get('inputs', [])
                for input_item in inputs:
                    if isinstance(input_item, dict):
                        # Look for text/string input
                        if input_item.get('name') in ['text', 'prompt', 'string']:
                            widget = input_item.get('widget', {})
                            if isinstance(widget, dict):
                                prompt_text = widget.get('value') or widget.get('default')
                            break
            
            # Skip if still no valid text
            if not prompt_text or not prompt_text.strip():
                continue
            
            # Validate that this is actual prompt text, not a node reference
            if not is_valid_prompt_text(prompt_text):
                continue
            
            # Only include if it's likely a positive prompt
            is_positive = (
                'positive' in title or 
                'pos' in title or
                (title == '' and 'negative' not in prompt_text.lower()[:50]) or
                (title == 'untitled' and 'negative' not in prompt_text.lower()[:50])
            )
            
            # Exclude obvious negative prompts
            is_negative = (
                'negative' in title or 
                'neg' in title or
                prompt_text.lower().strip().startswith('negative')
            )
            
            if is_positive and not is_negative:
                prompt_info = {
                    'text': prompt_text
                }
                
                positive_prompts.append(prompt_info)
                processed_nodes.add(node_id)
    
    return positive_prompts

def resolve_node_reference(node_ref: Any, prompt_data: Dict, depth: int = 0) -> Any:
    """
    Resolve a node reference to its actual value.
    Node references are typically [node_id, output_index] lists.
    
    Args:
        node_ref: The reference to resolve (could be a list, string, or direct value)
        prompt_data: The full prompt data dictionary
        depth: Recursion depth (to prevent infinite loops)
        
    Returns:
        The resolved value, or the original if it can't be resolved
    """
    # Prevent infinite recursion
    if depth > 10:
        return node_ref
    
    # If it's a list with 2 elements [node_id, output_index], it's a node reference
    if isinstance(node_ref, list) and len(node_ref) == 2:
        node_id = str(node_ref[0])
        
        # Look up the referenced node
        if node_id in prompt_data:
            referenced_node = prompt_data[node_id]
            class_type = referenced_node.get('class_type', '')
            inputs = referenced_node.get('inputs', {})
            
            # For String nodes, try multiple field name variations (case-sensitive)
            if class_type == 'String':
                for field in ['String', 'string', 'text', 'value']:
                    if field in inputs:
                        result = inputs[field]
                        if result:  # Only recurse if we got a non-empty value
                            return resolve_node_reference(result, prompt_data, depth + 1)
            
            # For KepStringLiteral nodes
            elif class_type == 'KepStringLiteral':
                for field in ['string', 'String', 'text', 'value']:
                    if field in inputs:
                        result = inputs[field]
                        if result:
                            return resolve_node_reference(result, prompt_data, depth + 1)
            
            # For other text-holding nodes, try common field names (case-insensitive search)
            for field in inputs.keys():
                field_lower = field.lower()
                if field_lower in ['text', 'string', 'value', 'prompt', 'content']:
                    value = inputs[field]
                    if value:
                        return resolve_node_reference(value, prompt_data, depth + 1)
    
    # If it's not a reference, return as-is
    return node_ref



def extract_positive_from_prompt_data(prompt_data: Dict, processed_nodes: set) -> List[Dict]:
    """Extract positive prompts from prompt data structure"""
    positive_prompts = []
    
    for key, value in prompt_data.items():
        if isinstance(value, dict):
            class_type = value.get('class_type', '')
            
            # Skip if already processed
            if key in processed_nodes:
                continue
            
            if class_type == 'CLIPTextEncode':
                inputs = value.get('inputs', {})
                
                # Look for text input - try multiple possible field names
                text_content = None
                for field_name in ['text', 'prompt', 'string', 'conditioning']:
                    if field_name in inputs:
                        text_value = inputs[field_name]
                        
                        # Resolve node references
                        resolved_value = resolve_node_reference(text_value, prompt_data)
                        
                        # Handle both string and list types
                        if resolved_value is not None:
                            if isinstance(resolved_value, list):
                                text_content = ' '.join(str(item) for item in resolved_value if item)
                            elif isinstance(resolved_value, str):
                                text_content = resolved_value
                            else:
                                text_content = str(resolved_value)
                            
                            if text_content and text_content.strip():
                                break
                
                # Validate that this is actual prompt text, not a node reference
                if text_content and text_content.strip() and is_valid_prompt_text(text_content):
                    # Only include if it looks like a positive prompt
                    is_negative = (
                        'negative' in str(text_content).lower()[:50]
                    )
                    
                    if not is_negative:
                        prompt_info = {
                            'text': text_content
                        }
                        
                        positive_prompts.append(prompt_info)
                        processed_nodes.add(key)
    
    return positive_prompts


def load_parquet_db(parquet_path: Path) -> pd.DataFrame:
    """
    Load existing Parquet database or create empty DataFrame with correct schema.
    
    Args:
        parquet_path: Path to the Parquet file
        
    Returns:
        DataFrame with image_path, prompt, description, created_at, and modified_at columns
    """
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        
        # Add created_at column if it doesn't exist
        if 'created_at' not in df.columns:
            df['created_at'] = pd.Timestamp.now()
            print(f"⚠ Added 'created_at' column to existing database")
        
        # Add modified_at column if it doesn't exist
        if 'modified_at' not in df.columns:
            df['modified_at'] = pd.NaT
            print(f"⚠ Added 'modified_at' column to existing database")
        
        return df
    else:
        # Create empty DataFrame with correct schema including datetime columns
        return pd.DataFrame(columns=['image_path', 'prompt', 'description', 'created_at', 'modified_at'])


def save_parquet_db(df: pd.DataFrame, parquet_path: Path):
    """
    Save DataFrame to Parquet file.
    
    Args:
        df: DataFrame to save
        parquet_path: Path to save the Parquet file
    """
    # Ensure parent directory exists
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Parquet
    df.to_parquet(parquet_path, index=False, engine='pyarrow')


def signal_handler(sig, frame):
    """
    Handle Ctrl-C (SIGINT) gracefully by saving progress before exit.
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    print("\n\n⚠ Interrupt received (Ctrl-C). Saving progress...")
    
    saved_count, total_count = save_progress()
    
    if saved_count > 0:
        print(f"✓ Progress saved: {saved_count} new entries added to database")
        print(f"  Total entries in database: {total_count}")
        print(f"  Database: {parquet_path}")
    else:
        print("⊘ No progress to save")
    
    print("\nExiting...")
    sys.exit(0)


def should_process_image(image_path: Path, df: pd.DataFrame, override: bool) -> tuple:
    """
    Check if an image should be processed based on existence in Parquet database.
    
    Args:
        image_path: Full path to the image file
        df: DataFrame containing existing entries
        override: Whether to override existing entries
        
    Returns:
        Tuple of (should_process: bool, existing_entry: dict or None)
        If override is True and entry exists, returns the existing entry for datetime preservation
    """
    image_path_str = str(image_path)
    
    # Check if image_path already exists in database
    existing_mask = df['image_path'] == image_path_str
    
    if existing_mask.any():
        existing_entry = df[existing_mask].iloc[0].to_dict()
        if override:
            # Process it, but preserve the created_at timestamp
            return True, existing_entry
        else:
            # Skip it
            return False, None
    else:
        # New entry
        return True, None


def is_supported_image(file_path: Path) -> bool:
    """
    Check if a file is a supported PNG image.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is a PNG
    """
    return file_path.suffix.lower() == '.png'


def get_image_files_from_directory(directory: Path) -> List[Path]:
    """
    Get all PNG files from a directory with full paths.
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of absolute PNG file paths
    """
    image_files = []
    
    image_files.extend(directory.glob("*.png"))
    image_files.extend(directory.glob("*.PNG"))
    
    # Convert to absolute paths
    return sorted([img.resolve() for img in image_files])


def get_image_files_from_pattern(pattern: str) -> List[Path]:
    """
    Get all PNG files matching a glob pattern.
    
    Args:
        pattern: Glob pattern (e.g., "*.png", "image?.png")
        
    Returns:
        List of absolute PNG file paths
    """
    matched_files = glob.glob(pattern, recursive=False)
    image_files = []
    
    for file_path in matched_files:
        path = Path(file_path).resolve()
        if path.is_file() and is_supported_image(path):
            image_files.append(path)
    
    return sorted(image_files)


def get_image_files_from_list_file(list_file: Path) -> List[Path]:
    """
    Get image files from a text file (paths can be separated by newlines or spaces).
    
    Args:
        list_file: Path to text file containing file paths
        
    Returns:
        List of absolute PNG file paths
    """
    image_files = []
    
    try:
        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                
                for path_str in line.split():
                    file_path = Path(path_str).resolve()
                    if file_path.is_file() and is_supported_image(file_path):
                        image_files.append(file_path)
                    else:
                        print(f"⚠ Warning: Skipping invalid or unsupported file: {path_str}")
    
    except Exception as e:
        print(f"✗ Error reading file list '{list_file}': {str(e)}")
        return []
    
    return sorted(image_files)


def collect_image_files(args) -> List[Path]:
    """
    Collect image files based on the input arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        List of absolute PNG file paths to process
    """
    all_files = []
    
    # Priority 1: File list from text file
    if args.file_list:
        list_path = Path(args.file_list)
        if not list_path.exists():
            print(f"✗ Error: File list '{args.file_list}' does not exist")
            return []
        if not list_path.is_file():
            print(f"✗ Error: '{args.file_list}' is not a file")
            return []
        
        print(f"Reading file list from: {list_path}")
        all_files = get_image_files_from_list_file(list_path)
        return all_files
    
    # Priority 2: Input path (directory, file, or pattern)
    if args.input:
        input_path = Path(args.input)
        
        # Check if it's a directory
        if input_path.exists() and input_path.is_dir():
            all_files = get_image_files_from_directory(input_path)
        
        # Check if it's a single file
        elif input_path.exists() and input_path.is_file():
            if is_supported_image(input_path):
                all_files = [input_path.resolve()]
            else:
                print(f"✗ Error: '{args.input}' is not a PNG file")
                print("Only PNG files are supported")
                return []
        
        # Otherwise, treat as a glob pattern
        else:
            all_files = get_image_files_from_pattern(args.input)
            if not all_files:
                print(f"✗ Error: No matching PNG files found for pattern '{args.input}'")
                return []
    
    # Legacy support: --directory flag (for backward compatibility)
    elif args.directory:
        input_path = Path(args.directory)
        if not input_path.exists():
            print(f"✗ Error: Directory '{args.directory}' does not exist")
            return []
        if not input_path.is_dir():
            print(f"✗ Error: '{args.directory}' is not a directory")
            return []
        all_files = get_image_files_from_directory(input_path)
    
    return all_files


def process_image(
    image_path: Path,
    existing_entry: dict = None,
    pbar: tqdm = None
) -> tuple:
    """
    Process a single image to extract prompts.
    
    Args:
        image_path: Full path to the PNG file
        existing_entry: Existing database entry (if overriding), used to preserve created_at
        pbar: Progress bar instance
        
    Returns:
        Tuple of (success: bool, message: str, prompt_text: str, timestamps: dict)
    """
    try:
        # Extract prompts from PNG metadata
        if use_parameters_mode:
            result = extract_positive_prompts_parameters(str(image_path))
        else:
            result = extract_positive_prompts_only(str(image_path))

        positive_prompts = result.get('positive_prompts', [])

        
        # Concatenate multiple prompts with separator
        if positive_prompts:
            prompt_text = ' | '.join([p['text'] for p in positive_prompts])
        else:
            prompt_text = ''
        
        # Prepare timestamps
        current_time = pd.Timestamp.now()
        timestamps = {}
        
        if existing_entry:
            # Override mode: preserve created_at, update modified_at
            timestamps['created_at'] = existing_entry.get('created_at', current_time)
            timestamps['modified_at'] = current_time
        else:
            # New entry: set created_at, leave modified_at as NaT
            timestamps['created_at'] = current_time
            timestamps['modified_at'] = pd.NaT
        
        if pbar:
            pbar.set_postfix_str(f"✓ {image_path.name}")
        
        return True, "Success", prompt_text, timestamps
        
    except Exception as e:
        if pbar:
            pbar.set_postfix_str(f"✗ {image_path.name}")
        return False, f"Error: {str(e)}", "", {}



def save_progress():
    """
    Save current progress to the Parquet database.
    Called when Ctrl-C is pressed or at the end of processing.
    """
    global new_entries, db_df, parquet_path, override_mode
    
    if new_entries and db_df is not None and parquet_path is not None:
        try:
            new_df = pd.DataFrame(new_entries)
            
            if override_mode:
                # Remove old entries for processed images
                processed_paths = [entry['image_path'] for entry in new_entries]
                db_df_filtered = db_df[~db_df['image_path'].isin(processed_paths)]
            else:
                db_df_filtered = db_df
            
            # Append new entries - handle empty DataFrame case
            if len(db_df_filtered) == 0:
                updated_df = new_df
            else:
                updated_df = pd.concat([db_df_filtered, new_df], ignore_index=True)
            
            # Save updated database
            save_parquet_db(updated_df, parquet_path)
            return len(new_entries), len(updated_df)
        except Exception as e:
            print(f"\n✗ Error saving database: {str(e)}", file=sys.stderr)
            return 0, 0
    return 0, 0



def main():
    """Main function to parse arguments and process images."""
    global new_entries, db_df, parquet_path, override_mode
    
    # Register signal handler for Ctrl-C
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Extract prompts from ComfyUI PNG files and save to Parquet database",
        epilog="""
Examples:
  # Process all PNGs in a directory
  %(prog)s -i /path/to/images --database prompts.parquet
  
  # Process a single file
  %(prog)s -i image.png --database prompts.parquet
  
  # Process files with wildcards
  %(prog)s -i "*.png" --database prompts.parquet
  %(prog)s -i "image?.png" --database prompts.parquet
  
  # Process files from a text file (paths separated by newlines or spaces)
  %(prog)s -f filelist.txt --database prompts.parquet
  
  # Override existing entries
  %(prog)s -i /path/to/images --database prompts.parquet --override
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Create mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        '-i',
        type=str,
        help='Input: directory, single file, or glob pattern (e.g., *.png, image?.png)'
    )
    input_group.add_argument(
        '--file-list',
        '-f',
        type=str,
        help='Text file containing list of image paths (separated by newlines or spaces)'
    )
    input_group.add_argument(
        '--directory',
        '-d',
        type=str,
        help='[DEPRECATED] Use --input instead. Directory containing images to process'
    )
    
    parser.add_argument(
        '--database',
        '--db',
        type=str,
        required=True,
        help='Path to Parquet database file (required)'
    )
    parser.add_argument(
        '--override',
        action='store_true',
        help='Override existing entries in database for images being processed (default: skip existing)'
    )
    parser.add_argument(
        '--use-parameters',
        action='store_true',
        help='Use A1111/parameters-style extraction instead of ComfyUI workflow/prompt JSON'
    )

    
    args = parser.parse_args()
    # Expose parameters-mode choice globally so process_image can use it
    global use_parameters_mode
    use_parameters_mode = args.use_parameters

    
    # Collect image files based on input method
    all_image_files = collect_image_files(args)
    
    if not all_image_files:
        print("✗ No PNG files found")
        return
    
    # Parse database path and set global variable
    parquet_path = Path(args.database).resolve()
    override_mode = args.override
    
    # Load existing Parquet database (will add datetime columns if missing)
    db_df = load_parquet_db(parquet_path)
    
    # Determine input description for display
    if args.file_list:
        input_desc = f"File list: {args.file_list}"
    elif args.input:
        input_desc = f"Input: {args.input}"
    else:
        input_desc = f"Directory: {args.directory}"
    
    print(f"ComfyUI Prompt Extractor")
    print(f"{input_desc}")
    print(f"Parquet database: {parquet_path}")
    print(f"Existing entries in database: {len(db_df)}")
    print(f"Override existing: {args.override}")
    print(f"\n💡 Tip: Press Ctrl-C anytime to save progress and exit gracefully")
    print("-" * 60)
    
    print(f"Found {len(all_image_files)} PNG file(s) total")
    
    # Filter images based on idempotency and collect existing entries for override mode
    images_to_process = []
    image_existing_entries = {}
    
    for img in all_image_files:
        should_process, existing_entry = should_process_image(img, db_df, args.override)
        if should_process:
            images_to_process.append(img)
            if existing_entry:
                image_existing_entries[str(img)] = existing_entry
    
    skipped_count = len(all_image_files) - len(images_to_process)
    
    if skipped_count > 0:
        print(f"Skipping {skipped_count} image(s) already in database")
    
    if not images_to_process:
        print("\n✓ No images to process. All images already exist in database.")
        print("Use --override to reprocess all images.")
        return
    
    print(f"Processing {len(images_to_process)} image(s)\n")
    
    # Process each image with progress bar
    success_count = 0
    error_count = 0
    no_prompt_count = 0
    new_entries = []
    
    with tqdm(
        total=len(images_to_process),
        desc="Processing images",
        unit="img",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    ) as pbar:
        for image_path in images_to_process:
            # Get existing entry if in override mode
            existing_entry = image_existing_entries.get(str(image_path))
            
            success, message, prompt_text, timestamps = process_image(
                image_path,
                existing_entry,
                pbar
            )
            
            if success:
                success_count += 1
                
                # Track images with no prompts
                if not prompt_text:
                    no_prompt_count += 1
                
                # Add new entry with timestamps
                # 'prompt' is empty for compatibility, 'description' contains the extracted prompt
                new_entries.append({
                    'image_path': str(image_path),
                    'prompt': '',  # Empty for compatibility
                    'description': prompt_text,  # Extracted prompt goes here
                    'created_at': timestamps['created_at'],
                    'modified_at': timestamps['modified_at']
                })

            else:
                error_count += 1
                tqdm.write(f"✗ {image_path.name}: {message}")
            
            pbar.update(1)
    
    # Save progress at the end
    print("\nSaving results to database...")
    saved_count, total_count = save_progress()
    
    if saved_count > 0:
        print(f"✓ Database updated: {parquet_path}")
        print(f"  New entries added: {saved_count}")
        print(f"  Total entries in database: {total_count}")
    
    print("-" * 60)
    print(f"Processing complete!")
    print(f"✓ Successfully processed: {success_count}")
    if no_prompt_count > 0:
        print(f"⚠ Images with no prompts found: {no_prompt_count}")
    if error_count > 0:
        print(f"✗ Errors: {error_count}")
    if skipped_count > 0:
        print(f"⊘ Skipped (already in database): {skipped_count}")


if __name__ == "__main__":
    main()

