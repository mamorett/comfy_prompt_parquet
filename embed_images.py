#!/usr/bin/env python3
"""
Parquet Image Embedder
Converts a Parquet database with image paths into one with embedded image binary data.

Usage:
    python embed_images.py --input database.parquet --output embedded_database.parquet
"""

import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
from PIL import Image
import io

def embed_images(input_path: str, output_path: str = None, scale: float = 0.25, image_column: str = 'image_path', binary_column: str = 'image_bytes', dry_run: bool = False):
    """
    Reads a parquet file, loads images from paths, resizes them, and saves them as bytes in a new parquet.
    Original files are NEVER modified.
    """
    input_p = Path(input_path)
    output_p = Path(output_path) if output_path else None
    
    if not input_p.exists():
        print(f"❌ Error: Input file '{input_path}' not found.")
        return

    print(f"📖 Reading {input_p}...")
    try:
        df = pd.read_parquet(input_p)
    except Exception as e:
        print(f"❌ Error reading parquet: {e}")
        return

    if image_column not in df.columns:
        print(f"❌ Error: Column '{image_column}' not found in database.")
        print(f"Available columns: {list(df.columns)}")
        return

    if dry_run:
        print(f"🔍 DRY RUN: Scanning {len(df)} entries...")
        found = 0
        missing = []
        total_original_bytes = 0
        
        for path_str in tqdm(df[image_column], desc="Verifying paths"):
            if pd.isna(path_str):
                continue
            p = Path(path_str)
            if p.exists() and p.is_file():
                found += 1
                total_original_bytes += p.stat().st_size
            else:
                missing.append(path_str)
        
        print("\n--- Dry Run Report ---")
        print(f"Total entries: {len(df)}")
        print(f"Images found:  {found}")
        print(f"Images missing: {len(missing)}")
        
        # Estimate size: scaling dimensions by S scales area/pixels by S^2
        # We assume file size correlates roughly with pixel count
        est_bytes = total_original_bytes * (scale ** 2)
        est_mb = est_bytes / (1024 * 1024)
        print(f"Estimated image data size (at {scale*100:.0f}% scale): ~{est_mb:.2f} MB")
        
        if missing:
            print("\nFirst 10 missing paths:")
            for m in missing[:10]:
                print(f"  - {m}")
        
        print("\n✅ Dry run complete. No files were written.")
        return

    print(f"🖼️ Embedding images (scaled to {scale*100:.0f}%) from '{image_column}' into '{binary_column}'...")
    
    image_data = []
    missing_count = 0
    error_count = 0
    
    # Use tqdm for progress bar
    for path_str in tqdm(df[image_column], desc="Processing images"):
        if pd.isna(path_str):
            image_data.append(None)
            continue
            
        img_path = Path(path_str)
        if img_path.exists() and img_path.is_file():
            try:
                # Open the original image
                with Image.open(img_path) as img:
                    # Calculate new dimensions
                    new_width = max(1, int(img.width * scale))
                    new_height = max(1, int(img.height * scale))
                    
                    # Resize in memory (Lanczos is high quality)
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save to a buffer in its original format (or JPEG if you prefer)
                    img_format = img.format if img.format else "PNG"
                    buffer = io.BytesIO()
                    resized_img.save(buffer, format=img_format)
                    
                    image_data.append(buffer.getvalue())
            except Exception as e:
                print(f"\n⚠️ Warning: Could not process {img_path}: {e}")
                image_data.append(None)
                error_count += 1
        else:
            image_data.append(None)
            missing_count += 1

    df[binary_column] = image_data
    
    if missing_count > 0:
        print(f"⚠️ Warning: {missing_count} images were not found.")
    if error_count > 0:
        print(f"⚠️ Warning: {error_count} images encountered processing errors.")

    print(f"💾 Saving to {output_p}...")
    try:
        output_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_p, index=False, engine='pyarrow')
        print(f"✅ Success! Embedded database saved to {output_p}")
        
        # Comparison of file sizes
        in_size = input_p.stat().st_size / (1024 * 1024)
        out_size = output_p.stat().st_size / (1024 * 1024)
        print(f"📊 Size: {in_size:.2f} MB -> {out_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error saving parquet: {e}")

def main():
    parser = argparse.ArgumentParser(description="Embed resized images from paths into a Parquet file as binary data.")
    parser.add_argument("-i", "--input", required=True, help="Path to input Parquet file")
    parser.add_argument("-o", "--output", help="Path to output Parquet file (required unless -d is used)")
    parser.add_argument("-s", "--scale", type=float, default=0.25, help="Scale factor for resizing (default: 0.25)")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Scan paths and estimate size without writing output")
    parser.add_argument("--col", default="image_path", help="Column name containing image paths (default: image_path)")
    parser.add_argument("--target", default="image_bytes", help="Column name for embedded bytes (default: image_bytes)")
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.output:
        parser.error("the following arguments are required: -o/--output (use -d/--dry-run to skip output)")

    embed_images(args.input, args.output, args.scale, args.col, args.target, args.dry_run)

if __name__ == "__main__":
    main()
