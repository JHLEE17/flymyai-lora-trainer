#!/usr/bin/env python3
"""
Script to convert Korean text metadata dataset to LoRA training format.
This script reads metadata.csv and creates img{index}.jpg and img{index}.txt pairs.
"""

import os
import csv
import shutil
from pathlib import Path

def setup_korean_dataset():
    """
    Convert the Korean dataset from metadata.csv format to LoRA training format.
    """
    # Paths
    source_dir = Path("/home/user/datasets/korean_large_text")
    metadata_file = source_dir / "metadata.csv"
    target_dir = Path("korean_dataset")
    
    # Create target directory
    target_dir.mkdir(exist_ok=True)
    
    print(f"Setting up Korean dataset from {source_dir} to {target_dir}")
    print(f"Reading metadata from {metadata_file}")
    
    # Read metadata.csv
    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found!")
        return False
    
    successful_pairs = 0
    skipped_files = 0
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        
        for index, row in enumerate(reader, 1):
            # Skip empty rows
            if not row or len(row) < 2:
                continue
                
            original_image_name = row[0].strip()
            prompt = row[1].strip()
            
            # Skip if prompt is empty
            if not prompt:
                print(f"Skipping {original_image_name}: empty prompt")
                skipped_files += 1
                continue
            
            # Source and target file paths
            source_image_path = source_dir / original_image_name
            target_image_path = target_dir / f"img{index:03d}.jpg"  # img001.jpg, img002.jpg, etc.
            target_text_path = target_dir / f"img{index:03d}.txt"
            
            # Check if source image exists
            if not source_image_path.exists():
                print(f"Warning: Source image {source_image_path} not found, skipping...")
                skipped_files += 1
                continue
            
            try:
                # Copy image file
                shutil.copy2(source_image_path, target_image_path)
                
                # Write prompt to text file
                with open(target_text_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(prompt)
                
                successful_pairs += 1
                if successful_pairs % 10 == 0:
                    print(f"Processed {successful_pairs} image-text pairs...")
                    
            except Exception as e:
                print(f"Error processing {original_image_name}: {e}")
                skipped_files += 1
                continue
    
    print(f"\nDataset setup complete!")
    print(f"Successfully created {successful_pairs} image-text pairs")
    print(f"Skipped {skipped_files} files")
    print(f"Dataset saved to: {target_dir.absolute()}")
    
    # Validate the dataset
    print("\nValidating dataset structure...")
    validate_dataset(target_dir)
    
    return successful_pairs > 0

def validate_dataset(dataset_dir):
    """
    Validate that each image has a corresponding text file.
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_path} does not exist!")
        return False
    
    # Get all files
    image_files = list(dataset_path.glob("*.jpg"))
    text_files = list(dataset_path.glob("*.txt"))
    
    print(f"Found {len(image_files)} image files and {len(text_files)} text files")
    
    # Check for mismatched pairs
    image_stems = {f.stem for f in image_files}
    text_stems = {f.stem for f in text_files}
    
    missing_texts = image_stems - text_stems
    missing_images = text_stems - image_stems
    
    if missing_texts:
        print(f"Missing text files for: {sorted(missing_texts)}")
    if missing_images:
        print(f"Missing image files for: {sorted(missing_images)}")
    
    if not missing_texts and not missing_images and len(image_files) > 0:
        print("✅ Dataset structure is valid!")
        print(f"Ready for LoRA training with {len(image_files)} image-text pairs")
        return True
    else:
        print("❌ Dataset structure has issues!")
        return False

if __name__ == "__main__":
    success = setup_korean_dataset()
    if success:
        print("\n🎉 Korean dataset is ready for LoRA training!")
        print("You can now update the training config to use: ./korean_dataset")
    else:
        print("\n❌ Failed to setup dataset. Please check the error messages above.")
