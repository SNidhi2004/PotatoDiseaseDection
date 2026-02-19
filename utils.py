"""
Utility functions for data preparation and visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def organize_dataset(source_dir, target_dir, split_ratio=0.8):
    """
    Organize dataset into train/validation splits
    
    Args:
        source_dir: Directory containing class folders
        target_dir: Target directory for organized data
        split_ratio: Train/validation split ratio
    """
    
    # Create target directories
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'validation')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get class folders
    classes = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        images = os.listdir(class_path)
        
        # Split images
        train_images, val_images = train_test_split(
            images, train_size=split_ratio, random_state=42
        )
        
        # Create class directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # Copy images
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)
        
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy2(src, dst)
        
        print(f"Class {class_name}: {len(train_images)} train, {len(val_images)} validation")

def create_augmentation_preview(image_path, output_dir='augmentation_preview'):
    """
    Create a preview of different augmentations applied to an image
    
    Args:
        image_path: Path to source image
        output_dir: Directory to save preview images
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.reshape((1,) + img_array.shape)
    
    # Create augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    
    # Generate and save augmented images
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Generate augmentations
    for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
        if i >= 11:  # Generate 11 augmented images
            break
        
        aug_img = batch[0].astype(np.uint8)
        axes[i+1].imshow(aug_img)
        axes[i+1].set_title(f'Augmentation {i+1}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'augmentation_preview.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Augmentation preview saved to {output_dir}/augmentation_preview.png")

def calculate_dataset_statistics(data_dir):
    """
    Calculate and display dataset statistics
    
    Args:
        data_dir: Directory containing class folders
    """
    
    total_images = 0
    class_counts = {}
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            n_images = len(os.listdir(class_path))
            class_counts[class_name] = n_images
            total_images += n_images
    
    print("=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    
    for class_name, count in class_counts.items():
        percentage = (count / total_images) * 100
        print(f"{class_name}: {count} images ({percentage:.1f}%)")
    
    print("-" * 50)
    print(f"Total: {total_images} images")
    print(f"Number of classes: {len(class_counts)}")
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Dataset Distribution by Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')