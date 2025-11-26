"""
Dataset Structure Normalizer

Normalizes arbitrary dataset structures to ARC's standard format.

Standard Format (Segmentation):
    dataset_root/
        images/
            *.jpg or *.png
        masks/ (if segmentation)
            *.png
        metadata.json

Standard Format (Classification):
    dataset_root/
        train/
            glaucoma/
                *.png
            normal/
                *.png
        val/
            glaucoma/
            normal/
        test/
            glaucoma/
            normal/
        labels.csv
        metadata.json

Handles:
- Nested folder structures
- Separate train/val/test splits
- MATLAB .mat files (common in RIM-ONE)
- CSV metadata
- Various naming conventions
- DrishtiGS format (Test/Images/{class}/*.png)
- RIMONE format (partitioned_by_hospital/{split}/{class}/*.png)
"""

import os
import shutil
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

# Standard class names for glaucoma classification
GLAUCOMA_CLASS_NAMES = ['normal', 'glaucoma']
CLASS_ALIASES = {
    'glaucoma': 'glaucoma',
    'glaucomatous': 'glaucoma',
    'positive': 'glaucoma',
    'normal': 'normal',
    'healthy': 'normal',
    'negative': 'normal',
}


class DatasetNormalizerError(Exception):
    """Raised when dataset normalization fails."""
    pass


def normalize_dataset_structure(
    input_dir: str,
    output_dir: str,
    dataset_name: str,
    mode: str = "copy"  # or "move"
) -> Dict[str, Any]:
    """
    Normalize dataset structure to ARC standard format.

    Args:
        input_dir: Input dataset directory (potentially messy)
        output_dir: Output directory for normalized dataset
        dataset_name: Dataset name for metadata
        mode: "copy" or "move" files

    Returns:
        Dict with normalization results

    Raises:
        DatasetNormalizerError: If normalization fails
    """
    logger.info(f"Normalizing dataset structure: {input_dir} → {output_dir}")

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise DatasetNormalizerError(f"Input directory does not exist: {input_dir}")

    # Create output structure
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Find all images and masks
    images_found, masks_found = _find_images_and_masks(input_path)

    logger.info(f"Found {len(images_found)} images and {len(masks_found)} masks")

    # Copy/move images
    images_copied = 0
    for src_path in images_found:
        dest_path = images_dir / src_path.name
        if mode == "copy":
            shutil.copy2(src_path, dest_path)
        else:
            shutil.move(str(src_path), dest_path)
        images_copied += 1

    # Copy/move masks
    masks_copied = 0
    for src_path in masks_found:
        dest_path = masks_dir / src_path.name
        if mode == "copy":
            shutil.copy2(src_path, dest_path)
        else:
            shutil.move(str(src_path), dest_path)
        masks_copied += 1

    # Process MATLAB files if present
    mat_files = list(input_path.rglob("*.mat"))
    mat_processed = 0
    if mat_files and HAS_SCIPY:
        for mat_file in mat_files:
            try:
                _process_matlab_file(mat_file, output_path)
                mat_processed += 1
            except Exception as e:
                logger.warning(f"Failed to process MATLAB file {mat_file}: {e}")

    # Create metadata.json
    metadata = {
        "name": dataset_name,
        "description": f"Normalized dataset from {input_dir}",
        "source": str(input_path),
        "normalized_at": datetime.utcnow().isoformat(),
        "statistics": {
            "total_images": images_copied,
            "total_masks": masks_copied,
            "has_segmentation": masks_copied > 0,
            "matlab_files_processed": mat_processed
        },
        "structure": {
            "format": "arc_standard",
            "images_dir": "images/",
            "masks_dir": "masks/" if masks_copied > 0 else None
        }
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Normalization complete: {images_copied} images, {masks_copied} masks")

    return {
        "status": "success",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "images_copied": images_copied,
        "masks_copied": masks_copied,
        "matlab_files_processed": mat_processed,
        "metadata_path": str(metadata_path)
    }


def _find_images_and_masks(root_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Find all images and masks in directory tree.

    Args:
        root_dir: Root directory to search

    Returns:
        Tuple of (image_paths, mask_paths)
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    images = []
    masks = []

    for file_path in root_dir.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()

            if ext in image_extensions:
                # Heuristic detection of masks
                path_lower = str(file_path).lower()

                if any(keyword in path_lower for keyword in ['mask', 'segmentation', 'label', 'ground_truth', 'gt']):
                    masks.append(file_path)
                else:
                    images.append(file_path)

    return images, masks


def _process_matlab_file(mat_file: Path, output_dir: Path):
    """
    Process MATLAB .mat file (common in RIM-ONE dataset).

    Args:
        mat_file: Path to .mat file
        output_dir: Output directory for extracted data
    """
    if not HAS_SCIPY:
        logger.warning("scipy not available, skipping MATLAB file processing")
        return

    try:
        mat_data = sio.loadmat(str(mat_file))

        # Extract relevant data
        # (Structure depends on specific dataset format)
        logger.info(f"Loaded MATLAB file: {mat_file.name}")
        logger.debug(f"Keys: {list(mat_data.keys())}")

        # Save metadata about MATLAB file
        mat_metadata = {
            "filename": mat_file.name,
            "keys": [k for k in mat_data.keys() if not k.startswith('__')],
            "processed_at": datetime.utcnow().isoformat()
        }

        metadata_path = output_dir / f"matlab_{mat_file.stem}.json"
        with open(metadata_path, 'w') as f:
            json.dump(mat_metadata, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to process MATLAB file: {e}")
        raise


def merge_dataset_splits(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    output_dir: str,
    dataset_name: str
) -> Dict[str, Any]:
    """
    Merge separate train/val/test splits into single normalized dataset.

    Args:
        train_dir: Training set directory
        val_dir: Validation set directory
        test_dir: Test set directory
        output_dir: Output directory
        dataset_name: Dataset name

    Returns:
        Dict with merge results
    """
    logger.info(f"Merging dataset splits into {output_dir}")

    output_path = Path(output_dir)
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_dir,
        "val": val_dir,
        "test": test_dir
    }

    total_images = 0
    total_masks = 0
    split_info = {}

    for split_name, split_dir in splits.items():
        if not split_dir or not Path(split_dir).exists():
            continue

        images, masks = _find_images_and_masks(Path(split_dir))

        # Copy images with split prefix
        for img_path in images:
            dest_name = f"{split_name}_{img_path.name}"
            shutil.copy2(img_path, images_dir / dest_name)
            total_images += 1

        # Copy masks with split prefix
        for mask_path in masks:
            dest_name = f"{split_name}_{mask_path.name}"
            shutil.copy2(mask_path, masks_dir / dest_name)
            total_masks += 1

        split_info[split_name] = {
            "images": len(images),
            "masks": len(masks)
        }

    # Create metadata
    metadata = {
        "name": dataset_name,
        "description": f"Merged dataset from train/val/test splits",
        "merged_at": datetime.utcnow().isoformat(),
        "splits": split_info,
        "statistics": {
            "total_images": total_images,
            "total_masks": total_masks,
            "has_segmentation": total_masks > 0
        }
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Merge complete: {total_images} images, {total_masks} masks")

    return {
        "status": "success",
        "output_dir": str(output_dir),
        "total_images": total_images,
        "total_masks": total_masks,
        "split_info": split_info
    }


def detect_dataset_format(dataset_dir: str) -> Dict[str, Any]:
    """
    Auto-detect dataset format and structure.

    Args:
        dataset_dir: Directory to analyze

    Returns:
        Dict with detected format information
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        return {"detected": False, "error": "Directory does not exist"}

    # Check for standard ARC format
    has_images_dir = (dataset_path / "images").exists()
    has_masks_dir = (dataset_path / "masks").exists()
    has_metadata = (dataset_path / "metadata.json").exists()

    if has_images_dir and has_metadata:
        return {
            "detected": True,
            "format": "arc_standard",
            "ready": True,
            "needs_normalization": False
        }

    # Check for split-based format (train/val/test)
    has_train = (dataset_path / "train").exists()
    has_val = (dataset_path / "val").exists()
    has_test = (dataset_path / "test").exists()

    if has_train or has_val or has_test:
        return {
            "detected": True,
            "format": "split_based",
            "ready": False,
            "needs_normalization": True,
            "splits_found": {
                "train": has_train,
                "val": has_val,
                "test": has_test
            }
        }

    # Check for flat structure (all images in root)
    images, masks = _find_images_and_masks(dataset_path)

    if len(images) > 0:
        return {
            "detected": True,
            "format": "flat",
            "ready": False,
            "needs_normalization": True,
            "image_count": len(images),
            "mask_count": len(masks)
        }

    return {
        "detected": False,
        "format": "unknown",
        "ready": False,
        "needs_normalization": True
    }


# ============================================================================
# Classification Dataset Normalization (DrishtiGS, RIMONE, etc.)
# ============================================================================

def _extract_class_from_path(file_path: Path) -> Optional[str]:
    """
    Extract class label from file path based on parent folder names.

    Handles patterns like:
    - .../glaucoma/image.png -> glaucoma
    - .../normal/image.png -> normal
    - .../Images/glaucoma/image.png -> glaucoma

    Args:
        file_path: Path to image file

    Returns:
        Normalized class name or None if not detected
    """
    path_parts = [p.lower() for p in file_path.parts]

    for part in reversed(path_parts[:-1]):  # Skip filename
        if part in CLASS_ALIASES:
            return CLASS_ALIASES[part]

    return None


def _find_images_with_classes(root_dir: Path) -> List[Tuple[Path, str, Optional[str]]]:
    """
    Find all images with their class labels and split info.

    Args:
        root_dir: Root directory to search

    Returns:
        List of (image_path, class_label, split_name) tuples
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    results = []

    for file_path in root_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # Skip mask files
            path_lower = str(file_path).lower()
            if any(kw in path_lower for kw in ['mask', 'segmentation', 'ground_truth', 'gt']):
                continue

            # Extract class
            class_label = _extract_class_from_path(file_path)

            # Extract split (train/val/test)
            split_name = None
            path_parts_lower = [p.lower() for p in file_path.parts]
            for part in path_parts_lower:
                if part in ['train', 'training', 'training_set']:
                    split_name = 'train'
                    break
                elif part in ['val', 'validation', 'valid']:
                    split_name = 'val'
                    break
                elif part in ['test', 'testing', 'test_set']:
                    split_name = 'test'
                    break

            results.append((file_path, class_label, split_name))

    return results


def normalize_classification_dataset(
    input_dir: str,
    output_dir: str,
    dataset_name: str,
    mode: str = "copy",
    create_val_split: bool = True,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, Any]:
    """
    Normalize classification dataset to ARC standard format with class labels.

    Creates structure:
        output_dir/
            train/
                glaucoma/
                normal/
            val/
                glaucoma/
                normal/
            test/
                glaucoma/
                normal/
            labels.csv
            metadata.json

    Args:
        input_dir: Input dataset directory
        output_dir: Output directory for normalized dataset
        dataset_name: Dataset name for metadata
        mode: "copy" or "move" files
        create_val_split: Create validation split if not present
        val_ratio: Ratio for validation split (if creating)
        test_ratio: Ratio for test split (if no test data)

    Returns:
        Dict with normalization results

    Raises:
        DatasetNormalizerError: If normalization fails
    """
    logger.info(f"Normalizing classification dataset: {input_dir} → {output_dir}")

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise DatasetNormalizerError(f"Input directory does not exist: {input_dir}")

    # Find all images with classes
    images_with_labels = _find_images_with_classes(input_path)

    if not images_with_labels:
        raise DatasetNormalizerError(f"No images found in {input_dir}")

    # Count classes
    class_counts = {}
    split_counts = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}

    for _, class_label, split_name in images_with_labels:
        if class_label:
            class_counts[class_label] = class_counts.get(class_label, 0) + 1
        split_counts[split_name or 'unknown'] += 1

    logger.info(f"Found {len(images_with_labels)} images")
    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"Split distribution: {split_counts}")

    # Determine if we need to create splits
    has_splits = split_counts['train'] > 0 or split_counts['test'] > 0

    # Create output directory structure
    for split in ['train', 'val', 'test']:
        for class_name in GLAUCOMA_CLASS_NAMES:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)

    # Process images
    labels_data = []
    files_copied = {'train': 0, 'val': 0, 'test': 0}
    class_per_split = {
        'train': {'glaucoma': 0, 'normal': 0},
        'val': {'glaucoma': 0, 'normal': 0},
        'test': {'glaucoma': 0, 'normal': 0}
    }

    # If no splits exist, we need to create them
    if not has_splits:
        import random
        random.seed(42)
        random.shuffle(images_with_labels)

        n_total = len(images_with_labels)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)

        for i, (img_path, class_label, _) in enumerate(images_with_labels):
            if i < n_test:
                split = 'test'
            elif i < n_test + n_val:
                split = 'val'
            else:
                split = 'train'

            # Use default class if not detected
            if not class_label:
                class_label = 'normal'
                logger.warning(f"No class detected for {img_path.name}, defaulting to 'normal'")

            # Copy/move file
            dest_path = output_path / split / class_label / img_path.name
            if mode == "copy":
                shutil.copy2(img_path, dest_path)
            else:
                shutil.move(str(img_path), dest_path)

            files_copied[split] += 1
            class_per_split[split][class_label] += 1

            # Add to labels CSV
            labels_data.append({
                'filename': img_path.name,
                'split': split,
                'class': class_label,
                'label': 1 if class_label == 'glaucoma' else 0
            })
    else:
        # Use existing splits
        for img_path, class_label, split_name in images_with_labels:
            # Default split assignment
            if not split_name:
                split_name = 'train'

            # Use default class if not detected
            if not class_label:
                class_label = 'normal'
                logger.warning(f"No class detected for {img_path.name}, defaulting to 'normal'")

            # Copy/move file
            dest_path = output_path / split_name / class_label / img_path.name
            if mode == "copy":
                shutil.copy2(img_path, dest_path)
            else:
                shutil.move(str(img_path), dest_path)

            files_copied[split_name] += 1
            class_per_split[split_name][class_label] += 1

            # Add to labels CSV
            labels_data.append({
                'filename': img_path.name,
                'split': split_name,
                'class': class_label,
                'label': 1 if class_label == 'glaucoma' else 0
            })

        # Create validation split from training if needed
        if create_val_split and files_copied['val'] == 0 and files_copied['train'] > 0:
            logger.info("Creating validation split from training data...")
            _create_val_split_from_train(output_path, val_ratio, labels_data, class_per_split, files_copied)

    # Write labels.csv
    labels_path = output_path / "labels.csv"
    with open(labels_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'split', 'class', 'label'])
        writer.writeheader()
        writer.writerows(labels_data)

    logger.info(f"Created labels.csv with {len(labels_data)} entries")

    # Create metadata.json
    metadata = {
        "name": dataset_name,
        "task": "classification",
        "description": f"Normalized classification dataset from {input_dir}",
        "source": str(input_path),
        "normalized_at": datetime.utcnow().isoformat(),
        "classes": GLAUCOMA_CLASS_NAMES,
        "num_classes": len(GLAUCOMA_CLASS_NAMES),
        "statistics": {
            "total_images": len(labels_data),
            "train_images": files_copied['train'],
            "val_images": files_copied['val'],
            "test_images": files_copied['test'],
            "class_distribution": class_counts,
            "split_class_distribution": class_per_split
        },
        "structure": {
            "format": "arc_classification",
            "splits": ["train", "val", "test"],
            "labels_file": "labels.csv"
        }
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Classification normalization complete: {sum(files_copied.values())} images")

    return {
        "status": "success",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "task": "classification",
        "total_images": len(labels_data),
        "files_per_split": files_copied,
        "class_per_split": class_per_split,
        "labels_path": str(labels_path),
        "metadata_path": str(metadata_path)
    }


def _create_val_split_from_train(
    output_path: Path,
    val_ratio: float,
    labels_data: List[Dict],
    class_per_split: Dict,
    files_copied: Dict
):
    """Move some training files to validation split."""
    import random
    random.seed(42)

    train_dir = output_path / 'train'
    val_dir = output_path / 'val'

    for class_name in GLAUCOMA_CLASS_NAMES:
        class_dir = train_dir / class_name
        if not class_dir.exists():
            continue

        files = list(class_dir.glob('*'))
        n_val = max(1, int(len(files) * val_ratio))

        random.shuffle(files)
        val_files = files[:n_val]

        for f in val_files:
            dest = val_dir / class_name / f.name
            shutil.move(str(f), dest)

            # Update labels data
            for entry in labels_data:
                if entry['filename'] == f.name and entry['split'] == 'train':
                    entry['split'] = 'val'
                    break

            files_copied['train'] -= 1
            files_copied['val'] += 1
            class_per_split['train'][class_name] -= 1
            class_per_split['val'][class_name] += 1


def normalize_drishti_gs(
    zip_path: str,
    output_dir: str,
    mode: str = "copy"
) -> Dict[str, Any]:
    """
    Normalize DrishtiGS dataset from ZIP archive.

    DrishtiGS structure:
        Test/Images/glaucoma/*.png
        Test/Images/normal/*.png

    Args:
        zip_path: Path to DrishtiGS ZIP file
        output_dir: Output directory
        mode: "copy" or "move"

    Returns:
        Dict with normalization results
    """
    import zipfile
    import tempfile

    logger.info(f"Normalizing DrishtiGS dataset: {zip_path}")

    # Extract to temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)

        # Find the extracted root (may have nested folder)
        temp_path = Path(temp_dir)
        extracted_dirs = list(temp_path.iterdir())

        if len(extracted_dirs) == 1 and extracted_dirs[0].is_dir():
            input_dir = extracted_dirs[0]
        else:
            input_dir = temp_path

        # Normalize
        return normalize_classification_dataset(
            input_dir=str(input_dir),
            output_dir=output_dir,
            dataset_name="drishti_gs",
            mode="copy",  # Always copy from temp
            create_val_split=True,
            val_ratio=0.15,
            test_ratio=0.0  # DrishtiGS already has test split
        )


def normalize_rimone(
    zip_path: str,
    output_dir: str,
    mode: str = "copy",
    preserve_hospital_splits: bool = True
) -> Dict[str, Any]:
    """
    Normalize RIMONE dataset from ZIP archive.

    RIMONE structure (partitioned_by_hospital):
        RIM-ONE_DL_images/partitioned_by_hospital/
            training_set/glaucoma/*.png  (r1_, r2_, r3_ prefixes = hospitals)
            training_set/normal/*.png
            test_set/glaucoma/*.png
            test_set/normal/*.png

    IMPORTANT: The original RIMONE split mixes hospitals between train/test,
    which causes data leakage. When preserve_hospital_splits=True (default),
    we re-split by hospital to ensure no hospital appears in both train and test.

    Hospital distribution in RIMONE:
    - r1: Hospital 1 (smaller)
    - r2: Hospital 2 (largest)
    - r3: Hospital 3 (medium)

    Args:
        zip_path: Path to RIMONE ZIP file
        output_dir: Output directory
        mode: "copy" or "move"
        preserve_hospital_splits: If True, split by hospital (recommended)

    Returns:
        Dict with normalization results
    """
    import zipfile
    import tempfile
    import random

    logger.info(f"Normalizing RIMONE dataset: {zip_path}")
    logger.info(f"Hospital-based splitting: {preserve_hospital_splits}")

    output_path = Path(output_dir)

    # Extract to temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)

        temp_path = Path(temp_dir)

        # Find partitioned_by_hospital directory
        partitioned_dir = None
        for p in temp_path.rglob('partitioned_by_hospital'):
            partitioned_dir = p
            break

        if not partitioned_dir:
            logger.warning("partitioned_by_hospital not found, falling back to generic normalization")
            rimone_root = temp_path
            for d in temp_path.iterdir():
                if d.is_dir():
                    rimone_root = d
                    break
            return normalize_classification_dataset(
                input_dir=str(rimone_root),
                output_dir=output_dir,
                dataset_name="rimone",
                mode="copy",
                create_val_split=True
            )

        # Collect all images with hospital info
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        images_by_hospital: Dict[str, List[Tuple[Path, str]]] = {}  # hospital -> [(path, class)]

        for file_path in partitioned_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # Extract hospital from filename (r1_, r2_, r3_)
                filename = file_path.name
                hospital = None
                for prefix in ['r1_', 'r2_', 'r3_']:
                    if filename.lower().startswith(prefix):
                        hospital = prefix.rstrip('_')
                        break

                if not hospital:
                    # Try to extract from filename pattern
                    if filename.startswith('r') and '_' in filename:
                        hospital = filename.split('_')[0]
                    else:
                        hospital = 'unknown'

                # Extract class
                class_label = _extract_class_from_path(file_path)
                if not class_label:
                    class_label = 'normal'

                if hospital not in images_by_hospital:
                    images_by_hospital[hospital] = []
                images_by_hospital[hospital].append((file_path, class_label))

        # Log hospital distribution
        logger.info("Hospital distribution:")
        for hospital, images in sorted(images_by_hospital.items()):
            class_counts = {}
            for _, c in images:
                class_counts[c] = class_counts.get(c, 0) + 1
            logger.info(f"  {hospital}: {len(images)} images {class_counts}")

        if not preserve_hospital_splits:
            # Use original splits (not recommended - causes data leakage)
            return normalize_classification_dataset(
                input_dir=str(partitioned_dir.parent),
                output_dir=output_dir,
                dataset_name="rimone",
                mode="copy",
                create_val_split=True
            )

        # Hospital-based splitting strategy:
        # - Use smallest hospital (r1) for test to ensure clean separation
        # - Split remaining hospitals between train and val
        hospitals = sorted(images_by_hospital.keys())
        hospital_sizes = {h: len(imgs) for h, imgs in images_by_hospital.items()}

        # Assign hospitals to splits
        # Strategy: smallest hospital -> test, others -> train (with val split)
        sorted_hospitals = sorted(hospitals, key=lambda h: hospital_sizes.get(h, 0))

        # r1 (smallest) -> test, r2 and r3 -> train/val
        test_hospitals = [sorted_hospitals[0]] if len(sorted_hospitals) > 0 else []
        train_hospitals = sorted_hospitals[1:] if len(sorted_hospitals) > 1 else []

        logger.info(f"Test hospitals: {test_hospitals}")
        logger.info(f"Train hospitals: {train_hospitals}")

        # Create output structure
        for split in ['train', 'val', 'test']:
            for class_name in GLAUCOMA_CLASS_NAMES:
                (output_path / split / class_name).mkdir(parents=True, exist_ok=True)

        labels_data = []
        files_copied = {'train': 0, 'val': 0, 'test': 0}
        class_per_split = {
            'train': {'glaucoma': 0, 'normal': 0},
            'val': {'glaucoma': 0, 'normal': 0},
            'test': {'glaucoma': 0, 'normal': 0}
        }

        # Copy test hospital images
        for hospital in test_hospitals:
            for img_path, class_label in images_by_hospital.get(hospital, []):
                dest = output_path / 'test' / class_label / img_path.name
                shutil.copy2(img_path, dest)
                files_copied['test'] += 1
                class_per_split['test'][class_label] += 1
                labels_data.append({
                    'filename': img_path.name,
                    'split': 'test',
                    'class': class_label,
                    'label': 1 if class_label == 'glaucoma' else 0,
                    'hospital': hospital
                })

        # Copy train hospital images (with val split)
        random.seed(42)
        val_ratio = 0.15

        for hospital in train_hospitals:
            hospital_images = images_by_hospital.get(hospital, [])
            random.shuffle(hospital_images)

            n_val = max(1, int(len(hospital_images) * val_ratio))

            for i, (img_path, class_label) in enumerate(hospital_images):
                if i < n_val:
                    split = 'val'
                else:
                    split = 'train'

                dest = output_path / split / class_label / img_path.name
                shutil.copy2(img_path, dest)
                files_copied[split] += 1
                class_per_split[split][class_label] += 1
                labels_data.append({
                    'filename': img_path.name,
                    'split': split,
                    'class': class_label,
                    'label': 1 if class_label == 'glaucoma' else 0,
                    'hospital': hospital
                })

        # Write labels.csv (with hospital column for transparency)
        labels_path = output_path / "labels.csv"
        with open(labels_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'split', 'class', 'label', 'hospital'])
            writer.writeheader()
            writer.writerows(labels_data)

        # Create metadata
        metadata = {
            "name": "rimone",
            "task": "classification",
            "description": "RIMONE dataset normalized with hospital-based splitting",
            "source": str(zip_path),
            "normalized_at": datetime.utcnow().isoformat(),
            "classes": GLAUCOMA_CLASS_NAMES,
            "num_classes": len(GLAUCOMA_CLASS_NAMES),
            "splitting_strategy": "hospital_based",
            "test_hospitals": test_hospitals,
            "train_hospitals": train_hospitals,
            "statistics": {
                "total_images": len(labels_data),
                "train_images": files_copied['train'],
                "val_images": files_copied['val'],
                "test_images": files_copied['test'],
                "class_distribution": {
                    'glaucoma': sum(1 for d in labels_data if d['class'] == 'glaucoma'),
                    'normal': sum(1 for d in labels_data if d['class'] == 'normal')
                },
                "split_class_distribution": class_per_split,
                "hospital_distribution": {h: len(imgs) for h, imgs in images_by_hospital.items()}
            },
            "structure": {
                "format": "arc_classification",
                "splits": ["train", "val", "test"],
                "labels_file": "labels.csv"
            },
            "data_leakage_prevention": {
                "method": "hospital_separation",
                "description": "No hospital appears in both train and test sets",
                "test_hospitals": test_hospitals,
                "train_val_hospitals": train_hospitals
            }
        }

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"RIMONE normalization complete (hospital-based): {len(labels_data)} images")
        logger.info(f"  Train: {files_copied['train']}, Val: {files_copied['val']}, Test: {files_copied['test']}")

        return {
            "status": "success",
            "input_dir": str(zip_path),
            "output_dir": str(output_dir),
            "task": "classification",
            "splitting_strategy": "hospital_based",
            "test_hospitals": test_hospitals,
            "train_hospitals": train_hospitals,
            "total_images": len(labels_data),
            "files_per_split": files_copied,
            "class_per_split": class_per_split,
            "labels_path": str(labels_path),
            "metadata_path": str(metadata_path)
        }


def normalize_from_zip(
    zip_path: str,
    output_dir: str,
    dataset_name: Optional[str] = None,
    task: str = "auto"
) -> Dict[str, Any]:
    """
    Auto-detect dataset format from ZIP and normalize.

    Args:
        zip_path: Path to ZIP archive
        output_dir: Output directory
        dataset_name: Optional dataset name (auto-detected if not provided)
        task: Task type ("classification", "segmentation", or "auto")

    Returns:
        Dict with normalization results
    """
    import zipfile

    zip_path_obj = Path(zip_path)

    if not zip_path_obj.exists():
        raise DatasetNormalizerError(f"ZIP file not found: {zip_path}")

    # Auto-detect dataset type from filename
    zip_name_lower = zip_path_obj.name.lower()

    if 'drishti' in zip_name_lower or 'test-2021' in zip_name_lower:
        logger.info("Detected DrishtiGS dataset")
        return normalize_drishti_gs(zip_path, output_dir)

    elif 'rimone' in zip_name_lower:
        logger.info("Detected RIMONE dataset")
        return normalize_rimone(zip_path, output_dir)

    else:
        # Generic normalization
        logger.info("Using generic classification normalization")

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)

            return normalize_classification_dataset(
                input_dir=temp_dir,
                output_dir=output_dir,
                dataset_name=dataset_name or zip_path_obj.stem,
                mode="copy"
            )
