"""
Dataset Validator: Great Expectations-based validation for medical imaging datasets

Validates dataset integrity before training to prevent ARC from training on corrupted data.

Validations:
- Image dimensions (height, width, channels)
- Pixel value ranges (0-255 for uint8)
- Mask dimensions match image dimensions
- File format consistency
- No missing files
- Unique patient IDs (no data leakage)
- Train/val/test split integrity
- Metadata completeness

This is CRITICAL for autonomous operation - prevents training failures and invalid results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


class DatasetValidator:
    """
    Validates medical imaging datasets using Great Expectations-like checks.

    Prevents ARC from training on corrupted or invalid data.
    """

    def __init__(self):
        """Initialize dataset validator."""
        self.validation_results = []

    def validate_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        task_type: str = "segmentation",
        expected_image_size: Tuple[int, int] = (512, 512),
        expected_channels: int = 3
    ) -> Dict[str, Any]:
        """
        Run complete dataset validation.

        Args:
            dataset_path: Path to dataset directory
            dataset_name: Dataset name
            task_type: Task type (segmentation or classification)
            expected_image_size: Expected (height, width)
            expected_channels: Expected number of channels

        Returns:
            Validation results dict

        Raises:
            DatasetValidationError: If critical validations fail
        """
        logger.info(f"Validating dataset: {dataset_name}")

        dataset_dir = Path(dataset_path)

        if not dataset_dir.exists():
            raise DatasetValidationError(f"Dataset directory not found: {dataset_path}")

        results = {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_dir),
            "task_type": task_type,
            "validated_at": datetime.utcnow().isoformat(),
            "validations": [],
            "passed": True,
            "warnings": [],
            "errors": []
        }

        # Run all validations
        self._validate_directory_structure(dataset_dir, task_type, results)
        self._validate_file_counts(dataset_dir, task_type, results)
        self._validate_image_dimensions(dataset_dir, expected_image_size, expected_channels, results)
        self._validate_pixel_ranges(dataset_dir, results)

        if task_type == "segmentation":
            self._validate_mask_consistency(dataset_dir, results)

        self._validate_metadata(dataset_dir, results)

        # Check if any critical errors
        if results["errors"]:
            results["passed"] = False

        logger.info(f"Validation complete: {dataset_name} - {'PASSED' if results['passed'] else 'FAILED'}")

        return results

    def validate_splits(
        self,
        dataset_path: str,
        dataset_name: str,
        check_patient_leakage: bool = True
    ) -> Dict[str, Any]:
        """
        Validate train/val/test splits for classification tasks.

        Args:
            dataset_path: Path to dataset root (should contain train/val/test)
            dataset_name: Dataset name
            check_patient_leakage: Check for patient ID overlap between splits

        Returns:
            Validation results
        """
        logger.info(f"Validating splits for: {dataset_name}")

        dataset_dir = Path(dataset_path)

        results = {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_dir),
            "validated_at": datetime.utcnow().isoformat(),
            "validations": [],
            "passed": True,
            "warnings": [],
            "errors": []
        }

        # Check split directories exist
        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val"
        test_dir = dataset_dir / "test"

        for split_dir, split_name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
            if not split_dir.exists():
                results["errors"].append(f"Missing {split_name} directory")
                results["passed"] = False
            else:
                # Count files in split
                image_files = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.png"))
                results["validations"].append({
                    "type": "split_count",
                    "split": split_name,
                    "count": len(image_files),
                    "passed": len(image_files) > 0
                })

        # Check for patient leakage if requested
        if check_patient_leakage and all(d.exists() for d in [train_dir, val_dir, test_dir]):
            leakage_check = self._check_patient_leakage(train_dir, val_dir, test_dir)
            results["validations"].append(leakage_check)

            if not leakage_check["passed"]:
                results["errors"].append("Patient ID leakage detected between splits")
                results["passed"] = False

        return results

    # Private validation methods

    def _validate_directory_structure(
        self,
        dataset_dir: Path,
        task_type: str,
        results: Dict[str, Any]
    ):
        """Validate expected directory structure."""
        images_dir = dataset_dir / "images"

        if not images_dir.exists():
            results["errors"].append("Missing images/ directory")
            results["passed"] = False
            return

        if task_type == "segmentation":
            masks_dir = dataset_dir / "masks"
            if not masks_dir.exists():
                results["warnings"].append("Missing masks/ directory for segmentation task")

        results["validations"].append({
            "type": "directory_structure",
            "passed": True
        })

    def _validate_file_counts(
        self,
        dataset_dir: Path,
        task_type: str,
        results: Dict[str, Any]
    ):
        """Validate file counts and consistency."""
        images_dir = dataset_dir / "images"

        # Count images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        image_count = len(image_files)

        if image_count == 0:
            results["errors"].append("No images found in images/ directory")
            results["passed"] = False
            return

        validation = {
            "type": "file_counts",
            "image_count": image_count,
            "passed": True
        }

        # For segmentation, check mask count matches
        if task_type == "segmentation":
            masks_dir = dataset_dir / "masks"
            if masks_dir.exists():
                mask_files = list(masks_dir.glob("*.png"))
                mask_count = len(mask_files)

                validation["mask_count"] = mask_count

                if mask_count != image_count:
                    results["warnings"].append(
                        f"Image count ({image_count}) != mask count ({mask_count})"
                    )
                    validation["passed"] = False

        results["validations"].append(validation)

    def _validate_image_dimensions(
        self,
        dataset_dir: Path,
        expected_size: Tuple[int, int],
        expected_channels: int,
        results: Dict[str, Any]
    ):
        """Validate image dimensions."""
        images_dir = dataset_dir / "images"
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

        if not image_files:
            return

        # Sample first 10 images
        sample_size = min(10, len(image_files))
        sample_files = image_files[:sample_size]

        try:
            import cv2

            dimension_issues = []

            for img_file in sample_files:
                img = cv2.imread(str(img_file))
                if img is None:
                    dimension_issues.append(f"Failed to load: {img_file.name}")
                    continue

                h, w, c = img.shape

                if (h, w) != expected_size:
                    dimension_issues.append(
                        f"{img_file.name}: size {(h, w)} != expected {expected_size}"
                    )

                if c != expected_channels:
                    dimension_issues.append(
                        f"{img_file.name}: channels {c} != expected {expected_channels}"
                    )

            validation = {
                "type": "image_dimensions",
                "expected_size": expected_size,
                "expected_channels": expected_channels,
                "sampled": sample_size,
                "passed": len(dimension_issues) == 0
            }

            if dimension_issues:
                validation["issues"] = dimension_issues[:5]  # Limit to 5
                results["warnings"].extend(dimension_issues[:5])

            results["validations"].append(validation)

        except ImportError:
            logger.warning("OpenCV not available, skipping dimension validation")

    def _validate_pixel_ranges(
        self,
        dataset_dir: Path,
        results: Dict[str, Any]
    ):
        """Validate pixel value ranges."""
        images_dir = dataset_dir / "images"
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

        if not image_files:
            return

        # Sample first 5 images
        sample_size = min(5, len(image_files))
        sample_files = image_files[:sample_size]

        try:
            import cv2

            range_issues = []

            for img_file in sample_files:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                min_val = img.min()
                max_val = img.max()

                # Check if in valid range [0, 255]
                if min_val < 0 or max_val > 255:
                    range_issues.append(
                        f"{img_file.name}: pixel range [{min_val}, {max_val}] outside [0, 255]"
                    )

            validation = {
                "type": "pixel_ranges",
                "expected_range": [0, 255],
                "sampled": sample_size,
                "passed": len(range_issues) == 0
            }

            if range_issues:
                validation["issues"] = range_issues
                results["errors"].extend(range_issues)
                results["passed"] = False

            results["validations"].append(validation)

        except ImportError:
            logger.warning("OpenCV not available, skipping pixel range validation")

    def _validate_mask_consistency(
        self,
        dataset_dir: Path,
        results: Dict[str, Any]
    ):
        """Validate mask consistency for segmentation."""
        images_dir = dataset_dir / "images"
        masks_dir = dataset_dir / "masks"

        if not masks_dir.exists():
            return

        image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        mask_files = sorted(list(masks_dir.glob("*.png")))

        if not image_files or not mask_files:
            return

        # Sample first 5 pairs
        sample_size = min(5, min(len(image_files), len(mask_files)))

        try:
            import cv2

            consistency_issues = []

            for i in range(sample_size):
                img = cv2.imread(str(image_files[i]))
                mask = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)

                if img is None or mask is None:
                    consistency_issues.append(f"Failed to load pair {i}")
                    continue

                # Check dimensions match
                if img.shape[:2] != mask.shape[:2]:
                    consistency_issues.append(
                        f"Dimension mismatch: {image_files[i].name} {img.shape[:2]} != "
                        f"{mask_files[i].name} {mask.shape[:2]}"
                    )

                # Check mask is binary or near-binary
                unique_vals = np.unique(mask)
                if len(unique_vals) > 10:  # Allow some tolerance for anti-aliasing
                    consistency_issues.append(
                        f"Mask {mask_files[i].name} has {len(unique_vals)} unique values (expected ~2)"
                    )

            validation = {
                "type": "mask_consistency",
                "sampled": sample_size,
                "passed": len(consistency_issues) == 0
            }

            if consistency_issues:
                validation["issues"] = consistency_issues
                results["warnings"].extend(consistency_issues)

            results["validations"].append(validation)

        except ImportError:
            logger.warning("OpenCV/NumPy not available, skipping mask validation")

    def _validate_metadata(
        self,
        dataset_dir: Path,
        results: Dict[str, Any]
    ):
        """Validate metadata.json if present."""
        metadata_file = dataset_dir / "metadata.json"

        if not metadata_file.exists():
            results["warnings"].append("No metadata.json found")
            results["validations"].append({
                "type": "metadata",
                "passed": False,
                "found": False
            })
            return

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check for required fields
            required_fields = ["dataset_name", "created_at"]
            missing_fields = [f for f in required_fields if f not in metadata]

            validation = {
                "type": "metadata",
                "passed": len(missing_fields) == 0,
                "found": True
            }

            if missing_fields:
                validation["missing_fields"] = missing_fields
                results["warnings"].append(f"Metadata missing fields: {missing_fields}")

            results["validations"].append(validation)

        except Exception as e:
            results["warnings"].append(f"Failed to parse metadata.json: {e}")
            results["validations"].append({
                "type": "metadata",
                "passed": False,
                "error": str(e)
            })

    def _check_patient_leakage(
        self,
        train_dir: Path,
        val_dir: Path,
        test_dir: Path
    ) -> Dict[str, Any]:
        """
        Check for patient ID overlap between splits.

        Assumes filenames contain patient IDs (e.g., patient_001_img.jpg).
        """
        def extract_patient_ids(directory: Path) -> set:
            """Extract patient IDs from filenames."""
            patient_ids = set()
            for img_file in directory.rglob("*.jpg"):
                # Simple heuristic: first part before underscore
                filename = img_file.stem
                parts = filename.split('_')
                if len(parts) > 0:
                    patient_ids.add(parts[0])
            return patient_ids

        train_patients = extract_patient_ids(train_dir)
        val_patients = extract_patient_ids(val_dir)
        test_patients = extract_patient_ids(test_dir)

        # Check for overlaps
        train_val_overlap = train_patients & val_patients
        train_test_overlap = train_patients & test_patients
        val_test_overlap = val_patients & test_patients

        has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)

        validation = {
            "type": "patient_leakage",
            "passed": not has_leakage,
            "train_patient_count": len(train_patients),
            "val_patient_count": len(val_patients),
            "test_patient_count": len(test_patients)
        }

        if has_leakage:
            validation["leakage"] = {
                "train_val_overlap": len(train_val_overlap),
                "train_test_overlap": len(train_test_overlap),
                "val_test_overlap": len(val_test_overlap)
            }

        return validation


# Global singleton
_validator: Optional[DatasetValidator] = None


def get_dataset_validator() -> DatasetValidator:
    """Get global dataset validator instance."""
    global _validator

    if _validator is None:
        _validator = DatasetValidator()

    return _validator
