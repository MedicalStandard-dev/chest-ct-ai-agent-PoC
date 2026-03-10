# monai_pipeline/data_processing/lidc_preprocessor.py
"""
LIDC-IDRI Preprocessor
DICOM + XML annotations → NIfTI + heatmap labels
"""
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import json
import xml.etree.ElementTree as ET

try:
    import pydicom
    import nibabel as nib
    from scipy.ndimage import zoom
except ImportError as e:
    raise ImportError(
        "Required packages not found. Install with: "
        "pip install pydicom nibabel scipy"
    ) from e

from utils.logger import logger


class LIDCPreprocessor:
    """
    LIDC-IDRI 데이터 전처리기
    
    LIDC 구조:
    - LIDC-IDRI/
      - LIDC-IDRI-0001/
        - ... / ... / DICOM files
      - ...
      - LIDC-XML-only/
        - ... / ... / XML annotations
    
    출력:
    - processed/
      - images/
        - LIDC-IDRI-0001.nii.gz
      - heatmaps/
        - LIDC-IDRI-0001_heatmap.nii.gz
      - annotations/
        - LIDC-IDRI-0001.json
    """
    
    DEFAULT_CONFIG = {
        "target_spacing": (1.0, 1.0, 1.0),  # Isotropic 1mm
        "intensity_range": (-1000, 400),     # Lung window
        "gaussian_sigma": 3.0,               # Heatmap sigma
        "min_diameter_mm": 3.0,              # Minimum nodule size
        "nodule_agreement_threshold": 2      # Min radiologists agreeing
    }
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        logger.info("LIDCPreprocessor initialized")
    
    def load_dicom_series(self, dicom_dir: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load DICOM series and return volume + metadata
        
        Returns:
            volume: (D, H, W) numpy array in Hounsfield Units
            metadata: dict with spacing, origin, z_positions, etc.
        """
        dicom_files = list(dicom_dir.glob("*.dcm"))
        if not dicom_files:
            dicom_files = [f for f in dicom_dir.iterdir() if f.is_file()]
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")
        
        # Load and sort slices
        slices = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f)
                slices.append(ds)
            except Exception:
                continue
        
        if not slices:
            raise ValueError(f"Could not read any DICOM files in {dicom_dir}")
        
        # Sort by ImagePositionPatient or InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except (AttributeError, TypeError):
            slices.sort(key=lambda x: int(x.InstanceNumber))
        
        # Extract z positions for world-to-voxel coordinate conversion
        z_positions = []
        for s in slices:
            try:
                z_positions.append(float(s.ImagePositionPatient[2]))
            except (AttributeError, TypeError):
                z_positions.append(float(s.InstanceNumber))
        
        # Extract pixel data
        volume = np.stack([s.pixel_array for s in slices], axis=0)
        
        # Convert to Hounsfield Units
        slope = float(getattr(slices[0], "RescaleSlope", 1))
        intercept = float(getattr(slices[0], "RescaleIntercept", 0))
        volume = volume.astype(np.float32) * slope + intercept
        
        # Extract spacing
        pixel_spacing = slices[0].PixelSpacing
        try:
            slice_thickness = float(slices[0].SliceThickness)
        except (AttributeError, TypeError):
            # Estimate from positions
            if len(slices) > 1:
                pos0 = float(slices[0].ImagePositionPatient[2])
                pos1 = float(slices[1].ImagePositionPatient[2])
                slice_thickness = abs(pos1 - pos0)
            else:
                slice_thickness = 1.0
        
        metadata = {
            "spacing": (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1])),
            "z_positions": z_positions,  # World Z coordinates for each slice
            "z_origin": z_positions[0] if z_positions else 0.0,
            "series_uid": str(getattr(slices[0], "SeriesInstanceUID", "unknown")),
            "patient_id": str(getattr(slices[0], "PatientID", "unknown")),
            "study_uid": str(getattr(slices[0], "StudyInstanceUID", "unknown")),
            "shape": volume.shape
        }
        
        return volume, metadata
    
    def resample_volume(
        self,
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Resample volume to target spacing"""
        target_spacing = target_spacing or self.config["target_spacing"]
        
        zoom_factors = tuple(
            o / t for o, t in zip(original_spacing, target_spacing)
        )
        
        resampled = zoom(volume, zoom_factors, order=1, mode="nearest")
        
        return resampled, target_spacing
    
    def normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Clip and normalize to [0, 1]"""
        a_min, a_max = self.config["intensity_range"]
        volume = np.clip(volume, a_min, a_max)
        volume = (volume - a_min) / (a_max - a_min)
        return volume.astype(np.float32)
    
    def parse_lidc_xml(self, xml_path: Path) -> List[Dict]:
        """
        Parse LIDC XML annotation file
        
        Returns:
            List of nodule annotations with center, diameter, etc.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # LIDC XML namespace
        ns = {"lidc": "http://www.nih.gov"}
        
        nodules = []
        
        # Find all reading sessions
        for session in root.findall(".//lidc:readingSession", ns):
            reader_id = session.find("lidc:servicingRadiologistID", ns)
            reader_id = reader_id.text if reader_id is not None else "unknown"
            
            # Find nodules (>= 3mm)
            for nodule in session.findall(".//lidc:unblindedReadNodule", ns):
                nodule_id = nodule.find("lidc:noduleID", ns)
                nodule_id = nodule_id.text if nodule_id is not None else "unknown"
                
                # Get characteristics (if present)
                chars = nodule.find("lidc:characteristics", ns)
                
                # Get ROI (contours)
                rois = nodule.findall(".//lidc:roi", ns)
                
                contour_points = []
                z_positions = []
                
                for roi in rois:
                    z_pos = roi.find("lidc:imageZposition", ns)
                    if z_pos is not None:
                        z_positions.append(float(z_pos.text))
                    
                    for edge in roi.findall(".//lidc:edgeMap", ns):
                        x = edge.find("lidc:xCoord", ns)
                        y = edge.find("lidc:yCoord", ns)
                        if x is not None and y is not None:
                            contour_points.append((
                                float(x.text),
                                float(y.text),
                                z_positions[-1] if z_positions else 0
                            ))
                
                if contour_points:
                    # Compute centroid
                    xs = [p[0] for p in contour_points]
                    ys = [p[1] for p in contour_points]
                    zs = [p[2] for p in contour_points]
                    
                    center = (
                        np.mean(zs),
                        np.mean(ys),
                        np.mean(xs)
                    )
                    
                    # Estimate diameter
                    diameter = max(
                        max(xs) - min(xs),
                        max(ys) - min(ys),
                        max(zs) - min(zs) if len(set(zs)) > 1 else 0
                    )
                    
                    nodules.append({
                        "nodule_id": nodule_id,
                        "reader_id": reader_id,
                        "center_zyx": center,
                        "diameter_mm": diameter,
                        "z_range": (min(zs), max(zs)) if zs else (0, 0),
                        "num_contours": len(rois)
                    })
        
        return nodules
    
    def aggregate_nodules(self, nodules: List[Dict]) -> List[Dict]:
        """
        Aggregate nodules from multiple readers
        
        Returns nodules with >= threshold agreement
        """
        if not nodules:
            return []
        
        # Group by approximate position (within 10mm)
        threshold = self.config["nodule_agreement_threshold"]
        distance_threshold = 10.0  # mm
        
        groups = []
        for nodule in nodules:
            center = np.array(nodule["center_zyx"])
            
            # Find matching group
            matched = False
            for group in groups:
                group_center = np.mean([np.array(n["center_zyx"]) for n in group], axis=0)
                if np.linalg.norm(center - group_center) < distance_threshold:
                    group.append(nodule)
                    matched = True
                    break
            
            if not matched:
                groups.append([nodule])
        
        # Filter by agreement threshold and minimum diameter
        min_diameter = self.config["min_diameter_mm"]
        aggregated = []
        skipped_small = 0
        for group in groups:
            if len(group) >= threshold:
                # Compute consensus values
                centers = [n["center_zyx"] for n in group]
                diameters = [n["diameter_mm"] for n in group]
                mean_diameter = float(np.mean(diameters))

                # Skip nodules below minimum diameter
                if mean_diameter < min_diameter:
                    skipped_small += 1
                    continue

                aggregated.append({
                    "center_zyx": tuple(np.mean(centers, axis=0)),
                    "diameter_mm": mean_diameter,
                    "agreement": len(group),
                    "reader_ids": [n["reader_id"] for n in group]
                })

        if skipped_small > 0:
            logger.info(
                f"Filtered out {skipped_small} nodules with diameter < {min_diameter}mm"
            )

        return aggregated
    
    def generate_heatmap(
        self,
        volume_shape: Tuple[int, int, int],
        nodule_centers: List[Tuple[float, float, float]],
        spacing: Tuple[float, float, float],
        diameters_mm: Optional[List[float]] = None
    ) -> np.ndarray:
        """Generate Gaussian heatmap from nodule centers

        Args:
            volume_shape: (D, H, W)
            nodule_centers: List of (z, y, x) voxel coordinates
            spacing: (sz, sy, sx) in mm
            diameters_mm: Per-nodule diameter in mm. When provided, sigma
                is scaled proportionally (radius / 2) so larger nodules
                produce wider Gaussians. Falls back to config gaussian_sigma.
        """
        from scipy.ndimage import gaussian_filter

        heatmap = np.zeros(volume_shape, dtype=np.float32)
        base_sigma = self.config["gaussian_sigma"]

        if not nodule_centers:
            return heatmap

        # Per-nodule heatmap to allow different sigma per nodule
        for idx, center in enumerate(nodule_centers):
            z, y, x = int(center[0]), int(center[1]), int(center[2])

            if not (0 <= z < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= x < volume_shape[2]):
                continue

            # Determine sigma for this nodule
            if diameters_mm is not None and idx < len(diameters_mm):
                # sigma_mm = radius / 2, clamped to [base_sigma, 3*base_sigma]
                sigma_mm = max(base_sigma, min(diameters_mm[idx] / 4.0, base_sigma * 3.0))
            else:
                sigma_mm = base_sigma

            sigma_voxels = tuple(sigma_mm / s for s in spacing)

            single = np.zeros(volume_shape, dtype=np.float32)
            single[z, y, x] = 1.0
            single = gaussian_filter(single, sigma=sigma_voxels)
            single = single / (single.max() + 1e-8)
            heatmap = np.maximum(heatmap, single)

        return heatmap
    
    def process_case(
        self,
        dicom_dir: Path,
        xml_path: Optional[Path],
        output_dir: Path,
        case_id: str
    ) -> Dict:
        """
        Process single LIDC case
        
        Returns:
            dict with paths to output files and metadata
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        heatmaps_dir = output_dir / "heatmaps"
        annotations_dir = output_dir / "annotations"
        
        for d in [images_dir, heatmaps_dir, annotations_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Load DICOM
        logger.info(f"Processing {case_id}: Loading DICOM...")
        volume, metadata = self.load_dicom_series(dicom_dir)
        
        # Resample
        logger.info(f"Processing {case_id}: Resampling...")
        volume, new_spacing = self.resample_volume(
            volume,
            metadata["spacing"]
        )
        
        # Normalize
        volume_norm = self.normalize_intensity(volume)
        
        # Parse annotations
        nodules = []
        if xml_path and xml_path.exists():
            logger.info(f"Processing {case_id}: Parsing annotations...")
            raw_nodules = self.parse_lidc_xml(xml_path)
            nodules = self.aggregate_nodules(raw_nodules)
            
            # Convert world coordinates to voxel coordinates
            z_positions = metadata.get("z_positions", [])
            original_shape = metadata["shape"]
            
            for nodule in nodules:
                z_world, y_pixel, x_pixel = nodule["center_zyx"]
                
                # Convert Z: world coordinate (mm) -> slice index
                if z_positions:
                    # Find closest slice index
                    z_voxel = self._world_z_to_slice_index(z_world, z_positions)
                else:
                    # Fallback: assume z_world is already in voxel
                    z_voxel = z_world
                
                # X, Y are already in pixel coordinates
                # Clamp to valid range
                z_voxel = max(0, min(z_voxel, original_shape[0] - 1))
                y_pixel = max(0, min(y_pixel, original_shape[1] - 1))
                x_pixel = max(0, min(x_pixel, original_shape[2] - 1))
                
                # Store voxel coordinates in original space
                nodule["center_zyx_original"] = (z_voxel, y_pixel, x_pixel)
                
                # Scale to resampled coordinates
                scale_factors = tuple(
                    o / n for o, n in zip(metadata["spacing"], new_spacing)
                )
                nodule["center_zyx"] = (
                    z_voxel * scale_factors[0],
                    y_pixel * scale_factors[1],
                    x_pixel * scale_factors[2]
                )
            
            logger.info(f"Converted {len(nodules)} nodules to voxel coordinates")
        
        # Generate heatmap
        logger.info(f"Processing {case_id}: Generating heatmap...")
        centers = [n["center_zyx"] for n in nodules]
        diameters = [n["diameter_mm"] for n in nodules]
        heatmap = self.generate_heatmap(volume_norm.shape, centers, new_spacing, diameters_mm=diameters)
        
        # Save NIfTI files
        affine = np.eye(4)
        affine[0, 0] = new_spacing[2]
        affine[1, 1] = new_spacing[1]
        affine[2, 2] = new_spacing[0]
        
        image_path = images_dir / f"{case_id}.nii.gz"
        nib.save(nib.Nifti1Image(volume_norm, affine), image_path)
        
        heatmap_path = heatmaps_dir / f"{case_id}_heatmap.nii.gz"
        nib.save(nib.Nifti1Image(heatmap, affine), heatmap_path)
        
        # Save annotations JSON
        annotation_path = annotations_dir / f"{case_id}.json"
        annotation_data = {
            "case_id": case_id,
            "original_spacing": metadata["spacing"],
            "resampled_spacing": new_spacing,
            "original_shape": metadata["shape"],
            "resampled_shape": volume_norm.shape,
            "series_uid": metadata["series_uid"],
            "patient_id": metadata["patient_id"],
            "nodules": nodules
        }
        with open(annotation_path, "w") as f:
            json.dump(annotation_data, f, indent=2)
        
        logger.info(f"Processed {case_id}: {len(nodules)} nodules found")
        
        return {
            "image_path": str(image_path),
            "heatmap_path": str(heatmap_path),
            "annotation_path": str(annotation_path),
            "num_nodules": len(nodules),
            "metadata": annotation_data
        }
    
    def process_dataset(
        self,
        lidc_root: Path,
        output_dir: Path,
        max_cases: Optional[int] = None
    ) -> List[Dict]:
        """
        Process entire LIDC-IDRI dataset
        
        Args:
            lidc_root: Path to LIDC-IDRI root directory
            output_dir: Output directory
            max_cases: Maximum number of cases to process (for testing)
            
        Returns:
            List of processing results
        """
        lidc_root = Path(lidc_root)
        output_dir = Path(output_dir)
        
        # Find all cases
        case_dirs = sorted([
            d for d in lidc_root.iterdir()
            if d.is_dir() and d.name.startswith("LIDC-IDRI-")
        ])
        
        if max_cases:
            case_dirs = case_dirs[:max_cases]
        
        logger.info(f"Found {len(case_dirs)} cases to process")
        
        results = []
        for case_dir in case_dirs:
            case_id = case_dir.name
            
            # Find DICOM directory (navigate subdirs)
            dicom_dir = self._find_dicom_dir(case_dir)
            if not dicom_dir:
                logger.warning(f"No DICOM found for {case_id}")
                continue
            
            # Find XML annotation (usually in separate folder)
            xml_path = self._find_xml_file(lidc_root, case_id)
            
            try:
                result = self.process_case(
                    dicom_dir=dicom_dir,
                    xml_path=xml_path,
                    output_dir=output_dir,
                    case_id=case_id
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {case_id}: {e}")
        
        # Save manifest
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Processed {len(results)} cases, manifest saved to {manifest_path}")
        return results
    
    def _find_dicom_dir(self, case_dir: Path) -> Optional[Path]:
        """Find CT DICOM directory within case folder"""
        import os
        # LIDC has nested structure: case/study/series/dicoms
        # Multiple series may exist (CT, DX, etc.) - find CT with most files
        candidate_dirs = []
        
        for root, dirs, files in os.walk(case_dir):
            dcm_files = [f for f in files if f.endswith(".dcm")]
            if len(dcm_files) > 10:  # CT series typically has many slices
                # Check if it's CT modality
                try:
                    sample_dcm = Path(root) / dcm_files[0]
                    ds = pydicom.dcmread(sample_dcm, stop_before_pixels=True)
                    if getattr(ds, "Modality", "") == "CT":
                        candidate_dirs.append((Path(root), len(dcm_files)))
                except Exception:
                    continue
        
        # Return directory with most CT files
        if candidate_dirs:
            candidate_dirs.sort(key=lambda x: x[1], reverse=True)
            return candidate_dirs[0][0]
        
        return None
    
    def _find_xml_file(self, lidc_root: Path, case_id: str) -> Optional[Path]:
        """
        Find CT annotation XML file for case
        
        LIDC has two types of XML:
        - CXR XML (IdriReadMessage) - for chest X-ray
        - CT XML (LidcReadMessage) - for CT scans
        
        We need the CT XML.
        """
        # Check common locations
        xml_dir = lidc_root / "LIDC-XML-only"
        if xml_dir.exists():
            for xml_file in xml_dir.rglob("*.xml"):
                if case_id in str(xml_file):
                    if self._is_ct_xml(xml_file):
                        return xml_file
        
        # Check within case directory
        case_dir = lidc_root / case_id
        if case_dir.exists():
            for xml_file in case_dir.rglob("*.xml"):
                if self._is_ct_xml(xml_file):
                    return xml_file
        
        return None
    
    def _is_ct_xml(self, xml_path: Path) -> bool:
        """
        Check if XML is CT annotation (LidcReadMessage), not CXR (IdriReadMessage)
        """
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                # Read first 500 chars to check root element
                header = f.read(500)
                return 'LidcReadMessage' in header
        except Exception:
            return False
    
    def _world_z_to_slice_index(
        self, 
        z_world: float, 
        z_positions: List[float]
    ) -> float:
        """
        Convert world Z coordinate (mm) to slice index
        
        LIDC XML uses ImageZposition which is the Z coordinate in patient space.
        We need to map this to the corresponding slice index.
        
        Args:
            z_world: Z coordinate from XML (mm)
            z_positions: List of Z positions for each slice (from DICOM)
            
        Returns:
            Interpolated slice index (float)
        """
        if not z_positions:
            return z_world
        
        # z_positions may be increasing or decreasing
        z_min, z_max = min(z_positions), max(z_positions)
        is_ascending = z_positions[0] < z_positions[-1]
        
        # Clamp to valid range
        z_clamped = max(z_min, min(z_max, z_world))
        
        # Binary search for closest slice
        # For interpolation: find two closest slices
        best_idx = 0
        best_dist = abs(z_positions[0] - z_clamped)
        
        for i, z_pos in enumerate(z_positions):
            dist = abs(z_pos - z_clamped)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        # Interpolate between neighboring slices if possible
        if best_idx > 0 and best_idx < len(z_positions) - 1:
            z_prev = z_positions[best_idx - 1]
            z_curr = z_positions[best_idx]
            z_next = z_positions[best_idx + 1]
            
            # Check which neighbor is closer
            if abs(z_clamped - z_prev) < abs(z_clamped - z_next):
                # Interpolate between prev and curr
                if abs(z_curr - z_prev) > 1e-6:
                    t = (z_clamped - z_prev) / (z_curr - z_prev)
                    return (best_idx - 1) + t
            else:
                # Interpolate between curr and next
                if abs(z_next - z_curr) > 1e-6:
                    t = (z_clamped - z_curr) / (z_next - z_curr)
                    return best_idx + t
        
        return float(best_idx)


def create_training_split(
    manifest_path: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    require_nodules: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create train/val split from manifest
    
    Args:
        manifest_path: Path to manifest.json
        train_ratio: Ratio for training set
        seed: Random seed
        require_nodules: If True, exclude cases with 0 nodules (for nodule detection)
    
    Returns:
        (train_files, val_files) in MONAI format
    """
    np.random.seed(seed)
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Filter cases with nodules if required
    if require_nodules:
        original_count = len(manifest)
        manifest = [m for m in manifest if m.get("num_nodules", 0) > 0]
        filtered_count = len(manifest)
        logger.info(f"Filtered to cases with nodules: {filtered_count}/{original_count}")
    
    if len(manifest) == 0:
        raise ValueError("No cases with nodules found in manifest!")
    
    # Shuffle
    np.random.shuffle(manifest)
    
    # Split
    n_train = int(len(manifest) * train_ratio)
    train_manifest = manifest[:n_train]
    val_manifest = manifest[n_train:]
    
    # Convert to MONAI format
    def to_monai_format(items):
        return [
            {"image": item["image_path"], "heatmap": item["heatmap_path"]}
            for item in items
        ]
    
    return to_monai_format(train_manifest), to_monai_format(val_manifest)
