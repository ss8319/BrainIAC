import SimpleITK as sitk
import os
import argparse
import glob
from tqdm import tqdm

def convert_dicom_series_to_nifti(dicom_dir, output_file):
    """
    Convert a single DICOM series to NIFTI format
    Args:
        dicom_dir: Directory containing DICOM files for one scan
        output_file: Output NIFTI file path
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        reader = sitk.ImageSeriesReader()
        
        # Try to get DICOM series IDs (works regardless of file extension)
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
        
        if not series_IDs:
            print(f"No DICOM series found in: {dicom_dir}")
            return False
        
        # Use the first series found (most ADNI scans should have just one series per directory)
        series_ID = series_IDs[0]
        dicom_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, series_ID)
        
        if not dicom_files:
            print(f"No DICOM files found for series in: {dicom_dir}")
            return False
        
        # Print number of files found
        print(f"Found {len(dicom_files)} DICOM files in: {dicom_dir}")
            
        reader.SetFileNames(dicom_files)
        
        # load dicom images
        image = reader.Execute()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        sitk.WriteImage(image, output_file)
        print(f"Successfully converted to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error converting {dicom_dir}: {str(e)}")
        return False

def find_deepest_dicom_dir(start_path):
    """
    Recursively find the deepest directory containing DICOM files
    
    Args:
        start_path: Starting directory to search from
    
    Returns:
        str: Path to deepest directory containing DICOM files, or None if not found
    """
    # Check if current directory contains DICOM series
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(start_path)
    if series_IDs:
        return start_path
    
    # If not, search subdirectories
    subdirs = [d for d in os.listdir(start_path) 
              if os.path.isdir(os.path.join(start_path, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(start_path, subdir)
        dicom_dir = find_deepest_dicom_dir(subdir_path)
        if dicom_dir:
            return dicom_dir
    
    # No DICOM files found in this branch
    return None

def convert_adni_structure(input_root, output_root):
    """
    Convert DICOM to NIFTI while preserving ADNI directory structure
    
    Args:
        input_root: Root directory containing ADNI data structure
        output_root: Output root directory for preserving the same structure
    """
    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)
    
    print(f"Input root directory: {input_root}")
    print(f"Output root directory: {output_root}")
    
    # Create output root if it doesn't exist
    os.makedirs(output_root, exist_ok=True)
    
    # Track statistics
    successful = 0
    failed = 0
    
    # Find all subject directories
    subject_dirs = [d for d in os.listdir(input_root) 
                   if os.path.isdir(os.path.join(input_root, d))]
    
    if not subject_dirs:
        print(f"No subject directories found in {input_root}")
        return
    
    print(f"Found {len(subject_dirs)} subject directories")
    
    # Process each subject
    for subject in tqdm(subject_dirs, desc="Processing subjects"):
        subject_path = os.path.join(input_root, subject)
        
        # Find all modality directories for this subject
        modality_dirs = [d for d in os.listdir(subject_path)
                        if os.path.isdir(os.path.join(subject_path, d))]
        
        for modality in modality_dirs:
            modality_path = os.path.join(subject_path, modality)
            
            # Find all scan date directories for this modality
            scan_dirs = [d for d in os.listdir(modality_path)
                        if os.path.isdir(os.path.join(modality_path, d))]
            
            for scan_date in scan_dirs:
                scan_path = os.path.join(modality_path, scan_date)
                
                # Find the actual directory containing DICOM files (might be deeper)
                dicom_dir = find_deepest_dicom_dir(scan_path)
                
                if not dicom_dir:
                    print(f"✗ No DICOM directory found in: {scan_path}")
                    failed += 1
                    continue
                
                # Create corresponding output path - use the scan_path for output structure
                relative_path = os.path.relpath(scan_path, input_root)
                output_path = os.path.join(output_root, relative_path)
                output_file = os.path.join(output_path, f"{scan_date}.nii.gz")
                
                # Convert DICOM to NIFTI using the actual DICOM directory
                print(f"Found DICOM directory: {dicom_dir}")
                if convert_dicom_series_to_nifti(dicom_dir, output_file):
                    successful += 1
                    print(f"✓ Converted: {relative_path}")
                else:
                    failed += 1
                    print(f"✗ Failed: {relative_path}")
    
    print("\nConversion Summary:")
    print(f"Successfully converted: {successful} scans")
    print(f"Failed conversions: {failed} scans")
    print(f"Output directory: {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ADNI DICOM series to NIFTI format while preserving directory structure")
    parser.add_argument("--input", "-i", required=True, 
                      help="Input root directory containing ADNI data structure")
    parser.add_argument("--output", "-o", required=True, 
                      help="Output root directory for preserving the same structure")
    
    args = parser.parse_args()
    
    convert_adni_structure(args.input, args.output)
