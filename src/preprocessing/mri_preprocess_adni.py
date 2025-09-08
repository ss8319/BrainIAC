import sys
import os
import glob
import SimpleITK as sitk
from tqdm import tqdm
import random
from HD_BET.hd_bet import hd_bet
import argparse
import torch
import shutil
import logging
import traceback
import psutil
import time
import gc

def brain_extraction(input_dir, output_dir, device):
    """
    Brain extraction using HDBET package (UNet based DL method)
    Args:
        input_dir {path} -- input directory for registered images
        output_dir {path} -- output directory for brain extracted images
    Returns:
        Brain images
    """
    print("Running brain extraction...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Debug: Check input files
    input_files = os.listdir(input_dir)
    print(f"Input files: {input_files}")
    
    # Run HD-BET directly with the output directory
    try:
        hd_bet(input_dir, output_dir, device=device, mode='fast', tta=0)
        print('HD-BET completed without errors')
    except Exception as e:
        print(f'HD-BET failed with error: {e}')
        raise e
    
    print('Brain extraction complete!')
    print("\nContents of output directory after brain extraction:")
    output_files = os.listdir(output_dir)
    print(f"Output files: {output_files}")
    
    # Debug: Check for expected output pattern
    for file in output_files:
        if '_bet.nii.gz' in file:
            print(f"Found brain extraction output: {file}")
        else:
            print(f"Other file: {file}")

def registration(input_file, output_file, temp_img, interp_type='linear'):
    """
    MRI registration with SimpleITK for a single file
    Args:
        input_file {path} -- Input NIfTI file
        output_file {path} -- Output NIfTI file path
        temp_img {str} -- Registration image template
    Returns:
        bool -- True if successful, False otherwise
    """
    
    try:
        # Read the template image
        fixed_img = sitk.ReadImage(temp_img, sitk.sitkFloat32)
        
        # Read and preprocess moving image
        moving_img = sitk.ReadImage(input_file, sitk.sitkFloat32)
        moving_img = sitk.N4BiasFieldCorrection(moving_img)

        # Resample fixed image to 1mm isotropic
        old_size = fixed_img.GetSize()
        old_spacing = fixed_img.GetSpacing()
        new_spacing = (1, 1, 1)
        new_size = [
            int(round((old_size[0] * old_spacing[0]) / float(new_spacing[0]))),
            int(round((old_size[1] * old_spacing[1]) / float(new_spacing[1]))),
            int(round((old_size[2] * old_spacing[2]) / float(new_spacing[2])))
        ]

        # Set interpolation type
        if interp_type == 'linear':
            interp_type = sitk.sitkLinear
        elif interp_type == 'bspline':
            interp_type = sitk.sitkBSpline
        elif interp_type == 'nearest_neighbor':
            interp_type = sitk.sitkNearestNeighbor

        # Resample fixed image
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputOrigin(fixed_img.GetOrigin())
        resample.SetOutputDirection(fixed_img.GetDirection())
        resample.SetInterpolator(interp_type)
        resample.SetDefaultPixelValue(fixed_img.GetPixelIDValue())
        resample.SetOutputPixelType(sitk.sitkFloat32)
        fixed_img = resample.Execute(fixed_img)

        # Initialize transform
        transform = sitk.CenteredTransformInitializer(
            fixed_img, 
            moving_img, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY)

        # Set up registration method
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0, 
            numberOfIterations=100, 
            convergenceMinimumValue=1e-6, 
            convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetInitialTransform(transform)

        # Execute registration
        final_transform = registration_method.Execute(fixed_img, moving_img)

        # Apply transform and save registered image
        moving_img_resampled = sitk.Resample(
            moving_img, 
            fixed_img, 
            final_transform, 
            sitk.sitkLinear, 
            0.0, 
            moving_img.GetPixelID())
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save with _0000 suffix as required by HD-BET
        sitk.WriteImage(moving_img_resampled, output_file)
        print(f"Saved registered image to: {output_file}")
        return True

    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

# This function is replaced by the new process_single_image function and main function

def setup_logging(output_dir):
    """
    Set up logging to file and console
    """
    log_file = os.path.join(output_dir, 'preprocessing.log')
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_system_info():
    """
    Log system information for debugging
    """
    logging.info("=== System Information ===")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"SimpleITK version: {sitk.Version()}")
    logging.info(f"Torch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Memory information
    mem = psutil.virtual_memory()
    logging.info(f"Total memory: {mem.total / (1024**3):.1f} GB")
    logging.info(f"Available memory: {mem.available / (1024**3):.1f} GB")
    logging.info(f"Used memory: {mem.used / (1024**3):.1f} GB")
    logging.info(f"Memory percent: {mem.percent}%")
    
    # CPU information
    logging.info(f"CPU count: {psutil.cpu_count()}")
    logging.info(f"CPU percent: {psutil.cpu_percent()}%")
    
    logging.info("========================")

def process_single_image(nifti_file, temp_img, input_root, output_root, device):
    """
    Process a single image completely (registration + skull stripping)
    
    Args:
        nifti_file: Path to input NIfTI file
        temp_img: Path to template image
        input_root: Root input directory
        output_root: Root output directory
        device: Device for HD-BET
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create temporary directories for processing
        temp_reg_dir = os.path.join(output_root, 'temp_reg')
        temp_bet_dir = os.path.join(output_root, 'temp_bet')
        os.makedirs(temp_reg_dir, exist_ok=True)
        os.makedirs(temp_bet_dir, exist_ok=True)
        
        # Get base name and prepare paths
        base_name = os.path.splitext(os.path.splitext(os.path.basename(nifti_file))[0])[0]
        temp_reg_file = os.path.join(temp_reg_dir, f"{base_name}_0000.nii.gz")
        
        # Log memory usage before registration
        mem = psutil.virtual_memory()
        logging.info(f"Memory before registration: {mem.percent}% used, {mem.available / (1024**3):.1f} GB available")
        
        # Step 1: Registration
        logging.info(f"Registering {nifti_file}")
        if not registration(nifti_file, temp_reg_file, temp_img):
            logging.error(f"Registration failed for {nifti_file}")
            # Clean up temporary directories
            shutil.rmtree(temp_reg_dir, ignore_errors=True)
            shutil.rmtree(temp_bet_dir, ignore_errors=True)
            return False
        
        # Force garbage collection
        gc.collect()
        
        # Log memory usage after registration
        mem = psutil.virtual_memory()
        logging.info(f"Memory after registration: {mem.percent}% used, {mem.available / (1024**3):.1f} GB available")
        
        # Step 2: Brain extraction
        logging.info(f"Performing brain extraction on {temp_reg_file}")
        try:
            # Use different input and output directories for HD-BET
            brain_extraction(temp_reg_dir, temp_bet_dir, device)
            
        except Exception as e:
            logging.error(f"Brain extraction failed: {str(e)}")
            logging.error(traceback.format_exc())
            # Clean up temporary directories
            shutil.rmtree(temp_reg_dir, ignore_errors=True)
            shutil.rmtree(temp_bet_dir, ignore_errors=True)
            return False
        
        # Force garbage collection
        gc.collect()
        
        # Log memory usage after brain extraction
        mem = psutil.virtual_memory()
        logging.info(f"Memory after brain extraction: {mem.percent}% used, {mem.available / (1024**3):.1f} GB available")
        
        # Step 3: Move to final location
        # HD-BET doesn't add _bet suffix, it just processes the file in place
        extracted_file = os.path.join(temp_bet_dir, f"{base_name}_0000.nii.gz")
        if not os.path.exists(extracted_file):
            logging.error(f"Expected brain extraction output not found: {extracted_file}")
            # Clean up temporary directories
            shutil.rmtree(temp_reg_dir, ignore_errors=True)
            shutil.rmtree(temp_bet_dir, ignore_errors=True)
            return False
        
        # Get relative path to maintain ADNI directory structure
        # ADNI structure: input_dir/subject/scan_type/scan_date/file.nii.gz
        # We want to preserve: subject/scan_type/scan_date/file.nii.gz
        rel_path = os.path.relpath(nifti_file, input_root)
        output_file = os.path.join(output_root, rel_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Copy the file
        shutil.copy(extracted_file, output_file)
        logging.info(f"Saved processed image to: {output_file}")
        
        # Clean up temporary directories
        shutil.rmtree(temp_reg_dir, ignore_errors=True)
        shutil.rmtree(temp_bet_dir, ignore_errors=True)
        
        return True
    
    except Exception as e:
        logging.error(f"Error processing {nifti_file}: {str(e)}")
        logging.error(traceback.format_exc())
        # Clean up temporary directories if they exist
        if 'temp_reg_dir' in locals():
            shutil.rmtree(temp_reg_dir, ignore_errors=True)
        if 'temp_bet_dir' in locals():
            shutil.rmtree(temp_bet_dir, ignore_errors=True)
        return False

def main():
    """
    Main function to process brain MRI images
    """
    parser = argparse.ArgumentParser(description="Process brain MRI registration and skull stripping while preserving ADNI directory structure")
    parser.add_argument("--temp_img", type=str, required=True, help="Path to the atlas template image.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input ADNI directory structure.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the processed images with the same structure.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging.")
    parser.add_argument("--resume", action="store_true", help="Resume processing from where it left off.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of files to process (for testing).")
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Log system information
    log_system_info()
    
    try:
        # Find all NIfTI files recursively
        input_root = os.path.abspath(args.input_dir)
        output_root = os.path.abspath(args.output_dir)
        temp_img = os.path.abspath(args.temp_img)
        
        logging.info(f"Input root directory: {input_root}")
        logging.info(f"Output root directory: {output_root}")
        logging.info(f"Template image: {temp_img}")
        
        nifti_files = []
        # Define brain scan patterns (skip calibration and localizer scans)
        brain_patterns = ['MPRAGE', 'MP-RAGE', 'T1', 'T1W', 'T1w', 'FSPGR', 'SPGR', 'BRAVO']
        skip_patterns = ['localizer', 'calibration', 'survey', 'scout', 'B1-calibration']
        
        for root, _, files in os.walk(input_root):
            for file in files:
                if file.endswith(".nii.gz"):
                    file_path = os.path.join(root, file)
                    # Check if this is a brain scan
                    is_brain_scan = any(pattern in file_path for pattern in brain_patterns)
                    is_skip_scan = any(pattern in file_path.lower() for pattern in skip_patterns)
                    
                    if is_brain_scan and not is_skip_scan:
                        nifti_files.append(file_path)
                        logging.info(f"Added brain scan: {file_path}")
                    else:
                        logging.info(f"Skipped non-brain scan: {file_path}")
        
        if not nifti_files:
            logging.error(f"No NIfTI files found in {input_root}")
            return
        
        # Sort files for consistent processing order
        nifti_files.sort()
        
        # Limit files if specified
        if args.limit:
            nifti_files = nifti_files[:args.limit]
            logging.info(f"Limited to processing {args.limit} files")
        
        logging.info(f"Found {len(nifti_files)} NIfTI files")
        
        # Skip already processed files if resuming
        if args.resume:
            processed_files = []
            for nifti_file in nifti_files:
                # Use the same relative path logic as in processing
                rel_path = os.path.relpath(nifti_file, input_root)
                output_file = os.path.join(output_root, rel_path)
                
                if os.path.exists(output_file):
                    logging.info(f"Skipping already processed file: {rel_path}")
                    processed_files.append(nifti_file)
            
            # Remove processed files from the list
            for processed in processed_files:
                nifti_files.remove(processed)
            
            logging.info(f"Resuming with {len(nifti_files)} files left to process")
        
        # Set device for brain extraction
        device = "0" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        
        # Process each file
        successful = 0
        failed = 0
        
        for i, nifti_file in enumerate(nifti_files):
            logging.info(f"Processing file {i+1}/{len(nifti_files)}: {nifti_file}")
            
            start_time = time.time()
            if process_single_image(nifti_file, temp_img, input_root, output_root, device):
                successful += 1
                logging.info(f"Successfully processed {nifti_file}")
            else:
                failed += 1
                logging.error(f"Failed to process {nifti_file}")
            
            elapsed_time = time.time() - start_time
            logging.info(f"Processing took {elapsed_time:.1f} seconds")
            
            # Force garbage collection
            gc.collect()
        
        logging.info("\nProcessing Summary:")
        logging.info(f"Successfully processed: {successful} images")
        logging.info(f"Failed processing: {failed} images")
        logging.info(f"Output directory: {output_root}")
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
