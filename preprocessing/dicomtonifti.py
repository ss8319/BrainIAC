# Import libraries
import os
import shutil
import pydicom
from datetime import datetime
import csv
import SimpleITK as sitk
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert DICOM images to NIFTI format')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing DICOM files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for NIFTI files')
    
    args = parser.parse_args()
    
    input_path_main = args.input
    output_path = args.output
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    error_log = []

    for input_path in os.listdir(input_path_main):
        if ".DS" not in input_path and '.localized' not in input_path and '.zip' not in input_path:
            input_path = input_path_main+"/"+input_path
            study_id=input_path.split("/")[-1]      ## revert it back to "_" if "/" doesnt work
            print(input_path)
            patient_dir = output_path+"/"+study_id    
            if os.path.exists(patient_dir):
                shutil.rmtree(patient_dir)
            os.mkdir(patient_dir)
            
            matched_lst=[]
            
            for study_date in os.listdir(input_path):
                if ".DS" not in study_date:
                    ## curate dicoms
                    study_path = patient_dir+"/"+study_date
                    try:
                        os.mkdir(study_path)
                    except:
                        pass
                    print(study_path)
                    
                    for scan_type in os.listdir(input_path+"/"+study_date):
                        input_folder = input_path+"/"+study_date + "/" + scan_type
                        output_folder = patient_dir+"/"+study_date+"/"+scan_type
                        print(input_folder, output_folder)

                        reader = sitk.ImageSeriesReader()
                        dicom_names = reader.GetGDCMSeriesFileNames(input_folder)
                        reader.SetFileNames(dicom_names)
                        try:
                            image = reader.Execute()
                            # Added a call to PermuteAxes to change the axes of the data
                            image = sitk.PermuteAxes(image, [2, 1, 0])

                            sitk.WriteImage(image, output_folder+".nii.gz")

                            full_dicom_path = os.path.join(input_folder, os.listdir(input_folder)[0])
                            ds = pydicom.filereader.dcmread(full_dicom_path)
                            # save dicom header 
                            with open(output_folder+".csv", mode='w') as csv_file:
                                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                for elem in ds:
                                    csv_writer.writerow([f"{elem.tag.group:04X}", f"{elem.tag.element:04X}", elem.description(), elem.VR, str(elem.value)])
                        except Exception as e:
                            error_log.append([input_folder,e])
                            pass
                        
                    ## get header metadata
                    for dicom_path in os.listdir(input_path+"/"+study_date):
                        if ".DS" not in dicom_path:
                            full_dicom_path = input_path+"/"+study_date+"/"+dicom_path +"/"
                            print("_____",full_dicom_path)
                            ds = pydicom.filereader.dcmread(full_dicom_path+"/"+os.listdir(full_dicom_path)[0], force=True)
                            try:
                                dob = ds[0x10,0x30].value
                                sex = ds[0x10,0x40].value
                                scan_date = ds[0x08,0x20].value
                                patient_id = ds[0x10,0x20].value
                                try:
                                    weight = ds[0x101030].value
                                except:
                                    weight = 0
                                datetime_object_scan_time = datetime.strptime(scan_date, '%Y%m%d')
                                datetime_object_dob = datetime.strptime(dob, '%Y%m%d')

                                years_at_scan = (datetime_object_scan_time-datetime_object_dob).days/365
                                matched_lst.append([patient_id, input_path+"/"+study_date, study_path,
                                                    dicom_path, dob, years_at_scan, sex, weight, scan_date])
                            except:
                                continue
    
    # error log 
    if error_log:
        print("\nErrors encountered:")
        for error in error_log:
            print(f"File: {error[0]}")
            print(f"Error: {error[1]}\n")

if __name__ == "__main__":
    main()