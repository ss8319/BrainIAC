# BrainIAC


this is divi 

## command to install the environement 

python == 3.9

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

### runnign the dicom to nifti conversion 
python dicomtonifti.py -i /path/to/input/directory -o /path/to/output/directory

### runnnign the preprocessing 
python your_script.py --temp_img /path/to/temp_img.nii.gz --T2W_dir /path/to/T2W_dir --output_path /path/to/output_dir

### loading the brainiac 

go to util.ipynb  


### getting features from brainiac 

python get_brainiac_features.py \
    --checkpoint /path/to/brainiac/checkpoint.ckpt \
    --input_csv /path/to/input/data.csv \
    --output_csv /path/to/output/features.csv


python get_brainiac_features2.py \
    --checkpoint /media/sdb/divyanshu/divyanshu/foundation_model/code/simclr/checkpoints/WORKS_V3_jointcsvalldataV3_easiertransform_latent2048_adamwnewschrluer_simclr_singlescan_FM_train_RESNET50_1000epcs_batch32_lr0.0001_best-model-epoch=07-train_loss=0.00.ckpt" \
    --input_csv /media/sdb/divyanshu/divyanshu/project_XX/pilot/csvs/oasis3_csvs/saliencymap.csv \
    --output_csv features.csv
    --root_dir /media/sdb/divyanshu/divyanshu/longitudinal_fm/datasets


### get the saliency map from brainiac 

python get_saliency_maps.py \
    --checkpoint /path/to/model/checkpoint.ckpt \
    --input_csv /path/to/input/data.csv \
    --output_dir /path/to/output/saliency_maps \
    --root_dir /path/to/data/root \


python get_brainiac_saliencymap.py \
    --checkpoint /media/sdb/divyanshu/divyanshu/foundation_model/code/simclr/checkpoints/WORKS_V3_jointcsvalldataV3_easiertransform_latent2048_adamwnewschrluer_simclr_singlescan_FM_train_RESNET50_1000epcs_batch32_lr0.0001_best-model-epoch=07-train_loss=0.00.ckpt \
    --input_csv /media/sdb/divyanshu/divyanshu/project_XX/pilot/csvs/oasis3_csvs/saliencymap.csv \
    --output_dir /media/sdb/divyanshu/divyanshu/BrainIAC/BrainIAC \
    --root_dir /media/sdb/divyanshu/divyanshu/longitudinal_fm/datasets


### downstream tasks 

###
