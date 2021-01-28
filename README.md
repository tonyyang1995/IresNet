# Deep-learning-Reveals-Brain-structural-Difference-in-Autism-Spectrum-Disorder
## requirements
Please install Python3.6+ and install the following python libraries:
```
pip3 install -r requirements.txt
```
We are using GPU to speed up the computation for deel learning models. The CUDA can be installed from [here](https://developer.nvidia.com/Cuda-downloads)
We are using CUDA 10.2, but you can install the most suitable version for your computers

## dataset
We are testing on two different dataset: preschooler and abide-I.
Abide-I can be download from [here](http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html), we are using the MPRAGE file.
After download the dataset, put data into the dataset file. The strcutral should look like the following:
```
dataset/abide/mprage/caltech/A00033257/MPRAGE.nii.gz
dataset/mri/nii_reori/ASDP200_FAN_YI_NING_Y3846941_32626.nii.gz
```
We maintain a labelfile to get all the paths, the label files are located at:
```
data/abide/labels/ours_test_reori.txt
```
The following is an example in the label files, please change the path into your settings.
"you should put [path]\t[subid]\t[label 1 for asd, 0 for td]". If there is no subid, just put "None".
```
/home/dao2/Desktop/fmri/dataset/mri/nii_reori/ASDP200_FAN_YI_NING_Y3846941_32626.nii.gz	None	1
```
To save your time, we provide an zip file on [onedrive](https://purdue0-my.sharepoint.com/:u:/g/personal/tang385_purdue_edu/EZYPXGqCHJ1Oo7HQ-hJRESYBcza3AYowWt3A7120hlzhoA?e=y3DepZ)

## checkpoints
Our best model are on [onedrive](https://purdue0-my.sharepoint.com/:u:/g/personal/tang385_purdue_edu/EcPFRK6Hkr9Lnw_EnRWOhNgBGdglI_ve1sSdbVo68sxZnw?e=Q3JTwM)
Your can download it and upzip into the checkpoints. 

## run the code
To test abide, run the code like the following. Modify the model and data path into your own settings
```
python3 test_abide.py --load_model checkpoints/abide/abide_best.pth --dataroot dataset/abide/labels/test_abide_7.txt 
python3 test_preschooler.py --load_model checkpoints/preschooler/preschooler_best.pth --dataroot dataset/abide/labels/ours_test_reori.txt 
```

## Highlight features
We save the highlight features using numpy.
sample code is like the following:
```
python3 get_preschooler_features.py
python3 get_abide_features.py
```
The scipts will generate the file with all the features in the form of .npy.
We share zip files of the features on [onedrive](https://purdue0-my.sharepoint.com/:f:/g/personal/tang385_purdue_edu/Etx_9CMjeUpDpLFHVRudrBMB09U2SHr4hlEAYBSlPN9J-A?e=65w9De)

Then we use the script highligh_ba.py to generate correspond nii.gz files:
```
python3 highlight_ba.py
```
your can modify the path in the main function of this scipts.

## Visualization
We are using fsleyes to draw the images. the tutorial of fsl library can be found [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation).
We first load the original nii.gz files of the MPRAGE scans, then overlay the features on it.
Then We set up the contrast and lightness with different color bars. Details are descripted in our manuscriptes.
Demos of asd screenshot are shown in the demo file.
