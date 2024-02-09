# A Cross-Platform app for Lesion-Symptom Mapping Analysis using Multi-perturbation Shapley-Value Analysis

## Introduction
This application is a graphical user interface version for the methodology described in this [paper](https://academic.oup.com/braincomms/article/3/3/fcab204/6362866), allowing users without programming experience to perform Lesion-Symptom mapping on stroke datasets.

## Instructions
### Downloading
Download the latest release for your operating system from the [release section](https://github.com/ShreyDixit/MSA-App/releases). Follow the OS-specific instructions provided below.

**Note:** Your browser or antivirus might issue a warning due to the application being bundled into a single file using `PyInstaller`. This is a standard caution for unrecognized apps but rest assured, the software is secure and open-source.

Example warning in Chrome:

<img src="assets/suspicios-file-windows.png" alt="Warning Saying application is suspicious" width="400"/>

Ignore any such warnings and proceed with the download.

### Starting the application
#### Windows
You might encounter a security warning when running the application:

<img src="assets/defender-warning-windws.png" alt="Warning Saying application is suspicious" width="300"/>

Click on `More Info` followed by `Run Anyway` to launch the app.

#### Remote Linux Server
For running the application on a remote Linux server, X11 forwarding needs to be enabled to support the GUI. Achieve this with:
```bash
ssh -X username@host
```
Now you need to modify the permissions of the file, making the file executable. For this, run the following command:
```bash
chmod +x msa_app_linux
```
Now you can simply start the application using the following command:
```bash
./msa_app_linux
```

**Note:** The application is compiled for `ubuntu-20.04`. Compatibility issues may arise with other Linux distributions.

### Data Format
The application requires two files:

1. **Data File:** Acceptable in either `csv` or `xlsx` format. An example of this file can be found [here](https://github.com/ShreyDixit/MSA-App/blob/master/data/example_data.xlsx). The file should contain patient data with each row representing a patient. Columns should label brain ROIs, showing the percentage alteration in each (values between 0 and 100). The final column should have the NIHSS score or a performance metric for each patient.
2. Voxels File: TThis optional file is needed if you wish to run the analysis until only a few ROIs are left, excluding the non-significant contribution from the Rest of the Brain (ROB). An example is available [here](https://github.com/ShreyDixit/MSA-App/blob/master/data/NumVoxels.xlsx). This file has the number of voxels for each brain ROI.

### Options
The GUI offers several configurable options:

1. NIHSS Score / Performance Dropdown: Choose whether the final data column represents the NIHSS Score or a performance metric. Performance metrics should inversely correlate with stroke severity. For NIHSS scores, the application converts them into performance metrics by subtracting the maximum NIHSS Score from the score of each patient.
2. Machine Learning Model: You can use a variety of Machine Learning models. However, according to our data and experimentation, a Support Vector Regressor works best.
3. Run Iteratve: If enabled, it will run the MSA iteratively until the smalles set of ROI are left with a non-significant contribution from the ROB.
4. Run Network Interaction: If enabled, it will calculate the network interaction of each ROI pairs after the MSA. The interaction between two regions quantifies how much the contribution of the two regions considered jointly is larger or smaller than the sum of the contribution of each of them individually when the other one is lesioned.
5. Binarize Data: Binarizes stroke data by setting values below the median to 0, and above to 1.

## TODO:
- Better Styling of the Application
- More Tests
- Docs and Instructions for Operating Systems I missed
- Add none ML model (Add a check)