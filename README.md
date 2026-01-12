# 1.Introduction
This repository contains the implementation code for the paper titled "Deep Learning-Based Phaseless Near-Field to Near-Field Transformation via Joint Height-Frequency Embedding"
# 2.Usage Instructions
1) Execute 'Data generation.py' in the Data Generation directory. Transfer the generated data file (Data.h5) to both the JHFE-Net Training and Validation directories.
2) Run 'JHFE-Net training.py' in the JHFE-Net Training directory. Move the trained model file (Trained JHFE-Net.pth) to the Validation directory.
3) Execute the following scripts to produce the results:
'Testing set.py'
'Patch antennas array.py'.
# 3.Notes
1) Data used in the paper can be downloaded at https://drive.google.com/file/d/1fcRQ08oD-SqCmT4TCmwOD4d-QfBm1pnp/view?usp=sharing
2) 'Trained JHFE-Net.pth' is available for direct download and can be utilized for testing without requiring additional training.
# 4.Maintainers
This project is owned and managed by Dong-Hao Han and Xing-Chang Wei from Zhejiang University, China.
