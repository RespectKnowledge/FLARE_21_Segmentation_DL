# FLARE21_Segmentation_DL

# My Paper Title
This repository is the official implementation of "Efficient Segmentation of Abdominal Organs using Skip 
Residual Block UNet Model"
# Environments and Requirements
•	Windows
•	CPU, RAM, GPU information
•	CUDA version (11.3)
•	python version (3.7)
# To install requirements:
pip install -r requirements.txt
# Dataset
Please you can donwload the dataset from the following website
https://flare.grand-challenge.org/Data/

•	The dataset is divided into training and validation. Inside the training and validation further two folders are

created with name images and masks

# Preprocessing
•	intensity normalization used to process the training, validation and testing images.

Running the data preprocessing code:

python Data_Preprocessing_Flare2021.py

# Training
Please run this python code for training:

python Training_Flare_model.py

# Trained Models
The trained weights can be download here:

# Here is the docker link for retrained on 1k abdominal CT images
https://www.dropbox.com/s/j6ue6it0nrwadjc/aq_enib_flare_seg.tar.gz?dl=0

# Inference
To infer the testing cases, run this command:

python prediction_flare21.py

# Results
Please check the validation results on leaderboard
https://flare.grand-challenge.org/evaluation/challenge/leaderboard/

# Acknowledgement
We thank the contributors of public datasets.
