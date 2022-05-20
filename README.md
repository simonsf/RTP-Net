# RTP-Net

RTP-Net (v1.0)

Introduction
This repository is for the proposed RTP-Net used for AI-assisted contouring in radiotherapy.

Installation
This repository is based on PyTorch 1.1.0, developed in Ubuntu 16.04 environment. The typical install time is usually several miniutes. 

Training
1.	Prepare the training data csv/txt: the full path of training data, including intensity image and mask image (.nii.gz).
2.	Prepare the configuration file (config.py) for training.
3.	Train the model: python test.py –i config.py.

Inference
1.	Prepare the testing data csv/txt: the full path of training data, including intensity image (.nii.gz).
2.	Run the model: python test.py –i test.csv/txt –m model_directory –o output_ directory.
3.	The segmented data could be found in the output directory.

Data 
We have released the anonymized data with annotations of 5 regions.

AI inference application
Due to the commercial issue, the trained model on large dataset is not included.
