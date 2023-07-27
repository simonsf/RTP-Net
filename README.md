# RTP-Net
Deep learning empowered volume delineation of whole-body organs-at-risk for accelerated radiotherapy

RTP-Net (v1.1)

Release notes

v1.1 

Updated the test.py, uploaded the csv examples for train and test, and changed functions in utils accordingly. 

v1.0 

Orignial version.

Introduction

This repository is for the proposed RTP-Net used for AI-assisted contouring in radiotherapy. Please refer to the below paper for the detailed description. 

Shi, F., Hu, W., Wu, J. et al. Deep learning empowered volume delineation of whole-body organs-at-risk for accelerated radiotherapy. Nat Commun 13, 6566 (2022). https://doi.org/10.1038/s41467-022-34257-x

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

We have released 50 anonymized data with annotations of CTV and PTV for rectum cancer patients.

AI inference application

Due to the commercial issue, the trained model on large dataset is not included.
