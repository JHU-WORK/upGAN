# ml-dl-final Super-Resolution of Facial Images Using Deep Learning

## Introduction

In the past decade, the field of image super-resolution has seen significant development due to its unique challenges and the abundance of available data. This technology is especially relevant in digital forensics, security systems, facial recognition, and social media algorithms. Our project, ml-dl-final, focuses on developing a deep learning model for 4x super-resolution of facial images. The aim is to improve texture while preserving facial features, ensuring the model's robustness to various poses and lighting conditions, and allowing for handling of arbitrary image sizes. Future enhancements include supporting variable scaling factors at runtime.

## Features and Highlights

* **4x Super-Resolution:** Specifically designed for facial images.
* **Robust Performance:** Effective across various lighting conditions and poses.
* **Flexible Image Size Handling:** Can process images of different sizes.
* **Data Augmentation:** Incorporates random blackout techniques for training

## Folder Structure and Contents

* **./nets:** Contains our model implementations (PaperSR, FSRCNN, ResFSRCNN).
* **./model:** Stores the trained and tested models.
* **train.ipynb:** Jupyter notebook for training PaperSR, FSRCNN, and ResFSRCNN.
* **trainGAN.ipynb:** Jupyter notebook for training the SRGAN model.
* **corruption.py:** Includes functions for data augmentation, like random blackout.
* **./dataset:** In the folder we implement the dataset for Referencing and Inferencing, we create the lower resolution as our input to our model
* if you want to train on colab, you should zip the folders and upload to colab and use the function in the train.ipynb or test.ipynb to unpack the zip file

## Data Augmentation Techniques

We utilize a unique data augmentation strategy - random blackout - to enhance our model's robustness. This method is designed to simulate real-world image corruptions, ensuring our model's effectiveness in diverse conditions.

## Future Work and Contributions

Future development plans include extending the model to support variable scaling factors. Contributions to this project are highly welcome to help achieve these goals and improve the model's capabilities further.
