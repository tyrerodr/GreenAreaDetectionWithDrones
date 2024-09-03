# Green Area Detection with Drones

## Description

This project focuses on the detection and segmentation of green areas in aerial images captured by drones. It utilizes a semantic segmentation model based on U-Net with transfer learning, using EfficientNetB3 as the backbone. The dataset consists of approximately 400 high-resolution images (6000x4000 pixels), captured from altitudes ranging from 5 to 30 meters. The images are available from the [Drone Dataset](http://dronedataset.icg.tugraz.at).

![image](https://github.com/user-attachments/assets/f01eb3d4-a149-4d3e-aba3-1297339171fc)

## Methodology

### 1. Dataset Preparation

- **Image Collection**: High-resolution images obtained from the provided dataset.
- **Mask Configuration**: Annotation of images with masks to identify green areas.
- **Preprocessing**: Includes normalization of images and adjustment of masks for segmentation.

### 2. Model Implementation

- **U-Net Model with EfficientNetB3**: Configuration and training of U-Net with EfficientNetB3 as the backbone to improve precision and efficiency.
- **Transfer Learning**: Utilization of pre-trained features to enhance the model's generalization capability.

### 3. Training and Evaluation

- **Model Training**: Use the `train.ipynb` notebook to train the U-Net model.
- **Metric Calculation**: Evaluation of the model with metrics such as precision, recall, and F1-score.
- **Result Visualization**: Implementation of a function to visualize segmentation compared to the original masks.

### 4. Result Analysis

- **Green Area Percentage Calculation**: Estimation of the green area detected compared to the original masks.
- **Comparison and Adjustments**: Comparison of results and adjustments to improve model precision.

## Project Files

- `DataAugmentation.py`: Script for performing data augmentation on the dataset.
- `DataGenerator.py`: Data generator for loading images and masks during training.
- `Model.py`: Definition of the U-Net model with EfficientNetB3 as the backbone.
- `greenAreaCalculate.py`: Script for calculating the percentage of green area in segmented images.
- `resize_and_delete_images.py`: Script for resizing and removing unnecessary images from the dataset.
- `predict.ipynb`: Notebook for making predictions and visualizing results.
- `train.ipynb`: Notebook for training the U-Net model.
- `requirements.txt`: File containing project dependencies.
- `best_model.weights.h5`: Weights of the best-trained model.
- `image.jpg`: Example image from the dataset.
- `mask_image.jpg`: Example mask corresponding to the example image.

## Installation

Make sure you have Python 3.x installed. Then, install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Contributing

This project was developed with contributions from:

- **Eng. Bryan Paul Alava Calderón**  
  Faculty of Electrical and Computing Engineering (FIEC)  
  Escuela Superior Politécnica del Litoral - ESPOL  
  Guayaquil, Ecuador  
  [bpalava@espol.edu.ec](mailto:bpalava@espol.edu.ec)

- **Eng. Paul del Pezo**  
  Faculty of Electrical and Computing Engineering (FIEC)  
  Escuela Superior Politécnica del Litoral - ESPOL  
  Guayaquil, Ecuador  
  [paudpez@espol.edu.ec](mailto:paudpez@espol.edu.ec)

- **Eng. Aaron Villao**  
  Faculty of Electrical and Computing Engineering (FIEC)  
  Escuela Superior Politécnica del Litoral - ESPOL  
  Guayaquil, Ecuador  
  [avillao@espol.edu.ec](mailto:avillao@espol.edu.ec)

- **Eng. Tyrone Eduardo Rodriguez Motato**  
  Faculty of Electrical and Computing Engineering (FIEC)  
  Escuela Superior Politécnica del Litoral - ESPOL  
  Guayaquil, Ecuador  
  [tyrerodr@espol.edu.ec](mailto:tyrerodr@espol.edu.ec)

- **PhD. Miguel Andrés Realpe Robalino**  
  Faculty of Electrical and Computing Engineering (FIEC)  
  Escuela Superior Politécnica del Litoral - ESPOL  
  Guayaquil, Ecuador  
  [mrealpe@espol.edu.ec](mailto:mrealpe@espol.edu.ec)  
  Project Advisor
