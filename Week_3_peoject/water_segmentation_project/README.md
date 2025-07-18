# Water Segmentation with U-Net

This project provides an end-to-end deep learning pipeline for high-accuracy water body segmentation. It uses a **U-Net architecture** implemented in TensorFlow to analyze 12-channel multispectral and optical data from Harmonized Sentinel-2/Landsat satellites. The pipeline covers everything from data preprocessing and visualization to model training and evaluation, making it a robust tool for environmental monitoring, flood management, and conservation.

## Data Overview

The model processes 12-channel multispectral and optical data from Harmonized Sentinel-2/Landsat satellites to produce a binary water mask.


A sample visualization of the 12 channels for a single image is shown below:

![Sample 12-Channel Image Bands](./data/visualizations/sample_bands_253.png)
_Figure 1: Visualization of the 12 channels for a sample image._

## Model Architecture

The core of this project is a **U-Net model**, implemented in the `UNetMultiChannel` class in `src/model.py`. The U-Net is a convolutional neural network designed for fast and precise semantic segmentation.

## Data Specifications

The model is designed to work with 12-channel data.

| Channel # | Band Name                 | Description                               |
| :-------: | :------------------------ | :---------------------------------------- |
| 1         | Coastal aerosol           | Aerosol detection & atmospheric correction |
| 2         | Blue                      | Visible blue band                         |
| 3         | Green                     | Visible green band                        |
| 4         | Red                       | Visible red band                          |
| 5         | NIR                       | Near-Infrared                             |
| 6         | SWIR1                     | Short-Wave Infrared 1                     |
| 7         | SWIR2                     | Short-Wave Infrared 2                     |
| 8         | QA Band                   | Quality assessment band                   |
| 9         | Merit DEM                 | Digital Elevation Model                   |
| 10        | Copernicus DEM            | Copernicus Digital Elevation Model        |
| 11        | ESA world cover map       | Land cover classification map             |
| 12        | Water occurrence probability | Historical water presence probability     |


![Data Structure Diagram](./data/WhatsApp%20Image%202025-07-13%20at%2021.34.01_117d2a2b.jpg)

_Figure21: The 12 input channels from Harmonized Sentinel-2/Landsat data and the binary water mask._

## Evaluation Metrics

Model performance is assessed using the following standard segmentation metrics:

| Metric          | Description                                               |
| :-------------- | :-------------------------------------------------------- |
| **IoU Score**   | (Intersection over Union) Measures the overlap between predicted and true masks. |
| **Precision**   | Measures the accuracy of positive predictions.            |
| **Recall**      | Measures the model's ability to find all relevant instances. |
| **F1-Score**    | The harmonic mean of Precision and Recall.                |


