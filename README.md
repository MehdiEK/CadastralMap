# Automating Cadastral Maps Updates and Application on Undeclared Swimming Pools Detection

## Abstract
This project addresses the challenge of updating cadastral maps by automating the detection of undeclared swimming pools using traditional computer vision techniques on satellite imagery. Our approach demonstrates promising accuracy in recognizing registered land parcels and structures, offering an efficient tool for urban authorities.

## Introduction
Urban planning and regulatory compliance are integral to sustainable development. This project introduces a computer vision-based approach to update cadastral maps and detect undeclared swimming pools using satellite imagery.

## Problem Definition
The problem is defined as the detection of physical structures within satellite images that do not correspond to registered cadastral data, indicative of potential unauthorized constructions.

## Related Work
This section discusses relevant research papers and insights into non-deep learning techniques for automating cadastral map updates and undeclared swimming pool detection.

### Automated Boundary and Feature Detection
- Ragia and Sarri (2019) proposed a low-cost method for cadastral object extraction using uncalibrated cameras.

### Image Processing and Feature Detection
- Crommelinck et al. (2017) investigated the use of gPb contour detection for automated cadastral mapping using UAV imagery.

## Methodology
### Cadastral Matching
#### Pretreatment
The pretreatment process enhances the quality of satellite images for feature extraction and analysis, including steps like Grayscale Conversion, Gaussian Blur, and Edge Detection.

#### Shape Extraction
This phase involves identifying and delineating relevant features from the satellite and cadastral images.

#### Cadastral Matching
This process aligns cadastral plans with satellite images, incorporating feature detection, descriptor extraction, feature matching, and homography estimation.

### Pool/Houses Detection
We approach pool detection based on their color, using the HSV (Hue-Saturation-Value) space and implementing the Ramer-Douglas-Peucker algorithm.

### Fraud Detection Application
The application analyzes discrepancies between cadastral plans and actual land use, employing dynamic map visualization and geospatial fraud analysis.

## Evaluation
The evaluation involved a comparative analysis of our method against a ground truth dataset, assessing accuracy using pixel accuracy and F1-score metrics.

## Conclusions
The project demonstrates the feasibility of using traditional computer vision techniques for cadastral map updating and unauthorized structure detection, suggesting a cost-effective alternative to deep learning models.

## Figures
Figures referenced in the report can be added here using markdown image syntax.

![Figure 1: Example of Cadastral Government Data](Images/Cad.png)
![Figure 2: Example of Satellite Images from Esri Library](Images/Sat.png)
![Figure 3: Example of Fraud Detection Analysis](Images/cat_sat_pool.png)

## References
- Ragia, L., Sarri, F., (2019). Automated Boundary and Feature Detection. 
- Crommelinck, S., et al. (2017). Image Processing and Feature Detection.

