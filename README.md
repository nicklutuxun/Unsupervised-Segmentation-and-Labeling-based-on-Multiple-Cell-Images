# Unsupervised Segmentation and Labeling based on Multiple Cell Images

by Chongjun Yang, Jiayue Zhang, Tuxun(Nick) Lu

> This project aims to achieve automated cell classification by processing stained cell data. The main steps are as follows: reconstructing images from stained cell data, pre-processing the images through operations such as histogram adjustment and Gaussian filtering, segmenting the images using the cellpose segmentation method, and employing the ResNet model for deep learning to achieve cell classification.

## 1.Introduction
Cell classification plays a crucial role in various biomedical research and applications. Traditional manual methods for cell classification are time-consuming and subjective. Therefore, there is a growing interest in developing automated techniques that can accurately classify cells.

In this project, we propose a novel approach to cell classification by leveraging deep learning and color analysis. By processing-stained cell data, we aim to automate the cell classification process. The key idea is to harness the power of
deep learning algorithms, specifically the ResNet model, to learn and extract meaningful features from the images.

The proposed methodology involves several steps. Firstly, we reconstruct images from the stained cell data. Then, we apply pre-processing techniques such as histogram adjustment and Gaussian filtering to enhance the image quality and remove noise. Next, we utilize the cellpose segmentation method to segment the images and isolate individual cells. Finally, we employ the ResNet model, a state-of-the-art deep learning architecture, to perform classification based on the extracted features.
By combining deep learning and color analysis, we expect to achieve more accurate and efficient cell classification compared to traditional manual methods. The automated approach has the potential to greatly accelerate research in areas such as pathology, drug discovery, and cell biology.

## 2.Image Preprocess Pre-processing Methods and Results
The preprocessing steps for the experimental data are as follows:

<p align="center">
  <img width="750" src="https://github.com/nicklutuxun/Unsupervised-Segmentation-and-Labeling-based-on-Multiple-Cell-Images/assets/41639441/f574ae21-1c6a-4ac5-a79e-c6372c564727">
</p>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(a)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(b)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(c)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(d)

1. Original Dataset: The original experimental data in (a) consists of 4 sets of datasets. Each dataset contains images from different channels. 
2. Slicing the Data: The first step is to extract the photos from all channels in a sliced form in (b). This involves separating the individual images from each dataset and organizing them based on their respective channels.
3. Reconstructing Color Channels: Next, the individual color channels (Red, Green, Blue, and Cyan) in (b) from the dataset in (a) are reconstructed. This reconstruction as shown in (c) involves combining the corresponding color channels from each dataset based on a specified ratio to create the original experimental color photos.
4. Histogram Adjustment: The reconstructed color channel photos in (c) undergo histogram adjustment. This operation adjusts the intensity distribution of the pixels in the images, enhancing the overall contrast and improving the visibility of details.
5. Gaussian Filtering: Following histogram adjustment, the preprocessed color channel photos are subjected to Gaussian filtering. This process involves applying a Gaussian filter to the images, which helps to reduce noise and blur, further enhancing the quality of the images.
6. Preprocessed Experimental Images: After the histogram adjustment and Gaussian filtering steps, the images as shown in (d) are effectively enhanced and ready for further analysis. The result is a set of preprocessed experimental “*.tif” images that have undergone these enhancement techniques to improve their visual quality and highlight relevant features.

## 3.Image segmentation
### 3.1 Segmentation Methods
Based on the above image pre-processing results, we can find the overlap of multiple cells, the diversity of shapes, and the different shades of colors in images. Therefore, in order to achieve better segmentation, we need to take the above characteristics into account during segmentation. We consider two segmentation methods, one is the classical image segmentation method called watershed, which is often used in cell segmentation. Another is cellpose segmentation based on deep learning.

Watershed segmentation is a classic image segmentation technique that separates objects in an image based on local minima and maxima of the image intensity. The algorithm starts by flooding the image from its regional minima, which creates a set of catchment basins. These basins are then merged based on a user-defined threshold to obtain the final segmentation.

Cellpose Segmentation is a useful deep learning method to accurately segment cells and other biological structures in microscopy images. This method analyzes image pixels and classify them as either foreground or background based on convolutional neural network (CNN). Cellpose uses a deep learning model trained on a large dataset of microscopy images to identify and segment cells or cell nuclei. CNN takes a gray or RGB image as input and outputs a probability map showing how likely each pixel is to belong to an object. Objects are identified based on the threshold applied to the probability map. Once objects are identified, the next step is to segment them by assigning each pixel to a specific object. Cellpose Segmentation is a combination of watershed segmentation and Deep Learning to segment objects accurately and efficiently. Cellpose Segmentation is used in a variety of research areas such as cell biology and cancer research.  Compared with watershed segmentation, cellpose segmentation is able to handle more complex objects and overlapping regions.

### 3.2 Segmentation Result
We first performed watershed segmentation and met the problem of over-segmentation, where the image has been divided into too many small regions. One way to address this is by using a post-processing step called "morphological opening" to smooth out small regions and merge them with neighboring regions. Another way is applying a clustering algorithm to group together similar sub-regions based on some similarity criterion. However, the results were not satisfactory, and a large number of cells were lost. Then, we performed cellpose segmentation on a total of 120 multiple images. We define the Cellpose model and set the diameter parameter and get the results. The segmentation mask images were obtained and the partial results are as shown in figure.

<p align="center">
  <img width="750" alt="image" src="https://github.com/nicklutuxun/Unsupervised-Segmentation-and-Labeling-based-on-Multiple-Cell-Images/assets/41639441/4124d4e6-b4db-4dae-96c8-a7d3a26f6e86">
</p>

From the comparison between the mask image and the original image, the result of cellpose segmentation is satisfactory, and the cells with different shapes, shades of color and connected or partially overlapping cells are also well segmented.

## 4.Cell Classification
### 4.1 Classification Methods
In order to generate a labeled image, we need to first classify all individual cells in the image. As shown in Figure, the previous steps provide the corresponding segmentation mask for all cells in the image, so we are able to crop all individual cell images from all images to create a dataset. These images in the dataset are then used as the input of a convolutional neural network to extract representative features. Specifically, we used a pretrained ResNet152 model that has been proven suitable for feature extraction tasks. The output of ResNet for one cell image is a 1000-dimensional array. Combining all features from all cell images, we input this list of features to the k-means clustering algorithm. The choice of hyperparameter for the number of clusters is determined by specific dataset. For the dataset we use, we set the number of clusters to 11 based on the dataset description. Lastly we overlay a colormap of labels onto the original image based on individual cell labels to generate the final segmented and labeled image.

<p align="center">
  <img width="750" alt="image" src="https://github.com/nicklutuxun/Unsupervised-Segmentation-and-Labeling-based-on-Multiple-Cell-Images/assets/41639441/d81eb878-3205-4f54-b218-432c4d5e7e3e">
</p>

### 4.2 Classification Result
As shown in Figure, our model successfully groups most of the cells with others of the same type. However, for some edge cases where cell pose is different from the majority, the model misclassified. The problem is likely to be related with the generalizability of the pretrained ResNet. The ResNet we used is trained on ImageNet, ImageNet is a large-scale visual recognition challenge that was created to advance the state-of-the-art in image recognition. The ImageNet dataset contains over 14 million images that are classified into thousands of categories. However, cell images rarely appear in the dataset. We are compelled that this pretrained ResNet transfers its feature extraction ability poorly to cell images and, therefore, the extracted feature vector is not representative of the cell's true features.

<p align="center">
  <img width="750" alt="image" src="https://github.com/nicklutuxun/Unsupervised-Segmentation-and-Labeling-based-on-Multiple-Cell-Images/assets/41639441/bb95b522-fd78-4cc4-9f89-464f2719b55f">
</p>
<p align="center">
  <img width="750" alt="image" src="https://github.com/nicklutuxun/Unsupervised-Segmentation-and-Labeling-based-on-Multiple-Cell-Images/assets/41639441/08b703fb-3c53-4432-966a-0e42dd9baf89">
</p>

## 5. Conclusion and Outlook
Our model successfully segments and labels all cells in an image. Unsupervised cell segmentation and labeling is a complex problem that has not received enough attention.

This project could benefit from more improvement. A list of possible future directions are listed as follows:
- Optimize cellpose segmentation with less manual intervention in adjusting parameters and thresholds.
- Pretrained ResNet is trained on ImageNet, which could potentially generalize not so well on cell images. We could find other suitable pretrained models or train our own model if we can obtain the access to labeled data.

## Acknowledgements
We would like to thank Dr. Micheli Mario for his valuable advice, support, and help with our project. We would also like to thank the MEOW Lab for providing us with data for analysis. In addition, we would like to thank TA Oscar Liu for his continuous assistance.

## Contributions:
Chongjun Yang proposed the idea of the project and performed the preprocessing part of this project. Jiayue Zhang performed the segmentation part of this project. Nick Lu performed the classification part of this project. All the members wrote and revised the final report collaboratively. All members have read and approved the final manuscript.

## Conflicts of Interest
The authors declare no conflict of interest.

## References
[1] Chen, B., Kao, E., & Tward, A. D. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1), 100-106.

[2] Angulo, J., Sánchez, J., & García-Sánchez, F. (2019). An overview of watershed segmentation. In Handbook of Mathematical Methods in Imaging (pp. 157-185). Springer, Cham.

[3] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 234-241). Springer, Cham.

[4] Gonzales, R. C., & Wintz, P. (1987). Digital image processing. Addison-Wesley Longman Publishing Co., Inc..



