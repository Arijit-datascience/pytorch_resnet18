# CIFAR10 image recognition using ResNet-18 architecture

Let's look into some more advanced concepts.

## Learning Time

### Grad-CAM: Gradient-weighted Class Activation Mapping
Convolutional Neural Network (CNN)-based models can be made more transparent by visualizing the regions of input that are "important" for predictions from these models - or visual explanations. Gradient-weighted Class Activation Mapping (Grad-CAM), uses the class-specific gradient information flowing into the final convolutional layer of a CNN to produce a coarse localization map of the important regions in the image.

![image](/Output%20Images%20and%20Logs/gradcam_flow.JPG)

Gradient-weighted Class Activation Mapping (GradCAM) uses the gradients of any target concept (say logits for 'dog' or even a caption), flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept. We take the final convolutional feature map, and then we weigh every channel in that feature with the gradient of the class with respect to the channel. It tells us how intensely the input image activates different channels by how important each channel is with regard to the class. It does not require any re-training or change in the existing architecture.


## Objective

* [x] Train for 40 Epochs
* [x] Display 20 misclassified images
* [x] Display 20 GradCam output on the SAME misclassified images
* [x] Apply the following transforms while training:
  * [x] RandomCrop(32, padding=4)
  * [x] CutOut(16x16)
  * [x] Rotate(±5°)
* [x] Must use ReduceLROnPlateau
* [x] Must use LayerNormalization ONLY

## Results

  1. Model: ResNet18
  2. Total Train data: 60,000 | Total Test Data: 10,000
  3. Total Parameters: 11,173,962
  4. Test Accuracy: 90.03%
  5. Epochs: Run till 40 epochs
  6. Normalization: Layer Normalization
  7. Regularization: L2 with factor 0.0001
  8. Optimizer: Adam with learning rate 0.001
  9. Loss criterion: Cross Entropy
  10. Scheduler: ReduceLROnPlateau
  11. Albumentations: 
      1. RandomCrop(32, padding=4)
      2. CutOut(16x16)
      3. Rotate(5 degree)
      4. CoarseDropout
      5. Normalization 
   12. Misclassified Images: 1104 images were misclassified out of 10,000

## Code Structure

* [resnet.py](https://github.com/Arijit-datascience/pytorch_cifar10/blob/main/model/resnet.py): This describes the ResNet-18 architecture with Layer Normalization  
<i>Referrence: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py</i>  

* [utils](https://github.com/Arijit-datascience/pytorch_cifar10/blob/main/utils/utils.py): Utils code contains the following components:-  
  1. Data Loaders  
  2. Albumentations  
  3. Accuracy Plots
  4. Misclassification Image Plots
  5. Seed

* [main.py](https://github.com/Arijit-datascience/pytorch_cifar10/blob/main/main.py): Main code contains the following functions:-  
  1. Train code
  2. Test code
  3. Main function for training and testing the model  

* [Colab file](/pytorch_cifar10_resnet.ipynb): The Google Colab file contains the following steps:-  
  1. Cloning the GIT Repository
  2. Loading data calling the data loader function from utils file
  3. Model Summary
  4. Running the model calling the main file
  5. Plotting Accuracy Plots
  6. Plotting 20 Misclassification Images
  7. Plotting the Gradcam for same 20 misclassified images

## Model Summary
![image](/images_and_logs/model_summary.JPG)

## Plots

  1. Train & Test Loss, Train & Test Accuracy  
  ![image](/images_and_logs/accuracy_and_loss.png)  

  2. Misclassified Images  
  ![image](/images_and_logs/misclassified_images.png)  

  3. Gradcam Images  
  ![image](/images_and_logs/gradcam.png)  

## Collaborators
Abhiram Gurijala  
Arijit Ganguly  
Rohin Sequeira
