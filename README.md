# ResNet18_DL_MiniProject

In this project, we explore the impact of tuning hyperparameters, using data augmentations, and changing optimizers on ResNet-18(reduced number of parameters by tuning block size) training for achieving a 90% test accuracy on the CIFAR-10 dataset. We experimented with different combinations of hyperparameters and data augmentations, along with various optimizers, such as SGD and Adam, to find the optimal configuration that yields the highest accuracy. Our results demonstrate that tuning hyperparameters and using data augmentations significantly improve the model's accuracy. Additionally, changing optimizers can also have a significant impact on the performance of ResNet-18 on CIFAR-10. We provide insights into best practices for ResNet-18 training on CIFAR-10 and achieve our goal of 90% test accuracy.

## Dataset:
The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck

The dataset is split into 50,000 training images and 10,000 testing images. The training set contains exactly 5,000 images randomly selected from each class, while the test set contains the remaining images. The images in the dataset were collected by the Canadian Institute for Advanced Research (CIFAR).

The CIFAR-10 dataset is commonly used in machine learning research for image classification tasks, and has become a standard benchmark for evaluating the performance of new algorithms.

## Model Architecture:

<img width="1108" alt="image" src="https://user-images.githubusercontent.com/64778259/232172272-10b269d8-3940-47c9-a459-62fee275a719.png">


<img width="777" alt="image" src="https://user-images.githubusercontent.com/64778259/232172261-47697b75-eaf4-4ffc-b642-11fcfbdacbb5.png">


## Usage
The `resnet18Model.ipynb` file can run locally through Jupyter Notebook or Google Colab.
The `resnet18Model.py` file can be run using the command `python resnet18Model.py`

Requirements:

Run the below command in the terminal to install dependencies.

`pip install -r requirements.txt`


## Performance and Results:

| Optimizer	| Test Accuracy |	
------------|--------------------------|
| SGD |	87 |
| RMSProp |	89.5 |	
| AdaDelta |	77.7 |	
| AdaGrad |	86.5 |	
| Adam F1 Score |	93 |	

| Learning Rate (Adam) |	Test Accuracy |
----------|--------------------------|
| 0.0008 |	90.7 |
| 0.001 |	93 |	
| 0.0012 |	84.6 |
| 0.01 |	70.7 |

<img width="575" alt="image" src="https://user-images.githubusercontent.com/64778259/232172310-e4482a84-56aa-47c9-925e-cbc14b828790.png">

    
<img width="557" alt="image" src="https://user-images.githubusercontent.com/64778259/232172417-dcb9b61e-a3b9-420d-a293-aef5c212eb7d.png">
