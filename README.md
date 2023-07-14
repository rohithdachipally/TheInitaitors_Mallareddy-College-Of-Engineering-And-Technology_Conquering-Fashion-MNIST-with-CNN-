# CONQUERING  FASHION MNIST WITH CNNs USING COMPUTER VISION


Fashion MNIST is a widely used benchmark dataset in the field of computer vision, specifically for image classification tasks. It serves as a replacement for the traditional MNIST dataset, which consists of grayscale images of handwritten digits. The FashionMNIST dataset consists of 60,000 training images and 10,000 testing images, divided into 10 different categories of fashion items. This abstract focuses on the application of Convolutional Neural Networks (CNNs) for the classification of FashionMNIST dataset.


In this study, we propose a CNN architecture to classify the FashionMNIST dataset. The proposed model consists of multiple convolutional layers, pooling layers, and fully connected layers. We utilize the rectified linear activation function (ReLU) for the convolutional layers and softmax activation for the final output layer to obtain class probabilities.


To train and evaluate the CNN model, we split the FashionMNIST dataset into training and testing sets. We preprocess the images by normalizing pixel values and converting them into a suitable format for training the CNN. The model is trained using the training set, and its performance is evaluated using the testing set. We utilize popular optimization techniques such as stochastic gradient descent (SGD) with backpropagation to update the model weights and minimize the loss function during training.


The application of CNNs on the FashionMNIST dataset not only provides accurate classification of fashion items but also opens doors for various practical applications such as image search, recommendation systems, and virtual try-on experiences in the fashion industry. The proposed CNN architecture serves as a foundation for further research and exploration in the field of computer vision and deep learning, aiming to improve the accuracy and efficiency of image classification tasks.


Convolutional Neural Networks (CNN)
A Convolutional Neural Network (CNN/ConvNet) is a specialized type of deep neural network primarily used for analyzing visual data, such as images or 
videos. 


▪ Unlike traditional neural networks that rely on general matrix multiplication, CNNs utilize convolution operations in at least one of their layers.


▪ Convolution, in the context of CNNs, refers to the process of applying a set of filters or kernels to the input image. Each filter detects specific patterns or 
features, such as edges or textures, by convolving it across the image. 

▪ This results in the extraction of relevant features and the creation of feature maps that highlight the presence of these patterns in different regions of the 
image

Computer Vision Tasks

Classification: categorizing an image or an object into predefined classes or categories. The goal is to assign a label or a class to the entire image or a 
specific region within the image.


▪ Localization: It refers to the task of identifying and localizing objects within an image. It involves not only determining the class or category of an object but also providing the precise location or bounding box coordinates that enclose the object.

▪ Object Detection: It combines classification and localization to identify and locate multiple objects within an image. It involves detecting and classifying 
objects while providing bounding box coordinates for each detected object. 



▪ Instance segmentation: It involves identifying and delineating individual objects within an image. It goes a step beyond object detection by not only 
detecting and classifying objects but also providing pixel-level segmentation masks that separate each object instance from its surroundings.





Dataset size is more than 25 mb which can't be uploaded in github
here, we are providing links to download datasets
link: https://www.kaggle.com/datasets/zalando-research/fashionmnist


