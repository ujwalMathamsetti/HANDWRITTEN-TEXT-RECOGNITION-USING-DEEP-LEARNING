# HANDWRITTEN-TEXT-RECOGNITION-USING-DEEP-LEARNING
Handwritten text recognition (HTR) is a technique for receiving and interpreting handwritten input from sources such as documents, touch screens, photographs, and so on. Handwritten text recognition is a form of pattern recognition. The conventional system for this recognition is based on the characteristics of hand-written work and a vast amount of prior knowledge. In recent years, the focus of research in this field has been on deep learning techniques, which have achieved impressive results. Neural Networks (NN) are highly effective at recognising handwritten structures, such as characters/symbols/numbers and words, that support automatically extracting unique features. The proposed system is used to identify writings in a variety of formats. The EMNIST (Extended Modified National Institute of Standards and Technology database) dataset includes handwritten text such as digits, symbols, and numbers, with a high degree of complexity. The purpose of this study is to compare the accuracy of handwritten text recognition algorithms developed by CNN and ResNet – 50. In addition, some characters may be missed during text recognition due to incorrect detection, which will be minimised. The final step of this paper is to choose the most accurate algorithm which is CNN with a training accuracy of 93.27 per cent and then deployed the CNN model in real-time development using the FLASK framework. Python programming language was utilised for the implementation of both algorithms. 


![image](https://user-images.githubusercontent.com/94397783/169290830-80b286dd-a6ad-497c-ace7-dbd8ccce9214.jpeg)

The block diagram above shows how a deployment model works

•	Initially, required libraries are downloaded and imported for the model

•	Data pre-processing is performed on the input data, iterating over the entire data set with an easy-to-use reshape method on each image

•	Defining the architecture of the model, such as the required number of layers for the model to function properly

•	The subsequent step entails evaluating the model and carrying out the process of fitting the model with the necessary hyperparameters

•	The next step is saving the model that has been executed with the hyperparameters that have been adjusted
 
•	Developing an Application for the Flask to Serve the Model. In the process of developing, an HTML file is required for the user to give an input image

•	Integrating the saved model with the HTML file using APIs (Application Programming Interface) 

•	According to the figure above, when a user selects an image on the web screen and submits it, the image loads in the flask where the saved CNN model was integrated, and the results are predicted on the web screen.

