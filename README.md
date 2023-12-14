## MNIST Fashion Image Classification with MLflow

This project implements a convolutional neural network (CNN) for classifying images from the Fashion MNIST dataset. The Fashion MNIST dataset has been expanded to include 70,000 28x28 grayscale images across 10 fashion categories, with 7,000 images per category. The primary objective is to train a CNN model capable of accurately predicting the category of a given image.

In addition to the model development, this project integrates MLflow for experiment tracking and model serving. MLflow provides a centralized platform for managing the end-to-end machine learning lifecycle, enabling seamless experimentation, reproducibility, and deployment.

#### Conceptual diagram as illustrated below.
![image_classification_concept](https://github.com/1010sb/ImageClass/assets/96765388/57c951a5-5e49-4c1f-90ba-076aff088c3a)

### Getting Started
1. Clone this repository
2. Navigate to the project directory
3. Install dependencies using requirements.txt
4. Run the Flask app
  - python app2.py
5. Load test images to the Flask endpoint
  - http://127.0.0.1:5000
6. Retrieve the CSV file
  - The Flask app will process the images and generate a CSV file with label classes and prediction percentages. The file will be available in the project directory 


#### License
This project is licensed under the MIT License.
