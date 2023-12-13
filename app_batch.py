from flask import Flask, render_template, request, jsonify
import os
import io
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
import mlflow

# Load the saved model using the model ID (run ID)
model_id = "a11ef41c562e42a895b00b732a0e930c"
model_path = "models/" + model_id
loaded_model = tf.keras.models.load_model(model_path)

# Function to perform individual prediction
def perform_prediction(file_content):
    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(io.BytesIO(file_content), target_size=(28, 28), color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)  # Add batch and channel dimensions

    # Perform prediction
    prediction = loaded_model.predict(img_array)
    predicted_label = np.argmax(prediction[0])
    predicted_probabilities = tf.nn.softmax(prediction).numpy()[0]  # Apply softmax

    # Get the index of the maximum probability
    max_prob_index = np.argmax(predicted_probabilities)

    # Define class labels
    class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    predicted_class = class_labels[predicted_label]

    # Return the prediction result as a dictionary
    result = {
        "filename": "",
        "predicted_class": predicted_class,
        "predicted_probability": int(predicted_probabilities[max_prob_index] * 100)  # Convert to percentage
    }

    return result

# Function to perform batch prediction on images in a folder
def batch_predict_images(folder_path, batch_size=100):
    results = []
    num_processed = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            with open(file_path, 'rb') as file:
                file_content = file.read()
                result = perform_prediction(file_content)
                result['filename'] = secure_filename(filename)
                results.append(result)

            num_processed += 1

            # Check if a batch is complete
            if num_processed % batch_size == 0:
                print(f"Processed {num_processed} images")
                results = []  # Reset results for the next batch

    # Check if there are remaining results after the last batch
    if results:
        print(f"Processed {num_processed} images")
        # Save the remaining results to a CSV file
        results_df = pd.DataFrame(results)
        # Generate a unique filename based on the current date and time
        now = datetime.now()
        date_part = now.strftime("%Y%m%d")
        time_part = now.strftime("%H%M%S")
        filename = f"batch_result_{date_part}_{time_part}.csv"
        results_df.to_csv(filename, index=False)

    print("Batch processing completed.")

if __name__ == '__main__':
    # Provide the path to the folder containing images
    images_folder_path = r'C:\Users\sahma\Desktop\Model_to_Production\image_classification\test_images_1'

    # Perform batch prediction
    batch_predict_images(images_folder_path)
