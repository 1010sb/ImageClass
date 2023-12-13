from flask import Flask, render_template, request, jsonify
import os
import io
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
import mlflow

app = Flask(__name__)

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

# Function to perform batch prediction and save results to CSV
def perform_batch_prediction(folder_path):
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "rb") as f:
                file_content = f.read()

            result = perform_prediction(file_content)
            result['filename'] = secure_filename(filename)
            results.append(result)

    # Save results to CSV
    date_part = datetime.now().strftime("%Y%m%d")
    time_part = datetime.now().strftime("%H%M%S")
    filename = f"batch_result_{date_part}_{time_part}.csv"
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

    return filename

# Define a route for individual prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_content = uploaded_file.read()
            result = perform_prediction(file_content)
            result['filename'] = secure_filename(uploaded_file.filename)
            return jsonify(result)

# Define a route for batch prediction
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if request.method == 'POST':
        results = []
        files = request.files.getlist('file')

        for uploaded_file in files:
            if uploaded_file.filename != '':
                file_content = uploaded_file.read()
                result = perform_prediction(file_content)
                result['filename'] = secure_filename(uploaded_file.filename)
                results.append(result)

        return jsonify(results)

# Define a route to save batch results to CSV
@app.route('/save_batch_results', methods=['POST'])
def save_batch_results():
    if request.method == 'POST':
        results = request.get_json()
        df = pd.DataFrame(results)

        # Generate a unique filename based on the current date and time
        now = datetime.now()
        date_part = now.strftime("%Y%m%d")
        time_part = now.strftime("%H%M%S")
        filename = f"batch_result_{date_part}_{time_part}.csv"

        # Save the DataFrame to a CSV file in the main folder
        df.to_csv(filename, index=False)

        return jsonify({"message": f"Batch results saved to {filename}"})

# Define a route for automated batch prediction (scheduled task)
@app.route('/automated_batch_predict', methods=['GET'])
def automated_batch_predict():
    folder_path = "C:\\Users\\sahma\\Desktop\\Model_to_Production\\image_classification\\test_images_1"
    result_file = perform_batch_prediction(folder_path)

    return jsonify({"message": f"Automated batch prediction completed. Results saved to {result_file}"})

# Define a route for the root URL
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
