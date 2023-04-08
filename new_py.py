import cv2
import numpy as np
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from flask import Flask, request, jsonify, render_template


# Load the trained machine learning model
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

# Create the Flask app
app = Flask(__name__)

# Define the home page route
@app.route('/')
def home():
    return render_template('home.html')

# Define the pneumonia detection endpoint
@app.route('/detect_pneumonia', methods=['POST'])
def detect_pneumonia():
    # Read in the image data from the request
    image_data = request.files['image'].read()

    # Convert the image data to a NumPy array
    image_array = np.frombuffer(image_data, np.uint8)

    # Decode the image array into an OpenCV image
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image data
    image = cv2.resize(image, (224, 224)) / 255.0
    image = np.expand_dims(image, axis=0)

    # Pass the preprocessed image data to the model to get a prediction
    prediction = model.predict(image)

    # Return the prediction to the client
    return jsonify({'prediction': str(prediction)})

# Run the Flask app
if __name__ == '_main_':
    app.run(debug=True)