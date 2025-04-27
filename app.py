from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow.lite as tflite  # Revert to TensorFlow Lite
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load your TFLite model
interpreter = tflite.Interpreter(model_path="bamboo_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get the image from the request
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        image_file = request.files['image']

        # Preprocess the image
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Shape: [1, 224, 224, 3]

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = int(np.argmax(output_data[0]))

        return jsonify({"class_index": predicted_class_index})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)  # Hugging Face Spaces uses port 8000 by default