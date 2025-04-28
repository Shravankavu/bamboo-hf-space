from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Load the Keras model from .h5 file
model = tf.keras.models.load_model("bamboo_modelFinal.h5")

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))  # Matches ResNet50 input
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        predicted_class_index = int(np.argmax(predictions[0]))
        return jsonify({"class_index": predicted_class_index})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
