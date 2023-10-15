from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model_path = "models/imageclassifier.h5"  # Replace 'your_model_name.h5' with your model's name
model = tf.keras.models.load_model(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction='Error: Image not provided')

        file = request.files['image']
        image = Image.open(file).convert("RGB").resize((256, 256))  # Adjust this size as per your model's input shape
        image_array = np.expand_dims(np.array(image), axis=0) / 255.0

        predictions = model.predict(image_array)
        predicted_class = "Sad" if predictions[0][0] > 0.5 else "Happy"  # Adjust this based on your model's output

        return render_template('index.html', prediction=f'The person in the photo seems {predicted_class}.')

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
