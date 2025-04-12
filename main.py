import os
from flask import Flask, request, render_template
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load class names (labels)
class_names = open("labels.txt", "r").readlines()

# Folder to save uploaded images temporarily
UPLOAD_FOLDER = 'static/uploaded_images/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to predict action
def predict_action(image_path):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:], confidence_score

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded file
        file = request.files["image"]
        
        if file:
            # Save the uploaded file
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            
            # Predict the action6
            action, confidence = predict_action(image_path)

            # Render result
            return render_template("index.html", action=action, confidence=confidence*100)
    
    return render_template("index.html", action=None, confidence=None)

if __name__ == "__main__":
    app.run(debug=True)
