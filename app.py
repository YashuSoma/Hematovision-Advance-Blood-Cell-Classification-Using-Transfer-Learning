import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import io
import base64

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Rice classes (matching the training order)
RICE_CLASSES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Load the trained model
MODEL_PATH = 'rice_classification_model.h5'
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            # Load model with custom objects for TensorFlow Hub
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'KerasLayer': hub.KerasLayer}
            )
            print("✅ TensorFlow Hub model loaded successfully!")
        else:
            print(f"❌ Model file {MODEL_PATH} not found. Please train the model first.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    """Preprocess image for model prediction (matching training preprocessing)"""
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Resize image to 224x224 (MobileNetV2 input size)
    img_resized = cv2.resize(img_cv, (224, 224))
    
    # Convert back to RGB and normalize
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = np.array(img_rgb)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

def predict_rice_type(img):
    """Predict rice type from image using TensorFlow Hub model"""
    if model is None:
        return None, None
    
    try:
        # Preprocess image
        processed_img = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_class = RICE_CLASSES[predicted_class_idx]
        
        return predicted_class, confidence
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Read image from memory
            img = Image.open(io.BytesIO(file.read()))
            
            # Convert image to base64 for display
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Make prediction
            predicted_class, confidence = predict_rice_type(img)
            
            if predicted_class is None:
                flash('Error in prediction. Please try again.')
                return redirect(url_for('index'))
            
            # Get all class probabilities for detailed results
            processed_img = preprocess_image(img)
            all_predictions = model.predict(processed_img)[0]
            
            # Create results dictionary
            results = {}
            for i, class_name in enumerate(RICE_CLASSES):
                results[class_name] = float(all_predictions[i])
            
            return render_template('result.html', 
                                 predicted_class=predicted_class,
                                 confidence=confidence,
                                 results=results,
                                 image_data=img_base64)
            
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
