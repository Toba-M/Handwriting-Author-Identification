import csv
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, redirect
import os
from werkzeug.utils import secure_filename
import joblib
from keras.models import load_model
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage import data




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

label_encoder = joblib.load("C:/Users/PC/Desktop/PROJECT/deploy/label_encoder.pkl")
scaler = joblib.load("C:/Users/PC/Desktop/PROJECT/deploy/scaler.pkl")
model = load_model("C:/Users/PC/Desktop/PROJECT/deploy/handwriting_recognition_model.keras")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename})

def apply_gabor_filter(image, ksize, sigma, theta, lambd, gamma):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(image,cv2.CV_32F, kernel)
    return filtered_image

def extract_glcms(image):
    image = np.uint8(image)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    return np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])

def preprocess_image(file_path):
    img_array = np.array(Image.open(file_path))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img_size = 850
    new_array = cv2.resize(img_array, (img_size, img_size))
    img_gray = cv2.adaptiveThreshold(new_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 11)
    bilateral_filtered = cv2.bilateralFilter(img_gray, 5, 6, 6)
    gaussian_blur = cv2.GaussianBlur(bilateral_filtered, (7, 7), 2)
    new_image = cv2.addWeighted(bilateral_filtered, 1.5, gaussian_blur, -0.5, 0)
    ksize = 30
    sigma = 0.2
    theta = 0
    lambd = 10
    gamma = 0.5
    gabor_filtered = apply_gabor_filter(new_image, ksize, sigma, theta, lambd, gamma)
    gabor_filtered = np.expand_dims(gabor_filtered, axis=-1)
    hpp = horizontal_projection_profile(gabor_filtered)
    word_line_spaces = word_line_space_formation(hpp)
    padded_image = image_padding(gabor_filtered, target_size=(650, 650))
    normalized_blocks = block_normalization(padded_image, block_size=(650, 650))
    reshaped_blocks = normalized_blocks.reshape(normalized_blocks.shape[1], normalized_blocks.shape[2], normalized_blocks.shape[0]).squeeze()
    glcm_features = extract_glcms(reshaped_blocks)
    return glcm_features

def horizontal_projection_profile(image):
    hpp = np.sum(image, axis=1)
    return hpp

def word_line_space_formation(hpp):
    threshold = np.mean(hpp) * 0.1
    space_indices = np.where(hpp < threshold)[0]
    word_line_spaces = np.diff(space_indices)
    return word_line_spaces

def image_padding(image, target_size=(650, 650)):
    h, w, c = image.shape
    padded_image = np.zeros((target_size[0], target_size[1], c))
    padded_image[:min(h, target_size[0]), :min(w, target_size[1])] = image[:min(h, target_size[0]), :min(w, target_size[1])]
    return padded_image

def block_normalization(image, block_size=(650, 650)):
    normalized_blocks = []
    for i in range(0, image.shape[0], block_size[0]):
        for j in range(0, image.shape[1], block_size[1]):
            block = image[i:i+block_size[0], j:j+block_size[1]]
            block_mean = np.mean(block)
            block_std = np.std(block)
            normalized_block = (block - block_mean) / (block_std + 1e-5)
            normalized_blocks.append(normalized_block)
    return np.array(normalized_blocks)

def make_prediction(img_path, model, scaler, threshold=0.8):
    features = preprocess_image(img_path)
    features_normalized = scaler.transform([features])
    prediction = model.predict(features_normalized.reshape(1, 20, 1))
    predicted_class = np.argmax(prediction)
    confidence = float(prediction[0][predicted_class])
    if confidence < threshold:
        return "unrecognized", confidence
    else:
        return predicted_class, confidence

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        predicted_class, confidence = make_prediction(file_path, model, scaler, threshold=0.8)
        if predicted_class == "unrecognized":
            return jsonify({'prediction': predicted_class, 'confidence': confidence}), 200
        predicted_author = label_encoder.inverse_transform([predicted_class])[0]
        return jsonify({'prediction': str(predicted_author), 'confidence': confidence}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400


@app.route('/start')
def start():
    return render_template('start.html')

@app.route('/files', methods=['POST'])
def files():
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    message = request.form.get('message')


    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'contact_form_data.csv')

    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([name, email, phone, message])
    return redirect(url_for('file3'))

@app.route('/file3')
def file3():
    return render_template('file3.html')

if __name__ == '__main__':
    
    app.run(host="0.0.0.0", port=8000, debug=True)