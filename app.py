import os
import numpy as np
from flask import Flask, render_template, request
from keras.preprocessing import image
from keras.models import load_model
import csv
from datetime import datetime
from collections import Counter

app = Flask(__name__)

# Load the trained model
model_path = '/Users/harshkesarwani/Desktop/MajorProject/model.keras'
model = load_model(model_path)

# Disease class labels
disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
                 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# Directory for storing uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, color_mode='rgb', target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    return x

# Function to make prediction
def predict_disease(image_path):
    x = preprocess_image(image_path)
    prediction = model.predict(x)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class = disease_class[predicted_class_index]
    return predicted_class

# Function to save prediction information to CSV file
def save_prediction_info(image_path, predicted_class):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('historical_data.csv', mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([image_path, predicted_class, timestamp])

# Function to get most common diseases in a given month
def get_most_common_diseases(month, year):
    with open('historical_data.csv', mode='r') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    predictions = [(row[1], datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")) for row in data]
    filtered_predictions = [(disease, timestamp) for disease, timestamp in predictions if timestamp.month == month and timestamp.year == year]
    counter = Counter(disease for disease, _ in filtered_predictions)
    most_common = counter.most_common()
    return most_common

# Route to upload an image
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', message='No image selected')
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            prediction = predict_disease(filename)
            save_prediction_info(filename, prediction)
            current_month = datetime.now().month
            current_year = datetime.now().year
            most_common_disease = get_most_common_diseases(current_month, current_year)[0][0]  # Get the most common disease
            return render_template('result.html', prediction=prediction, image=filename, most_common_disease=most_common_disease)
    return render_template('upload.html')

# Route to display most common diseases in current month
@app.route('/common_diseases')
def display_common_diseases():
    current_month = datetime.now().month
    current_year = datetime.now().year
    most_common_diseases = get_most_common_diseases(current_month, current_year)
    return render_template('common_diseases.html', most_common_diseases=most_common_diseases)

if __name__ == '__main__':
    app.run(debug=True)
