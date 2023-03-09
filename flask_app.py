import cv2
import tensorflow as tf 
import numpy as np
from keras.models import load_model
import sys
from flask import Flask, request, render_template, Response
from streamlit_option_menu import option_menu

#Loading the VGG16 model
model= load_model('model.h5',compile=(False))

#Functions
def predict(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

    # Predict with the VGG16 model
    prediction = model.predict(img)

    # Convert the prediction into text
    pred_text = tf.keras.applications.inception_v3.decode_predictions(prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        label  = ("{}: {:.2f}%".format(label, prob * 100))

    return label


def predict2(frame, model):
    # Pre-process the image for model prediction
    img = cv2.resize(frame, (299, 299))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img /= 255.0

    # Predict with the VGG16 model
    prediction = model.predict(img)

    # Convert the prediction into text
    pred_text = tf.keras.applications.vgg16.decode_predictions(prediction, top=1)
    for (i, (imagenetID, label, prob)) in enumerate(pred_text[0]):
        pred_class = label

    return pred_class

def object_detection(search_key,frame, model):
    label = predict2(frame,model)
    label = label.lower()
    try:
        if label.find(search_key) > -1:
            sys.exit(predict(frame, model))
        else:
            return 'Not found'

    except:
        print('')

# Flask app
app = Flask(__name__)

# Main App
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Check if video file is uploaded
        if 'file' not in request.files:
            return 'No file found'
        file = request.files['file']
        # Check if file is selected
        if file.filename == '':
            return 'No file selected'
        path = file.filename
        file.save(path)

        cap = cv2.VideoCapture(path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

        # Start the video prediction loop
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Perform object detection
            object_detection(request.form.get('search_key', '').lower(), frame, model)

            # Display the resulting frame

        cap.release()
        output.release()
        cv2.destroyAllWindows()

    return 'File uploaded'

if __name__ == '__main__':
    app.run(debug=True)
