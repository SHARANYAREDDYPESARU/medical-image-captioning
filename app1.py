from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3, preprocess_input 
from PIL import Image
import numpy as np
from flask_cors import CORS
import json
import io

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = load_model('model1006qqqqqq.keras')

# Load the word-to-index and index-to-word mappings
with open('wordtoixqqqq.json', 'r') as f:
    wordtoix = json.load(f)

with open('ixtowordqqqq.json', 'r') as f:
    ixtoword = json.load(f)

base_model = InceptionV3(weights = 'imagenet') 
model1 = Model(base_model.input, base_model.layers[-2].output)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((299, 299))  # Resize to the size expected by the model
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def encode(image): 
	image = preprocess_image(image) 
	vec = model1.predict(image) 
	vec = np.reshape(vec, (vec.shape[1])) 
	return vec 

# Function to generate caption
def generate_caption(image):
    

    # Placeholder: Start the caption generation with "startseq"
    caption = 'startseq'
    
    # Maximum length of caption
    max_length = 31

    # Generate the caption
    for i in range(max_length):
        # Convert caption words to indices
        sequence = [wordtoix[w] for w in caption.split() if w in wordtoix]
        sequence = np.pad(sequence, (0, max_length - len(sequence)), mode='constant')
        sequence = np.array([sequence])

        # Predict the next word
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[str(yhat)]
        
        # Append the predicted word to the caption
        caption += ' ' + word

        # Stop if the end sequence token is predicted
        if word == 'endseq':
            break

    # Remove the start and end sequence tokens from the caption
    caption = caption.split()
    caption = caption[1:-1]
    caption = ' '.join(caption)

    return caption

@app.route('/upload', methods=['POST'])
def upload_image():
    print("image uploaded.....")
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Open the image file
    image = Image.open(io.BytesIO(file.read()))
    print("image is converted into file...")
    # Generate the caption
    image_features = encode(image).reshape(1,2048)
    caption = generate_caption( image_features)

    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)



















"""from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS
import json
import io

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = load_model('model1006qqqqqq.keras')

# Load the word-to-index and index-to-word mappings
with open('wordtoixqqqq.json', 'r') as f:
    wordtoix = json.load(f)

with open('ixtowordqqqq.json', 'r') as f:
    ixtoword = json.load(f)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((299, 299))  # Resize to the size expected by the model
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to generate caption
def generate_caption(image):
    # Preprocess the image
    
    image = preprocess_image(image)

    # Placeholder: Start the caption generation with "startseq"
    caption = 'startseq'
    
    # Maximum length of caption
    max_length = 20

    # Generate the caption
    for i in range(max_length):
        # Convert caption words to indices
        sequence = [wordtoix[w] for w in caption.split() if w in wordtoix]
        sequence = np.pad(sequence, (0, max_length - len(sequence)), mode='constant')
        sequence = np.array([sequence])

        # Predict the next word
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[str(yhat)]
        
        # Append the predicted word to the caption
        caption += ' ' + word

        # Stop if the end sequence token is predicted
        if word == 'endseq':
            break

    # Remove the start and end sequence tokens from the caption
    caption = caption.split()
    caption = caption[1:-1]
    caption = ' '.join(caption)

    return caption

@app.route('/upload', methods=['POST'])
def upload_image():
    print("image uploaded.....")
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Open the image file
    image = Image.open(io.BytesIO(file.read()))
    print("image is converted into file...")
    # Generate the caption
    caption = generate_caption(image)

    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)"""




















"""from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import json

app = Flask(__name__)
CORS(app)

# Define the custom NotEqual layer
class NotEqual(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)

    def call(self, inputs):
        # Assuming inputs is a list of two tensors
        return tf.cast(tf.not_equal(inputs[0], inputs[1]), tf.float32)

# Register the custom layer
custom_objects = {'NotEqual': NotEqual}

# Load the model with custom objects
model = load_model("C:/Users/Public/mernproject2/server/model2504000000.h5", custom_objects=custom_objects)

# Parameters (adjust as needed)
max_length = 34
vocab_size = 5000  # This should match the vocab size used in training

# Load your tokenizer here
with open("C:/Users/Public/mernproject2/server/wordtoix.json") as f:
    wordtoix = json.load(f)
ixtoword = {v: k for k, v in wordtoix.items()}

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))  # Resize as per your model requirements
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize if needed
    return img

def generate_caption(image, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[word] for word in in_text.split() if word in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat, axis=-1)
        word = ixtoword.get(yhat[0], '')
        if word is None or word == '':
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = ' '.join(in_text.split()[1:-1])  # Remove 'startseq' and 'endseq'
    return final_caption

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    file_path = f'./{file.filename}'
    file.save(file_path)

    image = preprocess_image(file_path)
    caption = generate_caption(image, max_length)
    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)"""




    
"""from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM, add
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the deep learning model for image classification
model_path = "C:/Users/Public/mernproject2/server/model (1).h5"
model = tf.keras.models.load_model(model_path)

# Define functions for deep learning predictions
def preprocess_image(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(image):
    predicted_prob = model.predict(image)[0][0]
    predicted_class = 'Normal' if predicted_prob > 0.5 else 'Effusion'
    return predicted_class

# Route for image classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        image = preprocess_image(file_path)
        predicted_class = predict_image(image)
        return jsonify({'predicted_class': predicted_class})

# Define the image captioning model
max_length = 34  # Example value, replace with your actual max_length
vocab_size = 445  # Replace with your actual vocab_size
emb_dim = 200  # Replace with your actual emb_dim

ip1 = Input(shape=(2048,))
fe1 = Dropout(0.2)(ip1)
fe2 = Dense(256, activation='relu')(fe1)

ip2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, emb_dim, mask_zero=True)(ip2)
se2 = Dropout(0.2)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

caption_model = Model(inputs=[ip1, ip2], outputs=outputs)
caption_model.load_weights("C:/Users/Public/mernproject2/server/model2504000000.h5")  # Path to your model weights

# Dummy functions to demonstrate feature extraction and caption generation
def extract_features(image_path):
    # This function should extract features from the image using a pre-trained model
    # Example: using InceptionV3, ResNet, or any other CNN model to get image features
    return np.random.rand(2048)  # Replace with actual feature extraction

def generate_caption(image_features):
    # This function should generate a caption using the LSTM model
    # Placeholder example, replace with your actual implementation
    return "A placeholder caption for the input image."

# Route for image captioning
@app.route('/generate_caption', methods=['POST'])
def generate_caption_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        features = extract_features(file_path)
        caption = generate_caption(features)
        return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)"""
    
