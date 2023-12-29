import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Flatten, Dense, LSTM, Dropout, Embedding, Activation
from keras.layers import concatenate, BatchNormalization, Input
from tensorflow.keras.layers import add
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import cv2
import string
import glob
from keras.preprocessing.image import load_img, img_to_array

# Set your local paths for the dataset
images = '/path/to/your/local/flickr_data/Flickr_Data/Images/'
token_path = '/path/to/your/local/flickr_data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'
train_path = '/path/to/your/local/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'

# Function to load image descriptions from text file
def load_description(text):
    mapping = dict()
    for line in text.split("\n"):
        token = line.split("\t")
        if len(line) < 2:
            continue
        img_id = token[0].split('.')[0]
        img_des = token[1]
        if img_id not in mapping:
            mapping[img_id] = list()
        mapping[img_id].append(img_des)
    return mapping

# Function to clean image descriptions
def clean_description(desc):
    for key, des_list in desc.items():
        for i in range(len(des_list)):
            caption = des_list[i]
            caption = [ch for ch in caption if ch not in string.punctuation]
            caption = ''.join(caption)
            caption = caption.split(' ')
            caption = [word.lower() for word in caption if len(word) > 1 and word.isalpha()]
            caption = ' '.join(caption)
            des_list[i] = caption

# Function to create vocabulary from descriptions
def to_vocab(desc):
    words = set()
    for key in desc.keys():
        for line in desc[key]:
            words.update(line.split())
    return words

# Function to load and clean descriptions based on the dataset
def load_clean_descriptions(desc, dataset):
    dataset_des = dict()
    for key, des_list in desc.items():
        if key + '.jpg' in dataset:
            if key not in dataset_des:
                dataset_des[key] = list()
            for line in des_list:
                desc = 'startseq ' + line + ' endseq'
                dataset_des[key].append(desc)
    return dataset_des

# Load descriptions from token file
text = open(token_path, 'r', encoding='utf-8').read()
descriptions = load_description(text)
clean_description(descriptions)
vocab = to_vocab(descriptions)

# Load image paths
img = glob.glob(images + '/*.jpg')

# Load training image paths
train_images = open(train_path, 'r', encoding='utf-8').read().split("\n")
train_img = [im for im in img if im[len(images):] in train_images]

# Load and encode images using InceptionV3
def preprocess_img(img_path):
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess_img(image)
    vec = model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec

base_model = InceptionV3(weights='imagenet')
model = Model(base_model.input, base_model.layers[-2].output)

encoding_train = {}
for img in train_img:
    encoding_train[img[len(images):]] = encode(img)

# Prepare captions for training
all_train_captions = []
for key, des_list in train_descriptions.items():
    for caption in des_list:
        all_train_captions.append(caption)

threshold = 10
word_counts = {}
for cap in all_train_captions:
    for word in cap.split(' '):
        word_counts[word] = word_counts.get(word, 0) + 1

vocab = [word for word in word_counts if word_counts[word] >= threshold]

# Create index mappings for words to integers and vice versa
ixtoword = {i + 1: word for i, word in enumerate(vocab)}
wordtoix = {word: i + 1 for i, word in enumerate(vocab)}

max_length = max(len(desc.split()) for desc in all_train_captions)

# Prepare data for training
X1, X2, y = list(), list(), list()
for key, des_list in train_descriptions.items():
    pic = encoding_train[key + '.jpg']
    for cap in des_list:
        seq = [wordtoix[word] for word in cap.split(' ') if word in wordtoix]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=len(vocab) + 1)[0]
            X1.append(pic)
            X2.append(in_seq)
            y.append(out_seq)

X2 = np.array(X2)
X1 = np.array(X1)
y = np.array(y)

# Prepare word embeddings using GloVe
embeddings_index = {}
glove_path = '/content/glove.6B.200d.txt'  # Provide the correct path to your GloVe file
glove = open(glove_path, 'r', encoding='utf-8').read()
for line in glove.split("\n"):
    values = line.split(" ")
    word = values[0]
    indices = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = indices

emb_dim = 200
vocab_size = len(vocab) + 1  # Add 1 for the zero-padding
emb_matrix = np.zeros((vocab_size, emb_dim))
for word, i in wordtoix.items():
    emb_vec = embeddings_index.get(word)
    if emb_vec is not None:
        emb_matrix[i] = emb_vec

# Define the model architecture
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
model = Model(inputs=[ip1, ip2], outputs=outputs)

# Load pre-trained word embeddings into the model and freeze the embedding layer
model.layers[2].set_weights([emb_matrix])
model.layers[2].trainable = False

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit([X1, X2], y, epochs=50, batch_size=256)

# Function for generating captions using greedy search
def greedy_search(pic): 
    start = 'startseq'
    for i in range(max_length): 
        seq = [wordtoix[word] for word in start.split() if word in wordtoix] 
        seq = pad_sequences([seq], maxlen=max_length) 
        yhat = model.predict([pic, seq]) 
        yhat = np.argmax(yhat) 
        word = ixtoword[yhat] 
        start += ' ' + word 
        if word == 'endseq': 
            break
    final = start.split() 
    final = final[1:-1] 
    final = ' '.join(final) 
    return final