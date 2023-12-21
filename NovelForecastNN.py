#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras_nlp
import tensorflow as tf
import numpy as np

import collections
import pathlib

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization

keras.utils.set_random_seed(13)


# In[ ]:


BATCH_SIZE = 50
EPOCHS = 15
VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH=512
EMBED_DIM = 128


# In[ ]:


# Reviews supposed to be located in the same directory separated in two folders: Train and Test
# Structure of those folders is the same: each label for each sub-folder (1,2,3,4,5)
train_ds = keras.utils.text_dataset_from_directory(
    ".\Reviews\\Train",
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="training",
    seed=13,
)


# In[ ]:


val_ds = keras.utils.text_dataset_from_directory(
    ".\Reviews\\Train",
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="validation",
    seed=13,
)


# In[ ]:


test_ds = keras.utils.text_dataset_from_directory(".\Reviews\\Test", batch_size=BATCH_SIZE)


# In[ ]:


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    stripped_extra = tf.strings.regex_replace(stripped_html, '[image error]', '')
    return tf.strings.regex_replace(stripped_extra, '\r\n', '')


# In[ ]:


train_ds = train_ds.map(lambda x, y: (custom_standardization(x), y))
val_ds = val_ds.map(lambda x, y: (custom_standardization(x), y))
test_ds = test_ds.map(lambda x, y: (custom_standardization(x), y))


# In[ ]:


train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# In[ ]:


int_vectorize_layer = TextVectorization(max_tokens=VOCAB_SIZE, ngrams=2, split='whitespace', output_mode='int',
                                        output_sequence_length=MAX_SEQUENCE_LENGTH, standardize='strip_punctuation')


# In[ ]:


train_text = train_ds.map(lambda text, labels: text)
int_vectorize_layer.adapt(train_text)


# In[ ]:


def create_model(vocab_size, num_labels, vectorizer):
    my_layers = [tf.keras.Input(shape=(1,), dtype=tf.string), vectorizer]

    my_layers.extend([
        layers.Embedding(vocab_size + 1, EMBED_DIM),
        layers.Dropout(0.2)])
    
    for _ in range(num_labels-1):
        my_layers.extend([layers.Dense(units=64, activation='relu')])
        my_layers.extend([layers.Dropout(rate=0.2)])
    
    my_layers.extend([
        layers.Conv1D(32, num_labels, padding="valid", activation="relu", strides=2),
        layers.GlobalMaxPooling1D(),
        layers.Dense(32, activation='softmax')
    ])
    
    model = tf.keras.Sequential(my_layers)
    return model


# In[ ]:


int_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=5, vectorizer=int_vectorize_layer)


# In[ ]:


int_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# In[ ]:


int_history = int_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
int_model.save('novel-forecast.keras')


# In[ ]:


loss = plt.plot(int_history.epoch, int_history.history['loss'], label='int-loss')
plt.plot(int_history.epoch, int_history.history['val_loss'], '--', color=loss[0].get_color(), label='int-val_loss')

plt.legend()
plt.xlabel('Epoch')
plt.ylabel('CE/token')
plt.show()


# In[ ]:


int_model.evaluate(test_ds)

