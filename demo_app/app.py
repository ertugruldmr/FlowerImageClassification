import gradio as gr
import tensorflow as tf
from glob import glob
import numpy as np
import pickle
import json

# loading the files
model_path = "tuned_ResNetV2"
model = tf.keras.models.load_model(model_path)
labels = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

# examples
examples_path = "examples"

def process_image(image):
    # Convert into tensor
    image = tf.convert_to_tensor(image)

    # Cast the image to tf.float32
    image = tf.cast(image, tf.float32)
    
    # Resize the image to img_resize
    image = tf.image.resize(image, (224,224))
    
    # Normalize the image
    image /= 255.0
    
    # Return the processed image and label
    return image

def predict(image):

  # Pre-procesing the data
  images = process_image(image)

  # Batching
  batched_images = tf.expand_dims(images, axis=0)
  
  prediction = model.predict(batched_images).flatten()
  confidences = {labels[i]: float(prediction[i]) for i in range(len(labels))}
  return confidences

# declerating the params
component_params = {
    "fn":predict, 
    "inputs":gr.Image(shape=(224, 224)),
    "outputs":gr.Label(num_top_classes=len(labels)),
    "examples":examples_path,
}

# Instantiating example demo app
demo = gr.Interface(**component_params)
            
# Launching the demo
if __name__ == "__main__":
    demo.launch()
