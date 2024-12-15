# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
import numpy as np
import pandas as pd
from PIL import Image
import os

# Basic Streamlit config
st.set_page_config(
    page_title="Scene Classification",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class ModelBuilder:
    @staticmethod
    def build_model():
        """Build model with fixed architecture"""
        try:
            # Load VGG16 base
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            base_model.trainable = False
            
            # Build model
            inputs = Input(shape=(224, 224, 3))
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D(name='global_average_pooling2d')(x)
            x = BatchNormalization(name='batch_normalization')(x)
            x = Dense(512, activation='relu', name='dense')(x)
            x = Dropout(0.3, name='dropout')(x)
            x = BatchNormalization(name='batch_normalization_1')(x)
            x = Dense(256, activation='relu', name='dense_1')(x)
            x = Dropout(0.2, name='dropout_1')(x)
            x = BatchNormalization(name='batch_normalization_2')(x)
            outputs = Dense(6, activation='softmax', name='predictions')(x)
            
            model = Model(inputs=inputs, outputs=outputs, name='scene_classifier')
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        except Exception as e:
            st.error(f"Error building model: {str(e)}")
            return None

@st.cache_resource
def load_model():
    """Load or create model"""
    try:
        model = ModelBuilder.build_model()
        if model is None:
            return None
            
        # Try to load saved weights
        weights_path = 'scene_classifier_full.weights.h5'  # Updated path
        if os.path.exists(weights_path):
            try:
                st.info("Loading saved weights...")
                model.load_weights(weights_path, by_name=True)
                st.success("Successfully loaded saved weights")
            except Exception as e:
                st.warning(f"Could not load weights: {str(e)}")
                st.info("Using ImageNet weights")
        else:
            st.info("No saved weights found. Using ImageNet weights")
            
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize
        image = image.resize((224, 224))
        
        # Convert to array
        img_array = np.array(image)
        
        # Add batch dimension and preprocess
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed = preprocess_input(img_array)
        
        return preprocessed
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict(model, image):
    """Make prediction"""
    try:
        # Class names
        classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        
        # Preprocess image
        preprocessed = preprocess_image(image)
        if preprocessed is None:
            return None
            
        # Make prediction
        predictions = model.predict(preprocessed, verbose=0)
        
        # Create results dictionary
        results = {
            class_name: float(pred)
            for class_name, pred in zip(classes, predictions[0])
        }
        
        return results
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    # App title and description
    st.title("Scene Classification using VGG16")
    st.write("Upload an image to classify scenes into: buildings, forest, glacier, mountain, sea, or street")
    
    # Debug info in sidebar
    with st.sidebar:
        st.subheader("Debug Information")
        st.write(f"TensorFlow version: {tf.__version__}")
        
        # Model details toggle
        if st.checkbox("Show model details"):
            model = load_model()
            if model is not None:
                stringlist = []
                model.summary(print_fn=lambda x: stringlist.append(x))
                st.text("\n".join(stringlist))
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to initialize model")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Create columns
        col1, col2 = st.columns(2)
        
        # Display uploaded image
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Make prediction
        with col2:
            st.subheader("Prediction Results")
            predictions = predict(model, image)
            
            if predictions:
                # Get top prediction
                top_class = max(predictions.items(), key=lambda x: x[1])
                
                # Display results
                st.markdown(f"### Predicted Class: **{top_class[0].title()}**")
                st.markdown(f"### Confidence: **{top_class[1]:.2%}**")
                
                # Create DataFrame for visualization
                df = pd.DataFrame(
                    list(predictions.items()),
                    columns=['Class', 'Probability']
                )
                df = df.sort_values('Probability', ascending=True)
                
                # Show probabilities
                st.subheader("Class Probabilities")
                st.bar_chart(df.set_index('Class'))

if __name__ == "__main__":
    main()
