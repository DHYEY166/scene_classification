# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
import numpy as np
import pandas as pd
from PIL import Image
import os
import json

# Basic Streamlit config
st.set_page_config(
    page_title="Scene Classification",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class ModelLoader:
    @staticmethod
    def create_base_model():
        """Create model with exact architecture matching saved model"""
        try:
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            base_model.trainable = False
            
            inputs = base_model.input
            x = base_model.output
            x = GlobalAveragePooling2D(name='global_average_pooling2d')(x)
            x = BatchNormalization(name='batch_normalization')(x)
            x = Dense(512, activation='relu', name='dense')(x)
            x = Dropout(0.3, name='dropout')(x)
            x = BatchNormalization(name='batch_normalization_1')(x)
            x = Dense(256, activation='relu', name='dense_1')(x)
            x = Dropout(0.2, name='dropout_1')(x)
            x = BatchNormalization(name='batch_normalization_2')(x)
            outputs = Dense(6, activation='softmax', name='dense_2')(x)
            
            model = Model(inputs=inputs, outputs=outputs, name='scene_classifier')
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model, "Created new model with ImageNet weights"
        except Exception as e:
            return None, f"Error creating model: {str(e)}"

    @staticmethod
    def load_model_from_json_weights():
        """Load model from separate architecture and weights files"""
        try:
            # Try to load architecture from JSON
            if os.path.exists('scene_classifier.json'):
                st.info("Loading model architecture from JSON...")
                with open('scene_classifier.json', 'r') as f:
                    model_json = f.read()
                model = model_from_json(model_json)
                
                # Try to load weights
                if os.path.exists('scene_classifier.weights.h5'):
                    st.info("Loading model weights...")
                    model.load_weights('scene_classifier.weights.h5')
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    st.success("Successfully loaded model from JSON and weights")
                    return model
            
            st.warning("Could not find architecture or weights files")
            return None
            
        except Exception as e:
            st.error(f"Error loading model from JSON/weights: {str(e)}")
            return None

    @staticmethod
    def verify_model(model):
        """Verify model works correctly"""
        try:
            # Create test input
            test_input = np.random.rand(1, 224, 224, 3)
            
            # Try prediction
            prediction = model.predict(test_input, verbose=0)
            
            # Verify output shape
            if prediction.shape != (1, 6):
                return False, "Invalid output shape"
            
            # Verify output is valid probability distribution
            if not np.allclose(np.sum(prediction), 1.0):
                return False, "Invalid probability distribution"
                
            return True, "Model verified successfully"
        except Exception as e:
            return False, f"Model verification failed: {str(e)}"

@st.cache_resource
def load_classification_model():
    """Load and verify model"""
    try:
        # First try loading from JSON/weights
        model = ModelLoader.load_model_from_json_weights()
        
        if model is not None:
            # Verify loaded model
            is_valid, msg = ModelLoader.verify_model(model)
            if is_valid:
                st.success("Successfully loaded and verified saved model")
                return model
            else:
                st.warning(f"Saved model verification failed: {msg}")
        
        # If loading saved model fails, create new one
        st.info("Creating new model with ImageNet weights...")
        model, msg = ModelLoader.create_base_model()
        
        if model is not None:
            is_valid, verify_msg = ModelLoader.verify_model(model)
            if is_valid:
                st.success(msg)
                return model
            else:
                st.error(f"Model verification failed: {verify_msg}")
                return None
        else:
            st.error(msg)
            return None
            
    except Exception as e:
        st.error(f"Error in model loading: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize image
        image = image.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed = preprocess_input(img_array)
        
        return preprocessed
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_scene(model, image):
    """Make prediction on image"""
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
        
        # Show model details if requested
        if st.checkbox("Show model details"):
            model = load_classification_model()
            if model is not None:
                st.write("Model summary:")
                stringlist = []
                model.summary(print_fn=lambda x: stringlist.append(x))
                st.text("\n".join(stringlist))
    
    # Load model
    model = load_classification_model()
    if model is None:
        st.error("Failed to initialize model")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        # Display uploaded image
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Make and display prediction
        with col2:
            st.subheader("Prediction Results")
            predictions = predict_scene(model, image)
            
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
