import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
import numpy as np
import pandas as pd
from PIL import Image
import os

# Add test image for verification
test_image = np.random.rand(224, 224, 3)

class ModelLoader:
    @staticmethod
    def verify_model(model):
        """Verify model works correctly"""
        try:
            # Try to make a prediction
            test_input = np.expand_dims(test_image, axis=0)
            prediction = model.predict(test_input, verbose=0)
            
            # Check prediction shape
            if prediction.shape[1] != 6:  # Should have 6 classes
                return False, "Invalid model output shape"
                
            return True, "Model verified successfully"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def create_model():
        """Create new model with ImageNet weights"""
        try:
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            base_model.trainable = False
            
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = BatchNormalization()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = BatchNormalization()(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)
            outputs = Dense(6, activation='softmax')(x)
            
            model = Model(inputs=base_model.input, outputs=outputs)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model, "Created new model with ImageNet weights"
        except Exception as e:
            return None, f"Error creating model: {str(e)}"

    @staticmethod
    def load_saved_model():
        """Try to load saved model"""
        model_files = [
            'best_vgg16.keras',
            'best_vgg16.h5',
            'best_vgg16_weights.h5'
        ]
        
        for file_path in model_files:
            if os.path.exists(file_path):
                try:
                    st.info(f"Attempting to load model from {file_path}...")
                    
                    if file_path.endswith('_weights.h5'):
                        # Load weights into new model
                        model, msg = ModelLoader.create_model()
                        if model is None:
                            continue
                        model.load_weights(file_path)
                    else:
                        # Load complete model
                        model = load_model(file_path)
                    
                    # Verify model
                    is_valid, msg = ModelLoader.verify_model(model)
                    if is_valid:
                        st.success(f"Successfully loaded and verified model from {file_path}")
                        return model
                    else:
                        st.warning(f"Model validation failed: {msg}")
                except Exception as e:
                    st.warning(f"Error loading {file_path}: {str(e)}")
                    continue
        
        # If no saved model works, create new one
        st.warning("No valid saved model found. Creating new model...")
        model, msg = ModelLoader.create_model()
        if model is not None:
            st.success(msg)
        return model

@st.cache_resource
def load_classification_model():
    """Load model with validation"""
    return ModelLoader.load_saved_model()

def preprocess_image(image):
    """Preprocess image for prediction"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_scene(model, image):
    """Make prediction with error handling"""
    try:
        classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        preprocessed = preprocess_image(image)
        if preprocessed is None:
            return None
            
        predictions = model.predict(preprocessed, verbose=0)
        
        # Validate predictions
        if not isinstance(predictions, np.ndarray) or predictions.shape[1] != len(classes):
            st.error("Invalid prediction format")
            return None
            
        results = {
            class_name: float(pred)
            for class_name, pred in zip(classes, predictions[0])
        }
        
        return results
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    st.title("Scene Classification using VGG16")
    st.write("Upload an image to classify scenes into: buildings, forest, glacier, mountain, sea, or street")
    
    # Debug info
    with st.sidebar:
        st.subheader("Debug Information")
        st.write(f"TensorFlow version: {tf.__version__}")
        
        # Model info
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
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            predictions = predict_scene(model, image)
            
            if predictions:
                top_class = max(predictions.items(), key=lambda x: x[1])
                st.markdown(f"### Predicted Class: **{top_class[0].title()}**")
                st.markdown(f"### Confidence: **{top_class[1]:.2%}**")
                
                df = pd.DataFrame(
                    list(predictions.items()),
                    columns=['Class', 'Probability']
                )
                df = df.sort_values('Probability', ascending=True)
                
                st.subheader("Class Probabilities")
                st.bar_chart(df.set_index('Class'))

if __name__ == "__main__":
    main()
