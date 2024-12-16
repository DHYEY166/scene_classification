import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Scene Classification App",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Headers */
    .main h1 {
        color: #1E3A8A;
        margin-bottom: 2rem;
        text-align: center;
        padding: 1rem;
        border-bottom: 2px solid #E5E7EB;
    }
    
    /* Sidebar */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* Cards */
    .stMarkdown div {
        border-radius: 10px;
    }
    
    .prediction-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .info-box {
        background-color: #EFF6FF;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #BFDBFE;
        margin-bottom: 1rem;
    }
    
    /* File uploader */
    .css-1cpxqw2 {
        border: 2px dashed #CBD5E1;
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #2563EB !important;
    }
    
    /* Bar chart */
    .stChart {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

def create_model():
    """Create identical model architecture"""
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D(name='gap')(x)
    x = BatchNormalization(name='bn_1')(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01), name='dense_1')(x)
    x = Dropout(0.3, name='dropout_1')(x)
    x = BatchNormalization(name='bn_2')(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01), name='dense_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)
    x = BatchNormalization(name='bn_3')(x)
    
    outputs = Dense(6, activation='softmax', kernel_regularizer=l2(0.01), name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        # Create model architecture
        model = create_model()
        model.summary()  # Print summary to verify layer count
        
        # Load weights
        weights_path = 'scene_classifier.weights.h5'
        if not os.path.exists(weights_path):
            st.error(f"Weights not found at: {weights_path}")
            return None
        
        # Load weights without by_name parameter
        model.load_weights(weights_path)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        st.success("Model loaded successfully!")
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
        
        # Resize to match training dimensions
        image = image.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        processed_img = preprocess_input(img_array)
        
        return processed_img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_scene(model, image):
    """Make prediction on image"""
    try:
        # Class names must match training order
        classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        
        # Preprocess image
        processed_img = preprocess_image(image)
        if processed_img is None:
            return None
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        
        # Create results dictionary
        results = {
            class_name: float(prob)
            for class_name, prob in zip(classes, predictions[0])
        }
        
        return results
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    # Page title and description
    st.markdown("<h1>🏞️ Scene Classification AI</h1>", unsafe_allow_html=True)
    
    # Add sidebar with info
    with st.sidebar:
        st.markdown("### 📋 About")
        st.markdown("""
        <div class='info-box'>
        This AI-powered application uses advanced deep learning to classify scenes 
        into six different categories. Perfect for photographers, researchers, 
        and nature enthusiasts.
        
        Upload any scene image to get instant classification results!
        </div>
        """, unsafe_allow_html=True)
        
        # Model details
        with st.expander("🔍 Technical Details"):
            st.markdown("""
            <div class='info-box'>
            
            * **Base Model:** VGG16 Architecture
            * **Input Size:** 224x224 pixels
            * **Categories:** 6 scene types
            * **Technology:** Transfer Learning
            * **Backend:** TensorFlow 2.x
            
            Supported scene categories:
            * 🏢 Buildings
            * 🌲 Forest
            * ❄️ Glacier
            * ⛰️ Mountain
            * 🌊 Sea
            * 🛣️ Street
            </div>
            """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a scene image to classify",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
    
    with col2:
        st.markdown("### 💡 Tips")
        st.markdown("""
        <div class='info-box'>
        
        * Use clear, well-lit images
        * Avoid blurry photos
        * Center the main scene
        * Landscape orientation preferred
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Process uploaded image
    if uploaded_file is not None:
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 📸 Uploaded Scene")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col4:
            st.markdown("### 🎯 Analysis Results")
            
            with st.spinner('🔄 Analyzing scene...'):
                predictions = predict_scene(model, image)
            
            if predictions:
                top_class = max(predictions.items(), key=lambda x: x[1])
                
                # Display results
                st.markdown(f"""
                <div class='prediction-box'>
                    <h3>🎪 Predicted Scene: <span style='color: #2563EB'>{top_class[0].title()}</span></h3>
                    <h3>📊 Confidence: <span style='color: #2563EB'>{top_class[1]:.1%}</span></h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Show all probabilities
                st.markdown("### 📈 Detailed Analysis")
                
                # Create and sort DataFrame
                df = pd.DataFrame(
                    list(predictions.items()),
                    columns=['Class', 'Probability']
                )
                df = df.sort_values('Probability', ascending=True)
                
                # Custom chart colors
                st.bar_chart(
                    df.set_index('Class'),
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
