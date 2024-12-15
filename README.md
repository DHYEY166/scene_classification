# Scene Classification using Transfer Learning

This project implements a scene classification system using VGG16 with transfer learning. The project includes both model training implementations and a Streamlit web application for real-time scene classification.

## Live Demo
🌐 [Scene Classification App](your_streamlit_app_link_here)

## Project Structure
```
project_directory/
├── data.zip
│   ├── seg_train.zip
│   └── seg_test.zip
├── notebooks/
│   ├── scene_classification.ipynb     # Main implementation (VGG16, ResNet50, ResNet101, EfficientNetB0)
│   ├── efficientnet_modified.ipynb    # Modified EfficientNetB0 implementation
│   └── app.py                        # Streamlit application
└── README.md
```

## Models Implemented
- VGG16 (Main implementation)
- ResNet50
- ResNet101
- EfficientNetB0

## Dataset
The dataset consists of scene images divided into six categories:
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

## Implementation Details

### Model Training
1. Data preprocessing and augmentation
2. Transfer learning with VGG16 base
3. Custom layers added for classification
4. Training with early stopping
5. Model weights saving for deployment

### Streamlit Application
The web application allows users to:
- Upload images for classification
- View prediction results
- See confidence scores for each class
- Visualize probability distribution

## Setup and Installation

### Local Development
1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run Streamlit app:
```bash
streamlit run notebooks/app.py
```

### Model Training
1. Extract data:
```python
# Extract training and test data
extract_zip('seg_train.zip', extract_path)
extract_zip('seg_test.zip', extract_path)
```

2. Train model:
```python
# Run the training notebook
jupyter notebook notebooks/scene_classification.ipynb
```

3. Save weights:
```python
# Save model weights
model.save_weights('scene_classifier_full.weights.h5')
```

## Usage

### Web Application
1. Visit the [Scene Classification App](your_streamlit_app_link_here)
2. Upload an image using the file uploader
3. View the classification results and confidence scores

### Local Testing
1. Start the Streamlit server:
```bash
streamlit run notebooks/app.py
```
2. Open browser at `http://localhost:8501`
3. Upload images for classification

## Model Performance
The VGG16 implementation achieves:
- Training Accuracy: XX%
- Validation Accuracy: XX%
- Test Accuracy: XX%

## Dependencies
```
numpy==1.24.3
pandas==2.0.3
Pillow==9.5.0
streamlit==1.24.0
tensorflow==2.13.0
```

## Implementation Notes
- Using on-the-fly data augmentation during training
- Transfer learning with frozen VGG16 base layers
- Custom top layers for classification
- Early stopping for optimal model selection
- Memory-efficient data handling

## Deployment
The application is deployed on Streamlit Cloud:
1. Model weights are saved as 'scene_classifier_full.weights.h5'
2. Streamlit app loads weights dynamically
3. Real-time image processing and prediction

## Authors
- DHYEY DESAI
- Github Username: DHYEY166
- USC ID: 6337508262

## License
[Your License Information]

## Acknowledgments
- Dataset source
- TensorFlow/Keras team
- Streamlit team

## Contact
For questions or feedback:
[Your Contact Information]
