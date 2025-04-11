# FruitSnap: AI-Driven System for Fruit Identification, Ripeness Assessment, and Recipe Recommendation
![image](https://github.com/user-attachments/assets/cfcd2d10-dbdf-4fa2-aa97-ecea9d5b6bde)


## Overview

FruitSnap is an intelligent computer vision application that allows users to take a single image of a fruit, which then triggers a cascade of AI capabilities leading to personalized culinary experiences. The system identifies fruit types along with their ripeness level (Raw, Ripe, or Rotten) and provides tailored recipe recommendations based on contextual factors including temperature, weather conditions, and time of day.

## Features

- **Fruit Identification**: Accurately identifies 5 fruit types (Apple, Banana, Mango, Orange, Papaya)
- **Ripeness Assessment**: Classifies fruits into three stages - Raw, Ripe, and Rotten
- **Environmental Context**: Integrates real-time weather data and user location
- **Fuzzy Logic**: Uses fuzzy temperature classification for more natural decision-making
- **Recipe Recommendation**: Suggests appropriate recipes based on fruit condition and environmental factors
- **User-Friendly Interface**: Simple GUI for image upload and recommendation display

## Model Architecture
![image](https://github.com/user-attachments/assets/d1d04349-fda1-41d9-afc7-26c6053fca1a)
The system uses a MobileNetV2-based CNN architecture, fine-tuned on our custom dataset with the following components:

- Transfer learning from pre-trained weights
- Custom top layers for 15-class classification (5 fruits × 3 ripeness states)
- Data augmentation to enhance model robustness
- Feature extraction using convolutional layers

## Dataset
Our model was trained on a diverse dataset compiled through:
1. **Web Scraping**: Automated collection of diverse fruit images across various environments
2. **User Contributions**: Crowd-sourced submissions via Google Forms with metadata
3. **Data Augmentation**: Rotation, flipping, translation, brightness adjustment, and color jittering

## Results

- **Validation Accuracy**: 99.28%
- Strong generalization across all fruit types and ripeness stages
- Minimal confusion between visually similar classes

## Recipe Recommendation Logic

The system uses a rule-based approach with:

- **Contextual Matching**: Finds recipes that match the fruit type, ripeness, and environmental conditions
- **Weighted Similarity Scoring**: Prioritizes recipes based on relevance to current context
- **Fuzzy Logic**: Categorizes temperatures into overlapping sets (cold, cool, moderate, warm, hot)
- **Fallback Recommendations**: Provides general advice when no specific recipe matches

## System Requirements
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Tkinter (for GUI)
- Internet connection (for weather API)


## Usage
1. Launch the application
2. Click "Upload Image" and select a fruit image
3. The system will identify the fruit and its ripeness
4. Current weather information is automatically fetched
5. An appropriate recipe suggestion is displayed


## Future Work
- Expand the fruit variety (currently supporting 5 types)
- Add nutrition information alongside recipes
- Develop a mobile application for on-the-go use
- Include seasonal availability information
- Implement user feedback and recipe rating system

## Team Members
- Satviki Budhia (22070122186)
- Tanisha Vyas (22070122232)
- Yash Parkhi (22070122256)

## Acknowledgments

Special thanks to:
- Professor Akanksha Kulkarni
- Dr. Renuka Agrawal
- Dr. Usha Jogalekar
