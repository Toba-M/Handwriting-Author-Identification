# Handwriting Author Identification

This project is designed to identify the author of handwriting samples using machine learning and deep learning techniques. The system preprocesses handwriting images, extracts features, and classifies them to identify the author.

## Features
- Preprocessing of handwriting images for noise reduction and feature extraction.
- Model training and evaluation for handwriting author identification.
- Flask-based web application for deployment.
- Test folder for validating the system with new handwriting samples.

## Project Structure

The project is organized as follows:
```
```
project_name/
│
├── README.md              # Documentation for the project
├── FINAL_TEST.ipynb      # The main notebook file
│
├── deployment/           # Contains the Flask app and related files
│   ├── app.py           # Flask app entry point
│   ├── templates/       # HTML files for the web interface
│   ├── static/         # CSS, JavaScript, images
│   └── other files     # Other deployment-related files
│
├── handwriting/         # Handwriting folder
│   ├── preprocessing.py # Preprocessing scripts
│   ├── model.py        # Model-related code
│   └── other files     # Scripts and modules
│
├── source6/            # Contains preprocessed images
│   ├── image1.png      # Example image
│   ├── image2.png
│   └── ...            # Other preprocessed images
│
└── test/              # Test folder
    ├── test_image1.png # Example test image
    ├── test_image2.png
    └── ...            # Other test files
```
```
