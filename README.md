# Handwriting Author Identification

This project is designed to identify the author of handwriting samples using machine learning and deep learning techniques. The system preprocesses handwriting images, extracts features, and classifies them to identify the author.

## Features
- Preprocessing of handwriting images for noise reduction and feature extraction.
- Model training and evaluation for handwriting author identification.
- Flask-based web application for deployment.
- Test folder for validating the system with new handwriting samples.

## Key Components

- **README.md**: Documentation for the project.

- **FINAL_TEST.ipynb**: The main notebook file containing preprocessing, training, and evaluation code.

- **deployment/**: Contains Flask application files.
  - `app.py`: Flask app entry point.
  - `templates/`: HTML files for the web interface.
  - `static/`: CSS, JavaScript, and images.
  - Other files related to deployment.

- **handwriting/**: Core scripts and modules for handwriting processing.
  - `preprocessing.py`: Preprocessing scripts.
  - `model.py`: Model-related code.
  - Other files for handwriting processing.

- **source6/**: Contains preprocessed handwriting images.
  - `image1.png`: Example image.
  - `image2.png`: Example image.
  - `...`: Other preprocessed images.

- **test/**: Test files and images.
  - `test_image1.png`: Example test image.
  - `test_image2.png`: Example test image.
  - `...`: Other test files.
