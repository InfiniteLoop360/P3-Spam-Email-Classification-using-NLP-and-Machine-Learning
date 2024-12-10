# P3-Spam-Email-Classification-using-NLP-and-Machine-Learning

Spam Email Classification Using NLP and Machine Learning
This project demonstrates a spam email classification system using Natural Language Processing (NLP) and Machine Learning techniques. The system processes email content, extracts features using TF-IDF, and classifies emails as spam or ham using a Logistic Regression model.

Table of Contents
Introduction
Features
Technologies Used
Dataset
Project Workflow
Installation
Usage
Results
Future Improvements
Contributors
License
Introduction
Spam emails are a significant challenge for email users worldwide, affecting productivity and posing security risks. This project aims to create a machine learning-based solution to classify emails into spam and ham (non-spam) categories using a labeled dataset and various NLP techniques.

Features
Preprocessing of email content (cleaning, tokenization, stopword removal).
Feature extraction using TF-IDF vectorization.
Spam classification using Logistic Regression.
Visualization of email category distribution and word frequencies.
Evaluation metrics like accuracy and confusion matrix.
Technologies Used
Programming Language: Python
Libraries:
Pandas: For data manipulation and cleaning.
NumPy: For numerical computations.
Scikit-learn: For feature extraction, model training, and evaluation.
NLTK: For text preprocessing.
Matplotlib and Seaborn: For data visualization.
Dataset
The dataset contains 5,169 emails labeled as spam or ham.
Source: Provided dataset in CSV format.
Preprocessing steps:
Dropped unnecessary columns and duplicate rows.
Converted text labels (spam and ham) to binary format (0 and 1).
Project Workflow
Data Preprocessing:

Clean and preprocess the text data.
Remove duplicates, stopwords, and special characters.
Feature Extraction:

Use TF-IDF Vectorizer to convert email content into numerical form.
Model Training:

Train a Logistic Regression model on the training dataset.
Evaluation:

Evaluate the model using metrics like accuracy and a confusion matrix.
Visualization:

Plot email category distribution and word frequency charts for spam and ham emails.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/<YourUsername>/spam-email-classification.git
cd spam-email-classification
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Download the dataset and place it in the project directory.

Usage
Run the script to preprocess data, train the model, and evaluate results:

bash
Copy code
python main.py
Test the model with custom email inputs:

python
Copy code
input_mail = ["Your sample email text here."]
Visualize results such as the confusion matrix and word frequency charts.

Results
Training Accuracy: 96.1%
Test Accuracy: 96.4%
Confusion Matrix: Minimal false positives and negatives.
Sample Visualization:
Category Distribution

Confusion Matrix

Word Frequencies

Future Improvements
Implement advanced deep learning models like LSTMs or Transformers.
Extend the dataset to include multilingual emails.
Develop a web interface for real-time email classification.
Contributors
Sudipto Maity
Email: sudiptomaity08@gmail.com
LinkedIn: Your LinkedIn Profile
GitHub: InfiniteLoop360
License
This project is licensed under the MIT License.
