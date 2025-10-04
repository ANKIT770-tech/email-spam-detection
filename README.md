SMS Spam Classifier

This project is a machine learning model designed to classify SMS messages as either spam or ham (non-spam). It provides a lightweight and effective solution for real-time spam filtering in various applications.

Key Features

    Text Preprocessing: The model cleans and normalizes raw SMS messages using techniques such as regular expressions to remove unwanted characters, stopword removal to filter out common words, and Porter Stemming to reduce words to their root form.

    Vectorization: It converts the preprocessed messages into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer, which weighs the importance of words based on their frequency and rarity across the dataset.

    Modeling: A Logistic Regression classifier is used to perform the classification. The entire workflow, from preprocessing to modeling, is streamlined using a scikit-learn pipeline, which ensures consistency and prevents data leakage.

    High Accuracy: The model achieved an accuracy of over 95% on the test dataset.

    Prediction Function: A custom, easy-to-use function is included to classify new, unseen messages as "Spam" or "Ham".

How to Use

Prerequisites

    Python 3.x

    The following Python libraries:

        pandas

        scikit-learn

        nltk

You can install the required libraries using pip:
Bash

pip install pandas scikit-learn nltk

Running the Code

    Place the provided Python script (your_script_name.py) in the same directory as your spam.csv dataset.

    Run the script from your terminal:

Bash

python your_script_name.py

The script will automatically train the model and print the accuracy score and test predictions.

Making a New Prediction

You can use the predict_spam function directly in your script or an interactive Python session:
Python

from your_script_name import predict_spam

# Test a new message
print(predict_spam("Congratulations! You've won a FREE vacation!"))
print(predict_spam("Hey, are you free for coffee this weekend?"))

How It Works

The core of the system is a pipeline that automates the classification process.

    Input: A new SMS message is provided.

    Preprocessing: The message is cleaned and stemmed.

    Vectorization: The cleaned message is transformed into a numerical vector using the trained TF-IDF vectorizer.

    Classification: The vector is fed into the Logistic Regression model, which outputs a prediction (Spam or Ham).
