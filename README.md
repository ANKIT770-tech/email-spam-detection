SMS Spam Classifier

This project is a machine learning model designed to classify SMS messages as either spam or ham (non-spam). It provides a lightweight and effective solution for real-time spam filtering in various applications.

Key Features:

Text Preprocessing: The model cleans and normalizes raw SMS messages using techniques such as regular expressions to remove unwanted characters, stopword removal to filter out common words, and Porter Stemming to reduce words to their root form.

Vectorization: It converts the preprocessed messages into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer, which weighs the importance of words based on their frequency and rarity across the dataset.

Modeling: A Logistic Regression classifier is used to perform the classification. The entire workflow, from preprocessing to modeling, is streamlined using a scikit-learn pipeline, which ensures consistency and prevents data leakage.


How It Works:

The core of the system is a pipeline that automates the classification process.

  Input: A new SMS message is provided.

  preprocessing: The message is cleaned and stemmed.

  Vectorization: The cleaned message is transformed into a numerical vector using the trained TF-IDF vectorizer.

  Classification: The vector is fed into the Logistic Regression model, which outputs a prediction (Spam or Ham).

Technologies Used:

  Python

  Scikit-learn

  NLTK (Natural Language Toolkit)

  Pandas (for data handling)
