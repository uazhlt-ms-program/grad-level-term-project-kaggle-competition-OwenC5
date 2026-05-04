'''
Name: Owen Crandall
Class: DATA 539
Instructor: Dr. Eric Jackson
Due Date: May 6th, 2026
Project Title: Term Asssignment - Class Kaggle Competition

Project Description: 
This project builds a supervised text classification pipeline using scikit-learn to 
predict labels from raw text data of potential movie/TV reviews. The text is transformed into a 
machine-readable format using a CountVectorizer with unigram and bigram features, followed by 
training a logistic regression model on the encoded labels. Model performance is evaluated on 
the training set, and predictions are generated for the test set and exported as a csv submission file.
The predictions are integer values, with 0 meaning no review, 1 meaning a positive review, and 2 meaning
a negative review.
'''

# Importing necessary packages, mostly from scikit-learn.
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Reading in the training and test sets.
train = pd.read_csv('/Users/owencrandall/Downloads/Spring 26/DATA 539/grad-level-term-project-kaggle-competition-OwenC5/train.csv')
test = pd.read_csv('/Users/owencrandall/Downloads/Spring 26/DATA 539/grad-level-term-project-kaggle-competition-OwenC5/test.csv')

# Separating the data into "x" (the text) and "y" (the label).
TEXT = 'TEXT'
LABEL = 'LABEL'
x_train = train[TEXT].fillna("")
y_train = train[LABEL].fillna("")
x_test = test[TEXT]

# Converting the target labels into numeric form so the model can use them.
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Creating the vectorizer to separate the text data.
vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True, strip_accents='unicode', token_pattern=r'\b\w+\b')
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Fitting the logistic regression model on the training data with 1,000 maximum optimization steps and regularization of 1.5.
model = LogisticRegression(
    max_iter=1000,
    C=1.5
)
model.fit(x_train, y_train)

# Calculating the training predictions and evaluation metrics.
y_train_pred = model.predict(x_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_error = 1 - train_accuracy
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
print("Training Accuracy:", train_accuracy)
print("Training Error:", train_error)
print("Training F1:", train_f1)
print(classification_report(y_train, y_train_pred, 
                            target_names=[str(c) for c in le.classes_]))

# Generating the label predictions for the test set.
y_pred = model.predict(x_test)
y_pred_labels = le.inverse_transform(y_pred)
output = pd.DataFrame({
    'ID': test['ID'],
    'LABEL': y_pred_labels
})

# Creating the submission file containing the IDs and predicted labels for the test set.
output.to_csv('/Users/owencrandall/Downloads/Spring 26/DATA 539/grad-level-term-project-kaggle-competition-OwenC5/owen_crandall_submission.csv', index=False)


