#!/usr/bin/env python
# coding: utf-8

# ### Title - Fake News Prediction using Machine Learning
# 
# #### From: Anish Moodi

# #### About the Dataset:
# 
# 1) id: Unique Id for a News Article
# 2) title: The Title of a News Article
# 3) author: Author of the News Article
# 4) text: The Text of the Article; could be Incomplete
# 5) label: A Label that Marks whether the News Article is Real or Fake - 
#        1:- Fake News & 
#        0:- Real News

# #### Importing the Dependencies

# In[118]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import matplotlib.pyplot as plt


# In[119]:


# Print the Stopwords in English
print(stopwords.words('english'))


# #### Data Pre-Processing

# In[120]:


# Loading the Dataset into a Pandas DataFrame
news_dataset = pd.read_csv(r'C:\Users\msani\Downloads\train.csv')


# In[121]:


# Display the Shape of the Dataset
print(news_dataset.shape)


# In[122]:


# Display the First 5 Rows of the Dataframe
print(news_dataset.head())


# In[123]:


# Counting the number of Missing Values in the Dataset
print(news_dataset.isnull().sum())


# In[124]:


# Replacing the Null Values with an Empty String
news_dataset = news_dataset.fillna('')


# In[125]:


# Merging the Author Name and News Title
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']


# In[126]:


# Display the Merged Content
print(news_dataset['content'])


# In[127]:


# Separating the Data and Labels
X = news_dataset['content'].values
Y = news_dataset['label'].values


# ### Stemming - 

# In[128]:


# Text Preprocessing using Stemming
port_stem = PorterStemmer()


# In[129]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[130]:


news_dataset['content'] = news_dataset['content'].apply(stemming)


# In[131]:


# Display the Preprocessed Content
print(news_dataset['content'])


# In[132]:


# Separating the Data and Labels after Preprocessing
X = news_dataset['content'].values
Y = news_dataset['label'].values


# In[133]:


# Display the Data and Labels
print(X)
print("\n")
print(Y)


# In[134]:


# Converting the Textual Data to Numerical Data using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[135]:


# Display the Transformed data
print(X)


# In[136]:


# Splitting the Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, stratify=Y, random_state=2)


# In[137]:


# Building and Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)


# ### Evaluation

# #### Accuracy Score

# In[138]:


# Accuracy Score on the Training Data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[139]:


print('Accuracy Score on the Training Data: ', training_data_accuracy)


# In[140]:


# Accuracy Score on the Test Data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[141]:


print('Accuracy Score on the Test Data: ', test_data_accuracy)


# ### Making a Predictive System

# In[142]:


# Making Predictions on a New Data Point
X_new = X_test[3]
prediction = model.predict(X_new.reshape(1, -1))


# In[143]:


# Displaying the Prediction Result
if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')


# In[144]:


# Display the Actual Label for Comparison
print('Actual label:', Y_test[3])
