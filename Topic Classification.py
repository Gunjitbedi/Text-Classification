
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import pandas as pd
import nltk
nltk.download('brown')
from textblob import Word
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import csv


#setting random seed
np.random.seed(500)

#cleaning training data
input_0=pd.read_csv(r'\\Mac\Home\Desktop\parallel data\data_6.csv')
stop_words = set(stopwords.words('english'))
limit_0=len(input_0.index)
for i in range(0,limit_0):
    a = TextBlob(input_0.iloc[i,0])
    #a = a.correct()
    a = a.lower()
    w = Word(a)
    a = w.lemmatize()
    input_0.iloc[i,0] = str(a)
    word_tokens = word_tokenize(input_0.iloc[i,0])
    #input_0.iloc[i,0] = input_0.iloc[i,0].lemmatize()
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    input_0.iloc[i,0] = " ".join(filtered_sentence)


print('file tokenized')
input_0.to_csv(r'\\Mac\Home\Desktop\input_0.csv')
print('file printed')

#assigning training data

Train_X = input_0.iloc[:,0]
Train_Y = input_0.iloc[:,1]

#encoding
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
print(Train_Y)

#vectorization
vectorizer = TfidfVectorizer(max_features=5000)
vectors = vectorizer.fit_transform(Train_X)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()

#denselist = dense.tolist()

df = pd.DataFrame(dense, columns=feature_names)
print(df)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(vectors,Train_Y)
print('model trained')

# importing testing data and cleaning it
input=pd.read_excel(r'\\Mac\Home\Desktop\daily.xls')
Output=pd.DataFrame(columns=['Comment','Date','Time','Browser','Email','GUID','URL','Category','Top Customer','Exp User ID','Gross Profit','Repeat Purchase Probability','Centile','NID','IP','Language'])
#print(input.iloc[0,0])
limit=len(input.index)

for i in range(0,limit):
    a = TextBlob(input.iloc[i,0])
    #a = a.correct()
    a = a.lower()
    w = Word(a)
    a = w.lemmatize()
    input.iloc[i,0] = str(a)
    word_tokens = word_tokenize(input.iloc[i,0])
    #input_0.iloc[i,0] = input_0.iloc[i,0].lemmatize()
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    input.iloc[i,0] = " ".join(filtered_sentence)


print('testing file tokenized')

Test_X = input.iloc[:,0]

#vectorization
vectorizer_2 = TfidfVectorizer(max_features=5000)
vectors_2 = vectorizer_2.fit_transform(Test_X)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(vectors_2)
print(predictions_SVM)

# Use accuracy_score function to get the accuracy
#print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

for i in range(0,limit):
    Output.at[i,'Comment']=input.iloc[i,0]
    #cat=cl.classify(input.iloc[i,0])
    blob1 = TextBlob(input.iloc[i,0])
    if(float(format(blob1.sentiment.polarity))>0.3):
        predictions_SVM[i] = '1'
    else:
        if ((input.iloc[i,11]=='1.0') or (input.iloc[i,11]=='2.0') or (input.iloc[i,11]=='3.0') or (input.iloc[i,11]=='4.0') or (input.iloc[i,11]=='5.0') or (input.iloc[i,11]=='6.0') or (input.iloc[i,11]=='7.0') or (input.iloc[i,11]=='8.0') or (input.iloc[i,11]=='9.0') or (input.iloc[i,11]=='10.0')) :
            predictions_SVM[i]='0'
        elif ((input.iloc[i,7]=='GOLD') or (input.iloc[i,7]=='Gold') or (input.iloc[i,7]=='SILVER') or  (input.iloc[i,7]=='Silver') or (input.iloc[i,7]=='PLATINUM')):
            predictions_SVM[i]='0'
        elif ((input.iloc[i,14]!='English') or (input.iloc[i,14]!='en')):
            predictions_SVM[i]='0'
        else :
            print(i)
    
    
    if(predictions_SVM[i]==0):
        cat='Actionable'
    else:
        if(predictions_SVM[i]==1):
            cat='Non Actionable'
    Output.at[i,'Category']=cat
    Output.at[i,'Date']=input.iloc[i,1]
    Output.at[i,'Time']=input.iloc[i,2]
    Output.at[i,'Browser']=input.iloc[i,3]
    Output.at[i,'Email']=input.iloc[i,4]
    Output.at[i,'GUID']=input.iloc[i,5]
    Output.at[i,'URL']=input.iloc[i,6]
    Output.at[i,'Top Customer']=input.iloc[i,7]
    Output.at[i,'Exp User ID']=input.iloc[i,8]
    Output.at[i,'Gross Profit']=input.iloc[i,9]
    Output.at[i,'Repeat Purchase Probability']=input.iloc[i,10]
    Output.at[i,'Centile']=input.iloc[i,11]
    Output.at[i,'NID']=input.iloc[i,12]
    Output.at[i,'IP']=input.iloc[i,13]
    Output.at[i,'Language']=input.iloc[i,14]
 
#b=cl.accuracy(test)
Output.to_excel(r'\\Mac\Home\Desktop\output_NLP.xlsx')
print('output written')








