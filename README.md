# Text Classification 

This is an easy to understand script for 'Text Classfication' using SVM and Naive Bayes. 

The input file is also uploaded - corpus.csv

Refer medium link for detailed explanation
https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

# Version 2: 
For people facing challenges in using the models to inference on unseen data. I have adapted the code into two modules - training.py and inference.py.

Below is a short description of them.

training.py - This will train a model on your data and save the trained LabelEncoder, TFIDF Vector and the model itself on disk.
inference.py - This will load these saved files on the disk and do prediction on unseen text.

Feel free to play with them and adapt them further as per your requirements.
