# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:03:45 2019

@author: Sanket Dhabale
"""
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords 
from nltk import word_tokenize
from nltk.stem import PorterStemmer 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")





nltk.download('stopwords')
nltk.download('punkt')

stop= stopwords.words('english')

test = pd.read_csv(r'C:\Users\Sanket Dhabale\Desktop\FYP2018\bbc-text.csv')


data= pd.DataFrame(test)
data.columns=["category","text"]


print(data.head())
fig = plt.figure(figsize=(7,5))
data.groupby('category').text.count().plot.bar(ylim=0)
plt.show()

tfidf = TfidfVectorizer(analyzer='word', stop_words = 'english')
score = tfidf.fit_transform(data['text'])

# New data frame containing the tfidf features and their scores
df = pd.DataFrame(score.toarray(), columns=tfidf.get_feature_names())
print(tfidf.get_feature_names())

# Filter the tokens with tfidf score greater than 0.0
Tfidf_values = df.max()[df.max() > 0.0].sort_values(ascending=False)

print(Tfidf_values)


# Preprocessing of the text in dataset
data['stopwords_removed_text']=test['text'].apply(lambda x: ' '.join([w for w in x.split() if w not in (stop) and not w.isdigit()]))

def apwords(words):
    filtered_sentence = []
    words = word_tokenize(words)
    for w in words:
        filtered_sentence.append(w)
    return filtered_sentence
addwords = lambda x: apwords(x) 
     
data['token']=data['stopwords_removed_text'].apply(addwords)
ps= PorterStemmer()
data['token']=data['token'].apply(lambda x : [ps.stem(y) for y in x])


print(data)
 
#Done with Preproceesing

X = data.text
y = data.category
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 1st classifier NV-MNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
#print(y_pred)
accu=np.mean(y_pred==y_test)
A1=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred)) 
print("Accuracy using MNB is ",round(accu,4))
print(A1)
df_cm = pd.DataFrame(A1, range(5),range(5))
plt.figure(figsize = (7,5))
sn.set(font_scale=1)#for label size
sn.heatmap(df_cm, annot=True,cmap='Blues', fmt='g',
           xticklabels=['Business','Entertainment','Politics','Sports','Tech'],
           yticklabels=['Business','Entertainment','Politics','Sports','Tech'],
           annot_kws={"size": 16} # font size
           )
ax=plt.subplot()
ax.set_title('confusion matrix of \nMultinomialNB')
plt.ylabel('Actual')
plt.xlabel('Predicted\naccuracy={:0.4f}; misclass={:0.4f}'.format(accu,1-accu))


# 2nd Classifier Random forest classifier
vect = CountVectorizer(min_df=1)
X = vect.fit_transform(data.text).toarray()
y = vect.fit_transform(data.category).toarray()
from sklearn.ensemble import RandomForestClassifier
rfc = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=0)),
              ])


rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)


#print(y_pred)
accu1=np.mean(y_pred==y_test)
A2=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred)) 
print("Accuracy using Random Forest is  ",round(accu1,4))
print(confusion_matrix(y_test,y_pred))
df_cm = pd.DataFrame(A2, range(5),range(5))
plt.figure(figsize = (7,5))
sn.set(font_scale=1)#for label size
sn.heatmap(df_cm, annot=True,cmap='Blues', fmt='g',
           xticklabels=['Business','Entertainment','Politics','Sports','Tech'],
           yticklabels=['Business','Entertainment','Politics','Sports','Tech'],
           annot_kws={"size": 16} # font size
           )
ax=plt.subplot()
ax.set_title('confusion matrix of \nRandom forest ')
plt.ylabel('Actual')
plt.xlabel('Predicted\naccuracy={:0.4f}; misclass={:0.4f}'.format(accu1,1-accu1))




#3rd classifier linear SVM
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(random_state=42,max_iter=5, tol=-np.infty)),
                   ])
text_clf.fit(X_train, y_train)
y_pred = text_clf.predict(X_test)


#print(y_pred)
accu2=np.mean(y_pred==y_test)
A3=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred)) 
print("Accuracy using Linear SVM is  ",round(accu2,4))
print(A3)

df_cm = pd.DataFrame(A3, range(5),range(5))
plt.figure(figsize = (7,5))
sn.set(font_scale=1)#for label size
sn.heatmap(df_cm, annot=True,cmap='Blues', fmt='g',
           xticklabels=['Business','Entertainment','Politics','Sports','Tech'],
           yticklabels=['Business','Entertainment','Politics','Sports','Tech'],
           annot_kws={"size": 16} # font size
           )
ax=plt.subplot()
ax.set_title('confusion matrix of \nlinear SVM')
plt.ylabel('Actual')
plt.xlabel('Predicted\naccuracy={:0.4f}; misclass={:0.4f}'.format(accu2,1-accu2))




#4th Classifier KNN
from sklearn.neighbors import KNeighborsClassifier
knn=    Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', KNeighborsClassifier(n_neighbors=5)),
                ])

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


#print(y_pred)
accu3=np.mean(y_pred==y_test)
A4=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred)) 
print("Accuracy using KNN is  ",round(accu3,4))
print(A4)

df_cm = pd.DataFrame(A4, range(5),range(5))
plt.figure(figsize = (7,5))
sn.set(font_scale=1)#for label size
sn.heatmap(df_cm, annot=True,cmap='Blues', fmt='g',
           xticklabels=['Business','Entertainment','Politics','Sports','Tech'],
           yticklabels=['Business','Entertainment','Politics','Sports','Tech'],
           annot_kws={"size": 16} # font size
           )
ax=plt.subplot()
ax.set_title('confusion matrix of \nKNN')
plt.ylabel('Actual')
plt.xlabel('Predicted\naccuracy={:0.4f}; misclass={:0.4f}'.format(accu3,1-accu3))


#5th Classifier Logistic Regression

from sklearn.linear_model import LogisticRegression
lr =    Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', LogisticRegression(solver='liblinear',multi_class='ovr',random_state=0)),
                ])

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


#print(y_pred)
accu4=np.mean(y_pred==y_test)
A5=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred)) 
print("Accuracy using logistic regression is  ",round(accu4,4))
print(A5)

df_cm = pd.DataFrame(A5,range(5),range(5))
plt.figure(figsize = (7,5))
sn.set(font_scale=1)#for label size
sn.heatmap(df_cm, annot=True,cmap='Blues', fmt='g',
          xticklabels=['Business','Entertainment','Politics','Sports','Tech'],
           yticklabels=['Business','Entertainment','Politics','Sports','Tech'],
           annot_kws={"size": 16} # font size
           )

ax=plt.subplot()
ax.set_title('confusion matrix of \nLogistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted\naccuracy={:0.4f}; misclass={:0.4f}'.format(accu4,1-accu4))
# print all the efficiency
print("Accuracy using MultiNominal-NB is ",round(accu*100,2),"%")
print("Accuracy using Random Forest is ",round(accu1*100,2),"%")
print("Accuracy using Linear SVM is ",round(accu2*100,2),"%")
print("Accuracy using KNN is ",round(accu3*100,2),"%")
print("Accuracy using Logistic Regression is ",round(accu4*100,2),"%")
