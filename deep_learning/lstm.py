# LSTM  for multi-class classification
import pandas as pd
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import nltk
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from keras.layers import LSTM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Activation, Dropout, BatchNormalization,Embedding
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D
nltk.download('stopwords')
nltk.download('punkt')

# download stop word list
stopwords =nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')

#Load dataset 
df = pd.read_csv("econbiz.csv")
df = df[['id','title','labels']]

# Separate labels and save as list
df['labels'] = df['labels'].str.split()


#Pre-process title data

df['title'] = df['title'].str.lower()
df['title'] = df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
exclude = ['!','"','#','$','%','&',"'",'(',')', '*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','`','{','|','}','~']
df['title'] =df['title'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

X = []
titles = list(df['title'])

for idx,title in enumerate(titles):
    X.append(titles)

#Preprocess labels
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df.labels)
labels = multilabel_binarizer.classes_
y = multilabel_binarizer.transform(df.labels)

#Divide dataset into training and validation set (80/20)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9000)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9000)

#Initialize tokenizer from keras that will vectorize title values
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

vocabulary_size = len(tokenizer.word_index) + 1

x_train = pad_sequences(x_train, padding= 'post',maxlen=51)
x_test = pad_sequences(x_test,padding='post',maxlen=51)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

max_features = vocabulary_size
maxlen = 51
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 300
epochs = 2

embedding_vector_features=45
model = Sequential()
model.add(Embedding(vocabulary_size, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# Evaluate classifier
scores = []
    for i in range(0, 10):
    prediction = model.predict(x_test)
    scores.append(tuple((f1_score(y_test, preds, average="samples"),prob)))

print("EconBiz average F-1 score:", np.mean(scores))