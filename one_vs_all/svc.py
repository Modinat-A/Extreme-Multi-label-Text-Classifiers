
# logistic regression for multi-class classification using a one-vs-rest
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# download stop word list
stop = stopwords.words('english')

# Create Classifier using logistic regression
clf = Pipeline([("vectorizer",
                 TfidfVectorizer(max_features=25000)),
                ("classifier", OneVsRestClassifier(LinearSVC(), n_jobs=4))])


# creating a function to split a portion of dataset for training and testing 
def load_dataset(df, fold, all_titles):
    if all_titles != "True":
        print("Using a portion of the dataset")
        df = df[df["fold"].isin(range(0,10))]
    else:
        print("Using all dataset")

    labels = df["labels"].values
    labels = [[l for l in label_string.split()] for label_string in labels]
    multilabel_binarizer = MultiLabelBinarizer(sparse_output = True)
    multilabel_binarizer.fit(labels)

    def to_indicator_matrix(some_df):
        some_df_labels = some_df["labels"].values
        some_df_labels = [[l for l in label_string.split()] for label_string in some_df_labels]
        return multilabel_binarizer.transform(some_df_labels)
    test_df = df[df["fold"] == fold]
    X_test = test_df["title"].values
    y_test = to_indicator_matrix(test_df)

    train_df = df[df["fold"] != fold]
    X_train = train_df["title"].values
    y_train = to_indicator_matrix(train_df)
    
    return X_train, y_train, X_test, y_test

### Evaluate classifier


SINGLE_FOLD = True
ALL_TITLES = False


def evaluate(dataset):
    df = pd.read_csv(dataset)
    df= df.drop(['id'], axis=1)
    df['title'] = df['title'].str.lower()
    df['title'] = df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    scores = []
    for i in range(0, 10):
        train_df, y_train, test_df, y_test = load_dataset(df, i, all_titles = ALL_TITLES)
        clf.fit(train_df, y_train)
        y_pred = clf.predict(test_df)

        scores.append(f1_score(y_test, y_pred, average="samples"))

        if SINGLE_FOLD:
            break
    return np.mean(scores)


print("EconBiz average F-1 score:", evaluate('econbiz.csv'))