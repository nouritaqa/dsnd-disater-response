import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
import nltk
nltk.download(['punkt', 'stopwords','wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import timeit
import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM DisasterResponse', engine)
    X = df.message
    X_genre = pd.get_dummies(df.genre, drop_first=True)
    Y = df.iloc[:,4:]
    category_names = list(df.iloc[:,4:].columns)
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text.lower().strip())
    tokens = [w for w in tokens if w not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    clean_tokens = [lemmatizer.lemmatize(tok,pos='v') for tok in tokens]
    return clean_tokens

# Custom feature1: Number of words in sentence
class WordCount(BaseEstimator, TransformerMixin):
    def word_counts(self, text):
        words = word_tokenize(text.lower().strip())
        return len(words)

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        count = pd.Series(x).apply(self.word_counts)
        return pd.DataFrame(count)

# Custome feature2: Number of noun in Sentence
class NounCount(BaseEstimator, TransformerMixin):
    def noun_counts(self,text):
        count = 0
        pos_tags = nltk.pos_tag(word_tokenize(text.lower().strip()))
        for token, tag in pos_tags:
            if tag in ['PRP', 'NN']:
                count += 1
        return count

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        count = pd.Series(x).apply(self.noun_counts)
        return pd.DataFrame(count)

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            #text pipeline
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            #additional custom features
            ('word_count', WordCount()),
            ('noun_count', NounCount())
        ])),

        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
    'features__text_pipeline__vect__max_df': [1, 0.75],
    'features__text_pipeline__vect__max_features':[None, 5000],
    'features__text_pipeline__tfidf__use_idf': [False, True],
    'features__vect__ngram_range': [(1,1),(1,2)],
    'clf__estimator__n_estimators': [10,100],
    'clf__estimator__min_samples_split': [2,10],
    'features__transformer_weights':(
        {'text_pipeline':1, 'word_count':0.5, 'noun_count':0.5},
        {'text_pipeline':0.8, 'word_count':1, 'noun_count':1})
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=12, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print("Best Parameters:", model.best_params_)
    for i,name in enumerate(category_names):
        print("Classification report of {}".format(name))
        print(classification_report(Y_test[name],Y_pred[:,i]))

    results_dict = {}
    for i,col in enumerate(Y_test.columns):
        results_dict[col] = f1_score(Y_test[col],Y_pred[:,i],average='micro')
    df_eva = pd.DataFrame([results_dict],index=['f1_score']).T
    print("The average f1-score is {}".format(df_eva.mean()))
    print("The 5 lowest f1-score are: ")
    return df_eva.sort_values('f1_score').head()

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
