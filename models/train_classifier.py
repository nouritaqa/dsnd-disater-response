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


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50,100,200],
        'clf__estimator__min_samples_split': [2,3,4]
        ''
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=12)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print("Best Parameters:", model.best_params_)
    for i,name in enumerate(category_names):
        print("Classification report of {}".format(name))
        print(classification_report(Y_test[name],Y_pred[:,i]))


def evaluate_f1(Y_test, Y_pred):
    results_dict = {}
    for i,col in enumerate(Y_test.columns):
        results_dict[col] = f1_score(Y_test[col],Y_pred[:,i],average='micro')
    df_eva = pd.DataFrame([results_dict],index=['f1_score']).T
    return df_eva


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
