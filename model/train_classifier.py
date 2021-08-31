# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import pickle
from functools import partial

# python model/train_classifier.py data/DisasterResponse.db model/model.pkl

def load_data(data_file):
    '''Return the data separated by X, y (features and target)

    Parameters:
        data_file (str): Data path

    Return:
        X, y (Panda DataFrames): Features and Targets

    '''
    path = data_file

    engine = create_engine('sqlite:///' + path)
    df = pd.read_sql_table(table_name= 'T_DisasterResponse', con= engine)

    # define features and label arrays

    X = df['message']
    y = df.drop(columns= ['original', 'id', 'message', 'genre'])

    return X, y


def tokenize(text):
    '''Return a list of string after tokenize the string

    Parameters:
        text (str): String to be tokenized


    Return:
        clean_tokens (list): List of string
    
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''Return a pipeline builded with CountVectorizer and TfidTransformer

    Parameters:
        None

    Return:
        pipeline (object): builded pipeline
    
    '''
    # text processing and model pipeline

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer= partial(tokenize))),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    # define parameters for GridSearchCV

    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_leaf': [5, 10]
        }

    # create gridsearch object and return as final model pipeline

    model_pipeline = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose= 2)

    return model_pipeline


def train(X, y, model):
    '''Return trained model
    
    Parameters:
        X (Pandas Dataframe): Features of the model
        y (Pandas Series): Target
        model (object): Builded Pipeline

    Return:
        model (object): Trained model
    
    '''
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # fit model
    model.fit(X_train, y_train)

    # output model test results
    y_pred_improved = model.predict(X_test)
    print(classification_report(y_test.values, y_pred_improved, target_names= y.columns, zero_division= False))
    
    return model


def export_model(model, model_path):
    '''Export the trained model
    
    Parameters:
        model (object): Trained model
        model_path (string): Path to save the trained model
        
    Return:
        None
        
    '''
    # Export model as a pickle file

    pickle.dump(model, open(model_path, 'wb'))

def run_pipeline(data_file, model_path):
    '''Function to load the data, build, train and export the trained model

    Parameters:
        data_file (string): Path to the data
        model_path (string): Path to save the trained model

    Return:
        None
    
    '''
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model, model_path)  # save model


if __name__ == '__main__':
    # get filename of dataset
    #data_file = sys.argv[1]
    #model_path = sys.argv[2]
    
    data_file = 'data/DisasterResponse.db'
    model_path = 'model/model.pkl'
    run_pipeline(data_file, model_path)  # run data pipeline