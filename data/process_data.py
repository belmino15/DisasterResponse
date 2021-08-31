import sys
import pandas as pd
from sqlalchemy import create_engine
from numpy import nan

# python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db

def load_data(messages_filepath, categories_filepath):
    '''Load the data

    Parameters:
        messages_filepath (str): Messages path
        categories_filepath (str): Categories path

    Return:
        df (Panda DataFrames): DataFrame with the data

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, how= 'inner', on= 'id')

    return df


def clean_data(df):
    '''Clean the data

    Parameters:
        df (Panda DataFrames): DataFrame with the data

    Return:
        df (Panda DataFrames): DataFrame with the cleaning data

    '''
    categories = df.categories.str.split(';', expand= True)

    row = categories.head(1)
    category_colnames = [r.split('-')[0] for r in row.values[0]]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand= True)[1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    del df['categories']

    df = pd.concat([df, categories], axis= 1)

    df = df.drop_duplicates()

    df = df[df.related != 2]
    df = df[df.message != '#NAME?']

    duplicated_messages = list(df[df.message.duplicated()].message.unique())

    for message in duplicated_messages:
        # create new row joining the information of the duplicated row (same message) 
        new_row = df[df.message == message].fillna(-1).groupby(['id', 'message', 'original', 'genre']).any().astype(int).reset_index()
        new_row.original = nan

        # delete duplicated rows (same message)
        df = df[df.message != message]
        
        # concat the dataframe and the new_row
        df = pd.concat([df, new_row])

    return df


def save_data(df, database_filename):
    '''Create a DataBase with the clean data

    Parameters:
        df (Panda DataFrames): DataFrame with the data
        database_filename (string): DataBase filename to save the data

    Return:
        None

    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('T_DisasterResponse', engine, if_exists= 'replace', index=False)


def main():
    '''Receive file paths then load, clean and save the data

    Parameters:
        None

    Return:
        None
    
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()