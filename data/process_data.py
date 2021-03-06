import sys
import pandas as pd
import numpy as np

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load data from csv fiels and merge to a single pandas dataframe with 1/0 number of categorical values

    Input:
    messages_filepath   filepath to messages csv file
    categories_filepath filepath to categories csv file

    Returns
    df      dataframe merging categories and messages
    '''
    #loda data into padans dataframe and merge them
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id',how='left')

    #create dataframe of the 36 indivifudal categoreis plus 'id' column to merge
    categories = pd.concat([categories['id'],categories['categories'].str.split(';',expand=True)],axis=1)

    #rename the columns of categories
    category_colnames = categories.iloc[0,1:].apply(lambda x: x[:-2]).values
    categories.columns = ['id'] + list(category_colnames)

    #convert category variable to 1 or 0 number
    for column in categories:
        if column == 'id':
            pass
        else:
            # set each value to be the last character of the string and convert column from string to numeric
            categories[column] = categories[column].apply(lambda x: int(str(x)[-1]))

    # replace categories columns in df with new category column
    df.drop(['categories'],axis=1,inplace=True)
    df = pd.merge(df,categories, on='id',how="left")

    return df


def clean_data(df):
    '''
    clean_data
    - Remove the duplicated rows from pandas dataframe
    - Conver values of related column to binary

    Input:
    df      pandas dataframe

    Output:
    df      cleaned pandas dataframe

    '''
    #drop duplicates
    df.drop_duplicates(inplace=True)

    #rel
    df.loc[(df_merged.related == 2), 'related'] = 1
    return df


def save_data(df, database_filepath):
    '''
    save_data
    Save the pandas dataframe to SQL database

    Inputs:
    df                  pandas dataframe
    database_filepath   filepath to SQL database

    Output:
    'DisasterResponse' SQL database will be saved in the database_filepath
    '''
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
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
