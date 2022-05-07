import sys
import pandas as pd
import numpy as np

def load_data(messages_filepath, categories_filepath):
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
    
    # rpelace categories columns in df with new category column
    df.drop(['categories'],axis=1,inplace=True)
    df = pd.merge(df,categories, on='id',how="left")
    
    return df


def clean_data(df):
    #drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filepath):
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('DisasterResponse', engine, index=False)


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