# Disaster Response Pipeline Project
### Table of Contents
1. [Instructions](#installation)
2. [Project Objectives](#objectives)
3. [File Descriptions](#files)
4. [Results](#results)
 4.1. [Data Overview](#data)
 4.2. [Machine Learning](#ml)
 4.3. [Web App](#app)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## 1. Instructions <a name='installation'></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## 2. Project Objectives <a name='objectives'></a>
The project main objective is to build the machine learning model to classify disaster message data from [Figture Eight](https://appen.com), and deploy the web app, where an potential emergency worker can input a new message and get relevant classification results in several categories as well as see the data visualization of original dataset.

The project consists of the following components:
1. ETL Pipeline - Load and clean data, and the store the data in a SQLite database.
2. ML Pipeline - Text processing, feature extraction with BoW and TF-IDF, and machine learning processes. This process includes the model tuning by GridSearchCV with a few additional features.
3. Flask Web App - Deployment of web app for users and visualization of original dataset using Plotly.

This project was completed as part of the course requirements of Udacity's [Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) certification.

## 3. File Descriptions <a name="files"></a>
There are three folders for each project components.
- data
  - original dataset from Figure Eight
    - [disaster_categories.csv](data/disaster_categories.csv)
    - [disaster_messages.csv](data/disaster_messages.csv)
  - [process_data.py](data/process_data.py) - code for ETL pipleline
  - [DisasterResponse.db](data/DisasterResponse.db) - cleaned data stored in SQLite database.
- models
  - [train_classifier.py](models/train_classifier.py) - code for ML Pipeline
  - [classifier.pkl](models/classifier.pkl) - pickle file that stores the trained models
- app
  - templates - html files
  - [run.py](app/run.py) - code for Web App deployment

## 4. Results <a name="results"></a>
### 4.1 Data Overview <a name="data"></a>
Original dataset from Figure Eight consists of the messages from several disaster events and their categories encoded into 36 types. There are 26248 messages with 452 duplicates. The types of messages are 1) direct(10766), 2) news(13054), and 3) social(2396). The categories of messages are all related to disaster and some of them are about particular needs/requests such as foods, medical products and other aids.

It is important for aid organization to identify the emergent needs at the ground-level. Thus, the project aims at supporting them to quickly identify such needs of categories from text messages by using machine learning algorithms.

### 4.2 Machine Learning <a name="ml"></a>
Text processing and feature extraction was undertaken for machine learning. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) was selected for machine learning classifier.

Without tuning, the model's average f1-score is 0.943094 and the 5 lowest f1-scores among categories are as follows:
![this is an image](/images/f1_ml1.png)

To improve the model, the following 5 parameters were tested by GridSearchCV:
- [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
  - max_df: [1, 0.5]
  - max_feature: [None, 5000]
  - ngram_range: [(1,1),(1,2)]
- [TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html?highlight=tfidftransformer#sklearn.feature_extraction.text.TfidfTransformer)
  - use_idf: [True, False]
- [RandomForestClassifier]
  - n_estimators: [100,200]
  - min_samples_split: [2,10]

After tuning, the model predicted as follows:

The best parameters for the above parameters combination were:



### 4.3 Web App <a name="app"></a>

## 5. Licensing, Authors, Acknowledgements<a name="licensing"></a>
