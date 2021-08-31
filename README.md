# Disaster Response Pipeline Project

## Summary:
- Installation
- Motivation
- File Description
- Results
- Licensing, Authors and Acknoledgement

## Installation:

Create a new environment:
> conda create -n new_env python=3.8.3

Activate it:
> conda activate new_env

Install requirements:
> pip install -r requirements.txt

## Motivation:

Analyze disaster data from [Figure Eight](https://appen.com/) to build a machine lerning pipeline model for an API that classifies disaster messages so that you can send the messages to an appropriate disaster relief agency.

A data set contains real messages that were sent during disaster events.

## Instructions:

Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database:
>       python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
- To run ML pipeline that trains classifier and saves:
>       python model/train_classifier.py data/DisasterResponse.db model/model.pkl
This step can take a while to download nltk "punkt" and "wordnet".

## Results

Run `python app/run.py` in the directory where app is downloaded.
Go to http://0.0.0.0:3001/ (or try http://localhost:3001) in a browser.

![Web App](https://github.com/belmino15/DisasterResponse/tree/master/app/imagens/web_app.png)

## Licensing, Authors, Acknowledgements

Must give credit to Figure Eight for the data. You can find the Licensing for the data and other descriptive information [here](https://appen.com/).

