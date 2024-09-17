import model
import argparse
import pandas as pd
import joblib

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_df = pd.read_csv('data/train_payment_data.csv', sep=',')
train_loan_df = pd.read_csv('data/train_loan_data.csv', sep=',')
test_df = pd.read_csv('data/test_loan_data.csv', sep=',')


parser = argparse.ArgumentParser(description='Machine Learning Model')
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()
input = args.input
if input == 'train':
    train_main_data = model.get_data(train_loan_df)
    trained_model, X = model.train_model(train_main_data)
    saved_model = model.save_model(trained_model)
    #import the saved model
    saved_model = joblib.load('model.pkl')  
    #predict the results for the test data
    print("predicting the results for the test data ...")
    #test_df = model.predict(trained_model, test_df)
    #group the results by business_id
    #test_df_aggregated = test_df.groupby('business_id').agg({'paid_late':'mean'}).reset_index()
    #print("test_df_aggregated", test_df_aggregated)
    #save results to a file
    #test_df.to_csv('/app/model_output/test_loan_result.csv', index=False)
else:
    print('Invalid input')
