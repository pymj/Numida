#import files
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


# #load data from a file
# train_df = pd.read_csv('/sme-hiring-assessment/machine-learning/data/train_payment_data.csv', sep=',')
# train_loan_df = pd.read_csv('/sme-hiring-assessment/machine-learning/data/train_loan_data.csv', sep=',')
# test_df = pd.read_csv('/sme-hiring-assessment/machine-learning/data/test_loan_data.csv', sep=',')


def get_data(df):
    #identify the outliers using interqurtile range
    Q1 = df['total_owing_at_issue'].quantile(0.25)
    Q3 = df['total_owing_at_issue'].quantile(0.75)
    lower_bound = Q1 - 1.5*(Q3-Q1)
    upper_bound = Q3 + 1.5*(Q3-Q1)
    outliers = df[df['total_owing_at_issue'] > upper_bound]
    train_main_data = df[df['total_owing_at_issue'] < upper_bound]
    #fill the missing values in the train_main_data
    train_main_data['employee_count'].fillna(train_main_data['employee_count'].mean(), inplace=True)
    return train_main_data

#normalize the data imbalance with SMOTE
def train_model(train_main_data):
    #split the data into train and test
    X = train_main_data.drop('paid_late', axis=1)
    y = train_main_data['paid_late']
    X = X.drop('credit_officer_id', axis=1)
    X = X.drop('acquisition_channel', axis=1)
    X = X.drop('applying_for_loan_number', axis=1)
    X = X.drop('dismissal_description', axis=1)
    #X = X.drop('paid_at', axis=1)
    #categorical data with one hot encoding
    #convert the categorical columns to one hot encoding
    X = pd.get_dummies(X, columns=['business_id'], drop_first=True)
    #X = pd.get_dummies(X, columns=['transaction_type'], drop_first=True)
    X = pd.get_dummies(X, columns=['sector'], drop_first=True)
    X = pd.get_dummies(X, columns=['approval_status'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #normalize the data imbalance with SMOTE
    os = SMOTE(sampling_strategy=0.8,random_state=42)
    print("resampling data due to imbalance, this may take while ...")
    X_train_smote, y_train_smote = os.fit_resample(X_train, y_train)
    #train the model
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    print("training model ...")
    y_pred = model.predict(X_test)
    #print the classification report
    print(classification_report(y_test, y_pred))
    #print the confusion matrix
    print("confusion matris on resampled data", confusion_matrix(y_test, y_pred))
    #print the accuracy score
    print(accuracy_score(y_test, y_pred))
    print("predict on full dataset ...")
    y_pred_full = model.predict(X)
    #add the predicted values to the dataframe
    X['paid_late_predict'] = y_pred_full
    #print the classification report
    print("classification report on full training data ", classification_report(y, y_pred_full))
    #print the confusion matrix
    print("confusion matrix", confusion_matrix(y, y_pred_full))
    return model, X

def save_model(model):
    #save the model
    joblib.dump(model, 'model.pkl')
    return model

#inferencing the model
def predict(model, test_df):
    #fill the missing values in the test data
    test_df['employee_count'].fillna(test_df['employee_count'].mean(), inplace=True)
    test_df = test_df.drop('credit_officer_id', axis=1)
    test_df = test_df.drop('acquisition_channel', axis=1)
    test_df = test_df.drop('applying_for_loan_number', axis=1)
    test_df = test_df.drop('paid_late', axis=1)
    test_df = test_df.drop('payment_status', axis=1)
    test_df = test_df.drop('dismissal_description', axis=1)
    test_df = pd.get_dummies(test_df, columns=['approval_status'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['sector'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['business_id'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['loan_id'], drop_first=True)
    #fiil the missing values in the test data
    test_df['employee_count'].fillna(test_df['employee_count'].mean(), inplace=True)
    #drop null values
    test_df.dropna(inplace=True)
    #predict the model
    y_pred = model.predict(test_df)
    #add the predicted values to the dataframe
    test_df['paid_late_predict'] = y_pred
    return test_df




