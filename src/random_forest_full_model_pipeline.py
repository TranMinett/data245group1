#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import warnings
warnings.filterwarnings("ignore") #i hate the subsetting failure messages

#load data
#faster than reading one by one actually
file_paths = [
    "/Users/Minett/Downloads/animalshelterdata_fy2425.csv",
    "/Users/Minett/Downloads/animalshelterdata_fy2324.csv",
    "/Users/Minett/Downloads/animalshelterdata_fy2223.csv",
    "/Users/Minett/Downloads/animalshelterdata_fy2122.csv",
    "/Users/Minett/Downloads/animalshelterdata_fy2021.csv",
    "/Users/Minett/Downloads/animalshelterdata_fy1920.csv",
    "/Users/Minett/Downloads/animalshelterdata_fy1819.csv"
]
dataframes = [pd.read_csv(file) for file in file_paths]
shelter_data = pd.concat(dataframes)
#remove not valid outcomes
valid_outcomes = ['ADOPTION', 'FOSTER', 'DIED', 
                  'EUTH', 'RESCUE']
shelter_data_filtered = shelter_data[shelter_data['OutcomeType'].isin(valid_outcomes)]
invalid_combinations = [('FOSTER', 'ADOPTION'), ('FOSTER', 'RESCUE'), ('FOSTER', 'FOSTER')]
for intake, outcome in invalid_combinations:
    shelter_data_filtered = shelter_data_filtered[~((shelter_data_filtered['IntakeType'] == intake) &(shelter_data_filtered['OutcomeType'] == outcome))]
shelter_data_filtered_no_dupes = shelter_data_filtered.drop_duplicates(subset='AnimalID', keep='first').copy()

# date columns into datetime for splitting later
shelter_data_filtered_no_dupes['IntakeDate'] = pd.to_datetime(shelter_data_filtered_no_dupes['IntakeDate'])
shelter_data_filtered_no_dupes['OutcomeDate'] = pd.to_datetime(shelter_data_filtered_no_dupes['OutcomeDate'])
# new year,month,day columns from  previous datatime
for prefix, col in [('Intake', 'IntakeDate'), ('Outcome', 'OutcomeDate')]:
    shelter_data_filtered_no_dupes[f'{prefix}Year'] = 
    shelter_data_filtered_no_dupes[col].dt.year
    shelter_data_filtered_no_dupes[f'{prefix}Month'] = 
    shelter_data_filtered_no_dupes[col].dt.month
    shelter_data_filtered_no_dupes[f'{prefix}Day'] = 
    shelter_data_filtered_no_dupes[col].dt.day

# new variable for time spent in shelter which is LOS
shelter_data_filtered_no_dupes['Time_Spent_In_Shelter_Days'] = (shelter_data_filtered_no_dupes['OutcomeDate'] - shelter_data_filtered_no_dupes['IntakeDate']).dt.days

#convert age to numeric and strip out the words in the column
shelter_data_filtered_no_dupes['Age_Numeric'] = pd.to_numeric(
    shelter_data_filtered_no_dupes['Age'].str.extract('(\d+)')[0], 
    errors='coerce')


# features re-encoded or dropped
columns_to_drop = [
    'LastUpdate', 'IntakeDate', 'OutcomeDate', 'Age', 'Crossing', 'OutcomeSubtype', 'OutcomeCondition', 'DOB'
]
shelter_data_filtered_no_dupes.drop(columns=columns_to_drop, inplace=True)

numerical_cols = ['Age_Numeric', 'Time_Spent_In_Shelter_Days', 'IntakeYear', 'IntakeMonth', 'IntakeDay', 'OutcomeYear', 'OutcomeMonth', 'OutcomeDay']
categorical_cols = ['AnimalType', 'PrimaryColor', 'SecondaryColor', 'PrimaryBreed', 'Sex', 'IntakeCondition', 'IntakeType', 'IntakeSubtype', 'IntakeReason', 'Jurisdiction']

# Multiclass setup
#we also drop more variables we do not need
#in the future, you coudl sentiment analysis on the names though
#we didnt learn that though lol
X = shelter_data_filtered_no_dupes.drop(['AnimalID', 'AnimalName', 'OutcomeType'], axis=1)
y = shelter_data_filtered_no_dupes['OutcomeType']

# general train-test split setup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=245)


#HERE's the modelling

# Preprocessor setup
#some feature engineering here that worked out for me
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, interaction_only=True,
                                        include_bias=False))
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', 
                                     drop='first', sparse_output=False))
        ]), categorical_cols)])

# Random Forest model init
rf_model = RandomForestClassifier(random_state=245, class_weight='balanced') #balanced for OUR UNBALANCED classes

# Add the SMOTE to pipeline
pipeline_rf = ImbPipeline(steps=[('preprocessor', preprocessor), 
                                 ('smote', SMOTE(random_state=245)),  
                                 ('model', rf_model)])

#hyperparameter grid
#we use towards data science article hyper params 
#shotgun technqiue!
random_grid = {
    'model__n_estimators': [50, 100, 150],
    'model__max_features': ['auto', 'sqrt'],
    'model__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__bootstrap': [True, False]
}

#set randomsearch method to fit 
rf_random = RandomizedSearchCV(estimator=pipeline_rf, param_distributions=random_grid, 
                               cv=StratifiedKFold(n_splits=5),
                               n_iter=100, random_state=245, n_jobs=-1, scoring='accuracy')

# fit + predict probabiltiies
rf_random.fit(X_train, y_train)
best_model = rf_random.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

# Multiclass metrics
print("\nOptimized Random Forest Model Performance:")
print("Best Parameters:", rf_random.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC and MCC for multiclass
#yes it does work for multiclass
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba, multi_class="ovr"))
#we do have to make convert
print("Matthews Correlation Coefficient (MCC):", matthews_corrcoef(y_test, y_pred))  #this should work normally in a multinomial setting?
shelter_data_filtered_no_dupes['OutcomeType_Binary'] = shelter_data_filtered_no_dupes['OutcomeType'].replace(
    {'ADOPTION': 'POSITIVE_OUTCOME', 'FOSTER': 'POSITIVE_OUTCOME', 'RESCUE': 'POSITIVE_OUTCOME',
     'EUTH': 'NEGATIVE_OUTCOME', 'DIED': 'NEGATIVE_OUTCOME'}
)

#set x binary featurs
X_bin = shelter_data_filtered_no_dupes.drop(['AnimalID', 'AnimalName', 'OutcomeType', 'OutcomeType_Binary'], axis=1)
#set y binary outcomes
y_bin = shelter_data_filtered_no_dupes['OutcomeType_Binary']

# same old train test split
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_bin, y_bin, test_size=0.2, random_state=245)

# binary pipeline
pipeline_rf_bin = ImbPipeline(steps=[('preprocessor', preprocessor), 
                                     ('smote', SMOTE(random_state=245)),  
                                     ('model', rf_model)])

#binary grid search
rf_random_bin = RandomizedSearchCV(estimator=pipeline_rf_bin, param_distributions=random_grid, 
                                   n_iter=100, cv=StratifiedKFold(n_splits=5), 
                                   verbose=2, random_state=245, scoring='accuracy', n_jobs=-1)

# binary model fit start
rf_random_bin.fit(X_train_bin, y_train_bin)
#best binary model
best_model_bin = rf_random_bin.best_estimator_

# binary prediction
y_pred_bin = best_model_bin.predict(X_test_bin)
y_pred_proba_bin = best_model_bin.predict_proba(X_test_bin)[:, 1] #grab probabilities to use later

# Binary metrics
print("\nOptimized Binary Random Forest Model Performance:")
print("Best Parameters:", rf_random_bin.best_params_)
print("Accuracy:", accuracy_score(y_test_bin, y_pred_bin))
print("Classification Report:")
print(classification_report(y_test_bin, y_pred_bin))
print("Confusion Matrix:")
print(confusion_matrix(y_test_bin, y_pred_bin))

# ROC-AUC and MCC for binary classification
print("ROC-AUC Score:", roc_auc_score(y_test_bin, y_pred_proba_bin))
print("Matthews Correlation Coefficient (MCC):", matthews_corrcoef(y_test_bin, y_pred_bin))
#also use job lib to dump model
joblib.dump(best_model, 'final_best_random_forest_multiclass_model.joblib')
joblib.dump(best_model_bin, 'final_best_random_forest_binary_model.joblib')

