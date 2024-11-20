#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd

# Define file paths for the datasets
shelter_data_24 = pd.read_csv("/Users/Minett/Downloads/animalshelterdata_fy2425.csv")
shelter_data_23 = pd.read_csv("/Users/Minett/Downloads/animalshelterdata_fy2324.csv")
shelter_data_22 = pd.read_csv("/Users/Minett/Downloads/animalshelterdata_fy2223.csv")
shelter_data_21 = pd.read_csv("/Users/Minett/Downloads/animalshelterdata_fy2122.csv")
shelter_data_20 = pd.read_csv("/Users/Minett/Downloads/animalshelterdata_fy2021.csv")
shelter_data_19 = pd.read_csv("/Users/Minett/Downloads/animalshelterdata_fy1920.csv")
shelter_data_18 = pd.read_csv("/Users/Minett/Downloads/animalshelterdata_fy1819.csv")
#slap them on top of each other
shelter_data = pd.concat([shelter_data_24, shelter_data_23,
                       shelter_data_22, shelter_data_21,shelter_data_20, shelter_data_19,
                       shelter_data_18])

# Display all columns
pd.set_option('display.max_columns', None)



# filter  outcome types for modelling later
valid_outcomes = ['ADOPTION', 'FOSTER', 'DEAD', 'EUTH', 'RESCUE']
shelter_data_filtered = shelter_data[shelter_data['OutcomeType'].isin(valid_outcomes)]

# remove invalid intake/outcome combinations. we only want outcomes that constitute
# the instance of an animal leaving the shelter
for intake, outcome in [('FOSTER', 'ADOPTION'), ('FOSTER', 'RESCUE'), ('FOSTER', 'FOSTER')]:
    shelter_data_filtered = shelter_data_filtered[~((shelter_data_filtered['IntakeType'] == intake) & 
                                                     (shelter_data_filtered['OutcomeType'] == outcome))]

# remove duplicates, keeping the first instance
#the only way to ignore the value slice copy error is to make a copy of the dataframe here
shelter_data_filtered_no_dupes = shelter_data_filtered.drop_duplicates(subset='AnimalID', keep='first').copy()



# In[15]:


#this is a mis-entry from the clinic. the cat is definitely not 1 year in the future.
shelter_data_filtered_no_dupes.loc[shelter_data_filtered_no_dupes['AnimalID'] == 'A1334166', 'OutcomeDate'] = '2024-03-31'


# In[16]:


#save dataset for data visualization
shelter_data_filtered_no_dupes.to_csv('cleaned_shelter_data_for_EDA.csv', index=False)


# In[17]:


#the hardest part because datetime and pandas df like to fight each other
# convert date columns to datetime
shelter_data_filtered_no_dupes['IntakeDate'] = pd.to_datetime(shelter_data_filtered_no_dupes['IntakeDate'])
shelter_data_filtered_no_dupes['OutcomeDate'] = pd.to_datetime(shelter_data_filtered_no_dupes['OutcomeDate'])

# extract year, month, day from IntakeDate
shelter_data_filtered_no_dupes['IntakeYear'] = shelter_data_filtered_no_dupes['IntakeDate'].dt.year
shelter_data_filtered_no_dupes['IntakeMonth'] = shelter_data_filtered_no_dupes['IntakeDate'].dt.month
shelter_data_filtered_no_dupes['IntakeDay'] = shelter_data_filtered_no_dupes['IntakeDate'].dt.day

shelter_data_filtered_no_dupes['OutcomeYear'] = shelter_data_filtered_no_dupes['OutcomeDate'].dt.year
shelter_data_filtered_no_dupes['OutcomeMonth'] = shelter_data_filtered_no_dupes['OutcomeDate'].dt.month
shelter_data_filtered_no_dupes['OutcomeDay'] = shelter_data_filtered_no_dupes['OutcomeDate'].dt.day


# In[18]:


# calculate Time Spent In Shelter
shelter_data_filtered_no_dupes['Time_Spent_In_Shelter'] = shelter_data_filtered_no_dupes['OutcomeDate'] - shelter_data_filtered_no_dupes['IntakeDate']
shelter_data_filtered_no_dupes['Time_Spent_In_Shelter_Days'] = shelter_data_filtered_no_dupes['Time_Spent_In_Shelter'].dt.days

# previous check for bad date entries
shelter_data_filtered_no_dupes['Is_Outcome_After_Intake'] = shelter_data_filtered_no_dupes['OutcomeDate'] >= shelter_data_filtered_no_dupes['IntakeDate']

# convert Age to numeric values
shelter_data_filtered_no_dupes['Age_Numeric'] = pd.to_numeric(shelter_data_filtered_no_dupes['Age'].str.extract('(\d+)')[0], errors='coerce')

# drop unnecessary columns not needed in model
columns_to_drop = ['LastUpdate', 'IntakeDate', 'OutcomeDate', 'Age', 'Time_Spent_In_Shelter','Is_Outcome_After_Intake',
                  'Crossing','OutcomeSubtype', 'OutcomeCondition', 'DOB'  ]
shelter_data_filtered_no_dupes.drop(columns=columns_to_drop, inplace=True)




# In[9]:


#uncomment and run this if you want to do one hot encoding before model
#other models may not need this so beware
#also no preprocessor pipeline needed here on accident
#THIS INCLUDES MEAN AND MODE IMPUTATION
#IMPUTATION TYPE SUBJECT TO CHANGE IF I FIND A BETTER ONE
#-------------------------------------------------
from sklearn.preprocessing import OneHotEncoder

numerical_cols = ['Age_Numeric', 'Time_Spent_In_Shelter_Days', 'IntakeYear', 'IntakeMonth', 'IntakeDay', 'OutcomeYear', 'OutcomeMonth', 'OutcomeDay']

categorical_cols = ['AnimalType', 'PrimaryColor', 'SecondaryColor', 'PrimaryBreed', 'Sex', 'IntakeCondition', 'IntakeType', 'IntakeSubtype', 'IntakeReason', 'Jurisdiction']

#numerical var mean impute
for col in numerical_cols:
    shelter_data_filtered_no_dupes[col].fillna(shelter_data_filtered_no_dupes[col].mean(), inplace=True)

# cat var mode impute
for col in categorical_cols:
    shelter_data_filtered_no_dupes[col].fillna(shelter_data_filtered_no_dupes[col].mode()[0], inplace=True)

# one hot encoding from sk learn on cat var, drop first variable
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first' )

encoded_cats = pd.DataFrame(encoder.fit_transform(shelter_data_filtered_no_dupes[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))

# drop original cat
shelter_data_filtered_no_dupes_cleaned = shelter_data_filtered_no_dupes.drop(categorical_cols, axis=1, errors='ignore')


# Reset indices
shelter_data_filtered_no_dupes_cleaned = shelter_data_filtered_no_dupes_cleaned.reset_index(drop=True)
encoded_cats = encoded_cats.reset_index(drop=True)

# merge cleaned df with the one-hot encoded df
shelter_data_filtered_no_dupes = pd.concat([shelter_data_filtered_no_dupes_cleaned, encoded_cats], axis=1)

shelter_data_filtered_no_dupes.to_csv('cleaned_shelter_data_for_models_one_hot.csv', index=False)


# In[10]:


# save this dataframe for modelling later
shelter_data_filtered_no_dupes.to_csv('cleaned_shelter_data_for_models.csv', index=False)

