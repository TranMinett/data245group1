#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import joblib
from datetime import datetime, timedelta

class SimpleOutcomePredictor:
    #load and run both models
    def __init__(self):
        try:
            self.multiclass_model = joblib.load('final_best_random_forest_multiclass_model.joblib')
            self.binary_model = joblib.load('final_best_random_forest_binary_model.joblib')
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    def calculate_outcome_dates(self, intake_year, intake_month, intake_day, shelter_days):
        #force uses input into readable for model
        intake_date = datetime(int(intake_year), int(intake_month), int(intake_day))
        outcome_date = intake_date + timedelta(days=int(shelter_days))
        return {
            'OutcomeYear': outcome_date.year,
            'OutcomeMonth': outcome_date.month,
            'OutcomeDay': outcome_date.day
        }
    def get_user_inputs(self):
        #prompt user for information
        print()
        print("Please enter the following information:")
        
        #animal info
        inputs = {
            'AnimalType': input("Is this a DOG or CAT? ").upper(),
            'Age_Numeric': float(input("Age in years: ")),
            'PrimaryBreed': input("Primary breed (e.g., LABRADOR RETRIEVER, SIAMESE): ").upper(),
            'Sex': input("Sex (M/F): ").upper(),
            'PrimaryColor': input("Primary color (e.g., BLACK, WHITE): ").upper(),
        }
        
        #intake
        print()
        print("Intake Information:")
        inputs['IntakeType'] = input("Intake type (STRAY or OWNER SURRENDER): ").upper()
        inputs['IntakeCondition'] = input("Condition at intake (NORMAL, INJURED, SICK): ").upper()
        inputs['IntakeReason'] = input("Reason for intake (e.g., STRAY, ABANDONED): ").upper()
        
        # dates
        print()
        print("Date Information:")
        inputs['IntakeYear'] = int(input("Intake year (e.g., 2024): "))
        inputs['IntakeMonth'] = int(input("Intake month (1-12): "))
        inputs['IntakeDay'] = int(input("Intake day (1-31): "))
        inputs['Time_Spent_In_Shelter_Days'] = float(input("Number of days in shelter: "))
        #dummy var fix
        #model wont run w/o this but these are I believe also the mean/mode imputation
        inputs.update({
            'SecondaryColor': 'NONE',
            'IntakeSubtype': 'FIELD',
            'Jurisdiction': 'SAN JOSE'
        })
        outcome_dates = self.calculate_outcome_dates(
            inputs['IntakeYear'],
            inputs['IntakeMonth'],
            inputs['IntakeDay'],
            inputs['Time_Spent_In_Shelter_Days']
        )
        inputs.update(outcome_dates)
        return pd.DataFrame([inputs])
    def predict_outcome(self): #MAIN
        try:
            #start input section
            input_df = self.get_user_inputs()
            
            #predictions
            overall_outcome = self.binary_model.predict(input_df)[0]
            overall_proba = self.binary_model.predict_proba(input_df)[0]
            specific_outcome = self.multiclass_model.predict(input_df)[0]
            specific_proba = self.multiclass_model.predict_proba(input_df)[0]
            outcome_classes = self.multiclass_model.classes_
            # display
            print()
            print("----- PREDICTION RESULTS -----")
            print()
            print(f"Overall Outcome: {overall_outcome}")
            print(f"Confidence: {max(overall_proba)*100:.1f}%")
            print()
            print(f"Most Likely Specific Outcome: {specific_outcome}")
            print()
            print("Probabilities for each outcome:")
            
            #multinomial probabilities
            probs = list(zip(outcome_classes, specific_proba))
            probs.sort(key=lambda x: x[1], reverse=True)
            for outcome, prob in probs:
                print(f"{outcome}: {prob*100:.1f}%")
        #debug error
        except Exception as e:
            print(f"Error making prediction: {str(e)}")

    def run(self):
        #start loop to run
        #print example to run
        #iirc picking something else makes it into a null value
        print("Welcome to the Animal Shelter Outcome Predictor!")
        print()
        print("Example inputs:")
        print("Animal Type: DOG")
        print("Age: 2")
        print("Primary Breed: LABRADOR RETRIEVER")
        print("Sex: M")
        print("Primary Color: BLACK")
        print("Intake Type: STRAY")
        print("Intake Condition: NORMAL")    
        while True:
            self.predict_outcome()
            if input("\nMake another prediction? (y/n): ").lower() != 'y': #set lower for y
                print("\nStopped the Outcome Prediction Program")
                break  #terminate

if __name__ == "__main__": #start program on py run
    predictor = SimpleOutcomePredictor()
    predictor.run()


# In[ ]:




