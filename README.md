# Evaluating San Jose Animal Shelter Management Effectiveness via Machine Learning Techniques

## Project Overview
This repostitory contains the full scripts required to train, run, and save multiple classification models based on the San Jose Animal Shelter record data. Included models are Naive Bayes, SVM, and Random Forest. 
Note that training times may take a while.
## Requirements
```
python >= 3.8
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
joblib
```

## Project Structure
```
├── data/
│   ├── raw/               # Raw data files. These are the fiscal year animals shelter csvs.
│   ├── processed/               #Processed and cleaned data. Some models use this, but in general the pipeline will take care of pre-processing

├── notebooks/            # Full Jupyter notebooks for running if you prefer to run it code block by block
├── src/                 # Source code
│   ├── full_rf_model.py  # Random Forest Model
│   ├── full_svm_model.py         # SVM model
│   └── full_bayes.py         # Naive Bayes model
├── results/             # Output files and visualizations
│   ├── models/               # Saved models
│   ├── visualizations/               # EDA and other visualizations
└── requirements.txt     # Project dependencies
```
