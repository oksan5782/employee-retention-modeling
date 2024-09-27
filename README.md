# Employee Turnover Prediction
Predictive modeling project aimed at identifying key factors contributing to employee turnover. Using machine learning models such as Decision Trees, Random Forest, and XGBoost, this analysis provides insights into employee churn and helps HR departments improve retention strategies.


## Project Overview

This project focuses on analyzing and predicting employee turnover at comany experiencing high employee churn. The project is based on the [Hr Analytics Job Prediction](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv) dataset. The goal is to provide actionable insights to the HR department, helping them identify key factors contributing to turnover and develop strategies for improving employee retention. Used exploratory data analysis (EDA) and machine learning models to predict whether employees are likely to leave the company.


## Problem Statement

High employee turnover can lead to significant costs for companies in terms of hiring, training, and onboarding new employees. Understanding the factors that influence employees to leave is critical for improving retention and reducing costs. This project aims to analyze HR data and build predictive models to identify the key drivers behind employee churn. By accurately predicting which employees are likely to leave based on factors such as satisfaction level, number of projects, average monthly hours, time spent at the company, promotions, department, and salary, companies can take proactive steps to improve job satisfaction and retain valuable talent.

## Files in the Repository

- `helpers.py`: Contains all necessary imports and shared functions that are used across different notebooks.
- `eda_data_cleaning.ipynb`: Performs basic data exploration and cleaning. Includes steps like checking for missing values, duplicates, and outliers.
- `data_exploration.ipynb`: In-depth analysis of relationships between variables, featuring several visualizations such as histograms, scatter plots, box plots, and heatmaps.
- `model_building_1.ipynb`: Initial model building, including Logistic Regression, Naive Bayes, Decision Tree, and Random Forest. Evaluates model performance and refines assumptions.
- `model_building_2.ipynb`: Advanced model building and testing, including Decision Tree, Random Forest, and XGBoost models, with further feature engineering and cross-validation.
- `README.md`: This file explaining the project in detail.
- `models/`: After running a notebook, will contain trained models saved using the pickle module for reuse.

## Tools and Libraries Used

- `Python`: Core programming language for the analysis.
- `pandas`: Data manipulation and analysis.
- `matplotlib` & `seaborn`: Data visualization.
- `scikit-learn`: Machine learning models and evaluation metrics.
- `statsmodels`: Feature and model evaluation.
- `XGBoost`: Advanced model for building decision-tree-based models.
- `pickle`: Saving and reusing models.
- `Jupyter Notebook`: Interactive computing environment used for development.

## Setup Instructions

Run the Jupyter notebooks in the following order:

1. eda_data_cleaning.ipynb
2. data_exploration.ipynb
3. model_building_1.ipynb
4. model_building_2.ipynb

The trained models will be stored in the `models/` folder and can be reused for further analysis.

## Data Preprocessing

The dataset used in this project contains 15,000 rows and 10 columns, [Hr Analytics Job Prediction](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv). The key variables include:

* satisfaction_level
* last_evaluation
* number_project
* average_monthly_hours
* tenure
* work_accident
* left (target variable)
* promotion_last_5_years
* department
* salary
  
### Preprocessing Steps:

* Handling missing values: Checked for missing and null values, and handled them accordingly.
* Outlier detection: Identified outliers in key features like satisfaction_level and average_monthly_hours.
* Encoding categorical variables: One-hot encoding was applied to categorical variables like salary and department.
* Balancing classes: Addressed class imbalance by evaluating model performance.

## Exploratory Data Analysis (EDA)

To better understand the data, the following visualizations and insights were generated:

* Histograms: Showed the distribution of variables such as satisfaction_level and tenure.
* Correlation heatmap: Showed the relationships between features.
* Box plots and scatter plots: Explored relationships between churn (left) and variables like average_monthly_hours, number_project, and salary.
* Department analysis: Compared churn rates across different departments, revealing that some departments experienced higher turnover than others.
* Tenure and churn relationship: Examined how the number of years with the company affected the likelihood of leaving.

## Model Building

Implemented 5 machine learning models to predict employee turnover:

- Logistic Regression: Basic statistical model for classification.
- Naive Bayes: Simple probabilistic classifier based on Bayes' theorem.
- Decision Tree Classifier: A tree-based model that splits data based on feature importance.
- Random Forest Classifier: An ensemble model that aggregates multiple decision trees for better performance.
- XGBoost: A boosting model that builds on the weaknesses of previous models to improve predictive performance.

### Feature Engineering

* Replaced some variables (e.g., average_monthly_hours) with a binary feature (overworked).
* Used label encoding for ordinal variables such as salary.
* One-hot encoding for non-ordinal categorical variables like department.

### Cross-validation

Most models were cross-validated to ensure performance generalization, and hyperparameters were tuned for models like Decision Trees, Random Forest, and XGBoost.

### Evaluation Metrics

The following metrics were used to evaluate the models:

* Accuracy: Proportion of correctly classified instances.
* Precision: Proportion of positive identifications that are actually correct.
* Recall: Proportion of actual positives identified correctly (important for turnover prediction).
* F1 Score: Harmonic mean of precision and recall.
* AUC (Area Under the Curve): Evaluates how well the model distinguishes between classes.

## Results and Insights

Based on the evaluation of accuracy, precision, recall, F1 score, and AUC, XGBoost emerged as the best-performing model. The performance metrics for the models are as follows:

| Model          | Accuracy | Precision | Recall | F1   | AUC  |
|----------------|----------|-----------|--------|------|------|
| Decision Tree  | 0.960    | 0.863     | 0.903  | 0.883| 0.958|
| Random Forest  | 0.955    | 0.841     | 0.902  | 0.870| 0.962|
| XGBoost        | 0.964    | 0.890     | 0.894  | 0.892| 0.970|


### Key Insights
  Feature Importance: The most important features for predicting employee churn in the XGBoost model were:
    * last_evaluation
    * tenure
    * number_project
    * salary
    * Being overworked (working excessive hours)

    
  Churn Drivers: Employees with high evaluations, many projects, long tenures, and those classified as overworked were more likely to leave. Salary played a significant role as well, with those in lower salary bands more likely to churn.
  
<img width="900" alt="feature importance" src="https://github.com/user-attachments/assets/d474ef34-b905-4efa-a253-c15a7183acb7">

## Conclusion

The XGBoost model was the most effective in predicting employee turnover, providing HR with actionable insights to address high-risk employees. By focusing on key factors like evaluation scores, workload, and salary, the company can implement targeted retention strategies, reduce turnover, and save on recruitment costs.


<img width="630" alt="confusion_matrix" src="https://github.com/user-attachments/assets/428e3518-a697-430b-8075-bca8395ad874">


  Confusion Matrix above showes the distribution of true positives, true negatives, false positives, and false negatives for the XGBoost model.
  The model predicts almost the same number of false positives (53) as false negatives (55), which means that some employees may be identified as at risk of quitting or getting fired, when that's actually not the case and vice versa with the same chances. True positives and true negatives prediction rate is high.

