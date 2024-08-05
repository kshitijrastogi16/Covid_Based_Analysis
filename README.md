# Covid Based Analysis

## Machine Learning-Based Research for COVID-19 Analysis and Prediction

## Project Overview
This project leverages machine learning techniques to analyze and predict COVID-19 cases, focusing on both epidemiological and mental health aspects. The research demonstrates the versatility of machine learning in diagnosing, detecting, and forecasting COVID-19 cases, and extends to predicting mental health outcomes, particularly among pregnant women.

## Authors
Tejashri Chavan,
Kshitij Rastogi,
Esha Rai

## Institution
SRM Institute of Science and Technology, Chennai, India

## Abstract
The study utilizes machine learning to perform comprehensive analyses on COVID-19 datasets, including country-wise and month-wise assessments, to discern transmission patterns. Predictive models, specifically logistic regression, were employed to forecast the pandemic's death rate with an accuracy of 84.15%. Additionally, mental health implications were analyzed, achieving a prediction accuracy of 91.84% for outcomes among pregnant women. The findings highlight machine learning’s potential in informing public health strategies and enhancing preparedness for future health crises.

## Keywords
COVID-19
Machine Learning
Logistic Regression

## Introduction
The COVID-19 pandemic has prompted a significant scientific response, with data science and machine learning playing crucial roles in understanding and mitigating the virus's impact. This paper explores the contributions of machine learning in diagnosing, detecting, and predicting COVID-19 cases and its mental health implications. The study emphasizes the importance of a holistic approach in addressing the pandemic’s complexities.

## Literature Review
The paper reviews several studies that highlight the role of machine learning in COVID-19 research, addressing challenges such as data quality, detection, diagnosis, and prediction. Key models and datasets used by different researchers are discussed, showcasing the advancements and potential improvements in the field.

## Methodology
### Datasets
#### Pregnancy_Data:
Contains maternal demographics, health indicators, birth process specifics, and mental health risk factors.
Key features: Maternal_Age, Household_Income, Maternal_Education, Edinburgh_Postnatal_Depression_Scale, Gestational_Age_At_Birth, Delivery_Mode, NICU_Stay, etc.

#### Covid_Death:
Includes detailed medical information of patients, such as medical conditions, interventions, and demographic details.
Key features: MEDICAL_UNIT, SEX, PATIENT_TYPE, DATE_DIED, INTUBED, PNEUMONIA, AGE, DIABETES, etc.

#### Data Preprocessing
Missing Values: Handled using imputation techniques (mean, median, mode).
Outliers: Addressed through techniques like winsorization.
Feature Encoding: Categorical variables converted to binary using one-hot encoding or label encoding.
Normalization: Applied to numerical variables for consistent scaling.
Train-Test Split: Data divided into training and testing sets.

#### Redundancy Reduction
Combined similar threat labels to create a single binary indicator, reducing redundancy.
Preprocessing steps ensured data quality and readiness for model training.

#### Analysis and Visualization
Descriptive Statistics: Monthly distribution of confirmed cases, deaths, and recoveries visualized using bar charts and pie charts.
Ratio Analysis: Deaths per 100 confirmed cases analyzed across different WHO regions.
Predictive Modelling: Logistic regression used for forecasting death rates and mental health outcomes, achieving high accuracy.

#### Results
COVID-19 Death Rate Prediction: Achieved 84.15% accuracy using logistic regression.
Mental Health Prediction for Pregnant Women: Achieved 91.84% accuracy, highlighting the intersection of mental health and infectious disease.

#### Conclusion
The research underscores the pivotal role of machine learning in addressing the multifaceted challenges posed by the COVID-19 pandemic. By providing accurate predictions and insights, machine learning aids in public health decision-making and preparedness for future health emergencies.

#### Contact
Email: kg4225@srmist.edu.in

#### Future Work
Explore additional biomarkers and meta-learning techniques for improved prediction accuracy.
Extend research to other vulnerable demographics and integrate more comprehensive datasets.

This README provides a detailed explanation of the research project, covering every aspect from the introduction to future work. It aims to guide users through the project's objectives, methodologies, and findings comprehensively.
