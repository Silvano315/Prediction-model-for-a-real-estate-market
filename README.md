# Prediction-model-for-a-real-estate-market

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methods](#methods)
4. [Results](#results)
5. [Extra: Explainability](extra-explainability)
6. [Requirements](#requirements)


## Introduction

This repository is the third project of the master's degree in AI Engineering with [Profession AI](https://profession.ai), all the credits for the requests and idea go to this team. This project was awarded as the 'Best Project' in August 2024, you can see the LinkedIn post [here](https://www.linkedin.com/feed/update/urn:li:activity:7236997652976988160/).

RealEstateAI Solutions aims to optimize real estate price valuation through the use of advanced regularization techniques in linear regression models. The goal is to provide more accurate and reliable price predicts, reducing the risk of overfitting and improving the generalization ability of the model. In the real estate industry, obtaining accurate property price estimates is crucial to make informed decisions. However, traditional linear regression models can suffer from overfitting, compromising the accuracy of the predicts. Effective regularization methods need to be explored to improve the predictive performance and manage the complexity of the model. By implementing and comparing regularization methods such as Lasso, Ridge and Elastic Net, RealEstateAI Solutions will offer a system capable of providing more accurate and stable real estate price predicts. This will allow real estate agents and investors to make decisions based on more reliable data, increasing their competitiveness in the market.

Project Steps:
- Dataset Preprocessing
- Implementation of Regression Models with Ridge, Lasso and Elastic Net regularization
- Performance Evaluation in cross-validation techniques:
- Use of cross-validation techniques.
- Visualization of Results

## Dataset

The dataset is structured in such a way as to emphasize the complexity due to the strong multicollinearity of the features. The dataset has 13 features and 545 samples and here there is a description of the features:
- Price: the price, the target to be predicted
- Area: surface area of ​​the property
- Bedrooms: number of bedrooms
- Bathrooms: number of bathrooms
- Stories: number of floors
- Mainroad: is 1 if the property overlooks a main road, 0 otherwise
- guestroom: is 1 if the property has a guest room, 0 otherwise
- basement: is 1 if the property has a basement, 0 otherwise
- hotwaterheating: is 1 if the property has a boiler, 0 otherwise
- airconditioning: is 1 if the property has air conditioning, 0 otherwise
- parking: number of parking spaces
- prefarea: is 1 if the property is in a prestigious area, 0 otherwise
- Furnishingstatus: is 0 if the property is unfurnished, 1 if it is partially furnished, 2 if it is fully furnished

## Methods

To complete all the project requirements, I have created a single [notebook](prediction_model.ipynb) where all operations are performed and results are visualized. Additionally, I have organized the code into a [`src`](src/) folder containing Python files with the methods used in the notebook.

### Data Engineering

In the [`data_engineering.py`](src/data_engineering.py) file, I have gathered all methods for managing dataset preparation, including:

- **Feature Conversion:** Handling and converting categorical and numerical variables.
- **Duplicate and Missing Data Management:** Functions to identify and handle duplicate and missing values.
- **Interactive Feature Visualization:** Functions to visualize categorical features through bar plots and numerical features through histograms, both implemented with **Plotly**.
- **One-Hot Encoding:** Transformation of categorical features into dummy variables.
- **Standardization:** Functions to standardize numerical features, ensuring they have a distribution with mean 0 and standard deviation 1.

### Model Training & Evaluation

In the [`models.py`](src/models.py) file, I have implemented methods for training and evaluating four regression models:

- **Models:** Linear Regression, Ridge, Lasso, Elastic Net.
- **Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R2 Score.
- **Evaluation:** Training models using both a simple train-test split and K-Fold Cross-Validation.
- **Result Visualization:** Functions to visualize results through bar charts, showing the mean and standard deviation of the requested metrics for each model.
- **Model Complexity Evaluation:** Functions to analyze and visualize model complexity by evaluating the number of non-zero coefficients.

### Results Visualization

In the [`results_viz.py`](src/results_viz.py) file, I have created functions to visualize the results using **Matplotlib** and **Seaborn**:

- **Model Performance:** Functions to create bar plots and violin plots to evaluate and compare model performance based on a specific metric.
- **Residual Evaluation:** For each cross-validation fold, I created a function that generates a histogram of residuals (including mean, standard deviation, and skewness p-value) alongside a scatter plot comparing predicted values with true values.
- **Coefficient Visualization:** A function to visualize the behavior of model coefficients as the regularization parameter `alpha` varies.

## Results

The detailed description of the results, including both the Exploratory Data Analysis (EDA) and model performance evaluations, can be found in the markdown cells of the [notebook](prediction_model.ipynb). It is highly recommended to view the results directly within the notebook, as interactive **Plotly** graphs have been utilized for a more dynamic exploration of the data.

### Key Findings:

- **Results Comparison in Bar Plots (cross-validated results):**
    * With a parameter configuration of alpha = 0.1 and l1_ratio = 0.5 for Elastic Net, it was observed that Linear Regression, Lasso, and Ridge performed quite similarly. Ridge seemed to achieve the best results for both the test and training datasets.

- **Models Complexity Comparison with Non-Zero Coefficients (cross-validated results):**
    * As expected, Linear Regression has no zero coefficients, and the regularization methods set only a few coefficients to zero, confirming the importance of most features.
    * Interestingly, Lasso has fewer zero-coefficients than initially expected!

- **Models Coefficients Trend in Relation to Alpha Regularization Parameter:**
    * The coefficients distributions for several regularization methods were similar when using low alpha values. However, this situation changed with higher alpha values.
    * As alpha increased, Ridge seemed to maintain consistent coefficients, while Lasso tended to set coefficients to zero or near zero, as expected.


## Extra: Explainability

In this project, **SHAP** (SHapley Additive exPlanations) was implemented to make the machine learning models more interpretable and to understand the internal decision-making process of the predictive models. SHAP allows me to break down each prediction to understand how different features contribute to the final output. The following visualizations, saved in the [`images` folder](images/), were generated using SHAP:

- **Summary Plot**
- **Feature Importance Plot**
- **Dependence Plot**
- **Force Plots**

Incorporating SHAP into this project provided a significant advantage in terms of interpretability. SHAP allowed me to go beyond simple performance metrics and understand **why** certain predictions were made. For instance:
1. I was able to see that properties with larger areas consistently had positive SHAP values, indicating that larger properties are strong predictors of higher prices.
2. Binary features like the presence of air conditioning or a guestroom had positive SHAP values when set to 1, revealing that these attributes add significant value to a property.
3. The SHAP analysis made it easier to identify potential **feature interactions**, such as how the number of bathrooms influences the effect of other features like property area or location.
  
This additional layer of insight helps not only in **validating the model** but also in providing real estate agents and decision-makers with a more transparent understanding of which property attributes contribute the most to price estimates. With SHAP, I don't just know the model's prediction but also have a **clear justification** for it, making the model more **trustworthy** in real-world applications.

## Requirements

To run this project, I used a 3.11.x Python version. You need to installed the packages in the [requirements](requirements.txt):

```bash
pip install -r requirements.txt
