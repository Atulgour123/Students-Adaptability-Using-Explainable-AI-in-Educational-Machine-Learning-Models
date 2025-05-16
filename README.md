# Students-Adaptability-Using-Explainable-AI-in-Educational-Machine-Learning-Models
Approach:-

The project aims to predict students' adaptability levels in online education.
The approach involves several key steps:
1. Data Loading and Initial Exploration: Load the dataset, check for missing values, data types, and duplicates.
2. Data Visualization: Analyze the distribution of the target variable ('Adaptivity Level') and key features like 'Gender', 'Age', 'Internet Type', and 'Network Type' in relation to the target.
3. Data Preprocessing:
   - Handle missing values (although the code shows dropping columns with all missing values, which in this dataset appears to be none, so effectively no missing values are handled).
   - Apply Label Encoding to convert categorical features into numerical format.
   - Analyze feature correlations using a heatmap.
   - Perform Feature Selection using the Chi-squared test to identify the most important features.
   - Apply MinMaxScaler to scale the selected numerical features.
   - Address class imbalance in the target variable using Random Oversampling.
4. Data Splitting: Split the balanced dataset into training and testing sets.
5. Model Training and Evaluation:
   - Train and evaluate a Stacking Classifier model, which combines predictions from Gradient Boosting, Support Vector Machines, XGBoost, and LightGBM.
   - Evaluate the Stacking Classifier using accuracy, F1 score, precision, recall, classification report, confusion matrix, and ROC curves on both training and testing data.
   - Train and evaluate an Extra Trees Classifier model.
   - Evaluate the Extra Trees Classifier using accuracy, F1 score, precision, recall, classification report, confusion matrix, and ROC curves on both training and testing data.
6. Model Comparison: Visualize the performance metrics (Accuracy, F1 Score, Precision, Recall) of the Stacking Classifier and Extra Trees Classifier on the testing data to compare their effectiveness.
7. Model Explainability:
   - Use SHAP (SHapley Additive exPlanations) to understand the contribution of each feature to the Extra Trees Classifier's predictions.
   - Use LIME (Local Interpretable Model-agnostic Explanations) to explain individual predictions made by the Extra Trees Classifier.


Summary:-

- Imports necessary libraries for data manipulation, visualization, machine learning, and explainable AI (SHAP, LIME).
- Loads the 'students_adaptability_level_online_education.csv' dataset.
- Performs initial data checks (`.head()`, `.isnull().sum()`, `.info()`, `.columns`, `.duplicated().sum()`, `.describe()`).
- Visualizes data distributions using `seaborn.countplot` and `matplotlib.pyplot.pie`.
- Applies `LabelEncoder` to all object type columns.
- Computes and visualizes the correlation matrix using `dfrm.corr()` and `seaborn.heatmap`.
- Performs feature selection using `SelectKBest` with `chi2`.
- Selects and scales features using `MinMaxScaler`.
- Addresses class imbalance using `RandomOverSampler` and visualizes the distribution before and after oversampling.
- Splits the data into training and testing sets using `train_test_split`.
- Defines and trains a `StackingClassifier` with specified base estimators (GradientBoostingClassifier, SVC, XGBClassifier, LGBMClassifier) and a final estimator (LogisticRegressionCV).
- Evaluates the Stacking Classifier on training and testing data, printing accuracy, F1 score, precision, recall, and classification reports. Plots confusion matrices and ROC curves for the Stacking Classifier.
- Defines and trains an `ExtraTreesClassifier`.
- Evaluates the Extra Trees Classifier on training and testing data, printing accuracy, F1 score, precision, recall, and classification reports. Plots confusion matrices and ROC curves for the Extra Trees Classifier.
- Creates a bar plot to compare the performance metrics of the Stacking Classifier and Extra Trees Classifier on the testing data.
- Uses the trained Extra Trees Classifier, training data, and testing data to generate SHAP summary plots.
- Uses the trained Extra Trees Classifier and test data to generate LIME explanations for a specific instance, showing the explanations in a notebook table format.
