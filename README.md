# Insurance Cost Optimization: A Predictive and Prescriptive Statistical Analysis
This project is based on the regression analysis of insurance charge prediction, where the insurance customer's data is taken from a particular insurance company. Various machine learning algorithms are used, out of which one best model is chosed for prescriptive analysis.

# Sections
- Overvview
- Dataset Information
- Methodology
- Installation
- How to run
- Authors
- Relevant Links

# Overview
This project holds substantial promise for reshaping the insurance landscape. By focusing on transparency and fairness, the project addresses a persistent concern by providing accurate and comprehensible estimates of insurance charges. This empowers policyholders to make well-informed decisions and plan their finances effectively. Additionally, insurance providers benefit from more precise risk assessment and pricing strategies, contributing to their competitiveness and efficiency. The adaptability of the models to dynamic variables and strict adherence to data privacy regulations underscore ethical and regulatory considerations. Most notably, the project introduces a prescriptive dimension, offering actionable insights for optimizing insurance costs and coverage. 

# Dataset Information
The study uses and compares different machine learning algorithm for predictive modelling in Insurance Cost Optimization (regression problem). The dataset is expected to be a csv file of type Index,Age,Gender, BMI, Children, Smoker, Region, InsuranceCharges where the index is a unique integer identifying the record of insurance customer, Age,Gender, BMI are the age, gender, BMI's of insurance customer respectively, Children is the number of dependents, Smoker is the smoking status of respective customer, Region is the area of resident and InsuranceCharges is nothing but the insurance charges in Rupees. Please note that csv headers are not expected and should be removed from the dataset.

# Methodology
- Data Collection:
  Initiated the project by acquiring the complex dataset from the designated insurance company. Ensured that data sources are reliable, and the dataset encompasses a wide range of variables, including customer demographics, personal information.

- Data Preprocessing:
  Began with data preprocessing to enhance data quality and suitability for analysis. This phase includes addressing missing data, handling outliers, and removing duplicate records. Standardize data formats, encode categorical variables, and ensure consistency in data representations.

- Exploratory Data Analysis (EDA):
  Conducted an exploratory data analysis to gain initial insights into the dataset's characteristics. Generated summary statistics, visualize data distributions (e.g., histograms, box plots), and create scatter plots or heatmaps to identify potential correlations and trends.

- Descriptive Statistical Analysis:
  Performed a detailed descriptive statistical analysis to further understand the dataset. Calculated central tendency measures (mean, median), measures of dispersion (standard deviation, range), and correlation coefficients. Summarize findings in tables and charts for clarity.

- Diagnostic Statistical Analysis:
  Diagnosed potential issues in the dataset that might affect subsequent analyses. Used correlation plot matrices to detect multicollinearity among variables and regression plots to identify heteroscedasticity or non-linear relationships.

- Statistical Testing:
  Employed two specific statistical tests to assess the data:
  - Shapiro-Wilk's Test: Evaluated the normality of data distributions to determine the applicability of parametric statistical tests.
  - Chi-squared Test: Assessed the dependency of attributes, particularly in categorical data, to identify significant relationships.

- Predictive Statistical Model (Machine Learning):
  Developed a predictive statistical model using machine learning techniques. Choosed appropriate algorithms, such as linear regression, decision trees, random forest, based on the project's objectives. Split the dataset into training and testing subsets for model validation.

- Model Comparison:
Trained and evaluated multiple predictive models, assessing their performance using key metrics, including:
  - R-squared: Measured the proportion of variance explained by the model.
  - Adjusted R-squared: Accounted for the number of predictors in the model.
  - Testing R-squared: Measured the model’s accuracy on test set.
  - Mean Squared Error (MSE): Quantified the model's predictive accuracy.

- Best-Fitted Model Selection:
  Selected the best-fitted model based on the model comparison results. Choosed the model that demonstrates the highest predictive accuracy and aligns with the project's goals.

- Prescriptive Statistical Analysis:
  Utilized the selected predictive model to conduct prescriptive analysis. Generated actionable recommendations and strategies based on the model's predictions. These recommendations aim to optimize operational efficiency, enhance customer satisfaction, or drive revenue growth for the insurance company.

- Documentation and Reporting:
  Compiled a comprehensive project report that documents all phases of the analysis. Included detailed explanations of the methodology, data preprocessing steps, EDA findings, descriptive and diagnostic statistical analyses, results of statistical tests, model comparison details, and prescriptive recommendations. Utilized visualizations, tables, and charts to aid in data interpretation and presentation.

Throughout the project, utilized statistical software tools like Python, or specialized machine learning libraries to execute the analysis and generate visualizations. This methodical approach ensures a 
rigorous analysis of the complex insurance company dataset, delivering valuable insights and actionable recommendations to address specific business challenges.

# Installation
There is a general requirement of libraries for the project and some of which are specific to individual methods. The required libraries are as follows:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

These libraries can be installed by using the pip installer.

# How to run
For the sake of simplicity I have added everything in a single Python Notebook file. Follow the Notebook, each cell in the notebook is well commented which will help to understand the project steps.

# Author
[Tushar Kshirsagar](https://github.com/KshirsagarTushar)

# Relevant Links
LinkedIn : www.linkedin.com/in/tushar-kshirsagar-8459bb245

GitHub : https://github.com/KshirsagarTushar
