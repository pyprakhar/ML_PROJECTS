
# Credit Card Fraud Detection using Classification Models

This repository contains Python code for Credit Card Fraud Detection using various Classification Models. The dataset used for this project is sourced from "credit_data.csv" and contains information about credit card transactions, including legitimate and fraudulent transactions.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Selection](#model-selection)
- [Model Evaluation](#model-evaluation)
- [Results and Insights](#results-and-insights)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
Credit card fraud detection is a crucial task in the financial industry to safeguard customers from unauthorized transactions. In this project, we explore different Classification Models to predict and detect fraudulent transactions from the dataset.

## Dependencies
To run the code in this repository, you need the following Python libraries:
- NumPy
- pandas
- scikit-learn
- seaborn
- matplotlib

You can install these dependencies using the following command:
```
pip install numpy pandas scikit-learn seaborn matplotlib
```

## Dataset
The dataset "credit_data.csv" contains various attributes related to credit card transactions, including the transaction amount and whether the transaction is legitimate or fraudulent.
https://www.kaggle.com/datasets/teralasowmya/credit-datacsv

## Data Preprocessing
- Load the dataset into a Pandas DataFrame.
- Check for any missing values and handle them if necessary.
- Analyze the distribution of legitimate and fraudulent transactions.
- Apply under-sampling to create a balanced dataset.

## Model Selection
We train and evaluate the following Classification Models:
- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest
- Gaussian Naive Bayes
- Decision Tree

## Model Evaluation
For each model, we measure accuracy, precision, recall, and F1-score to assess their performance in fraud detection.

## Results and Insights
We analyze the strengths and weaknesses of each model and discuss their implications in correctly identifying legitimate and fraudulent transactions.

## Usage
- Clone this repository:
```
git clone https://github.com/your_username/credit-card-fraud-detection.git
```
- Navigate to the project directory:
```
cd credit-card-fraud-detection
```
- Run the Python script:
```
python credit_card_fraud_detection.py
```

## Contributing
Contributions to this repository are welcome! If you have any suggestions, improvements, or bug fixes, feel free to create a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
