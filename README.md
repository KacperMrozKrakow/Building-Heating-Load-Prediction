# Energy Consumption Prediction in Buildings Using Machine Learning

This project predicts the heating load and energy consumption of buildings by analyzing structural parameters such as wall area, roof area, and overall height. It uses several machine learning regression models, including Decision Trees, Random Forest, Gradient Boosting, and CatBoost, to determine the most accurate method for forecasting energy usage.

## Table of Contents
- [Overview](#overview)
- [Project Goals](#project-goals)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributions](#contributions)
- [License](#license)

## Overview
This project aims to predict the heating load and energy consumption of buildings by analyzing their structural parameters, such as wall area, roof area, and overall height. By testing multiple regression models, including Decision Trees, Random Forest, Gradient Boosting, and CatBoost, the project seeks to determine the most accurate algorithm for forecasting energy usage in energy-efficient buildings.

The primary focus is on leveraging machine learning techniques to optimize energy consumption based on building characteristics. This allows for the exploration of how different models handle numerical data in the context of building automation, where efficiency is crucial. The models are compared based on their prediction accuracy and performance, with the goal of identifying the best approach for minimizing energy waste and improving energy management in smart buildings.

By testing a variety of machine learning algorithms, this project provides insights into how advanced data-driven techniques can be applied to real-world energy management problems, paving the way for their integration into intelligent building control systems.

## Project Goals
- Predict energy usage based on various structural parameters of buildings.
- Compare machine learning models, including Decision Trees, Random Forest, Gradient Boosting, and CatBoost.
- Provide insights into how machine learning can improve energy efficiency in building automation systems.

## Technologies Used
- **Python**
- **Pandas** (for data manipulation)
- **NumPy** (for numerical computations)
- **Matplotlib** & **Seaborn** (for data visualization)
- **Scikit-learn** (for various regression models)
- **CatBoost** (for gradient boosting regressor)

## Dataset

The dataset used for this project is **Building Energy Efficiency.csv**, which contains data on building features and their corresponding heating load and cooling load. This dataset is available directly in the repository under the `data/` directory.

**Dataset Details:**
- **Columns**:
  - Relative Compactness
  - Surface Area
  - Wall Area
  - Roof Area
  - Overall Height
  - Orientation
  - Glazing Area
  - Glazing Area Distribution
  - Heating Load (target)
  - Cooling Load

You can find the dataset in the repository here:
- `data/Building Energy Efficiency.csv`

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KacperMrozKrakow/Building-Energy-Consumption-Prediction.git
   cd Building-Energy-Consumption-Prediction

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate

3. Install the required libraries:
```bash
   pip install -r requirements.txt

## Usage
Once you have installed the dependencies and downloaded the dataset, you can run the script to perform energy consumption predictions based on building features.

This will:

- Load the Building Energy Efficiency.csv dataset.
- Preprocess the data.
- Train several machine learning models (e.g., Decision Tree, Random Forest, Gradient Boosting).
- Compare the performance of the models based on Mean Absolute Error (MAE).
- Visualize the prediction results.

# Results
The project compares various regression models in predicting building energy consumption and concludes which model is most effective. Key visualizations include:

- Energy consumption prediction results: A comparison between actual and predicted heating load values.
- Model error distribution: A histogram of prediction errors for each model.
- Model performance comparison: A bar chart displaying the MAE for each regression model.

# Contributions
Contributions to the project are welcome! If you would like to improve the energy consumption models or add new algorithms, feel free to fork the repository and submit a pull request.
