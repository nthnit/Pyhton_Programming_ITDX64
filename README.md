# Pyhton_Programming_ITDX64
# Wine Quality Prediction using k-NN



## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Modules Description](#modules-description)

## Overview

The main objectives of this project are to:
- Analyze and visualize the wine quality dataset.
- Build and train a k-NN classifier to predict the quality of red wine based on its physicochemical properties.
- Evaluate the model performance using key metrics (accuracy, confusion matrix, classification report).
- Provide an interactive web application where users can explore the data, train the model, and predict wine quality.
- Support multiple languages (English and Vietnamese) to reach a broader audience.

## Project Structure

```
.
├── app.py                      # Main application file that handles navigation and module imports
├── README.md                   # Project documentation and instructions (this file)
├── Data
│   └── winequality-red.csv     # Red wine quality dataset
├── lang
│   └── translations.json       # Language translations for the app (English and Vietnamese)
└── modules
    ├── introduction.py         # Module for the Introduction section
    ├── eda.py                  # Module for Exploratory Data Analysis (EDA)
    ├── training.py             # Module for data preprocessing & model training
    └── prediction.py           # Module for wine quality prediction
```

## Dataset

The project uses the **winequality-red.csv** dataset provided by the University of California, Berkeley. This dataset contains several physicochemical properties of red wine (e.g., fixed acidity, volatile acidity, citric acid, residual sugar, etc.) along with a quality rating. The file is stored in the `Data` folder.

## Technologies Used

- **Python**: Programming language.
- **Streamlit**: Framework for building interactive web applications.
- **Pandas & NumPy**: Libraries for data manipulation and numerical computations.
- **Matplotlib & Seaborn**: Libraries for data visualization.
- **Scikit-learn**: Machine learning library for building and evaluating the k-NN classifier.

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **(Optional) Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**

   Ensure you have a `requirements.txt` file with the following content or similar:

   ```txt
   streamlit
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   ```

   Then run:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:

```bash
streamlit run app.py
```

When the application runs, you'll see a sidebar with navigation options that allow you to switch between the following sections:
- **Introduction:** Overview and details about the project.
- **EDA:** Interactive exploratory data analysis with various visualization options.
- **Preprocessing & Training:** Data preprocessing, model training, and evaluation with the k-NN classifier.
- **Prediction:** Input wine properties and predict wine quality using the trained model.

Additionally, you can select the language (English / Tiếng Việt) in the sidebar.

## Modules Description

- **introduction.py:**  
  Contains the introduction page with the project overview and background.

- **eda.py:**  
  Provides Exploratory Data Analysis (EDA) including:
  - Displaying dataset summary and samples.
  - Multiple visualizations such as histograms, heatmaps, box plots, count plots, pair plots, violin plots, and KDE plots.

- **training.py:**  
  Handles data preprocessing (feature scaling, train/test split) and trains the k-NN classifier. It also evaluates the model using metrics like accuracy, confusion matrix, and classification report.

- **prediction.py:**  
  Offers an interactive interface for users to input wine features and predict wine quality with the trained model.
