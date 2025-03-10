
# rMLai-Mikan
A Shiny application for interactive **machine learning** model building, prediction, report generation and **AI assistant (Local LLM)**.

(This app is part of the **WakaCitrus-Informatics** series. For other apps in the **WakaCitrus-Informatics** series, please visit:　https://github.com/yin4dev?tab=repositories)

Welcome to **rMLai-Mikan**! This open-source project is designed to provide a user-friendly interface for data analysis, model training, prediction (including inverse prediction via genetic algorithms), similarity search, and an integrated AI assistant (Local LLM via ollama) for chat-based interaction. Contributions, improvements, and customizations are welcome—feel free to fork and submit pull requests.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Dependencies](#dependencies)
4. [Installation and Setup](#installation-and-setup)
   - [Required Tools](#required-tools)
5. [Usage](#usage)
   - [Running the Application](#running-the-application)
   - [Feature Management](#feature-management)
6. [Configuration](#configuration)
7. [Special Thanks](#special-thanks)
8. [License](#license)

---

## Overview
**rMLai-Mikan** is a Shiny application built with R that offers an interactive platform for data exploration, model training, and evaluation. The app supports both regression and classification tasks, allowing users to upload datasets, visualize data distributions, train various machine learning models, make predictions, and generate comprehensive reports. It also integrates an AI assistant for chat-based interaction to enhance user support and guidance.

---

## Key Features
- **Data Visualization and Exploration**: Upload CSV files and visualize data distributions, correlations, and associations between variables.
 - **Similarity Search**: Identify and compare similar records from the dataset based on normalized feature values (using Cosine Similarity).
- **Model Building and Evaluation**: Train multiple machine learning models (e.g., decision trees, linear regression, random forests, xgboost, LASSO, SVM, logistic regression, naive Bayes, k-nearest neighbors) with configurable hyperparameters.
- **Prediction and Inverse Prediction**: Make predictions on new data and perform inverse prediction using genetic algorithms to optimize input parameters.
- **Integrated AI Assistant**: Chat with a built-in AI assistant powered by ollama to answer questions and provide analytical insights.
- **Report Generation**: Automatically generate detailed reports summarizing data statistics, model performance, and prediction results.

---

## Dependencies
- **Programming Language/Framework**: R, Shiny
- **Libraries/Packages**:  
  `shiny`, `dplyr`, `ggplot2`, `skimr`, `tidyr`, `DT`, `rhandsontable`, `shinyBS`, `glmnet`, `GA`, `randomForest`, `rpart`, `xgboost`, `pROC`, `e1071`, `kknn`, `nnet`, `shinyjs`, `httr`, `jsonlite`, `pdftools`, `readtext`, `tools`
- **System Tools**:  
  - [R](https://www.r-project.org/)
  - [RStudio](https://www.rstudio.com/)
  -  [Ollama](https://www.ollama.com/)  (Optional): To use the AI Assistant feature, please install [Ollama](https://www.ollama.com/) and download the desired LLM. **The core functionality of the app works well without the AI function**.

---

## Installation and Setup

### Required Tools
1. **Clone or Download the Repository**  
   Clone this repository or download the source files

2. **Install External Tools/Services**  
   Make sure you have R installed along with your preferred IDE (e.g., RStudio). No additional external services are required for running the Shiny app.

3. **Install Required Packages**  
   Install the necessary packages in R by running:
   ```r
   install.packages(c("shiny", "dplyr", "ggplot2", "skimr", "tidyr", "DT", "rhandsontable", "shinyBS", 
                        "glmnet", "GA", "randomForest", "rpart", "xgboost", "pROC", "e1071", "kknn", 
                        "nnet", "shinyjs", "httr", "jsonlite", "pdftools", "readtext", "tools"))
   ```
   Alternatively, you can install packages using your preferred package manager.

---

## Usage

### Running the Application
Launch the application by opening the project in RStudio and running the following command in the R console:
```r
library(shiny)
runApp("path_to_your_app_directory")
```
Alternatively, if you are using the command line, navigate to the app’s directory and run:
```r
R -e "shiny::runApp('.')"
```

### Feature Management
- **Data Upload and Visualization**: Use the sidebar to upload a CSV file, select the target and feature variables, and visualize distributions and correlations.
-  **Similarity Search**: Choose a record by name or enter manual values to find similar records in the dataset.
- **Model Training**: Select one or more models from the provided list, configure hyperparameters, and click "Train Model" to view performance metrics and variable importance.
- **Prediction**: Input new data manually or via CSV for predictions, and download prediction results.
- **Inverse Prediction**: Configure target values and variable ranges to run a genetic algorithm for inverse prediction.
- **AI Assistant**: Interact with the integrated AI assistant to get guidance, generate reports, and more.

---

## Configuration
Customize various settings to tailor the application to your needs:
- **System Prompts/Profiles**: Modify the default LLM system prompt and adjust parameters like temperature, top_k, and top_p.
- **API Endpoints**: Update the endpoint URL for LLM interactions if needed.
- **Parameter Settings**: Change default hyperparameters for machine learning models through the UI.

---

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

Copyright (C) 2025 Hongrong Yin

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

---

## Acknowledgments

Special thanks to my loving Aimi Yago for her continuous support, inspiration, and contributions to this project's success! 🎉
