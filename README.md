README.md
Bachelors F1 Bayes Predictor
A machine learning project developed as part of a bachelor's thesis that predicts Formula 1 driver performance using a Bayesian classifier—and also compares predictions using Support Vector Machines (SVM). The repository includes scripts for data processing, prediction generation, and visualization of results.

Table of Contents
Overview
Features
Project Structure
Installation
Usage
Requirements
Contributing
License
Contact
Overview
This project leverages historical Formula 1 data to predict:

Driver points per race
Likely winners for upcoming races
It implements two machine learning approaches:

Naive Bayes: For probabilistic prediction of driver points and winning chances.
Support Vector Machines (SVM): As a comparative method to assess prediction performance.
The repository contains scripts to create and manage the F1 database, perform predictions, and visualize the output.

Features
Data Processing: Scripts to build a database (createDB.py) and create tables (createTable.py).
Prediction Models:
naiveBayesClass.py implements the Bayesian classifier.
points_bayes.py and winner_bayes.py predict points and race winners respectively using the Naive Bayes method.
points_svm.py and winner_svm.py provide SVM-based predictions.
Visualization:
visual_bayes_points.py and visuals_svm_points.py for generating graphical representations of the prediction results.
Data Files: CSV files such as final_table.csv, driver_points_per_circuit_with_names.csv, and others that support model training and evaluation.


Analysis: Additional scripts like winner_bayes_analysis.py for deeper performance evaluation.
Project Structure
bachelors_f1_bayes_predictor/
├── .idea/                         # IDE configuration files
├── createDB.py                    # Script to create the database
├── createTable.py                 # Script to create tables in the database
├── main.py                        # Main driver script to run predictions
├── naiveBayesClass.py             # Naive Bayes classifier implementation
├── points_bayes.py                # Prediction of driver points using Naive Bayes
├── points_svm.py                  # Prediction of driver points using SVM
├── visual_bayes_points.py         # Visualization for Naive Bayes points prediction
├── visuals_svm_points.py          # Visualization for SVM points prediction
├── winner_bayes.py                # Winner prediction using Naive Bayes
├── winner_bayes_analysis.py       # Analysis script for Bayesian predictions
├── winner_svm.py                  # Winner prediction using SVM
├── driver_points_per_circuit_with_names.csv  # Data file with driver points per circuit
├── final_table.csv                # Processed data table for training/evaluation
├── predicted_driver_points.csv    # Output CSV for predicted driver points
├── test_driverform.csv            # Test dataset file
├── winning_predictions_naive_bayes.csv  # Output CSV for winning predictions (Naive Bayes)
├── winning_predictions_svm.csv    # Output CSV for winning predictions (SVM)
└── f1_database.db                 # SQLite database file

Installation
1. Clone the repository:
git clone https://github.com/nikawork03/bachelors_f1_bayes_predictor.git
cd bachelors_f1_bayes_predictor

2.Set up a virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

3.Install required dependencies: Ensure you have Python 3 installed. Install necessary packages (e.g., pandas, numpy, scikit-learn, matplotlib) via pip:
pip install -r requirements.txt

Usage
1.Database Setup: Run the following scripts to initialize your database and tables:
python createDB.py
python createTable.py

2. Run Predictions: Use the main script to execute the prediction workflow:
python main.py

3. Visualization: After predictions are generated, run the visualization scripts to see graphical output:
python visual_bayes_points.py
python visuals_svm_points.py

4.Results: Check the CSV output files (e.g., predicted_driver_points.csv, winning_predictions_naive_bayes.csv, winning_predictions_svm.csv) for detailed results.

Requirements
Python 3.x
Pandas
NumPy
Scikit-learn
Matplotlib
SQLite3 (for database management)
Contributing
Contributions, issues, and feature requests are welcome!
Feel free to check issues page if you want to contribute.

Contact
For questions or suggestions, please contact nagliashvilinik@gmail.com

