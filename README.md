# AI project

## Flask Machine Learning Models Application

This application is a simple Flask web application that uses machine learning models (Linear Regression and XGBoost) to predict outcomes based on user input. And it has some notebooks in it.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

Note: There may be problems with the relative path of the files in the notebooks and flask app. If you have problems, please change the path to the absolute path of your PC.

1.  Create a virtual environment:

    `python -m venv venv`

    This will create a new Python virtual environment in a folder named `venv`. If you are not running a virtual env just run pip install on your global python.

2.  Activate the virtual environment:

    `source venv/bin/activate`

3.  Install the required packages:

    Copy code

    `pip install -r requirements.txt`

4.  Run the notebooks

## Running the Application

1.  Set the FLASK_APP environment variable (not sure if needed):

    `export FLASK_APP=app.py`

2.  Run the Flask application:

    `python flask/app.py`

3.  Open a web browser and navigate to `http://127.0.0.1:5000/` to see the application in action.

    **Troubleshoot:** if there is a problem of loading models for the flask app, change the path of models and csv files in `flask/app.py` file.

    ex: from

    ```python
     data = pd.read_csv("described.csv")
     data_head = pd.read_csv("data.csv").head()
      model = joblib.load("models/linear_regression.pkl")
      model = joblib.load("models/XGB.pkl")
    ```

    to

    ```python
     data = pd.read_csv("../described.csv")
     data_head = pd.read_csv("../data.csv").head()
      model = joblib.load("../models/linear_regression.pkl")
      model = joblib.load("../models/XGB.pkl")
    ```

    You should do the same for the models
