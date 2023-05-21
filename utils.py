import warnings
from typing import Any, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor as gb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter("ignore")
import xgboost as xgb
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def helper_has_fields_compared_to(
    df: pd.DataFrame,
    columns: List[str],
    target: Union[int, float],
    what: str,
    operator: str,
) -> pd.Series:
    """
    This function takes in a pandas DataFrame, a list of column names, a target value,
    a string indicating whether to return rows where all or any of the conditions are
    met, and an operator to use for comparison. It returns a pandas Series of boolean
    values indicating whether the conditions are met for each row.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame to search through.
    columns (List[str]): A list of column names to compare against the target value.
    target (Union[int, float]): The target value to compare against.
    what (str): A string indicating whether to return rows where all or any of the
                 conditions are met. Must be either "all" or "any".
    operator (str): The operator to use for comparison. Must be one of the
                 following: ">", ">=", "<=", "<", "==", "!=".


    Returns:
    pd.Series: A pandas Series of boolean values indicating whether the conditions are
                met for each row.


    Raises:
    ValueError: If the what parameter is not "all" or "any".
    ValueError: If the operator parameter is not one of the allowed values.
    """
    col = columns[0]
    if operator == ">":
        res = df[col] > target
    elif operator == ">=":
        res = df[col] >= target
    elif operator == "<=":
        res = df[col] <= target
    elif operator == "<":
        res = df[col] < target
    elif operator == "==":
        res = df[col] == target
    elif operator == "!=":
        res = df[col] != target
    for col in columns[1:]:
        if operator == ">":
            tmp = df[col] > target
        elif operator == ">=":
            tmp = df[col] >= target
        elif operator == "<=":
            tmp = df[col] <= target
        elif operator == "<":
            tmp = df[col] < target
        elif operator == "==":
            tmp = df[col] == target
        elif operator == "!=":
            tmp = df[col] != target
        if what == "all":
            res = res & tmp
        elif what in ["any"]:
            res = res | tmp
    return res


def helper_has_any_field_greater_than(df, columns, target):
    res = helper_has_fields_compared_to(df, columns, target, "any", ">")
    return res


def helper_has_any_field_smaller_than(df, columns, target):
    res = helper_has_fields_compared_to(df, columns, target, "any", "<")
    return res


def helper_has_all_field_greater_than(df, columns, target):
    res = helper_has_fields_compared_to(df, columns, target, "all", ">")
    return res


def helper_has_all_field_smaller_than(df, columns, target):
    res = helper_has_fields_compared_to(df, columns, target, "all", "<")
    return res


def helper_has_all_field_equal_to(df, columns, target):
    res = helper_has_fields_compared_to(df, columns, target, "all", "==")
    return res


def models(X_train, y_train, X_test, y_test):
    """
    This function takes in four parameters: X_train, y_train, X_test, and y_test. These
    parameters are the training and testing data for a machine learning model.

    The function prompts the user to select a machine learning model from a list of
        options. The options are:
    1. Naive Bayes
    2. Support Vector Machines
    3. Logistic Regression
    4. Decision Tree
    5. Random Forest Classifier
    6. Extreme Gradient Boosting

    The function then fits the selected model to the training data and makes predictions on the testing data. It prints out the testing set accuracy score, testing set accuracy mean, classification report, and confusion matrix.

    Parameters:
    X_train (array-like, shape (n_samples, n_features)): Training data features
    y_train (array-like, shape (n_samples,)): Training data labels
    X_test (array-like, shape (n_samples, n_features)): Testing data features
    y_test (array-like, shape (n_samples,)): Testing data labels

    Returns:
    None

    Raises:
    ValueError: If any of the input parameters are not of the expected type or shape
    """

    print(
        "Select 1 : Naive Bayes, 2: Support Vector Machines, 3: Logistic Regression,"
        + " 4: Decision Tree, 5: RandomForestClassifier,6: Extreme gradient boosting"
    )
    mo = int(input())
    list = [1, 2, 3, 4, 5, 6]

    if mo == 1:
        model = GaussianNB()
    elif mo == 2:
        model = svm.SVC()
    elif mo == 3:
        model = LogisticRegression()
    elif mo == 4:
        model = DecisionTreeClassifier()
    elif mo == 5:
        model = RandomForestClassifier(random_state=15325)
    elif mo == 6:
        model = xgb.XGBClassifier()
    else:
        print("Invalid Entry")
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    print("testing set accuracy score: ", accuracy_score(y_test, predict))
    accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
    print("testing set accuracy mean: ", accuracies.mean())
    print(classification_report(y_test, predict))
    print("confusion matrix: ")
    print(confusion_matrix(y_test, predict))


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    title: str = "Confusion matrix",
    cmap: str = "Blues",
) -> None:
    """
    Plots a confusion matrix using matplotlib.

    Parameters:
    -----------
    cm : np.ndarray
        A confusion matrix as a 2D numpy array.
    classes : list[str]
        A list of class names in the order they appear in the confusion matrix.
    title : str, optional
        The title of the plot, by default "Confusion matrix".
    cmap : str, optional
        The color map to use for the plot, by default "Blues".

    Returns:
    --------
    None

    Raises:
    -------
    ValueError
        If the dimensions of the confusion matrix and the number of classes do not match
    """

    import itertools

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


# define accuracy function
def predAcc(x_train, x_test, y_train, y_test, model):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    model.fit(x_train, y_train)
    predictTR = model.predict(x_train)
    predictTT = model.predict(x_test)
    print("Accuracy on train set: {:.3f}".format(accuracy_score(y_train, predictTR)))
    print("Accuracy on test set: {:.3f}".format(accuracy_score(y_test, predictTT)))
    print("Model Evaluation:")
    print("classification report of train set: ")
    print(classification_report(y_train, predictTR))
    print("classification report of test set: ")
    print(classification_report(y_test, predictTT))

    print("Confusion matrix of test set: ")
    cnf_matrix = confusion_matrix(y_test, predictTT)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix,
        classes=["Inactive sellers", "Occational sellers", "Frequent sellers"],
        title="Confusion matrix  accumulate",
    )


# define feature selection function
def feaSelect(
    x_train: pd.DataFrame, y_train: pd.Series, func: object, x: pd.DataFrame
) -> None:
    """
    Selects the most important features from a given dataset using Recursive Feature
    Elimination (RFE) algorithm.

    Parameters:
    x_train (pd.DataFrame): The training dataset containing the features.
    y_train (pd.Series): The training dataset containing the target variable.
    func (object): The estimator object used to fit the data.
    x (pd.DataFrame): The dataset containing the features to be ranked.

    Returns:
    None: The function prints the ranked features in descending order of importance.

    Raises:
    None: No exceptions are raised.

    """
    import warnings

    warnings.simplefilter("ignore")
    from sklearn.feature_selection import RFE

    predictors = x_train
    selector = RFE(func, n_features_to_select=1)
    selector = selector.fit(predictors, y_train)
    order = selector.ranking_

    feature_ranks = []
    for i in order:
        feature_ranks.append(f"{i-1}.{x.columns[i-1]}")

    print(feature_ranks)


def org(x: pd.DataFrame, y: pd.Series, model: Any) -> None:
    """
    This function takes in a pandas DataFrame x, a pandas Series y, and a machine
    learning model object and performs the following steps:
    1. Splits the dataset into training and testing sets using train_test_split()
    function from sklearn.model_selection module.
    2. Fits the model on the training data using the fit() method of the model object.
    3. Prints the evaluation of the original model using the predAcc() function.
    4. Prints the features selected by the original model using the feaSelect() function

    Parameters:
    x (pd.DataFrame): A pandas DataFrame containing the features of the dataset.
    y (pd.Series): A pandas Series containing the target variable of the dataset.
    model (Any): A machine learning model object that has a fit() method.

    Returns:
    None: This function does not return anything.

    Raises:
    TypeError: If x is not a pandas DataFrame or y is not a pandas Series or model does
    not have a fit() method.
    ValueError: If x and y have different lengths or if test_size is not between 0 and 1
    """

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.33, random_state=42
    )
    # rebuild model
    model.fit(X_train, Y_train)
    print("Original Model Evaluation: ")
    predAcc(X_train, X_test, Y_train, Y_test, model)
    print("Original Model's features selection: ")
    feaSelect(X_train, Y_train, model, x)


def filt(xf: pd.DataFrame, yf: pd.Series, model: Any) -> None:
    """
    This function takes in a filtered dataset (xf), its corresponding labels (yf), and a
    machine learning model (model).
    It then splits the dataset into training and testing sets, rebuilds the model using
    the training set, and evaluates the model's performance on the testing set. Finally,
    it prints out the feature importance ranking of the model.

    Parameters:
    xf (pd.DataFrame): A filtered dataset.
    yf (pd.Series): The corresponding labels for the filtered dataset.
    model (Any): A machine learning model object.

    Returns:
    None

    Raises:
    TypeError: If the input parameters are not of the expected type.
    ValueError: If the input parameters are not of the expected value.
    """
    # re-split the filtered dataset
    Xf_train, Xf_test, Yf_train, Yf_test = train_test_split(
        xf, yf, test_size=0.33, random_state=42
    )
    # rebuild model
    model.fit(Xf_train, Yf_train)
    print("Filtered Model Evaluation: ")
    predAcc(Xf_train, Xf_test, Yf_train, Yf_test, model)
    print("Filtered Model's Feature Importance Ranking: ")
    feaSelect(Xf_train, Yf_train, model, xf)


def balanced(x_ros: np.ndarray, y_ros: np.ndarray, model: Any) -> None:
    """
    This function takes in a balanced dataset and a machine learning model, and
    performs the following steps:
    1. Splits the dataset into training and testing sets using a 33% test size and a
    random state of 42.
    2. Fits the model on the training data.
    3. Prints the accuracy of the model on the testing data.
    4. Prints the features selected by the model.
    5. Prints the shape of the resampled training and testing datasets.

    Parameters:
    x_ros (np.ndarray): A numpy array containing the features of the balanced dataset.
    y_ros (np.ndarray): A numpy array containing the labels of the balanced dataset.
    model (Any): A machine learning model that has a fit method and a predict method.

    Returns:
    None

    Raises:
    TypeError: If x_ros or y_ros is not a numpy array.
    TypeError: If model does not have a fit method or a predict method.
    """
    # re-split the balanced dataset
    X_ros_train, X_ros_test, Y_ros_train, Y_ros_test = train_test_split(
        x_ros, y_ros, test_size=0.33, random_state=42
    )
    # rebuild model
    model.fit(X_ros_train, Y_ros_train)
    print("Balanced Model Evaluation: ")
    predAcc(X_ros_train, X_ros_test, Y_ros_train, Y_ros_test, model)
    print("Balanced Model's feature Selections: ")
    feaSelect(X_ros_train, Y_ros_train, model, x_ros)
    print("Resample training dataset shape", Y_ros_train.shape[0])
    print("Resample testing dataset shape", Y_ros_test.shape[0])


def balancedFilt(xf_ros: pd.DataFrame, yf_ros: pd.Series, model: Any) -> None:
    """
    Splits the balanced filtered dataset into training and testing sets, rebuilds the
    model, and evaluates its performance.

    Parameters:
    xf_ros (pd.DataFrame): The balanced filtered dataset's feature matrix.
    yf_ros (pd.Series): The balanced filtered dataset's target variable.
    model (Any): The machine learning model to be rebuilt and evaluated.

    Returns:
    None

    Raises:
    N/A

    """
    # re-split the balanced filtered dataset
    Xf_ros_train, Xf_ros_test, Yf_ros_train, Yf_ros_test = train_test_split(
        xf_ros, yf_ros, test_size=0.33, random_state=42
    )
    # Rebuild model
    model.fit(Xf_ros_train, Yf_ros_train)
    print("Balanced Filtered Model Evaluation: ")
    predAcc(Xf_ros_train, Xf_ros_test, Yf_ros_train, Yf_ros_test, model)
    print("Balanced Filtered Model's Feature Importance Ranking: ")
    feaSelect(Xf_ros_train, Yf_ros_train, model, xf_ros)


def get_data_from_db():
    import sqlite3

    # Connect to the SQLite database
    conn = sqlite3.connect("db/raw_data.sqlite3")

    # Write the SQL query
    query = "SELECT * FROM model_data"

    # Use pandas to pass SQL query using connection from SQLite3
    db = pd.read_sql_query(query, conn)

    # Show the resulting DataFrame
    db.head()

    # Don't forget to close the connection
    conn.close()
    return db
