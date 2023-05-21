import joblib
import pandas as pd

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/described-data")
def described_data():
    data = pd.read_csv("described.csv")
    data_head = pd.read_csv("data.csv").head()
    return render_template(
        "/described.html", table=data.to_html(), table2=data_head.to_html()
    )


@app.route("/predict", methods=["POST"])
def predict():
    model_choice = int(request.form.get("model_choice"))
    if model_choice == 1:
        input_data = [
            [
                int(request.form.get("productsWished")),
                int(request.form.get("civilityGenderId")),
                int(request.form.get("daysSinceLastLogin")),
                int(request.form.get("language_encoded")),
                int(request.form.get("countryCode_encoded")),
                int(request.form.get("hasAndroidApp_encoded")),
                int(request.form.get("hasIosApp_encoded")),
            ]
        ]
        df = pd.DataFrame(input_data)

    else:
        nUser = {
            "socialNbFollowers": [int(request.form.get("socialNbFollowers"))],
            "socialNbFollows": [int(request.form.get("socialNbFollows"))],
            "socialProductsLiked": [int(request.form.get("socialProductsLiked"))],
            "civilityGenderId": [int(request.form.get("civilityGenderId"))],
            "language_encoded": [int(request.form.get("language_encoded"))],
            "countryCode_encoded": [int(request.form.get("countryCode_encoded"))],
            "hasAnyApp_encoded": [int(request.form.get("hasAnyApp_encoded"))],
        }
        df = pd.DataFrame(nUser)

    if model_choice == 1:
        model = joblib.load("models/linear_regression.pkl")
    else:
        model = joblib.load("models/XGB.pkl")

    prediction = model.predict(df)

    return render_template("/predict.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
