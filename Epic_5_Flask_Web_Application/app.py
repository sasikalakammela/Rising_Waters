from flask import Flask, render_template, request
import pandas as pd
import joblib

# Create Flask App
app = Flask(__name__)

# Load Trained Model
model = joblib.load("model/best_model.pkl")

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    temp = float(request.form["Temp"])
    humidity = float(request.form["Humidity"])
    cloud_cover = float(request.form["Cloud Cover"])
    annual = float(request.form["ANNUAL"])
    jan_feb = float(request.form["Jan-Feb"])
    mar_may = float(request.form["Mar-May"])
    jun_sep = float(request.form["Jun-Sep"])
    oct_dec = float(request.form["Oct-Dec"])
    avgjune = float(request.form["avgjune"])
    sub = float(request.form["sub"])

    input_data = pd.DataFrame(
        [[
            temp,
            humidity,
            cloud_cover,
            annual,
            jan_feb,
            mar_may,
            jun_sep,
            oct_dec,
            avgjune,
            sub
        ]],
        columns=[
            "Temp",
            "Humidity",
            "Cloud Cover",
            "ANNUAL",
            "Jan-Feb",
            "Mar-May",
            "Jun-Sep",
            "Oct-Dec",
            "avgjune",
            "sub"
        ]
    )

    # Print input values
    print(input_data)

    # Make prediction
    prediction = model.predict(input_data)

    # Print prediction in terminal
    print("Prediction:", prediction)

    if prediction[0] == 1:
        result = "⚠ High Flood Risk Detected"
    else:
        result = "✅ Low Flood Risk - No Immediate Danger"

    return render_template("result.html", prediction=result)

# Run App
if __name__ == "__main__":
    app.run(debug=True)