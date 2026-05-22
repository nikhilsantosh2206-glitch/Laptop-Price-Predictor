from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model/laptop_price_model.pkl", "rb"))
brand_encoder = pickle.load(open("model/brand_encoder.pkl", "rb"))
processor_encoder = pickle.load(open("model/processor_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    brand = request.form['brand']
    ram = int(request.form['ram'])
    storage = int(request.form['storage'])
    processor = request.form['processor']
    screen = float(request.form['screen'])

    # Encode categorical values
    brand = brand_encoder.transform([brand])[0]
    processor = processor_encoder.transform([processor])[0]

    # Prediction
    features = np.array([[brand, ram, storage, processor, screen]])

    prediction = model.predict(features)[0]

    return render_template(
        'index.html',
        prediction_text=f"Estimated Laptop Price: ₹ {round(prediction,2)}"
    )

if __name__ == "__main__":
    app.run(debug=True)