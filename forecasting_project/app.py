from flask import Flask, jsonify
import joblib

# Create Flask app
app = Flask(__name__)

# Load saved model
model = joblib.load('best_xgboost_model.pkl')

@app.route('/predict')

def predict():

    # Example sample input
    sample_data = [[
        100000000,   # lag_1
        95000000,    # lag_7
        98000000,    # rolling_mean_7
        12,           # month
        6             # day_of_week
    ]]

    # Predict
    prediction = model.predict(sample_data)

    return jsonify({
        'Predicted Sales': float(prediction[0])
    })

# Run app
if __name__ == '__main__':
    app.run(debug=True)