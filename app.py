from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Load the model
try:
    model = joblib.load('models/house_price_model.pkl')
    features = joblib.load('models/features.pkl')
except:
    model = None
    features = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained yet'}), 400
    
    try:
        data = request.form.to_dict()
        input_data = [float(data[feature]) for feature in features]
        prediction = model.predict([input_data])[0]
        
        # In California Housing dataset, price is in $100,000s
        formatted_price = f"${prediction * 100000:,.2f}"
        
        return render_template('index.html', prediction_text=f'Estimated House Value: {formatted_price}')
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
