from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load data
data = pd.read_csv('/workspaces/ANA-680/StudentsPerformance.csv')
x = data[['math score', 'writing score', 'reading score']]
y = data['race/ethnicity']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from the form
        features = [
            float(request.form['math_score']),
            float(request.form['reading_score']),
            float(request.form['writing_score'])
        ]
        
        # Make prediction
        prediction = model.predict(np.array([features]))
        return render_template('index.html', prediction=prediction[0])
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
