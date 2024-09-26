from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model..pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    math_score = float(request.form['math_score'])
    reading_score = float(request.form['reading_score'])
    writing_score = float(request.form['writing_score'])
    
    # Prepare the input data for the model
    input_features = np.array([[math_score, reading_score, writing_score]])
    
    # Make prediction
    prediction = model.predict(input_features)
    
    # Map prediction to class name (optional, adjust if needed)
    # For example: mapping 0 to 'Group A', 1 to 'Group B', etc.
    class_names = ['Group A', 'Group B', 'Group C', 'Group D', 'Group E']
    result = class_names[int(prediction[0])]

    return f'<h1>Predicted Race/Ethnicity: {result}</h1>'

if __name__ == '__main__':
    app.run(debug=True)
