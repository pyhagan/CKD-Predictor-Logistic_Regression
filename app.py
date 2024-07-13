from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the dataset and prepare it
ckd = pd.read_csv('kidney_disease_complete_2c.csv')

X = ckd.drop(columns='classification', axis=1)
Y = ckd['classification']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the CKD Prediction API!"


def diet_plan(result):
    if result == "The patient is likely suffering from CKD":
        diet_suggestion = """
        To improve your condition, do the following:
        
        - Consult with a dietitian for a personalized plan.
        - Drink adequate water, but not excessively.
        - Minimize intake of fizzy drinks.
        - Avoid alcohol
        - Limit protein intake to reduce the workload on kidneys.
        - Choose high-quality protein sources like fish, poultry, and eggs.
        - Reduce sodium intake to control blood pressure.
        - Limit foods high in phosphorus such as dairy, nuts, seeds.
        - Limit potassium-rich foods such as bananas, oranges, potatoes.
        """
    else:
        diet_suggestion = """
        Diet Suggestions for Healthy Kidneys:
        
        - Eat plenty of fruits and vegetables.
        - Minimize intake of fizzy drinks.
        - Minimize alcohol intake
        - Maintain a balanced diet with a variety of foods.
        - Ensure adequate hydration by drinking enough water.
        - Include whole grains and lean proteins.
        - Limit intake of processed foods and high-sodium snacks.
        - Avoid excessive amounts of sugar and saturated fats.
        """
    return diet_suggestion
    
@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        data = request.json
        input_data = [
            data['age'], data['diastolic bp'], data['sg'], data['al'], data['su'], data['bgr'], data['bu'],
            data['sc'], data['sod'], data['pot'], data['hemo'], data['pcv'], data['wc'], data['rc'],
            data['rbc'], data['pc'], data['pcc'], data['ba'], data['htn'], data['dm'], data['cad'],
            data['pe'], data['ane']
        ]
        
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        input_data_df = pd.DataFrame(input_data_as_numpy_array, columns=X.columns)
        
        prediction = model.predict(input_data_df)
        prediction_proba = model.predict_proba(input_data_df)
        
        probability = prediction_proba[0][prediction[0]] * 100
        
        if prediction[0] == 0:
            prediction_result = f'The patient is likely suffering from CKD with a probability of {probability:.2f}%'
            diet_suggestion = diet_plan("The patient is likely suffering from CKD")
        else:
            prediction_result = f'Patient is healthy with a probability of {probability:.2f}%'
            diet_suggestion = diet_plan("Patient is healthy")
        
        return jsonify({'Prediction': prediction_result, 'Diet Suggestion': diet_suggestion})
    
    except Exception as e:
        return jsonify({'error': str(e)})





if __name__ == '__main__':
    app.run(debug=True)
