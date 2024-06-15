from flask import render_template,Flask,request,redirect,url_for
import joblib
import numpy as np
app=Flask(__name__)

model=joblib.load('svc_SONAR_model.pkl')

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    features = []
    for i in range(1, 61):
        value = request.form.get(f'value{i}')
        features.append(float(value))

    # Convert to numpy array
    features_array = np.array([features])

    # Make prediction
    prediction = model.predict(features_array)
    predicted_label = prediction[0]  # Assuming a single prediction is made
    predicted_class = 'M' if predicted_label == 1 else 'R'  # Adjust as per your model's class mapping

    if predicted_class == 'M':
        return render_template('index1_2.html')
    else:
        return render_template('index1_3.html')

    if prediction == '[M]':
        return render_template('index1_2.html')
    else:
        return render_template('index1_3.html')



if __name__=='__main__':
    app.run(debug=True)