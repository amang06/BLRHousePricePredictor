from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    area = request.form.get('area')
    print(location,bhk,bath,area)
    input = pd.DataFrame([[location, area, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    return str(np.round(pipe.predict(input)[0] * 1e5, 2))


if __name__ == '__main__':
    app.run(debug=True)
