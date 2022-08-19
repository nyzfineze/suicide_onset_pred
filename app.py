from flask import Flask, render_template, request, redirect
import pickle
import numpy as np

model = pickle.load(open('final_model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/resume')
def resume():
    return render_template('resume.html')


@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/model')
def lr_model():
    return render_template('model.html')

@app.route('/predict', methods=['POST'])
def prediction():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['l']
    data13 = request.form['m']
    data14 = request.form['n']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)



if __name__ == "__main__":
    app.run(debug=True)















