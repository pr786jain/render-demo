from flask import Flask ,render_template,request,url_for
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)
model=pickle.load(open("linear_Reg.pkl","rb"))

@app.route('/',methods=['GET'])
def welcome():
    return render_template('index.html')

standar_to=StandardScaler()
@app.route("/predict",methods=['POST'])

def predict():
    if request.method =='POST':
      yearexp=float(request.form['number'])
      x_future = np.array([[yearexp]]).reshape(-1, 1)
      prediction=model.predict(x_future)
      return render_template('test.html', prediction_text="Your salary is {:.2f} RS".format(prediction[0]))

    else:
      return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)