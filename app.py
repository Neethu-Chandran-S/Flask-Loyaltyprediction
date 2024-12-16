from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
#import joblib

app = Flask(__name__)
# Load the trained models and encoders
#scaler = joblib.load('std_scalar (1).pkl')



#best_model = joblib.load('lr_newmodel.pkl')


@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')




@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route("/prediction", methods = ["GET","POST"])
def predicted():
        if request.method == 'POST':
             
             revenue = request.form['revenue'] 
             total_spent = request.form['total_spent']
             satisfaction_score = request.form['satisfaction_score'] 
             
             
             loyalty_prediction =  {
                                    "Revenue": revenue,
                                    "Total Spent": total_spent,
                                    "Satisfaction Score": satisfaction_score,
                                   
                                    
                                   
                                    
                                    }
             
             #dataframe
             loyalty_predictdf = pd.DataFrame([loyalty_prediction])

             # encoding
            # encoder = pickle.load(open('l_encoder (1).pkl','rb'))
            
             x = loyalty_predictdf
             print(x)

             #scaling
             scalar = pickle.load(open('std_scalar (1).pkl','rb'))

            
             loyalty_predict_scaled = scalar.transform(loyalty_predictdf)
             
             print("Scaled ", loyalty_predict_scaled)
             #x = loyalty_predict_scaled

             #modeling
             pickled_model = pickle.load(open('lr_newmodel.pkl','rb'))

             loyalty_score = pickled_model.predict(loyalty_predict_scaled)[0] 

             return render_template('pred.html',revenue = revenue, 
             total_spent = total_spent,
             satisfaction_score = satisfaction_score,
             loyalty_score =loyalty_score
             )
     
             
             
         

if __name__ == "__main__":
    app.run()