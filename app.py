from flask import Flask, render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import uuid
import os
import mysql.connector
from datetime import datetime


app = Flask(__name__)
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="@appu1111abi",
    database="loan_data"
)
cursor = conn.cursor()
# Load models and scalers
classifier = pickle.load(open('best_clf.pkl', 'rb'))
regressor = pickle.load(open('clf2.pkl', 'rb'))
scaler1 = pickle.load(open('scale.pkl', 'rb'))
scaler2 = pickle.load(open('scale2nd.pkl', 'rb'))

@app.route('/')
def login():
    return render_template('login.html')  # Show login page first

@app.route('/login',methods=['POST'])
def home():
    global name, mobile, gmail 
    name= request.form['name']
    mobile = int(request.form['mobile'])
    gmail= request.form['gmail']
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    no_of_dependents = int(request.form['no_of_dependents'])
    education = int(request.form['education'])
    self_employed = int(request.form['self_employed'])
    income_annum = int(request.form['income_annum'])
    loan_term = int(request.form['loan_term'])
    cibil_score = int(request.form['cibil_score'])
    residential_assets_value = int(request.form['residential_assets_value'])
    commercial_assets_value = int(request.form['commercial_assets_value'])
    luxury_assets_value = int(request.form['luxury_assets_value'])
    bank_asset_value = int(request.form['bank_asset_value'])

    # Combine features
    features = [[
        no_of_dependents, education, self_employed,
        income_annum, loan_term, cibil_score,
        residential_assets_value, commercial_assets_value,
        luxury_assets_value, bank_asset_value
    ]]

    # Scale features and predict
    scaled_features = scaler1.transform(features)
    classification = classifier.predict(scaled_features)[0]

    amount = None
    plot_url = None

    if classification == 0:
        message = "❌ Sorry, you are not eligible for a loan."
        st=0
        amount=0
    else:
        message = "✅ Congratulations! You are eligible for a loan."
        amount = regressor.predict(scaler2.transform(features))[0] * 10000000

        # Create plot
        plot_id = uuid.uuid4().hex
        plot_path = f'static/plot_{plot_id}.png'

        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, amount], marker='o', label=f'₹{amount:,.2f}', color='green')
        plt.ylim(0, amount*1.10)
        plt.xticks([0, 1], ['Min Amount', 'Max Amount'])  # Clean X-axis scale
        plt.xlabel("Loan Amount Scale")
        plt.ylabel('Amount (in ₹)')
        plt.title('Loan Amount Visualization')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plot_url = plot_path   
        st=1
    
    global name, mobile, gmail 
    # geting date time
    now = datetime.now()
    login_date = now.date()
    login_time = now.time()
    #insert to database
    insert_query = """
    INSERT INTO search_history 
    (User, Mobile, Gmail, login_date, login_time, Loan_status, Loan_amount)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    data = (name, mobile, gmail, login_date, login_time,st,int(amount))
    cursor.execute(insert_query, data)    
    conn.commit()
    cursor.close()
    conn.close()
    return render_template("result.html", message=message, amount=amount, plot_url=plot_url)
 

if __name__ == '__main__':
    app.run(debug=True)
 