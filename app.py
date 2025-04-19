from flask import Flask, render_template, request, session
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import uuid
import os
import sqlite3
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'secret-key-123'  # Required for session

# Load models and scalers
classifier = pickle.load(open('best_clf.pkl', 'rb'))
regressor = pickle.load(open('clf2.pkl', 'rb'))
scaler1 = pickle.load(open('scale.pkl', 'rb'))
scaler2 = pickle.load(open('scale2nd.pkl', 'rb'))

# SQLite DB setup
def init_db():
    conn = sqlite3.connect('loan_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            User TEXT,
            Mobile TEXT,
            Gmail TEXT,
            login_date TEXT,
            login_time TEXT,
            Loan_status INTEGER,
            Loan_amount INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def home():
    session['name'] = request.form['name']
    session['mobile'] = request.form['mobile']
    session['gmail'] = request.form['gmail']
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Input values
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

    # Column names (match training)
    columns = [
        "no_of_dependents", "education", "self_employed",
        "income_annum", "loan_term", "cibil_score",
        "residential_assets_value", "commercial_assets_value",
        "luxury_assets_value", "bank_asset_value"
    ]

    features = [[
        no_of_dependents, education, self_employed,
        income_annum, loan_term, cibil_score,
        residential_assets_value, commercial_assets_value,
        luxury_assets_value, bank_asset_value
    ]]
    input_df = pd.DataFrame(features, columns=columns)

    # Scale + predict
    scaled_features = scaler1.transform(input_df)
    classification = classifier.predict(scaled_features)[0]

    amount = 0
    plot_url = None

    if classification == 0:
        message = "❌ Sorry, you are not eligible for a loan."
        st = 0
    else:
        message = "✅ Congratulations! You are eligible for a loan."
        scaled_input = scaler2.transform(input_df)
        amount = regressor.predict(scaled_input)[0] * 10000000

        # Create a unique plot
        plot_id = uuid.uuid4().hex
        plot_path = f'static/plots/plot_{plot_id}.png'  
        
        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, amount], marker='o', label=f'₹{amount:,.2f}', color='green')
        plt.ylim(0, amount * 1.10)
        plt.xticks([0, 1], ['Min Amount', 'Max Amount'])
        plt.xlabel("Loan Amount Scale")
        plt.ylabel('Amount (in ₹)')
        plt.title('Loan Amount Visualization')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plot_url = plot_path
        st = 1

    # Save user info
    name = session.get('name')
    mobile = session.get('mobile')
    gmail = session.get('gmail')
    now = datetime.now()
    login_date = now.date().isoformat()
    login_time = now.time().strftime('%H:%M:%S')

    # Insert into SQLite
    conn = sqlite3.connect('loan_data.db')
    cursor = conn.cursor()
    insert_query = '''
        INSERT INTO search_history 
        (User, Mobile, Gmail, login_date, login_time, Loan_status, Loan_amount)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    '''
    data = (name, mobile, gmail, login_date, login_time, st, int(amount))
    cursor.execute(insert_query, data)
    conn.commit()
    conn.close()

    return render_template("result.html", message=message, amount=amount, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
