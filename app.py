# app.py

from flask import Flask, request, render_template_string
import os
import io
import pandas as pd
from jinja2 import Template
import subprocess
from cc_prediction import main as run_prediction  # Importing main function from cc_prediction

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Customer Churn Prediction</title>
</head>
<body style="text-align: center;">
    <h1>Upload your dataset</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv" required>
        <button type="submit">Predict</button>
    </form>
</body>
</html>
''')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)

            with io.StringIO() as buffer:
                df.to_csv(buffer, index=False)
                uploaded_csv_path = os.path.join(os.getcwd(), 'uploaded_dataset.csv')
                with open(uploaded_csv_path, 'w') as f:
                    f.write(buffer.getvalue())

            result_text = run_prediction(uploaded_csv_path)

            total = 'Total Customers:'
            start_index = result_text.index(total) + len(total)
            end_index = result_text.index('\n', start_index)
            total_customers = int(result_text[start_index:end_index])

            churn_customers_text = 'Churn Customers:'
            start_index = result_text.index(churn_customers_text) + len(churn_customers_text)
            end_index = result_text.index('\n', start_index)
            churn_customers = int(result_text[start_index:end_index])

            non_churn_customers_text = 'Non-Churn Customers:'
            start_index = result_text.index(non_churn_customers_text) + len(non_churn_customers_text)
            end_index = result_text.index('\n', start_index)
            non_churn_customers = int(result_text[start_index:end_index])

            return render_template_string(f'''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Customer Churn Prediction</title>
</head>
<body style="text-align: center;">
    <h1>Prediction Results</h1>
    <p>Number of Customers: {total_customers}</p>
    <p>Number of Churn Customers: {churn_customers}</p>
    <p>Number of Non-Churn Customers: {non_churn_customers}</p>
</body>
</html>
''')

if __name__ == '__main__':
    app.run(debug=True)