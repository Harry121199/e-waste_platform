from flask import Flask, render_template, request
import pandas as pd
import os
import joblib
import plotly.express as px
import plotly.io as pio

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'dataset', 'e-waste_final.csv')
model_path = os.path.join(script_dir, 'models', 'ewaste_predictor.joblib') # IMPROVEMENT: Robust model path

app = Flask(__name__)

try:
    model = joblib.load(model_path)
    df = pd.read_csv(dataset_path)
except FileNotFoundError as e:
    print(f"Error loading assets: {e}")
    print("Please ensure 'e-waste_final.csv' is in the 'dataset' folder and 'ewaste_predictor.joblib' is in the 'models' folder.")
    exit()

def get_form_options():
    """Helper function to get unique options for form dropdowns."""
    options = {
        'states': sorted(df['state'].unique()),
        'localities': sorted(df['locality_type'].unique()),
        'incomes': ['Low', 'Middle', 'Upper-Middle', 'High'],
        'e_literacies':['Basic','Intermediate','Advanced'],
        'tendencies': ['Low', 'Medium', 'High'],
        'disposals': sorted(df['disposal_method'].unique()),
        'awareness': ['Low', 'Medium', 'High']
    }
    return options

def create_awareness_charts():
    df['disposal_method'] = df['disposal_method'].astype(str).str.strip().str.title()
    disposal_counts = df['disposal_method'].value_counts().reset_index()
    disposal_counts.columns = ['disposal_method', 'count']

    fig1 = px.pie(
        disposal_counts,
        values='count',
        names='disposal_method',
        title='Frequency of E-Waste Disposal Methods'
    )

    income_waste = df.groupby('income_bracket')['ewaste_kg_per_year'].mean().reset_index()
    income_order = ['Low', 'Middle', 'Upper-Middle', 'High']
    income_waste['income_bracket'] = pd.Categorical(income_waste['income_bracket'], categories=income_order, ordered=True)
    income_waste = income_waste.sort_values('income_bracket')

    fig2 = px.bar(
        income_waste,
        x='income_bracket',
        y='ewaste_kg_per_year',
        title='Average Annual E-Waste per Household by Income'
    )

    # Convert figures to JSON
    chart1_json = pio.to_json(fig1)
    chart2_json = pio.to_json(fig2)

    return chart1_json, chart2_json



# --- FLASK ROUTES ---
@app.route('/')
def index():
    """Renders the main page with awareness charts."""
    chart1_json, chart2_json = create_awareness_charts()
    return render_template('index.html', chart1_json=chart1_json, chart2_json=chart2_json)

@app.route('/predict', methods=['GET', 'POST']) 
def predict():
    prediction_result = None
    form_options = get_form_options()
    if request.method == 'POST':
        form_data = {
            'state': [request.form['state']],
            'locality_type': [request.form['locality_type']],
            'household_size': [int(request.form['household_size'])],
            'income_bracket': [request.form['income_bracket']],
            'e-literacy_level': [request.form['e_literacy_level']],
            'total_devices_owned': [int(request.form['total_devices_owned'])],
            'avg_device_age_years': [float(request.form['avg_device_age_years'])],
            'broken_devices_stored': [int(request.form['broken_devices_stored'])],
            'upgrade_tendency': [request.form['upgrade_tendency']],
            'disposal_method': [request.form['disposal_method']],
            'recycling_awareness': [request.form['recycling_awareness']]
        }
        input_data = pd.DataFrame(form_data)
        prediction = model.predict(input_data)
        prediction_result = f"{prediction[0]:.2f} kg/year"

    return render_template('predict.html', options=form_options, prediction=prediction_result)


@app.route('/traceability')
def traceability():
    """Renders the traceability page."""
    return render_template('traceability.html')

if __name__ == '__main__':
    app.run(debug=True)