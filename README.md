# E-Waste Platform

The E-Waste Platform is a Flask web application for exploring electronic-waste information and predicting annual household e-waste generation. It provides awareness charts, a prediction form, and a traceability page.

## Features

- Displays the frequency of e-waste disposal methods.
- Shows average annual e-waste per household by income bracket.
- Predicts annual household e-waste generation in kilograms.
- Provides a traceability page.
- Uses a saved machine-learning pipeline for predictions.

## Technology Stack

- **Language:** Python 3
- **Web framework:** Flask
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Model serialization:** joblib
- **Charts:** Plotly
- **Model evaluation and visualization:** Matplotlib, Seaborn

## Getting Started

### Prerequisites

- Python 3
- The included dataset:
  - `ewaste_platform/dataset/e-waste_final.csv`
- The saved model:
  - `ewaste_platform/models/ewaste_predictor.joblib`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Harry121199/e-waste_platform.git
   cd e-waste_platform
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment.

   **macOS/Linux:**

   ```bash
   source .venv/bin/activate
   ```

   **Windows PowerShell:**

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. Install the dependencies:

   ```bash
   pip install -r ewaste_platform/requirements.txt
   ```

### Run the Application

Start the Flask application from the application directory:

```bash
cd ewaste_platform
python app.py
```

The application runs with Flask's debug mode enabled. Open the local address displayed in the terminal in a web browser.

## Web Pages

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Displays e-waste awareness charts. |
| `GET` | `/predict` | Displays the prediction form. |
| `POST` | `/predict` | Submits prediction inputs and displays the estimated annual e-waste. |
| `GET` | `/traceability` | Displays the traceability page. |

## Making a Prediction

The prediction form accepts the following information:

- State
- Locality type
- Household size
- Income bracket
- E-literacy level
- Total devices owned
- Average device age in years
- Broken devices stored
- Upgrade tendency
- Disposal method
- Recycling awareness

The result is displayed in kilograms per year using the format:

```text
0.00 kg/year
```

The available state, locality, and disposal-method options are loaded from the dataset. Other form choices are defined by the application.

## Machine-Learning Model

The training script is located at `ewaste_platform/model.py`.

The model:

1. Loads `dataset/e-waste_final.csv`.
2. Uses household, device, awareness, and disposal information as input features.
3. One-hot encodes categorical features.
4. Passes numeric features through the preprocessing pipeline.
5. Applies a `RandomForestRegressor` with 100 estimators and `random_state=42`.
6. Splits the data into training and test sets using `test_size=0.8` and `random_state=42`.
7. Evaluates predictions using RMSE and R-squared.
8. Displays actual-versus-predicted and residual-distribution charts.
9. Saves the trained pipeline to `models/ewaste_predictor.joblib`.

To retrain the model:

```bash
cd ewaste_platform
python model.py
```

The web application loads the saved model when it starts. Keep the dataset and model in their existing directories.

## Project Structure

```text
.
├── README.md
└── ewaste_platform/
    ├── app.py                         # Flask application and routes
    ├── model.py                       # Model training and evaluation
    ├── requirements.txt               # Python dependencies
    ├── dataset/
    │   └── e-waste_final.csv          # Dataset used for charts and training
    ├── models/
    │   └── ewaste_predictor.joblib    # Saved prediction pipeline
    ├── static/
    │   └── css/
    │       └── style.css              # Application styles
    └── templates/
        ├── index.html                 # Main awareness page
        ├── layout.html                # Shared page layout
        ├── predict.html               # Prediction form
        └── traceability.html          # Traceability page
```