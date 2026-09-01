# E-Waste Interactive Platform

A Flask-based web application for e-waste awareness, household e-waste prediction, and recycling traceability demonstrations.

The platform provides:

- Interactive charts showing e-waste disposal methods and average annual e-waste by income bracket.
- An awareness quiz about responsible e-waste disposal.
- A machine-learning prediction form for estimating household e-waste generation in kilograms per year.
- A traceability interface for uploading an e-waste image and generating a demonstration tracking result.

## Tech Stack

- **Language:** Python
- **Web framework:** Flask
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Model serialization:** joblib
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Templates:** Jinja2/HTML
- **Styling:** CSS

## Project Structure

```text
ewaste_platform/
├── app.py                         # Flask application and routes
├── model.py                       # Model training and evaluation script
├── requirements.txt               # Python dependencies
├── dataset/
│   └── e-waste_final.csv          # Training and application dataset
├── models/
│   └── ewaste_predictor.joblib    # Serialized prediction pipeline
├── static/
│   └── css/
│       └── style.css              # Application styles
└── templates/
    ├── index.html                 # Awareness dashboard and quiz
    ├── layout.html                # Shared page layout
    ├── predict.html               # Prediction form
    └── traceability.html          # Traceability demonstration page
```

## Getting Started

### Prerequisites

- Python 3
- `pip`
- The dataset file at `ewaste_platform/dataset/e-waste_final.csv`
- The trained model at `ewaste_platform/models/ewaste_predictor.joblib`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Harry121199/e-waste_platform.git
   cd e-waste_platform/ewaste_platform
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Run the Application

From the `ewaste_platform` directory, start the Flask application:

```bash
python app.py
```

The application runs with Flask's debug mode enabled through the `app.run(debug=True)` configuration.

Open the local address reported by Flask in a browser.

## Machine-Learning Model

The prediction pipeline is trained by `model.py` using the dataset in `dataset/e-waste_final.csv`.

### Features

The model uses the following input features:

- `state`
- `locality_type`
- `household_size`
- `income_bracket`
- `e_literacy_level`
- `total_devices_owned`
- `avg_device_age_years`
- `broken_devices_stored`
- `upgrade_tendency`
- `disposal_method`
- `recycling_awareness`

The prediction target is:

- `ewaste_kg_per_year`

Categorical features are one-hot encoded, while the remaining numeric features pass through the preprocessing pipeline. The model is a `RandomForestRegressor` with 100 estimators and `random_state=42`.

### Retrain the Model

From the `ewaste_platform` directory:

```bash
python model.py
```

The script:

1. Loads the CSV dataset.
2. Splits the data into training and test sets.
3. Trains the Random Forest regression pipeline.
4. Prints RMSE and R-squared metrics.
5. Displays actual-versus-predicted and residual-distribution charts.
6. Saves the trained model to:

   ```text
   models/ewaste_predictor.joblib
   ```

## Application Pages and Routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Displays the awareness hub, Plotly charts, and e-waste disposal quiz. |
| `GET` | `/predict` | Displays the household e-waste prediction form. |
| `POST` | `/predict` | Predicts annual household e-waste generation from submitted form data. |
| `GET` | `/traceability` | Displays the recycling traceability demonstration page. |

## Prediction Inputs

The `/predict` form accepts the following values:

| Input | Type |
|---|---|
| State | Dataset-derived selection |
| Locality type | Dataset-derived selection |
| Household size | Integer |
| Income bracket | Low, Middle, Upper-Middle, or High |
| E-literacy level | Basic, Intermediate, or Advanced |
| Total devices owned | Integer |
| Average device age | Decimal number of years |
| Broken devices stored | Integer |
| Upgrade tendency | Low, Medium, or High |
| Primary disposal method | Dataset-derived selection |
| Recycling awareness | Low, Medium, or High |

Successful predictions are displayed in the format:

```text
<number> kg/year
```

## Awareness Dashboard

The home page generates two Plotly charts from the dataset:

- Frequency of e-waste disposal methods.
- Average annual household e-waste by income bracket.

It also includes a client-side quiz about the recommended disposal method for an old mobile phone.

Plotly is loaded in the browser from the Plotly CDN.

## Traceability Demonstration

The `/traceability` page contains an image upload form. Its JavaScript currently uses a mock identification flow with predefined items such as smartphones, laptops, CRT monitors, printers, keyboards, and refrigerators.

The implementation generates a random item and tracking ID in the browser rather than uploading the image to a server or running an image-classification model.

> **Current limitation:** The traceability script references a `statuses` variable that is not defined in the provided code. As a result, submitting an image may raise a JavaScript error before the tracking result is displayed.

## Data and Model Assets

The Flask application loads these files relative to the application directory:

```text
dataset/e-waste_final.csv
models/ewaste_predictor.joblib
```

If either file is missing, the application prints an asset-loading error and exits.

## License

No license file or license declaration is included in the provided repository content.