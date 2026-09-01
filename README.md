# E-Waste Interactive Platform

A Flask-based web application for e-waste awareness, household e-waste prediction, and recycling traceability demonstrations.

The platform includes:

- Interactive charts for e-waste disposal methods and average annual e-waste by income bracket
- An awareness quiz about responsible e-waste disposal
- A machine-learning form for predicting annual household e-waste generation
- A browser-based traceability demonstration for uploaded e-waste images

## Tech Stack

- **Language:** Python
- **Web framework:** Flask
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Model serialization:** joblib
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Templating:** Jinja2 and HTML
- **Styling:** CSS

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

The application starts with Flask debug mode enabled. Open the local address reported by Flask in a browser.

The application loads the dataset and serialized model relative to the application directory. If either required asset is missing, the application prints an error and exits.

## Application Routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Displays the awareness hub, charts, and e-waste disposal quiz |
| `GET` | `/predict` | Displays the household e-waste prediction form |
| `POST` | `/predict` | Predicts annual household e-waste generation from submitted form data |
| `GET` | `/traceability` | Displays the recycling traceability demonstration |

## Awareness Dashboard

The home page provides:

- A Plotly pie chart showing the frequency of e-waste disposal methods
- A Plotly bar chart showing average annual household e-waste by income bracket
- A client-side quiz about the recommended disposal method for an old mobile phone

Plotly is loaded in the browser from the Plotly CDN.

## Household E-Waste Prediction

The `/predict` form accepts the following inputs:

| Input | Type or values |
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

Successful predictions are displayed in kilograms per year:

```text
<number> kg/year
```

### Model Details

The prediction pipeline uses:

- One-hot encoding for categorical features
- A `StandardScaler` configured for sparse data
- A `RandomForestRegressor` with 100 estimators
- `random_state=42`

The model features are:

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

The prediction target is `ewaste_kg_per_year`.

## Retrain the Model

From the `ewaste_platform` directory, run:

```bash
python model.py
```

The training script:

1. Loads `dataset/e-waste_final.csv`
2. Splits the data into training and test sets
3. Trains the Random Forest regression pipeline
4. Prints RMSE and R-squared metrics
5. Displays actual-versus-predicted and residual-distribution charts
6. Saves the trained model to:

   ```text
   models/ewaste_predictor.joblib
   ```

The script creates the `models` directory if it does not already exist.

## Traceability Demonstration

The `/traceability` page provides an image upload interface. The current implementation does not send the image to the Flask server or perform image classification.

Instead, the browser uses predefined e-waste items and statuses to generate a demonstration result. It randomly selects:

- An item, such as a smartphone, laptop, CRT monitor, printer, keyboard, or refrigerator
- A recycling status
- A tracking ID beginning with `EW-`

The image must be selected before submitting the form.

## Project Structure

```text
ewaste_platform/
├── app.py                          # Flask application and routes
├── model.py                        # Model training and evaluation script
├── requirements.txt                # Python dependencies
├── dataset/
│   └── e-waste_final.csv           # Dataset used by the application and model
├── models/
│   └── ewaste_predictor.joblib     # Serialized prediction pipeline
├── static/
│   └── css/
│       └── style.css               # Application styles
└── templates/
    ├── index.html                  # Awareness dashboard and quiz
    ├── layout.html                 # Shared page layout
    ├── predict.html                # Prediction form
    └── traceability.html           # Traceability demonstration page
```

## Testing

No automated test suite or test configuration is included in the repository content.

## License

No license file or license declaration is included in the repository content.