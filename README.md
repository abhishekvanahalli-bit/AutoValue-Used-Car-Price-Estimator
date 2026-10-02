# AutoValue: Used-Car Price Estimator

MSc neural-network project using the Indian CarDekho dataset.

## Project components

- Linear regression baseline.
- Keras ANN with learned brand embedding.
- NumPy ANN implemented from scratch.
- Optimizer, learning-rate, loss and target-transform experiments.
- Residual analysis by fuel and actual-price band.
- Monte Carlo dropout using 50 draws.
- Validation-set permutation feature importance.
- FastAPI prediction service and prediction logging.
- Streamlit interface with age/kilometre what-if controls.

## Environment

Packaged from Python 3.13.15.
Installed dependency versions are in requirements.txt.
A clean-environment installation check is still required.

## Installation

Extract the ZIP and open a terminal in its folder.

Create a virtual environment:

    python -m venv .venv

Activate on Windows PowerShell:

    .venv\Scripts\Activate.ps1

Activate on macOS/Linux:

    source .venv/bin/activate

Install dependencies:

    python -m pip install -r requirements.txt

## Run the application

In the first terminal, start the API:

    python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1

In a second terminal, activate the same environment and run:

    python -m streamlit run streamlit_app.py

Open the Streamlit address printed in the terminal.
API documentation: http://127.0.0.1:8000/docs
For the Colab demonstration, run notebook Cell 60 instead.

## API endpoints

- GET /health
- GET /metadata
- GET /feature-importance
- POST /predict

Successful predictions are logged in logs/predictions.jsonl.
Runtime logs are excluded from this distribution ZIP.

## Input interpretation

Vehicle age = 2020 - manufacturing_year.
Keep this reference year fixed to match training.
Estimates reflect the training dataset, not live market quotations.
Enter actual vehicle specifications and the correct mileage unit.
Individual values and categories are validated, but the application
does not fully verify that a combination describes a real vehicle.

## Prediction and uncertainty

The main price is predicted with dropout switched off.
The range uses the 5th and 95th percentiles of 50 MC-dropout draws.
This is an uncalibrated model-uncertainty range.
It is not a guaranteed 90% interval for actual selling prices.
Observed validation coverage: 49.66%.

## Evaluation

Final test metrics: results/final_test_results.csv
Test data was reserved for final evaluation.
Do not use test results for further model selection or tuning.
The linear baseline had non-positive predictions, so its log-price
R² is undefined. Those predictions remain in the raw-price metrics.

## Feature importance

Permutation importance was calculated on validation data.
It shows global model reliance, not causation or an explanation
of an individual prediction. Correlated features can share importance.

## Remaining submission items

- Include the final training notebook separately.
- Include the final report and experiment tables.
- Verify installation in a clean environment.
- Record the application demonstration.
- Include a Git repository with contributions from both members.

The training notebook contains preprocessing, split construction,
seeds, experiments, and the from-scratch NumPy implementation.
