
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Load files relative to this API file
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"

MODEL = tf.keras.models.load_model(
    str(ARTIFACTS / "car_price_model.keras"),
    compile=False
)

PREP = joblib.load(ARTIFACTS / "preprocessing.joblib")

with open(ARTIFACTS / "app_metadata.json", encoding="utf-8") as file:
    META = json.load(file)

IMPORTANCE = pd.read_csv(ARTIFACTS / "feature_importance.csv")

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MODEL_LOCK = Lock()

app = FastAPI(title="Used-Car Price Estimator API", version="1.0")


# Required car details
class CarDetails(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True
    )

    brand: str = Field(min_length=1, max_length=80)

    car_age: float = Field(
        ge=0,
        description="Saved age_reference_year minus manufacturing year."
    )

    km_driven: float = Field(ge=0)
    mileage_value: float = Field(gt=0)
    engine: float = Field(gt=0)
    max_power: float = Field(gt=0)
    seats: int = Field(ge=1)

    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage_unit: str


@app.get("/health")
def health():
    return {
        "status": "ready",
        "experiment": META["selected_experiment"]
    }


@app.get("/metadata")
def metadata():
    return META


@app.get("/feature-importance")
def feature_importance():
    return IMPORTANCE.to_dict(orient="records")


@app.post("/predict")
def predict(car: CarDetails):
    row = car.model_dump()
    notes = []

    # Match categorical values to the original training labels
    for column, options in META["categorical_options"].items():
        lookup = {value.casefold(): value for value in options}
        supplied = row[column].casefold()

        if supplied not in lookup:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid {column}. Allowed values: {options}"
            )

        row[column] = lookup[supplied]

    # Preserve the original brand mapping
    brand_lookup = {
        value.casefold(): value
        for value in META["brand_options"]
    }

    row["brand"] = brand_lookup.get(
        row["brand"].casefold(),
        row["brand"]
    )

    brand_id = PREP["brand_to_id"].get(
        row["brand"],
        PREP["unknown_brand_id"]
    )

    if brand_id == PREP["unknown_brand_id"]:
        notes.append(
            "This brand uses the shared rare/unseen-brand representation."
        )

    # Identify inputs outside the numeric ranges seen during training
    for column, bounds in META["numeric_training_summary"].items():
        if not bounds["min"] <= row[column] <= bounds["max"]:
            notes.append(f"{column} is outside its observed training range.")

    frame = pd.DataFrame([row])

    # Apply the saved preprocessing without fitting anything
    try:
        with np.errstate(over="raise", invalid="raise"):
            numeric = PREP["numeric_scaler"].transform(
                frame[PREP["numeric_columns"]]
            )

            categories = PREP["ann_encoder"].transform(
                frame[PREP["categorical_columns"]]
            )

            if hasattr(categories, "toarray"):
                categories = categories.toarray()

            other = np.hstack(
                [numeric, categories]
            ).astype("float32")

    except (ValueError, FloatingPointError):
        raise HTTPException(
            status_code=422,
            detail="Numeric inputs could not be processed. Check their units and values."
        )

    if not np.isfinite(other).all():
        raise HTTPException(
            status_code=422,
            detail="Processed inputs contain non-finite values."
        )

    inputs = {
        "brand": np.array([[brand_id]], dtype="int32"),
        "other": other
    }

    def predict_price(dropout_on):
        scaled = MODEL(inputs, training=dropout_on).numpy()

        target = PREP["target_scaler"].inverse_transform(
            scaled
        ).astype("float64").ravel()[0]

        try:
            with np.errstate(over="raise", invalid="raise"):
                price = (
                    float(np.exp(target))
                    if PREP["uses_log_target"]
                    else float(target)
                )
        except FloatingPointError:
            raise HTTPException(
                status_code=422,
                detail="These inputs produced an invalid price estimate."
            )

        if not np.isfinite(price) or price <= 0:
            raise HTTPException(
                status_code=422,
                detail="These inputs produced an invalid price estimate."
            )

        return price

    # Serialize model calls and log writes within this process
    with MODEL_LOCK:
        # Normal prediction: dropout OFF
        point_price = predict_price(False)

        # Fifty stochastic predictions: dropout ON
        sample_count = int(META["uncertainty"]["samples"])

        samples = np.array([
            predict_price(True)
            for _ in range(sample_count)
        ])

        percentiles = [
            META["uncertainty"]["lower_percentile"],
            META["uncertainty"]["upper_percentile"]
        ]

        low, high = np.percentile(samples, percentiles)

        result = {
            "prediction_inr": point_price,
            "prediction_lakh": point_price / 100000,
            "low_inr": float(low),
            "high_inr": float(high),
            "low_lakh": float(low / 100000),
            "high_lakh": float(high / 100000),
            "mc_samples": sample_count,
            "range_label": META["uncertainty"]["label"],
            "range_calibrated": False,
            "validation_coverage_percent":
                META["uncertainty"]["validation_coverage_percent"],
            "age_reference_year": META["age_reference_year"],
            "notes": notes
        }

        # Record each successful prediction
        log_entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": META["selected_experiment"],
            "inputs": row,
            "result": result
        }

        with open(
            LOG_DIR / "predictions.jsonl",
            "a",
            encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(log_entry, ensure_ascii=False, allow_nan=False)
                + "\n"
            )

    return result
