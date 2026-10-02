import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

import joblib


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)



sensor = pd.read_csv(
    DATA_DIR / "road_traffic_sensor_data.csv"
)

accident = pd.read_csv(
    DATA_DIR / "traffic_accident_data.csv"
)


print("Sensor data:", sensor.shape)
print("Accident data:", accident.shape)



sensor["Date_Time"] = pd.to_datetime(
    sensor["Date_Time"]
)

accident["Date_Time"] = pd.to_datetime(
    accident["Date_Time"]
)



sensor["Hour"] = sensor["Date_Time"].dt.hour

sensor["Date"] = sensor["Date_Time"].dt.date



vehicle_min = sensor["Vehicle_Count"].min()
vehicle_max = sensor["Vehicle_Count"].max()

speed_min = sensor["Average_Speed"].min()
speed_max = sensor["Average_Speed"].max()


vehicle_normalized = (
    (sensor["Vehicle_Count"] - vehicle_min)
    /
    (vehicle_max - vehicle_min)
)


speed_normalized = (
    (sensor["Average_Speed"] - speed_min)
    /
    (speed_max - speed_min)
)


sensor["Congestion_Index"] = (
    0.60 * vehicle_normalized * 100
    +
    0.40 * (1 - speed_normalized) * 100
)


sensor["Congestion_Index"] = (
    sensor["Congestion_Index"]
    .round(2)
)




sensor.to_csv(
    OUTPUT_DIR / "clean_sensor_data.csv",
    index=False
)



traffic_profile = (
    sensor
    .groupby(
        ["Location", "Hour"],
        as_index=False
    )
    .agg(

        Avg_Vehicle_Count=(
            "Vehicle_Count",
            "mean"
        ),

        Avg_Speed=(
            "Average_Speed",
            "mean"
        ),

        Avg_Congestion_Index=(
            "Congestion_Index",
            "mean"
        ),

        High_Congestion_Rate=(
            "Congestion_Level",
            lambda x:
            (x == "High").mean()
        )
    )
)


traffic_profile.to_csv(
    OUTPUT_DIR /
    "traffic_location_hour_profile.csv",
    index=False
)



accident["Hour"] = (
    accident["Date_Time"].dt.hour
)

accident["Date"] = (
    accident["Date_Time"].dt.date
)



severity_mapping = {

    "Minor": 1,

    "Moderate": 2,

    "Severe": 3,

    "Fatal": 4
}


accident["Accident_Severity_Score"] = (
    accident["Accident_Severity"]
    .map(severity_mapping)
)



accident["High_Risk"] = (
    accident["Accident_Severity"]
    .isin(
        ["Severe", "Fatal"]
    )
    .astype(int)
)


peak_hours = [
    7,
    8,
    9,
    17,
    18,
    19,
    20
]


accident["Peak_Hour"] = (
    accident["Hour"]
    .isin(peak_hours)
    .astype(int)
)



integrated = accident.merge(

    traffic_profile,

    on=[
        "Location",
        "Hour"
    ],

    how="left"
)

location_profile = (

    sensor
    .groupby(
        "Location",
        as_index=False
    )
    .agg(

        Location_Avg_Vehicle_Count=(
            "Vehicle_Count",
            "mean"
        ),

        Location_Avg_Speed=(
            "Average_Speed",
            "mean"
        ),

        Location_Avg_Congestion_Index=(
            "Congestion_Index",
            "mean"
        ),

        Location_High_Congestion_Rate=(
            "Congestion_Level",
            lambda x:
            (x == "High").mean()
        )
    )
)


integrated = integrated.merge(

    location_profile,

    on="Location",

    how="left"
)



integrated["Avg_Vehicle_Count"] = (
    integrated["Avg_Vehicle_Count"]
    .fillna(
        integrated[
            "Location_Avg_Vehicle_Count"
        ]
    )
)


integrated["Avg_Speed"] = (
    integrated["Avg_Speed"]
    .fillna(
        integrated[
            "Location_Avg_Speed"
        ]
    )
)


integrated["Avg_Congestion_Index"] = (
    integrated["Avg_Congestion_Index"]
    .fillna(
        integrated[
            "Location_Avg_Congestion_Index"
        ]
    )
)


integrated["High_Congestion_Rate"] = (
    integrated["High_Congestion_Rate"]
    .fillna(
        integrated[
            "Location_High_Congestion_Rate"
        ]
    )
)


integrated.to_csv(

    OUTPUT_DIR /
    "integrated_accident_traffic_data.csv",

    index=False
)




features = [

    "Location",

    "Weather_Condition",

    "Road_Condition",

    "Vehicle_Type",

    "Traffic_Density",

    "Number_of_Vehicles",

    "Hour",

    "Peak_Hour",

    "Avg_Vehicle_Count",

    "Avg_Speed",

    "Avg_Congestion_Index",

    "High_Congestion_Rate"
]


target = "High_Risk"


X = integrated[features]

y = integrated[target]




categorical_features = [

    "Location",

    "Weather_Condition",

    "Road_Condition",

    "Vehicle_Type",

    "Traffic_Density"
]


numeric_features = [

    "Number_of_Vehicles",

    "Hour",

    "Peak_Hour",

    "Avg_Vehicle_Count",

    "Avg_Speed",

    "Avg_Congestion_Index",

    "High_Congestion_Rate"
]



preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numeric",

            StandardScaler(),

            numeric_features
        )
    ]
)


model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",

            RandomForestClassifier(

                n_estimators=300,

                random_state=42,

                class_weight="balanced"
            )
        )
    ]
)



X_train, X_test, y_train, y_test = (

    train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )
)


model.fit(
    X_train,
    y_train
)




predictions = model.predict(
    X_test
)


probabilities = (
    model
    .predict_proba(X_test)[:, 1]
)




print("\n========== MODEL RESULTS ==========")

print(
    classification_report(
        y_test,
        predictions
    )
)


print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            probabilities
        ),
        4
    )
)


print(
    "Confusion Matrix:"
)


print(
    confusion_matrix(
        y_test,
        predictions
    )
)




joblib.dump(

    model,

    OUTPUT_DIR /
    "accident_risk_model.joblib"
)


location_summary = (

    integrated
    .groupby(
        "Location",
        as_index=False
    )
    .agg(

        Accidents=(
            "Accident_ID",
            "count"
        ),

        Casualties=(
            "Casualties",
            "sum"
        ),

        Avg_Severity_Score=(
            "Accident_Severity_Score",
            "mean"
        ),

        High_Risk_Rate=(
            "High_Risk",
            "mean"
        ),

        Avg_Vehicle_Count=(
            "Avg_Vehicle_Count",
            "mean"
        ),

        Avg_Speed=(
            "Avg_Speed",
            "mean"
        ),

        Avg_Congestion_Index=(
            "Avg_Congestion_Index",
            "mean"
        )
    )
)


location_summary["High_Risk_Rate"] = (

    location_summary["High_Risk_Rate"]
    * 100
).round(2)


location_summary.to_csv(

    OUTPUT_DIR /
    "dashboard_location_summary.csv",

    index=False
)



hourly_summary = (

    integrated
    .groupby(
        "Hour",
        as_index=False
    )
    .agg(

        Accidents=(
            "Accident_ID",
            "count"
        ),

        Casualties=(
            "Casualties",
            "sum"
        ),

        High_Risk_Rate=(
            "High_Risk",
            "mean"
        ),

        Avg_Congestion_Index=(
            "Avg_Congestion_Index",
            "mean"
        )
    )
)


hourly_summary["High_Risk_Rate"] = (

    hourly_summary["High_Risk_Rate"]
    * 100
).round(2)


hourly_summary.to_csv(

    OUTPUT_DIR /
    "dashboard_hourly_summary.csv",

    index=False
)


print(
    "\nPipeline completed successfully."
)

print(
    "Outputs saved in:",
    OUTPUT_DIR
)