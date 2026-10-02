import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

sensor_path = BASE_DIR / "data" / "road_traffic_sensor_data.csv"
accident_path = BASE_DIR / "data" / "traffic_accident_data.csv"

sensor = pd.read_csv(sensor_path)
accident = pd.read_csv(accident_path)

print("\n========== SENSOR DATA ==========")
print(sensor.head())
print(sensor.shape)
print(sensor.info())
print(sensor.isnull().sum())

print("\n========== ACCIDENT DATA ==========")
print(accident.head())
print(accident.shape)
print(accident.info())
print(accident.isnull().sum())

print("\nSensor locations:")
print(sensor["Location"].value_counts())

print("\nAccident severity:")
print(accident["Accident_Severity"].value_counts())

print("\nAccident locations:")
print(accident["Location"].value_counts())