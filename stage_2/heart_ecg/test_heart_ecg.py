import pandas as pd
from heart_inference import heart_ecg_inference


# Load one ECG signal from dataset
df = pd.read_csv("stage_2/heart_ecg/data/mitbih_test.csv", header=None)

signal = df.iloc[0, :-1].values


result = heart_ecg_inference(signal)

print("\n===== HEART ECG INFERENCE =====")

if result == 1:
    print("Abnormal ECG detected")
else:
    print("Normal ECG")