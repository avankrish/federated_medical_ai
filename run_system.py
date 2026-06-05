from inference.stage1_inference import stage1_inference

from stage_2.heart_ecg.heart_inference import heart_ecg_inference
from stage_2.kidney_ultrasound.kidney_inference import kidney_inference

def explain_stage1(patient):

    explanations = []

    if patient.get("egfr", 100) < 60:
        explanations.append("Low eGFR")

    if patient.get("sc", 0) > 1.8:
        explanations.append("High Creatinine")

    if patient.get("Glucose", 0) > 150:
        explanations.append("High Glucose")

    if patient.get("BMI", 0) > 30:
        explanations.append("High BMI")

    if patient.get("trestbps", 0) > 150:
        explanations.append("High Blood Pressure")

    if patient.get("chol", 0) > 250:
        explanations.append("High Cholesterol")

    return explanations

def run_full_system(patient_data, ecg_path=None, kidney_image=None):

    print("\n===== STAGE 1 : FEDERATED SCREENING =====")

    stage1_result = stage1_inference(patient_data)

    prediction = stage1_result["final_output"]

    ckd_risk = prediction[0]
    diabetes_risk = prediction[1]
    heart_risk = prediction[2]
    explanations=explain_stage1(patient_data)

    print("CKD Risk      :", "HIGH" if ckd_risk else "LOW")
    if ckd_risk:
        print("Reason        :", ", ".join(explanations))
    print("Diabetes Risk :", "HIGH" if diabetes_risk else "LOW")
    if ckd_risk:
        print("Reason        :", ", ".join(explanations))
    print("Heart Risk    :", "HIGH" if heart_risk else "LOW")
    if heart_risk:
        print("Reason       :",",".join(explanations))


    print("\n===== STAGE 2 : CONFIRMATION =====")

    if heart_risk and ecg_path:

        heart_result = heart_ecg_inference(ecg_path)

        print("\nHeart ECG Result:")
        print(heart_result)

    if ckd_risk and kidney_image:

        kidney_result = kidney_inference(kidney_image)

        print("\nKidney Ultrasound Result:")
        print(kidney_result)

    if not heart_risk and not ckd_risk:

        print("\nNo further imaging required.")



if __name__ == "__main__":

    patient = {

        "Age": 65,
        "age": 65,

        "Glucose": 180,
        "BMI": 34,
        "Insulin": 140,

        "sc": 2.3,
        "egfr": 28,
        "al": 2,

       # "trestbps": 160,
        #"chol": 280
    }


    run_full_system(
        patient_data=patient,
        ecg_path="stage_2/heart_ecg/mitbih_train.csv",
        kidney_image="stage_2/kidney_ultrasound/data/Stone/Stone_11.jpg"
    )