import numpy as np
from stage1_inference import stage1_inference


def stress_test_ckd():
    print("\n=== CKD STRESS TEST ===")
    egfr_values = [90, 75, 60, 45, 30]

    base_input = {
        "age": 60,
        "Age": 60,
        "sc": 1.9,
        "al": 1
    }

    for egfr in egfr_values:
        user = base_input.copy()
        user["egfr"] = egfr
        output = stage1_inference(user)

        print(
            f"eGFR={egfr:>3} → "
            f"CKD_prob={output['probabilities'].get('CKD', None):.3f}, "
            f"CKD_pred={output['final_output'][0]}"
        )


def stress_test_diabetes():
    print("\n=== DIABETES STRESS TEST ===")
    glucose_values = [90, 110, 125, 140, 180]

    base_input = {
        "Age": 45,
        "age": 45,
        "BMI": 28,
        "Insulin": 100
    }

    for g in glucose_values:
        user = base_input.copy()
        user["Glucose"] = g
        output = stage1_inference(user)

        print(
            f"Glucose={g:>3} → "
            f"Diabetes_prob={output['probabilities'].get('Diabetes', None):.3f}, "
            f"Diabetes_pred={output['final_output'][1]}"
        )


def stress_test_heart():
    print("\n=== HEART STRESS TEST ===")
    bp_values = [110, 130, 150, 170]

    base_input = {
        "age": 55,
        "Age": 55,
        "chol": 220
    }

    for bp in bp_values:
        user = base_input.copy()
        user["trestbps"] = bp
        output = stage1_inference(user)

        print(
            f"BP={bp:>3} → "
            f"Heart_prob={output['probabilities'].get('Heart', None):.3f}, "
            f"Heart_pred={output['final_output'][2]}"
        )


if __name__ == "__main__":
    print("\n===== RUNNING STAGE-1 STRESS TESTS =====")
    stress_test_ckd()
    stress_test_diabetes()
    stress_test_heart()
