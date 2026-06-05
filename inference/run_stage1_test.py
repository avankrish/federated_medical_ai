from stage1_inference import stage1_inference
from test_scenarios import test_cases
from sklearn.metrics import precision_score,recall_score,f1_score
import numpy as np


def run_tests():

    print("\n===== STAGE-1 COMPREHENSIVE EVALUATION =====\n")

    total = len(test_cases)
    exact_match = 0

    y_true = []
    y_pred = []

    for i, (name, input_data, expected) in enumerate(test_cases, start=1):

        output = stage1_inference(input_data)
        predicted = output["final_output"]

        y_true.append(expected)
        y_pred.append(predicted)

        print(f"Test {i}: {name}")
        print("Input:", input_data)
        print("Predicted:", predicted)
        print("Expected :", expected)

        if predicted == expected:
            print("Result: ✅ PASS\n")
            exact_match += 1
        else:
            print("Result: ❌ FAIL\n")

        print("-" * 50)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ==============================
    # METRICS
    # ==============================

    # Exact match accuracy
    exact_accuracy = exact_match / total

    # Hamming accuracy (per label)
    hamming_acc = (y_true == y_pred).mean()

    # Precision / Recall / F1
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # False negative rate
    fnr = FN / (FN + TP + 1e-8)
    # Micro
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    # Macro
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

   
        # ==============================
        # PRINT SUMMARY
        # ==============================

    print("\n===== SUMMARY METRICS =====")
    print(f"Total tests           : {total}")
    print(f"Exact Match Accuracy  : {exact_accuracy:.2%}")
    print(f"Hamming Accuracy      : {hamming_acc:.2%}")
    print(f"Precision             : {precision:.3f}")
    print(f"Recall (Sensitivity)  : {recall:.3f}")
    print(f"F1 Score              : {f1:.3f}")
    print(f"False Negative Rate   : {fnr:.3f}")
    print("\n=== MICRO AVERAGE ===")
    print(f"Precision_micro : {precision_micro:.3f}")
    print(f"Recall_micro    : {recall_micro:.3f}")
    print(f"F1_micro        : {f1_micro:.3f}")

    print("\n=== MACRO AVERAGE ===")
    print(f"Precision_macro : {precision_macro:.3f}")
    print(f"Recall_macro    : {recall_macro:.3f}")
    print(f"F1_macro        : {f1_macro:.3f}")


if __name__ == "__main__":
    run_tests()