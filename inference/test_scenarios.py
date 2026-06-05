test_cases = [("Healthy adult",{"Age": 22, "age": 22, "Glucose": 88, "BMI": 20}, [0, 0, 0]),
("Healthy adult",{"Age": 30, "age": 30, "Glucose": 95, "BMI": 22}, [0, 0, 0]),
("Healthy adult",{"Age": 35, "age": 35, "Glucose": 100, "BMI": 24}, [0, 0, 0]),
("Healthy adult",{"Age": 40, "age": 40, "Glucose": 90, "BMI": 23}, [0, 0, 0]),
("Healthy adult",{"Age": 27, "age": 27, "Glucose": 92, "BMI": 21}, [0, 0, 0]),
("Healthy adult",{"Age": 33, "age": 33, "Glucose": 85, "BMI": 22}, [0, 0, 0]),
("Healthy adult",{"Age": 45, "age": 45, "Glucose": 98, "BMI": 24}, [0, 0, 0]),
("Healthy adult",{"Age": 50, "age": 50, "Glucose": 102, "BMI": 25}, [0, 0, 0]),
("Healthy adult",{"Age": 29, "age": 29, "Glucose": 93, "BMI": 21}, [0, 0, 0]),
("Healthy adult",{"Age": 38, "age": 38, "Glucose": 97, "BMI": 23}, [0, 0, 0]),
("Diabetes only",{"Age": 45, "age": 45, "Glucose": 160, "BMI": 31, "Insulin": 130}, [0, 1, 0]),
("Diabetes only",{"Age": 52, "age": 52, "Glucose": 190, "BMI": 36, "Insulin": 150}, [0, 1, 0]),
("Diabetes only",{"Age": 48, "age": 48, "Glucose": 170, "BMI": 33, "Insulin": 140}, [0, 1, 0]),
("Diabetes only",{"Age": 60, "age": 60, "Glucose": 200, "BMI": 38, "Insulin": 180}, [0, 1, 0]),
("Diabetes only",{"Age": 55, "age": 55, "Glucose": 175, "BMI": 34, "Insulin": 160}, [0, 1, 0]),
("Diabetes only",{"Age": 42, "age": 42, "Glucose": 150, "BMI": 30, "Insulin": 120}, [0, 1, 0]),
("Diabetes only",{"Age": 63, "age": 63, "Glucose": 210, "BMI": 39, "Insulin": 190}, [0, 1, 0]),
("Diabetes only",{"Age": 50, "age": 50, "Glucose": 165, "BMI": 32, "Insulin": 140}, [0, 1, 0]),
("Diabetes only",{"Age": 46, "age": 46, "Glucose": 155, "BMI": 31, "Insulin": 135}, [0, 1, 0]),
("Diabetes only",{"Age": 58, "age": 58, "Glucose": 185, "BMI": 35, "Insulin": 165}, [0, 1, 0]),
("Diabetes only",{"Age": 61, "age": 61, "Glucose": 195, "BMI": 37, "Insulin": 175}, [0, 1, 0]),
("Diabetes only",{"Age": 47, "age": 47, "Glucose": 168, "BMI": 32, "Insulin": 138}, [0, 1, 0]),


# =========================
# CKD ONLY (12)
# =========================
("CKD",{"Age": 60, "age": 60, "sc": 2.2, "egfr": 35, "al": 2}, [1, 0, 0]),
("CKD",{"Age": 70, "age": 70, "sc": 2.8, "egfr": 22, "al": 3}, [1, 0, 0]),
("CKD",{"Age": 65, "age": 65, "sc": 2.5, "egfr": 30, "al": 2}, [1, 0, 0]),
("CKD",{"Age": 58, "age": 58, "sc": 2.0, "egfr": 40, "al": 1}, [1, 0, 0]),
("CKD",{"Age": 75, "age": 75, "sc": 3.1, "egfr": 18, "al": 3}, [1, 0, 0]),
("CKD",{"Age": 68, "age": 68, "sc": 2.7, "egfr": 25, "al": 2}, [1, 0, 0]),
("CKD",{"Age": 72, "age": 72, "sc": 2.9, "egfr": 20, "al": 3}, [1, 0, 0]),
("CKD",{"Age": 66, "age": 66, "sc": 2.4, "egfr": 32, "al": 2}, [1, 0, 0]),
("CKD",{"Age": 63, "age": 63, "sc": 2.3, "egfr": 36, "al": 1}, [1, 0, 0]),
("CKD",{"Age": 77, "age": 77, "sc": 3.2, "egfr": 15, "al": 4}, [1, 0, 0]),
("CKD",{"Age": 69, "age": 69, "sc": 2.6, "egfr": 28, "al": 2}, [1, 0, 0]),
("CKD",{"Age": 62, "age": 62, "sc": 2.1, "egfr": 38, "al": 1}, [1, 0, 0]),


# =========================
# HEART ONLY (12)
# =========================
("Heart",{"Age": 55, "age": 55, "trestbps": 150, "chol": 260}, [0, 0, 1]),
("Heart",{"Age": 65, "age": 65, "trestbps": 170, "chol": 300}, [0, 0, 1]),
("Heart",{"Age": 60, "age": 60, "trestbps": 160, "chol": 280}, [0, 0, 1]),
("Heart",{"Age": 58, "age": 58, "trestbps": 155, "chol": 270}, [0, 0, 1]),
("Heart",{"Age": 62, "age": 62, "trestbps": 165, "chol": 290}, [0, 0, 1]),
("Heart",{"Age": 70, "age": 70, "trestbps": 175, "chol": 310}, [0, 0, 1]),
("Heart",{"Age": 59, "age": 59, "trestbps": 150, "chol": 265}, [0, 0, 1]),
("Heart",{"Age": 67, "age": 67, "trestbps": 168, "chol": 295}, [0, 0, 1]),
("Heart",{"Age": 72, "age": 72, "trestbps": 180, "chol": 320}, [0, 0, 1]),
("Heart",{"Age": 64, "age": 64, "trestbps": 158, "chol": 275}, [0, 0, 1]),
("Heart",{"Age": 57, "age": 57, "trestbps": 152, "chol": 268}, [0, 0, 1]),
("Heart",{"Age": 69, "age": 69, "trestbps": 172, "chol": 305}, [0, 0, 1]),
("Heart",{"Age": 50, "age": 50, "Glucose": 135, "BMI": 29}, [0, 0, 0]),
("Heart",{"Age": 55, "age": 55, "Glucose": 140, "BMI": 30}, [0, 0, 0]),
("Heart",{"Age": 60, "age": 60, "egfr": 60, "sc": 1.5, "al": 0}, [0, 0, 0]),
("Heart",{"Age": 65, "age": 65, "trestbps": 140, "chol": 240}, [0, 0, 0]),
("Heart",{"Age": 58, "age": 58, "Glucose": 145, "BMI": 31}, [0, 0, 0]),
("Heart",{"Age": 62, "age": 62, "egfr": 58, "sc": 1.6, "al": 0}, [0, 0, 0]),
("Heart",{"Age": 67, "age": 67, "trestbps": 142, "chol": 245}, [0, 0, 0]),
("Heart",{"Age": 52, "age": 52, "Glucose": 138, "BMI": 29}, [0, 0, 0]),


# =========================
# MISSING FEATURES (6)
# =========================
("missing features",{"Age": 62, "age": 62, "sc": 2.1}, [0, 0, 0]),
("missing features",{"Age": 45, "age": 45, "Glucose": 165}, [0, 0, 0]),
("missing features",{"Age": 70, "age": 70, "trestbps": 170}, [0, 0, 0]),
("missing features",{"Age": 55, "age": 55}, [0, 0, 0]),
("missing features",{"Glucose": 180, "BMI": 35}, [0, 0, 0]),
("missing features",{"sc": 2.4, "egfr": 30}, [0, 0, 0]),
("multiple disease",{"Age": 58, "age": 58, "Glucose": 170, "BMI": 30, "Insulin": 120,
  "sc": 1.9, "egfr": 45, "al": 1}, [1, 1, 0]),

("multiple disease",{"Age": 65, "age": 65, "Glucose": 180, "BMI": 34, "Insulin": 140,
  "sc": 2.3, "egfr": 28, "al": 2,
  "trestbps": 160, "chol": 280}, [1, 1, 1]),

("multiple disease",{"Age": 62, "age": 62, "Glucose": 175, "BMI": 32, "Insulin": 130,
  "sc": 2.2, "egfr": 35, "al": 2}, [1, 1, 0]),

("multiple disease",{"Age": 70, "age": 70, "Glucose": 190, "BMI": 36, "Insulin": 150,
  "trestbps": 170, "chol": 300}, [0, 1, 1]),

("multiple disease",{"Age": 66, "age": 66, "sc": 2.6, "egfr": 30, "al": 2,
  "trestbps": 165, "chol": 290}, [1, 0, 1]),
("multiple disease",{"Age": 63, "age": 63, "Glucose": 185, "BMI": 33, "Insulin": 145,
  "sc": 2.4, "egfr": 32, "al": 2,
  "trestbps": 160, "chol": 285}, [1, 1, 1]),

("multiple disease",{"Age": 68, "age": 68, "Glucose": 195, "BMI": 37, "Insulin": 160,
  "sc": 2.7, "egfr": 26, "al": 3}, [1, 1, 0]),

("multiple disease",{"Age": 72, "age": 72, "Glucose": 205, "BMI": 39, "Insulin": 180,
  "trestbps": 178, "chol": 315}, [0, 1, 1]),

("multiple disease",{"Age": 61, "age": 61, "sc": 2.1, "egfr": 38, "al": 1,
  "trestbps": 155, "chol": 270}, [1, 0, 1]),

("multiple disease",{"Age": 67, "age": 67, "Glucose": 178, "BMI": 31, "Insulin": 138,
  "sc": 2.3, "egfr": 34, "al": 2,
  "trestbps": 162, "chol": 282}, [1, 1, 1]),

("multiple disease",{"Age": 59, "age": 59, "Glucose": 165, "BMI": 30, "Insulin": 125,
  "trestbps": 150, "chol": 265}, [0, 1, 1]),

("multiple disease",{"Age": 64, "age": 64, "sc": 2.5, "egfr": 30, "al": 2,
  "Glucose": 172, "BMI": 33, "Insulin": 140}, [1, 1, 0]),

]