CKD – Model Comparison Notes

Logistic Regression:
- Achieved high accuracy (95%)
- Precision warnings indicate biased predictions
- Likely affected by class imbalance
- Suitable as baseline but not final model

MLP:
- Slower convergence but stable learning
- Improved balance between precision and recall
- Captured non-linear interactions between eGFR and albumin
- Selected as final CKD client model


Diabetes -Model Comparison Notes

Both logistic regression and MLP achieved similar performance on the diabetes dataset. This indicates that diabetes classification is primarily driven by glucose levels, with limited benefit from complex non-linear modeling. Therefore, a simpler and more interpretable model is sufficient.
