
# Evaluation Report

## Metrics
- **Accuracy**: 0.7950
- **Precision**: 0.7185
- **Recall**: 0.9700
- **F1 Score**: 0.8255
- **MCC**: 0.6298
- **Specificity (TNR)**: 0.6200
- **False Positive Rate (FPR)**: 0.3800
- **PR AUC**: 0.9061

## Visualizations
### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Confidence Distribution
![Confidence Distribution](confidence_dist.png)

### Precision–Recall Curve
![Precision–Recall Curve](precision_recall_curve.png)

## Classification Report
```
              precision    recall  f1-score   support

   Authentic       0.95      0.62      0.75       100
 Manipulated       0.72      0.97      0.83       100

    accuracy                           0.80       200
   macro avg       0.84      0.79      0.79       200
weighted avg       0.84      0.80      0.79       200

```
    