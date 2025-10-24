import joblib
from sklearn.metrics import roc_auc_score, confusion_matrix

def evaluate_model (model, X_test, y_test):
    pred = model.predict(model, X_test, y_test)
    print('ROC_AUG:', roc_auc_score(y_test, pred))
    print("Confusion Matrix:", confusion_matrix(y_test, pred))