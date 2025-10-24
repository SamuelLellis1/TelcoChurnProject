import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def train_model (df):
    df = df.copy()
    if "CustomerID" in df.columns:
        df.drop("CustomerID", axis = 1, inplace = True)

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(columns=[col for col in df.columns if "Churn" in col])
    y = df["Churn_Yes"] if "Churn_Yes" in df.columns else df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treino/teste
    sm = SMOTE(random_state=42)
    X_train_res,y_train_res  = sm.fit_resample(X_train_scaled, y_train)

    # -- Modelos --
    # RandomForest
    rf = RandomForestClassifier(random_state=42,n_estimators= 100, min_samples_split= 5, min_samples_leaf= 1, max_features= 'log2', max_depth= None, class_weight= 'balanced')
    rf_params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }
    rf_search = RandomizedSearchCV(rf, rf_params, cv=5, n_iter=20,scoring='f1', n_jobs=-1, random_state=42)
    rf_search.fit(X_train_res, y_train_res)
    best_rf = rf_search.best_estimator_

    # XGBoost
    xgb = XGBClassifier(
        n_estimators= 300,
        learning_rate= 0.05,
        max_depth= 6,
        subsample= 0.8,
        colsample_bytree= 0.8,
        random_state= 42,
        use_label_encoder = False,
        eval_metric= "logloss"
    )
    xgb_params = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.3]
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_params, cv = 5, n_iter= 20, scoring= 'f1', n_jobs=-1, random_state= 42)
    xgb_search.fit(X_train_res, y_train_res)
    best_xgb = xgb_search.best_estimator_

    # Stacking
    base_models = [
        ("rf", best_rf),
        ("xgb", best_xgb)
    ]
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    stacking= StackingClassifier(
        estimators = base_models,
        final_estimator= meta_model,
        cv= 5,
        stack_method= "predict_proba",
        n_jobs= -1
    )
    stacking.fit(X_train_res,y_train_res)

    # Avaliacao
    pred = stacking.predict(X_test_scaled)
    proba = stacking.predict_proba(X_test_scaled)[:, 1]


    print("Parâmetros randomforest:",rf_search.best_params_)
    print("Relatorio de classificação",classification_report(y_test, pred))
    print("ROC_AUG:",round(roc_auc_score(y_test, proba),3))
    print("Matriz de Confusão", confusion_matrix(y_test, pred))
    return stacking