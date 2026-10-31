import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from preprocessing import load_data, preprocess_data

def train_model():
    mlflow.set_experiment("telco_churn_experiment")
    
    with mlflow.start_run():
        print("Cargando y preprocesando datos...")
        df = load_data()
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
        
        n_estimators = 100
        random_state = 42
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("class_weight", "balanced")
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators, 
                random_state=random_state, 
                class_weight='balanced'
            ))
        ])
        
        print("Entrenando el modelo...")
        model.fit(X_train, y_train)
        
        print("Evaluando el modelo en el conjunto de prueba...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Registrar el modelo indicando explícitamente que confiamos en los árboles de Scikit-learn
        mlflow.sklearn.log_model(
            model, 
            name="model", 
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        
        print(f"\n--- Métricas del Modelo (Registradas en MLflow) ---")
        print(f"ROC-AUC Score: {roc_auc:.4f}\n")
        print(classification_report(y_test, y_pred))
        
    return model

if __name__ == "__main__":
    train_model()
