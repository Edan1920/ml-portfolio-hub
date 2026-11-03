import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import load_data, preprocess_data

def train_model():
    # Configurar el experimento en MLflow para B2B NBO
    mlflow.set_experiment("b2b_nbo_experiment")
    
    with mlflow.start_run():
        print("Cargando y preprocesando datos B2B...")
        df = load_data()
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
        
        # Definir hiperparámetros
        n_estimators = 150
        random_state = 42
        
        # Registrar parámetros en MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("task", "multiclass_recommendation")
        
        # Construir Pipeline completo
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators, 
                random_state=random_state, 
                class_weight='balanced'
            ))
        ])
        
        print("Entrenando el modelo de recomendación NBO...")
        model.fit(X_train, y_train)
        
        print("Evaluando el modelo en el conjunto de prueba...")
        y_pred = model.predict(X_test)
        
        # Calcular métrica principal (Accuracy para multiclase)
        acc = accuracy_score(y_test, y_pred)
        
        # Registrar métrica en MLflow
        mlflow.log_metric("accuracy", acc)
        
        # Registrar el modelo con seguridad para skops
        mlflow.sklearn.log_model(
            model, 
            name="model", 
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        
        print(f"\n--- Métricas del Modelo B2B NBO (Registradas en MLflow) ---")
        print(f"Accuracy Score: {acc:.4f}\n")
        print(classification_report(y_test, y_pred))
        
    return model

if __name__ == "__main__":
    train_model()
