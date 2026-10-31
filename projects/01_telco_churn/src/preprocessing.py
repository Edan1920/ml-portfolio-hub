import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path="projects/01_telco_churn/data/raw/telco_churn_synthetic.csv"):
    """Carga el dataset sintético."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Separa variables predictoras y objetivo, y construye el pipeline de transformación."""
    X = df.drop(columns=['customer_id', 'churn'])
    y = df['churn']
    
    # Identificar columnas numéricas y categóricas
    numeric_features = ['tenure', 'monthly_charges']
    categorical_features = ['contract', 'tech_support', 'payment_method']
    
    # Preprocesador para transformar características
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    # División en entrenamiento y prueba (train/test split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    print("Datos cargados y divididos con éxito:")
    print(f"Formas de entrenamiento (X_train): {X_train.shape}")
    print(f"Formas de prueba (X_test): {X_test.shape}")
    print(f"Proporción de churn en entrenamiento: {y_train.mean()*100:.2f}%")
