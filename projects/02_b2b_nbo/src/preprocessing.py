import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_data(filepath="projects/02_b2b_nbo/data/b2b_nbo_synthetic.csv"):
    """Carga el dataset sintético B2B."""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """
    Prepara las características (features) y la variable objetivo (target).
    Aplica transformaciones a variables numéricas y categóricas.
    """
    # Definir la variable objetivo (NBO - Next Best Offer) y las características predictoras
    target_col = 'next_best_offer'
    
    # Excluimos identificadores y el objetivo de las características de entrada
    X = df.drop(columns=['company_id', target_col, 'primary_product_used'])
    y = df[target_col]
    
    # Identificar tipos de columnas
    categorical_features = ['segment', 'industry']
    numerical_features = ['monthly_transaction_volume', 'avg_ticket_size', 'days_active']
    
    # Construir el transformador de columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # División de datos en entrenamiento y prueba (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    print("Probando el pipeline de preprocesamiento B2B...")
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    print(f"Dimensiones de entrenamiento (X_train): {X_train.shape}")
    print(f"Dimensiones de prueba (X_test): {X_test.shape}")
    print("¡Preprocesamiento configurado con éxito!")
