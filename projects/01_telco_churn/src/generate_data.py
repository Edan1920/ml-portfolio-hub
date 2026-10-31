import numpy as np
import pandas as pd
import os

def generate_telco_churn_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    
    customer_id = [f"CUST-{i:05d}" for i in range(1, n_samples + 1)]
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    
    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.5, 0.3, 0.2])
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples, p=[0.3, 0.5, 0.2])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n_samples)
    
    base_prob = 0.15
    prob_churn = (
        base_prob 
        + np.where(contract_types == 'Month-to-month', 0.30, 0.05)
        + np.where(monthly_charges > 80.0, 0.15, 0.0)
        + np.where(tech_support == 'No', 0.10, -0.05)
        - (tenure * 0.003)
    )
    prob_churn = np.clip(prob_churn, 0.01, 0.95)
    churn = np.random.binomial(1, prob_churn)
    
    df = pd.DataFrame({
        'customer_id': customer_id,
        'tenure': tenure,
        'monthly_charges': np.round(monthly_charges, 2),
        'contract': contract_types,
        'tech_support': tech_support,
        'payment_method': payment_method,
        'churn': churn
    })
    
    return df

if __name__ == "__main__":
    output_dir = "projects/01_telco_churn/data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    df_churn = generate_telco_churn_data()
    file_path = os.path.join(output_dir, "telco_churn_synthetic.csv")
    df_churn.to_csv(file_path, index=False)
    
    print(f"Dataset sintético generado con éxito en '{file_path}'")
    print(df_churn.head())
    print("\nTasa de abandono global:", f"{df_churn['churn'].mean()*100:.2f}%")
