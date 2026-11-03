import pandas as pd
import numpy as np

def generate_b2b_data(num_samples=1500, random_state=42):
    np.random.seed(random_state)
    
    segments = ['Pequeña', 'Mediana', 'Grande', 'Corporativa']
    segment_weights = [0.45, 0.30, 0.15, 0.10]
    
    industries = ['Retail', 'Tecnología', 'Manufactura', 'Servicios Financieros', 'Logística']
    
    catalog_products = [
        'Cuenta Corriente Básica', 
        'Línea de Crédito Comercial', 
        'Financiación de Comercio Exterior', 
        'Fideicomiso de Inversión', 
        'Pasarela de Pagos B2B'
    ]
    
    company_ids = [f"COMP_{i:04d}" for i in range(1, num_samples + 1)]
    chosen_segments = np.random.choice(segments, size=num_samples, p=segment_weights)
    chosen_industries = np.random.choice(industries, size=num_samples)
    
    monthly_transaction_volume = np.random.exponential(scale=50000, size=num_samples) + 5000
    avg_ticket_size = np.random.uniform(1000, 50000, size=num_samples)
    days_active = np.random.randint(30, 1800, size=num_samples)
    
    data = []
    for i in range(num_samples):
        seg = chosen_segments[i]
        ind = chosen_industries[i]
        vol = monthly_transaction_volume[i]
        
        # Lógica basada en volumen transaccional y segmento para una señal más clara
        if vol < 20000:
            nbo = catalog_products[0] # Cuenta Corriente Básica
        elif 20000 <= vol < 60000:
            nbo = catalog_products[1] # Línea de Crédito Comercial
        elif 60000 <= vol < 100000:
            nbo = catalog_products[2] # Financiación de Comercio Exterior
        elif 100000 <= vol < 150000:
            nbo = catalog_products[3] # Fideicomiso de Inversión
        else:
            nbo = catalog_products[4] # Pasarela de Pagos B2B
            
        data.append({
            'company_id': company_ids[i],
            'segment': seg,
            'industry': ind,
            'monthly_transaction_volume': round(vol, 2),
            'avg_ticket_size': round(avg_ticket_size[i], 2),
            'days_active': days_active[i],
            'primary_product_used': catalog_products[0],
            'next_best_offer': nbo
        })
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Regenerando dataset B2B con correlación transaccional...")
    df_b2b = generate_b2b_data()
    output_path = "projects/02_b2b_nbo/data/b2b_nbo_synthetic.csv"
    df_b2b.to_csv(output_path, index=False)
    print(f"Dataset actualizado con éxito en: {output_path}")
