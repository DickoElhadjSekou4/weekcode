import pandas as pd 

def get_downloaded_dataset() -> pd.DataFrame:
    df = pd.read_csv(r"data\risk_factors_cervical_cancer.csv")
    return df 
