import pandas as pd 
from ucimlrepo import fetch_ucirepo 

def get_downloaded_dataset() -> pd.DataFrame:
    df = pd.read_csv(r"data\risk_factors_cervical_cancer.csv")
    return df 
def load_dataset_from_ucirepo():
    cervical_cancer_risk_factors = fetch_ucirepo(id=383) 
    X = cervical_cancer_risk_factors.data.features 
    y = cervical_cancer_risk_factors.data.targets 
    return X, y
