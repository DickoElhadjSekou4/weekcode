import pandas as pd 
from ucimlrepo import fetch_ucirepo 

def get_downloaded_dataset() -> pd.DataFrame:
    df = pd.read_csv(r"data\risk_factors_cervical_cancer.csv")
    return df 

def load_dataset_from_ucirepo():
    cervical_cancer_risk_factors = fetch_ucirepo(id=383) 
    X = cervical_cancer_risk_factors.data.features 
    # y = cervical_cancer_risk_factors.data.targets 
    df = pd.DataFrame(X)
    return df

if __name__ == "__main__":
    df1 = get_downloaded_dataset()
    df2 = load_dataset_from_ucirepo()
