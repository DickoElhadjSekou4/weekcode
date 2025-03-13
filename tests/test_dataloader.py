import pytest 

import pandas as pd
import sys 

sys.path.append(".")

from src.dataloader import get_downloaded_dataset, load_dataset_from_ucirepo

def test_get_downloaded_dataset():
    df = get_downloaded_dataset()
    assert isinstance(df, pd.DataFrame), "Le dataset téléchargé doit être un DataFrame"
    assert not df.empty, "Le dataset ne doit pas être vide"

def test_load_dataset_from_ucirepo():
    df = load_dataset_from_ucirepo()
    
    assert isinstance(df, pd.DataFrame), "df doit être un DataFrame"
    assert not df.empty, "Le dataset ne doit pas être vide"
