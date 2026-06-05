import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH="data/raw/ckd/kidney_disease.csv"
PROCESSED_PATH="data/processed/ckd/ckd_processed.csv"

def calculate_egfr(scr,age):
    return 144*(scr/0.7)**-1.209*(0.993**age)

def preprocess_ckd():
    df=pd.read_csv(RAW_PATH)
    #selecting medically justified features 

    df['classification']=df['classification'].str.strip().str.lower()
    label_map={
        'ckd':1,'notckd':0,'not ckd':0
    }

    df['prognosis']=df['classification'].map(label_map)

    #drop the old label column

    df.drop(columns=['classification'],inplace=True)

    df=df[["sc","al","age","prognosis"]]
    for col in ['sc','al','age']:
        df=df.apply(pd.to_numeric,errors='coerce')
    
    #feature engineering
    df["egfr"]=calculate_egfr(df["sc"],df["age"])
    
    df.fillna(df.median(), inplace=True)
    
    Path("data/processed/ckd").mkdir(parents=True,exist_ok=True)
    df.to_csv(PROCESSED_PATH,index=False)

    print("CKD preprocessing complete")

if __name__=="__main__":
    preprocess_ckd()