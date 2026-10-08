import numpy as np
import pandas as pd
from .schema import FEATURES, TARGET, UCI_MAPPING

def normalize(df):
    df = df.rename(columns=UCI_MAPPING).copy()
    missing = [x for x in FEATURES+[TARGET] if x not in df.columns]
    if missing:
        raise ValueError('Missing required columns: '+', '.join(missing))
    df = df[FEATURES+[TARGET]].copy()
    for col in FEATURES[:-1]:
        df[col] = pd.to_numeric(df[col],errors='coerce')
    df['product_type'] = df['product_type'].astype('string').str.upper().str.strip()
    df[TARGET] = pd.to_numeric(df[TARGET],errors='coerce')
    if df.isna().any().any():
        raise ValueError('Dataset contains missing or nonnumeric required values; clean data before training')
    if not df.product_type.isin(['L','M','H']).all():
        raise ValueError('product_type must be L, M or H')
    if not df[TARGET].isin([0,1]).all():
        raise ValueError('machine_failure must be 0 or 1')
    return df

def demo_data(n=2400,seed=42):
    # Clearly synthetic demo data, not the UCI AI4I dataset.
    rng=np.random.default_rng(seed)
    air=rng.normal(300,2,n)
    process=air+rng.normal(10,1.4,n)
    rpm=np.clip(rng.normal(1500,280,n),900,2600)
    torque=np.clip(rng.normal(40,10,n),5,80)
    wear=rng.integers(0,250,n)
    typ=rng.choice(['L','M','H'],n,p=[.5,.3,.2])
    z=-4.0 + .05*(wear-120)+.07*(torque-40)-.002*(rpm-1500)+.25*(process-air-10)
    prob=1/(1+np.exp(-np.clip(z,-25,25)))
    failure=rng.binomial(1,prob)
    return pd.DataFrame(dict(air_temperature=air,process_temperature=process,rotational_speed=rpm,torque=torque,tool_wear=wear,product_type=typ,machine_failure=failure))
