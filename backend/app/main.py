import json,os
from pathlib import Path
from functools import lru_cache
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field,field_validator
from typing import Literal
from ml.schema import FEATURES

ROOT=Path(__file__).resolve().parents[2]
ARTIFACT=Path(os.getenv('MODEL_DIR',str(ROOT/'artifacts')))
app=FastAPI(title='Predictrix AI API',version='1.0.0',description='Reproducible machine failure risk prediction. For research, not safety-critical operations.')
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('CORS_ORIGINS','http://localhost:5173,http://localhost:8000').split(','),allow_methods=['GET','POST'],allow_headers=['*'])
class Reading(BaseModel):
    air_temperature:float=Field(ge=250,le=400)
    process_temperature:float=Field(ge=250,le=420)
    rotational_speed:float=Field(ge=0,le=10000)
    torque:float=Field(ge=0,le=500)
    tool_wear:float=Field(ge=0,le=1000)
    product_type:Literal['L','M','H']
class Batch(BaseModel):
    readings:list[Reading]=Field(min_length=1,max_length=500)
@lru_cache(maxsize=1)
def model():
    path=ARTIFACT/'model.joblib'
    if not path.exists():raise HTTPException(503,'Model not trained. Run python -m ml.train --demo or --csv data/ai4i2020.csv')
    return joblib.load(path)  # Only load trusted locally trained model artifacts.
def predict_many(readings):
    df=pd.DataFrame([r.model_dump() for r in readings],columns=FEATURES)
    probabilities=model().predict_proba(df)[:,1]
    return [{'failure_probability':round(float(v),5),'risk_level':'high' if v>=.7 else 'medium' if v>=.3 else 'low','threshold':.5,'predicted_failure':bool(v>=.5)} for v in probabilities]
@app.get('/api/health')
def health():return {'status':'ok','model_ready':(ARTIFACT/'model.joblib').exists()}
@app.get('/api/model')
def model_info():
    path=ARTIFACT/'report.json'
    if not path.exists():raise HTTPException(503,'No report. Train a model first.')
    return json.loads(path.read_text())
@app.post('/api/predict')
def predict(reading:Reading):return predict_many([reading])[0]
@app.post('/api/predict/batch')
def predict_batch(batch:Batch):return {'predictions':predict_many(batch.readings)}
@app.post('/api/explain')
def explain(reading:Reading):
    # Model-agnostic local perturbation: informative, NOT SHAP and NOT causal.
    base=predict_many([reading])[0]
    reference={'air_temperature':300,'process_temperature':310,'rotational_speed':1500,'torque':40,'tool_wear':120,'product_type':'M'}
    contributions=[]
    for feature in FEATURES:
        altered=reading.model_copy(update={feature:reference[feature]})
        alt=predict_many([altered])[0]['failure_probability']
        contributions.append({'feature':feature,'observed':getattr(reading,feature),'reference':reference[feature],'probability_difference':round(base['failure_probability']-alt,5)})
    contributions.sort(key=lambda x:abs(x['probability_difference']),reverse=True)
    return {'prediction':base,'method':'one-feature-at-a-time reference perturbation (not SHAP; not causal)','contributions':contributions,'warning':'Feature interactions mean differences do not add up to prediction.'}
@app.get('/')
def home():return FileResponse(ROOT/'frontend'/'index.html')
@app.get('/app.js')
def javascript():return FileResponse(ROOT/'frontend'/'app.js',media_type='application/javascript')
@app.get('/styles.css')
def styles():return FileResponse(ROOT/'frontend'/'styles.css',media_type='text/css')
