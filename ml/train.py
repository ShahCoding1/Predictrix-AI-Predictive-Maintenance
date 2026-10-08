import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score,roc_auc_score,precision_score,recall_score,f1_score,confusion_matrix,brier_score_loss)
from .schema import FEATURES,NUMERIC,TARGET
from .data import demo_data,normalize

ROOT=Path(__file__).resolve().parents[1]

def pipeline(model):
    numeric=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    categorical=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))])
    pre=ColumnTransformer([('num',numeric,NUMERIC),('cat',categorical,['product_type'])])
    return Pipeline([('preprocess',pre),('model',model)])

def metrics(y,prob,threshold=.5):
    pred=(prob>=threshold).astype(int)
    return {'average_precision':round(float(average_precision_score(y,prob)),5), 'roc_auc':round(float(roc_auc_score(y,prob)),5), 'precision':round(float(precision_score(y,pred,zero_division=0)),5),'recall':round(float(recall_score(y,pred,zero_division=0)),5),'f1':round(float(f1_score(y,pred,zero_division=0)),5),'brier':round(float(brier_score_loss(y,prob)),5),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def train(df,output=ROOT/'artifacts',demo=False):
    df=normalize(df)
    if len(df)<100 or df[TARGET].value_counts().min()<10:
        raise ValueError('Need >=100 records and >=10 examples in each target class')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    x=df[FEATURES];y=df[TARGET].astype(int)
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42,stratify=y)
    X_fit,X_val,y_fit,y_val=train_test_split(X_train,y_train,test_size=.25,random_state=42,stratify=y_train)
    candidates={'logistic_regression':LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42),'random_forest':RandomForestClassifier(n_estimators=140,min_samples_leaf=3,class_weight='balanced_subsample',random_state=42,n_jobs=-1),'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=100,random_state=42)}
    comparison={};fitted={}
    for name,estimator in candidates.items():
        model=pipeline(estimator).fit(X_fit,y_fit)
        prob=model.predict_proba(X_val)[:,1]
        comparison[name]=metrics(y_val,prob)
        fitted[name]=model
    winner=max(comparison,key=lambda k:comparison[k]['average_precision'])
    final=pipeline(candidates[winner]).fit(X_train,y_train)
    test_metrics=metrics(y_test,final.predict_proba(X_test)[:,1])
    joblib.dump(final,output/'model.joblib')
    report={'dataset':'SYNTHETIC DEMO DATA' if demo else 'USER-SUPPLIED DATASET','records':len(df),'positive_rate':round(float(y.mean()),5),'split':'60% fit / 20% validation / 20% held-out test, stratified','selection_metric':'validation average precision','selected_model':winner,'validation_comparison':comparison,'held_out_test':test_metrics,'threshold':0.5,'features':FEATURES,'random_seed':42,'note':'Test metrics are measured once after model selection; not a production guarantee.'}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',help='Path to UCI AI4I or compatible CSV')
    parser.add_argument('--demo',action='store_true',help='Generate explicitly synthetic sample data')
    parser.add_argument('--output',default=str(ROOT/'artifacts'))
    args=parser.parse_args()
    if not args.demo and not args.csv:parser.error('Specify --csv PATH or --demo')
    df=demo_data() if args.demo else pd.read_csv(args.csv)
    print(json.dumps(train(df,args.output,demo=args.demo),indent=2))
if __name__=='__main__':main()
