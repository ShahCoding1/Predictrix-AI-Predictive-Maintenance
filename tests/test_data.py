import pytest
from ml.data import demo_data,normalize
from ml.train import train

def test_demo_data_valid():
    data=normalize(demo_data(500))
    assert len(data)==500
    assert set(data.machine_failure.unique())=={0,1}

def test_reject_missing_columns():
    with pytest.raises(ValueError,match='Missing required columns'):
        normalize(demo_data(100).drop(columns=['torque']))

def test_training_creates_artifacts(tmp_path):
    report=train(demo_data(600),tmp_path,demo=True)
    assert (tmp_path/'model.joblib').exists()
    assert report['selected_model'] in report['validation_comparison']
    assert 0<=report['held_out_test']['average_precision']<=1
