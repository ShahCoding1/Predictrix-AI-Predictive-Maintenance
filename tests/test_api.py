import os
from fastapi.testclient import TestClient
from backend.app.main import app

client=TestClient(app)
SAMPLE={'air_temperature':300,'process_temperature':311,'rotational_speed':1420,'torque':48,'tool_wear':175,'product_type':'M'}
def test_health():
    assert client.get('/api/health').status_code==200

def test_invalid_prediction():
    response=client.post('/api/predict',json={**SAMPLE,'torque':-4})
    assert response.status_code==422

def test_predict_when_trained():
    if not client.get('/api/health').json()['model_ready']:return
    response=client.post('/api/predict',json=SAMPLE)
    assert response.status_code==200
    assert 0<=response.json()['failure_probability']<=1
    explain=client.post('/api/explain',json=SAMPLE)
    assert explain.status_code==200
    assert len(explain.json()['contributions'])==6
