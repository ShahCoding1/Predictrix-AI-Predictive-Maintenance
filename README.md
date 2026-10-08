# ⚙️ Predictrix AI
### Explainable Machine Learning for Intelligent Predictive Maintenance

**Predict equipment failures before they happen. Understand the predictions. Support smarter maintenance decisions.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

**Predictrix AI** is an end-to-end, machine-learning-powered predictive maintenance application designed to estimate the likelihood of industrial equipment failure using operational sensor measurements.

Traditional maintenance strategies often depend on scheduled inspections or repairs after equipment failure. Predictive maintenance offers a data-driven alternative by using machine learning to identify patterns associated with equipment degradation and potential failure.

Predictrix AI combines:

- Supervised machine learning for equipment failure prediction
- Sensor-data preprocessing and feature engineering
- Multiple classification algorithms and model comparison
- Probability-based failure risk assessment
- Local feature-sensitivity analysis for prediction interpretation
- RESTful API services for model inference
- An interactive web dashboard for equipment monitoring
- Automated testing and containerized deployment support

The project demonstrates how machine learning can be integrated into a practical software engineering system, from data preparation and model training to inference, visualization, and deployment.

---

## 🎯 Project Objectives

The primary objectives of Predictrix AI are to:

1. Develop a reproducible machine learning pipeline for industrial equipment failure prediction.
2. Compare different classification algorithms using appropriate evaluation metrics.
3. Estimate equipment failure probability from operational sensor data.
4. Provide interpretable insights into individual predictions.
5. Build a modular backend capable of serving machine learning predictions through REST APIs.
6. Create a responsive dashboard for interacting with the predictive model.
7. Establish a foundation for future research in explainable AI, industrial analytics, and intelligent maintenance systems.

---

## ✨ Key Features

### 1. AI-Powered Failure Prediction

Predicts the probability of equipment failure from operational measurements using trained machine learning models.

### 2. Multiple Machine Learning Algorithms

The training pipeline evaluates three supervised learning approaches:

- Logistic Regression
- Random Forest Classifier
- Histogram-Based Gradient Boosting Classifier

The pipeline compares candidate models using validation performance and selects a model for final evaluation.

### 3. Data Preprocessing

Includes structured data preparation and preprocessing to support consistent model training and inference.

### 4. Model Performance Evaluation

Reports relevant classification metrics, including:

- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision

Average Precision is particularly useful when evaluating failure prediction systems with imbalanced class distributions.

### 5. Prediction Interpretability

Includes a local feature-sensitivity mechanism that estimates how predictions change when individual feature values are replaced with reference values.

**Important:** The current explanation method is based on feature perturbation. It is not SHAP, and its outputs should not be interpreted as causal explanations.

### 6. Interactive Dashboard

A responsive, industrial-themed dashboard allows users to:

- Enter equipment sensor measurements
- Request machine learning predictions
- View predicted failure probability
- Inspect feature-sensitivity results
- Review model evaluation information
- Compare trained classification models

### 7. RESTful Prediction API

FastAPI exposes model information, prediction, batch prediction, and explanation endpoints.

### 8. Reproducible Development Workflow

Includes automated tests, GitHub Actions configuration, Docker support, and a modular codebase.

---

## 🏗️ System Architecture

```text
                   ┌─────────────────────────┐
                   │     Sensor Data         │
                   │  Equipment Measurements │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │   Data Preprocessing    │
                   │ Validation & Transform  │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │   ML Training Pipeline  │
                   │ Logistic Regression     │
                   │ Random Forest           │
                   │ HistGradientBoosting    │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Model Evaluation        │
                   │ & Model Selection       │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Saved Model Artifacts   │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │      FastAPI API        │
                   │ Prediction & Explanation│
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Interactive Web UI      │
                   │ Risk & Model Insights   │
                   └─────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | scikit-learn |
| Data Processing | pandas, NumPy |
| Backend Framework | FastAPI |
| API Validation | Pydantic |
| Web Server | Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Model Persistence | Joblib |
| Testing | pytest |
| Containerization | Docker, Docker Compose |
| Continuous Integration | GitHub Actions |
| Version Control | Git and GitHub |

---

## 📊 Dataset

Predictrix AI supports experimentation using the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

The dataset represents synthetic industrial operating conditions and includes features such as:

| Feature | Description |
|---|---|
| Type | Product quality category |
| Air Temperature | Environmental air temperature |
| Process Temperature | Operating process temperature |
| Rotational Speed | Equipment rotational speed |
| Torque | Applied torque |
| Tool Wear | Accumulated tool wear |

The prediction target is machine failure.

**Dataset source:**

[UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

The repository also supports a generated demonstration dataset for testing the application without downloading the external dataset.

**Dataset disclaimer:** Both the demonstration data and the AI4I dataset are synthetic. Performance on these datasets does not establish reliability in real industrial environments.

---

## 🧠 Machine Learning Methodology

### Data Preparation

The training pipeline prepares structured sensor measurements, handles the required preprocessing steps, and separates data into training, validation, and testing subsets.

### Dataset Splitting

The project uses stratified splitting:

| Subset | Proportion | Purpose |
|---|---:|---|
| Training | 60% | Model fitting |
| Validation | 20% | Model comparison and selection |
| Testing | 20% | Final held-out evaluation |

### Model Training

The pipeline trains multiple classification algorithms using consistent experimental data partitions.

### Model Selection

Candidate models are compared using validation Average Precision.

### Model Evaluation

The selected model is evaluated on held-out test data using several classification metrics.

This separation helps distinguish model selection from final performance evaluation.

### Model Persistence

The trained model and associated metadata are saved for later use by the API.

---

## 📈 Evaluation Metrics

**Precision**

Measures the proportion of predicted failures that correspond to actual failures.

**Recall**

Measures the proportion of actual failures successfully identified.

**F1-score**

Provides a harmonic mean of precision and recall.

**ROC-AUC**

Measures the model's ability to rank positive instances above negative instances across different thresholds.

**Average Precision**

Summarizes precision-recall performance and is especially informative for imbalanced classification problems.

> Reported model results depend on the dataset, training configuration, and experimental conditions. Reproduce the training pipeline to generate evaluation results for your environment.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have:

- Python 3.10 or newer
- Git
- pip
- A modern web browser

Docker is optional.

### 1. Clone the Repository

```bash
git clone https://github.com/ShahCoding1/Predictrix-AI-Predictive-Maintenance.git
```

```bash
cd Predictrix-AI-Predictive-Maintenance
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train a Demonstration Model

```bash
python -m ml.train --demo
```

This generates demonstration training data, trains candidate models, evaluates their performance, and saves the selected model artifacts.

### 5. Start the Backend

```bash
python -m uvicorn backend.app.main:app --reload --reload-dir backend --reload-dir ml
```

If automatic reload causes problems, run:

```bash
python -m uvicorn backend.app.main:app
```

### 6. Open the Dashboard

Visit:

**http://127.0.0.1:8000**

### 7. Explore the API

Interactive API documentation is available at:

**http://127.0.0.1:8000/docs**

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Check application health |
| GET | `/api/model` | Retrieve model information |
| POST | `/api/predict` | Predict failure risk |
| POST | `/api/predict/batch` | Generate predictions for multiple records |
| POST | `/api/explain` | Generate local feature-sensitivity information |
| GET | `/docs` | Access interactive API documentation |

The API accepts validated input data and returns structured prediction results.

For exact request and response schemas, use the interactive documentation generated by FastAPI.

---

## 🧪 Automated Testing

Run the project's test suite with:

```bash
python -m pytest -q
```

The tests cover core application functionality and help detect regressions as the project evolves.

---

## 🐳 Docker Deployment

The repository includes Docker configuration for containerized execution.

Start the application with:

```bash
docker compose up --build
```

The containerized application requires access to trained model artifacts. Generate these according to the project configuration before starting the service.

---

## 📁 Project Structure

```text
predictrix-ai/
│
├── backend/
│   └── app/
│       └── main.py
│
├── frontend/
│   └── ...
│
├── ml/
│   ├── train.py
│   └── ...
│
├── tests/
│   └── ...
│
├── docs/
│   ├── ARCHITECTURE.md
│   └── ROADMAP.md
│
├── artifacts/
│   └── ...
│
├── .github/
│   └── workflows/
│       └── ...
│
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── LICENSE
└── README.md
```

*The tree highlights major components rather than every source file.*

---

## 🔍 Explainability and Model Transparency

Predictrix AI aims to make model predictions more understandable.

The current system provides local sensitivity information by modifying one input feature at a time and observing the resulting prediction difference.

This technique can help users explore how individual measurements influence model output in a particular context.

However, there are important limitations:

- Sensitivity scores are not causal effects.
- Perturbed inputs may not always represent physically realistic equipment states.
- Correlated sensor variables can complicate interpretation.
- The current implementation does not compute SHAP values.

More advanced explanation methods are planned for future development.

---

## 🗺️ Development Roadmap

The current release provides the core machine learning pipeline, prediction API, explanation mechanism, dashboard, testing infrastructure, and Docker configuration.

Potential future improvements include:

- [ ] SHAP-based local and global explanations
- [ ] React and TypeScript frontend
- [ ] PostgreSQL for persistent prediction history
- [ ] Authentication and role-based access control
- [ ] Dataset upload and validation workflows
- [ ] MLflow experiment tracking
- [ ] Hyperparameter optimization
- [ ] Model calibration and threshold optimization
- [ ] Time-series modeling with LSTM and GRU networks
- [ ] Sensor anomaly detection
- [ ] Drift monitoring and retraining workflows
- [ ] Maintenance alerts and notifications
- [ ] Real-world industrial dataset evaluation
- [ ] Cloud deployment and observability

See [Development Roadmap](docs/ROADMAP.md) for additional details.

---

## 🔬 Research and Educational Relevance

Predictrix AI explores the intersection of several important computer science and engineering disciplines:

**Artificial Intelligence and Machine Learning:** Supervised learning, classification, evaluation, and predictive modeling.

**Software Engineering:** Modular architecture, API development, testing, and maintainability.

**Data Science:** Data preparation, feature analysis, class imbalance, and evaluation methodology.

**Explainable AI:** Local prediction interpretation and the limitations of explanation techniques.

**Industrial Informatics:** Data-driven equipment monitoring and intelligent maintenance support.

The project is intended as an educational and portfolio demonstration and a foundation for further experimentation.

---

## ⚠️ Limitations

The following considerations are important when interpreting this project's results:

1. Demonstration data is synthetic and is not a substitute for real equipment measurements.
2. Classification performance may vary substantially across operating environments.
3. Model predictions do not guarantee that equipment will fail or remain operational.
4. Prediction probabilities may require additional calibration before operational use.
5. Local feature perturbations do not establish causal relationships.
6. The current application is not validated for safety-critical industrial decision-making.

**Predictrix AI should not be used as the sole basis for real-world maintenance or safety decisions.**

---

## 🤝 Contributing

Contributions, suggestions, and constructive feedback are welcome.

To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Commit your changes with a clear message.
5. Open a pull request describing the proposed improvement.

Please include appropriate tests for changes affecting model training, prediction, or API behavior.

---

## 👨‍💻 Author

**Muhammad Shah Khalid**

Software Engineering Graduate | Python Developer | AI & Machine Learning Enthusiast

My interests include intelligent software systems, applied machine learning, backend engineering, and practical AI solutions.

**GitHub:** [@ShahCoding1](https://github.com/ShahCoding1)

---

## 📄 License

This project is distributed under the [MIT License](LICENSE).

---

## ⭐ Support

If you find Predictrix AI useful for learning about machine learning, predictive maintenance, or full-stack AI development, consider giving the repository a star.

**Predictrix AI — Turning equipment data into interpretable predictive insights.**
