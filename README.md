# DATA 6545 Final Project
## Predicting High-Severity Traffic Accidents Using Machine Learning

### Overview
This project applies machine learning techniques to predict whether a traffic accident is likely to be high severity using the US Accidents dataset (2016–2023). The objective is to support operational awareness and emergency response prioritization through predictive analytics and interpretable machine learning models.

The project was completed as part of Fairfield University DATA 6545: Data Science and MLOps.

---

## Business Problem
Traffic accidents create significant operational and public safety challenges. Rapid identification of potentially high-severity accidents can improve emergency response coordination, resource allocation, and situational awareness.

This project reframes accident severity prediction as a binary classification problem:

```python
is_high_severity = 1 if Severity >= 3 else 0
```

---

## Dataset
Source:
- US Accidents Dataset (2016–2023)
- Kaggle

The dataset contains millions of accident records across the United States, including:
- Weather conditions
- Geographic information
- Time and date variables
- Visibility and traffic-related indicators
- Environmental conditions

---

## Repository Structure

```text
DATA6545_Group_6_Final_Project/
│
├── app.py
├── requirements.txt
├── example_payload.json
├── DATA_6545_Final_Project.ipynb
│
├── models/
├── figures/
├── artifacts/
│
├── Final Report
└── Final Presentation
```

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- LightGBM
- Flask
- MLflow
- Docker
- Google Colab

---

## Machine Learning Workflow
The project workflow included:
- Data preprocessing
- Feature engineering
- Exploratory data analysis (EDA)
- Model training and evaluation
- Threshold optimization
- SHAP interpretability analysis
- Experiment tracking with MLflow
- Flask API deployment
- Docker containerization

Models evaluated:
- Logistic Regression
- Random Forest
- LightGBM

The final champion model was LightGBM.

---

## Model Deployment
A Flask API was developed for model inference.

Endpoints:
- `/health`
- `/predict`

Example payload:

```json
{
  "Temperature(F)": 72,
  "Visibility(mi)": 10,
  "State": "CT"
}
```

---

## Setup Instructions

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Flask API

```bash
python app.py
```

---

## Technical Artifacts
This repository includes:
- Trained model artifacts
- Flask API implementation
- Experiment tracking outputs
- Figures and evaluation outputs
- Final report
- Final presentation

---

## Author
Peter Bonnanzio  
Fairfield University  
DATA 6545 – Data Science and MLOps