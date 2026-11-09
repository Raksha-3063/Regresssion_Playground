# Regression Playground

<p align="center">
  **An Interactive Machine Learning Workspace for Regression Experiments**
  
  Explore regression algorithms, train models, visualize results, and compare performance — all in one place.
</p>

---

## Overview

**Regression Playground** is a full-stack machine learning application designed to make regression experimentation simple, interactive, and accessible.

Instead of manually writing separate scripts for every experiment, users can upload their own datasets, explore data characteristics, select regression algorithms, configure model parameters, and evaluate results through a unified interface.

The platform brings the complete regression workflow together — from dataset exploration and preprocessing to model training, hyperparameter tuning, and experiment comparison.

## Key Features

- **Dataset Upload:** Upload CSV datasets and use them for regression experiments.
- **Data Exploration:** Inspect dataset structure, column information, missing values, and data quality.
- **Target Selection:** Choose the target variable and relevant features for model training.
- **Multiple Regression Algorithms:** Experiment with eight supported regression estimators.
- **Automated Preprocessing:** Apply appropriate missing-value imputation, encoding, scaling, and feature selection through scikit-learn pipelines.
- **Model Training:** Train models using configurable train-test splits.
- **Hyperparameter Tuning:** Optimize model parameters using cross-validation.
- **Performance Evaluation:** Evaluate trained models using regression metrics.
- **Experiment Comparison:** Compare saved model runs to understand differences in performance.
- **Docker Deployment:** Run the frontend and backend together using Docker Compose.

## Machine Learning Workflow

The application follows a structured machine learning workflow:

1. Upload a CSV dataset.
2. Explore the dataset and inspect data quality.
3. Select the target column and input features.
4. Choose a regression algorithm.
5. Configure model parameters and training settings.
6. Train the model using a leakage-safe pipeline.
7. Evaluate model performance on held-out test data.
8. Tune hyperparameters using cross-validation.
9. Compare results across saved experiments.

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React.js, Vite |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn, NumPy, Pandas |
| Data Processing | Pandas, NumPy |
| Model Evaluation | Scikit-learn |
| Deployment | Docker, Docker Compose |
| Web Server | Nginx |

## Project Architecture

```text
Regression_Playground/
│
├── backend/
│   ├── app/
│   ├── requirements.txt
│   └── storage/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── Dockerfile
│   └── nginx.conf
│
├── docker-compose.yml
└── README.md
```

## Running the Application

### Option 1: Docker Compose (Recommended)

**Prerequisites**
- Docker Desktop
- Git

Clone the repository:

```bash
git clone https://github.com/Raksha-3063/Regresssion_Playground.git
cd Regresssion_Playground
```

Build and start the application:

```bash
docker compose up --build
```

Access the application:

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |

To stop the application:

```bash
docker compose down
```

### Option 2: Local Development

**Backend**

```bash
cd backend

python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

**Frontend**

Open another terminal:

```bash
cd frontend

npm install
npm run dev
```

## Engineering Considerations

- **Data Leakage Prevention:** Data splitting is performed before fitting preprocessing transformations and training workflows.
- **Pipeline-Based Processing:** Preprocessing and model estimation are integrated using scikit-learn pipelines.
- **Reliable Evaluation:** Hyperparameter tuning uses cross-validation without using the held-out test split.
- **Experiment Tracking:** Saved runs allow users to compare model performance.
- **Local Storage:** Uploaded datasets and trained artifacts are stored locally in `backend/storage`.

## Future Enhancements

- Interactive regression plots and residual analysis.
- Additional regression algorithms.
- Feature importance and model interpretability.
- Export trained models and experiment reports.
- User authentication and multi-user experiment management.
- Cloud-based model deployment.

## Project Goal

The goal of Regression Playground is to provide a practical environment for understanding and experimenting with machine learning regression algorithms while bringing data exploration, model development, and evaluation into one integrated application.

---

**Developed by Raksha C C**

[GitHub Repository](https://github.com/Raksha-3063/Regresssion_Playground)

