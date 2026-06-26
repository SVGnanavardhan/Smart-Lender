# Smart Lender

Smart Lender is a machine learning-powered web application designed to predict the creditworthiness of loan applicants. It enables banks and financial institutions to make faster, data-driven loan approval decisions by evaluating applicant data and predicting the likelihood of repayment or default.

## About the Project

Smart Lender processes structured applicant inputs such as:
- Gender
- Marital Status
- Dependents
- Education
- Self-Employment
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

The application trains and evaluates multiple classification algorithms:
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost

The best-performing model (XGBoost) is saved and integrated into a Flask web application for real-time prediction. In testing, the XGBoost model achieved approximately **94.7% training accuracy** and **81.1% testing accuracy**.

## Project Structure

- `Flask/app.py` - Flask application entry point
- `Flask/templates/` - HTML templates for the home, prediction form, and result pages
- `Flask/static/` - optional static assets such as CSS
- `Flask/model.pkl` - serialized trained ML model
- `Flask/scaler.pkl` - serialized preprocessing scaler
- `Dataset/loan_prediction.csv` - source dataset used for model training
- `Training/Loan_Prediction_using_Ml.ipynb` - exploratory notebook and training analysis
- `train_model.py` - training script that builds the ML model and saves the required artifacts
- `requirements.txt` - Python package dependencies

## Tech Stack

- Python
- Flask
- scikit-learn
- XGBoost
- pandas
- NumPy
- HTML / CSS
- IBM Cloud (deployment target)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SVGnanavardhan/Smart-Lender.git
cd "Smart-Lender"
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the environment:

```powershell
.venv\Scripts\activate
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Generate model artifacts (if not already present):

```powershell
python train_model.py
```

6. Run the Flask application:

```powershell
python Flask\app.py
```

7. Open the app in your browser:

```text
http://localhost:5000
```

## Workflow

1. **Data Preparation**
   - Load and clean the loan dataset
   - Handle missing values and categorical encodings
   - Normalize numeric features with a scaler

2. **Model Training**
   - Train multiple classification models
   - Evaluate model performance using train/test splits
   - Select the best-performing model

3. **Model Serialization**
   - Save the trained model as `Flask/model.pkl`
   - Save the scaler as `Flask/scaler.pkl`

4. **Web Integration**
   - Build a Flask app to accept form inputs
   - Apply preprocessing to incoming inputs
   - Use the saved model for real-time predictions
   - Render results in a user-friendly interface

5. **Deployment**
   - Deploy the Flask app on IBM Cloud or any compatible web host
   - Use the interface for loan decision support

## Usage Scenarios

### Scenario 1: Fast-Track Approval for Low-Risk Applicants
A bank credit officer enters the details of a salaried applicant with good credit history and stable income. The model predicts loan approval with high confidence, allowing the officer to fast-track the application.

### Scenario 2: High-Risk Applicant Detection
An applicant with irregular self-employment income and no credit history submits a request. The system flags this application as high-risk, prompting further scrutiny.

### Scenario 3: High-Volume Batch Evaluation
A financial analyst batch-evaluates multiple applicants during a busy period. By using Smart Lender, the analyst reduces evaluation time while maintaining accuracy and compliance.

## Requirements for an ML Project

- Python 3.8+ (3.12 recommended)
- Flask
- pandas
- NumPy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib / seaborn (for exploratory data analysis)
- joblib or pickle for model serialization
- Git for version control

## Future Enhancements

- Add user authentication and role-based access control for credit officers and analysts
- Implement model explainability tools such as SHAP or LIME for transparent decision support
- Add batch-upload functionality to evaluate multiple applicants from CSV files
- Log prediction requests and store results for audit and performance tracking
- Add automatic retraining and monitoring for model drift
- Deploy with Docker and CI/CD pipelines for scalable production delivery
- Integrate with cloud databases or financial APIs for real-time applicant data

## Developer

**Samudrala Venkata Gnana Vardhan**
- B.Tech - Computer Science and Engineering
- Academic Project - 2026

---

Build a better loan approval pipeline with Smart Lender, and deploy the app on IBM Cloud for production-ready usage.