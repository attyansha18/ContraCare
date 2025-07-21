import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data():
    """Load the processed data"""
    try:
        df = pd.read_csv('processed_cmc.csv')
        return df
    except FileNotFoundError:
        print("Processed data not found. Please run download_data.py first.")
        return None

def add_derived_features(df):
    """Add derived features to match training data"""
    # Add socioeconomic score
    df['socioeconomic_score'] = df['wife_education'] + df['husband_education'] + df['standard_of_living']
    
    # Add interaction features
    df['age_children_interaction'] = df['wife_age'] * df['num_children']
    df['education_interaction'] = df['wife_education'] * df['husband_education']
    
    # Add age groups
    df['age_group'] = pd.cut(df['wife_age'], 
                            bins=[0, 25, 35, 45, 100],
                            labels=['18-25', '26-35', '36-45', '46+'])
    
    # Add children groups
    df['children_group'] = pd.cut(df['num_children'],
                                 bins=[-1, 0, 2, 4, 100],
                                 labels=['0', '1-2', '3-4', '5+'])
    
    # Add family planning stage
    df['family_planning_stage'] = np.where(
        df['num_children'] == 0, 'pre-family',
        np.where(df['num_children'] <= 2, 'growing-family',
                'completed-family')
    )
    
    # Add medical conditions (all set to 0 for evaluation)
    medical_conditions = [
        'high_blood_pressure', 'history_of_clots', 'breast_cancer_history',
        'liver_disease', 'thyroid_disorder', 'has_arthritis',
        'has_hypertension', 'has_diabetes', 'has_anxiety',
        'has_depression', 'has_chronic_kidney_disease',
        'has_chronic_liver_disease', 'has_chronic_heart_disease',
        'has_chronic_lung_disease', 'is_smoker', 'recent_childbirth',
        'is_breastfeeding', 'taking_tb_medication', 'taking_anticonvulsants',
        'has_surgery', 'pcos_diagnosed'
    ]
    
    for condition in medical_conditions:
        df[condition] = 0
    
    # Add health risk score
    df['health_risk_score'] = 0
    
    return df

def evaluate_model(X, y, model):
    """Evaluate model performance"""
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    
    return y_pred, y_pred_proba

def main():
    print("Loading data and model...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Load model
    try:
        model = joblib.load('contra_model.pkl')
    except FileNotFoundError:
        print("Model not found. Please run train_model.py first.")
        return
    
    # Add derived features
    df = add_derived_features(df)
    
    # Prepare features and target
    X = df.drop(['contraceptive_method'], axis=1)
    y = df['contraceptive_method']
    
    print("\nEvaluating model performance...")
    y_pred, y_pred_proba = evaluate_model(X, y, model)
    
    print("\nEvaluation complete. Results saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    main() 