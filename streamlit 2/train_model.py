import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def load_and_preprocess_data():
    # Load the processed data
    try:
        df = pd.read_csv('processed_cmc.csv')
    except FileNotFoundError:
        print("Processed data not found. Please run download_data.py first.")
        return None
    
    # Age-based features
    df['age_group'] = pd.cut(df['wife_age'], 
                            bins=[0, 25, 35, 45, 100],
                            labels=['18-25', '26-35', '36-45', '46+'])
    
    # Education features (based on dataset description)
    df['wife_education_level'] = pd.cut(df['wife_education'],
                                      bins=[0, 2, 4],
                                      labels=['low', 'high'])
    df['husband_education_level'] = pd.cut(df['husband_education'],
                                         bins=[0, 2, 4],
                                         labels=['low', 'high'])
    df['education_gap'] = df['husband_education'] - df['wife_education']
    
    # Family planning features
    df['children_group'] = pd.cut(df['num_children'],
                                 bins=[-1, 0, 2, 4, 100],
                                 labels=['0', '1-2', '3-4', '5+'])
    
    # Socioeconomic features
    df['standard_of_living_level'] = pd.cut(df['standard_of_living'],
                                          bins=[0, 2, 4],
                                          labels=['low', 'high'])
    
    # Add medical conditions columns with default values (0 = no condition)
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
        if condition not in df.columns:
            df[condition] = 0
    
    # Calculate health risk score
    df['health_risk_score'] = (
        df['high_blood_pressure'] * 2.5 +
        df['history_of_clots'] * 3.0 +
        df['breast_cancer_history'] * 3.0 +
        df['liver_disease'] * 2.0 +
        df['has_diabetes'] * 2.5 +
        df['has_hypertension'] * 2.5 +
        df['has_chronic_heart_disease'] * 3.0 +
        df['has_chronic_lung_disease'] * 2.5 +
        df['has_chronic_kidney_disease'] * 3.0 +
        df['has_chronic_liver_disease'] * 3.0 +
        df['is_smoker'] * 1.5 +
        df['taking_tb_medication'] * 2.0 +
        df['taking_anticonvulsants'] * 2.0
    )
    
    return df

def create_preprocessing_pipeline():
    # Define numerical features
    numerical_features = [
        'wife_age', 'num_children', 'education_gap', 'health_risk_score'
    ]
    
    # Add medical conditions to numerical features
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
    numerical_features.extend(medical_conditions)
    
    # Define categorical features
    categorical_features = [
        'wife_education', 'husband_education', 'wife_religion',
        'wife_now_working', 'husband_occupation', 'standard_of_living',
        'media_exposure', 'age_group', 'children_group',
        'wife_education_level', 'husband_education_level',
        'standard_of_living_level'
    ]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return preprocessor

def train_model(df):
    if df is None:
        return None, None
        
    # Prepare features and target
    X = df.drop(['contraceptive_method'], axis=1)
    y = df['contraceptive_method']
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Define models with optimized parameters
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=400,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=3,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
    }
    
    best_model = None
    best_score = 0
    best_model_name = None
    
    # Try different models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(
                sampling_strategy='auto',
                random_state=42,
                k_neighbors=5,
                n_jobs=-1
            )),
            ('model', model)
        ])
        
        # Use StratifiedKFold for cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Get cross-validation scores
        cv_scores = cross_val_score(
            pipeline,
            X, y,
            cv=cv,
            scoring='f1_weighted'
        )
        
        mean_score = cv_scores.mean()
        print(f"{name} Mean CV Score: {mean_score:.4f} (±{cv_scores.std():.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = pipeline
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with F1 score: {best_score:.4f}")
    
    # Evaluate on test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred))
    
    # Save model and components
    joblib.dump(best_model, 'contra_model.pkl')
    joblib.dump(X.columns, 'feature_columns.pkl')
    
    # Create SHAP explainer only for Random Forest
    if best_model_name == 'RandomForest':
        explainer = shap.TreeExplainer(best_model.named_steps['model'])
        shap_values = explainer.shap_values(
            best_model.named_steps['preprocessor'].transform(X)
        )
        joblib.dump(explainer, 'shap_explainer.pkl')
    else:
        print("\nNote: SHAP explainer not created for Gradient Boosting due to multi-class limitation")
        joblib.dump(None, 'shap_explainer.pkl')
    
    return best_model, X.columns

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    if df is not None:
        print("\nTraining model...")
        model, feature_columns = train_model(df)
        
        if model is not None:
            print("\nModel training completed successfully!")
        else:
            print("\nModel training failed.")
    else:
        print("\nData loading failed. Please check the data files.") 