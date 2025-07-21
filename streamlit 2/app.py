import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page config
st.set_page_config(
    page_title="ContraCare: AI Contraceptive Recommendation",
    page_icon="🚺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        model = joblib.load('contra_model.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        shap_explainer = joblib.load('shap_explainer.pkl')
        return model, feature_columns, shap_explainer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, feature_columns, shap_explainer = load_model()

# Define contraceptive methods mapping
method_mapping = {
    1: "No-use",
    2: "Short-term (e.g., pills, condoms)",
    3: "Long-term (e.g., IUD, implant)"
}

# Knowledge Base Content
knowledge_base = {
    "Types of Contraceptives": {
        "Barrier Methods": {
            "description": "Physical barriers that prevent sperm from reaching the egg.",
            "examples": "Condoms (male/female), diaphragms, cervical caps",
            "effectiveness": "70-98% with perfect use",
            "side_effects": "Rare, possible allergic reactions"
        },
        "Hormonal Methods": {
            "description": "Use hormones to prevent ovulation or thicken cervical mucus.",
            "examples": "Birth control pills, patches, vaginal rings, injections",
            "effectiveness": "91-99% with perfect use",
            "side_effects": "Headaches, nausea, breast tenderness, mood changes"
        },
        "IUDs": {
            "description": "Small devices inserted into the uterus, can be hormonal or copper.",
            "examples": "Copper IUD, hormonal IUD (Mirena, Kyleena)",
            "effectiveness": "Over 99%",
            "side_effects": "Cramping, irregular bleeding, possible expulsion"
        },
        "Natural Methods": {
            "description": "Tracking fertility signs to avoid intercourse during fertile periods.",
            "examples": "Fertility awareness, withdrawal method",
            "effectiveness": "76-88% with perfect use",
            "side_effects": "None, but requires strict discipline"
        },
        "Permanent Methods": {
            "description": "Surgical procedures for permanent contraception.",
            "examples": "Tubal ligation (women), vasectomy (men)",
            "effectiveness": "Over 99%",
            "side_effects": "Surgical risks, considered permanent"
        },
        "Emergency Contraception": {
            "description": "Used after unprotected sex to prevent pregnancy.",
            "examples": "Emergency contraceptive pills, copper IUD insertion",
            "effectiveness": "75-95% depending on timing",
            "side_effects": "Nausea, irregular bleeding"
        }
    },
    "FAQs": [
        {
            "question": "Do contraceptives protect against STDs?",
            "answer": "Only condoms (male and female) provide protection against STDs. Other methods only prevent pregnancy."
        },
        {
            "question": "Can I get pregnant immediately after stopping contraceptives?",
            "answer": "It depends on the method. With pills, patches, or rings, fertility returns quickly. With injections, it may take several months. IUD removal allows immediate conception."
        },
        {
            "question": "Are contraceptives 100% effective?",
            "answer": "No method is 100% effective. Effectiveness ranges from 70% (natural methods) to over 99% (IUDs, implants). Perfect use increases effectiveness."
        },
        {
            "question": "Do contraceptives cause infertility?",
            "answer": "No, contraceptives do not cause infertility. Fertility returns after stopping most methods, though it may take time depending on the method."
        },
        {
            "question": "Can I use contraceptives while breastfeeding?",
            "answer": "Some methods are safe during breastfeeding (progestin-only pills, IUDs, implants), while combined hormonal methods may affect milk supply. Consult your doctor."
        }
    ],
    "Mythbusters": [
        {
            "myth": "Contraceptives are only for married women.",
            "fact": "Contraceptives are for anyone who wants to prevent pregnancy, regardless of marital status."
        },
        {
            "myth": "Using contraceptives is against religious beliefs.",
            "fact": "Most major religions allow family planning methods that don't terminate pregnancy."
        },
        {
            "myth": "Contraceptives cause weight gain.",
            "fact": "While some hormonal methods may cause temporary fluid retention, significant weight gain is uncommon."
        },
        {
            "myth": "Emergency contraception is the same as abortion.",
            "fact": "Emergency contraception prevents pregnancy from occurring; it does not terminate an existing pregnancy."
        },
        {
            "myth": "Natural methods are just as effective as modern methods.",
            "fact": "Natural methods have much higher failure rates compared to modern contraceptives when not used perfectly."
        }
    ],
    "Safe Usage Tips": [
        "Always read and follow the instructions for your chosen method.",
        "Use condoms along with other methods for STD protection.",
        "Set reminders for methods that require regular use (pills, patches).",
        "Have emergency contraception available as backup.",
        "Get regular check-ups if using hormonal methods or IUDs.",
        "Discuss options with your healthcare provider to find the best method for you."
    ]
}

# Risk assessment logic
def calculate_risk_score(input_data):
    risk_score = 0
    
    # Age risk
    age = input_data.get('wife_age', 25)
    if age < 18 or age > 40:
        risk_score += 2
    elif age < 20 or age > 35:
        risk_score += 1
    
    # Medical conditions
    conditions = [
        'high_blood_pressure', 'history_of_clots', 'breast_cancer_history',
        'liver_disease', 'has_hypertension', 'has_diabetes',
        'has_chronic_heart_disease', 'has_chronic_lung_disease',
        'has_chronic_kidney_disease', 'has_chronic_liver_disease',
        'is_smoker', 'taking_tb_medication', 'taking_anticonvulsants'
    ]
    
    for condition in conditions:
        if input_data.get(condition, 0) == 1:
            risk_score += 2 if condition in ['high_blood_pressure', 'history_of_clots', 'has_diabetes'] else 1
    
    # Normalize score to 0-10 scale
    risk_score = min(10, risk_score)
    
    return risk_score

# User input form
def user_input_form():
    with st.sidebar:
        st.header("Personal Information")
        
        with st.form("user_input"):
            # Basic information
            age = st.number_input("Age", min_value=15, max_value=60, value=25)
            wife_education = st.selectbox("Education Level", [1, 2, 3, 4], 
                                        format_func=lambda x: ["Low", "Medium-Low", "Medium-High", "High"][x-1])
            husband_education = st.selectbox("Partner's Education Level", [1, 2, 3, 4], 
                                           format_func=lambda x: ["Low", "Medium-Low", "Medium-High", "High"][x-1])
            num_children = st.number_input("Number of Children", min_value=0, max_value=15, value=1)
            
            # Socioeconomic factors
            st.subheader("Socioeconomic Factors")
            wife_religion = st.selectbox("Religion", [0, 1], format_func=lambda x: ["Other", "Islam"][x])
            wife_now_working = st.selectbox("Currently Working", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            standard_of_living = st.selectbox("Standard of Living Index", [1, 2, 3, 4], 
                                             format_func=lambda x: ["Low", "Medium-Low", "Medium-High", "High"][x-1])
            media_exposure = st.selectbox("Media Exposure", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            
            # Medical history
            st.subheader("Medical History")
            col1, col2 = st.columns(2)
            with col1:
                high_blood_pressure = st.checkbox("High Blood Pressure", value=False)
                history_of_clots = st.checkbox("History of Blood Clots", value=False)
                breast_cancer_history = st.checkbox("Family History of Breast Cancer", value=False)
                liver_disease = st.checkbox("Liver Disease", value=False)
                has_diabetes = st.checkbox("Diabetes", value=False)
            with col2:
                has_hypertension = st.checkbox("Hypertension", value=False)
                is_smoker = st.checkbox("Smoker", value=False)
                taking_tb_medication = st.checkbox("Taking TB Medication", value=False)
                taking_anticonvulsants = st.checkbox("Taking Anticonvulsants", value=False)
            
            submitted = st.form_submit_button("Get Recommendation")
            
            if submitted:
                input_data = {
                    'wife_age': int(age),
                    'wife_education': int(wife_education),
                    'husband_education': int(husband_education),
                    'num_children': int(num_children),
                    'wife_religion': int(wife_religion),
                    'wife_now_working': int(wife_now_working),
                    'standard_of_living': int(standard_of_living),
                    'media_exposure': int(media_exposure),
                    'high_blood_pressure': int(high_blood_pressure),
                    'history_of_clots': int(history_of_clots),
                    'breast_cancer_history': int(breast_cancer_history),
                    'liver_disease': int(liver_disease),
                    'has_diabetes': int(has_diabetes),
                    'has_hypertension': int(has_hypertension),
                    'is_smoker': int(is_smoker),
                    'taking_tb_medication': int(taking_tb_medication),
                    'taking_anticonvulsants': int(taking_anticonvulsants),
                    'thyroid_disorder': 0,
                    'has_arthritis': 0,
                    'has_anxiety': 0,
                    'has_depression': 0,
                    'has_chronic_kidney_disease': 0,
                    'has_chronic_liver_disease': 0,
                    'has_chronic_heart_disease': 0,
                    'has_chronic_lung_disease': 0,
                    'recent_childbirth': 0,
                    'is_breastfeeding': 0,
                    'has_surgery': 0,
                    'pcos_diagnosed': 0
                }
                
                # Calculate derived features
                input_data['education_gap'] = int(input_data['husband_education']) - int(input_data['wife_education'])
                input_data['health_risk_score'] = calculate_risk_score(input_data)
                
                # Create age group
                if age <= 25:
                    age_group = '18-25'
                elif age <= 35:
                    age_group = '26-35'
                elif age <= 45:
                    age_group = '36-45'
                else:
                    age_group = '46+'
                input_data['age_group'] = age_group
                
                # Create children group
                if num_children == 0:
                    children_group = '0'
                elif num_children <= 2:
                    children_group = '1-2'
                elif num_children <= 4:
                    children_group = '3-4'
                else:
                    children_group = '5+'
                input_data['children_group'] = children_group
                
                # Create education levels
                input_data['wife_education_level'] = 'low' if wife_education <= 2 else 'high'
                input_data['husband_education_level'] = 'low' if husband_education <= 2 else 'high'
                input_data['standard_of_living_level'] = 'low' if standard_of_living <= 2 else 'high'
                
                return input_data
    return None

# Doctor dashboard
def doctor_dashboard():
    st.title("Doctor Dashboard")
    
    # Simple authentication
    password = st.text_input("Enter Doctor Password", type="password")
    if password != "contracare123":
        st.warning("Please enter correct password to access doctor dashboard")
        return
    
    st.success("Authenticated successfully!")
    
    # Upload CSV for bulk predictions
    uploaded_file = st.file_uploader("Upload Patient CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Predict for All Patients"):
                required_cols = [
                    'wife_age', 'wife_education', 'husband_education', 'num_children',
                    'wife_religion', 'wife_now_working', 'standard_of_living', 'media_exposure'
                ]
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    return
                
                medical_conditions = [
                    'high_blood_pressure', 'history_of_clots', 'breast_cancer_history',
                    'liver_disease', 'has_diabetes', 'has_hypertension', 'is_smoker',
                    'taking_tb_medication', 'taking_anticonvulsants'
                ]
                
                for condition in medical_conditions:
                    if condition not in df.columns:
                        df[condition] = 0
                
                # Convert data types
                for col in required_cols + medical_conditions:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                
                # Make predictions
                predictions = model.predict(df)
                df['Predicted Method'] = [method_mapping[p] for p in predictions]
                
                st.subheader("Prediction Results")
                st.dataframe(df[required_cols + ['Predicted Method']])
                
                st.subheader("Prediction Distribution")
                fig, ax = plt.subplots()
                df['Predicted Method'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_ylabel('')
                st.pyplot(fig)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Predictions",
                    csv,
                    "contraceptive_predictions.csv",
                    "text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Knowledge Base section
def knowledge_base_section():
    st.title("Contraceptive Knowledge Base")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Types", "FAQs", "Mythbusters", "Usage Tips"])
    
    with tab1:
        st.subheader("Types of Contraceptives")
        for method, details in knowledge_base["Types of Contraceptives"].items():
            with st.expander(method):
                st.markdown(f"**Description**: {details['description']}")
                st.markdown(f"**Examples**: {details['examples']}")
                st.markdown(f"**Effectiveness**: {details['effectiveness']}")
                st.markdown(f"**Side Effects**: {details['side_effects']}")
    
    with tab2:
        st.subheader("Frequently Asked Questions")
        for faq in knowledge_base["FAQs"]:
            with st.expander(faq["question"]):
                st.write(faq["answer"])
    
    with tab3:
        st.subheader("Common Myths and Facts")
        for myth in knowledge_base["Mythbusters"]:
            with st.expander(myth["myth"]):
                st.markdown(f"**Fact**: {myth['fact']}")
    
    with tab4:
        st.subheader("Safe Usage Tips")
        for tip in knowledge_base["Safe Usage Tips"]:
            st.markdown(f"- {tip}")

# Model explanation section
def model_explanation():
    st.title("How Our Recommendation System Works")
    
    st.markdown("""
    Our AI model analyzes multiple factors to recommend the most suitable contraceptive method:
    - **Demographic factors**: Age, education level, number of children
    - **Socioeconomic factors**: Standard of living, media exposure
    - **Health factors**: Medical history, risk factors
    """)
    
    if shap_explainer is not None:
        st.subheader("Feature Importance")
        st.image("shap_summary.png", caption="SHAP Feature Importance", use_column_width=True)
        st.markdown("""
        This chart shows which factors most influence the model's recommendations:
        - Features extending to the right increase likelihood of long-term methods
        - Features extending to the left increase likelihood of no-use or short-term methods
        """)
    else:
        st.warning("Model explanation not available for the current model")

# Main app
def main():
    st.title("ContraCare: AI-Driven Contraceptive Recommendation")
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Home", "Get Recommendation", "Knowledge Base", "Doctor Login"])
    
    if page == "Home":
        st.markdown("""
        ## Welcome to ContraCare
        
        Our AI-powered system helps women and healthcare providers make informed decisions about 
        contraceptive methods based on individual health profiles and lifestyle factors.
        
        **How it works:**
        1. Provide your information in the "Get Recommendation" section
        2. Receive personalized contraceptive recommendation
        3. Learn about your options in the Knowledge Base
        
        **Disclaimer:**
        - This tool provides informational recommendations only
        - Always consult with a healthcare provider before making decisions
        - Your data is not stored permanently and remains confidential
        """)
        
        st.image("contraceptive_methods.jpg", caption="Various Contraceptive Methods", use_column_width=True)
    
    elif page == "Get Recommendation":
        st.header("Personalized Contraceptive Recommendation")
        
        input_data = user_input_form()
        
        if input_data is not None and model is not None and feature_columns is not None:
            try:
                # Convert input data to DataFrame for prediction
                input_df = pd.DataFrame([input_data])
                
                # Ensure all feature columns are present
                for col in feature_columns:
                    if col not in input_df.columns:
                        if col in ['wife_age', 'num_children', 'education_gap', 'health_risk_score']:
                            input_df[col] = 0.0
                        else:
                            input_df[col] = 0 if col in [
                                'high_blood_pressure', 'history_of_clots', 'breast_cancer_history',
                                'liver_disease', 'has_diabetes', 'has_hypertension', 'is_smoker',
                                'taking_tb_medication', 'taking_anticonvulsants'
                            ] else ''
                
                # Convert numerical columns to float
                numerical_cols = ['wife_age', 'num_children', 'education_gap', 'health_risk_score']
                input_df[numerical_cols] = input_df[numerical_cols].astype(float)
                
                # Convert categorical columns to string
                categorical_cols = [col for col in feature_columns if col not in numerical_cols]
                input_df[categorical_cols] = input_df[categorical_cols].astype(str)
                
                # Reorder columns to match training data
                input_df = input_df[feature_columns]
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]
                
                # Display results
                st.success(f"Recommended Method: **{method_mapping[prediction]}**")
                
                # Recommendation notes
                st.subheader("Recommendation Notes")
                if prediction == 1:
                    st.info("""
                    **No-use Recommendation Notes:**
                    - May be appropriate if not sexually active or trying to conceive
                    - Consider natural family planning methods if avoiding pregnancy
                    - Remember that no method is 100% effective against pregnancy
                    """)
                elif prediction == 2:
                    st.info("""
                    **Short-term Method Notes:**
                    - Good option for temporary pregnancy prevention
                    - Requires consistent daily/weekly/monthly use
                    - Easily reversible when you want to conceive
                    - Consider combining with condoms for STD protection
                    """)
                else:
                    st.info("""
                    **Long-term Method Notes:**
                    - Ideal for extended pregnancy prevention (3-10 years)
                    - Highly effective with minimal user effort
                    - Requires healthcare provider for insertion/removal
                    - Consider side effects and personal health factors
                    """)
                
                # Risk assessment
                st.subheader("Personalized Risk Assessment")
                risk_score = calculate_risk_score(input_data)
                st.progress(risk_score/10)
                st.write(f"Risk Score: {risk_score}/10")
                
                if risk_score < 3:
                    st.success("Low risk profile for contraceptive use")
                elif risk_score < 7:
                    st.warning("Moderate risk profile - consult with healthcare provider")
                else:
                    st.error("High risk profile - requires medical consultation before use")
                
                # Risk factors visualization
                risk_factors = []
                if input_data['wife_age'] < 18 or input_data['wife_age'] > 40:
                    risk_factors.append(("Age", 2))
                if input_data['is_smoker']:
                    risk_factors.append(("Smoking", 2))
                if input_data['high_blood_pressure']:
                    risk_factors.append(("High BP", 2.5))
                if input_data['has_diabetes']:
                    risk_factors.append(("Diabetes", 2.5))
                if input_data['history_of_clots']:
                    risk_factors.append(("Blood Clots", 3))
                
                if risk_factors:
                    st.subheader("Key Risk Factors")
                    risk_df = pd.DataFrame(risk_factors, columns=["Factor", "Risk Weight"])
                    fig, ax = plt.subplots()
                    sns.barplot(data=risk_df, x="Risk Weight", y="Factor", ax=ax)
                    ax.set_title("Your Top Risk Factors")
                    st.pyplot(fig)
                
                # Model explanation
                with st.expander("How was this recommendation determined?"):
                    model_explanation()
                
            except Exception as e:
                st.error(f"Error generating recommendation: {e}")
                st.error("Please check your inputs and try again.")
    
    elif page == "Knowledge Base":
        knowledge_base_section()
    
    elif page == "Doctor Login":
        doctor_dashboard()

if __name__ == "__main__":
    main()