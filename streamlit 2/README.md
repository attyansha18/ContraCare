# ContraCare - AI-Powered Contraceptive Method Recommender

ContraCare is a machine learning-based decision support system that recommends appropriate contraceptive methods while considering medical history, demographic factors, and lifestyle preferences. The system aims to provide personalized guidance while minimizing potential side effects.

## Features

- 🤖 AI-powered contraceptive method recommendations
- 🏥 Medical safety checks and contraindication warnings
- 📊 Detailed method effectiveness and suitability information
- 💡 Explainable AI with SHAP value analysis
- 📱 User-friendly web interface
- 🔒 Privacy-focused (no data storage)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/contracare.git
cd contracare
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model (first time only):
```bash
python train_model.py
```

2. Run the web application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to `http://localhost:8501`

## Data Source

This project uses the Contraceptive Method Choice (CMC) dataset from the UCI Machine Learning Repository, which includes factors such as:
- Age
- Education level
- Number of children
- Socioeconomic status
- Medical background

## Model Details

The system uses a Random Forest Classifier with:
- Cross-validation for robust performance
- Hyperparameter tuning for optimal results
- SHAP values for explainability
- Medical contraindication checks

## Safety and Disclaimer

⚠️ **Important:** This tool provides general guidance only. Always consult with a healthcare provider before making decisions about contraceptive methods. The recommendations are based on statistical patterns and may not account for all individual circumstances.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the CMC dataset
- WHO for contraceptive guidelines
- Contributors and maintainers 