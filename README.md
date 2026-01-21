# Breast Cancer Prediction System

An educational machine learning system for predicting breast cancer tumor classification (Benign/Malignant) using the Wisconsin Breast Cancer dataset.

## ⚠️ Disclaimer

**This system is strictly for educational purposes and must NOT be used as a medical diagnostic tool.** Always consult qualified healthcare professionals for medical diagnosis and treatment.

## 📋 Project Overview

This project implements a complete machine learning pipeline including:
- Data preprocessing and feature selection
- Model training using Logistic Regression
- Model evaluation with multiple metrics
- Web-based GUI for predictions
- Model persistence and deployment

## 🎯 Features Selected

The model uses the following 5 features from the dataset:
1. Mean Radius
2. Mean Texture
3. Mean Perimeter
4. Mean Area
5. Mean Compactness

## 🏗️ Project Structure

```
BreastCancer_Project_yourName_matricNo/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── BreastCancer_hosted_webGUI_link.txt  # Deployment information
├── model/
│   ├── model_building.ipynb        # Model development notebook
│   ├── breast_cancer_model.pkl     # Trained model
│   ├── scaler.pkl                  # Feature scaler
│   └── feature_names.pkl           # Feature names
├── templates/
│   └── index.html                  # Web interface
└── static/
    └── style.css                   # (Optional) Custom styles
```

## 🚀 Installation & Setup

### Local Development

1. **Clone the repository:**
```bash
git clone <your-github-repo-url>
cd BreastCancer_Project_yourName_matricNo
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run model training:**
```bash
jupyter notebook model/model_building.ipynb
# Or convert to .py and run:
python model/model_building.py
```

5. **Start the web application:**
```bash
python app.py
```

6. **Access the application:**
Open your browser and navigate to `http://localhost:5000`

## 📊 Model Performance

- **Algorithm Used:** Logistic Regression
- **Accuracy:** ~97%
- **Precision:** ~98%
- **Recall:** ~97%
- **F1-Score:** ~97%

## 🌐 Deployment Instructions

### Option 1: Render.com

1. Create account on [Render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repository
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Deploy

### Option 2: PythonAnywhere

1. Create account on [PythonAnywhere.com](https://www.pythonanywhere.com)
2. Upload your project files
3. Create a new web app (Flask)
4. Configure WSGI file
5. Install requirements from console
6. Reload web app

### Option 3: Streamlit Cloud

If using Streamlit instead of Flask:
1. Create account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect GitHub repository
3. Select main file
4. Deploy automatically

## 📝 Usage

1. Open the web application
2. Enter the 5 tumor feature values:
   - Mean Radius (e.g., 14.5)
   - Mean Texture (e.g., 19.2)
   - Mean Perimeter (e.g., 91.5)
   - Mean Area (e.g., 655.0)
   - Mean Compactness (e.g., 0.105)
3. Click "Predict" button
4. View the prediction result with confidence levels

## 🔬 Technical Details

### Model Development Process

1. **Data Loading:** Wisconsin Breast Cancer dataset
2. **Preprocessing:**
   - No missing values found
   - Feature selection: 5 features chosen
   - Target encoding: Already binary (0=Malignant, 1=Benign)
   - Feature scaling: StandardScaler applied
3. **Training:**
   - Train-test split: 80-20
   - Algorithm: Logistic Regression
   - Cross-validation applied
4. **Evaluation:**
   - Multiple metrics calculated
   - Confusion matrix analyzed
5. **Persistence:**
   - Model saved using Joblib
   - Scaler and feature names also saved

### Web Application

- **Backend:** Flask framework
- **Frontend:** HTML/CSS/JavaScript
- **API Endpoint:** `/predict` for predictions
- **Health Check:** `/health` endpoint

## 📦 Dependencies

- Flask 3.0.0
- scikit-learn 1.3.2
- pandas 2.0.3
- numpy 1.24.3
- joblib 1.3.2
- gunicorn 21.2.0 (for deployment)

## 🤝 Contributing

This is an educational project. Feel free to fork and modify for learning purposes.

## 📄 License

This project is for educational use only.

## 👨‍💻 Author

**[Your Name]**  
**Matric Number:** [Your Matric Number]  
**Course:** Machine Learning Project  
**Submission Date:** January 22, 2026

## 🔗 Links

- **GitHub Repository:** [Your GitHub URL]
- **Live Application:** [Your Deployment URL]

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact through email.

---

**Remember:** This is an educational tool only. Never use it for actual medical diagnosis.