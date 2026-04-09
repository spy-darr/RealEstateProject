# 🏠 PropSense AI — Real Estate Analytics & Property Valuation Platform

> **MCA Final Year Project** · AI/ML · Full-Stack · Production-Ready

An end-to-end machine learning system for predicting Indian property prices, analysing real estate investments, detecting listing anomalies, and recommending similar properties — served via a FastAPI backend and a modern HTML/JS dashboard.

---

## 📸 Screenshots

> **Dashboard — Market Overview**
> ![Dashboard](outputs/screenshots/dashboard.png)
> *(Screenshot placeholder — run the app to see the live interface)*

> **Price Prediction**
> ![Prediction](outputs/screenshots/prediction.png)

> **Investment Analysis**
> ![Investment](outputs/screenshots/investment.png)

---

## 🗂️ Project Structure

```
RealEstateAI/
├── data/                          # Raw & processed datasets (CSV)
├── notebooks/                     # Jupyter EDA / experiment notebooks
├── models/                        # Saved model artefacts (.pkl / .h5)
├── outputs/                       # EDA plots, predictions, reports (PNG, JSON, CSV)
│
├── src/
│   ├── data_collection/
│   │   └── data_loader.py         # Synthetic dataset generator & CSV loader
│   ├── preprocessing/
│   │   └── preprocess.py          # Missing-value imputation, encoding, scaling
│   ├── eda/
│   │   └── eda.py                 # EDA plots (7 charts saved to outputs/)
│   ├── feature_engineering/
│   │   └── feature_engineer.py    # price_per_sqft, amenities_score, value_score …
│   ├── models/
│   │   └── train_model.py         # Linear/Ridge/RandomForest/XGBoost training
│   ├── deep_learning/
│   │   └── deep_learning_model.py # Keras / sklearn MLP neural network
│   ├── recommendation/
│   │   └── recommendation.py      # Cosine-similarity property recommender
│   ├── investment_analysis/
│   │   └── investment_analysis.py # ROI, rental yield, EMI, investment grade
│   ├── anomaly_detection/
│   │   └── anomaly_detection.py   # Isolation Forest + Z-score + IQR flagging
│   ├── api/
│   │   └── api.py                 # FastAPI REST backend
│   └── dashboard/
│       └── index.html             # Single-file HTML/JS/Chart.js dashboard
│
├── train_pipeline.py              # One-shot end-to-end training script
├── app.py                         # Main entry point (CLI)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 – 3.11
- pip or conda

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/RealEstateAI.git
cd RealEstateAI

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full training pipeline (generates data, trains models, saves artefacts)
python train_pipeline.py

# 5. Start the API server
python app.py --api

# 6. Open the dashboard (in a second terminal)
python app.py --dashboard
```

The dashboard will open automatically at **http://localhost:8080**  
The API Swagger UI is available at **http://localhost:8000/docs**

---

## 🚀 Usage

### Option A — All-in-one
```bash
python app.py --all
```
Runs the training pipeline, starts the API, and opens the dashboard.

### Option B — Individually
```bash
python app.py --pipeline    # Train models only
python app.py --api         # Start API server
python app.py --dashboard   # Open dashboard
```

### Option C — Direct module execution
```bash
python -m src.data_collection.data_loader
python -m src.models.train_model
python -m src.recommendation.recommendation
```

---

## 🔌 API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint       | Description                          |
|--------|---------------|--------------------------------------|
| GET    | `/health`      | Service health check                 |
| POST   | `/predict`     | Predict property price               |
| POST   | `/recommend`   | Get similar property recommendations |
| POST   | `/investment`  | ROI, rental yield, EMI analysis      |
| POST   | `/anomaly`     | Detect anomalous listings            |
| GET    | `/cities`      | List supported cities                |
| GET    | `/model-info`  | Active model info                    |

### Example: Price Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Pune",
    "locality": "Koregaon Park",
    "property_type": "Apartment",
    "bedrooms": 3,
    "bathrooms": 2,
    "area_sqft": 1350,
    "age_years": 4,
    "floor_number": 6,
    "total_floors": 14,
    "location_score": 8.5,
    "amenities": "Gym|Pool|Parking|Security|Lift"
  }'
```

### Example: Investment Analysis
```bash
curl -X POST http://localhost:8000/investment \
  -H "Content-Type: application/json" \
  -d '{"price": 8500000, "city": "Pune"}'
```

---

## 📊 ML Models & Performance

| Model             | R²     | RMSE (₹)  | MAPE (%) |
|-------------------|--------|-----------|----------|
| Linear Regression | ~0.72  | —         | ~18%     |
| Ridge Regression  | ~0.73  | —         | ~18%     |
| Random Forest     | ~0.91  | —         | ~9%      |
| **XGBoost**       | **~0.93** | —      | **~8%**  |
| Neural Network    | ~0.89  | —         | ~10%     |

*Results vary by random seed. Best model is auto-selected and saved as `models/best_model.pkl`.*

---

## 🧠 Features Engineered

| Feature            | Description                                      |
|--------------------|--------------------------------------------------|
| `price_per_sqft`   | price ÷ area_sqft                               |
| `amenities_count`  | Count of amenities from pipe-separated string    |
| `amenities_score`  | Normalised 0-10 score                            |
| `bed_bath_ratio`   | bedrooms ÷ bathrooms                            |
| `floor_ratio`      | floor_number ÷ total_floors                     |
| `area_per_bedroom` | area_sqft ÷ bedrooms                            |
| `city_price_index` | City median ppsf normalised to [0,1]            |
| `is_premium_loc`   | 1 if premium locality                           |
| `property_age_group` | New / Mid / Old categorical                   |
| `value_score`      | Composite investment attractiveness 0-10        |

---

## 📦 Dataset

A synthetic dataset of **2,000 Indian residential properties** is auto-generated if no CSV is found.  
Cities: Mumbai, Pune, Bangalore, Hyderabad, Chennai, Delhi, Kolkata, Ahmedabad, Jaipur, Surat  
Property types: Apartment, Villa, Independent House, Studio, Penthouse  

The dataset is saved to `data/properties.csv` on first run.

---

## 🌐 GitHub Pages / Hosting

The dashboard (`src/dashboard/index.html`) is a **self-contained static file** that can be hosted anywhere:

```bash
# Deploy to GitHub Pages
git add .
git commit -m "Initial commit"
git push origin main
# Then enable GitHub Pages for the /src/dashboard folder (or copy index.html to docs/)
```

For the API, deploy to **Render**, **Railway**, or **Heroku**:
```bash
# Procfile (Heroku)
web: uvicorn src.api.api:app --host 0.0.0.0 --port $PORT
```

---

## 🔧 Configuration

Create a `.env` file in the project root to override defaults:
```
API_HOST=0.0.0.0
API_PORT=8000
DASHBOARD_PORT=8080
DATA_PATH=data/properties.csv
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

Developed as an MCA Final Year Project.  
Designed to be extended into a startup MVP for Indian real estate market intelligence.

---

*Built with ❤️ using Python · scikit-learn · XGBoost · TensorFlow · FastAPI · Chart.js*
