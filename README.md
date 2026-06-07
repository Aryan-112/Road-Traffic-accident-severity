# 🚗 Road Traffic Accident Severity Prediction

A machine learning project that classifies road traffic accident severity into **Slight Injury**, **Serious Injury**, or **Fatal Injury** using XGBoost.

---

## 📌 Overview

Road accidents are a major global concern. In 2024, Australia recorded 1,291 road fatalities (DITRDC, 2025). This project builds a multi-class classification model using real-world accident data to predict injury severity based on driver demographics, vehicle characteristics, road conditions, and environmental variables.

The goal is to identify factors that contribute to the most dangerous accidents, which can inform emergency response prioritisation and transport safety policy.

---

## 📂 Dataset

- **Source:** [Kaggle – RTA Dataset: Accident Prediction Dataset](https://www.kaggle.com/datasets/adityaakuskar/accident-prediction-dataset) (Aditya Akuskar)
- **Format:** CSV
- **Size:** ~12,300 rows, 32 columns
- **Target variable:** `Accident_severity` (Slight Injury / Serious Injury / Fatal Injury)

Key features include: time of accident, day of week, driver age/sex/education, vehicle type, road alignment, weather conditions, light conditions, collision type, and more.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Google Colab | Development environment |
| Pandas | Data manipulation |
| Scikit-learn | Preprocessing, train-test split, evaluation |
| XGBoost | Classification model |
| Matplotlib / Seaborn | Visualisation |

---

## ⚙️ Project Pipeline

### 1. Data Exploration
- Identified missing values across 20+ columns (ranging from 1% to 35% missing)
- Generated a missing value heatmap for visual analysis
- Documented a full data dictionary for all 32 attributes

### 2. Preprocessing
- **Dropped** `Work_of_casualty` (25.97% missing, low correlation to target)
- **Filled with "Unknown"** context-sensitive columns with high missingness: `Defect_of_vehicle`, `Fitness_of_casuality`, `Service_year_of_vehicle`, `Driving_experience`, `Owner_of_vehicle`, `Area_accident_occured`
- **Filled with Mode** columns with lower variance and low missingness: `Educational_level`, `Type_of_vehicle`, `Vehicle_driver_relation`, and others
- **Time conversion:** Extracted hour of day from the `Time` column; dropped the original

### 3. Encoding
- **Label Encoding** applied to ordinal columns (e.g., `Educational_level`, `Age_band_of_driver`)
- **One-Hot Encoding** applied to nominal columns via `pd.get_dummies()` with `drop_first=True` to avoid the dummy variable trap
- **Target Encoding:** `Accident_severity` encoded as `0 = Fatal Injury`, `1 = Serious Injury`, `2 = Slight Injury`

### 4. Training & Testing
- **Split:** 80% training / 20% testing
- Stratified split to maintain class proportions
- `random_state=42` for reproducibility

### 5. Model – XGBoost Classifier
```python
model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42
)
```
- `multi:softprob` outputs a probability distribution across all 3 classes
- Sequential boosted decision trees correct errors from prior trees
- Loss function: **Multiclass Log Loss**

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **85.51%** |
| Weighted F1-Score | **0.8048** |
| Final Training Log Loss | 0.1941 |
| Final Test Log Loss | 0.4467 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fatal Injury | 1.00 | 0.10 | 0.18 | 31 |
| Serious Injury | 0.69 | 0.09 | 0.17 | 349 |
| Slight Injury | 0.86 | 0.99 | 0.92 | 2,084 |

**Key observations:**
- The model excels at predicting **Slight Injury** (most frequent class) with a 0.92 F1-score
- **Serious Injury** recall is low (0.09) due to class imbalance and similarity to Slight Injury
- **Fatal Injury** has perfect precision (1.00) — when the model flags a fatal case, it is always correct, though it rarely does so (low recall)

---

## 🔗 Links

- 📓 **Google Colab Notebook:** [Open in Colab](https://colab.research.google.com/drive/1h3S4IovpL2o3PZNpPykAviXCLxHLASTG?usp=sharing)
- 🤖 **AI Assistance Log:** [ChatGPT conversation](https://chatgpt.com/share/68e615c7-eec4-800b-a76e-8770bdd2ec51)

---

## 🚀 Future Work

- Address class imbalance using techniques such as SMOTE or class weighting to improve recall for Serious and Fatal Injury classes
- Experiment with hyperparameter tuning (learning rate, max depth, n_estimators)
- Explore feature importance to identify the strongest predictors of fatal accidents
- Test alternative models (LightGBM, CatBoost) for comparison
- Collect more data for underrepresented classes (Fatal Injury: only 31 test samples)

---

## 📖 References

- Analytics Vidhya. (2021). *Binary cross entropy / log loss for binary classification.* https://www.analyticsvidhya.com/blog/2021/03/binary-cross-entropy-log-loss-for-binary-classification/
- DITRDC. (2025). *Monthly road deaths.* https://datahub.roadsafety.gov.au/progress-reporting/monthly-road-deaths
- Akuskar, A. (2025). *RTA Dataset: Accident Prediction Dataset.* Kaggle. https://www.kaggle.com/datasets/adityaakuskar/accident-prediction-dataset
- MLJAR. (n.d.). *Visualize XGBoost tree.* https://mljar.com/blog/visualize-xgboost-tree/
- OpenAI. (2023). *ChatGPT (GPT-4).* https://openai.com/chatgpt

---

## 👤 Author

**Aryan Raheja** — Student ID: 24956928  
Bachelor of Computing Science (AI & Data Analytics), University of Technology Sydney  
Machine Learning Assignment 2
