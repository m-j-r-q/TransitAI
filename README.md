# TransitAI

## Project Description

TransitAI is a Machine Learning model with the goal of detecting exoplanets using the transit method, where the brightness of stars is measured over time to detect the periodic dimming caused by a planet passing in front of its host star. 

The model is a Random Forest Classifier trained on the Cumulative Kepler Objects of Interest (KOI) data set provided by the NASA Exoplanet Science Institute, consisting of over 9500 Treshold Crossing Events (TCEs) detected by NASA's Kepler Space Telescope. Threshold Crossing Events (TCEs) are periodic signals in Kepler's photometry that exceed a detection threshold and are flagged as potential planet transits. The model achieves 91.6% accuracy and an AUC of 0.972 in distinguishing planetary candidates from false positives.

TransitAI was deliberately trained without the koi_fpflag_* columns which are human-derived flags that directly veto specific known false positive scenarios such as eclipsing binaries and stray light contamination. By excluding these flags, the model makes predictions purely from raw observational physics like transit geometry, stellar properties and centroid motion statistics making it capable of identifying candidates that traditional vetting pipelines might miss.

**Model Performance:**

| Metric | Value |
|---|---|
| Accuracy | 91.57% |
| AUC Score | 0.972 |
| Precision | 0.92 |
| CANDIDATE Recall | 0.89 |

---

## How the Model Works

The model was trained using a `scikit-learn` Pipeline consisting of a Random Forest Clasifier, tuned via GridSearchCV with 5-fold cross validation across 450 hyperparameter combinations.

Feature selection was performed using `feature_importances_` from an initial Random Forest fit on the full dataset. The top 10 features were than selected. For more info refer to `Model.ipynb`.

**Selected Features:**

| Feature | Description |
|---|---|
| `koi_dikco_msky` | Sky offset between difference image and KIC position (arcsec) |
| `koi_prad` | Planet radius (Earth radii) |
| `koi_ror` | Ratio of planet to star radius |
| `koi_fwm_stat_sig` | Flux-weighted centroid offset significance |
| `koi_dor` | Ratio of orbital distance to star radius |
| `koi_dicco_msky` | Sky offset between OOT and difference image centroids (arcsec) |
| `koi_period` | Orbital period (days) |
| `koi_max_mult_ev` | Maximum Multiple Event Statistic |
| `koi_depth` | Transit depth (ppm) |
| `koi_max_sngle_ev` | Maximum Single Event Statistic |

For a full walkthrough of the data preprocessing, feature selection, hyperparameter tuning, and evaluation pipeline, refer to `Model.ipynb`.

---

## Application Usage

TransitAI provides a Streamlit web interface where users can:

- **Input KOI feature values**: by using the sidebar to obtain a prediction
- **Load a sample candidate**: to see the model in action instantly
- **View model performance metrics**: ROC curve and confusion matrix
- **Read feature descriptions**: with links to the NASA KOI column documentation

The app will returns a classification of either **Planetary Candidate** or **False Positive**, along with a probability confidence gauge.

> The app expects input values from the NASA Cumulative KOI Table. Column definitions are available at the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html).

---

## Tech Stack

| Component | Technology |
|---|---|
| Model Training | `scikit-learn` — Random Forest, GridSearchCV, Pipeline |
| Data Processing | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Model Export | `joblib` |
| Web Application | `streamlit` |
| Training Progress | `joblib-progress` |
| Language | Python 3.x |

---

## Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/m-j-r-q/TransitAI.git
cd TransitAI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**

Open and run `Model.ipynb` in Jupyter or VS Code..

**4. Run the Streamlit app**
```bash
streamlit run TransitAI.py
```

The app will open in your browser at `http://localhost:8501`.

**Requirements (`requirements.txt`):**
```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
plotly
joblib
joblib-progress
```

---

## Dataset

The model is trained on the **NASA Cumulative Kepler Objects of Interest (KOI) Table**, available from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative).

---

## Note:

I am a student developer, and have no formal education in the subject of Data Science or Machine Learning. This project was initially built as part of the NASA Space Apps Challenge 2025, but the initial project had several errors that needed to be addressed. I have decided to revise and upload the project. If you are a professional and find any scientific or technical inaccuracy, your feedback would be greatly appreciated. Thanks.