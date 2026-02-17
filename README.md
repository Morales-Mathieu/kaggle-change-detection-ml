# Kaggle Competition — 2EL1730 Machine Learning Project 2026

## Link
- **Rapport (PDF)** : [reports/armandcoiffe_coiffe_morales_grille.pdf](reports/armandcoiffe_coiffe_morales_grille.pdf)
- **Notebook HTML** : [docs/index.html](docs/index.html)


---

## Kaggle (Challenge)

### Overview
Analysing changes in multi-image, multi-date remote sensing data helps us to discover and understand global conditions. This challenge uses satellite imagery-derived geographical features. The data has been processed using computer vision techniques and is ready for exploration using machine learning methods.

The aim of this challenge is to classify a given geographical area into six categories.

### Classes
This challenge aims to classify a given geographical area into six classes:

- 'Demolition': 0  
- 'Road': 1  
- 'Residential': 2  
- 'Commercial': 3  
- 'Industrial': 4  
- 'Mega Projects': 5  

### Features
The geographical features are:
- An irregular polygon.
- Categorical values describing the status of the polygon on five different dates.
- Neighbourhood urban features (e.g., dense urban / industrial region).
- Neighbourhood geographic features (e.g., near a river and a hill).

### Pipeline
The proposed pipeline is similar to the one introduced in the first lecture and labs. The provided `skeleton_code.py` implements a simple k-NN baseline (~40%).

1. **Data preprocessing**: preprocess and convert the data into the appropriate format.
2. **Feature engineering & dimensionality reduction**:
   - Urban and Geographical types are multi-valued categorical columns (one-hot encoding can help).
   - Process polygons into features: area, perimeter, other geometric properties.
   - Days between consecutive dates can be useful.
3. **Learning algorithm**:
   - Choose a classifier: logistic regression / SVM / decision tree / neural net / ensemble, etc.
4. **Evaluation**:
   - Multi-class classification, metrics matter, generalization analysis.

### Evaluation
- Train with `train.geojson`, predict on `test.geojson`.
- Cross-validation is recommended: https://scikit-learn.org/stable/modules/cross_validation.html

**Evaluation metric:** Mean F1-Score.

**Submission format:** a CSV with:
```csv
Id,change_type
1,1
