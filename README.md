# Decision Tree Classifier - PlayTennis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)

## Overview

This project implements a **Decision Tree Classification algorithm** to predict whether to play tennis based on weather conditions. It demonstrates the complete machine learning workflow including data preprocessing, model training, hyperparameter tuning, and comprehensive evaluation.

### Key Features
- ✅ Complete data preprocessing pipeline with One-Hot Encoding
- ✅ Decision Tree classifier implementation from scikit-learn
- ✅ Hyperparameter tuning using GridSearchCV
- ✅ Model visualization with decision boundaries
- ✅ Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- ✅ Confusion matrix and classification report
- ✅ Model serialization with pickle

---

## Dataset Description

The dataset contains weather conditions and a binary target variable indicating whether to play tennis.

### Features:
- **Outlook**: Sunny, Overcast, Rain
- **Temperature**: Hot, Mild, Cool
- **Humidity**: High, Normal
- **Windy**: True, False

### Target:
- **Play**: Yes/No (binary classification)

**Dataset Size**: 14 samples with 4 categorical features

---

## Project Structure

```
Decision-Tree-Classifier-PlayTennis/
├── decision_tree_classifier.ipynb    # Main Jupyter Notebook
├── play_decision_tree_pipeline.pkl   # Trained pipeline model
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
├── LICENSE                           # MIT License
└── .gitignore                        # Git ignore file
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/meet2121/Decision-Tree-Classifier-PlayTennis.git
cd Decision-Tree-Classifier-PlayTennis
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Notebook

```bash
jupyter notebook decision_tree_classifier.ipynb
```

### Making Predictions

```python
import pickle
import pandas as pd

# Load the trained pipeline
with open('play_decision_tree_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Create a sample input
sample_data = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['High'],
    'Windy': [False]
})

# Make prediction
prediction = pipeline.predict(sample_data)
print(f"Prediction: {prediction[0]}")  # Output: 'No' or 'Yes'

# Get prediction probabilities
proba = pipeline.predict_proba(sample_data)
print(f"Probabilities: {proba}")
```

---

## Model Performance

### Test Set Results
- **Accuracy**: 75%
- **Precision (Yes)**: 100%
- **Recall (Yes)**: 67%
- **F1-Score (Yes)**: 0.80

### Confusion Matrix
```
          Predicted
          No    Yes
Actual No  1     0
Actual Yes 1     2
```

---

## Implementation Details

### Data Preprocessing
- **Categorical Encoding**: One-Hot Encoding applied to all features
- **Train-Test Split**: 75%-25% stratified split
- **Random State**: 42 (for reproducibility)

### Model Configuration
- **Algorithm**: Decision Tree Classifier
- **Criterion**: Gini impurity
- **Splitter**: Best split
- **Max Depth**: Tuned via GridSearchCV
- **Min Samples Leaf**: 1 (tuned parameter)

### Hyperparameter Tuning
- **GridSearchCV**: 3-fold cross-validation
- **Parameter Grid**:
  - `max_depth`: [None, 1, 2, 3, 4]
  - `min_samples_leaf`: [1, 2]

---

## Key Components

### 1. Data Exploration
- Dataset loading and initial exploration
- Feature distribution analysis
- Class balance verification

### 2. Preprocessing Pipeline
```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(sparse_output=False, drop='first'),
         ['Outlook', 'Temperature', 'Humidity', 'Windy'])
    ],
    remainder='drop'
)
```

### 3. Model Training
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

gridsearch = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
gridsearch.fit(X_train, y_train)
```

### 4. Model Evaluation
```python
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## Visualization

The notebook includes:
- **Decision Tree Visualization**: Full tree structure with decision boundaries
- **Feature Importance**: Relative importance of features
- **Performance Metrics**: Detailed classification metrics

---

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

See `requirements.txt` for complete list.

---

## Results & Interpretation

The trained model achieves:
- **Good predictive performance** on the test set (75% accuracy)
- **Perfect precision** for positive class (100%)
- **Reasonable recall** for positive class (67%)

### Key Insights
- Weather conditions are moderately predictive of play decisions
- Some weather combinations lead to clear decisions
- The model works well for class predictions

---

## Future Improvements

- ⏳ Expand dataset with more weather conditions
- ⏳ Experiment with ensemble methods (Random Forest, XGBoost)
- ⏳ Implement cross-validation strategies
- ⏳ Add feature scaling and normalization
- ⏳ Deploy model as a REST API
- ⏳ Create web interface for predictions

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**meet2121**
- GitHub: [@meet2121](https://github.com/meet2121)
- Email: your-email@example.com

---

## References

- [scikit-learn Decision Trees Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [Decision Tree Algorithm Explained](https://en.wikipedia.org/wiki/Decision_tree)
- [PlayTennis Dataset - Classic ML Example](https://archive.ics.uci.edu/ml/datasets/Weather+and+Play+Tennis)

---

## Acknowledgments

- Built with [❤️](https://github.com) and Python
- Powered by scikit-learn and pandas
- Inspired by classic machine learning projects
