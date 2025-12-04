"""
Decision Tree Classifier - PlayTennis

A machine learning implementation that predicts whether to play tennis
based on weather conditions using Decision Tree Classification.

Author: meet2121
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


def load_data():
    """Load the PlayTennis dataset."""
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 
                    'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 
                    'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 
                        'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
                     'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Windy': [False, True, False, False, False, True, True, False, False, 
                  False, True, True, False, True],
        'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 
                 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    
    df = pd.DataFrame(data)
    return df


def create_pipeline():
    """Create preprocessing and model pipeline."""
    # Preprocessing step: One-Hot Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(sparse_output=False, drop='first'),
             ['Outlook', 'Temperature', 'Humidity', 'Windy'])
        ],
        remainder='drop'
    )
    
    # Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', dt_classifier)
    ])
    
    return pipeline


def perform_grid_search(pipeline, X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    param_grid = {
        'classifier__max_depth': [None, 1, 2, 3, 4],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    gridsearch = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    
    gridsearch.fit(X_train, y_train)
    
    print("Best Parameters:", gridsearch.best_params_)
    print("Best CV Score:", gridsearch.best_score_)
    
    return gridsearch


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    return accuracy, cm, report


def visualize_tree(model, feature_names):
    """Visualize the decision tree."""
    clf = model.named_steps['classifier']
    
    plt.figure(figsize=(14, 8))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title('Decision Tree for PlayTennis Prediction')
    plt.tight_layout()
    plt.show()


def save_model(model, filename='decision_tree_pipeline.pkl'):
    """Save trained model to disk."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")


def main():
    """Main execution function."""
    print("=" * 50)
    print("Decision Tree Classifier - PlayTennis")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    # Prepare features and target
    X = df.drop('Play', axis=1)
    y = df['Play']
    
    # Train-test split
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )
    print(f"Train set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Create pipeline
    print("\n3. Creating pipeline...")
    pipeline = create_pipeline()
    
    # Hyperparameter tuning
    print("\n4. Performing hyperparameter tuning...")
    best_model = perform_grid_search(pipeline, X_train, y_train)
    
    # Evaluate model
    print("\n5. Evaluating model...")
    accuracy, cm, report = evaluate_model(best_model, X_test, y_test)
    
    # Save model
    print("\n6. Saving model...")
    save_model(best_model.best_estimator_)
    
    # Make predictions
    print("\n7. Making sample predictions...")
    sample = pd.DataFrame({
        'Outlook': ['Sunny'],
        'Temperature': ['Cool'],
        'Humidity': ['High'],
        'Windy': [False]
    })
    prediction = best_model.predict(sample)
    print(f"Sample prediction: {prediction[0]}")
    
    print("\n" + "=" * 50)
    print("Process completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()
