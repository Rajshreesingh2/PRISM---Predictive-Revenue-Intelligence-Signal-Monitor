"""Machine learning modeling module for PRISM"""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import pickle
from pathlib import Path


logger = logging.getLogger(__name__)


class MLModeler:
    """Train and evaluate ML models"""
    
    def __init__(self, config):
        """Initialize ML modeler"""
        self.config = config
        self.models = {}
        self.results = {}
    
    def train_models(self, X, y):
        """Train multiple models"""
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"  Train set: {X_train.shape[0]} samples")
        logger.info(f"  Test set: {X_test.shape[0]} samples")
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        }
        
        best_auc = 0
        best_model = None
        best_name = None
        
        for name, model in models.items():
            logger.info(f"  Training {name}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            self.results[name] = {
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
            
            logger.info(f"    ROC-AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            
            self.models[name] = model
            
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_name = name
        
        logger.info(f"✅ Best model: {best_name} (AUC: {best_auc:.4f})")
        
        # Save best model
        model_path = Path('models') / 'best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        logger.info(f"  Model saved to {model_path}")
        
        return best_model, X_test, y_test
    
    def generate_predictions(self, model, X):
        """Generate churn predictions and risk scores"""
        logger.info("Generating predictions...")
        
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Risk tiers
        risk_tiers = pd.cut(y_pred_proba, bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])
        
        logger.info(f"✅ Generated {len(y_pred)} predictions")
        logger.info(f"  High Risk: {(risk_tiers == 'High').sum()} customers")
        logger.info(f"  Medium Risk: {(risk_tiers == 'Medium').sum()} customers")
        logger.info(f"  Low Risk: {(risk_tiers == 'Low').sum()} customers")
        
        return y_pred, y_pred_proba, risk_tiers
