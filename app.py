import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Setup paths
CURRENT_DIR = Path(__file__).parent
MODEL_DIR = CURRENT_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

# Load preprocessing examples to verify our preprocessing steps
PREPROCESS_FILE = CURRENT_DIR / 'output/preprocessing_examples.json'

class TextPreprocessor:
    """Text preprocessing class that can be used in sklearn Pipeline"""
    
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Complete preprocessing pipeline for a single text"""
        # Noise removal
        text = re.sub(r'http\S+|www.\S+', '', str(text))
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalization
        text = text.lower().strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Stop-word removal and lemmatization
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        
        return ' '.join(tokens)
    
    def fit(self, X, y=None):
        """Required for sklearn Pipeline"""
        return self
    
    def transform(self, X):
        """Transform a series of texts"""
        return [self.preprocess_text(text) for text in X]

def load_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv(CURRENT_DIR / 'sample1.csv')
        return df
    except FileNotFoundError:
        print("Error: Could not find the data file")
        return None

def create_model_pipeline():
    """Create the complete model pipeline
    The pipeline consists of three main components:
    a) Text Preprocessor
    b) TF-IDF Vectorizer
    c) Random Forest Classifier
    """
    return Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('vectorizer', TfidfVectorizer(max_features=5000)), # max_features: Maximum number of features (words) in the vocabulary
        ('classifier', RandomForestClassifier(
            n_estimators=100, # Number of trees in the Forest
            max_depth=None, # Maximum depth of the tree
            min_samples_split=2, # Minimum number of samples required to split an internal node
            min_samples_leaf=1, # Minimum number of samples required to be at a leaf node
            random_state=42 # Random seed for reproducibility
        ))
    ])

def train_and_evaluate_model(df):
    """Train and evaluate the model"""
    # Prepare data
    X = df['issue_body'].fillna('')  # Handle missing values
    y = df['issue_label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the pipeline
    pipeline = create_model_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\nModel Evaluation:")
    print("----------------")
    print(classification_report(y_test, y_pred))
    
    return pipeline

def save_model(pipeline, filename='model_pipeline.joblib'):
    """Save the trained model pipeline"""
    model_path = MODEL_DIR / filename
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")

def verify_preprocessing():
    """Verify preprocessing against saved examples"""
    try:
        with open(PREPROCESS_FILE, 'r') as f:
            examples = json.load(f)
        
        preprocessor = TextPreprocessor()
        
        print("Verifying preprocessing pipeline...")
        for i, example in enumerate(examples):
            processed = preprocessor.preprocess_text(example['original'])
            print(f"\nExample {i+1}:")
            print(f"Original (first 100 chars): {example['original'][:100]}")
            print(f"Processed (first 100 chars): {processed[:100]}")
            
    except FileNotFoundError:
        print(f"Warning: Could not find preprocessing examples at {PREPROCESS_FILE}")

def main():
    # Verify preprocessing pipeline
    verify_preprocessing()
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Train and evaluate model
    pipeline = train_and_evaluate_model(df)
    
    # Save model
    save_model(pipeline)

if __name__ == "__main__":
    main()
