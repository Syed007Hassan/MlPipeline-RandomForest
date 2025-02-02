# Random Forest Model Pipeline

## 1. Model Pipeline Components

The pipeline consists of three main components:

### a) Text Preprocessor
- Custom transformer class compatible with sklearn
- Implements the preprocessing steps from Exercise 02:
  1. Noise removal (URLs, code blocks, special characters)
  2. Text normalization (lowercase, whitespace)
  3. Tokenization
  4. Stop-word removal
  5. Lemmatization
- Maintains preprocessing consistency between training and prediction

### b) TF-IDF Vectorizer
- Converts preprocessed text into numerical features
- Parameters:
  - max_features=5000 (limits vocabulary size)
- Benefits:
  - Captures word importance in documents
  - Handles varying document lengths
  - Reduces impact of common words

### c) Random Forest Classifier
- Final classification model
- Parameters:
  - n_estimators=100 (number of trees)
  - max_depth=None (allows full tree growth)
  - min_samples_split=2 (minimum samples for split)
  - random_state=42 (reproducibility)
- Benefits:
  - Handles high-dimensional data well
  - Resistant to overfitting
  - Provides feature importance

## 2. Model Modularization for Integration

### Modularization Approach:

1. **Separate Preprocessing Class**
   - TextPreprocessor class is self-contained
   - Can be imported and used independently
   - Maintains consistent preprocessing across applications

2. **Serialized Model Pipeline**
   - Complete pipeline saved using joblib
   - Includes preprocessor, vectorizer, and classifier
   - Can be loaded and used in Flask application

3. **Verification System**
   - Preprocessing examples stored in JSON
   - Allows verification of preprocessing consistency
   - Useful for testing and debugging

4. **Future Integration**
   - Flask app can import TextPreprocessor class
   - Load saved pipeline using joblib
   - Use for real-time predictions

## 3. Random Forest and Concept Drift

### Drawbacks of Random Forest

1. **Static Model Structure**
   - Trees are fixed after training
   - Cannot adapt to new patterns without retraining
   - May become outdated as issue patterns change

2. **Memory Intensive**
   - Stores many decision trees
   - Difficult to update incrementally
   - Requires full retraining for updates

3. **Feature Space Limitations**
   - Fixed vocabulary from training data
   - Cannot handle new terms or patterns
   - May miss emerging topics

### Alternative Model: Online Learning with SGDClassifier

#### Benefits for Concept Drift:

1. **Incremental Learning**
   - Can update with new data points
   - Adapts to changing patterns
   - Supports partial_fit method

2. **Memory Efficient**
   - Doesn't store training data
   - Lighter memory footprint
   - Easier to deploy and update

3. **Adaptive Learning Rate**
   - Adjusts to data changes
   - Balances old and new knowledge
   - Better handles concept drift

4. **Implementation Strategy**
   - Regular model updates with new data
   - Monitoring of prediction confidence
   - Sliding window for recent patterns


After the execution of app.py:

```markdown
### 1. Preprocessing Verification
First, the program verified the preprocessing pipeline using the examples saved from exercise02:

```python
Example 1: GitHub issue about Entities and fields
- Original: Technical issue about '__tileSrcRect' fields
- Processed: Cleaned, tokenized, and lemmatized version without URLs and special characters

Example 2: Bug report about blog link
- Original: Markdown-formatted bug report about updating website links
- Processed: Clean text with key terms preserved but formatting removed

Example 3: Technical discussion about expressions
- Original: Code example with markdown formatting
- Processed: Plain text with code-related terms preserved
```

### 2. Model Training and Evaluation
The program then trained and evaluated the Random Forest model:

```
Model Evaluation Results:
- Bug issues:
  - Precision: 0.76 (76% of predicted bugs were actual bugs)
  - Recall: 0.79 (79% of actual bugs were correctly identified)
  - F1-score: 0.77 (harmonic mean of precision and recall)

- Enhancement issues:
  - Precision: 0.70
  - Recall: 0.80
  - F1-score: 0.75

- Question issues:
  - Precision: 0.61
  - Recall: 0.09 (very low - model struggles with questions)
  - F1-score: 0.16 (poor performance on questions)

Overall:
- Accuracy: 0.73 (73% of all predictions were correct)
- The model performs well on bugs and enhancements
- Struggles with questions (likely due to class imbalance)
```

### 3. Model Serialization
```python
model_pipeline.joblib
```
This file contains:
- The complete trained pipeline including:
  1. TextPreprocessor
  2. TF-IDF Vectorizer
  3. Random Forest Classifier
- Can be loaded later using:
  ```python
  loaded_model = joblib.load('model_pipeline.joblib')
  prediction = loaded_model.predict(['new issue text'])
  ```

### Key Observations:
1. Preprocessing works consistently across different types of issues
2. Model performs well on majority classes (bugs and enhancements)
3. Poor performance on minority class (questions) suggests need for:
   - Class balancing techniques
   - More training data for questions
   - Possibly different model architecture for better minority class handling

The saved model can now be used in a Flask application for real-time predictions on new GitHub issues.
