# Project Report

## 1. Cover Page
**Title:** AI-Powered Student Success & Academic Risk Predictor  
**Technology:** Python, Machine Learning, Streamlit, scikit-learn

## 2. Introduction
Educational institutions often have data related to attendance, scores, study behavior,
and student engagement. Machine learning can use these factors to identify patterns
associated with academic risk and provide early decision support.

## 3. Problem Statement
The project predicts whether a student may be academically at risk using measurable
academic and lifestyle-related inputs.

## 4. Functional Requirements
1. Load or generate student data.
2. Validate required input columns.
3. Preprocess numerical and categorical data.
4. Train and compare two ML classifiers.
5. Evaluate the selected model.
6. Accept individual student information.
7. Return risk class and probability.
8. Display aggregate analytics.

## 5. Non-Functional Requirements
- **Performance:** Training should complete quickly for small/medium educational datasets.
- **Usability:** Simple web UI with clear forms and outputs.
- **Reliability:** Input validation and reproducible random state.
- **Maintainability:** Modular package structure with separate responsibilities.
- **Error Handling:** Invalid datasets raise meaningful errors.
- **Security/Privacy:** Avoid storing personally identifying student data in the demo.

## 6. System Architecture
See `docs/architecture.md`.

## 7. Design Diagrams
Architecture, workflow, use case, component, and sequence diagrams are provided as
Mermaid diagrams in `docs/architecture.md`.

## 8. Design Decisions & Rationale
A pipeline-based architecture prevents preprocessing mismatch between training and
prediction. Logistic Regression is used as an interpretable linear baseline. Random
Forest is used because it can model non-linear relationships and interactions. F1-score
is used for selection because it balances precision and recall for the at-risk class.

## 9. Implementation Details
The project is implemented as independent modules for configuration, data generation,
loading/validation, preprocessing, training, evaluation, and prediction. The Streamlit
app integrates these modules into an interactive interface.

## 10. Dataset Description
The included demo dataset is synthetically generated and contains:
- attendance percentage
- daily study hours
- sleep hours
- assignment score
- previous academic score
- extracurricular hours
- internet access
- binary academic-risk target

The synthetic target is generated from a noisy combination of these factors so that the
project can be demonstrated without exposing real student records.

## 11. Model Selection Rationale
- **Logistic Regression:** transparent and effective baseline for binary classification.
- **Random Forest:** handles nonlinear interactions without requiring manual feature rules.
The project compares both and chooses the model with the better validation F1-score.

## 12. Evaluation Methodology
A stratified 75/25 train-test split is used. Evaluation metrics are accuracy, precision,
recall, F1-score, and confusion matrix. Stratification preserves the target-class ratio.

## 13. Testing Approach
Pytest tests verify generated data validity, model training, metric range, and prediction
output structure.

## 14. Challenges Faced
- Handling mixed numerical and categorical features
- Avoiding preprocessing inconsistency between training and inference
- Selecting a metric suitable for an imbalanced-risk scenario
- Designing a reusable, modular project structure

## 15. Learnings & Key Takeaways
- Building reusable scikit-learn pipelines
- Comparing classification algorithms
- Evaluating binary classifiers using multiple metrics
- Structuring an ML application into independent modules
- Building a simple ML interface with Streamlit

## 16. Future Enhancements
- Train on anonymized real-world data
- Add cross-validation and hyperparameter tuning
- Explain predictions using SHAP or feature importance
- Add authentication and role-based dashboards
- Store prediction history in a database
- Add drift monitoring

## 17. Ethical Considerations
Predictions should not be used as the sole basis for high-stakes decisions about
students. Real deployments should check data quality, bias, privacy, and fairness.

## 18. References
- Python documentation
- scikit-learn documentation
- Streamlit documentation
- Pandas documentation
