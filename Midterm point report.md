# Using Neural Network to Predict Earthquake Damage Grade

**Midpoint Report**

## Introduction

This project aims to develop a neural network model to predict the damage grade of buildings after an earthquake, based on the dataset provided in the Kaggle competition *Earthquake Prediction ML*. The damage grade is a categorical variable with three classes (1, 2, and 3), representing increasing levels of structural damage. Accurate prediction of this grade can support disaster response and resource allocation efforts.

Our approach leverages deep learning with PyTorch, incorporating techniques such as feature normalization, class weighting for imbalanced data, and cross-validation to improve generalization. This report outlines the methodology, preliminary results, current challenges, and future work.

------

## Methodology

### Data Preprocessing

- **Categorical Encoding**: All non-numeric (object-type) features were converted to numeric codes using pandas' `category.codes`.
- **Feature Scaling**: StandardScaler was applied to normalize input features, improving convergence during training.
- **Label Adjustment**: The original labels (1–3) were shifted to 0–2 to align with PyTorch’s expected class indexing.

### Model Architecture

We experimented with two variants of a fully connected feedforward neural network:

1. Larger Network (Baseline)

   :

   - Hidden layers: 256 → 128 → 64 → 32 → 16
   - ReLU activations, BatchNorm, and Dropout (0.4) after each hidden layer.

2. Smaller Network (Attempted Improvement)

   :

   - Hidden layers: 128 → 64 → 32 → 16
   - Reduced dropout rates (0.3 → 0.2) and reordered BatchNorm before ReLU.

Both output a 3-dimensional logits vector for the three damage grades.

### Training Strategy

- **Loss Function**: Weighted CrossEntropyLoss using class weights computed via `compute_class_weight('balanced', ...)` to address severe class imbalance (class 1 is underrepresented).
- **Optimizer**: Adam with learning rate 0.001.
- **Validation**: 5-fold StratifiedKFold cross-validation to ensure representative splits across damage grades.
- **Early Stopping**: Patience increased from 20 to 40 epochs with a minimum delta of 0.001 to allow longer training before halting.

### Evaluation

- Primary metric: **Accuracy** (as used in the competition leaderboard).
- Secondary: Per-class precision, recall, and F1-score via `classification_report`.

------

## Preliminary Results

### Baseline (Before Improvements)

- **Mean CV Accuracy**: 0.5382
- **Full Train Accuracy**: 0.5465
- **Macro Recall** 0.5097
- **Macro F1 score ** 0.4554
- **Key Observation**: Strong recall for class 1 (damage grade 2), but **class 2 (damage grade 3) had 0% recall**, indicating the model failed to predict the most severe damage.

### After Introducing Class Weighting + Normalization + Smaller Network

- **Mean CV Accuracy dropped to 0.4682**
- **Macro Recall 0.5852**
- **Macro F1 score 0.4431**
- **Class 1 recall collapsed to 9%**, while class 2 improved (88% recall).
- This suggests the smaller network over-corrected for class imbalance, now **under-predicting the majority class**.

### Subsequent Adjustments

- Reverting to the larger network and increasing early stopping patience **did not recover performance** (Mean Acc: ~0.48).
- Removing normalization also **failed to improve results**, indicating scaling is not the primary issue.

> **Key Insight**: The model is highly sensitive to architectural and regularization choices under class imbalance. Simply applying class weights is insufficient without careful tuning of capacity and regularization.

------

## Plan of Work

1. **Hyperparameter Tuning**:
   - Systematically explore learning rates, dropout rates, and layer widths using validation performance.
   - Consider using `torch.optim.lr_scheduler` for adaptive learning rates.
2. **Alternative Architectures**:
   - Test shallower networks or residual connections.
   - Explore ensemble methods (e.g., averaging predictions from all 5 folds).
3. **Advanced Imbalance Handling**:
   - Replace class weighting with **Focal Loss** to focus on hard examples.
   - Experiment with **oversampling** (SMOTE) or **undersampling** on training folds.
4. **Feature Engineering**:
   - Analyze feature importance (e.g., via permutation or SHAP) to reduce noise.
   - Create interaction features (e.g., height × age) based on domain knowledge.
5. **Evaluation Focus**:
   - Shift from accuracy to **macro-F1** during development, as accuracy is misleading under imbalance.
   - Monitor per-class performance closely.

------

## Milestones

| Milestone                                  | Status                         |
| ------------------------------------------ | ------------------------------ |
| Data loading and preprocessing             | ✅ Completed                    |
| Baseline neural network implementation     | ✅ Completed                    |
| Cross-validation framework                 | ✅ Completed                    |
| Class imbalance mitigation (weighting)     | ✅ Implemented (but suboptimal) |
| Submission pipeline                        | ✅ Functional                   |
| Hyperparameter tuning                      | ⏳ Planned                      |
| Advanced loss functions (e.g., Focal Loss) | ⏳ Planned                      |
| Final model ensemble & submission          | ⏳ Future                       |

------

## Problems and Deficiencies Faced

1. **Severe Class Imbalance**:
   - The dataset has a long-tailed distribution: damage grade 2 (class 1) dominates, while grades 1 and 3 are underrepresented.
   - Class weighting improved recall for minority classes but **harmed overall accuracy** and destabilized training.
2. **Model Sensitivity**:
   - Small changes in architecture (e.g., reducing width or dropout) caused large performance swings.
   - The model appears to **overfit to the majority class** unless heavily regularized—yet heavy regularization hurts minority class learning.
3. **Evaluation Misalignment**:
   - Kaggle uses **F1-score**, and now I have not a great idea about it. So I will improve both recall and precision.
4. **Lack of Robust Validation Signal**:
   - Cross-validation scores are unstable across folds (not shown, but observed during debugging), suggesting high variance.
   - Early stopping may halt training prematurely due to noisy validation loss.
5. **Feature Quality**:
   - Many features are categorical with high cardinality (e.g., `ward_id`, `building_id` proxy). Simple label encoding may introduce spurious ordinal relationships.

------

**Next Steps**: Focus on stabilizing training through better loss design (e.g., Focal Loss) and systematic hyperparameter search, while monitoring macro-F1 as a proxy for real-world utility.