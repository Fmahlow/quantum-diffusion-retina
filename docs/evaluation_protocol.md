# Evaluation protocol beyond FID and Inception Score

FID and Inception Score remain useful as comparability metrics, but they should not be the main evidence for RetinaMNIST because they use natural-image Inception features. The recommended evaluation stack is:

## Domain classifier utility

Train a RetinaMNIST classifier only on the real training split. Use it as a fixed evaluator.

Report:

- `Real_Target_Accuracy`: sanity check that the evaluator recognizes real images of each class.
- `Fake_Target_Accuracy`: fraction of generated samples classified as the intended class.
- `Fake_Target_Confidence`: mean classifier probability assigned to the intended class.
- `Fake_Prediction_MAE`: ordinal distance between intended class and predicted class.

## Task-specific feature distribution

Use the penultimate layer of the real-trained classifier as a RetinaMNIST feature space.

Report:

- `Feature_FID`: FID computed in RetinaMNIST classifier feature space.
- `Feature_MMD_RBF`: kernel two-sample distance between real and generated feature distributions.

These are more relevant than Inception features because the representation is trained on the target medical-image task.

## Diversity and memorization checks

Report:

- `Fake_To_Real_NN_Distance`: mean nearest-neighbor distance from generated features to real features.
- `Real_Within_NN_Distance`: real within-class nearest-neighbor distance.
- `Fake_Within_NN_Distance`: generated within-class nearest-neighbor distance.
- `Fake_Diversity_Ratio`: generated within-class distance divided by real within-class distance.

Interpretation:

- Very low fake-to-real distance can indicate memorization.
- Very low fake within-class distance or diversity ratio far below 1 can indicate mode collapse.
- Very high fake within-class distance can indicate unrealistic dispersion or class leakage.

## Downstream augmentation

The strongest scientific test is still downstream utility:

1. Train classifier on real train only.
2. Train classifier on real train plus synthetic train.
3. Train classifier on synthetic train only.
4. Evaluate all on the untouched official test split.

Report accuracy, balanced accuracy, macro-F1, ordinal MAE, and confidence intervals across at least 3 seeds.

