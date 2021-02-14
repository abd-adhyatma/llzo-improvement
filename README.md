# The improvement of ionic conductivity in doped Li<sub>7</sub>La<sub>3</sub>Zr<sub>2</sub>O<sub>12</sub> using optimized machine learning with simplistic descriptors
Li<sub>7</sub>La<sub>3</sub>Zr<sub>2</sub>O<sub>12</sub> (LLZO) is hailed as one of the most promising electrolytes for solid-state Li-ion batteries. However, numerous viable doping strategies add further design complexities to the developmental process. Nowadays, the dawn of machine learning methods brings a possible solution to efficiently traverse the vast design space of doped LLZO. This repository describes the tools used to build a machine learning model to classify the ionic conductivity of doped LLZO using features derived from molecular, structural, and electronic descriptors. Overall, this study illustrates the role of powerful data-driven methods with easily obtainable features in accelerating the process of novel solid-state electrolyte design.
## 1. Data processing and EDA
* Unused columns are removed from the DataFrame. 
* EDA uses a heatmap that illustrates the Pearson Correlation Coefficients between features and the target property.
* Scatterplots can also be used to further visualize how features correlate with ionic conductivity.
## 2. LazyClassifier for initial model screening
* Cut and full versions of the dataset are defined. The cut dataset has samples without relative density information removed. The full dataset is imputed using the mean value after the split.
* Run LazyClassifier to initiate the model list.
* For both versions of the dataset, run LazyClassifier over 1000 unique splitting replications by altering the splitting randomness.
* Present top 5 models in a DataFrame to ease viewing.
## 3a. Establishing leave-one-out cross-validation on dataset for model validation
* Define leave-one-out cross-validation as a function to measure model generalizability on the dataset.
## 3b. Nested cross-validation to prove Bayesian optimization's effectiveness
* Nested cross-validation is normally used to measure a certain model's unbiased performance on a dataset.
* Bias prevention can also validate a hyperparameter optimizer's effectiveness.
## 4. Optimizing the model candidates
* Optuna searches for hyperparameters within predefined constraints. The constraints were defined to maximize search thoroughness.
* Optuna runs until a limit of trials has been reached. The limit is set to 100 for LGBM and RFC, and set to 200 for NuSVC.
* __Notice__: Bayesian optimization results might not be identical due to the stochastic nature of Gaussian processes, the underlying mechanism of Bayesian optimization.
## 5. Model interpretation
* Model interpretation uses the feature importance scores of the optimized models.
* If the model does not support built-in feature importance scores, other methods such as permutation importance may be used.
