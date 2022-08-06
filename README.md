# Rock-vs-Mine-Prediction
# numpy
NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices. NumPy was created in 2005 by Travis Oliphant. It is an open source project and you can use it freely. NumPy stands for Numerical Python.
# pandas
Pandas is a Python library that is used for faster data analysis, data cleaning and data pre-processing. Pandas is built on top of numpy. So, numpy gets some superpower with pandas. You might have heard about data-frames, which is a common term in machine learning
# sklearn.model_selection
The package sklearn.model_selection offers a lot of functionalities related to model selection and validation, including the following: Cross-validation; Learning curves; Hyperparameter tuning; Cross-validation is a set of techniques that combine the measures of prediction performance to get more accurate model estimations.
# sklearn.model_selection import train_test_split
sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None) Split arrays or matrices into random train and test subsets.

Quick utility that wraps input validation and next(ShuffleSplit().split(X, y)) and application to input data into a single call for splitting (and optionally subsampling) data in a oneliner.

# sklearn.linear_model import LogisticRegression
sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’, and uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’.

# sklearn.metrics import accuracy_score
sklearn.metrics.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None) Accuracy classification score.

In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true
