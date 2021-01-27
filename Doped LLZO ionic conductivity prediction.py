#!/usr/bin/env python
# coding: utf-8

# # Preparation and initialization

# In[ ]:


pip install scikit-optimize


# In[ ]:


pip install xgboost


# In[ ]:


pip install lightgbm


# In[ ]:


# Importing all necessary tools
# Importing necessary libraries for LazyClassifier
from lazypredict.Supervised import LazyClassifier
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skopt import gp_minimize
from sklearn.svm import NuSVC
import optuna

# Importing table and visualization libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing dataset splitting and imputing libraries
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut


# # 1. Dataset processing and EDA

# In[ ]:


# Reading .csv file for complete, cleaned dataset
df_path='https://raw.githubusercontent.com/abd-adhyatma/llzo-improvement/main/llzo_dataset_clean.csv'


# In[ ]:


df_raw=pd.read_csv(df_path)
df = df_raw[0:176]
df.head(176)


# Drop unnecessary columns for prediction

# In[ ]:


# Dropping unnecessary columns for prediction
df = df.drop(['conductivity','log_cond','li_dopant','la_dopant','zr_dopant','source'], axis=1)
df.head()


# In[ ]:


# Finding correlation between features
corr = df.corr()
plt.subplots(figsize=(10,10))
ax = sns.heatmap(
    corr,
    vmin = -1, vmax = 1, center = 0,
    linewidths = .5, annot = True,
    cmap = sns.diverging_palette(20, 220, n = 200),
    square = True)


# In[ ]:


# Making scatterplots to illustrate feature data distribution in respect to conductivity
for col in df.columns.values:
    if col == 'good_cond':
        continue
    plt.figure(figsize=(20,5))
    sns.scatterplot(df.dropna()[col], df.dropna()['good_cond'])
    plt.title(col)
    plt.show()


# # 2. Lazy classifier for initial model screening

# Drop the features with nan

# In[ ]:


# Defining X (features) and y (target property) for cut dataset
X = df.dropna().drop('good_cond', axis=1)
y = df.dropna()['good_cond']
print(X)

X_full = df.drop('good_cond', axis=1)
y_full = df['good_cond']


# Perform lazy classifier once to get the list of all models:

# In[ ]:


# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Using LazyClassifier for cut dataset
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
modellist = list(models.index.values) # Get the list of the methods' names
models


# Perform replications with 75% data as the training set. The R2 scores of the model are recorded. The models are also ranked according to the R2 scores. These scores and rank are then averaged.

# In[ ]:


Nrep = 1000 # Number of replications, the higher the better
r2score = np.zeros((len(modellist),Nrep)) # Initialize the r2score
position = np.zeros((len(modellist),Nrep)) # Initialize the position (rank)
for LOOP in range(0,Nrep):

    #Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = LOOP)

    #Using LazyRegressor for cut dataset
    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    modelstemp, predictionstemp = clf.fit(X_train, X_test, y_train, y_test)
 
    modellisttemp = list(modelstemp.index.values)
    
    for i, mdl in zip(range(0,len(modellist)),modellist):
        search_pos = int(modellisttemp.index(models.index.values[i]))
        r2score[i,LOOP] = modelstemp.iloc[:,0][search_pos]
        position[i,LOOP] = search_pos
    


# In[ ]:


print('------------------ CUT DATASET ------------------')
idx = np.argmax(np.mean(r2score, axis=1))
print('The best model according to the mean acc. score is ',modellist[idx],'with score',max(np.mean(r2score,axis=1)))
idx = np.argmax(np.median(r2score,axis=1))
print('The best model according to the median acc. score is ',modellist[idx],'with score',max(np.median(r2score,axis=1)))
idx = np.argmin(np.mean(position,axis=1))
print('The best model according to the mean ranking is ',modellist[idx],'with score',min(np.mean(position,axis=1)))
idx = np.argmin(np.median(position,axis=1))
print('The best model according to the median ranking is ',modellist[idx],'with score',min(np.median(position,axis=1)))


# In[ ]:


modellist_df = pd.DataFrame(modellist).rename(columns = {0:'Model'})
mean_df = pd.DataFrame(np.mean(r2score, axis = 1)).rename(columns = {0:'Accuracy mean'})
med_df = pd.DataFrame(np.median(r2score, axis = 1)).rename(columns = {0:'Accuracy median'})
rankmean_df = pd.DataFrame(np.mean(position, axis = 1)).rename(columns = {0:'Rank mean'})
rankmed_df = pd.DataFrame(np.median(position, axis = 1)).rename(columns = {0:'Rank median'})
models_df = pd.concat([modellist_df, mean_df, rankmean_df, med_df, rankmed_df], axis = 1)
models_df.sort_values(['Accuracy mean', 'Accuracy median'], ascending = False).head(5)


# ### The three model candidates are LGBM classifier, random forest, and NuSVC

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
Nrep = 200 # Number of replications, the higher the better
r2score_full = np.zeros((len(modellist),Nrep)) # Initialize the r2score
position_full = np.zeros((len(modellist),Nrep)) # Initialize the position (rank)
for LOOP in range(0,Nrep):

    #Splitting
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.25, random_state = LOOP)

    #Imputing X_train and X_test
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_full))
    X_train_imputed.columns = X_train_full.columns
    X_train_imputed.index = X_train_full.index

    X_test_imputed = pd.DataFrame(imputer.fit_transform(X_test_full))
    X_test_imputed.columns = X_test_full.columns
    X_test_imputed.index = X_test_full.index
    
    #Using LazyClassifier for cut dataset
    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    modelstemp,predictionstemp = clf.fit(X_train_imputed, X_test_imputed, y_train_full, y_test_full)
 
    modellisttemp = list(modelstemp.index.values)
    
    for i, mdl in zip(range(0,len(modellist)),modellist):
        search_pos = int(modellisttemp.index(models.index.values[i]))
        r2score_full[i,LOOP] = modelstemp.iloc[:,0][search_pos]
        position_full[i,LOOP] = search_pos
    


# In[ ]:


print('------------------ IMPUTED DATASET ------------------')
idx_full = np.argmax(np.mean(r2score_full,axis=1))
print('The best model according to the mean acc. score is ',modellist[idx_full],'with score',max(np.mean(r2score_full,axis=1)))
idx_full = np.argmax(np.median(r2score_full,axis=1))
print('The best model according to the median acc. score is ',modellist[idx_full],'with score',max(np.median(r2score_full,axis=1)))
idx_full = np.argmin(np.mean(position_full,axis=1))
print('The best model according to the mean ranking is ',modellist[idx_full],'with score',min(np.mean(position_full,axis=1)))
idx_full = np.argmin(np.median(position_full,axis=1))
print('The best model according to the median ranking is ',modellist[idx_full],'with score',min(np.median(position_full,axis=1)))


# #### Using cut dataset leads to better results, while LGBMClassifier consistently scores the best, score-wise and ranking-wise

# # 3a. Establishing LOOCV on dataset for model validation

# Start by leave-one-out-cross validation first

# In[ ]:


# Leave one out cross-validation (un-optimized)
cvpred = np.zeros([len(X)]) #Creating array of zeros as big as the length of X
Xnp = X.to_numpy() #Converts feature set to np array
ynp = y.to_numpy() #Converts target property to numpy
for i in range(0,len(X)):
    xpred = Xnp[i,:].reshape(1,-1) #Define X_val
    XLOO = np.delete(Xnp,i,axis=0) #Define X_train
    yLOO = np.delete(ynp,i).reshape(-1,1) #Define y_train
    modelLOO = LGBMClassifier() #Define model
    modelLOO.fit(XLOO, yLOO) #Fitting model to training set
    cvpred[i] = modelLOO.predict(xpred) #Adding predict score to array of zeros
LOOCVscore = np.sum(cvpred == ynp)/len(X)

print('LOOCV error of LGBMClassifier is ', LOOCVscore)


# In[ ]:


# Defining LOOCV function, takes input of X, y, and model and returns the mean score
def LOO_cross_val (X, y, model):
    cvpred = np.zeros([len(X)]) #Creating array of zeros as big as the length of X
    Xnp = X.to_numpy() #Converts feature set to np array
    ynp = y.to_numpy() #Converts target property to numpy
    for i in range(0,len(X)):
        xpred = Xnp[i,:].reshape(1,-1) #Define X_val
        XLOO = np.delete(Xnp,i,axis=0) #Define X_train
        yLOO = np.delete(ynp,i).reshape(-1,1) #Define y_train
        modelLOO = model #Define model
        modelLOO.fit(XLOO, yLOO) #Fitting model to training set
        cvpred[i] = modelLOO.predict(xpred) #Adding predict score to array of zeros
    LOOCVscore = np.sum(cvpred == ynp)/len(X)
    return LOOCVscore


# In[ ]:


from sklearn.ensemble import BaggingClassifier
candidates = {'NuSVC' : NuSVC(), 'LGBM' : LGBMClassifier(), 'RF': RandomForestClassifier()}
candidates
for candidate, model in candidates.items():
    print('The candidate {} scored {} in a baseline LOOCV'.format(candidate, LOO_cross_val(X, y, model)))


# # 3b. Nested cross validation to prove Bayesian optimization's effectiveness

# Nested cross validation optimizing towards LOOCV score on the training set, measuring the score improvement on the test set.

# In[ ]:


from sklearn.model_selection import StratifiedKFold
# Split train data to 5 outer folds and 4 inner folds
skfold_outer = StratifiedKFold(n_splits=5, shuffle = True, random_state = None)


# In[ ]:


# Executing nested cross-validation for DecisionTree
i = 1
folds = []
for train_index_outer, test_index_outer in skfold_outer.split(X, y):
    fold_data = {}
    
    # Splitting the outer folds
    print("Outer Fold %s" % i)
    data_X_train = X.iloc[train_index_outer]
    data_y_train = y.iloc[train_index_outer]
    data_X_test = X.iloc[test_index_outer]
    data_y_test = y.iloc[test_index_outer]
    
    # Testing on test_index_outer and LOOCV score on train_index_outer using one of the candidates, RFClassifier
    clf_base = LGBMClassifier(random_state = None)
    fold_data['outer_fold'] = i
    clf_base_fit = clf_base.fit(data_X_train, data_y_train)
    clf_base_test = clf_base_fit.score(data_X_test, data_y_test)
    fold_data['Base test score'] = clf_base_test
    fold_data['Base LOOCV score'] = LOO_cross_val(data_X_train, data_y_train, clf_base)
    
    
    # Optimization process in inner fold, claiming best parameters
    def objective(trial):
        num_leaves_lgbm = trial.suggest_int('num_leaves', 10, 50)
        learning_rate_lgbm = trial.suggest_loguniform('learning_rate', 1e-4, 1e2)
        n_estimators_lgbm = trial.suggest_int('n_estimators', 50, 500)
        subsample_for_bin_lgbm = trial.suggest_int('subsample_for_bin', 1e5, 3e5)
    
        classifier = LGBMClassifier(random_state = None,
                                num_leaves = num_leaves_lgbm,
                                subsample_for_bin = subsample_for_bin_lgbm,
                                n_estimators = n_estimators_lgbm,
                                learning_rate = learning_rate_lgbm)
                                
        score = LOO_cross_val(data_X_train, data_y_train, classifier)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, n_jobs = -1, gc_after_trial=True)
    print('Best hyperparameters:\n{}'.format(study.best_params))
    print('Best LOOCV score:\n{}'.format(study.best_value))
        
    # Entering optimization results
    fold_data['best_param'] = study.best_params
    fold_data['Optimized LOOCV score'] = study.best_value
    
    # Testing on test data in outer loop
    model = LGBMClassifier(random_state=None,                           num_leaves=fold_data['best_param']['num_leaves'],                           learning_rate=fold_data['best_param']['learning_rate'],                           n_estimators=fold_data['best_param']['n_estimators'],                           subsample_for_bin=fold_data['best_param']['subsample_for_bin'])

    fit = model.fit(data_X_train, data_y_train)
    test_score = fit.score(data_X_test, data_y_test)
    fold_data['Optimized test score'] = test_score
    print("Test Outer Score: %s" % test_score, '\n')
    
    folds.append(fold_data)
    
    i+=1


# In[ ]:


df_outer_fold_result = pd.DataFrame({"outer_fold":[],                                     "num_leaves":[],                                     "learning_rate":[],                                     "n_estimators":[],                                     "subsample_for_bin": [],                                     "Base LOOCV score":[],                                     "Optimized LOOCV score":[],                                     "Base test score":[],                                     "Optimized test score": []})
for fold in folds:
    res = pd.concat([pd.DataFrame(fold['best_param'], index=[0]),                    pd.DataFrame(fold, index=[0]).drop('best_param', axis=1)], axis=1)
    df_outer_fold_result = pd.concat([df_outer_fold_result, res])

df_outer_fold_result


# Optimizing towards the test set score, measuring the improvement of the LOOCV score.

# In[ ]:


# Executing nested cross-validation for DecisionTree
i = 1
folds = []
for train_index_outer, test_index_outer in skfold_outer.split(X, y):
    fold_data = {}
    
    # Splitting the outer folds
    print("Outer Fold %s" % i)
    data_X_train = X.iloc[train_index_outer]
    data_y_train = y.iloc[train_index_outer]
    data_X_test = X.iloc[test_index_outer]
    data_y_test = y.iloc[test_index_outer]
    
    # Testing on test_index_outer and LOOCV score on train_index_outer using one of the candidates, RFClassifier
    clf_base = LGBMClassifier(random_state = None)
    fold_data['outer_fold'] = i
    clf_base_fit = clf_base.fit(data_X_train, data_y_train)
    clf_base_test = clf_base_fit.score(data_X_test, data_y_test)
    fold_data['Base test score'] = clf_base_test
    fold_data['Base LOOCV score'] = LOO_cross_val(data_X_train, data_y_train, clf_base)
    
    
    # Optimization process in inner fold, claiming best parameters
    def objective(trial):
        num_leaves_lgbm = trial.suggest_int('num_leaves', 10, 50)
        learning_rate_lgbm = trial.suggest_loguniform('learning_rate', 1e-4, 1e2)
        n_estimators_lgbm = trial.suggest_int('n_estimators', 50, 500)
        subsample_for_bin_lgbm = trial.suggest_int('subsample_for_bin', 1e5, 3e5)
    
        classifier = LGBMClassifier(random_state = None,
                                num_leaves = num_leaves_lgbm,
                                subsample_for_bin = subsample_for_bin_lgbm,
                                n_estimators = n_estimators_lgbm,
                                learning_rate = learning_rate_lgbm)
                                
        fit = classifier.fit(data_X_train, data_y_train)
        score = fit.score(data_X_test, data_y_test)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, n_jobs = -1, gc_after_trial=True)
    print('Best hyperparameters:\n{}'.format(study.best_params))
    print('Best test score:\n{}'.format(study.best_value))
        
    # Entering optimization results
    fold_data['best_param'] = study.best_params
    fold_data['Optimized test score'] = study.best_value
    
    # Testing on test data in outer loop
    model = LGBMClassifier(random_state=None,                           num_leaves=fold_data['best_param']['num_leaves'],                           learning_rate=fold_data['best_param']['learning_rate'],                           n_estimators=fold_data['best_param']['n_estimators'],                           subsample_for_bin=fold_data['best_param']['subsample_for_bin'])

    test_score = LOO_cross_val(data_X_train, data_y_train, model)
    fold_data['Optimized LOOCV score'] = test_score
    print("Optimized LOOCV score: %s" % test_score, '\n')
    
    folds.append(fold_data)
    
    i+=1


# In[ ]:


df_outer_fold_result = pd.DataFrame({"outer_fold":[],                                     "num_leaves":[],                                     "learning_rate":[],                                     "n_estimators":[],                                     "subsample_for_bin": [],                                     "Base test score":[],                                     "Optimized test score":[],                                     "Base LOOCV score":[],                                     "Optimized LOOCV score": []})
for fold in folds:
    res = pd.concat([pd.DataFrame(fold['best_param'], index=[0]),                    pd.DataFrame(fold, index=[0]).drop('best_param', axis=1)], axis=1)
    df_outer_fold_result = pd.concat([df_outer_fold_result, res])

df_outer_fold_result


# # 4. Optimizing the model candidates
# The optimization uses a Bayesian optimization approach inside a leave-one-out cross-validation.

# In[ ]:


def objective_lgbm(trial):
    num_leaves_lgbm = trial.suggest_int('num_leaves', 10, 50)
    learning_rate_lgbm = trial.suggest_loguniform('learning_rate', 1e-4, 1e2)
    n_estimators_lgbm = trial.suggest_int('n_estimators', 50, 500)
    subsample_for_bin_lgbm = trial.suggest_int('subsample_for_bin', 1e5, 3e5)
    
    classifier = LGBMClassifier(random_state = None,
                                num_leaves = num_leaves_lgbm,
                                subsample_for_bin = subsample_for_bin_lgbm,
                                n_estimators = n_estimators_lgbm,
                                learning_rate = learning_rate_lgbm)
                                
    score = LOO_cross_val(X, y, classifier)
    return score

study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm, n_trials=100, n_jobs = -1)
print(study_lgbm.best_trial, '\n')
print('The best hyperparameters for LGBM:\n{}\n'.format(study_lgbm.best_params))
print('The best leave-one-out cross-validation score for LGBM:\n{}\n'.format(study_lgbm.best_value))


# In[ ]:


def objective_rf(trial):
    n_estimators_rf = trial.suggest_int('n_estimators', 50, 500)
    criterion_rf = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    bootstrap_rf = trial.suggest_categorical('bootstrap', [True, False])
    msl_rf = trial.suggest_int('min_samples_leaf', 1, 10)
    mss_rf = trial.suggest_int('min_samples_split', 2, 12)
    
    classifier = RandomForestClassifier(random_state = 0,
                                       n_estimators = n_estimators_rf,
                                       criterion = criterion_rf,
                                       bootstrap = bootstrap_rf,
                                       min_samples_leaf = msl_rf,
                                       min_samples_split = mss_rf)
    
    score = LOO_cross_val(X, y, classifier)
    return score

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=100, n_jobs = -1)
print(study_rf.best_trial, '\n')
print('The best hyperparameters for RF:\n{}\n'.format(study_rf.best_params))
print('The best leave-one-out cross-validation score for RF:\n{}\n'.format(study_rf.best_value))


# In[ ]:


def objective_nusvc(trial):
    nu_nusvc = trial.suggest_uniform('nu', 0.4, 0.6)
    tol_nusvc = trial.suggest_loguniform('tol', 1e-5, 1e-1)
    cache_size_nusvc = trial.suggest_uniform('cache_size', 100.0, 300.0)
    kernel_type = trial.suggest_categorical('kernel_type', ['linear', 'poly', 'rbf', 'sigmoid'])
    if kernel_type == 'linear':
        classifier = NuSVC(random_state = 0,
                          nu = nu_nusvc,
                          tol = tol_nusvc,
                          cache_size = cache_size_nusvc,
                          kernel = 'linear')
    if kernel_type == 'poly':
        degree_nusvc = trial.suggest_int('degree', 2, 5)
        gamma_nusvc = trial.suggest_categorical('gamma', ['scale', 'auto'])
        coef0_nusvc = trial.suggest_uniform('coef0', 0.0, 0.5)
        classifier = NuSVC(random_state = 0,
                          nu = nu_nusvc,
                          tol = tol_nusvc,
                          cache_size = cache_size_nusvc,
                          kernel = 'poly',
                          degree = degree_nusvc,
                          gamma = gamma_nusvc,
                          coef0 = coef0_nusvc)
    if kernel_type == 'sigmoid':
        gamma_nusvc = trial.suggest_categorical('gamma', ['scale', 'auto'])
        coef0_nusvc = trial.suggest_uniform('coef0', 0.0, 0.5)
        classifier = NuSVC(random_state = 0,
                          nu = nu_nusvc,
                          tol = tol_nusvc,
                          cache_size = cache_size_nusvc,
                          kernel = 'sigmoid',
                          gamma = gamma_nusvc,
                          coef0 = coef0_nusvc)
    else:
        gamma_nusvc = trial.suggest_categorical('gamma', ['scale', 'auto'])
        classifier = NuSVC(random_state = 0,
                          nu = nu_nusvc,
                          tol = tol_nusvc,
                          cache_size = cache_size_nusvc,
                          kernel = 'rbf',
                          gamma = gamma_nusvc)
        
    score = LOO_cross_val(X, y, classifier)
    return score

study_nusvc = optuna.create_study(direction="maximize")
study_nusvc.optimize(objective_nusvc, n_trials=200, n_jobs = -1)
print(study_nusvc.best_trial, '\n')
print('The best hyperparameters for NuSVC:\n{}\n'.format(study_nusvc.best_params))
print('The best leave-one-out cross-validation score for NuSVC:\n{}\n'.format(study_nusvc.best_value))


# # 5. Model interpretation

# In[ ]:


#Define optimal models
rf_opt = RandomForestClassifier(random_state = 0,
                               n_estimators = 214,
                               criterion = 'entropy',
                               bootstrap = True,
                               min_samples_leaf = 9,
                               min_samples_split = 9)
lgbm_opt = LGBMClassifier(random_state = 0,
                         num_leaves = 38,
                         learning_rate = 25.75602656,
                         n_estimators = 331,
                         subsample_for_bin = 122386)


# In[ ]:


#Showing GradientBoostingRegressor's built-in feature importance
rf_fit = rf_opt.fit(X, y)
feat_importance = rf_fit.feature_importances_
sorted_idx = np.argsort(feat_importance)
pos = np.arange(sorted_idx.shape[0]) + 1
fig = plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
plt.barh(pos, feat_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.title('Feature Importance')
sorted_feat = feat_importance[sorted_idx]
sort_fe = []
for feat in sorted_feat:
    fe = round(feat, 4)
    sort_fe.append(fe)
for index, value in enumerate(sort_fe):
    plt.text(value, index, str(value))


# In[ ]:


from lightgbm import plot_importance
lgbm_fit = lgbm_opt.fit(X, y)
plot_importance(lgbm_fit)


# In[ ]:


lgbm_df = study_lgbm.trials_dataframe()
rf_df = study_rf.trials_dataframe()
nu_df = study_nusvc.trials_dataframe()

