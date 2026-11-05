from sklearn.model_selection import GridSearchCV, StratifiedKFold


def cross_validation(model, param_grid, Xtr, ytr, seed):
    
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed) # shuffle data which gets split into 4 folds
    
    search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=cv, refit=True, verbose=True) # loop through hyperparameters
    
    search.fit(Xtr, ytr) # fit model to training data
    
    return search.best_params_, search.best_score_, search.best_estimator_
    