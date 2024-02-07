import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('../input/csi-machine-learning-competition/star_train.csv')
train_data = train_data.dropna(subset=['class'])

label_encoder = LabelEncoder()
train_data['class'] = label_encoder.fit_transform(train_data['class'])

train_data['mean_magnitude'] = train_data[['u', 'g', 'r', 'i', 'z']].mean(axis=1)
train_data['color_index'] = train_data['u'] - train_data['r']
train_data['u-g'] = train_data['u'] - train_data['g']
train_data['g-r'] = train_data['g'] - train_data['r']

X = train_data[['alpha', 'delta', 'mean_magnitude', 'color_index', 'cam_col', 'field_ID', 'spec_obj_ID', 'redshift', 'plate', 'MJD', 'fiber_ID', 'u-g', 'g-r']]
y = train_data['class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X.columns)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val_imputed), columns=X.columns)

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=rf_param_grid, cv=3, scoring='accuracy', verbose=1)
rf_grid_search.fit(X_train_scaled, y_train)
best_rf_model = rf_grid_search.best_estimator_
y_val_pred_rf_tuned = best_rf_model.predict(X_val_scaled)

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb_model = XGBClassifier(random_state=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train)
best_xgb_model = grid_search.best_estimator_
y_val_pred_xgb = best_xgb_model.predict(X_val_scaled)

soft_voting_model = VotingClassifier(estimators=[('rf', best_rf_model), ('xgb', best_xgb_model)], voting='soft')
soft_voting_model.fit(X_train_scaled, y_train)
y_val_pred_soft_voting = soft_voting_model.predict(X_val_scaled)

print("Tuned RF acc:", accuracy_score(y_val, y_val_pred_rf_tuned))
print("xgboost acc:", accuracy_score(y_val, y_val_pred_xgb))
print("Soft Voting acc:", accuracy_score(y_val, y_val_pred_soft_voting))
