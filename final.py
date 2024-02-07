import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('./star_train.csv')
train_data = train_data.dropna(subset=['class'])

label_encoder = LabelEncoder()
train_data['class'] = label_encoder.fit_transform(train_data['class'])

train_data['mean_magnitude'] = train_data[['u', 'g', 'r', 'i', 'z']].mean(axis=1)
train_data['color_index'] = train_data['u'] - train_data['r']

X = train_data[['alpha', 'delta', 'mean_magnitude', 'color_index', 'cam_col', 'field_ID', 'spec_obj_ID', 'redshift', 'plate', 'MJD', 'fiber_ID']]
y = train_data['class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X.columns)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val_imputed), columns=X.columns)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_val_pred_rf = rf_model.predict(X_val_scaled)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb_model = XGBClassifier(random_state=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train)

best_xgb_model = grid_search.best_estimator_
y_val_pred_xgb = best_xgb_model.predict(X_val_scaled)

voting_model = VotingClassifier(estimators=[('rf', rf_model), ('xgb', best_xgb_model)], voting='hard')
voting_model.fit(X_train_scaled, y_train)
y_val_pred_voting = voting_model.predict(X_val_scaled)

print("RF acc:", accuracy_score(y_val, y_val_pred_rf))
print("xgboost acc:", accuracy_score(y_val, y_val_pred_xgb))
print("VC acc:", accuracy_score(y_val, y_val_pred_voting))

final_model = rf_model


test_data = pd.read_csv('./star_test.csv')

test_data['mean_magnitude'] = test_data[['u', 'g', 'r', 'i', 'z']].mean(axis=1)
test_data['color_index'] = test_data['u'] - test_data['r']

X_test = test_data[['alpha', 'delta', 'mean_magnitude', 'color_index', 'cam_col', 'field_ID', 'spec_obj_ID', 'redshift', 'plate', 'MJD', 'fiber_ID']]

X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns)

y_test_pred = final_model.predict(X_test_scaled)

submission_df = pd.DataFrame({'ids': range(1, len(test_data) + 1), 'class': y_test_pred})
submission_df['class'] = label_encoder.inverse_transform(submission_df['class'])

submission_df.to_csv('./submission.csv', index=False)