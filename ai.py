import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv('./star_train.csv')
train_data = train_data.dropna(subset=['class'])
X = train_data[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'cam_col', 'field_ID', 'spec_obj_ID', 'redshift', 'plate', 'MJD', 'fiber_ID']]
y = train_data['class']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X.columns)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val_imputed), columns=X.columns)

model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_scaled, y_train)

y_val_pred = model.predict(X_val_scaled)
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f'vs accuracry: {accuracy_val}')

print("validation set check:")
print(classification_report(y_val, y_val_pred))

test_data = pd.read_csv('./star_test.csv')
X_test = test_data[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'cam_col', 'field_ID', 'spec_obj_ID', 'redshift', 'plate', 'MJD', 'fiber_ID']]

X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X.columns)
y_test_pred = model.predict(X_test_scaled)

submission_df = pd.DataFrame({'ids': range(1, len(y_test_pred)+1), 'predicted_class': y_test_pred})
submission_df.to_csv('./submission.csv', index=False)




