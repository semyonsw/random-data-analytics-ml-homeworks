import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('Semyon-Housing.csv')

df = pd.get_dummies(df, drop_first=True)

scaler = MinMaxScaler()
num_features = ['price', 'area', 'bedrooms', 'bathrooms', 'stories']
df[num_features] = scaler.fit_transform(df[num_features])

X = df.drop('price', axis=1)
y = df['price']

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()

r2_scores = []
sse_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    sse = mean_squared_error(y_test, y_pred) * len(y_test)

    r2_scores.append(r2)
    sse_scores.append(sse)

print(f"Average R²: {np.mean(r2_scores)}")
print(f"Average SSE: {np.mean(sse_scores)}")

fold_prices = [pd.concat([y.iloc[train_index], y.iloc[test_index]]) for train_index, test_index in kf.split(X)]
plt.boxplot(fold_prices)
plt.title("Price Distribution across Cross-Validation Folds")
plt.ylabel("Price")
plt.show()

selected_features = []
best_score = -np.inf
metrics = []

for _ in range(X.shape[1]):
    best_feature = None
    best_r2 = -np.inf

    for feature in X.columns:
        if feature not in selected_features:
            current_features = selected_features + [feature]
            X_train_selected = X[current_features]
            r2 = cross_val_score(model, X_train_selected, y, cv=kf, scoring='r2').mean()

            if r2 > best_r2:
                best_r2 = r2
                best_feature = feature

    selected_features.append(best_feature)
    metrics.append(best_r2)

plt.plot(range(1, len(metrics) + 1), metrics, marker='o')
plt.title("R² Score Improvement by Adding Features")
plt.xlabel("Number of Features Added")
plt.ylabel("R² Score")
plt.show()

X_selected = X[selected_features]
model.fit(X_selected, y)
final_r2 = r2_score(y, model.predict(X_selected))
print(f"Final R² with selected features: {final_r2}")