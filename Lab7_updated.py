import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv('Semyon-Housing.csv')

df['unfurnished'] = (df['furnishingstatus'] == 'unfurnished').astype(int)
df['furnished'] = (df['furnishingstatus'] == 'furnished').astype(int)
df['semi-furnished'] = (df['furnishingstatus'] == 'semi-furnished').astype(int)
df.drop('furnishingstatus', axis=1, inplace=True)

yes_no_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[yes_no_columns] = df[yes_no_columns].replace({'yes': 1, 'no': -1})


scaler = MinMaxScaler()
num_features = ['price', 'area', 'bedrooms', 'bathrooms', 'stories']
df[num_features] = scaler.fit_transform(df[num_features])


X = df.drop('price', axis=1) # for train
y = df['price'] # y - label

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()

r2_scores = []
sse_scores = []

vat_y = df['price']

vat_y_pred = np.full_like(y, y.mean())


vat_sse = mean_squared_error(y, vat_y_pred) * len(y)
vat_r2 = r2_score(y, vat_y_pred)

print(f"vat SSE: {vat_sse}")
print(f"vat R²: {vat_r2}")

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    sse = mean_squared_error(y_test, y_pred) * len(y_test)

    r2_scores.append(r2)
    sse_scores.append(sse)


average_r2 = np.mean(r2_scores)
average_sse = np.mean(sse_scores)
print(f"Միջին R²: {average_r2}")
print(f"Միջին SSE: {average_sse}")


fold_prices = [pd.concat([y.iloc[train_index], y.iloc[test_index]]) for train_index, test_index in kf.split(X)]
plt.boxplot(fold_prices)
plt.title("Գնի բաշխումը ֆոլդներում")
plt.ylabel("Գին")
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
plt.title("R²֊ը ավելացումով")
plt.xlabel("Ավելացված սյուների քանակը")
plt.ylabel("R² ցուցանիշը")
plt.show()


X_selected = X[selected_features]
model.fit(X_selected, y)
final_r2 = r2_score(y, model.predict(X_selected))
print(f"Վերջնական R²: {final_r2}")