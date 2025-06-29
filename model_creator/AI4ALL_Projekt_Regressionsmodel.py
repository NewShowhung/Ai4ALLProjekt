# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:55:15 2025

@author: GongChitmon2018
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# โหลดข้อมูล
df = pd.read_csv("C:/Users/GongChitmon2018/Documents/parkhaeuser_wetter_merged_allvars.csv")
features = ['temp', 'humidity', 'wind_speed', 'wind_gust', 'precip']
target = 'Alsterhaus_auslastung'
df_model = df.dropna(subset=features + [target])
X = df_model[features]
y = df_model[target]

# แบ่ง train+val กับ test (80/20)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# เตรียม pipeline สำหรับ linear และ polynomial regression
linear_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
])
poly_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('lin_reg', LinearRegression())
])

# Cross-validation 10 รอบ หาโมเดลที่ดีที่สุดด้วย validation set
best_linear_mse = np.inf
best_poly_mse = np.inf
best_linear_model = None
best_poly_model = None

for i in range(10):
    # 25% ของ trainval เป็น validation (=> 60/20/20)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=100+i)
    
    # Linear
    linear_pipeline.fit(X_train, y_train)
    y_val_pred_lin = linear_pipeline.predict(X_val)
    y_val_pred_lin = np.clip(y_val_pred_lin, 0, None)
    mse_lin = mean_squared_error(y_val, y_val_pred_lin)
    if mse_lin < best_linear_mse:
        best_linear_mse = mse_lin
        best_linear_model = Pipeline([
            ('scaler', StandardScaler()),
            ('lin_reg', LinearRegression())
        ])
        best_linear_model.fit(X_train, y_train)
    
    # Poly
    poly_pipeline.fit(X_train, y_train)
    y_val_pred_poly = poly_pipeline.predict(X_val)
    y_val_pred_poly = np.clip(y_val_pred_poly, 0, None)
    mse_poly = mean_squared_error(y_val, y_val_pred_poly)
    if mse_poly < best_poly_mse:
        best_poly_mse = mse_poly
        best_poly_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
            ('lin_reg', LinearRegression())
        ])
        best_poly_model.fit(X_train, y_train)

# ทดสอบกับ test set
y_pred_lin = best_linear_model.predict(X_test)
y_pred_lin = np.clip(y_pred_lin, 0, 87)
y_pred_poly = best_poly_model.predict(X_test)
y_pred_poly = np.clip(y_pred_poly, 0, 87)

mae_lin = mean_absolute_error(y_test, y_pred_lin)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Plot ผลลัพธ์
plt.figure()
plt.scatter(y_test, y_pred_lin, label='Lineare Regression')
plt.scatter(y_test, y_pred_poly, label='Polynomiale Regression (Grad 2)')
min_val = min(y_test.min(), y_pred_lin.min(), y_pred_poly.min())
max_val = max(y_test.max(), y_pred_lin.max(), y_pred_poly.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
plt.xlabel('Ist-Wert (Auslastung)')
plt.ylabel('Vorhergesagt (Auslastung)')
plt.legend()
plt.title('Vorhersage vs. Ist - Alsterhaus (Lineare- und Polynomiale Regression) ')

textstr = (
    f"Linear:\n"
    f"  MAE: {mae_lin:.2f}\n"
    f"  MSE: {mse_lin:.2f}\n"
    f"  R²: {r2_lin:.3f}\n\n"
    f"Poly (2):\n"
    f"  MAE: {mae_poly:.2f}\n"
    f"  MSE: {mse_poly:.2f}\n"
    f"  R²: {r2_poly:.3f}"
)
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.show()