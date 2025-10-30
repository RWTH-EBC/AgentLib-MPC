import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from agentlib_mpc.models.casadi_predictor import FunctionalWrapper
from physXAI import models
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error



# Load models
model_keras = keras.models.load_model(r'D:\phe\Git\testhall_offices_experiment\physXAI\unittests\modular\models\model.keras')
model_casadi = FunctionalWrapper(model_keras).functional

# Load sample data
data = pd.read_csv('data/sample_data.csv', sep=';', index_col=0)

# Separate features and target
X = data[['x1', 'x2', 'x3']].values
y_true = data['x4'].values

# Make predictions with both models
y_keras = model_keras.predict(X, verbose=0).flatten()

# Predict with casadi model
y_casadi = np.array([float(model_casadi(X[i, :])) for i in range(len(X))])

# Calculate differences
abs_diff = np.abs(y_keras - y_casadi)
rel_error = (abs_diff / np.abs(y_keras)) * 100

# Find 10 rows with largest relative error
top_10_indices = np.argsort(rel_error)[-10:][::-1]

print("Top 10 Rows with highest relative error:")
print("="*100)
print(f"{'Index':<8} {'Keras':<15} {'CasADi':<15} {'Abs. Diff':<15} {'Rel. Error (%)':<15}")
print("-"*100)
for idx in top_10_indices:
    print(f"{idx:<8} {y_keras[idx]:<15.6f} {y_casadi[idx]:<15.6f} {abs_diff[idx]:<15.6f} {rel_error[idx]:<15.6f}")

# Calculate R2 and RMSE
r2 = r2_score(y_keras, y_casadi)
rmse = np.sqrt(mean_squared_error(y_keras, y_casadi))

print("\n" + "="*100)
print(f"R² Score: {r2:.6f}")
print(f"RMSE: {rmse:.6f}")
print("="*100)

# Plot results - sorted by keras predictions
sort_indices = np.argsort(y_keras)
y_keras_sorted = y_keras[sort_indices]
y_casadi_sorted = y_casadi[sort_indices]
original_indices = sort_indices

# Create interactive plot with Plotly
fig = go.Figure()

# Add CasADi predictions (red)
fig.add_trace(go.Scatter(
    x=list(range(len(y_casadi_sorted))),
    y=y_casadi_sorted,
    mode='markers',
    name='CasADi Model',
    marker=dict(color='red', size=6, opacity=0.6),
    hovertemplate='<b>CasADi Model</b><br>' +
                  'Sorted Index: %{x}<br>' +
                  'Original Index: %{customdata}<br>' +
                  'Prediction: %{y:.4f}<br>' +
                  '<extra></extra>',
    customdata=original_indices
))

# Add Keras predictions (green)
fig.add_trace(go.Scatter(
    x=list(range(len(y_keras_sorted))),
    y=y_keras_sorted,
    mode='markers',
    name='Keras Model',
    marker=dict(color='green', size=6, opacity=0.6),
    hovertemplate='<b>Keras Model</b><br>' +
                  'Sorted Index: %{x}<br>' +
                  'Original Index: %{customdata}<br>' +
                  'Prediction: %{y:.4f}<br>' +
                  '<extra></extra>',
    customdata=original_indices
))

# Update layout
fig.update_layout(
    title='Comparison: Keras vs CasADi Model Predictions',
    xaxis_title='Sorted Index (by Keras Predictions)',
    yaxis_title='Prediction Value',
    hovermode='closest',
    width=None,
    height=None,
    template='plotly_white',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Show the interactive plot (full page)
fig.show(config={'displayModeBar': True, 'responsive': True})