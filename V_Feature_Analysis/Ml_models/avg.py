import pandas as pd

# Load your results CSV
df = pd.read_csv("linear_regression_daywise_results.csv")

# Compute averages
avg_mse = df["mse"].mean()
avg_mae = df["mae"].mean()
avg_r2 = df["r2"].mean()
avg_direction_acc = df["direction_accuracy"].mean()

# Compute standard deviations (optional but useful)
std_mse = df["mse"].std()
std_mae = df["mae"].std()
std_r2 = df["r2"].std()
std_direction_acc = df["direction_accuracy"].std()

# Print results
print("====== Average Metrics Across Days ======")
print(f"Average MSE: {avg_mse:.10f}")
print(f"Average MAE: {avg_mae:.10f}")
print(f"Average R²:  {avg_r2:.10f}")
print(f"Average Direction Accuracy: {avg_direction_acc:.4f}")

print("\n====== Standard Deviations ======")
print(f"MSE Std: {std_mse:.10f}")
print(f"MAE Std: {std_mae:.10f}")
print(f"R² Std:  {std_r2:.10f}")
print(f"Direction Accuracy Std: {std_direction_acc:.4f}")
