import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming your data is in a pandas DataFrame called df
# and the 'price' column contains the data you want to visualize

# Set the style for the plot

df = pd.read_csv('Semyon-Housing.csv')
sns.set(style="whitegrid")

# Plot the distribution of the 'price' column
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=30, kde=True, color='skyblue')

# Labeling the plot
plt.title("Distribution of Price")
plt.xlabel("Price")
plt.ylabel("Frequency")

# Show plot
plt.show()