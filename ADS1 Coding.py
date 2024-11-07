import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def describe_data(data):
    """Print basic descriptive statistics for the dataset, including skewness and kurtosis for numeric columns."""
    print("Data Description:")
    print(data.describe())

    # Select only numeric columns for skewness and kurtosis calculations
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    print("\nSkewness:")
    print(numeric_data.skew())

    print("\nKurtosis:")
    print(numeric_data.kurtosis())

def correlation_heatmap(data):
    """Generate and display a heatmap of correlations among numerical and boolean columns."""
    plt.figure(figsize=(10, 8))
    corr_data = data[['age', 'time_spent', 'income', 'indebt', 'isHomeOwner', 'Owns_Car']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', square=True, linewidths=0.5)
    plt.title("Correlation Heatmap of Numerical and Boolean Features", fontweight='bold')
    plt.show()

def plot_age_distribution(data):
    """Display a box plot showing the distribution of age."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='age', data=data, color='lightblue')
    plt.title("Age Distribution", fontweight='bold')
    plt.xlabel("Age", fontweight='bold')
    plt.show()

def plot_income_by_demographics(data):
    """Display a box plot for income distribution across different demographic categories."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='demographics', y='income', data=data, palette='Set3')
    plt.title("Income Distribution by Demographics", fontweight='bold')
    plt.xlabel("Demographics", fontweight='bold')
    plt.ylabel("Income", fontweight='bold')
    plt.show()

def plot_platform_usage_by_gender(data):
    """Display a stacked bar chart for platform usage by gender."""
    platform_gender_counts = data.groupby(['gender', 'platform']).size().unstack()
    platform_gender_counts.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 6))
    plt.title("Platform Usage by Gender", fontweight='bold')
    plt.xlabel("Gender", fontweight='bold')
    plt.ylabel("Number of Users", fontweight='bold')
    plt.xticks(rotation=45)
    plt.show()

def plot_average_income_by_age(data):
    """Display a line chart of average income by age group."""
    age_income_data = data.groupby('age')['income'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(age_income_data.index, age_income_data.values, marker='o', linestyle='-', color='b', alpha=0.7)
    plt.title("Average Income by Age", fontweight='bold')
    plt.xlabel("Age", fontweight='bold')
    plt.ylabel("Average Income", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Load and analyze data
file_path = 'dummy_data.csv'  # Replace with your actual file path if different
data = load_data(file_path)

# Perform descriptive statistics, skewness, and kurtosis analysis
describe_data(data)

# Plot visualizations
plot_age_distribution(data)
plot_income_by_demographics(data)
plot_platform_usage_by_gender(data)
plot_average_income_by_age(data)
correlation_heatmap(data)