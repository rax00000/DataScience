import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px

data =pd.read_csv('data/merged_data_for_analysis.csv')
data = data.drop(columns='Unnamed: 0')


# 1. Descriptive Statistics
descriptive_stats = data.describe()

descriptive_stats.to_excel('EDA/descriptive_statistics.xlsx')

# 2. Count of Unique Values
unique_values = data[['Nationality', 'League', 'Club', 'Position']].nunique()

unique_values.to_excel('EDA/unique_values_EDA.xlsx')

# 3. Check missing values
df = pd.DataFrame(data)

df['Position Group'] = df['Position'].apply(lambda x: 'GK' if x == 'GK' else 'non-GK')

# Grouping by Position Group and League, then counting missing values in 'Annual wage in EUR'
missing_values_count = df.groupby(['Position Group', 'League'])['Annual wage in EUR'].apply(lambda x: x.isnull().sum()).reset_index()

# Calculate the total count for each group
total_counts = df.groupby(['Position Group', 'League']).size().reset_index(name='Total Count')

# Merge the total counts with the missing values count dataframe
missing_values_count = missing_values_count.merge(total_counts, on=['Position Group', 'League'])
missing_values_count['Percentage Missing'] = (missing_values_count['Annual wage in EUR'] / missing_values_count['Total Count'])

missing_values_count.to_excel('EDA/missing_data_by_position_league.xlsx')

# 4. Distributions
for column in data.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(df, x=column, title=f'Distribution of {column}')
        fig.write_html(f'EDA/{column}_histogram.html')
    else:
        fig = px.histogram(df, x=column, title=f'Distribution of {column}')
        fig.write_html(f'EDA/{column}_histogram.html')

#5. Correlation matrix

import plotly.graph_objects as go
df = data.drop(columns=['Player Name', 'Nationality', 'Position', 'League', 'Club'])
# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
                   z=correlation_matrix.values,
                   x=correlation_matrix.columns,
                   y=correlation_matrix.index,
                   colorscale='Viridis'))

fig.update_layout(title='Correlation Matrix', xaxis_nticks=36)

# Save the correlation matrix as an HTML file
fig.write_html('EDA/correlation_matrix.html')

import numpy as np

# Make table
correlation_matrix = df.corr()

# Extract the upper triangle of the correlation matrix, excluding the diagonal
correlation_matrix = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

# Stack the matrix and drop NaN values
correlation_pairs = correlation_matrix.stack().reset_index()
correlation_pairs.columns = ['Variable1', 'Variable2', 'Correlation']

# Rank the pairs by the absolute correlation
correlation_pairs['Absolute Correlation'] = correlation_pairs['Correlation'].abs()
ranked_pairs = correlation_pairs.sort_values(by='Absolute Correlation', ascending=False).reset_index(drop=True)

ranked_pairs.to_excel('EDA/correlation_pairs_ranked.xlsx')

import pandas as pd
import plotly.express as px

# Assuming 'data' is your DataFrame
# Replace this with the actual DataFrame load code if needed
# data = pd.read_csv('your_data.csv')

import pandas as pd
import plotly.express as px

# Assuming 'data' is your DataFrame
# Replace this with the actual DataFrame load code if needed
# data = pd.read_csv('your_data.csv')

# Get unique leagues
leagues = data['League'].unique()

# Define the upper threshold
upper_threshold = 25_000_000

# Loop through each league and create a box plot
for league in leagues:
    # Filter data for the current league
    league_data = data[data['League'] == league]

    # Create the box plot with y-axis range
    fig = px.box(league_data, y='Annual wage in EUR', title=f'Annual Wage in EUR for {league}')
    fig.update_yaxes(range=[0, upper_threshold])  # Set the y-axis range to have a maximum of 25 million

    # Save the plot as HTML
    fig.write_html(f"{league}_box.html")

print("Box plots saved successfully.")
