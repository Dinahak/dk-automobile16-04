import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.title("🚗 Automobile Data Analysis")

# Load data
path = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
df = pd.read_csv(path)

st.subheader("1. Dataset Preview")
st.dataframe(df.head())

st.subheader("2. Data Types")
st.write(df.dtypes)

st.subheader("3. Correlation of Selected Features")
selected_cols = ['bore', 'stroke', 'compression-ratio', 'horsepower']
st.write(df[selected_cols].corr())

st.subheader("4. Visualization: Engine Size vs. Price")
fig, ax = plt.subplots()
sns.regplot(x="engine-size", y="price", data=df, ax=ax)
st.pyplot(fig)

st.subheader("5. Visualization: Highway MPG vs. Price")
fig, ax = plt.subplots()
sns.regplot(x="highway-mpg", y="price", data=df, ax=ax)
st.pyplot(fig)

st.subheader("6. Visualization: Peak RPM vs. Price")
fig, ax = plt.subplots()
sns.regplot(x="peak-rpm", y="price", data=df, ax=ax)
st.pyplot(fig)

st.subheader("7. Visualization: Stroke vs. Price")
fig, ax = plt.subplots()
sns.regplot(x="stroke", y="price", data=df, ax=ax)
st.pyplot(fig)

st.subheader("8. Boxplot: Body Style vs. Price")
fig, ax = plt.subplots()
sns.boxplot(x="body-style", y="price", data=df, ax=ax)
st.pyplot(fig)

st.subheader("9. Boxplot: Engine Location vs. Price")
fig, ax = plt.subplots()
sns.boxplot(x="engine-location", y="price", data=df, ax=ax)
st.pyplot(fig)

st.subheader("10. Boxplot: Drive Wheels vs. Price")
fig, ax = plt.subplots()
sns.boxplot(x="drive-wheels", y="price", data=df, ax=ax)
st.pyplot(fig)

st.subheader("11. Descriptive Statistics")
st.write(df.describe())
st.write(df.describe(include=['object']))

st.subheader("12. Value Counts: Drive Wheels")
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
st.write(drive_wheels_counts)

st.subheader("13. Grouping: Drive Wheels and Price")
df_group_one = df[['drive-wheels','body-style','price']].copy()
body_style  = {'sedan': 0, 'hatchback': 1, 'wagon': 2, 'convertible': 3, 'hardtop': 4}
df_group_one['body-style'] = df_group_one['body-style'].map(body_style)
grouped = df_group_one.groupby(['drive-wheels'], as_index=False).mean()
st.write(grouped)

st.subheader("14. Pivot Table and Heatmap")
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'], as_index=False).mean()
grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')
grouped_pivot = grouped_pivot.fillna(0)

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='Pastel1')

# labels
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5)
ax.set_xticklabels(row_labels, rotation=90)
ax.set_yticklabels(col_labels)
fig.colorbar(im)
st.pyplot(fig)

st.subheader("15. Correlation Matrix")
df_numeric = df.select_dtypes(include='number')
correlation_matrix = df_numeric.corr()
st.write(correlation_matrix)

st.markdown("---")
st.markdown("🔍 *This app explores relationships between features and car prices to identify key predictors.*")
