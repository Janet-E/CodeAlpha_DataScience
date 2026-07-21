import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Set clean styling for the charts
sns.set_theme(style="darkgrid")
print("STEP 1: PROCESSING EMPOWERED ECONOMIC DATA ")
# 1. Load the dataset
# Stripping whitespace in columns
df = pd.read_csv('Unemployment in India.csv', encoding='latin1')
df.columns = df.columns.str.strip()
# 2. Cleaning structural data & converting Date types
df.dropna(subset=['Date', 'Estimated Unemployment Rate (%)'], inplace=True)
df['Date'] = pd.to_datetime(df['Date'].str.strip(), dayfirst=True)
print(f"Data successfully cleaned. Rows: {df.shape[0]}")
print("Variables tracked:", list(df.columns))
print("\n STEP 2: INVESTIGATING THE COVID-19 DISRUPTION IMPACT ")
# Filtering data to divide pre-lockdown and lockdown peak (April 2020)
lockdown_peak = df[df['Date'] == '2020-04-30']
pre_pandemic = df[df['Date'] < '2020-03-01']
avg_pre = pre_pandemic['Estimated Unemployment Rate (%)'].mean()
avg_lockdown = lockdown_peak['Estimated Unemployment Rate (%)'].mean()
print(f"Average Pre-Pandemic Unemployment Rate: {avg_pre:.2f}%")
print(f"Average Peak Covid Lockdown Unemployment Rate: {avg_lockdown:.2f}%")
print(f"Economic Impact Shift: {((avg_lockdown - avg_pre) / avg_pre) * 100:.1f}% surge")
print("\n  STEP 3: MAPPING REGIONAL TRENDS & GENERATING VISUALS ")
# Chart 1: Time Series Timeline Trend
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)', hue='Area', errorbar=None, linewidth=2.5)
plt.axvline(pd.to_datetime('2020-03-25'), color='red', linestyle='--', label='Lockdown Start')
plt.title('Unemployment Rate Trajectory Over Time (Rural vs Urban)', fontsize=14, pad=15)
plt.xlabel('Timeline')
plt.ylabel('Unemployment Percentage (%)')
plt.legend()
plt.tight_layout()
plt.savefig('unemployment_timeline.png')
print("-> Saved visual: unemployment_timeline.png")
plt.close()
# Chart 2: Regional Impact Scale
plt.figure(figsize=(12, 8))
# Dynamically find the column name that contains the word "Region"
region_col = [col for col in df.columns if 'Region' in col][0]
# Group by the dynamically matched column name safely
state_rankings = df.groupby(region_col)['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=state_rankings, x='Estimated Unemployment Rate (%)', y=region_col, palette='viridis')
plt.title('Regional Disparity Breakdown: Highest Affected Areas', fontsize=14, pad=15)
plt.xlabel('Average Unemployment Rate (%)')
plt.ylabel('State / Region')
plt.tight_layout()
plt.savefig('regional_impact_bar.png')
print("-> Saved visual: regional_impact_bar.png")
plt.close()


