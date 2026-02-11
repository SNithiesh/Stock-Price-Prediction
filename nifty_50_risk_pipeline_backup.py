"""
NIFTY 50: RISK AND DECISION-FOCUSED PIPELINE
=========================================

WHAT THIS FILE IS SUPPOSED TO DO:
--------------------------------
This example focuses on DECISION MAKING rather than prediction,
applied specifically to the NIFTY 50 Index using 20 years of historical data.

Questions answered:
1. How risky is the NIFTY 50 index?
2. Does volatility cluster in the Indian market?
3. What is a fair option price for the index?
4. How do statistics support decisions?

DATA SOURCE:
-----------
Uses local CSV file with 20 years of NIFTY 50 historical data
Location: C:\Users\mksni\OneDrive\Desktop\DA &DP\nifty50_historical_data.csv

"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from arch import arch_model
import QuantLib as ql
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# =========================
# DATA INGESTION
# =========================
print("Loading NIFTY 50 historical data from CSV (20 years)")

# Path to your CSV file
csv_path = r"C:\Users\mksni\OneDrive\Desktop\DA &DP\nifty50_historical_data.csv"

# Read CSV file with proper date parsing
df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')

# Select only the Close column for analysis
# Note: CSV has columns like: sector, open, close, high, low, volume, dividend, stock split
df = df[["Close"]].copy()

print(f"Loaded {len(df)} days of historical data")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# =========================
# LOAD INDIVIDUAL STOCKS DATA
# =========================
print("\nLoading individual stock statistics for comparative analysis...")
stocks_path = r"C:\Users\mksni\OneDrive\Desktop\DA &DP\nifty50_summary_statistics.csv"
stocks_df = pd.read_csv(stocks_path)

print(f"Loaded {len(stocks_df)} individual stocks from NIFTY 50")
print(f"Sectors covered: {stocks_df['Sector'].nunique()}")

# =========================
# RETURNS AND RISK
# =========================
df["Return"] = df["Close"].pct_change()
df.dropna(inplace=True)

# Basic risk metrics
mean_return = np.mean(df["Return"])
std_risk = np.std(df["Return"])

print(f"\n{'='*60}")
print("NIFTY 50 INDEX RISK METRICS")
print(f"{'='*60}")
print(f"Mean Daily Return: {mean_return:.6f} ({mean_return*100:.4f}%)")
print(f"Risk (Std Dev):    {std_risk:.6f} ({std_risk*100:.4f}%)")
print(f"Annualized Return: {mean_return*252:.6f} ({mean_return*252*100:.2f}%)")
print(f"Annualized Vol:    {std_risk*np.sqrt(252):.6f} ({std_risk*np.sqrt(252)*100:.2f}%)")

# Calculate Sharpe Ratio (assuming risk-free rate of 6.5% for India)
risk_free_rate = 0.065
sharpe_ratio = (mean_return * 252 - risk_free_rate) / (std_risk * np.sqrt(252))
print(f"Sharpe Ratio:      {sharpe_ratio:.4f}")

# =========================
# COMPARATIVE ANALYSIS
# =========================
print(f"\n{'='*60}")
print("COMPARATIVE ANALYSIS: NIFTY 50 vs INDIVIDUAL STOCKS")
print(f"{'='*60}")

# Calculate index annualized metrics for comparison
index_annual_return = mean_return * 252 * 100
index_annual_vol = std_risk * np.sqrt(252) * 100

# Compare with individual stocks
better_return_count = len(stocks_df[stocks_df['Avg_Daily_Return_%'] * 252 > index_annual_return])
lower_vol_count = len(stocks_df[stocks_df['Volatility_%'] < index_annual_vol])
better_sharpe_stocks = stocks_df.copy()
better_sharpe_stocks['Sharpe_Ratio'] = (
    (better_sharpe_stocks['Avg_Daily_Return_%'] * 252 - risk_free_rate * 100) / 
    better_sharpe_stocks['Volatility_%']
)
better_sharpe_count = len(better_sharpe_stocks[better_sharpe_stocks['Sharpe_Ratio'] > sharpe_ratio])

print(f"\nStocks outperforming NIFTY 50 Index:")
print(f"  • Better Returns:     {better_return_count}/{len(stocks_df)} stocks ({better_return_count/len(stocks_df)*100:.1f}%)")
print(f"  • Lower Volatility:   {lower_vol_count}/{len(stocks_df)} stocks ({lower_vol_count/len(stocks_df)*100:.1f}%)")
print(f"  • Better Sharpe:      {better_sharpe_count}/{len(stocks_df)} stocks ({better_sharpe_count/len(stocks_df)*100:.1f}%)")

# Index percentile ranking
return_percentile = (stocks_df['Avg_Daily_Return_%'] * 252 < index_annual_return).sum() / len(stocks_df) * 100
vol_percentile = (stocks_df['Volatility_%'] > index_annual_vol).sum() / len(stocks_df) * 100

print(f"\nNIFTY 50 Index Percentile Rankings:")
print(f"  • Return Percentile:  {return_percentile:.1f}th (higher is better)")
print(f"  • Risk Percentile:    {vol_percentile:.1f}th (higher means lower risk)")

# Top performers
print(f"\nTop 5 Stocks by Sharpe Ratio:")
top_sharpe = better_sharpe_stocks.nlargest(5, 'Sharpe_Ratio')[['Company_Name', 'Sector', 'Avg_Daily_Return_%', 'Volatility_%', 'Sharpe_Ratio']]
for idx, row in top_sharpe.iterrows():
    print(f"  {idx+1}. {row['Company_Name'][:25]:25s} | Sharpe: {row['Sharpe_Ratio']:.3f} | Return: {row['Avg_Daily_Return_%']*252:6.2f}% | Vol: {row['Volatility_%']:5.2f}%")


# =========================
# STATISTICAL CONFIDENCE
# =========================
stat, p_value = stats.shapiro(df["Return"])
print(f"Normality p-value: {p_value:.6f}")

# =========================
# STATISTICAL MODEL
# =========================
df["Lag1"] = df["Return"].shift(1)
df.dropna(inplace=True)

X = sm.add_constant(df["Lag1"])
y = df["Return"]

ols = sm.OLS(y, X).fit()
print(ols.summary())

# =========================
# ML (SECONDARY)
# =========================
ml = LinearRegression()
ml.fit(df[["Lag1"]], df["Return"])

df["ML_Return"] = ml.predict(df[["Lag1"]])

# =========================
# VOLATILITY CLUSTERING
# =========================
# Modeling volatility for Indian Markets
garch = arch_model(df["Return"] * 100, p=1, q=1)
garch_fit = garch.fit(disp="off")
print(garch_fit.summary())

# =========================
# OPTION PRICING FOR DECISION
# =========================
ql.Settings.instance().evaluationDate = ql.Date.todaysDate()

spot = float(df["Close"].iloc[-1])
strike = spot * 1.05  # out-of-the-money
rate = 0.065  # Approximate risk-free rate for India (higher than US)
vol = np.std(df["Return"]) * np.sqrt(252)  # annualized
maturity = ql.Date(31, 12, 2027)  # Adjusted for future maturity

day_count = ql.Actual365Fixed()
# Using India calendar if available, otherwise fallback (QuantLib supports India)
calendar = ql.India()

option = ql.VanillaOption(
    ql.PlainVanillaPayoff(ql.Option.Call, strike),
    ql.EuropeanExercise(maturity)
)

process = ql.BlackScholesProcess(
    ql.QuoteHandle(ql.SimpleQuote(spot)),
    ql.YieldTermStructureHandle(
        ql.FlatForward(0, calendar, rate, day_count)
    ),
    ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(0, calendar, vol, day_count)
    )
)

option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
print("Option price for decision making:", option.NPV())

# =========================
# COMPREHENSIVE VISUALIZATION DASHBOARD
# =========================
print(f"\n{'='*60}")
print("GENERATING VISUALIZATION DASHBOARD")
print(f"{'='*60}")

# Set up the color palette
colors = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Rich purple
    'success': '#06A77D',      # Vibrant green
    'warning': '#F18F01',      # Warm orange
    'danger': '#C73E1D',       # Deep red
    'index': '#FFD23F',        # Gold for NIFTY 50
}

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ========== PANEL 1: NIFTY 50 Return Distribution ==========
ax1 = fig.add_subplot(gs[0, :2])
sns.histplot(df["Return"], bins=50, kde=True, color=colors['primary'], alpha=0.6, ax=ax1)
ax1.axvline(mean_return, color=colors['danger'], linestyle='--', linewidth=2, label=f'Mean: {mean_return*100:.4f}%')
ax1.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax1.set_title('NIFTY 50 Index: Return Distribution (20 Years)', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Daily Returns', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# ========== PANEL 2: Key Statistics Box ==========
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
NIFTY 50 INDEX STATISTICS

Daily Metrics:
  Mean Return: {mean_return*100:.4f}%
  Std Dev: {std_risk*100:.4f}%
  
Annualized Metrics:
  Return: {mean_return*252*100:.2f}%
  Volatility: {std_risk*np.sqrt(252)*100:.2f}%
  Sharpe Ratio: {sharpe_ratio:.4f}

Comparative Performance:
  Outperformed by: {better_return_count} stocks
  Lower Vol than: {lower_vol_count} stocks
  
Percentile Rank:
  Returns: {return_percentile:.1f}th
  Risk: {vol_percentile:.1f}th
"""
ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ========== PANEL 3: Risk-Return Scatter Plot ==========
ax3 = fig.add_subplot(gs[1, :])

# Prepare data for scatter
scatter_data = better_sharpe_stocks.copy()
scatter_data['Annual_Return'] = scatter_data['Avg_Daily_Return_%'] * 252

# Color by Sharpe Ratio
scatter = ax3.scatter(scatter_data['Volatility_%'], 
                     scatter_data['Annual_Return'],
                     c=scatter_data['Sharpe_Ratio'],
                     s=scatter_data['Current_Market_Cap'] / 1e11,  # Size by market cap
                     alpha=0.6,
                     cmap='RdYlGn',
                     edgecolors='black',
                     linewidth=0.5)

# Add NIFTY 50 Index point
ax3.scatter(index_annual_vol, index_annual_return, 
           s=500, c=colors['index'], marker='*',  # Star marker
           edgecolors='black', linewidth=2,
           label='NIFTY 50 Index', zorder=5)

# Add labels for top 5 performers
top_5 = scatter_data.nlargest(5, 'Sharpe_Ratio')
for idx, row in top_5.iterrows():
    ax3.annotate(row['Company_Name'].split()[0], 
                xy=(row['Volatility_%'], row['Annual_Return']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

ax3.set_title('Risk-Return Profile: All NIFTY 50 Stocks (Size = Market Cap)', 
             fontsize=16, fontweight='bold', pad=20)
ax3.set_xlabel('Annualized Volatility (%)', fontsize=12)
ax3.set_ylabel('Annualized Return (%)', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Sharpe Ratio', fontsize=10)

# ========== PANEL 4: Sector Analysis ==========
ax4 = fig.add_subplot(gs[2, 0])

sector_stats = scatter_data.groupby('Sector').agg({
    'Annual_Return': 'mean',
    'Volatility_%': 'mean',
    'Sharpe_Ratio': 'mean'
}).sort_values('Sharpe_Ratio', ascending=True)

colors_sectors = plt.cm.viridis(np.linspace(0, 1, len(sector_stats)))
sector_stats['Sharpe_Ratio'].plot(kind='barh', ax=ax4, color=colors_sectors)
ax4.set_title('Average Sharpe Ratio by Sector', fontsize=14, fontweight='bold', pad=15)
ax4.set_xlabel('Sharpe Ratio', fontsize=11)
ax4.set_ylabel('')
ax4.grid(True, alpha=0.3, axis='x')

# ========== PANEL 5: Top 10 Performers ==========
ax5 = fig.add_subplot(gs[2, 1])

top_10 = scatter_data.nlargest(10, 'Sharpe_Ratio')
colors_top10 = plt.cm.RdYlGn(np.linspace(0.5, 1, 10))
y_pos = np.arange(len(top_10))

ax5.barh(y_pos, top_10['Sharpe_Ratio'].values, color=colors_top10)
ax5.set_yticks(y_pos)
ax5.set_yticklabels([name.split()[0][:15] for name in top_10['Company_Name']], fontsize=9)
ax5.set_title('Top 10 Stocks by Sharpe Ratio', fontsize=14, fontweight='bold', pad=15)
ax5.set_xlabel('Sharpe Ratio', fontsize=11)
ax5.grid(True, alpha=0.3, axis='x')
ax5.invert_yaxis()

# ========== PANEL 6: Volatility Distribution Comparison ==========
ax6 = fig.add_subplot(gs[2, 2])

# Box plot comparing volatility
volatility_data = [
    scatter_data['Volatility_%'].values,
    [index_annual_vol]
]

bp = ax6.boxplot(volatility_data[0:1], labels=['Individual\nStocks'], 
                patch_artist=True, widths=0.5)
for patch in bp['boxes']:
    patch.set_facecolor(colors['secondary'])
    patch.set_alpha(0.6)

# Add index line
ax6.axhline(y=index_annual_vol, color=colors['index'], 
           linestyle='--', linewidth=2, label='NIFTY 50 Index')

ax6.set_title('Volatility Distribution', fontsize=14, fontweight='bold', pad=15)
ax6.set_ylabel('Annualized Volatility (%)', fontsize=11)
ax6.grid(True, alpha=0.3, axis='y')
ax6.legend(fontsize=9)

# Add overall title
fig.suptitle('NIFTY 50 COMPREHENSIVE RISK ANALYSIS DASHBOARD', 
            fontsize=20, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('nifty50_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Dashboard saved as 'nifty50_dashboard.png'")
plt.show()

print(f"\n{'='*60}")
print("NIFTY 50 RISK PIPELINE COMPLETED")
print(f"{'='*60}")
