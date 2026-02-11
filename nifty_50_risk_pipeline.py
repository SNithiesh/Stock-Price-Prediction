# -*- coding: utf-8 -*-
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
5. How does NIFTY 50 compare to individual stocks?

DATA SOURCE:
-----------
Uses local CSV files with 20 years of NIFTY 50 historical data
- Index data: nifty50_historical_data.csv
- Individual stocks: nifty50_summary_statistics.csv (49 stocks)

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
# DATA INGESTION & PROCESSING
# =========================
print("Loading NIFTY 50 historical data from CSV (20 years)")

# Path to your CSV file
csv_path = r"C:\Users\mksni\OneDrive\Desktop\DA &DP\nifty50_historical_data.csv"

# Read CSV file with proper date parsing
raw_df = pd.read_csv(csv_path, parse_dates=['Date'])

# Pivot data to get individual stock columns (Date x Ticker)
# This prevents mixing different stocks into one series
pivot_close = raw_df.pivot(index='Date', columns='Ticker', values='Close')

# Calculate returns for all stocks
stock_returns = pivot_close.pct_change()

# Calculate Equal-Weighted Index Return (Proxy for NIFTY 50 Index)
# We take the mean return of all available stocks for each day
index_returns = stock_returns.mean(axis=1)
df = pd.DataFrame({'Close': index_returns}) # Create dummy Close frame for compatibility
df['Return'] = index_returns
df.dropna(inplace=True)

# Reconstruct a "Close" price series starting at 100 for visualization
# (Cumulative product of returns)
df['Close'] = 100 * (1 + df['Return']).cumprod()

print(f"Loaded data for {pivot_close.shape[1]} stocks")
print(f"Constructed NIFTY 50 Equal-Weighted Index from components")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Total trading days: {len(df)}")

# =========================
# LOAD INDIVIDUAL STOCKS DATA (ALREADY LOADED)
# =========================
# We can use our pivoted data for more accurate comparative analysis
# calculating metrics directly from the time series rather than summary file
print("\nCalculating individual stock statistics from time series...")

# Calculate metrics for each stock from the actual daily data
stocks_metrics = pd.DataFrame(index=stock_returns.columns)
stocks_metrics['Avg_Daily_Return_%'] = stock_returns.mean() * 100
stocks_metrics['Volatility_%'] = stock_returns.std() * np.sqrt(252) * 100
stocks_metrics['Sharpe_Ratio'] = (stocks_metrics['Avg_Daily_Return_%'] * 252 - 6.5) / stocks_metrics['Volatility_%']

# Add sector info and market cap from the summary file if available
summary_path = r"C:\Users\mksni\OneDrive\Desktop\DA &DP\nifty50_summary_statistics.csv"
try:
    summary_df = pd.read_csv(summary_path)
    # create maps for quick lookup
    sector_map = dict(zip(summary_df['Ticker'], summary_df['Sector']))
    name_map = dict(zip(summary_df['Ticker'], summary_df['Company_Name']))
    mcap_map = dict(zip(summary_df['Ticker'], summary_df['Current_Market_Cap']))
    
    stocks_metrics['Sector'] = stocks_metrics.index.map(sector_map)
    stocks_metrics['Company_Name'] = stocks_metrics.index.map(name_map)
    stocks_metrics['Current_Market_Cap'] = stocks_metrics.index.map(mcap_map)
    
    # Fill missing values
    stocks_metrics['Sector'].fillna('Unknown', inplace=True)
    stocks_metrics['Company_Name'].fillna(stocks_metrics.index, inplace=True)
    stocks_metrics['Current_Market_Cap'].fillna(stocks_metrics['Current_Market_Cap'].median(), inplace=True)
    
    stocks_df = stocks_metrics.reset_index().rename(columns={'index': 'Ticker'})
    print(f"Successfully matched metadata for {len(stocks_df)} stocks")
except Exception as e:
    print(f"Warning: Could not load summary info: {e}")
    stocks_df = stocks_metrics.reset_index().rename(columns={'index': 'Ticker'})
    stocks_df['Sector'] = 'Unknown'
    stocks_df['Company_Name'] = stocks_df['Ticker']
    stocks_df['Current_Market_Cap'] = 1e11 # Default dummy market cap

# =========================
# RETURNS AND RISK
# =========================
# (Returns already calculated in Data Ingestion step)

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
print(f"  - Better Returns:     {better_return_count}/{len(stocks_df)} stocks ({better_return_count/len(stocks_df)*100:.1f}%)")
print(f"  - Lower Volatility:   {lower_vol_count}/{len(stocks_df)} stocks ({lower_vol_count/len(stocks_df)*100:.1f}%)")
print(f"  - Better Sharpe:      {better_sharpe_count}/{len(stocks_df)} stocks ({better_sharpe_count/len(stocks_df)*100:.1f}%)")

# Index percentile ranking
return_percentile = (stocks_df['Avg_Daily_Return_%'] * 252 < index_annual_return).sum() / len(stocks_df) * 100
vol_percentile = (stocks_df['Volatility_%'] > index_annual_vol).sum() / len(stocks_df) * 100

print(f"\nNIFTY 50 Index Percentile Rankings:")
print(f"  - Return Percentile:  {return_percentile:.1f}th (higher is better)")
print(f"  - Risk Percentile:    {vol_percentile:.1f}th (higher means lower risk)")

# Top performers
print(f"\nTop 5 Stocks by Sharpe Ratio:")
top_sharpe = better_sharpe_stocks.nlargest(5, 'Sharpe_Ratio')[['Company_Name', 'Sector', 'Avg_Daily_Return_%', 'Volatility_%', 'Sharpe_Ratio']]
for idx, row in top_sharpe.iterrows():
    print(f"  {idx+1}. {row['Company_Name'][:25]:25s} | Sharpe: {row['Sharpe_Ratio']:.3f} | Return: {row['Avg_Daily_Return_%']*252:6.2f}% | Vol: {row['Volatility_%']:5.2f}%")


# =========================
# STATISTICAL CONFIDENCE
# =========================
stat, p_value = stats.shapiro(df["Return"])
print(f"\nNormality p-value: {p_value:.6f}")

# =========================
# STATISTICAL MODEL
# =========================
df["Lag1"] = df["Return"].shift(1)
df.dropna(inplace=True)

X = sm.add_constant(df["Lag1"])
y = df["Return"]

ols = sm.OLS(y, X).fit()
print("\n" + "="*60)
print("OLS REGRESSION SUMMARY")
print("="*60)
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
print("\n" + "="*60)
print("GARCH MODEL FOR VOLATILITY CLUSTERING")
print("="*60)
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
print("\n" + "="*60)
print(f"Option price for decision making: {option.NPV():.2f}")
print("="*60)

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

# Create figure with improved spacing
fig = plt.figure(figsize=(22, 13))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35, 
                      left=0.05, right=0.98, top=0.94, bottom=0.05)

# ========== PANEL 1: NIFTY 50 Return Distribution ==========
ax1 = fig.add_subplot(gs[0, :2])
sns.histplot(df["Return"], bins=50, kde=True, color=colors['primary'], alpha=0.6, ax=ax1)
ax1.axvline(mean_return, color=colors['danger'], linestyle='--', linewidth=2.5, 
            label=f'Mean: {mean_return*100:.4f}%')
ax1.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
ax1.set_title('NIFTY 50 Index: Return Distribution (20 Years)', 
              fontsize=17, fontweight='bold', pad=20)
ax1.set_xlabel('Daily Returns', fontsize=13, fontweight='semibold')
ax1.set_ylabel('Frequency', fontsize=13, fontweight='semibold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')

# ========== PANEL 2: Key Statistics Box ==========
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')

# Create perfectly aligned statistics text
stats_lines = [
    "╔═══════════════════════════════════╗",
    "║  NIFTY 50 INDEX STATISTICS        ║",
    "╠═══════════════════════════════════╣",
    "║                                   ║",
    "║  Daily Metrics:                   ║",
    f"║    Mean Return : {mean_return*100:>8.4f}%      ║",
    f"║    Std Dev     : {std_risk*100:>8.4f}%      ║",
    "║                                   ║",
    "║  Annualized Metrics:              ║",
    f"║    Return      : {mean_return*252*100:>8.2f}%      ║",
    f"║    Volatility  : {std_risk*np.sqrt(252)*100:>8.2f}%      ║",
    f"║    Sharpe Ratio: {sharpe_ratio:>8.4f}       ║",
    "║                                   ║",
    "║  Comparative Performance:         ║",
    f"║    Outperformed: {better_return_count:>2d} stocks       ║",
    f"║    Lower Vol   : {lower_vol_count:>2d} stocks       ║",
    "║                                   ║",
    "║  Percentile Rank:                 ║",
    f"║    Returns     : {return_percentile:>6.1f}th        ║",
    f"║    Risk        : {vol_percentile:>6.1f}th        ║",
    "║                                   ║",
    "╚═══════════════════════════════════╝",
]

stats_text = '\n'.join(stats_lines)
ax2.text(0.05, 0.98, stats_text, transform=ax2.transAxes, fontsize=10.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF8DC', 
                   edgecolor='#8B7355', linewidth=2, alpha=0.9))

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
                     alpha=0.7,
                     cmap='RdYlGn',
                     edgecolors='black',
                     linewidth=0.8,
                     vmin=-5, vmax=15)

# Add NIFTY 50 Index point
ax3.scatter(index_annual_vol, index_annual_return, 
           s=600, c=colors['index'], marker='*',  # Star marker
           edgecolors='black', linewidth=2.5,
           label='NIFTY 50 Index', zorder=5)

# Add labels for top 5 performers
top_5 = scatter_data.nlargest(5, 'Sharpe_Ratio')
for idx, row in top_5.iterrows():
    ax3.annotate(row['Company_Name'].split()[0], 
                xy=(row['Volatility_%'], row['Annual_Return']),
                xytext=(8, 8), textcoords='offset points',
                fontsize=9, fontweight='bold', alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

ax3.set_title('Risk-Return Profile: All NIFTY 50 Stocks (Bubble Size = Market Cap)', 
             fontsize=17, fontweight='bold', pad=20)
ax3.set_xlabel('Annualized Volatility (%)', fontsize=13, fontweight='semibold')
ax3.set_ylabel('Annualized Return (%)', fontsize=13, fontweight='semibold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(fontsize=11, loc='upper left')

# Add colorbar with better formatting
cbar = plt.colorbar(scatter, ax=ax3, pad=0.01)
cbar.set_label('Sharpe Ratio', fontsize=12, fontweight='semibold')
cbar.ax.tick_params(labelsize=10)

# ========== PANEL 4: Sector Analysis ==========
ax4 = fig.add_subplot(gs[2, 0])

sector_stats = scatter_data.groupby('Sector').agg({
    'Annual_Return': 'mean',
    'Volatility_%': 'mean',
    'Sharpe_Ratio': 'mean'
}).sort_values('Sharpe_Ratio', ascending=True)

colors_sectors = plt.cm.viridis(np.linspace(0, 1, len(sector_stats)))
bars = ax4.barh(range(len(sector_stats)), sector_stats['Sharpe_Ratio'].values, 
                color=colors_sectors, edgecolor='black', linewidth=0.8)

ax4.set_yticks(range(len(sector_stats)))
ax4.set_yticklabels(sector_stats.index, fontsize=10)
ax4.set_title('Average Sharpe Ratio by Sector', fontsize=15, fontweight='bold', pad=15)
ax4.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='semibold')
ax4.grid(True, alpha=0.3, axis='x', linestyle='--')

# Add value labels on bars
for i, (idx, val) in enumerate(sector_stats['Sharpe_Ratio'].items()):
    ax4.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')

# ========== PANEL 5: Top 10 Performers ==========
ax5 = fig.add_subplot(gs[2, 1])

top_10 = scatter_data.nlargest(10, 'Sharpe_Ratio')
colors_top10 = plt.cm.RdYlGn(np.linspace(0.5, 1, 10))
y_pos = np.arange(len(top_10))

bars = ax5.barh(y_pos, top_10['Sharpe_Ratio'].values, color=colors_top10,
                edgecolor='black', linewidth=0.8)
ax5.set_yticks(y_pos)
ax5.set_yticklabels([name.split()[0][:15] for name in top_10['Company_Name']], fontsize=10)
ax5.set_title('Top 10 Stocks by Sharpe Ratio', fontsize=15, fontweight='bold', pad=15)
ax5.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='semibold')
ax5.grid(True, alpha=0.3, axis='x', linestyle='--')
ax5.invert_yaxis()

# Add value labels on bars
for i, val in enumerate(top_10['Sharpe_Ratio'].values):
    ax5.text(val + 0.2, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')

# ========== PANEL 6: Volatility Distribution Comparison ==========
ax6 = fig.add_subplot(gs[2, 2])

# Box plot comparing volatility
bp = ax6.boxplot([scatter_data['Volatility_%'].values], 
                 labels=['Individual\nStocks'], 
                 patch_artist=True, widths=0.6,
                 boxprops=dict(facecolor=colors['secondary'], alpha=0.7, linewidth=1.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5),
                 medianprops=dict(color='red', linewidth=2))

# Add index line
ax6.axhline(y=index_annual_vol, color=colors['index'], 
           linestyle='--', linewidth=2.5, label=f'NIFTY 50: {index_annual_vol:.1f}%')

ax6.set_title('Volatility Distribution', fontsize=15, fontweight='bold', pad=15)
ax6.set_ylabel('Annualized Volatility (%)', fontsize=12, fontweight='semibold')
ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
ax6.legend(fontsize=10, loc='upper right')

# Add statistics annotation
median_vol = scatter_data['Volatility_%'].median()
ax6.text(0.98, 0.02, f'Median: {median_vol:.1f}%\nNIFTY 50: {index_annual_vol:.1f}%', 
         transform=ax6.transAxes, fontsize=9, verticalalignment='bottom',
         horizontalalignment='right', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add overall title with better positioning
fig.suptitle('NIFTY 50 COMPREHENSIVE RISK ANALYSIS DASHBOARD', 
            fontsize=22, fontweight='bold', y=0.985)

# Save with high quality
plt.savefig('nifty50_dashboard.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Dashboard saved as 'nifty50_dashboard.png'")
plt.show()

print(f"\n{'='*60}")
print("NIFTY 50 RISK PIPELINE COMPLETED")
print(f"{'='*60}")
