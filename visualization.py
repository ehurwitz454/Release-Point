import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import pandas as pd
import scipy.stats as stats
def plot_rmse_vs_ops(merged_df, eligible_pitchers):
    merged_eligible = merged_df[merged_df['player_name'].isin(eligible_pitchers)]
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=merged_eligible, x='release_point_rmse', y='OPS_against', color='blue', alpha=0.6)
    plt.title('RMSE vs OPS Against for Eligible Games (80+ Pitches)')
    plt.xlabel('Release Point RMSE')
    plt.ylabel('OPS Against')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_whiff_rate_vs_rmse(rmse_whiff_merged):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=rmse_whiff_merged[rmse_whiff_merged['pitch_segment'] == 'First 40'],
                    x='release_point_rmse', y='whiff_rate', color='blue', alpha=0.6)
    plt.title('RMSE vs Whiff Rate – First 25 Pitches')
    plt.xlabel('Release Point RMSE')
    plt.ylabel('Whiff Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=rmse_whiff_merged[rmse_whiff_merged['pitch_segment'] == 'Rest'],
                    x='release_point_rmse', y='whiff_rate', color='green', alpha=0.6)
    plt.title('RMSE vs Whiff Rate – Pitches 25+')
    plt.xlabel('Release Point RMSE')
    plt.ylabel('Whiff Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_logit_results(results_df, top_n=20):
    # Filter for significant results only
    sig_results = results_df[results_df['P-Value'] < 0.05].copy()

    if sig_results.empty:
        print("No significant results at p < 0.05.")
        return

    # Optional: limit to top_n by absolute coefficient size
    sig_results = sig_results.reindex(sig_results['Coef'].abs().sort_values(ascending=False).index)
    sig_results = sig_results.head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=sig_results,
        x='Coef',
        y='From',
        hue='To',
        dodge=False
    )
    plt.title(f'Significant Transitions (p < 0.05): Release Diff → Whiff Coef')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Logistic Regression Coefficient")
    plt.ylabel("From Pitch Type")
    plt.legend(title="To Pitch Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_rmse_vs_whiff(df_metrics):
    top_3 = df_metrics.nlargest(3, 'release_point_rmse')
    bottom_3 = df_metrics.nsmallest(3, 'release_point_rmse')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_metrics, x='release_point_rmse', y='whiff_rate')

    for _, row in pd.concat([top_3, bottom_3]).iterrows():
        plt.text(row['release_point_rmse'], row['whiff_rate'], f"{row['player_name']} ({row['game_date']})", fontsize=9)

    plt.title('Game Relase Point RMSE vs. Whiff Rate (80+ pitch games)')
    plt.xlabel('Release Point RMSE')
    plt.ylabel('Whiff Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_regression_line(x, y, model, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, alpha=0.6)
    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals = model.predict(sm.add_constant(x_vals))
    plt.plot(x_vals, y_vals, color='red', label='Regression Line')
    plt.title(title)
    plt.xlabel('Release Point RMSE')
    plt.ylabel('Whiff Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_eligible_whiff(df, game_counts):
    eligible_pitchers = game_counts[game_counts['game_count'] >= 12]['player_name']
    df = df[df['player_name'].isin(eligible_pitchers)]
    summary = df.groupby('player_name').agg({
        'release_point_rmse': 'mean',
        'whiff_rate': 'mean'
    }).reset_index()
    top5 = summary.nlargest(5, 'release_point_rmse')
    bottom5 = summary.nsmallest(5, 'release_point_rmse')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=summary, x='release_point_rmse', y='whiff_rate', alpha=0.7)

    # Add labels
    for _, row in pd.concat([top5, bottom5]).iterrows():
        plt.text(row['release_point_rmse'] + 0.002, row['whiff_rate'],
                 row['player_name'], fontsize=9)

    plt.title('Release Point RMSE vs Whiff Rate (Eligible Pitchers)')
    plt.xlabel('Average Release Point RMSE')
    plt.ylabel('Average Whiff Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_player(df, player_name):
    # Filter data for Jesus Luzardo
    player_df = df[df['player_name'] == player_name].dropna()

    # Set style
    sns.set(style="whitegrid")

    # Create scatterplot with regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=player_df,
        x='release_point_rmse',
        y='whiff_rate',
        scatter_kws={'s': 70, 'color': '#1f77b4'},
        line_kws={'color': 'red', 'linewidth': 2}
    )

    # Labels and title
    plt.title(f"{player_name}: RMSE vs Whiff Rate (Per Game)", fontsize=14)
    plt.xlabel("Release Point RMSE", fontsize=12)
    plt.ylabel("Whiff Rate", fontsize=12)
    plt.tight_layout()
    plt.show()


def qq_plot(data, title="Q-Q Plot"):
    plt.figure(figsize=(6, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_release_positions_3d(avg_release_dict):
    df = pd.DataFrame(avg_release_dict, index=['release_pos_x', 'release_pos_y', 'release_pos_z'])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    for i, column in enumerate(df.columns):
        x, y, z = df.loc['release_pos_x', column], df.loc['release_pos_y', column], df.loc['release_pos_z', column]
        ax.scatter(x, y, z, color=colors[i % len(colors)], label=column, s=60)
        ax.text(x, y, z, column, fontsize=8, color=colors[i % len(colors)])

    ax.set_xlabel('release_pos_x')
    ax.set_ylabel('release_pos_y')
    ax.set_zlabel('release_pos_z')
    ax.set_title('Average Release Position by Group and Throwing Side')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_release_pos_xz(avg_release_dict):
    df = pd.DataFrame(avg_release_dict, index=['release_pos_x', 'release_pos_y', 'release_pos_z'])

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    for i, column in enumerate(df.columns):
        x = df.loc['release_pos_x', column]
        z = df.loc['release_pos_z', column]
        ax.scatter(x, z, color=colors[i % len(colors)], label=column, s=80)
        ax.text(x + 0.02, z + 0.02, column, fontsize=8, color=colors[i % len(colors)])

    ax.set_xlabel('release_pos_x')
    ax.set_ylabel('release_pos_z')
    ax.set_title('Average Release Position (x vs z)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()