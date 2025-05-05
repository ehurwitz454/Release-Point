import statsmodels.api as sm
from scipy.stats import ttest_rel, shapiro
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel, pearsonr
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
from scipy.stats import f_oneway

def perform_t_test(first_25_rmse, rest_rmse):
    t_stat, p_value = ttest_rel(first_25_rmse, rest_rmse)
    return t_stat, p_value

def perform_regression(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.summary()

def calculate_correlation(game_summary_df):
    correlation_results = []
    for pitcher, group in game_summary_df.groupby('player_name'):
        if len(group) >= 12:
            r, pval = pearsonr(group['release_point_rmse'], group['whiff_rate'])
            correlation_results.append({
                'player_name': pitcher,
                'n_games': len(group),
                'correlation': r,
                'p_value': pval
            })
    correlation_df = pd.DataFrame(correlation_results)
    return correlation_df

def run_linear_regression(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    return model


def safe_paired_ttest(data1, data2, alpha=0.05):
    # Ensure same length and type
    if len(data1) != len(data2):
        raise ValueError(f"Datasets must be the same length. Got {len(data1)} and {len(data2)}.")

    # Drop NaNs and ensure matching index
    paired_df = (
        pd.DataFrame({'A': data1, 'B': data2})
        .dropna()
    )

    if paired_df.empty or len(paired_df) < 5:
        raise ValueError("Not enough valid paired data to run t-test.")

    # Optional normality test of the differences
    diffs = paired_df['A'] - paired_df['B']
    stat, p_normality = shapiro(diffs)
    normal = p_normality > alpha

    print(f"Shapiro-Wilk normality test p = {p_normality:.3f} (Normal: {normal})")

    # Paired t-test
    t_stat, p_val = ttest_rel(paired_df['A'], paired_df['B'])

    print(f"Paired t-test t = {t_stat:.3f}, p = {p_val:.4f}")

    return {
        't-statistic': t_stat,
        'p-value': p_val,
        'n': len(paired_df),
        'normality_p': p_normality,
        'is_normal': normal
    }

def calculate_segmented_correlations(df_filtered, segment_size=15):
    results = []

    # Tag each pitch with its segment number
    df_filtered['segment'] = ((df_filtered['pitch_index'] - 1) // segment_size) + 1

    # Loop through each segment
    for segment in sorted(df_filtered['segment'].unique()):
        df_segment = df_filtered[df_filtered['segment'] == segment]

        # Group by pitcher and game
        grouped = df_segment.groupby(['player_name', 'game_date'])

        segment_data = []
        for (player, date), group in grouped:
            if len(group) < segment_size:
                continue  # skip incomplete segments

            rmse = np.sqrt(((group[['release_pos_x', 'release_pos_y', 'release_pos_z']] -
                             group[['release_pos_x', 'release_pos_y', 'release_pos_z']].mean())**2).sum(axis=1).mean())
            whiff_rate = group['is_whiff'].sum() / group['is_swing'].sum() if group['is_swing'].sum() > 0 else np.nan

            if not np.isnan(whiff_rate):
                segment_data.append({'player_name': player, 'game_date': date,
                                     'segment': segment, 'rmse': rmse, 'whiff_rate': whiff_rate})

        segment_df = pd.DataFrame(segment_data)
        if len(segment_df) >= 10:  # make sure there's enough data to calculate correlation
            r, p = pearsonr(segment_df['rmse'], segment_df['whiff_rate'])
            results.append({'segment': segment, 'correlation': r, 'p_value': p, 'n_games': len(segment_df)})

    return pd.DataFrame(results)


def one_way_anova(*groups):
    # Perform the ANOVA test
    f_stat, p_value = f_oneway(*groups)

    return f_stat, p_value

def fastball_anova(df_filtered, pos, neg, insig):
    def get_individual_fastball_pct(pitchers):
        results = []
        for pitcher in pitchers:
            pitches = df_filtered[df_filtered['player_name'] == pitcher]['pitch_name']
            total = len(pitches)
            if total == 0:
                continue
            fastballs = (pitches == '4-Seam Fastball').sum()
            pct = 100 * fastballs / total
            results.append(pct)
        return results

    # Get fastball usage for each group
    group_pos = get_individual_fastball_pct(pos)
    group_neg = get_individual_fastball_pct(neg)
    group_insig = get_individual_fastball_pct(insig)

    # Run one-way ANOVA
    f_stat, p_val = f_oneway(group_pos, group_neg, group_insig)

    return {
        'F-statistic': round(f_stat, 4),
        'p-value': round(p_val, 4),
        'group_means': {
            'Significant Positive': round(pd.Series(group_pos).mean(), 2),
            'Significant Negative': round(pd.Series(group_neg).mean(), 2),
            'Insignificant': round(pd.Series(group_insig).mean(), 2)
        }
    }

def fastball_ttests(df_filtered, pos, neg, insig):
    def get_fastball_pct(pitchers):
        return [
            100 * (df_filtered[df_filtered['player_name'] == p]['pitch_name'] == 'Fastball').mean()
            for p in pitchers
            if len(df_filtered[df_filtered['player_name'] == p]) > 0
        ]

    pos_fastball = get_fastball_pct(pos)
    neg_fastball = get_fastball_pct(neg)
    insig_fastball = get_fastball_pct(insig)

    # T-tests
    t_pos_vs_insig = ttest_ind(pos_fastball, insig_fastball, equal_var=False)
    t_neg_vs_insig = ttest_ind(neg_fastball, insig_fastball, equal_var=False)

    return {
        "Positive vs Insignificant": {
            "t-statistic": round(t_pos_vs_insig.statistic, 4),
            "p-value": round(t_pos_vs_insig.pvalue, 4),
            "means": (round(pd.Series(pos_fastball).mean(), 2), round(pd.Series(insig_fastball).mean(), 2))
        },
        "Negative vs Insignificant": {
            "t-statistic": round(t_neg_vs_insig.statistic, 4),
            "p-value": round(t_neg_vs_insig.pvalue, 4),
            "means": (round(pd.Series(neg_fastball).mean(), 2), round(pd.Series(insig_fastball).mean(), 2))
        }
    }



def calculate_avg_abs_release_positions_with_ttests(df_filtered, pos, neg, insig):
    # Helper to extract and normalize release data
    def extract_release_data(pitchers):
        filtered = df_filtered[df_filtered['player_name'].isin(pitchers)].copy()
        filtered['abs_release_pos_x'] = filtered['release_pos_x'].abs()
        return filtered[['abs_release_pos_x', 'release_pos_y', 'release_pos_z']]

    # Extract release data per group
    data = {
        'Significant Positive': extract_release_data(pos),
        'Significant Negative': extract_release_data(neg),
        'Insignificant': extract_release_data(insig)
    }

    # Calculate group means
    mean_df = pd.DataFrame({
        group: df.mean() for group, df in data.items()
    }).round(3)
    mean_df.index = ['|release_pos_x|', 'release_pos_y', 'release_pos_z']

    # Perform t-tests
    features = ['abs_release_pos_x', 'release_pos_y', 'release_pos_z']
    comparisons = [
        ('Significant Positive', 'Significant Negative'),
        ('Significant Positive', 'Insignificant'),
        ('Significant Negative', 'Insignificant')
    ]

    ttest_results = []
    for feature in features:
        for group1, group2 in comparisons:
            stat, pval = ttest_ind(
                data[group1][feature],
                data[group2][feature],
                equal_var=False,
                nan_policy='omit'
            )
            ttest_results.append({
                'Feature': feature,
                'Comparison': f"{group1} vs {group2}",
                't-stat': round(stat, 3),
                'p-value': round(pval, 15)
            })

    ttest_df = pd.DataFrame(ttest_results)

    return mean_df, ttest_df

def calculate_avg_abs_release_positions_with_anova(df_filtered, pos, neg, insig):
    # Helper to extract and normalize release data
    def extract_release_data(pitchers):
        filtered = df_filtered[df_filtered['player_name'].isin(pitchers)].copy()
        filtered['abs_release_pos_x'] = filtered['release_pos_x'].abs()
        return filtered[['abs_release_pos_x', 'release_pos_y', 'release_pos_z']]

    # Extract release data per group
    data = {
        'Significant Positive': extract_release_data(pos),
        'Significant Negative': extract_release_data(neg),
        'Insignificant': extract_release_data(insig)
    }

    # Calculate group means
    mean_df = pd.DataFrame({
        group: df.mean() for group, df in data.items()
    }).round(3)
    mean_df.index = ['|release_pos_x|', 'release_pos_y', 'release_pos_z']

    # Perform one-way ANOVA for each feature across the three groups
    features = ['abs_release_pos_x', 'release_pos_y', 'release_pos_z']
    anova_results = []

    for feature in features:
        samples = [df[feature].dropna() for df in data.values()]
        stat, pval = f_oneway(*samples)
        anova_results.append({
            'Feature': feature,
            'F-statistic': round(stat, 3),
            'p-value': round(pval, 15)
        })

    anova_df = pd.DataFrame(anova_results)

    return mean_df, anova_df