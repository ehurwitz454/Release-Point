from data_preprocessing import load_and_preprocess_data, filter_valid_games, add_pitch_transitions, load_data
from analysis_functions import calculate_rmse, calculate_metrics, calculate_whiff_rate, calculate_segmented_rmse, calculate_game_count, calculate_release_diff, run_logit_release_whiff, compute_game_level_metrics, eligible_pitchers, calculate_pitch_mix_and_spin_rate_by_group, calculate_lefty_percentage_by_group, calculate_avg_release_position_by_group, calculate_avg_release_position_by_group_and_side
from visualization import plot_rmse_vs_ops, plot_whiff_rate_vs_rmse, plot_logit_results, plot_rmse_vs_whiff, plot_regression_line, plot_eligible_whiff, plot_player, qq_plot, plot_release_positions_3d, plot_release_pos_xz
from statistical_analysis import perform_t_test, perform_regression, calculate_correlation, run_linear_regression, safe_paired_ttest, calculate_segmented_correlations, one_way_anova, fastball_anova, fastball_ttests, calculate_avg_abs_release_positions_with_ttests,calculate_avg_abs_release_positions_with_anova
import pandas as pd

# Load and preprocess data
df = load_and_preprocess_data('Statcast_2021.csv')
df_filtered = filter_valid_games(df)

# Calculate RMSE, metrics, and whiff rate
rmse_df = calculate_rmse(df_filtered)
metrics = calculate_metrics(df)
whiff_rate_df = calculate_whiff_rate(df_filtered)

# Calculate game count
game_count_df = calculate_game_count(df)

# Merge game count with metrics
metrics = metrics.merge(game_count_df, on='player_name', how='left')

# Calculate segmented RMSE
first_25_rmse, rest_rmse = calculate_segmented_rmse(df_filtered, 25) #for whiff rate late in game
first_40_rmse, rest2_rmse = calculate_segmented_rmse(df_filtered, 40) #for fatigue paired t-test

# Merge dataframes
merged_df = metrics.merge(rmse_df, on='player_name', how='inner')
merged_df = merged_df.merge(whiff_rate_df, on=['player_name', 'game_date'], how='inner')
plot_eligible_whiff(merged_df, game_count_df)
merged_df2 = merged_df[['player_name', 'game_date']]
first_25_rmse = first_25_rmse.merge(merged_df2, on = ['player_name', 'game_date'], how='right')
rest_rmse = rest_rmse.merge(merged_df2, on = ['player_name', 'game_date'], how='right')
# Plot and analyze
el_pitchers = metrics[metrics['game_count'] >= 12]['player_name']
plot_rmse_vs_ops(merged_df, el_pitchers)
merged_df3 = eligible_pitchers(merged_df, game_count_df)
model_pitch = run_linear_regression(merged_df3["release_point_rmse"], merged_df3["whiff_rate"])
print("\n Linear Regression of Mean RMSE vs Whiff Rate (Eligible Pitchers) (Full Season)")
print(model_pitch.summary())

# Perform statistical analysis
print(f"\n Mean Consistency RMSE for First 40 Pitches: {first_40_rmse["release_point_rmse"].mean()}. Variance: {first_40_rmse["release_point_rmse"].var()}")
print(f"\n Mean Consistency RMSE Pitches 41: {rest2_rmse["release_point_rmse"].mean()}. Variance: {rest2_rmse["release_point_rmse"].var()}")


t_stat, p_value = perform_t_test(first_40_rmse['release_point_rmse'], rest2_rmse['release_point_rmse'])
print(f"Paired t-test results:\n t-statistic = {t_stat:.4f}, p-value = {p_value:.20f}")

# Calculate correlation between RMSE and whiff rate
correlation_whiff_df = calculate_correlation(merged_df)
print(correlation_whiff_df.shape)
correlation_whiff_df_s = correlation_whiff_df[correlation_whiff_df['p_value'] < 0.15]
# Sort correlation_whiff_df by highest positive and highest negative correlations
sorted_positive_correlation_whiff = correlation_whiff_df_s.sort_values(by='correlation', ascending=False)
sorted_negative_correlation_whiff = correlation_whiff_df_s.sort_values(by='correlation', ascending=True)
sorted_insignificant_correlations = correlation_whiff_df.sort_values(by= 'p_value', ascending = False)
correlations_insig = correlation_whiff_df[correlation_whiff_df['p_value'] > 0.85]
correlations_pos = correlation_whiff_df_s[correlation_whiff_df_s['correlation'] > 0] #significant positive
correlations_neg = correlation_whiff_df_s[correlation_whiff_df_s['correlation'] < 0] #significant negative
print(correlations_insig.shape)
print(correlations_pos.shape)
print(correlations_neg.shape)
pos = calculate_avg_release_position_by_group(df_filtered, correlations_pos["player_name"], correlations_neg["player_name"], correlations_insig["player_name"])
pos2 = calculate_avg_release_position_by_group_and_side(df_filtered, correlations_pos["player_name"], correlations_neg["player_name"], correlations_insig["player_name"])
pos3, pos4 = calculate_avg_abs_release_positions_with_ttests(df_filtered, correlations_pos["player_name"], correlations_neg["player_name"], correlations_insig["player_name"])
print(pos3)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(pos4)
mean2, anova = calculate_avg_abs_release_positions_with_anova(df_filtered, correlations_pos["player_name"], correlations_neg["player_name"], correlations_insig["player_name"])
print(mean2)
print(anova)
pos2 =  pos2.to_dict()
plot_release_pos_xz(pos2)
pitch_mix, spin_rate = calculate_pitch_mix_and_spin_rate_by_group(df_filtered, correlations_pos["player_name"], correlations_neg["player_name"], correlations_insig["player_name"])
print("\n Pitch Mixes For Each Group")
print(pitch_mix)
print("\n Spin Rates by Pitch For Each Group")
print(spin_rate)
# Display top 5 positive and negative correlations for whiff rate
print("\nTop 5 Positive Correlations (Whiff Rate) (Significant at 10%):")
print(sorted_positive_correlation_whiff[['player_name', 'correlation', 'p_value', 'n_games']].head(5))

print("\nTop 5 Negative Correlations (Whiff Rate) (Significant at 10%):")
print(sorted_negative_correlation_whiff[['player_name', 'correlation', 'p_value', 'n_games']].head(5))

print("\n 5 Most Insignificant Correlations")
print(sorted_insignificant_correlations[['player_name', 'correlation', 'p_value', 'n_games']].head(5))

plot_rmse_vs_whiff(merged_df)

# Full game regression
model_full = run_linear_regression(merged_df['release_point_rmse'], merged_df['whiff_rate'])
print("Full Game Regression Summary:")
print(model_full.summary())

# First 25 pitches
model_first25 = run_linear_regression(first_25_rmse['release_point_rmse'], first_25_rmse['whiff_rate'])
print("First 25 Pitches Regression Summary:")
print(model_first25.summary())

# Pitches 26+
model_rest = run_linear_regression(rest_rmse['release_point_rmse'], rest_rmse['whiff_rate'])
print("Rest of Game Regression Summary:")
print(model_rest.summary())

# Plot regression for pitches 26+
plot_regression_line(rest_rmse['release_point_rmse'], rest_rmse['whiff_rate'], model_rest,
                     title="Regression: RMSE vs. Whiff Rate (Pitches 26+)")

df_p = load_data('Statcast_2021.csv')
df_p = add_pitch_transitions(df_p)
results_df = run_logit_release_whiff(df_p)

print(results_df[['From', 'To', 'Coef', 'P-Value', 'N']])
plot_logit_results(results_df)

plot_player(merged_df, "Singer, Brady")
plot_player(merged_df, "Luzardo, Jesús")

segments = calculate_segmented_correlations(df_filtered, 25)
print("\n RMSE to Whiff Rate Correlations Over Game Segments")
print(segments)