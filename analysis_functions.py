import numpy as np
import pandas as pd
from itertools import product
import statsmodels.formula.api as smf

def calculate_rmse(df_filtered):
    group_means = df_filtered.groupby(['player_name', 'game_date'])[['release_pos_x', 'release_pos_y', 'release_pos_z']].transform('mean')
    squared_diffs = (df_filtered[['release_pos_x', 'release_pos_y', 'release_pos_z']] - group_means) ** 2
    df_filtered['squared_error'] = squared_diffs.sum(axis=1)
    rmse_df = df_filtered.groupby(['player_name', 'game_date'])['squared_error'].mean().apply(np.sqrt).reset_index()
    rmse_df.rename(columns={'squared_error': 'release_point_rmse'}, inplace=True)
    return rmse_df
def eligible_pitchers(df_filtered, game_counts):
    games = game_counts[game_counts['game_count'] >= 12]['player_name']
    df = df_filtered[df_filtered['player_name'].isin(games)]
    summary = df.groupby('player_name').agg({
        'release_point_rmse': 'mean',
        'whiff_rate': 'mean'
    }).reset_index()
    return summary
def calculate_metrics(df):
    hit_values = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
    df['is_hit'] = df['events'].isin(hit_values)
    df['bases'] = df['events'].map(hit_values).fillna(0)
    df['is_at_bat'] = df['events'].isin([
        'field_out', 'strikeout', 'grounded_into_double_play', 'single',
        'double', 'triple', 'home_run', 'force_out', 'field_error',
        'fielders_choice', 'fielders_choice_out', 'strikeout_double_play',
        'other_out', 'double_play', 'triple_play', 'sac_bunt',
        'sac_bunt_double_play'
    ])
    df['is_pa'] = df['is_at_bat'] | df['events'].isin([
        'walk', 'hit_by_pitch', 'catcher_interf', 'sac_fly_double_play', 'sac_fly'
    ])
    df['is_bb'] = df['events'] == 'walk'
    df['is_k'] = df['events'].isin(['strikeout', 'strikeout_double_play'])
    df['is_hbp'] = df['events'] == 'hit_by_pitch'

    metrics = df.groupby(['player_name']).agg(
        hits=('is_hit', 'sum'),
        at_bats=('is_at_bat', 'sum'),
        total_bases=('bases', 'sum'),
        plate_appearances=('is_pa', 'sum'),
        walks=('is_bb', 'sum'),
        strikeouts=('is_k', 'sum'),
        hbp=('is_hbp', 'sum')
    ).reset_index()

    metrics['BAA'] = metrics['hits'] / metrics['at_bats']
    metrics['OBP_against'] = (metrics['hits'] + metrics['walks'] + metrics['hbp']) / metrics['plate_appearances']
    metrics['SLG_against'] = metrics['total_bases'] / metrics['at_bats']
    metrics['K_percent'] = metrics['strikeouts'] / metrics['plate_appearances']
    metrics['BB_percent'] = metrics['walks'] / metrics['plate_appearances']
    metrics['OPS_against'] = metrics['OBP_against'] + metrics['SLG_against']
    metrics = metrics.fillna(0)
    return metrics

def calculate_whiff_rate(df):
    # Define whiff and swing descriptions
    whiff_descriptions = ['swinging_strike', 'swinging_strike_blocked']
    swing_descriptions = [
        'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip',
        'hit_into_play', 'missed_bunt'
    ]

    # Create flags for whiffs and swings
    df['is_whiff'] = df['description'].isin(whiff_descriptions)
    df['is_swing'] = df['description'].isin(swing_descriptions)

    # Group by player and game to calculate whiff rate
    whiff_rate_df = df.groupby(['player_name', 'game_date']).agg(
        whiffs=('is_whiff', 'sum'),
        swings=('is_swing', 'sum')
    ).reset_index()

    whiff_rate_df['whiff_rate'] = whiff_rate_df['whiffs'] / whiff_rate_df['swings']
    whiff_rate_df = whiff_rate_df.fillna(0)  # Handle potential divide-by-zero
    return whiff_rate_df

def calculate_segmented_rmse(df_filtered, pitch_num):
    # Tag each pitch as in 'First 25' or 'Rest'
    df_filtered['pitch_segment'] = np.where(df_filtered['pitch_index'] <= pitch_num, 'First 25', 'Rest')

    # Calculate squared error from mean release point
    group_means = df_filtered.groupby(['player_name', 'game_date'])[['release_pos_x', 'release_pos_y', 'release_pos_z']].transform('mean')
    squared_diffs = (df_filtered[['release_pos_x', 'release_pos_y', 'release_pos_z']] - group_means) ** 2
    df_filtered['squared_error'] = squared_diffs.sum(axis=1)

    # Compute RMSE per pitcher-game-segment
    rmse_segment = (
        df_filtered.groupby(['player_name', 'game_date', 'pitch_segment'])['squared_error']
        .mean().apply(np.sqrt).reset_index()
        .rename(columns={'squared_error': 'release_point_rmse'})
    )

    # Compute Whiff Rate per segment
    whiff_rate_segment = (
        df_filtered.groupby(['player_name', 'game_date', 'pitch_segment'])
        .agg(total_pitches=('is_whiff', 'count'), total_whiffs=('is_whiff', 'sum'))
        .reset_index()
    )
    whiff_rate_segment['whiff_rate'] = whiff_rate_segment['total_whiffs'] / whiff_rate_segment['total_pitches']

    # Merge RMSE and whiff rate
    segment_summary = pd.merge(rmse_segment, whiff_rate_segment, on=['player_name', 'game_date', 'pitch_segment'])

    # Split into first 25 and rest
    first_25 = segment_summary[segment_summary['pitch_segment'] == 'First 25'].copy()
    rest = segment_summary[segment_summary['pitch_segment'] == 'Rest'].copy()

    return first_25, rest
def calculate_game_count(df):
    # Count the number of games each player has participated in
    game_counts = df.groupby('player_name')['game_date'].nunique().reset_index(name='game_count')
    return game_counts

def calculate_release_diff(df):
    df['release_diff'] = np.sqrt(
        (df['release_pos_x'] - df['prev_x'])**2 +
        (df['release_pos_y'] - df['prev_y'])**2 +
        (df['release_pos_z'] - df['prev_z'])**2
    )
    return df


def run_logit_release_whiff(df):
    whiff_descriptions = ['swinging_strike', 'swinging_strike_blocked']
    swing_descriptions = ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play']

    df['is_whiff'] = df['description'].isin(whiff_descriptions).astype(int)
    df['is_swing'] = df['description'].isin(swing_descriptions).astype(int)

    results = []
    pitch_names = df['pitch_name'].dropna().unique()

    for p1, p2 in product(pitch_names, repeat=2):
        if p1 == p2:
            continue

        df_pair = df[(df['pitch_name'] == p2) & (df['prev_pitch_type'] == p1)].copy()
        df_pair = df_pair.dropna(subset=[
            'release_pos_x', 'release_pos_y', 'release_pos_z',
            'prev_x', 'prev_y', 'prev_z', 'is_whiff'
        ])

        if len(df_pair) < 100:
            continue

        df_pair['release_diff'] = np.sqrt(
            (df_pair['release_pos_x'] - df_pair['prev_x']) ** 2 +
            (df_pair['release_pos_y'] - df_pair['prev_y']) ** 2 +
            (df_pair['release_pos_z'] - df_pair['prev_z']) ** 2
        )

        model_df = df_pair[['release_diff', 'is_whiff']].dropna()
        if model_df['is_whiff'].nunique() < 2:
            continue

        try:
            model = smf.logit('is_whiff ~ release_diff', data=model_df).fit(disp=0)
            results.append({
                'From': p1, 'To': p2,
                'Coef': model.params['release_diff'],
                'P-Value': model.pvalues['release_diff'],
                'N': len(model_df)
            })
        except Exception as e:
            print(f"Skipping {p1} -> {p2} due to error: {e}")

    return pd.DataFrame(results).sort_values(by='P-Value')


def compute_game_level_metrics(df):
    # Compute RMSE per game
    df['release_vector'] = list(zip(df.release_pos_x, df.release_pos_y, df.release_pos_z))

    game_groups = df.groupby(['player_name', 'game_date'])

    records = []
    for (player, date), group in game_groups:
        if len(group) < 80:
            continue

        release_array = np.array(list(group['release_vector']))
        rmse = np.sqrt(((release_array - release_array.mean(axis=0)) ** 2).mean())

        whiff_rate = group['is_whiff'].sum() / group['is_swing'].sum() if group['is_swing'].sum() > 0 else np.nan

        records.append({
            'player_name': player,
            'game_date': date,
            'rmse': rmse,
            'whiff_rate': whiff_rate,
            'num_pitches': len(group)
        })

    return pd.DataFrame(records).dropna()

def calculate_pitch_mix_and_spin_rate_by_group(df_filtered, pos, neg, insig):
    # Helper function to compute pitch distribution
    def get_pitch_distribution(pitchers, label):
        filtered = df_filtered[df_filtered['player_name'].isin(pitchers)]
        pitch_mix = (
            filtered['pitch_name']
            .value_counts(normalize=True)
            .sort_values(ascending=False) * 100
        )
        return pd.DataFrame({label: pitch_mix}).round(2)

    # Helper function to compute average spin rate by pitch type
    def get_spin_rate_by_pitch(pitchers, label):
        filtered = df_filtered[df_filtered['player_name'].isin(pitchers)]
        spin_rates = (
            filtered.groupby('pitch_name')['release_spin_rate']
            .mean()
            .round(2)
            .rename(label)
        )
        return pd.DataFrame(spin_rates)

    # Get pitch mix distributions
    dist_pos = get_pitch_distribution(pos, 'Significant Positive')
    dist_neg = get_pitch_distribution(neg, 'Significant Negative')
    dist_insig = get_pitch_distribution(insig, 'Insignificant')

    # Combine pitch mix tables
    pitch_mix_table = pd.concat([dist_pos, dist_neg, dist_insig], axis=1).fillna(0)

    # Get spin rates
    spin_pos = get_spin_rate_by_pitch(pos, 'Significant Positive')
    spin_neg = get_spin_rate_by_pitch(neg, 'Significant Negative')
    spin_insig = get_spin_rate_by_pitch(insig, 'Insignificant')

    # Combine spin rate tables
    spin_rate_table = pd.concat([spin_pos, spin_neg, spin_insig], axis=1).fillna(0)

    return pitch_mix_table, spin_rate_table

def calculate_lefty_percentage_by_group(df_filtered, pos, neg, insig):
    def lefty_pct(pitchers):
        filtered = df_filtered[df_filtered['player_name'].isin(pitchers)]
        # Drop duplicates to ensure we're counting unique pitchers
        unique_pitchers = filtered[['player_name', 'p_throws']].drop_duplicates()
        lefties = unique_pitchers[unique_pitchers['p_throws'] == 'L']
        return round(100 * len(lefties) / len(unique_pitchers), 2) if len(unique_pitchers) > 0 else 0.0

    return {
        'Significant Positive': lefty_pct(pos),
        'Significant Negative': lefty_pct(neg),
        'Insignificant': lefty_pct(insig)
    }

def calculate_avg_release_position_by_group(df_filtered, pos, neg, insig):
    def avg_release(pitchers):
        filtered = df_filtered[df_filtered['player_name'].isin(pitchers)]
        return (
            filtered[['release_pos_x', 'release_pos_y', 'release_pos_z']]
            .mean()
            .round(3)
        )

    return pd.DataFrame({
        'Significant Positive': avg_release(pos),
        'Significant Negative': avg_release(neg),
        'Insignificant': avg_release(insig)
    })

def calculate_avg_release_position_by_group_and_side(df_filtered, pos, neg, insig):
    def avg_release(pitchers, side):
        filtered = df_filtered[df_filtered['player_name'].isin(pitchers)]
        side_filtered = filtered[filtered['release_pos_x'] > 0] if side == "RHP" else filtered[filtered['release_pos_x'] < 0]
        return (
            side_filtered[['release_pos_x', 'release_pos_y', 'release_pos_z']]
            .mean()
            .round(3)
        )

    groups = {
        'Significant Positive': pos,
        'Significant Negative': neg,
        'Insignificant': insig
    }

    results = {}
    for label, group in groups.items():
        results[f'{label} - RHP (x > 0)'] = avg_release(group, "RHP")
        results[f'{label} - LHP (x < 0)'] = avg_release(group, "LHP")

    return pd.DataFrame(results)