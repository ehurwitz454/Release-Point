import pandas as pd

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(by=['player_name', 'game_date', "at_bat_number", "pitch_number"], ascending=[True, True, True, True])
    df["pitch_index"] = df.groupby(["player_name", "game_date"]).cumcount() + 1
    df = df.sort_values(by=["player_name", "game_date", "pitch_index"])
    return df

def filter_valid_games(df, min_pitch_count=80):
    pitch_counts = df.groupby(['player_name', 'game_date']).size().reset_index(name='pitch_count')
    valid_games = pitch_counts[pitch_counts['pitch_count'] >= min_pitch_count]
    df_filtered = pd.merge(df, valid_games[['player_name', 'game_date']], on=['player_name', 'game_date'])
    return df_filtered

def add_pitch_transitions(df):
    df = df.sort_values(by=['player_name', 'game_date', "at_bat_number", "pitch_number"])
    df["pitch_index"] = df.groupby(["player_name", "game_date", "at_bat_number"]).cumcount() + 1
    df['prev_pitch_type'] = df.groupby(['player_name', 'game_date'])['pitch_name'].shift(1)
    df['prev_x'] = df.groupby(['player_name', 'game_date'])['release_pos_x'].shift(1)
    df['prev_y'] = df.groupby(['player_name', 'game_date'])['release_pos_y'].shift(1)
    df['prev_z'] = df.groupby(['player_name', 'game_date'])['release_pos_z'].shift(1)
    df = df[df['at_bat_number'] == df['at_bat_number'].shift(1)]
    return df

def load_data(file):
    df = pd.read_csv(file)
    return df


