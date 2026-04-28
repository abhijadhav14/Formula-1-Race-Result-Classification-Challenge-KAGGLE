import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# 1. Load Data
train_df = pd.read_csv('/kaggle/input/competitions/formula-1-race-result-classification-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/competitions/formula-1-race-result-classification-challenge/test.csv')

# 2. Advanced F1 Relative Feature Engineering
def engineer_f1_features(df):
    df = df.copy()
    
    # Grid Delta
    df['rolling_avg_position'] = df['rolling_avg_position'].fillna(df['grid'])
    df['expected_performance_gap'] = df['grid'] - df['rolling_avg_position']
    
    # Teammate Dominance
    team_grid_avg = df.groupby(['raceId', 'constructorId'])['grid'].transform('mean')
    df['beat_teammate_qualifying'] = (df['grid'] < team_grid_avg).astype(int)
    df['career_efficiency'] = df['career_wins_so_far'] / (df['driver_age'] + 0.1)
    
    # === THE NEW GAME-CHANGING FEATURES ===
    
    # 1. Qualifying Pace Ratio (Driver Time / Fastest Time in that specific race)
    race_min_best_qual = df.groupby('raceId')['best_qual_ms'].transform('min')
    df['best_qual_pace_ratio'] = df['best_qual_ms'] / (race_min_best_qual + 1)
    # If they didn't set a time, penalize them by assuming they were 5% slower than the leader
    df['best_qual_pace_ratio'] = df['best_qual_pace_ratio'].fillna(1.05) 
    
    # 2. Driver Championship Dominance (Share of Total Grid Points)
    total_race_points = df.groupby('raceId')['prev_championship_points'].transform('sum')
    df['driver_champ_share'] = df['prev_championship_points'] / (total_race_points + 0.1)
    
    # 3. Constructor Championship Dominance
    total_constructor_points = df.groupby('raceId')['prev_constructor_points'].transform('sum')
    df['constructor_champ_share'] = df['prev_constructor_points'] / (total_constructor_points + 0.1)
    
    return df

# Apply Features
X_train_raw = engineer_f1_features(train_df)
X_test_raw = engineer_f1_features(test_df)

# Prepare Training data
drop_cols = ['id', 'finishing_position']
X_train = X_train_raw.drop(columns=drop_cols)
y_train = X_train_raw['finishing_position']
X_test = X_test_raw.drop(columns=['id'])

# 3. Train HistGradientBoostingRegressor
# We use the parameters that got you your best score, tuned slightly for the new features
print("Training model with Relative Pace & Championship Shares...")
model = HistGradientBoostingRegressor(
    max_iter=350, 
    learning_rate=0.05, 
    max_depth=9, 
    l2_regularization=2.0,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Make Predictions
print("Making predictions...")
predictions = model.predict(X_test)

# We return to Rounding to minimize DNF penalty
predictions_rounded = np.clip(np.round(predictions), 1, 39).astype(int)

# 5. Generate Final Submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'finishing_position': predictions_rounded
})

submission.to_csv('submission_relative_features.csv', index=False)
print("Successfully saved 'submission_relative_features.csv'!")
