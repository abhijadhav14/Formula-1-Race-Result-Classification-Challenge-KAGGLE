import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# 1. Load Data
train_df = pd.read_csv('/kaggle/input/competitions/formula-1-race-result-classification-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/competitions/formula-1-race-result-classification-challenge/test.csv')

# 2. Advanced Feature Engineering
def engineer_f1_features(df):
    df = df.copy()
    
    # Existing Momentum & Dominance Features
    df['rolling_avg_position'] = df['rolling_avg_position'].fillna(df['grid'])
    df['expected_performance_gap'] = df['grid'] - df['rolling_avg_position']
    
    team_grid_avg = df.groupby(['raceId', 'constructorId'])['grid'].transform('mean')
    df['beat_teammate_qualifying'] = (df['grid'] < team_grid_avg).astype(int)
    df['career_efficiency'] = df['career_wins_so_far'] / (df['driver_age'] + 0.1)
    
    # NEW: Track characteristics
    # Helps the model learn if a track is a high-speed straight or a winding street circuit
    df['track_complexity'] = df['number_of_turns'] / (df['track_length_km'] + 0.1)
    
    # NEW: Data Availability Flags (Helps the model distinguish eras of F1)
    df['has_qualifying'] = df['best_qual_ms'].notna().astype(int)
    df['has_pit_data'] = df['pit_stop_count'].notna().astype(int)
    
    return df

X_train_raw = engineer_f1_features(train_df)
X_test_raw = engineer_f1_features(test_df)

# 3. Categorical Casting
# We MUST tell the model these are labels, not mathematical numbers
categorical_cols = ['driverId', 'constructorId', 'circuitId', 
                    'driver_nationality_code', 'constructor_nationality_code']

for col in categorical_cols:
    X_train_raw[col] = X_train_raw[col].astype('category')
    X_test_raw[col] = X_test_raw[col].astype('category')

# Prepare final Training data
X_train = X_train_raw.drop(columns=['id', 'finishing_position'])
y_train = X_train_raw['finishing_position']
X_test = X_test_raw.drop(columns=['id'])

# 4. Train LightGBM with the Optimal Parameters from your Optuna run
print("Training Categorical-Aware LightGBM...")
model = LGBMRegressor(
    n_estimators=251,           # Best max_iter from your previous run
    learning_rate=0.1,
    max_depth=10,
    min_child_samples=31,       # Sklearn's min_samples_leaf equivalent
    num_leaves=17,              # Sklearn's max_leaf_nodes equivalent
    reg_lambda=5.0,             # Sklearn's l2_regularization equivalent
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, categorical_feature=categorical_cols)

# 5. Make Predictions
print("Making raw predictions...")
raw_predictions = model.predict(X_test)

# Add predictions to a temporary dataframe to rank them
post_process_df = pd.DataFrame({
    'id': test_df['id'],
    'raceId': test_df['raceId'],
    'raw_pred': raw_predictions
})

# 6. THE RANKING HACK: Rank drivers strictly 1 to 20 within their specific race!
print("Applying Within-Race Ranking post-processing...")
post_process_df['ranked_position'] = post_process_df.groupby('raceId')['raw_pred'].rank(method='first')

# 7. Generate Final Submission
submission = pd.DataFrame({
    'id': post_process_df['id'],
    'finishing_position': post_process_df['ranked_position'].astype(int)
})

submission.to_csv('submission_ranked.csv', index=False)
print("Successfully saved 'submission_ranked.csv'!")
