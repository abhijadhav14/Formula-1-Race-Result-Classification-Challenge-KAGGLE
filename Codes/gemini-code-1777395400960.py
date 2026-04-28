import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# 1. Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Prepare Features (X) and Target (y)
# Drop 'id' as it's a unique identifier, and 'finishing_position' as it's our target.
X_train = train_df.drop(columns=['id', 'finishing_position'])
y_train = train_df['finishing_position']

X_test = test_df.drop(columns=['id'])

# 3. Initialize the Model
# HistGradientBoosting handles missing values natively and trains incredibly fast.
model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_depth=7,
    random_state=42
)

# 4. Train the Model
print("Training the model...")
model.fit(X_train, y_train)

# 5. Make Predictions
print("Making predictions on the test set...")
predictions = model.predict(X_test)

# 6. Post-Process Predictions
# Finishing positions are discrete rankings (1st, 2nd, 3rd...). 
# We round the continuous predictions and clip them to stay within valid bounds (1 to 39).
predictions_rounded = np.clip(np.round(predictions), 1, 39).astype(int)

# 7. Create Submission File
submission = pd.DataFrame({
    'id': test_df['id'],
    'finishing_position': predictions_rounded
})

# Save to CSV for Kaggle submission
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv successfully!")