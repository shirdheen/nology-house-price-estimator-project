import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import joblib

# Loading the dataset
df = pd.read_csv('data/melbourne_housing_data.csv')

# Select the relevant columns
df = df[['Suburb', 'Type', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Distance', 'YearBuilt', 'Price']]

# Show basic info
print("Initial data:")
print(df.head()) # Used for previewing data
print("\nMissing values before cleaning:")
print(df.isnull().sum()) # Gives the count of missing values per column

# Drop rows with missing values in the price column
df = df.dropna(subset=['Price'])

# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
        print(f"Filled missing values in {col} with median.")

# Fill missing categorical values with mode
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])
        print(f"Filled missing values in {col} with mode.")

# FEATURE ENGINEERING

# Adding Age (of the house) feature
current_year = datetime.now().year
df['Age'] = current_year - df['YearBuilt']

# Adding Has Garage feature
df['Has_Garage'] = df['Car'].apply(lambda x : 1 if x > 0 else 0)

df = df.drop(columns=['YearBuilt', 'Car'])

# OUTLIER REMOVAL

# Remove the top 2% most expensive houses
price_cap = df['Price'].quantile(0.98)
df = df[df['Price'] < price_cap]
print(f"Outliers removed. Price cap set at: ${price_cap:,.2f}")

# Final check
print("\nMissing values after cleaning:", df.isnull().sum().sum()) # Gets the total number of missing entries in the dataframe (should be 0)
print(f"Final cleaned shape: {df.shape}") # Returns a tuple representing the dimensionality of the DataFrame as (rows, columns)

# Save cleaned version
df.to_csv('data/melbourne_housing_data_cleaned.csv', index=False) # Writes the DataFrame to a CSV file
print("Data has been cleaned and saved.")

# FEATURE SELECTION AND ENCODING

# Loading the cleaned dataset
df = pd.read_csv('data/melbourne_housing_data_cleaned.csv')

# Select features and target
X = df.drop(columns=['Price']) #Features
y = np.log1p(df['Price']) # log(price + 1) - log1p handles zero safely

# One-hot encode the 'Suburb' column (https://www.geeksforgeeks.org/ml-one-hot-encoding/)
# X_encoded = pd.get_dummies(X, columns=['Suburb', 'Type'])

# Label encode suburb and type
# le_suburb = LabelEncoder()
# le_type = LabelEncoder()

# X['Suburb'] = le_suburb.fit_transform(X['Suburb'])
# X['Type'] = le_type.fit_transform(X['Type'])

# Target encode Suburb
suburb_price_map = df.groupby('Suburb')['Price'].mean() # Groups the dataset by suburb and calculates the mean house price for each suburb
X['Suburb'] = df['Suburb'].map(suburb_price_map) # Replaces each row's Suburb string with the average price of that suburb

# Target encoding - it transforms a categorical variable into a numeric one based on its relationship with the target (Price), which gives the model strong predictive power

# Label encode Type
X['Type'] = df['Type'].astype('category').cat.codes # Converts the 'Type' column into a pandas categorical type; then .cat.codes assigns integer codes to each category

# Label encoding would work well here becaue 'Type' has a few categories

# Show the encoded features
print("\nEncoded feature columns:")
print(X.columns)
print(f"\nShape of encoded features: {X.shape}")

# TRAINING & EVALUATION

# Splitting the encoded dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# test_size: represents the proportion of the dataset to include in the test split (0 - 1). Here it is 20% of the data.
# random_state: Controls the shuffling applied to the data before applying the split (for reproducibility)

# MODEL 1: RANDOM FOREST

# -----------------------------------------------------------------------------

# HYPERPARAMETER TUNING PROCESS (commented-out)

# Parameter grid for tuning - tells the RandomizedSearchCV which combinations of hyperparameters to try when training the RandomForestRegressor
# param_dist = {
#     'n_estimators': [100, 200, 300], # Number of trees in the forest (more trees, better performance)
#     'max_depth': [None, 10, 20, 30, 50], # Maximum depth of each tree (deeper trees learn more complex patterns but may overfit)
#     'min_samples_split': [2, 5, 10], # Controls when a node should be split (higher the value, simpler the tree, and less prone to overfitting)
#     'min_samples_leaf': [1, 2, 4], # Minimum samples required at a leaf node
#     'max_features': ['sqrt', 'log2', None] # Number of features considered when looking for the best split (controls randomness of each tree's learning)
# }

# base_rf = RandomForestRegressor(random_state=42) # Creates a base (untuned) RandomForest model that will be passed to RandomizedSearchCV

# # Sets up the hyperparameter tuning process
# random_search = RandomizedSearchCV(
#     estimator=base_rf, # Model you want to tune
#     param_distributions=param_dist, # Parameters ranges to sample from
#     n_iter=40, # Try 30 random combinations from the grid
#     cv=3, # Use 3-fold cross-validation for evaluation each combination
#     verbose=1, # Print progress in the terminal
#     n_jobs=-1, # Use all available CPU cores to run faster
#     scoring='neg_mean_absolute_error', # Use MAE for scoring(negated, as scikit-learn maximises scores)
#     random_state=42 # Makes the randomness reproducible
#     )

# # Run search
# print("\n Running hyperparameter tuning...")
# random_search.fit(X_train, y_train)

# best_rf = random_search.best_estimator_
# print(f"\n Best Parameters: {random_search.best_params_}")

# -----------------------------------------------------------------------------

best_rf = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=20,
    random_state=42
)

best_rf.fit(X_train, y_train)

# Evaluation
y_pred_rf = np.expm1(best_rf.predict(X_test))
y_test_actual = np.expm1(y_test)
mae_rf = mean_absolute_error(y_test_actual, y_pred_rf)

print(f"\nRandom Forest Results:")
print(f"Mean Absolute Error: ${mae_rf:,.2f}")

joblib.dump(best_rf, 'models/final_rf_model.pkl')
print("Model saved!")