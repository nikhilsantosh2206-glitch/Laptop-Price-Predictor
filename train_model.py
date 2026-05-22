import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("laptop_data.csv")

# Encoding categorical columns
brand_encoder = LabelEncoder()
processor_encoder = LabelEncoder()

df['Brand'] = brand_encoder.fit_transform(df['Brand'])
df['Processor'] = processor_encoder.fit_transform(df['Processor'])

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor()

# Train
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model/laptop_price_model.pkl", "wb"))
pickle.dump(brand_encoder, open("model/brand_encoder.pkl", "wb"))
pickle.dump(processor_encoder, open("model/processor_encoder.pkl", "wb"))

print("Model Trained Successfully")