import joblib

feature_names = joblib.load("models/feature_names.pkl")

print("Type:", type(feature_names))
print("Length:", len(feature_names))
print("\nFeatures:\n")
print(feature_names)