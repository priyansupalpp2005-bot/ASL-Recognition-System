import joblib

encoder = joblib.load("models/label_encoder.pkl")

print("Total Classes:", len(encoder.classes_))
print("\nClasses:")
print(encoder.classes_)