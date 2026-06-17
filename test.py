import joblib

model = joblib.load("models/asl_model.pkl")

print(type(model))
print(model.n_features_in_)

encoder = joblib.load("models/label_encoder.pkl")
print(encoder.classes_)