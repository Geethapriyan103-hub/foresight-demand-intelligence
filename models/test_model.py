import joblib
print ("loading model feature")
x = joblib.load("models/model_features.pkl")
print ("Type:".type(x))
print ("Content:"x)