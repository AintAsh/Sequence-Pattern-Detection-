from tensorflow.keras.models import load_model

loaded_rnn = load_model("pattern_rnn_model.h5")


import joblib

loaded_scaler = joblib.load("scaler_pattern.pkl")


import numpy as np

# example sequence
test_seq = np.array([[0, 1, 0, 1, 0]])

# scale using loaded scaler
test_seq_scaled = loaded_scaler.transform(test_seq)

# reshape for RNN
test_seq_scaled = test_seq_scaled.reshape(1, 5, 1)

# predict
prediction = loaded_rnn.predict(test_seq_scaled)

print("Prediction probability:", prediction)
