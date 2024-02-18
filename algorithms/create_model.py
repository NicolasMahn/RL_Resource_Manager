from tensorflow import keras


def create_dqn_model(input_shape, num_actions):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(num_actions, activation="linear")
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model
