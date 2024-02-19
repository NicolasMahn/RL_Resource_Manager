from tensorflow import keras
from keras import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Lambda


def create_dqn_model(input_shape, num_actions, alpha):
    model = Sequential([
        Input(shape=input_shape),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Flatten(),
        Dense(num_actions, activation="linear")
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=alpha), metrics=['accuracy'])
    return model


def create_dueling_dqn_model(input_shape, num_actions, alpha):
    """
    Create a model with a value and an advantage stream outputs.
    """
    # Define the inputs
    inputs = Input(shape=input_shape)

    # Common base network
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)

    # Value stream
    value_stream = Dense(1, activation=None)(x)
    value_stream = Lambda(lambda s: keras.backend.expand_dims(s[:, 0], -1),
                                       output_shape=(num_actions,))(value_stream)

    # Advantage stream
    advantage_stream = Dense(num_actions, activation=None)(x)
    advantage_stream = Lambda(lambda a: a - keras.backend.mean(a, keepdims=True),
                                           output_shape=(num_actions,))(advantage_stream)

    # Combine streams into Q-values
    q_values = keras.layers.Add()([value_stream, advantage_stream])

    # Create the model
    model = Model(inputs=inputs, outputs=q_values)
    model.compile(loss='mse', optimizer=Adam(learning_rate=alpha), metrics=['accuracy'])

    return model


def create_actor_model(input_shape, num_actions, alpha):
    model = keras.Sequential([
        Input(shape=input_shape),
        Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(num_actions, activation="softmax")
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=alpha))
    return model


def create_critic_model(input_shape, alpha):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="linear")
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=alpha))
    return model