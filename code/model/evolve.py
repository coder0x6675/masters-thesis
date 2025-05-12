#!/usr/bin/env python
# Script that builds and tests hyperparameters for lightning strike prediction.

import os
import gc
import sys
import json
import time
import random
import pickle
import logging
import statistics
import multiprocessing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy
import pandas
import tensorflow
#tensorflow.compat.v1.disable_eager_execution()
import keras
keras.utils.set_random_seed(0)


VERSION = "1.0.0"
FOLD_COUNT = 10
LOOKBACK = 72

# ---

logging.basicConfig(
    level = logging.ERROR,
    datefmt = "%Y-%m-%d %H:%M:%S",
    format = "[%(asctime)s] %(levelname)s: %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evolve.log", mode="w", encoding="utf-8"),
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug(f"Script v{VERSION}")

# ---

def generate_timeseries(df_light, length=None):
    length = length or LOOKBACK
    x = pandas.merge(
        df_mesan, df_light,
        left_on  = "light.index",
        right_on = "index",
        how      = "inner",
    )[df_mesan.columns]
    x = x.groupby("light.index").tail(length)
    x = [group.drop("light.index", axis=1).to_numpy() for _, group in x.groupby("light.index")]
    y = df_light["lightning"]
    return numpy.array(x), numpy.array(y)

logger.debug("Loading MESAN data")
df_mesan = pandas.read_csv("./df_mesan.csv")
df_mesan.drop("timestamp", axis=1, inplace=True)

logger.debug("Loading LIGHT data")
paths = [f"df_light/df_light_part_{i:02}.csv" for i in range(FOLD_COUNT)]
with multiprocessing.Pool() as pool:
    df_lights = pool.map(pandas.read_csv, paths)
    folds = pool.map(generate_timeseries, df_lights)

# ---

class Candidate:

    MUTATION_RATE = 0
    TRAITS = {
        "model": ["lstm"],
        "batch_size": (32, 64, 128, 256, 512),
        "optimizer": ("adam", "sgd"),
        "validation_split": (0.1, 0.2, 0.3, 0.4),
        "loss": ["binary_crossentropy"],
        "r_activation": ("relu", "sigmoid", "tanh"),
        "d_activation": ("relu", "exponential", "sigmoid", "tanh"),
        "epochs": (8, 16),
        "r_dropout": (0.1, 0.2, 0.3, 0.4),
        "d_dropout": (0.1, 0.2, 0.3, 0.4),
        "r_units": (16, 32, 64, 128, 256, 512),
        "d_units": (8, 16, 32, 64, 128, 256, 512),
        "r_layers": (1, 2, 4, 8),
        "d_layers": (0, 1, 2, 4),
        "r_normalization": (True, False),
        "d_normalization": (True, False),
        "out_activ": ["sigmoid"],
    }

    """ Commented out to limit the testing scope
    TRAITS = {
        "model": ("rnn", "gru", "lstm"),
        "batch_size": (32, 64, 128, 256, 512),
        "optimizer": ("rmsprop", "adagrad", "adamax", "adam", "nadam", "sgd"),
        "validation_split": (0.1, 0.2, 0.3, 0.4),
        "loss": ("mean_squared_error", "log_cosh", "kl_divergence", "mean_squared_logarithmic_error", "mean_absolute_error", "mean_absolute_percentage_error", "binary_crossentropy", "squared_hinge"),
        "r_activation": ("linear", "relu", "elu", "exponential", "selu", "sigmoid", "tanh"),
        "d_activation": ("linear", "relu", "elu", "exponential", "selu", "sigmoid", "tanh"),
        "epochs": (2, 4, 8, 12, 16),
        "r_dropout": (0.1, 0.2, 0.3, 0.4),
        "d_dropout": (0.1, 0.2, 0.3, 0.4),
        "r_units": (16, 32, 64, 128, 256, 512),
        "d_units": (8, 16, 32, 64, 128, 256, 512),
        "r_layers": (1, 2, 4, 8),
        "d_layers": (0, 1, 2, 4),
        "r_normalization": (True, False),
        "d_normalization": (True, False),
        "out_activ": ("sigmoid", "tanh"),
    }
    """

    def __init__(self, traits=None):
        if traits:
            self.traits = traits
        else:
            self.traits = {}
            for trait, values in self.TRAITS.items():
                self.traits[trait] = random.choice(values)

    def __str__(self):
        return json.dumps(self.traits)

    def __repr__(self):
        return "Candidate(" + str(self.__hash__()) + ")"

    def __add__(self, other):
        offspring_traits = {}
        for trait in self.traits:
            choices = (self.traits[trait], other.traits[trait], random.choice(self.TRAITS[trait]))
            offspring_traits[trait] = random.choices(choices, weights=(1,1,self.MUTATION_RATE), k=1)[0]
        return Candidate(offspring_traits)

    def __eq__(self, other):
        return self.traits == other.traits

    def __hash__(self):
        return hash(frozenset(self.traits))


def build_model(training_data, candidate):

    input_shape = (training_data.shape[1], training_data.shape[2])
    normalization_layer = keras.layers.Normalization()
    normalization_layer.adapt(training_data)

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape))
    model.add(normalization_layer)

    if candidate.traits["model"] == "lstm":
        rec_layer = keras.layers.LSTM
    elif candidate.traits["model"] == "gru":
        rec_layer = keras.layers.GRU
    elif candidate.traits["model"] == "rnn":
        rec_layer = keras.layers.SimpleRNN

    for i in range(candidate.traits["r_layers"]):
        model.add(rec_layer(
            units = candidate.traits["r_units"],
            activation = candidate.traits["r_activation"],
            return_sequences = i != candidate.traits["r_layers"]-1,
        ))
        if candidate.traits["r_dropout"]:
            model.add(keras.layers.Dropout(candidate.traits["r_dropout"]))
        if candidate.traits["r_normalization"]:
            model.add(keras.layers.LayerNormalization())

    for i in range(candidate.traits["d_layers"]):
        model.add(keras.layers.Dense(
            units = candidate.traits["d_units"],
            activation = candidate.traits["d_activation"],
        ))
        if candidate.traits["d_dropout"]:
            model.add(keras.layers.Dropout(candidate.traits["d_dropout"]))
        if candidate.traits["d_normalization"]:
            model.add(keras.layers.LayerNormalization())

    model.add(keras.layers.Dense(1, activation=candidate.traits["out_activ"]))
    model.compile(
        optimizer = candidate.traits["optimizer"],
        loss = candidate.traits["loss"],
        metrics = ["binary_accuracy"]
    )

    return model


def fitness(candidate):
    logger.info(f"Evaluating candidate {candidate}")
    OBSERVATIONS = 3

    train_times = []
    accuracies  = []
    for i in range(OBSERVATIONS):

        x_test, y_test = folds[i]
        x_train = numpy.concatenate([folds[j][0] for j in range(len(folds)) if j != i])
        y_train = numpy.concatenate([folds[j][1] for j in range(len(folds)) if j != i])

        try:
            model = build_model(x_train, candidate)
            start_time = time.time()
            model.fit(
                x_train, y_train,
                epochs = candidate.traits["epochs"],
                batch_size = candidate.traits["batch_size"],
                validation_split = candidate.traits["validation_split"],
                callbacks = [
                    keras.callbacks.EarlyStopping(patience=2),
                    keras.callbacks.ReduceLROnPlateau(),
                ],
                verbose = 0,
            )
            end_time = time.time()
        except Exception as e:
            logger.warning(f"Candidate caused error: {e}")
            logger.warning("Candidate caused error, giving fitness 0")
            return 0

        train_times.append(end_time - start_time)
        accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
        accuracies.append(accuracy)

        logger.debug(f"Scored accuracy {accuracy} on round {i+1}")
        if accuracy < 0.8:
            break

        del x_train
        del y_train
        del x_test
        del y_test
        del model
        keras.backend.clear_session()
        tensorflow.compat.v1.reset_default_graph()
        gc.collect()

    time_penalty = statistics.fmean(train_times) / 24000 # 1% per 4th minute
    accuracy = statistics.fmean(accuracies)
    fitness = accuracy - time_penalty
    logger.info(f"Candidate fitness is {accuracy:0.4f}-{time_penalty:0.4f}={fitness:0.4f}")

    return fitness

# ---

def main():

    PARENT_OFFSPRING_RATIO = 0.1
    MUTATION_RATE = 0.05
    POPULATION = 100

    Candidate.MUTATION_RATE = MUTATION_RATE
    parent_count = max(1, int(POPULATION * PARENT_OFFSPRING_RATIO))

    if len(sys.argv) > 1:
        path = sys.argv[1]
        logger.info(f"Initializing population from {path}")
        with open(path, "rb") as file:
            population = pickle.load(file)
        POPULATION = len(population)
        START_GEN = int(path[-8:-4]) + 1
    else:
        logger.info("Initializing new population")
        population = [Candidate() for i in range(POPULATION)]
        START_GEN = 0

    logger.debug(f"{PARENT_OFFSPRING_RATIO=}")
    logger.debug(f"{MUTATION_RATE=}")
    logger.debug(f"{START_GEN=}")
    logger.debug(f"{POPULATION=}")
    logger.debug(f"{parent_count=}")

    # Survival
    logger.info("Starting simulation")
    scores = [fitness(candidate) for candidate in population]

    # Selection
    parents = numpy.argsort(scores)[-parent_count:]
    left = POPULATION - len(parents)
    children_fe = random.choices(population, weights=scores, k=left)
    children_ma = random.choices(population, weights=scores, k=left)

    # Evolution
    new_population = [population[parent] for parent in parents]
    new_population.extend([fe+ma for fe,ma in zip(children_fe, children_ma)])

    # Survival of the fittest
    population = new_population
    assert len(population) == POPULATION
    with open(f"generations/gen_{START_GEN:04d}.pkl", "wb") as file:
        pickle.dump(population, file)

    traits = [candidate.traits for candidate in population]
    with open("population.json", "w") as file:
        json.dump(traits, file)


if __name__ == "__main__":
    main()


