#!/usr/bin/env python
# Script that builds and tests ML models for lightning strike prediction.


import os
import gc
import sys
import time
import math
import random
import itertools
import statistics
import multiprocessing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy
import pandas
import keras
keras.utils.set_random_seed(0)


print("Initializing program")


USE_MP = False
FOLD_COUNT = 10
BIN_SIZE = 0
MTYPES = ("dense", "rnn", "lstm", "gru")

MAX_TIMESTEPS = 73
LOOKBACKS  = [1, 3, 6, 12, 24, 48, 72]
LOOKAHEADS = [1, 3, 6, 12, 24]


def structure_data(df_light_part, df_light_full, df_mesan, length, foresight, timeseries=True):
    assert length + foresight <= MAX_TIMESTEPS
    assert foresight > 0
    assert length > 0

    # Determine data shifts based on foresight
    n = len(df_light_part)
    lights = df_light_part["index"].values.tolist()
    lshifts = [random.randint(0, foresight-1) for i in range(n)]
    dshifts = dict(zip(lights, lshifts))

    # Build set X (features)
    x = pandas.merge(
        df_mesan, df_light_part,
        left_on  = "light.index",
        right_on = "index",
        how      = "inner",
    )[df_mesan.columns]

    if timeseries:
        groups = []
        for index, group in x.groupby("light.index"):
            shift = dshifts[index]
            t = group.tail(length + shift)
            t = group.head(len(t) - shift)
            t = group.drop("light.index", axis=1)
            t = t.to_numpy()
            groups.append(t)
        x = groups
    else:
        x = x.groupby("light.index").tail(1)
        x = x.drop("light.index", axis=1)

    # Build set Y (labels)
    foresight = pandas.to_timedelta(foresight, unit="h")
    df_light_full.latitude  = round(df_light_full.latitude,  BIN_SIZE) 
    df_light_full.longitude = round(df_light_full.longitude, BIN_SIZE) 

    shifts = pandas.to_timedelta(lshifts, unit="h")
    df_light_part.timestamp = df_light_part.timestamp - shifts
    df_light_part.longitude = round(df_light_part.longitude, BIN_SIZE)
    df_light_part.latitude = round(df_light_part.latitude, BIN_SIZE)

    y = []
    for _, row in df_light_part.iterrows():
        for ts in pandas.date_range(row.timestamp, row.timestamp+foresight, freq="1h"):

            condition = ((df_light_full.timestamp == ts) \
                & (df_light_full.longitude == row.longitude) \
                & (df_light_full.latitude == row.latitude) \
                & (df_light_full.lightning == True)
            )

            if condition.any():
                y.append(1)
                break

        else:
            y.append(0)

    return numpy.array(x), numpy.array(y)


def mp_structure_data(arg):
    return structure_data(*arg)


def get_f1_score(predictions, labels):
    tp = sum([1 for pred, label in zip(predictions, labels) if pred and label])
    fp = sum([1 for pred, label in zip(predictions, labels) if pred and not label])
    fn = sum([1 for pred, label in zip(predictions, labels) if not pred and label])
    f1_score = (2*tp) / (2*tp + fp + fn)
    return f1_score


def build_model(mtype, training_data):

    model = keras.models.Sequential()

    if mtype == "dense":
        input_shape = training_data.shape[1]
    else:
        input_shape = (training_data.shape[1], training_data.shape[2])
    model.add(keras.layers.InputLayer(input_shape))

    normalization_layer = keras.layers.Normalization()
    normalization_layer.adapt(training_data)
    model.add(normalization_layer)

    if mtype == "rnn":
        rec_layer = keras.layers.SimpleRNN
    elif mtype == "gru":
        rec_layer = keras.layers.GRU
    elif mtype == "lstm":
        rec_layer = keras.layers.LSTM

    layer_count = 2
    for i in range(layer_count):
        if mtype == "dense":
            model.add(keras.layers.Dense(
                units = 256,
                activation = "relu",
            ))
        else:
            model.add(rec_layer(
                units = 256,
                activation = "tanh",
                return_sequences = i != layer_count-1,
            ))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.LayerNormalization())

    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer = "adam",
        loss      = "binary_crossentropy",
        metrics   = ["binary_accuracy", "mean_absolute_error"],
    )

    return model


def evaluate(mtype, folds):

    train_times = []
    accuracies  = []
    f1_scores   = []
    maes        = []

    for i in range(FOLD_COUNT):

        x_test, y_test = folds[i]
        x_train = numpy.concatenate([folds[j][0] for j in range(len(folds)) if j != i])
        y_train = numpy.concatenate([folds[j][1] for j in range(len(folds)) if j != i])

        model = build_model(mtype, x_train)
        positive_balance = y_train.sum() / (len(y_train)/2)

        start_time = time.time()
        model.fit(
            x_train, y_train,
            epochs = 8,
            batch_size = 128,
            validation_split = 1 / FOLD_COUNT,
            class_weight = {1: 1, 0: positive_balance},
            callbacks = [
                keras.callbacks.ReduceLROnPlateau(),
            ],
        )
        stop_time = time.time()
        train_times.append(stop_time - start_time)

        measures = model.evaluate(x_test, y_test)
        accuracies.append(measures[1])
        maes.append(measures[2])

        predictions = model.predict(x_test)
        predictions = numpy.rint(predictions)
        f1_scores.append(get_f1_score(predictions, y_test))

        del model
        keras.backend.clear_session()
        gc.collect()

    train_time = statistics.fmean(train_times)
    accuracy   = statistics.fmean(accuracies)
    f1_score   = statistics.fmean(f1_scores)
    mae        = statistics.fmean(maes)

    error = (1 - accuracy)
    confidence = error + 1.96 * math.sqrt((error*accuracy) / len(x_test))

    return train_time, accuracy, f1_score, mae, confidence


def main():

    mtype = sys.argv[1]
    output = sys.argv[2]
    if mtype not in MTYPES:
        print(f"Error: Invalid model type '{mtype}'")
        sys.exit(1)

    print("Loading MESAN data...")
    df_mesan = pandas.read_csv("./df_mesan.csv")
    df_mesan.drop("timestamp", axis=1, inplace=True)

    print("Loading LIGHT data...")
    df_lights = [pandas.read_csv(f"df_light/df_light_part_{i:02}.csv") for i in range(FOLD_COUNT)]
    for dfl in df_lights:
        dfl.timestamp = pandas.to_datetime(dfl.timestamp)
    df_light = pandas.concat(df_lights, ignore_index=True)

    with open(output, "w") as f:
        for lookback, lookahead in itertools.product(LOOKBACKS, LOOKAHEADS):

            print(f"Lookback {lookback}, lookahead {lookahead}", file=f)
            if lookback + lookahead > MAX_TIMESTEPS:
                print("Insufficient timesteps for combination, skipping...", file=f)
                continue

            print("Structuring folds...")
            if USE_MP:
                args = [[fold, df_light, df_mesan, lookback, lookahead, mtype!="dense"] for fold in df_lights]
                with multiprocessing.Pool() as pool:
                    folds = pool.map(mp_structure_data, args)
            else:
                folds = [structure_data(fold, df_light, df_mesan, lookback, lookahead, mtype!="dense") for fold in df_lights]

            train_time, accuracy, f1_score, mae, confidence = evaluate(mtype, folds)
            print(f"- Training Time: {train_time:0.4f}", file=f)
            print(f"- Accuracy: {accuracy*100:0.4f}%", file=f)
            print(f"- F1 Score: {f1_score*100:0.4f}%", file=f)
            print(f"- Mean Absolute Error: {mae*100:0.4f}%", file=f)
            print(f"- Confidence: {confidence*100:0.4f}%", file=f)
            print(file=f)


if __name__ == "__main__":
    main()


