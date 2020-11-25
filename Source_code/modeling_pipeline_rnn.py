from tqdm import tqdm
import tensorflow as tf
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch, Hyperband
from kerastuner import HyperModel
import kerastuner as kt
import numpy as np
import pandas as pd

from Source_code.plot_metrics import plot_cm, plot_pr_curve, plot_roc


def get_results_table():
    results = []
    sampling = ['last']
    step_lengths = [15, 30, 60]
    time_of_day = [True, False]
    algorithm_list = ['GRU', 'LSTM']
    column = ['AUROC', 'AUPRC', 'sensibility', 'specificity', 'PPV', 'NPV', 'FScore']

    for algorithm in tqdm(algorithm_list):
        print(algorithm)
        for method in sampling:
            print(method)
            for time in time_of_day:
                print(time)
                for length in step_lengths:
                    print(length)
                    x = BuildAlgorithms(sampling_method=method, timestep_length=length,
                                        algorithm=algorithm, time_of_day=time)
                    results.append(list(x.get_results()))
    return pd.DataFrame(results, columns=column, index=pd.MultiIndex.from_product([algorithm_list, sampling,
                                                                                   time_of_day, step_lengths]))


class BuildAlgorithms(object):
    _DIRECTORY = '/home/user/dataset/'
    _EPOCHS = 1000
    _BATCH_SIZE = 128

    _early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_AUROC',
        verbose=0,
        patience=10,
        mode='max',
        restore_best_weights=True)
    _METRICS = [
        tf.keras.metrics.AUC(curve='ROC', name='AUROC'),
        tf.keras.metrics.AUC(curve='PR', name='AUPRC'),
        tf.keras.metrics.Precision(name='precision', thresholds=0.6),
        tf.keras.metrics.Recall(name='recall', thresholds=0.6),
        tf.keras.metrics.TruePositives(name='tp', thresholds=0.6),
        tf.keras.metrics.FalsePositives(name='fp', thresholds=0.6),
        tf.keras.metrics.TrueNegatives(name='tn', thresholds=0.6),
        tf.keras.metrics.FalseNegatives(name='fn', thresholds=0.6),
    ]

    def __init__(self, directory=None, sampling_method='last', timestep_length=60, algorithm=None, epochs=None,
                 batch_size=None, early_stopping=None, metrics=None, vitals=True, v_order=True,
                 med_order=True, comments=True, notes=True, time_of_day=True):
        self.__directory = directory or self._DIRECTORY
        self.__method = sampling_method
        self.__length = timestep_length
        self.__algorithm = algorithm
        self.__epochs = epochs or self._EPOCHS
        self.__batch_size = batch_size or self._BATCH_SIZE
        self.__early_stopping = early_stopping or self._early_stopping
        self.__metrics = metrics or self._METRICS
        self.__vitals = vitals
        self.__v_order = v_order
        self.__med_order = med_order
        self.__comments = comments
        self.__notes = notes
        self.__time_of_day = time_of_day

    def get_results(self):
        _, train_data, train_label, _, _, _ = _load_data(self.__directory, self.__method, self.__length)
        _, _, _, _, test_data, test_label = _load_data(self.__directory, self.__method, self.__length)
        train_data, test_data = \
            map(lambda x: _feature_selection(ds=x, vitals=self.__vitals, v_order=self.__v_order,
                                             med_order=self.__med_order, comments=self.__comments,
                                             notes=self.__notes, time_of_day=self.__time_of_day),
                [train_data, test_data])
        train_data, val_data, train_label, val_label = \
            train_test_split(train_data, train_label, test_size=0.25, random_state=42)
        train_data, train_label = _oversampling(train_data, train_label)
        train_ds = _Create_dataset(train_data, train_label, repeat=True)
        val_ds = _Create_dataset(val_data, val_label)
        test_ds = _Create_dataset(test_data, test_label)

        steps_per_epoch = len(train_data) // self.__batch_size
        input_shape = train_data.shape[-2:]
        RNN_model = self._make_model(self.__metrics, self.__algorithm, input_shape)
        RNN_history = RNN_model.fit(
            train_ds,
            epochs=self.__epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[self.__early_stopping],
            validation_data=val_ds,
            verbose=1
        )

        #plot_metrics(RNN_history)
        _, AUROC, AUPRC, precision, recall, TP, FP, TN, FN = \
            RNN_model.evaluate(test_ds, batch_size=self.__batch_size, verbose=0)
        NPV = TN/(TN+FN)
        specificity = TN/(TN+FP)
        FScore = 2/(recall**-1 + precision**-1)

        print('AUROC:{}, AUPRC:{}, sensitivity:{}, specificity:{}, PPV:{}, NPV:{}, F-Score:{}'
              .format(AUROC, AUPRC, recall, specificity, precision, NPV, FScore))
        return AUROC, AUPRC, recall, specificity, precision, NPV, FScore

    def _make_model(self, metrics, algorithm, input_shape):
        if algorithm == 'GRU':
            RNN_model = tf.keras.Sequential([
                tf.keras.layers.GRU(336, input_shape=input_shape),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            RNN_model = tf.keras.Sequential([
                tf.keras.layers.LSTM(240, input_shape=input_shape),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

        RNN_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics)

        return RNN_model


def _load_data(directory, sampling_method, timestep_length):
    folder = sampling_method
    timestep_length = str(timestep_length)
    point_train_data = np.load(directory+folder+'/len'+timestep_length + '_' + 'point_train_data.npy')
    train_data = np.load(directory+folder+'/len'+timestep_length + '_' + 'train_data.npy')
    train_label = np.load(directory+folder+'/len'+timestep_length + '_' + 'train_label.npy')
    point_test_data = np.load(directory+folder+'/len'+timestep_length + '_' + 'point_test_data.npy')
    test_data = np.load(directory+folder+'/len'+timestep_length + '_' + 'test_data.npy')
    test_label = np.load(directory+folder+'/len'+timestep_length + '_' + 'test_label.npy')
    return point_train_data, train_data, train_label, point_test_data, test_data, test_label


def _feature_selection(ds, vitals=True, v_order=True, med_order=True,
                       comments=True, notes=True, time_of_day=True):
    position = list(range(ds.shape[-1]))
    _VITALSIGNS = [0, 1, 2, 3, 4]
    _VITALORDERENTRY = [5, 6]
    _MEDORDERENTRY = [7, 8]
    _FLOAWSHEETCOMMENT = [9, 10, 11, 12, 13]
    _NOTES = [14]
    deposit = []

    if not time_of_day:
        position = position[:15]
    if not vitals:
        deposit += _VITALSIGNS
    if not v_order:
        deposit += _VITALORDERENTRY
    if not med_order:
        deposit += _MEDORDERENTRY
    if not comments:
        deposit += _FLOAWSHEETCOMMENT
    if not notes:
        deposit += _NOTES
    position = list(filter(lambda x: x not in deposit, position))
    return np.take(ds, position, axis=-1)


def _oversampling(train_data, train_label):
    pos_data, pos_label = resample(train_data[train_label == 1], train_label[train_label == 1],
                                   n_samples=(train_label == 0).sum(), replace=True, random_state=0)
    neg_data = train_data[train_label == 0]
    neg_label = train_label[train_label == 0]
    new_data = np.concatenate((pos_data, neg_data))
    new_label = np.concatenate([pos_label, neg_label], axis=None)
    new_data, new_label = shuffle(new_data, new_label, random_state=0)
    return new_data, new_label


def _Create_dataset(data, label, batch_size=128, repeat=False):
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    if repeat:
      dataset = dataset.batch(batch_size).repeat().prefetch(2)
    else:
       dataset = dataset.batch(batch_size).prefetch(2)
    return dataset


class RNNHyperModel(HyperModel):
    _METRICS = [
        tf.keras.metrics.AUC(curve='ROC', name='AUROC'),
        tf.keras.metrics.AUC(curve='PR', name='AUPRC'),
        tf.keras.metrics.Precision(name='precision', thresholds=0.6),
        tf.keras.metrics.Recall(name='recall', thresholds=0.6),
        tf.keras.metrics.TruePositives(name='tp', thresholds=0.6),
        tf.keras.metrics.FalsePositives(name='fp', thresholds=0.6),
        tf.keras.metrics.TrueNegatives(name='tn', thresholds=0.6),
        tf.keras.metrics.FalseNegatives(name='fn', thresholds=0.6),
    ]

    def __init__(self, input_shape, metrics=None, unit='GRU'):
        self._input_shape = input_shape
        self._metrics = metrics or self._METRICS
        self._unit = unit

    def build(self, hp):
        if self._unit == 'GRU':
            RNN_model = tf.keras.Sequential([
                tf.keras.layers.GRU(units=hp.Int('units',
                                                min_value=16,
                                                max_value=512,
                                                step=16),
                                     input_shape=self._input_shape),
                tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            RNN_model = tf.keras.Sequential([
                tf.keras.layers.LSTM(units=hp.Int('units',
                                                 min_value=16,
                                                 max_value=512,
                                                 step=16),
                                    input_shape=self._input_shape),
                tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        RNN_model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate',
                          values=[3e-2, 1e-3, 3e-4, 1e-4])),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=self._metrics)
        return RNN_model


def get_best_rnn(unit='LSTM'):
    _DIRECTORY = '/home/user/dataset/'
    _, train_data, train_label, _, _, _ = _load_data(_DIRECTORY, 'last', 60)
    train_data = _feature_selection(ds=train_data, time_of_day=True)
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.25,
                                                                    random_state=0)
    train_data, train_label = _oversampling(train_data, train_label)
    train_ds = _Create_dataset(train_data, train_label, repeat=True)
    val_ds = _Create_dataset(val_data, val_label)

    BATCH_SIZE = 128
    steps_per_epoch = len(train_data) // BATCH_SIZE
    input_shape = train_data.shape[-2:]

    RNN_model = RNNHyperModel(input_shape, unit=unit)

    tuner = Hyperband(
        RNN_model,
        objective=kt.Objective("val_AUPRC", direction="max"),
        max_epochs=10,
        factor=3,
        directory='/home/liheng/MAT1115/',
        project_name='MAT_HyperGRU_60lastPRC')
    tuner.search_space_summary()

    tuner.search(train_ds,
                 epochs=20,
                 steps_per_epoch=steps_per_epoch,
                 validation_data=val_ds)
    tuner.results_summary()
