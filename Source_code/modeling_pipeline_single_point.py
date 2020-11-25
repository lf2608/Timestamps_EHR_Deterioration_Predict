import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample, shuffle

from Source_code.data_preprocessing_pipeline import select_column
from Source_code.plot_metrics import plot_cm, plot_pr_curve, plot_roc


LR_model = LogisticRegression(random_state=0)
LR_paragram_grid = [{'penalty': ['l1', 'l2'], 'C': [10 ** n for n in range(-4, 5)],
                     'solver': ['liblinear']},
                    {'penalty': ['none'], 'solver': ['newton-cg']}]

RF_model = RandomForestClassifier(random_state=0)
RF_paragram_grid = [{'n_estimators': [50, 100, 200], 'max_depth': [20, 50, None],
                     'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5]}]


LR_best_model = LogisticRegression(random_state=1, penalty='l2', C=100, solver='liblinear')
RF_best_model = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_leaf=1, min_samples_split=2,
                                       max_depth=50)


def get_results():
    results = []
    time_of_day = [True, False]
    sampling = ['last']
    algorithm_list = ['Logistic_Regression', 'Random_Forest']
    column = ['AUROC', 'AUPRC', 'sensibility', 'specificity', 'PPV', 'NPV', 'FScore']
    thresholds = [0.6, 0.1]
    algorithms = {'Logistic_Regression': LR_best_model, 'Random_Forest': RF_best_model}
    paragram_grids = {'Logistic_Regression': LR_paragram_grid, 'Random_Forest': RF_paragram_grid}
    for algorithm, threshold in zip(algorithm_list, thresholds):
        for method in sampling:
            for time in time_of_day:
                x = BuildAlgorithmsSinglePoint(sampling_method=method, timestep_length=60,
                                               algorithm=algorithms[algorithm], paragram_grid=paragram_grids[algorithm],
                                               time_of_day=time)
                results.append(list(x.get_results(search=False, threshold=threshold)))
    return pd.DataFrame(results, columns=column,
                        index=pd.MultiIndex.from_product([algorithm_list, sampling, time_of_day]))


class BuildAlgorithmsSinglePoint(object):
    _DIRECTORY = '/home/user/dataset/'

    def __init__(self, directory=None, sampling_method='random', timestep_length=60, algorithm=None, paragram_grid=None,
                 vitals=True, v_order=True, med_order=True, comments=True, notes=True, time_of_day=True):
        self.__directory = directory or self._DIRECTORY
        self.__method = sampling_method
        self.__length = timestep_length
        self.__algorithm = algorithm
        self.__grid = paragram_grid
        self.__vitals = vitals
        self.__v_order = v_order
        self.__med_order = med_order
        self.__comments = comments
        self.__notes = notes
        self.__time_of_day = time_of_day

    def get_best_model(self, search=False):
        point_train_data, _, train_label, _, _, _ = self._load_data(self.__method, self.__length)
        point_train_data = self._standardize_numeric_col(point_train_data)
        point_train_data = \
            self._feature_selection(ds=point_train_data, vitals=self.__vitals, v_order=self.__v_order,
                                    med_order=self.__med_order, comments=self.__comments,
                                    notes=self.__notes, time_of_day=self.__time_of_day)
        point_train_data, train_label = self._oversampling(point_train_data, train_label)
        if search:
            best_model = model_searching(self.__algorithm, self.__grid, point_train_data, train_label)
        else:
            best_model = model_training(self.__algorithm, point_train_data, train_label)
        return best_model

    def get_results(self, search=False, threshold=0.5):
        _, _, _, point_test_data, _, test_label = self._load_data(self.__method, self.__length)
        point_test_data = self._standardize_numeric_col(point_test_data)
        point_test_data = \
            self._feature_selection(ds=point_test_data, vitals=self.__vitals, v_order=self.__v_order,
                                    med_order=self.__med_order, comments=self.__comments,
                                    notes=self.__notes, time_of_day=self.__time_of_day)
        best_model = self.get_best_model(search)
        AUROC, AUPRC, sensibility, specificity, PPV, NPV, FScore \
            = model_validation(best_model, point_test_data, test_label, threshold)
        print('AUROC:{}, AUPRC:{}, sensibility:{}, specificity:{}, PPV:{}, NPV:{}, F-Score:{}'
              .format(AUROC, AUPRC, sensibility, specificity, PPV, NPV, FScore))
        return AUROC, AUPRC, sensibility, specificity, PPV, NPV, FScore

    def _load_data(self, sampling_method, timestep_length):
        directory = self.__directory
        folder = sampling_method
        timestep_length = str(timestep_length)
        point_train_data = np.load(directory + folder + '/len' + timestep_length + '_' + 'point_train_data.npy')
        train_data = np.load(directory + folder + '/len' + timestep_length + '_' + 'train_data.npy')
        train_label = np.load(directory + folder + '/len' + timestep_length + '_' + 'train_label.npy')
        point_test_data = np.load(directory + folder + '/len' + timestep_length + '_' + 'point_test_data.npy')
        test_data = np.load(directory + folder + '/len' + timestep_length + '_' + 'test_data.npy')
        test_label = np.load(directory + folder + '/len' + timestep_length + '_' + 'test_label.npy')
        return point_train_data, train_data, train_label, point_test_data, test_data, test_label

    def _standardize_numeric_col(self, ds):
        nm_ds = ds[:, :15]
        mean = nm_ds.mean(axis=0)
        std = nm_ds.std(axis=0)
        standardized_ds = (nm_ds - mean) / std
        standardized_ds = np.concatenate((standardized_ds, ds[:, 15:]), axis=1)
        return standardized_ds

    def _feature_selection(self, ds, vitals=True, v_order=False, med_order=False, comments=False,
                           notes=False, time_of_day=True):
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

    def _oversampling(self, train_data, train_label):
        pos_data, pos_label = resample(train_data[train_label == 1], train_label[train_label == 1],
                                       n_samples=(train_label == 0).sum(), replace=True, random_state=0)
        neg_data = train_data[train_label == 0]
        neg_label = train_label[train_label == 0]
        new_data = np.concatenate((pos_data, neg_data))
        new_label = np.concatenate([pos_label, neg_label], axis=None)
        new_data, new_label = shuffle(new_data, new_label, random_state=0)
        return new_data, new_label

    def _column_name(self, freq):
        periods = int(1440 / freq)
        freq = str(freq) + 'T'
        col = select_column(base=False, vitals=True, v_order=True, med_order=True, comments=True, notes=True)
        col += ['Time_of_day_' + str(t)[7:] for t in pd.timedelta_range(0, periods=periods, freq=freq)]
        return col


def model_searching(algorithm, gridsearch_dict, training_data, training_label):
    search = GridSearchCV(algorithm, gridsearch_dict, cv=4)
    search.fit(training_data, training_label)
    best_model = search.best_estimator_
    # print(pd.DataFrame(search.cv_results_).sort_values(by='rank_test_score'))
    print('The best model hyperparameters: ', search.best_params_)
    return best_model


def model_training(best_model, training_data, training_label):
    best_model.fit(training_data, training_label)
    return best_model


def model_validation(model, test_data, true_label, threshold):
    pred_label = model.predict_proba(test_data)[:, 1]

    sensibility, specificity, PPV, NPV = plot_cm(true_label, pred_label, p=threshold)
    AUPRC = plot_pr_curve('AUPRC', true_label, pred_label)
    AUROC = plot_roc('AUROC', true_label, pred_label)
    FScore = 2 / (sensibility ** -1 + PPV ** -1)
    return AUROC, AUPRC, sensibility, specificity, PPV, NPV, FScore



