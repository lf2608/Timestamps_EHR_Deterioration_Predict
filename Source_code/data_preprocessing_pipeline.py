import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class DataPipeline(object):
    _FILE_NAME = '/home/liheng/Mat/Source_code/dataset/dataset_icu.csv'
    _FILTER_OUTLIERS = False
    _SAMPLE_LENGTH = 24
    _TIME_GAP = 12
    _FVATIALS = True
    _FVORDERS = True
    _FMEDORDERS = True
    _FCOMMENTS = True
    _FNOTES = True
    #_TOD = True
    _SAVE_DIRECTORY = '/home/liheng/Mat/Source_code/dataset/'

    def __init__(self, file_name=None, filter_outliers=None, starttime='last', sample_length=None, time_gap=None,
                 matching=False, f_vitals=None, f_vorder=None, f_medorder=None, f_comments=None, f_notes=None,
                 timestep_length=60, time_of_day=True, save_dir=None):
        self.__file_name = file_name or self._FILE_NAME
        self.__filter_outliers = filter_outliers or self._FILTER_OUTLIERS
        self.__starttime = starttime
        self.__sample_length = sample_length or self._SAMPLE_LENGTH
        self.__time_to_outcome = time_gap or self._TIME_GAP
        self.__matching = matching
        self.__timestep_length = timestep_length
        self.__f_vitals = f_vitals or self._FVATIALS
        self.__f_vorder = f_vorder or self._FVORDERS
        self.__f_medorder = f_medorder or self._FMEDORDERS
        self.__f_comments = f_comments or self._FCOMMENTS
        self.__f_notes = f_notes or self._FNOTES
        self.__time_of_day = time_of_day
        self.__save_dir = save_dir or self._SAVE_DIRECTORY

    def get_results(self):
        df = self._data_formating(self.__file_name)
        train_co, test_co = self._data_sampling(df, self.__starttime, self.__sample_length, self.__time_to_outcome)
        train_f, test_f = self._feature_engineering(train_co, test_co)
        print(self.__time_of_day)
        point_train_data, train_data, train_label, point_test_data, test_data, test_label = \
            self._create_dataset(train_f, test_f, matching=self.__matching,
                                 vitals=self.__f_vitals, v_order=self.__f_vorder,
                                 med_order=self.__f_medorder, comments=self.__f_comments,
                                 notes=self.__f_notes, time_of_day=self.__time_of_day,
                                 timestep_length=self.__timestep_length,
                                 time_to_outcome=self.__time_to_outcome, sample_length=self.__sample_length)
        return point_train_data, train_data, train_label, point_test_data, test_data, test_label

    def save_array(self, point_train_data, train_data, train_label, point_test_data, test_data, test_label):
        folder = self.__starttime + '/'
        timestep_lenght = 'len' + str(self.__timestep_length)
        directory = self.__save_dir + folder + timestep_lenght
        np.save(directory + '_point_train_data.npy', point_train_data)
        np.save(directory + '_train_data.npy', train_data)
        np.save(directory + '_train_label.npy', train_label)
        np.save(directory + '_point_test_data.npy', point_test_data)
        np.save(directory + '_test_data.npy', test_data)
        np.save(directory + '_test_label.npy', test_label)
        print('Dataset Saved. Sampling: {}, Timestep Length: {}'.format(self.__starttime, self.__timestep_length))
        print('Save to directory: ', self.__save_dir + folder)

    def _data_formating(self, file_name, filter_outliers=False):
        icu_df = pd.read_csv(file_name)

        # data cleaning
        icu_df = icu_df.astype({'outcome_time': 'datetime64[ns]', 'recorded_time': 'datetime64[ns]'})
        icu_df.replace({'Female': 1, 'Male': 0}, inplace=True)

        # admission time round by hour
        adm_t = icu_df.groupby('dummy_encounter_id')['recorded_time'].aggregate('min')
        adm_t = (adm_t.astype('int') - adm_t.astype('int') % (60 * 60 * 10 ** 9)).astype('datetime64[ns]')
        # outcome time round by hour
        proxyend = icu_df.groupby('dummy_encounter_id')['outcome_time'].aggregate('min')
        proxyend = (proxyend.astype('int') - proxyend.astype('int') % (60 * 60 * 10 ** 9)).astype('datetime64[ns]')

        #
        cohort_df = pd.merge(proxyend, icu_df, left_index=True, right_on='dummy_encounter_id').rename(
            columns={'outcome_time_x': 'proxyend_time', 'outcome_time_y': 'outcome_time'})
        #
        cohort_df = pd.merge(adm_t, cohort_df, left_index=True, right_on='dummy_encounter_id').rename(
            columns={'recorded_time_x': 'adm_time', 'recorded_time_y': 'recorded_time'})

        # calculate los
        cohort_df['los'] = cohort_df['outcome_time'] - cohort_df['adm_time']
        cohort_df['los'] = cohort_df['los'].astype('timedelta64[h]')

        # filter out top 1% longest stay
        if filter_outliers:
            los_table = cohort_df.groupby('dummy_encounter_id').first()[['outcome', 'los']]
            ninty_nine_quantile = los_table['los'].quantile(q=0.99, interpolation='lower')
            cohort_df = cohort_df[cohort_df['los'] <= ninty_nine_quantile]

        return cohort_df

    def _data_sampling(self, dataf, starttime, sample_length=24, time_to_outcome=12):
        # eligable
        icu_df = dataf[dataf['los'] >= (sample_length + time_to_outcome)]
        print("cohort met criteria: {}, control: {}, outcome: {}"
              .format(len(icu_df['dummy_encounter_id'].unique()),
                      len(icu_df[icu_df['outcome'] == 0]['dummy_encounter_id'].unique()),
                      len(icu_df[icu_df['outcome'] == 1]['dummy_encounter_id'].unique())))

        # split dataset into control and outcome set
        control_list = icu_df[icu_df['outcome'] == 0]['dummy_encounter_id'].unique()
        outcome_list = icu_df[icu_df['outcome'] == 1]['dummy_encounter_id'].unique()
        control_cohort = icu_df[icu_df['dummy_encounter_id'].isin(control_list)]
        outcome_cohort = icu_df[icu_df['dummy_encounter_id'].isin(outcome_list)]

        # split method
        dfs = [control_cohort, outcome_cohort]
        sampled = []

        for icu_df in dfs:
            # first 00hrs
            if starttime == 'first':
                icu_df['sample_start'] = icu_df['adm_time']
                sampled_df = icu_df[(icu_df['recorded_time'] < (icu_df['sample_start'] +
                                                            pd.Timedelta(hours=sample_length)))]
                sampled_df.sort_values(by=['dummy_encounter_id', 'recorded_time'], inplace=True)

            elif starttime == 'last':
                #
                icu_df['sample_start'] = icu_df['proxyend_time'] - pd.Timedelta(hours=sample_length + time_to_outcome)
                sampled_df = icu_df[(icu_df['recorded_time'] >= icu_df['sample_start'])
                                & (icu_df['recorded_time'] < (icu_df['sample_start'] +
                                                              pd.Timedelta(hours=sample_length)))]
                sampled_df.sort_values(by=['dummy_encounter_id', 'recorded_time'], inplace=True)

            elif starttime == 'random':
                sampled_df = random_sampling(icu_df, sample_length, time_to_outcome)
            sampled.append(sampled_df)

        control_cohort, outcome_cohort = sampled
        print("control: {}, outcome: {}"
              .format(len(control_list), len(outcome_list)))

        return control_cohort, outcome_cohort

    def _feature_engineering(self, control_cohort, outcome_cohort):
        columns = select_column(base=True, vitals=True, v_order=True, med_order=True, comments=True,
                              notes=True, nlp_topic=False)
        dfs = [control_cohort, outcome_cohort]
        selected_dfs = []
        for df in dfs:
            # Calculate time of day variable
            df['time_of_day'] = df['recorded_time'].dt.time
            tod = list(df['time_of_day'])
            minute = []
            for i in tod:
                h, m, s = str(i).split(':')
                count = (int(h) * 60 + int(m))
                minute.append(count)
            # time of day_(minute)
            df['time_of_day_minute'] = minute
            # calculate measurement time to sample starttime
            df['dt_start'] = df['recorded_time'] - df['sample_start']
            # select columns
            selectedcol_df = df.loc[:, columns]
            selected_dfs.append(selectedcol_df)
        control_df, outcome_df = selected_dfs
        return control_df, outcome_df

    def _create_dataset(self, control_df, outcome_df, matching=False, timestep_length=60,
                        sample_length=24, time_to_outcome=12,
                        time_of_day=True, vitals=True, v_order=True, med_order=True, comments=True, notes=True):
        steps = int(sample_length*60 / timestep_length)
        timestep_length_str = str(timestep_length) + 'T'
        dfs = [control_df, outcome_df]
        final_dataset = []
        columns = select_column(base=False, vitals=vitals, v_order=v_order, med_order=med_order,
                                comments=comments, notes=notes)
        print('with columns: ', columns)
        # loop thru icu stays, training
        for df in dfs:
            labels = []
            features = []
            overallcount = []
            icu_id = df['dummy_encounter_id'].unique()
            # df['timestep_iloc'] = pd.cut(df.time_of_day_minute, range(0, 1441, timestep_length), right=False,
            #                             labels=np.arange(1, 1+steps))

            timeframe = pd.DataFrame(0, columns=df.columns,
                                     index=pd.timedelta_range(0, periods=steps, freq=timestep_length_str))
            timeframe.drop(columns=['dummy_encounter_id', 'adm_time', 'outcome', 'time_of_day_minute',
                                    'sample_start', 'dt_start'],
                           inplace=True)
            for idx in tqdm(icu_id):
                df_time = df[df['dummy_encounter_id'] == idx]
                sample_start_hour = df_time['sample_start'].dt.hour.unique()
                # label
                label = df_time['outcome'].unique()[0]
                df_time = df_time.drop(
                    columns=['dummy_encounter_id', 'adm_time', 'outcome', 'time_of_day_minute', 'sample_start'])
                df_time = df_time.set_index('dt_start')
                # Create a single point dataset for logistic regression and tree models (overall counts of records)
                point_features = df_time.sum(axis=0).values
                # Resample data by given length of timestep
                df_time = pd.concat([timeframe, df_time])
                # Convert sequence to binary variables
                # df_time = df_time.resample(timestep_length_str).max()
                # df_time.iloc[:, :-1] = df_time.iloc[:, :-1] != 0

                # Convert sequence to counts per hour
                cond = {col: 'sum' for col in columns}
                df_time = df_time.resample(timestep_length_str).agg(cond)

                # Select features
                df_time = df_time.loc[:, columns]
                # Get the position of first record
                steps_per_day = int(1440 / timestep_length)
                start_step = sample_start_hour * 60 / timestep_length
                binary_features = df_time.to_numpy()

                labels.append([label, sample_start_hour])
                if time_of_day:
                    # concatenate features with time of day variable
                    binary_features = binary_features[np.arange(-start_step, steps_per_day - start_step, dtype='int')]
                    point_features = np.concatenate((point_features, np.array(sample_start_hour)), axis=None)
                    assert binary_features.shape == (steps, len(columns)), \
                        'matrix shape:{}, expected: {}'.format(binary_features.shape,
                                                               (steps, len(columns)))
                    assert point_features.shape == (len(columns) + 1,), \
                        'matrix shape:{}, expected: {}'.format(point_features.shape,
                                                               (len(columns) + 1,))
                else:
                    assert binary_features.shape == (steps, len(columns)), \
                        'matrix shape:{}, expected: {}'.format(binary_features.shape, (steps, len(columns)))
                    assert point_features.shape == (len(columns),), \
                        'matrix shape:{}, expected: {}'.format(point_features.shape, (len(columns, )))
                overallcount.append(point_features)
                features.append(binary_features)

            final_dataset.append(np.array(overallcount, dtype='float64'))
            final_dataset.append(np.array(features, dtype='float64'))
            final_dataset.append(np.array(labels, dtype='float64'))
        point_control_data, control_data, control_labels, point_outcome_data, outcome_data, outcome_labels \
            = final_dataset

        if matching:
            point_training_data, training_data, training_labels, point_holdout_data, holdout_data, holdout_labels = \
                _matching(point_control_data, control_data, control_labels, point_outcome_data, outcome_data,
                          outcome_labels,
                          match_ratio=40)

        else:
            point_data = np.concatenate((point_control_data, point_outcome_data))
            series_data = np.concatenate((control_data, outcome_data))
            labels = np.concatenate((control_labels, outcome_labels))
            labels, _ = _label_format(labels)

            point_training_data, point_holdout_data, training_labels, holdout_labels \
                = train_test_split(point_data, labels, test_size=0.25, random_state=10)

            training_data, holdout_data, training_labels, holdout_labels \
                = train_test_split(series_data, labels, test_size=0.25, random_state=10)

        return point_training_data, training_data, training_labels, point_holdout_data, holdout_data, holdout_labels


def _matching(point_control_data, control_data, control_labels, point_outcome_data, outcome_data, outcome_labels,
              match_ratio=5):
    # match by hour of outcome
    np.random.seed(20)
    # split outcome set into 120: 41
    training_outcome_args = np.random.choice(len(outcome_labels), 120, replace=False)
    training_point_outcome_data, training_outcome_data, training_outcome_labels = \
        map(lambda x: np.take(x, training_outcome_args, axis=0), [point_outcome_data, outcome_data, outcome_labels])
    holdout_point_outcome_data, holdout_outcome_data, holdout_outcome_labels = \
        map(lambda x: np.delete(x, training_outcome_args, axis=0), [point_outcome_data, outcome_data, outcome_labels])
    _, training_outcome_hour = _label_format(training_outcome_labels)
    #holdout_outcome_labels, _ = _label_format(holdout_outcome_labels)
    print("training outcome set shape: point_data {}, data {}, label{}"
          .format(training_point_outcome_data.shape, training_outcome_data.shape, training_outcome_labels.shape))
    print("holdout outcome set shape: point_data {}, data {}, label{}"
          .format(holdout_point_outcome_data.shape, holdout_outcome_data.shape, holdout_outcome_labels.shape))

    # split control set into 3:1
    holdout_control_args = np.random.choice(len(control_labels), int(len(control_labels)/4), replace=False)
    point_control_data_for_match, control_data_for_match, control_labels_for_match = \
        map(lambda x: np.delete(x, holdout_control_args.astype('int64'), axis=0),
            [point_control_data, control_data, control_labels])
    holdout_point_control_data, holdout_control_data, holdout_control_labels = \
        map(lambda x: np.take(x, holdout_control_args.astype('int64'), axis=0),
            [point_control_data, control_data, control_labels])
    #holdout_control_labels, _ = _label_format(holdout_control_labels)

    # match training outcome set to training control set
    training_control_args = np.array([])
    outcome_hour, counts = np.unique(training_outcome_hour, return_counts=True)
    for hour, count in zip(outcome_hour, counts):
        print('hour:{}, count:{}'.format(hour, count))
        control_args_part = np.argwhere(control_labels_for_match[:, 1] == hour).flatten()
        training_control_args_part = np.random.choice(control_args_part, count * match_ratio, replace=True)
        print('matched control count: ', len(training_control_args_part))
        training_control_args = np.append(training_control_args, training_control_args_part)

    training_point_control_data, training_control_data, training_control_labels = \
        map(lambda x: np.take(x, training_control_args.astype('int64'), axis=0),
            [point_control_data_for_match, control_data_for_match, control_labels_for_match])

    #training_control_labels, training_control_hour = _label_format(training_control_labels)

    print("training control set shape: point_data {}, data {}, label{}"
          .format(training_point_control_data.shape, training_control_data.shape, training_control_labels.shape))
    print("holdout control set shape: point_data {}, data {}, label{}"
          .format(holdout_point_control_data.shape, holdout_control_data.shape, holdout_control_labels.shape))

    point_training_data = np.concatenate((training_point_outcome_data, training_point_control_data), axis=0)
    training_data = np.concatenate((training_outcome_data, training_control_data), axis=0)
    training_labels = np.concatenate((training_outcome_labels, training_control_labels), axis=0)
    training_labels, _ = _label_format(training_labels)

    point_holdout_data = np.concatenate((holdout_point_outcome_data, holdout_point_control_data), axis=0)
    holdout_data = np.concatenate((holdout_outcome_data, holdout_control_data), axis=0)
    holdout_labels = np.concatenate((holdout_outcome_labels, holdout_control_labels), axis=0)
    holdout_labels, _ = _label_format(holdout_labels)

    print("point_training_data: {}, training_data: {}, training_labels: {}, "
          "point_holdout_data: {}, holdout_data: {}, holdout_labels: {}"
          .format(point_training_data.shape, training_data.shape, training_labels.shape,
                  point_holdout_data.shape, holdout_data.shape, holdout_labels.shape))
    return point_training_data, training_data, training_labels, point_holdout_data, holdout_data, holdout_labels


def _label_format(labels):
    labels, left_censor_hour = np.hsplit(labels, 2)
    return map(lambda x: x.flatten(), [labels, left_censor_hour])


def random_sampling(icu_df, sample_length, time_to_outcome):
    endpoint = icu_df['outcome'].values[0]

    # last 00hrs before 00hrs from outcomes for outcome groups
    if endpoint == 1:
        icu_df['sample_start'] = icu_df['proxyend_time'] - pd.Timedelta(hours=sample_length + time_to_outcome)
        sampled_df = icu_df[(icu_df['recorded_time'] >= icu_df['sample_start'])
                        & (icu_df['recorded_time'] < (icu_df['sample_start'] + pd.Timedelta(hours=sample_length)))]
        sampled_df.sort_values(by=['dummy_encounter_id', 'recorded_time'], inplace=True)

    # random slice before 00hrs from outcomes for survival groups
    if endpoint == 0:
        icu_st = icu_df.groupby('dummy_encounter_id').first()
        icu_st['upper_bound'] = icu_st['proxyend_time'] - pd.Timedelta(hours=(sample_length + time_to_outcome))
        icu_st['gap_unit'] = (icu_st['upper_bound'] - icu_st['adm_time']) / pd.Timedelta(minutes=60)

        # Randomly draw start time of slices
        sample_start = []
        np.random.seed(0)
        for i, unit in enumerate(icu_st['gap_unit'].to_list()):
            try:
                starttime = icu_st['adm_time'].iloc[i] + np.random.choice(int(unit + 1)) * pd.Timedelta(minutes=60)
            except:
                print(unit)
                starttime = icu_st['adm_time'].iloc[i]
            sample_start.append(starttime)
        icu_st['sample_start'] = sample_start
        # print('before concate; ', len(icu_st.index))
        icu_sm = pd.merge(icu_df, icu_st['sample_start'], right_index=True, left_on='dummy_encounter_id')
        # print('after concate; ', len(icu_sm['dummy_encounter_id'].unique()))
        # slice
        sampled_df = icu_sm[(icu_sm['recorded_time'] >= icu_sm['sample_start'])
                        & (icu_sm['recorded_time'] < icu_sm['sample_start'] + pd.Timedelta(hours=sample_length))]
        sampled_df.sort_values(by=['dummy_encounter_id', 'recorded_time'], inplace=True)
    # print('after slicing; ', len(icu_sm['dummy_encounter_id'].unique()))

    return sampled_df


def select_column(base=True, vitals=True, v_order=False, med_order=False, comments=False, notes=False, nlp_topic=False):
    _BASE = [
        'dummy_encounter_id',
        'adm_time',
        'dt_start',
        'outcome',
        'sample_start',
        'time_of_day_minute']
    _VITALSIGNS = [
        'hr_entered',
        'rr_entered',
        'bp_entered',
        'temp_entered',
        'spo2_entered']
    _VITALORDERENTRY = [
        'one_vital',
        'set_vital']
    _MEDORDERENTRY = [
        'prn',
        'withheld']
    _FLOAWSHEETCOMMENT = [
        'hr_comment',
        'rr_comment',
        'bp_comment',
        'temp_comment',
        'spo2_comment']
    _NOTES = [
        'notes']
    _NLPTOPIC = [
        'Fall down',
        'Abnormal rate rhythm depth and effort of respirations',
        'Abnormal Mental State',
        'Communication problem',
        'cognitive defects',
        'Impaired blood oxygen',
        'Delusions',
        'General concern',
        'Hallucinations',
        'Chest Pain',
        'Mood disorder',
        'Abnormal Blood Pressure',
        'Abnormal Heart Rhythm',
        'Weight alteration',
        'Improper renal function',
        'abnormal rate rhythm depth and effort of respirations_1',
        'Violence Gesture',
        'Abnormal lab test',
        'Restraint',
        'Aspiration',
        'Suicide Risk',
        'Abnormal Temperature',
        'Monitoring',
        'Incisional pain',
        'cranial nerve palsies',
        'Musculoskeletal Pain',
        'Sign Symptoms of infection',
        'ataxic patterns',
        'hypocalcemia',
        'seizure',
        'pain duration',
        'Diagnosis related with Infection',
        'Improper Airway Clearance',
        'abnormal reflex',
        'Acute onset pain',
        'Abuse',
        'Localized pain',
        'pain killer',
        'Back Pain',
        'Fluid Volume Alteration',
        'Dysuria',
        'Arthralgia',
        'delirium',
        'Cutaneous Pain',
        'Oxygen response',
        'headache',
        'Medication related with Infection']

    colname = []
    if base:
        colname += _BASE
    if vitals:
        colname += _VITALSIGNS
    if v_order:
        colname += _VITALORDERENTRY
    if med_order:
        colname += _MEDORDERENTRY
    if comments:
        colname += _FLOAWSHEETCOMMENT
    if notes:
        colname += _NOTES
    if nlp_topic:
        colname += _NLPTOPIC
    return colname
