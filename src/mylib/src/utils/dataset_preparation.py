import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder
from .utils import get_vocab, get_max_cat_len, get_fitted_discretizer, get_fitted_scaler

import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)


class OrderDataset:
    def __init__(self, prepared_folder, look_back):
        self.prepared_folder = prepared_folder

        self.train_file = 'train.csv'
        self.test_file = 'test.csv'
        self.valid_file = 'valid.csv'

        self.look_back = look_back

        self.id = 'id'
        self.categorical = 'label'
        self.amount = 'amount'
        self.date = 'date'
        self.dt = 'dt'

    def prepare_features(self, df, categorical_vocab, id_vocab):
        """
        Encode categorical feature with OrdinalEncoder.
        Process date column.
        Remove unnecessary columns.
        """
        df_upd = df[[self.id, self.categorical, self.amount, self.date]]
        if df_upd[self.date].dtype == 'O':
            df_upd[self.date] = pd.to_datetime(df_upd[self.date])

        cat_encoder = OrdinalEncoder(categories=categorical_vocab, dtype=np.int64)
        df_upd[self.categorical] = cat_encoder.fit_transform(df_upd[self.categorical].values.reshape(-1, 1))
        id_encoder = OrdinalEncoder(categories=id_vocab, dtype=np.int64)
        df_upd[self.id] = id_encoder.fit_transform(df_upd[self.id].values.reshape(-1, 1))

        df_final = df_upd.groupby([self.id, self.date, self.categorical]).agg({self.amount: 'sum'}).reset_index()
        return df_final

    def group_rows(self, df):
        """
        Rows, which relate to the same day, are combined into one row.
        """
        df_copy = df.copy(deep=True)
        df_grouped = df_copy.groupby([self.id, self.date]).agg({self.categorical: lambda x: list(x),
                                                                self.amount: lambda x: list(x)}).reset_index()
        return df_grouped

    def add_time_difference(self, df):
        """
        Add column 'dt' with time difference between orders.
        """
        all_differences = []
        interm_df = df.groupby(self.id)[self.date].apply(lambda x: x.diff())
        for ind in interm_df.index:
            try:
                corr_diff = np.nan_to_num(interm_df.iloc[ind].days).tolist()
            except:
                corr_diff = np.nan_to_num(interm_df.iloc[ind]).tolist()
            all_differences.extend(corr_diff if type(corr_diff) == list else [corr_diff])
        df.insert(2, self.dt, all_differences)

    def preprocess_dataframe(self):
        """
        Combine several steps of dataset preprocessing: encoding, scaling, grouping rows, adding 'dt' feature.
        """

        train = pd.read_csv(os.path.join(self.prepared_folder, self.train_file))
        test = pd.read_csv(os.path.join(self.prepared_folder, self.test_file))
        valid = pd.read_csv(os.path.join(self.prepared_folder, self.valid_file))

        categorical_vocab = get_vocab(train, test, valid, self.categorical)
        cat_vocab_size = categorical_vocab.shape[1]
        id_vocab = get_vocab(train, test, valid, self.id)
        id_vocab_size = id_vocab.shape[1]

        amount_discretizer = get_fitted_discretizer(train, test, valid, self.amount)
        amount_vocab_size = amount_discretizer.n_bins_[0]

        datasets = [train, test, valid]
        processed_datasets = []
        for df in datasets:
            prepared_df = self.prepare_features(df, categorical_vocab, id_vocab)
            prepared_df[self.amount] = amount_discretizer.transform(prepared_df[self.amount].values.reshape(-1, 1)).astype(np.int64)
            grouped_df = self.group_rows(prepared_df)
            self.add_time_difference(grouped_df)
            processed_datasets.append(grouped_df)

        max_cat_len = get_max_cat_len(*processed_datasets, self.categorical)

        dt_vocab = get_vocab(*processed_datasets, self.dt)
        dt_vocab_size = dt_vocab.shape[1]
        dt_encoder = OrdinalEncoder(categories=dt_vocab, dtype=np.int64)
        final_datasets = []
        for precessed_df in processed_datasets:
            precessed_df[self.dt] = dt_encoder.fit_transform(precessed_df[self.dt].values.reshape(-1, 1))
            final_datasets.append(precessed_df)

        return final_datasets, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, max_cat_len

    def window_combinations(self, df):
        """
        Specify indices of rows which we use as inputs and index of the row for which we make the prediction.
        """
        chunks_relevant_indices = []
        for num_id in df[self.id].unique():
            chunks_relevant_indices.append(df.loc[df[self.id] == num_id].index)

        all_combinations = []
        for relevant_indices in chunks_relevant_indices:
            for i in range(len(relevant_indices) - self.look_back):
                current_group = relevant_indices[i:i + self.look_back].tolist()
                index_to_pred = relevant_indices[i + self.look_back]
                all_combinations.append((current_group, index_to_pred))
        return all_combinations
