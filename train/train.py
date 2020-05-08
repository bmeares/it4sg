#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
WIP
"""

import pandas as pd
import numpy as np
import keras
import dateutil
from sklearn.model_selection import train_test_split
class main:
   
    def __init__(self):
        self.weights_filename = "Landshut.h5"
        self.df = pd.read_csv("data.csv")
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        self.df['year_percent'] = self.df['timestamp'].apply(lambda x : ((((x.dayofyear * 24) - 1) + x.hour) / 8760) % 1)
        self.normalize()

        self.create_arrays()
        self.create_model()

        choice = input("Train the model? (Y/n)")
        if choice.lower() == "y" or len(choice) == 0:
            self.train_model()

        self.model.load_weights(self.weights_filename)
        #  self.model.predict()
    
    # add normalized columns to df attribute
    def normalize(self):
        float_cols = ['Netz_Wirkleistung', 'PV_Wirkleistung',
                'Last_Rechenwert', 'Niederschlag',
                'Temperatur', 'rel_Feuchte',
                'Windgeschwindigkeit', 'Windrichtung',
                'Luftdruck']
        for col in float_cols:
            norm_col = 'normalized_' + col
            minimum = self.df[col].min()
            maximum = self.df[col].max()
            self.df[norm_col] = self.df[col].apply(lambda x : (x - minimum) / (maximum - minimum))

        bool_cols = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag',
                'Sonntag', 'Feiertag']
        for col in bool_cols:
            self.df[col] = self.df[col].apply(lambda x : int(1) if x is True else int(0))


        columns = ['normalized_' + c for c in float_cols[1:]] + bool_cols
        output_col = 'normalized_' + float_cols[0]
        columns.append(output_col)

        self.norm_df = self.df[columns]
        self.norm_df.to_csv("normalized.csv", index=False)

    # concatenate all rows and create labels
    def create_arrays(self):
        #  array = np.array(self.norm_df.iloc(0))
        output_col = "normalized_Netz_Wirkleistung"
        self.out_array = self.norm_df[output_col].to_numpy()
        self.array = self.norm_df.drop(output_col, axis=1).to_numpy()

    # create model
    def create_model(self):
        self.model = keras.Sequential()
        num_features = len(self.array[0])
        #  self.model.add(keras.layers.Input(shape=(num_features,)))
        self.model.add(keras.layers.Dense(int(num_features), input_shape=(int(num_features),)))
        self.model.add(keras.layers.Dense(int(num_features / 2)))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

        opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return True

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.array, self.out_array, test_size=0.3, random_state=42)
        history = self.model.fit(X_train, y_train, epochs=1, batch_size=32)

        eval_report = self.model.evaluate(x=X_test, y=y_test, batch_size=32, verbose=1,
                sample_weight=None, steps=None)

        print(history.history)

        # save model
        self.model.save(self.weights_filename)
        return True


if __name__ == "__main__":
    m = main()
