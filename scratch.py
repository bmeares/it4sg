#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 bmeares <bmeares@G752VT>
#
# Distributed under terms of the MIT license.

"""

"""
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "data/combinedData_cleared.csv"
base_df = pd.read_csv(INPUT_FILE, index_col="timestamp", parse_dates=True)
drop_cols = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag',
    'Sonntag', 'Feiertag'] 
df = base_df.drop(columns=drop_cols)

#  plt.matshow(base_df.corr())
#  plt.show()
print(df.corr())


f = plt.figure()
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=8, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=8)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=8)
#  plt.title('Correlation Matrix', fontsize=12);

plt.show()
