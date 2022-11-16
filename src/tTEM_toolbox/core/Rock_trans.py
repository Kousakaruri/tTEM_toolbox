from numba import jit
import numpy as np
import pandas as pd
import sys

def rock_transform(ttem_data, Resi_conf_df):
    rock_trans = ttem_data.copy()
    rock_trans["Identity"] = np.nan
    rock_trans["Identity_n"] = np.nan
    output_df = pd.DataFrame(columns=rock_trans.columns)
    conditionlist = [
        (ttem_data["Resistivity"] <= Resi_conf_df.Fine_conf.iloc[1]),
        ((Resi_conf_df.Coarse_conf.iloc[0] >ttem_data["Resistivity"])
         &(ttem_data["Resistivity"]>Resi_conf_df.Fine_conf.iloc[1])),
        (ttem_data["Resistivity"] >= Resi_conf_df.Coarse_conf.iloc[0])
    ]
    choicelist = [1, 2, 3]
    choicelist2 = ["Fine_grain","Mix_grain","Coarse_grain"]
    rock_trans["Identity"] = np.select(conditionlist, choicelist2)
    rock_trans["Identity_n"] = np.select(conditionlist, choicelist)
    return rock_trans

def pct_count(rk_transform_result, grain=False):
    grainsum = rk_transform_result.groupby(['UTMX', 'UTMY', 'Identity']).sum()
    grainsum.reset_index(inplace=True)
    thicksum = rk_transform_result.groupby(['UTMX', 'UTMY']).agg({'Thickness': 'sum'})
    thicksum.rename(columns={'Thickness': 'T_sum'}, inplace=True)
    thicksum.reset_index(inplace=True)
    ratio = pd.merge(grainsum, thicksum, on=['UTMX', 'UTMY'])
    ratio['ratio'] = round(ratio['Thickness'].div(ratio['T_sum']), 2)
    if grain is not False:
        try:
            ratio = ratio[ratio['Identity'] == grain]
        except:
            raise ('{} is not one of Fine_grain, Mix_grain, or Coarse_grain keyword'.format(grain))
    return ratio
'''
    for index, row in ttem_data.iterrows():
        if row.Resistivity <= Resi_conf_df.Fine_conf.iloc[1]:
            row["Identity"] = "Fine_grain"
            row["Identity_n"] = 0
        elif Resi_conf_df.Coarse_conf.iloc[0] > row.Resistivity> Resi_conf_df.Fine_conf.iloc[1]:
            row["Identity"] = "Mix_grain"
            row["Identity_n"] = 1
        elif row.Resistivity >= Resi_conf_df.Coarse_conf.iloc[0]:
            row["Identity"] = "Coarse_grain"
            row["Identity_n"]] = 2
        df = ttem_data
'''

