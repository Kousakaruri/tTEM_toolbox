import os
import numpy as np
import tTEM_tool as tt
import plotly.express as px
import plotly.graph_objects as go
import glob
import re
import xarray as xr
import pandas as pd
import dask
from dask.delayed import delayed
from itertools import compress
from pyproj import Transformer
import sys
location = r"B:\\Code\\Python\\Gamma\\location.csv"
welllog = r'B:\Code\Python\tTEM_test\Plot_with_well_log\Well_log.xlsx'
ttemname = r'B:\Code\Python\tTEM_test\Plot_with_well_log\PD22_I03_MOD.xyz'
ttemname2 = r'B:\Code\Python\tTEM_test\Plot_with_well_log\PD1_I01_MOD.xyz'
DOI = r'B:\Code\Python\tTEM_test\Plot_with_well_log\DOID1_DOIStaE.xyz'
def fill(group, factor=100):
    newgroup = group.loc[group.index.repeat(group.Thickness * factor)]
    mul_per_gr = newgroup.groupby('Elevation_Cell').cumcount()
    newgroup['Elevation_Cell'] = newgroup['Elevation_Cell'].subtract(mul_per_gr * 1 / factor)
    newgroup['Depth_top'] = newgroup['Depth_top'].add(mul_per_gr * 1 / factor)
    newgroup['Depth_bottom'] = newgroup['Depth_top'].add(1/factor)
    newgroup['Elevation_End'] = newgroup['Elevation_Cell'].subtract(1/factor)
    newgroup['Thickness'] = 1 / factor
    return newgroup
def upscale(ttem_data, factor=100):
    concatlist = []
    groups = ttem_data.groupby(['UTMX','UTMY'])
    total = len(list(groups.groups.keys()))
    count = 0
    for name, group in groups:
        newgroup = fill(group, factor)
        concatlist.append(newgroup)
        ### small progress bar
        count += 1
        print('\r', end='')
        print("Progress {}/{}".format(count, total), end='')
        sys.stdout.flush()
    result = pd.concat(concatlist)
    result.reset_index(drop=True, inplace=True)
    return result
ttem = tt.main.ProcessTTEM(ttem_path=ttemname2,
                              welllog=welllog,
                              DOI_path=DOI,
                              layer_exclude=[1, 2],
                              line_exclude=[180])
data = ttem.data()
#fig.add_trace(tt.plot.generate_trace(data, 'ttem'))
pre_bootstrap_ct, rk_trans_ct, pack_bootstrap_result_ct, Resi_conf_df_ct = ttem.ttem_well_connect()
#pre_bootstrap,rk_trans, pack_bootstrap_result, Resi_conf_df=ttem.ttem_well_connect()

