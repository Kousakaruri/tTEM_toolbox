# Todo: seperate ttEM into above and below water table, run bootstrap for two parts to see result
import os
import numpy as np
import tTEM_tool as tt
import plotly.express as px
import plotly.graph_objects as go
import glob
import re
import xarray as xr
import pandas as pd
from itertools import compress
from pyproj import Transformer
from scipy.stats import pearsonr
import sys


def splt_ttem(ttem_df, gwsurface_result):
    def get_distance(group1, group2):
        dis = np.sqrt((group1[0] - group2[0]) ** 2 + (group1[1] - group2[1]) ** 2)
        return dis
    abv_water_table = []
    blw_water_table = []
    ttem_groups = ttem_df.groupby(['UTMX', 'UTMY'])
    well_location = gwsurface_result[['UTMX', 'UTMY']].values
    for name, group in ttem_groups:
        ttem_xy = list(group[['UTMX', 'UTMY']].iloc[0])
        ttem_well_distance = list(map(lambda x: get_distance(ttem_xy, x), well_location))
        match = gwsurface_result.iloc[ttem_well_distance.index(min(ttem_well_distance))]
        elevation = match['water_elevation']
        ttem_abv = group[group['Elevation_End'] >= elevation]
        abv_water_table.append(ttem_abv)
        ttem_blw = group[group['Elevation_End'] < elevation]
        blw_water_table.append(ttem_blw)

    ttem_above = pd.concat(abv_water_table)
    ttem_below = pd.concat(blw_water_table)
    return ttem_above, ttem_below


#####Export html
def save_html(ttem_result, pack_bootstrap, title, distance, path):
    os.chdir(path)
    water_trace = tt.plot.generate_trace(water_format, 'water')
    fig_ttem = go.Figure()
    fig_ttem.add_trace(tt.plot.generate_trace(ttem_result, 'ttem'))
    fig_ttem.add_trace(water_trace)
    fig_ttem.update_layout(title=title)
    fig_pack = plot_bst(pack_bootstrap)
    fig_pack.update_layout(title=title)
    return fig_ttem.write_html(title + str(distance)+'.html'), fig_pack.write_html(title + str(distance)+'bootstrap.html')

def plot_bst(dataframe):
    """
    plot bootstrap result

    :param dataframe:
    :return: plotly fig
    """
    fig_hist = go.Figure()
    fig_hist.data = []
    fig_hist.add_trace(go.Histogram(x=dataframe.fine, name='Fine', marker_color='Blue', opacity=0.75))
    fig_hist.add_trace(go.Histogram(x=dataframe.coarse, name='Coarse', marker_color='Red', opacity=0.75))
    if dataframe.mix.sum() == 0:
        print("skip plot mix because there is no data")
    else:
        fig_hist.add_trace(go.Histogram(x=dataframe.mix, name='Mix', marker_color='Yellow', opacity=0.75))
    return fig_hist

def plot_ttem_well(ttem, bootstrappack):
    """
    plot ttem with selected well after process
    :param ttem:
    :param bootstrappack:
    :return:
    """

    def check_filtered_well(pre_bootstrap, welllog):
        wellname = list(pre_bootstrap.groupby('Bore').groups.keys())
        filtered_well = welllog[welllog['Bore'].isin(wellname)]
        trace = tt.plot.generate_trace(filtered_well, 'well')
        return trace

    fig = go.Figure()
    fig.add_trace(tt.plot.generate_trace(ttem, 'ttem'))
    fig.add_trace(check_filtered_well(bootstrappack[0], bootstrappack[-1]))
    return fig


if __name__ == '__main__':
    os.chdir(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah')
    well_info = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah\USGSdownload\NWISMapperExport.xls'
    location = r"C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\Gamma\location.csv"
    welllog = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\Well_log.xlsx'
    elevation = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah\usgs_water_elevation.csv'
    ttemname = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\PD1_I01_MOD.xyz'
    ttemname2 = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\PD22_I03_MOD.xyz'
    DOI = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\DOID1_DOIStaE.xyz'
    time = '2022-3'
    water = tt.main.GWSurface(waterwell=well_info, elevation_type='depth')
    water_format = water.format(elevation=elevation, time=time)
    # get raw ttem data for entire valley
    def runall(distance=1000, water_format=water_format, workpath=[], check_corr=np.nan):
        welllog = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\Well_log.xlsx'
        ttem = tt.main.ProcessTTEM(ttem_path=[ttemname, ttemname2],
                                   welllog=welllog,
                                   DOI_path=DOI,
                                   layer_exclude=[1, 2, 3])
        data = ttem.data()
        ttem_match, well_match = tt.bootstrap.select_closest(data,
                                                             tt.process_well.format_well(welllog, upscale=100),
                                                             distance=distance,
                                                             showskip=False)
        welllog, correxport = tt.bootstrap.corr_well_filter(ttem_match,
                                                            well_match,
                                                            corr_thershold=0.3,
                                                            correxport=True)

        ####plot match ttem and matched well seperately
        # get water elevation data
        ttem_above_wt, ttem_below_wt = splt_ttem(data, water_format)
        bootstrap_above = tt.main.ProcessTTEM(ttem_path=ttem_above_wt, welllog=welllog)
        bootstrap_above_result = bootstrap_above.ttem_well_connect(distance=distance, debug=True)
        bootstrap_below = tt.main.ProcessTTEM(ttem_path=ttem_below_wt, welllog=welllog)
        bootstrap_below_result = bootstrap_below.ttem_well_connect(distance=distance, debug=True)
        save_html(ttem_above_wt, bootstrap_above_result[2], 'ttem_above_wt', distance,workpath)
        save_html(ttem_below_wt, bootstrap_below_result[2], 'ttem_below_wt', distance,workpath)
        # seperate for north and center part
        ttem_n = tt.main.ProcessTTEM(ttem_path=ttemname,
                                     welllog=welllog,
                                     DOI_path=DOI,
                                     layer_exclude=[1, 2, 3],
                                     line_exclude=[])
        data_n = ttem_n.data()
        ttem_above_wt_n, ttem_below_wt_n = splt_ttem(data_n, water_format)
        bootstrap_above_n = tt.main.ProcessTTEM(ttem_path=ttem_above_wt_n, welllog=welllog)
        bootstrap_above_n_result = bootstrap_above_n.ttem_well_connect(distance=distance, debug=True, check_corr=check_corr)
        bootstrap_below_n = tt.main.ProcessTTEM(ttem_path=ttem_below_wt_n, welllog=welllog, )
        bootstrap_below_n_result = bootstrap_below_n.ttem_well_connect(distance=distance,debug=True,check_corr=check_corr)
        save_html(ttem_above_wt_n, bootstrap_above_n_result[2], 'ttem_above_wt_n', distance, workpath)
        save_html(ttem_below_wt_n, bootstrap_below_n_result[2], 'ttem_below_wt_n', distance, workpath)

        ttem_ct = tt.main.ProcessTTEM(ttem_path=ttemname2,
                                      welllog=welllog,
                                      DOI_path=DOI,
                                      layer_exclude=[1, 2, 3])

        data_ct = ttem_ct.data()
        ttem_above_wt_ct, ttem_below_wt_ct = splt_ttem(data_ct, water_format)
        bootstrap_above_ct = tt.main.ProcessTTEM(ttem_path=ttem_above_wt_ct, welllog=welllog)
        bootstrap_above_ct_result = bootstrap_above_ct.ttem_well_connect(distance=distance, debug=True,check_corr=check_corr)
        bootstrap_below_ct = tt.main.ProcessTTEM(ttem_path=ttem_below_wt_ct, welllog=welllog)
        bootstrap_below_ct_result = bootstrap_below_ct.ttem_well_connect(distance=distance, debug=True,check_corr=check_corr)
        save_html(ttem_above_wt_ct, bootstrap_above_ct_result[2], 'ttem_above_wt_ct', distance, workpath)
        save_html(ttem_below_wt_ct, bootstrap_below_ct_result[2], 'ttem_below_wt_ct', distance, workpath)
        return bootstrap_above_result, \
               bootstrap_below_result, \
               bootstrap_above_n_result, \
               bootstrap_below_n_result,\
               bootstrap_above_ct_result, \
               bootstrap_below_ct_result

#####For all wells check data correlation######


#####check filtered wells
####### check water elevation different
    water_format22_3 = water.format(elevation=elevation, time='2022-3')
    workdir = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\2022\Paper\meeting220822\htmls'
    result500 = runall(500, water_format22_3, workdir,check_corr=0.3)
    result1000 = runall(1000, water_format22_3, workdir, check_corr=0.3)
    result100000 = runall(100000, water_format22_3, workdir, check_corr=0.3)
    water_format14_3 = water.format(elevation=elevation, time='2014-3')
    water_format13_11 =  water.format(elevation=elevation, time='2013-11')
    water_format13_3 =  water.format(elevation=elevation, time='2013-3')
    water_format12_11 = water.format(elevation=elevation, time='2012-11')
    elevation_diff12_13 = water_format12_11['water_elevation'].subtract(water_format13_3['water_elevation'])
    elevation_diff13_14 = water_format13_11['water_elevation'].subtract(water_format14_3['water_elevation'])
    figwater = go.Figure()
    figwater.add_trace(tt.plot.generate_trace(water_format12_11,'water'))
    figwater.data[-1].name='2012-11'
    figwater.add_trace(tt.plot.generate_trace(water_format13_3,'water'))
    figwater.data[-1].name='2013-3'
    figwater.add_trace(tt.plot.generate_trace(water_format13_11,'water'))
    figwater.data[-1].name='2013-11'
    figwater.add_trace(tt.plot.generate_trace(water_format14_3,'water'))
    figwater.data[-1].name='2014-3'

    water_format22_3['water_elevation'] = water_format22_3['water_elevation'].add(elevation_diff12_13)

    workdir12 = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\2022\Paper\meeting220822\htmls\aftercorrection\correct12_13'
    result_crt12_500 = runall(500, water_format22_3, workdir12,check_corr=0.3)
    result_crt12_1000 = runall(1000, water_format22_3, workdir12, check_corr=0.3)
    result_crt12_100000 = runall(100000, water_format22_3, workdir12, check_corr=0.3)

    water_format22_3 = water.format(elevation=elevation, time='2022-3')
    water_format22_3['water_elevation'] = water_format22_3['water_elevation'].add(elevation_diff13_14)
    workdir13 = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\2022\Paper\meeting220822\htmls\aftercorrection\correct13_14'
    result_crt13_500 = runall(500, water_format22_3, workdir13, check_corr=0.3)
    result_crt13_1000 = runall(1000, water_format22_3, workdir13, check_corr=0.3)
    result_crt13_100000 = runall(100000, water_format22_3, workdir13, check_corr=0.3)