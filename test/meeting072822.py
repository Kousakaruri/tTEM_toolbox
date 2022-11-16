#goal
#plot the water level vs ttem data
#surface plot time series
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
import sys
#####Plot water elevation time series

#TODO: Apply Dask to the upscale function

'''
def dl2ds(filter='NAVD88'):
    """

    :return: return xarray dataset that include all availiable USGS wells
    """

    ds = xr.Dataset()
    problemtic = []
    tmp_list = []
    for index, row in well_info.iterrows():
        tmp_ds = tt.process_well.format_usgs_water(str(row['SiteNumber']), elevation_type=filter)

        try:
            ds = ds.merge(tmp_ds)
            #ds = ds.merge(tmp_ds)
        except:
            print('{} not able to merge, try to solve the problem by drop duplicates.'.format(str(row['SiteNumber'])))
            try:
                ds = ds.merge(tmp_ds[str(row['SiteNumber'])].drop_duplicates(dim='time').to_dataset())

            except:
                print('{} failed to merge'.format(str(row['SiteNumber'])))
                problemtic.append(str(row['SiteNumber']))
    print('All Wells Done!')
    return ds
def water_head_format(ds,time='2020-3',header='lev_va'):

    sitename = list(ds.keys())
    df = pd.concat([pd.DataFrame([da.attrs]) for varname, da in ds.data_vars.items()], axis=0)
    df.reset_index(drop=True,inplace=True)
    array = ds.sel(time=time,header=header).to_array().values.tolist()
    filter = list(map(lambda x: list(compress(x, ~pd.isna(x))), array))
    concatlist=[]
    for i in filter:
        if len(i) == 0:
            i = np.nan
            concatlist = concatlist + [i]
        else:
            concatlist = concatlist + i
    df[header] = concatlist
    transformer_27 = Transformer.from_crs('epsg:4267', 'epsg:32612')  # NAD27-->WGS84 UTM12N
    transformer_83 = Transformer.from_crs('epsg:4269', 'epsg:32612')  # NAD83-->WGS84 UTM12N
    NAD27 = df.groupby('datum').get_group('NAD27')
    NAD83 = df.groupby('datum').get_group('NAD83')
    UTMX27, UTMY27 = map(list,zip(*list(map(transformer_27.transform,NAD27['lat'].values, NAD27['long'].values))))
    #UTMX, UTMY = map(list, zip(*result))  split list of tuple into two lists
    NAD27 = NAD27.assign(UTMX=UTMX27,UTMY=UTMY27)
    UTMX83, UTMY83 = map(list,zip(*list(map(transformer_83.transform,NAD83['lat'].values, NAD83['long'].values))))
    NAD83 = NAD83.assign(UTMX=UTMX83,UTMY=UTMY83)
    df = pd.concat([NAD27,NAD83]).sort_index()
    return df
'''
def plot_water_elevation():
    time = np.datetime_as_string(np.arange(np.datetime64('2013-03'), np.datetime64('2022-03'),np.timedelta64(1,'Y')))
    time = [str(x) for x in time]
    water = tt.main.GWSurface(waterwell=well_info, elevation_type='depth', time=time)
    result = water.format()
    elevation = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah\usgs_water_elevation.csv'
    ele = pd.read_csv(elevation)
    result['elevation'] = ele['PAROWANDEM']
    #np.arange(np.datetime64('2013-03'), np.datetime64('2020-06'),np.timedelta64(1,'Y'))
    fig = go.Figure()
    for i in time:
        trace = go.Scatter3d(x=result['UTMX'].values,
                          y=result['UTMY'].values,
                          z=result['sl_lev_va'+i] ,
                          name=i)
        fig.add_trace(trace)
    return fig
##### add trace with tTEM data
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
#pre_bootstrap,rk_trans, pack_bootstrap_result, Resi_conf_df=ttem.ttem_well_connect()
def select_closest_NEW(ttemdata, welllog, distance=500):
    def get_distance(group1, group2):
        dis = np.sqrt((group1[0] - group2[0]) ** 2 + (group1[1] - group2[1]) ** 2)
        return dis
    concatlist = []
    concatwell = []
    count = 0
    ori_well = tt.process_well.format_well(welllog, upscale=False)
    groups_well = ori_well.groupby('Bore')
    total = len(list(groups_well.groups.keys()))
    ttem_location = list(ttemdata.groupby(['UTMX', 'UTMY']).groups.keys())
    for name, group in groups_well:
        wellxy = list(group[['UTMX','UTMY']].iloc[0])
        well_ttem_distance = list(map(lambda x: get_distance(wellxy, x), ttem_location))
        ### small progress bar
        count += 1
        print('\r', end='')
        print("Progress {}/{}".format(count, total), end='')
        sys.stdout.flush()
        ## small process bar
        minvalue = min(well_ttem_distance)
        if minvalue <= float(distance):
            point_match = ttem_location[well_ttem_distance.index(minvalue)]
            match = ttemdata[(ttemdata['UTMX'] == point_match[0])&(ttemdata['UTMY'] == point_match[1])]
            match['distance'] = minvalue

            concatlist.append(match)
            concatwell.append(group)
        else:
            print('skip {} due to min distance greater than limit {}, is {}'.format(name,distance,
                                                                                    round(minvalue)))
    matched_ttem = pd.concat(concatlist).reset_index(drop=True)
    matched_well = pd.concat(concatwell).reset_index(drop=True)
    return matched_ttem, matched_well
def ttem_well_lithology(ttemdata, welllogdf):
    resistivity_list = []
    for index, row in welllogdf.iterrows():
        elev_win_top = row['Elevation1']
        elev_win_bot = row['Elevation2']
        match_block = ttemdata[(ttemdata['Elevation_Cell'] <= elev_win_top) &
                               (ttemdata['Elevation_Cell'] > elev_win_bot)]
        if not match_block.empty:
            resistivity_avg = match_block['Resistivity'].mean()
            resistivity_list.append(resistivity_avg)
        else:
            resistivity_list.append(np.nan)
            continue
    welllogdf['Resistivity'] = resistivity_list
    return welllogdf

if __name__=="__main__":
    os.chdir(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah')
    well_info = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah\USGSdownload\NWISMapperExport.xls'
    location = r"C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\Gamma\location.csv"
    welllog = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\Well_log.xlsx'
    ttemname = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\PD1_I01_MOD.xyz'
    ttemname2 = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\PD22_I03_MOD.xyz'
    DOI = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\DOID1_DOIStaE.xyz'
    ttem = tt.main.ProcessTTEM(ttemname=[ttemname, ttemname2],
                               welllog=welllog,
                               DOI=DOI,
                               layer_exclude=[1, 2, 3, 4])
    data = ttem.data()
    # fig.add_trace(tt.plot.generate_trace(data, 'ttem'))
    result = upscale(data, factor=100)
    matched_ttem, matched_well = select_closest_NEW(result, welllog)
    welllogdf = ttem_well_lithology(matched_ttem, matched_well)
    welllogdf = welllogdf[welllogdf['Resistivity'].notna()]
    mean_r = welllogdf.groupby(['Bore', 'Keyword'])['Resistivity'].mean()
    mean_r = mean_r.to_frame().reset_index()
    fig = px.histogram(mean_r, x='Bore', y='Resistivity', color='Keyword', barmode='group', color_discrete_map={
        "fine grain": "blue",
        "mix grain": "yellow",
        "coarse grain": "red"
    })
    well_group = welllogdf.groupby('Bore')
    for name, group in well_group:
        elevation = group['Elevation1'] - group['Elevation2']
        figwell = px.bar(group, x='Resistivity', y='Elevation1', color='Keyword', text='Keyword',
               orientation='h', title=name,
               color_discrete_map={
                   "fine grain": "blue",
                   "mix grain": "yellow",
                   "coarse grain": "red"
               })
        #figwell.update_traces(width=5)
        figwell.write_html(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\2022\Paper\wells'+name+'.html')
