# process_well.py
# Created: 2023-11-18
# Version 11.18.2023
# Author: Jiawei Li
import os
import re
import requests
import numpy as np
import pandas as pd
from pyproj import Transformer
import sys
import xarray
import datetime
from itertools import compress
from progress.bar import Bar
from pathlib import Path



def format_well(welllog, upscale=1):
    if isinstance(welllog, pd.DataFrame):
        print('Will reuse cached well log in memory')
        return welllog
    def fill(group, factor=100):
        newgroup = group.loc[group.index.repeat(group.Thickness * factor)]
        mul_per_gr = newgroup.groupby('Elevation').cumcount()
        newgroup['Elevation'] = newgroup['Elevation'].subtract(mul_per_gr * 1 / factor)
        newgroup['Depth1_m'] = newgroup['Depth1_m'].subtract(mul_per_gr * 1 / factor)
        newgroup['Depth2_m'] = newgroup['Depth1_m'].add(1 / factor)
        newgroup['Elevation_End'] = newgroup['Elevation'].subtract(1 / factor)
        newgroup['Thickness'] = 1 / factor
        return newgroup
    transformer = Transformer.from_crs('epsg:4326', 'epsg:32612')
    lithology = pd.read_excel(welllog, sheet_name='Lithology').drop_duplicates()
    location = pd.read_excel(welllog, sheet_name='Location').drop_duplicates()
    lithology['Thickness'] = lithology['Depth2_m'].subtract(lithology['Depth1_m'])
    lithology_group = lithology.groupby('Bore')
    concatlist = []
    bar = Bar('Formating well logs', max=len(list(lithology_group.groups.keys())))
    bar.check_tty = False
    for name, group in lithology_group:
        group_location = location[location['Bore'] == name]
        if group_location.empty:
            group_location = location[location['Bore'] == name.strip()]
            if group_location.empty:
                continue
        long = group_location['X'].iloc[0]
        lat = group_location['Y'].iloc[0]
        x, y = transformer.transform(lat, long)
        group['Elevation'] = group_location['Z'].iloc[0]
        group['Elevation'] = group['Elevation'].subtract(group['Depth1_m'])
        group['UTMX'] = x
        group['UTMY'] = y
        group['Thickness'] = group['Depth2_m'].subtract(group['Depth1_m'])
        if upscale == 1:
            newgroup = group
        else:
            newgroup = fill(group, factor=upscale)
        concatlist.append(newgroup)
        bar.next()
    upscalled_well = pd.concat(concatlist)
    upscalled_well.reset_index(drop=True, inplace=True)
    conditionlist = [
        (upscalled_well["Keyword"] == "fine grain"),
        (upscalled_well["Keyword"] == "mix grain"),
        (upscalled_well["Keyword"] == "coarse grain")
    ]
    choicelist = [1, 2, 3]
    upscalled_well["Keyword_n"] = np.select(conditionlist, choicelist)
    upscalled_well['Bore'] = upscalled_well['Bore'].str.strip()
    bar.finish()
    return upscalled_well

def dl_usgs_water(wellname):
    workdir = os.getcwd()
    dlpath = Path(workdir+'\\'+ wellname)
    if wellname.isdigit():
        usgs = wellname
    else:
        try:
            usgs = re.findall('\d+',wellname)[0]
        except:
            raise("{} is not a usgs well name format, e.g.:'375006112554801'".format(wellname))
    url_1 = r'https://nwis.waterdata.usgs.gov/nwis/gwlevels?site_no=' + \
             usgs + r'&agency_cd=USGS&format=rdb'
    try:
        report = requests.get(url_1, stream=True)
    except:
        raise("Download failed! Check the Internet connection or {} is not reachable anymore".format(url_1))
    with open(usgs, 'wb') as f:
        for ch in report:
            f.write(ch)

    url_2 = r'https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=' + usgs
    pattern = re.compile(
        r'<title>USGS (.*\d)  .*?</title>.*?Latitude  (.*?), &nbsp; Longitude (.*?) &nbsp; (.*?)<br />.*?Well depth: (.*?) .*?Land surface altitude:  (.*?)'
    'feet above')
    pattern_coor = r'[&#;\\\'" ]'
    try:
        source = requests.get(url_2)
    except:
        raise("Download failed! Check the Internet connection or {} is not reachable anymore".format(url_1))
    match = re.findall(pattern, str(source.content))
    sitename = match[0][0]
    prelat = re.split(pattern_coor, match[0][1])
    lat = float(prelat[0]) + float(prelat[3]) / 60 + float(prelat[5]) / (60 * 60)
    prelong = re.split(pattern_coor, match[0][2])
    long = -(float(prelong[0]) + float(prelong[3]) / 60 + float(prelong[5]) / (60 * 60))
    datum = match[0][3]
    try:
        well_depth = float(match[0][4])
    except:
        well_depth = np.nan
    try:
        altitude = float(re.split(pattern_coor, match[0][5])[0].replace(',',''))
    except:
        altitude = np.nan
    metadata = pd.Series({'wellname': sitename, 'lat': lat, 'long': long, 'datum': datum,
                          'well_depth':well_depth, 'altitude':altitude})
    report = pd.read_fwf(usgs)
    return report, metadata

def format_usgs_water(usgs_well_NO, elevation_type='NAVD88'):
    report = pd.DataFrame()
    if isinstance(usgs_well_NO, str):
        try:
            report = pd.read_fwf(usgs_well_NO)
            _, meta = dl_usgs_water(usgs_well_NO)
        except:
            report, meta = dl_usgs_water(usgs_well_NO)
    elif isinstance(usgs_well_NO, pd.DataFrame):
        report = usgs_well_NO
    #because all reports are different in column so we need to find the exact column the header start
    try:
        row_start = report[report.apply(lambda row: row.astype(str).str.contains('agency_cd\tsite_no').any(), axis=1)].index.values[0]
    #https://datascientyst.com/search-for-string-whole-dataframe-pandas/
        messyinfo = report.iloc[:row_start, :]#other inforotion above header
        data = report.iloc[row_start:, :].copy()#real data
        data.rename(columns=data.iloc[0], inplace=True) #set data header
        data = data.iloc[2:, :]# reset data region
        columns = data.columns[0].split('\t') + data.columns[1:].to_list() #split columns they looks like aaa\tbbb\tccc
        split_data = list(data.iloc[:, 0].str.split('\t').values)#data also the same format
        formatdata = pd.DataFrame(split_data, columns=columns)#recombine the data into dataframe
        if elevation_type in ['NAVD88','NGVD29']:
            formatdata = formatdata[(formatdata['sl_lev_va'] != '') & (formatdata['sl_datum_cd'] == 'NAVD88') ]
        elif elevation_type.lower() == 'depth':
            formatdata = formatdata[formatdata['lev_va'] != '']
        formatdata['lev_dt'] = pd.to_datetime(formatdata['lev_dt'], format='%Y-%m-%d') #format to only keep year
        formatdata.reset_index(inplace=True, drop=True)
        _, meta = dl_usgs_water(formatdata.site_no[0]) #get metadata (lat long wellname datum)
        ds = xarray.Dataset.from_dataframe(formatdata) #convert dataframe to xarray format dataset
        ds = ds.set_index(index='lev_dt').to_array() #set index and convert to dataarray
        ds = ds.rename(meta.wellname) #set name
        ds = ds.rename({'index': 'time', 'variable': 'header'}) #set dimention names
        ds.attrs = meta.to_dict() #set attributes
        ds = ds.to_dataset() #convert back to dataset
        print("{} Done!".format(usgs_well_NO))
    except:
        ds = None
    return ds

def water_head_format(ds,time='2020-3',header='lev_va',elevation=None):
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
            concatlist = concatlist + [i[0]]
    df[header+time] = concatlist
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
    df[header+time] = df[header+time].astype(float).div(3.2808)#ft to m
    if elevation is not None:
        ele = pd.read_csv(elevation)
        df = pd.merge_asof(df.sort_values('UTMX'), ele.sort_values('X'), left_on='UTMX',
                           right_on='X', direction='nearest')
        df['water_elevation'] = df['PAROWANDEM'].subtract(df[header+time])
    return df





