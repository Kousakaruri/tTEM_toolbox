import sys
import pandas as pd
from pyproj import Transformer
from . import process_well
import numpy as np
import re


def load_gamma(fname, **kwargs):
    #gamma_file_path = folder
    #gamma_file = glob.glob(gamma_file_path + '\*.csv')
    #exclude_keyword = re.compile(r'fluid')
    #filt_gamma_file = list(filter(lambda x: not exclude_keyword.search(x), gamma_file))
    if re.search('fluid', fname):
        raise ('{} does not contains gamma data'.format(fname))
    else:
        df = pd.read_csv(fname, skiprows=[1])
        df.columns = df.columns.str.strip()
        df = df.replace(-999, np.nan)
        filename = fname[fname.rfind("\\")+1:-4]
        df["comment"] = filename
        df_hold = df[["Depth","comment"]]
        if np.isin("columns", list(kwargs.keys())):
            tmp = df[kwargs["columns"]]
            df = pd.concat([tmp, df_hold],axis=1)
        else:
            raise ("{} not the availiable option, availiable option contains 'columns'".format(list(kwargs.keys())))
    return df

def georeference(df, geo):
    output = df
    location = pd.read_csv(geo)
    location.loc[:, "name"] = location.loc[:, "name"].str.lower()
    df_name = output.loc[:, "comment"][0].split()[0]
    try:
        match = location[location.loc[:, "name"].str.contains(df_name)]
    except:
        raise ("{} does not found a match in location document".format(df_name))
    x_ori = match.iloc[0]["X"]
    y_ori = match.iloc[0]["Y"]
    transformer = Transformer.from_crs('epsg:4326', 'epsg:32612')  # WGS84-->UTM12N
    x, y = transformer.transform(y_ori, x_ori)
    output.loc[:, "X"] = x
    output.loc[:, "Y"] = y
    output.loc[:, "Z"] = match.loc[:, 'Z'].values[0]
    output.loc[:, "Elevation"] = output.loc[:, "Z"].sub(output.loc[:, "Depth"])

    return output


def rolling_average(df, window=5):
    addon = ["Depth", 'X', 'Y', "Z"]
    roll_avg = df.rolling(
        window=window,
        axis=0).mean()
    roll_avg = df.dropna().reset_index(drop=True)
    roll_avg.loc[:, "comment"] = df.loc[0, "comment"]
    roll_avg.loc[:, "Elevation"] = roll_avg.loc[:, "Z"].sub(roll_avg.loc[:, "Depth"])
    return roll_avg


def gamma_well_connect(gamma_df, welllog):
    if isinstance(welllog, str):
        ori_well = process_well.format_well(welllog, upscale=False)
    elif isinstance(welllog, pd.DataFrame):
        ori_well = welllog
    else:
        raise("welllog has to be either str or DataFrame not {}".format(type(welllog)))
    gamma_place = list(gamma_df.groupby("comment").groups.keys())
    utmx = gamma_df.loc[:, "X"].values[0]
    utmy = gamma_df.loc[:, "Y"].values[0]
    ori_well.loc[:,"distance"] = np.sqrt((ori_well.loc[:,"UTMX"]-utmx)**2+
                                         (ori_well.loc[:,"UTMY"]-utmy)**2)
    well_closest = ori_well[ori_well["distance"] == ori_well.distance.min()].reset_index(drop=True)
    ori_well_with_gamma = pd.DataFrame(columns=ori_well.columns)
    ori_well_with_gamma["GR"] = np.nan
    for index, row in well_closest.iterrows():
        if row["distance"] > 1000:
            print("\n {} is too far away from welllog".format(gamma_df["comment"][0]))
            break
        toplimit = row["Elevation1"]
        botlimit = row["Elevation2"]
        if botlimit > gamma_df.loc[0, "Elevation"]:
            continue
        if toplimit > gamma_df.loc[index, "Elevation"]:
            toplimit = gamma_df.loc[index, "Elevation"]
            row["Elevation1"] = toplimit
            row["Depth1"] = toplimit

        tmp_gamma_df = gamma_df[(gamma_df.loc[:,"Elevation"] >= botlimit) &
                                (gamma_df.loc[:,"Elevation"] < toplimit)]
        gamma_total = tmp_gamma_df.loc[:,"GR"].sum()
        gamma_total_series = pd.Series({"GR": gamma_total})
        row = row.append(gamma_total_series)
        ori_well_with_gamma = ori_well_with_gamma.append(row, ignore_index=True)
    ori_well_with_gamma["GRM"] =ori_well_with_gamma["GR"].div(ori_well_with_gamma["Elevation1"].sub(ori_well_with_gamma["Elevation2"]))
    ori_well_with_gamma = ori_well_with_gamma.drop(["Depth1","Depth2","Depth1_m","Depth2_m"],axis=1)
    ori_well_with_gamma["comment"] = gamma_place[0]
    return ori_well_with_gamma


def gamma_ttem_connect(gamma_df, ttem):
    gamma_place = list(gamma_df.groupby("comment").groups.keys())
    utmx = gamma_df.loc[:, "X"].values[0]
    utmy = gamma_df.loc[:, "Y"].values[0]
    ttem.loc[:, "gamma_distance"] = np.sqrt((ttem.loc[:, "UTMX"] - utmx) ** 2 +
                                          (ttem.loc[:, "UTMY"] - utmy) ** 2)
    ttem_closest = ttem[ttem["gamma_distance"] == ttem.gamma_distance.min()].reset_index(drop=True)
    ttem_with_gamma = pd.DataFrame(columns=ttem.columns)
    ttem_with_gamma["GR"] = np.nan
    for index, row in ttem_closest.iterrows():
        ### small progress bar
        print('\r', end='')
        print("Progress {}/{}".format(index, ttem_closest.shape[0]), end='')
        sys.stdout.flush()
        ###
        toplimit = row["Elevation_Cell"]
        botlimit = row["Elevation_End"]
        if row["gamma_distance"] > 1000:
            print("\n {} is too far away from ttem lines".format(gamma_df["comment"][0]))
            break
        if botlimit > gamma_df.loc[0, "Elevation"]:
            continue
        if toplimit > gamma_df.loc[0, "Elevation"]:
            toplimit = gamma_df.loc[index, "Elevation"]
            row["Elevation_Cell"] = toplimit
            row["Elevation_End"] = row["Elevation_Cell"] - row["Thickness"]

        tmp_gamma_df = gamma_df[(gamma_df.loc[:,"Elevation"] >= botlimit) &
                                (gamma_df.loc[:,"Elevation"] < toplimit)]
        gamma_total = tmp_gamma_df.loc[:,"GR"].sum()
        if gamma_total == 0:
            break
        gamma_total_series = pd.Series({"GR": gamma_total})
        row = row.append(gamma_total_series)
        ttem_with_gamma = ttem_with_gamma.append(row, ignore_index=True)
    ttem_with_gamma["GRM"] =ttem_with_gamma["GR"].div(ttem_with_gamma["Elevation_Cell"].sub(ttem_with_gamma["Elevation_End"]))
    ttem_with_gamma["comment"] = gamma_place[0]
    return ttem_with_gamma



