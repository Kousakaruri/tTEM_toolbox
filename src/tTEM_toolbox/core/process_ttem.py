import pandas as pd
import numpy as np
from progress.bar import Bar

def format_ttem(fname, layer_exclude=False,
                line_exclude=False,
                point_exclude=False,
                DOI_path=None,
                filling=False, factor=10, **kwargs):

    if isinstance(fname, str):
        print("\nReading data from source file...")
        tmp_df = pd.read_fwf(fname, skiprows=26)
        tmp_df = tmp_df.drop(columns="/")
        tmp_df = tmp_df[tmp_df.Thickness_STD != 9999]
    elif isinstance(fname, pd.DataFrame):
        print("Reusing cached dataframe...")
        tmp_df = fname
    elif isinstance(fname, list):
        tmp_df = pd.DataFrame()
        concatlist = []
        for i in fname:
            tmp_df = pd.read_fwf(i, skiprows=26)
            tmp_df = tmp_df.drop(columns="/")
            tmp_df = tmp_df[tmp_df.Thickness_STD != 9999]
            concatlist.append(tmp_df)
        tmp_df = pd.concat(concatlist)
    else:
        print('The input must be String or DataFrame')
    if layer_exclude is False:
        print("No layer was excluded")
    else:
        tmp_df = tmp_df[~np.isin(tmp_df["Layer_No"], layer_exclude)]
        print('Exclude layer {}'.format(layer_exclude))
    if line_exclude is False:
        print("No line was excluded")
    else:
        tmp_df = tmp_df[~np.isin(tmp_df["Line_No"], line_exclude)]
        print('Exclude line {}'.format(line_exclude))
    if point_exclude is False:
        print("No point was excluded")
    else:
        tmp_df = tmp_df[~tmp_df["UTMX"].isin(point_exclude['UTMX'].values)]
        tmp_df = tmp_df[~tmp_df["UTMY"].isin(point_exclude['UTMY'].values)]
        #tmp_df = tmp_df[~tmp_df[].isin(point_exclude.values[:, 1])]
        [print('Exclude point {},{}'.format(x[0], x[1])) for x in point_exclude[['UTMX','UTMY']].values]
    if DOI_path is None:
        print('Will skip filterting DOI since no DOI file exist')
    else:
        tmp_df = DOI(tmp_df, DOI=DOI_path)
    if filling is True:
        print("will filling the tTEM data with factor {}".format(factor))
        def fill(group, factor):
            newgroup = group.loc[group.index.repeat(group.Thickness * factor)]
            mul_per_gr = newgroup.groupby('Elevation_Cell').cumcount()
            newgroup['Elevation_Cell'] = newgroup['Elevation_Cell'].subtract(mul_per_gr * 1 / factor)
            newgroup['Depth_top'] = newgroup['Depth_top'].add(mul_per_gr * 1 / factor)
            newgroup['Depth_bottom'] = newgroup['Depth_top'].add(1 / factor)
            newgroup['Elevation_End'] = newgroup['Elevation_Cell'].subtract(1 / factor)
            newgroup['Thickness'] = 1 / factor
            return newgroup
        def upscale(ttem_data, factor):
            concatlist = []
            groups = ttem_data.groupby(['UTMX', 'UTMY'])
            bar = Bar('Filling tTEM data', max=len(list(groups.groups.keys())))
            bar.check_tty = False
            for name, group in groups:
                newgroup = fill(group, factor)
                concatlist.append(newgroup)
                ### small progress bar
                bar.next()
                ###
            result = pd.concat(concatlist)
            result.reset_index(drop=True, inplace=True)
            bar.finish()
            return result
        df = upscale(tmp_df, factor=factor)
    else:
        df = tmp_df
    return df

def DOI(dataframe, DOI):
    df_DOI = pd.read_fwf(DOI, skiprows=3)
    df_DOI = df_DOI.drop(columns="/")
    df_out = pd.DataFrame(columns=dataframe.columns.values)
    concatlist = []
    bar = Bar('Applying DOI', max=df_DOI.index.shape[0])
    bar.check_tty = False
    for index, row in df_DOI.iterrows():
        df_tmp = dataframe[((dataframe.loc[:,"UTMX"] == row["UTMX"]) &
                            (dataframe.loc[:,"UTMY"] == row["UTMY"]) &
                            (dataframe.loc[:,"Elevation_Cell"] >= row["Value"]))]
                #Match the exact ttem column and only keep elevation above the DOI value
        bar.next()
        concatlist.append(df_tmp)
    df_out = pd.concat(concatlist)
    df_out = df_out.reset_index(drop=True)
    df_out["Elevation_End"] = df_out["Elevation_Cell"].subtract(df_out["Thickness"]) #create new column for future usage
    bar.finish()
    return df_out


