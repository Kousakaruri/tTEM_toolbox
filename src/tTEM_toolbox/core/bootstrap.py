from numba import jit
import numpy as np
import pandas as pd
import sys
import tTEM_tool as tt
from progress.bar import Bar
from scipy.stats import pearsonr

'''def select_closest(dataframe, formatwell, WIN, distance=500):
    # preprocess 1. select line of tTEM data base on the location of well
    well_filt_WIN = formatwell[formatwell["Bore"] == WIN]
    well_filt_WIN.reset_index(drop=True, inplace=True)
    UTMX_well = well_filt_WIN["UTMX"].iloc[0]
    UTMY_well = well_filt_WIN["UTMY"].iloc[0]
    dataframe["distance"] = np.sqrt((dataframe["UTMX"] - UTMX_well) ** 2 +
                                    (dataframe["UTMY"] - UTMY_well) ** 2)
    #match base on distance equation l = sqrt((x1-x2)**2 + (y1-y2)**2)
    df_closest = dataframe[dataframe["distance"] == dataframe.distance.min()].reset_index(drop=True)
    df_closest = df_closest[df_closest["distance"] <= float(distance)]
    #filter out the closest ttem data and also output the matched single well log
    return df_closest, well_filt_WIN'''
def select_closest(ttemdata,
                   welllog,
                   distance=500,
                   showskip=False,
                   ):
    def get_distance(group1, group2):
        dis = np.sqrt((group1[0] - group2[0]) ** 2 + (group1[1] - group2[1]) ** 2)
        return dis
    concatlist = []
    concatwell = []
    skipname = []
    skipdistace = []
    if isinstance(welllog, str):
        ori_well = tt.process_well.format_well(welllog, wellupscale=10)
    elif isinstance(welllog, pd.DataFrame):
        ori_well = welllog
    else:
        raise TypeError('welllog should be path to text file or pd.Dataframe')
    groups_well = ori_well.groupby('Bore')
    ttem_location = list(ttemdata.groupby(['UTMX', 'UTMY']).groups.keys())
    bar = Bar('Selecting closest ttem trace', max=len(list(groups_well.groups.keys())))
    bar.check_tty = False
    for name, group in groups_well:
        wellxy = list(group[['UTMX','UTMY']].iloc[0])
        well_ttem_distance = list(map(lambda x: get_distance(wellxy, x), ttem_location))
        minvalue = min(well_ttem_distance)
        if minvalue <= float(distance):
            point_match = ttem_location[well_ttem_distance.index(minvalue)]
            matchpoint = ttemdata[(ttemdata['UTMX'] == point_match[0]) & (ttemdata['UTMY'] == point_match[1])].copy()
            matchpoint.loc[:, 'distance'] = minvalue
            matchpoint.loc[:, 'Bore'] = name
            concatlist.append(matchpoint)
            concatwell.append(group)
        else:
            skipname.append(name)
            skipdistace.append(minvalue)
        bar.next()
    matched_ttem = pd.concat(concatlist).reset_index(drop=True)
    matched_well = pd.concat(concatwell).reset_index(drop=True)
    skipped = pd.DataFrame({'Bore': skipname, 'Distance': skipdistace})
    bar.finish()
    print('Total of {} well with in radius ({}m), {} skipped'.format(len(concatlist), distance, len(skipname)))
    if showskip is False:
        return matched_ttem, matched_well
    else:
        return matched_ttem, matched_well, skipped

def sum_thickness(welllog): # receive single bore well_log and sum up the thickness of fine/mix/coarse material
    output = pd.DataFrame(columns=["Lithology", "Thickness"])
    init_lith = welllog["Keyword"].iloc[0]
    init_elev = welllog["Elevation"].iloc[0]
    concatlist=[]
    # TODO make pure ndarry to fit JIT to increase speed
    for index, row in welllog.iterrows():
        if row["Keyword"] == init_lith:
            pass
        elif row["Keyword"] != init_lith:
            pack = [init_lith, init_elev - row["Elevation"]]
            tmp = pd.DataFrame(pack,index=('Lithology','Thickness')).T
            concatlist.append(tmp)
            init_lith = row["Keyword"]
            init_elev = row["Elevation"]
        else:
            print("error")
        if row["Elevation"] == welllog.loc[welllog.index[-1], 'Elevation']:
            pack = [init_lith, init_elev - row["Elevation"]]
            tmp = pd.DataFrame(pack,index=('Lithology','Thickness')).T
            concatlist.append(tmp)
    output = pd.concat(concatlist)
    output = output.groupby("Lithology")["Thickness"].sum()
    return output

def corr_well_filter(matched_ttem, matched_well, corr_thershold=0.3, correxport=False):
    print('check_corr = True, will remove well log with correlation lower than {}'.format(corr_thershold))
    corrlist = []
    well_group = matched_well.groupby('Bore')
    ttem_group = matched_ttem.groupby('Bore')
    bar = Bar('working on correlation', max=len(list(well_group.groups.keys())))
    bar.check_tty = False
    for name, group in ttem_group:
        tmp_df = pd.merge_asof(group.sort_values('Elevation_Cell'),
                               well_group.get_group(name).sort_values('Elevation'), left_on='Elevation_Cell',
                               right_on='Elevation', direction='nearest')
        tmp_df['Elev_diff'] = tmp_df['Elevation_Cell'] - tmp_df['Elevation']
        tmp_df = tmp_df[tmp_df['Elev_diff'].abs() < 1]
        try:
            corr = pearsonr(tmp_df['Resistivity'], tmp_df['Keyword_n'])
        except:
            corr = [0, 0]
        checklist = {'corr': corr[0], 'pvalue': corr[1], 'Bore': name}
        list_df = pd.DataFrame(checklist, index=[0])
        corrlist.append(list_df)
        bar.next()
    corr_each_well = pd.concat(corrlist).reset_index(drop=True)
    remove_well_list = corr_each_well[~(corr_each_well['corr'] > corr_thershold)]['Bore'].values
    matched_well = matched_well[~(matched_well['Bore'].isin(remove_well_list))]
    bar.finish()
    if correxport is False:
        return matched_well
    else:
        return matched_well, corr_each_well

def ttem_well_connect(matched_ttem, matched_well):
    # use ttem data interval to filter out welllog data to make a bootstrap ready dataframe
    matched_ttem[["Fine", "Mix", "Coarse"]] = np.nan
    #TODO make pure numpy format to increase speed
    concatlist= []
    ttem_group = matched_ttem.groupby('Bore')
    bar = Bar('connecting ttem with well', max=len(list(ttem_group.groups.keys())))
    bar.check_tty = False
    for name, group in ttem_group:
        well_select = matched_well[matched_well['Bore'] == name].copy()
        for index, row in group.iterrows():
            top = row["Elevation_Cell"]
            bott = row["Elevation_End"]
            match_litho = well_select[(well_select['Elevation'] >= bott) & (well_select['Elevation'] < top)]
            if match_litho.empty:
                break
            thickness = dict(match_litho.groupby('Keyword')['Keyword'].count()*match_litho['Thickness'].iloc[0])

            if "fine grain" in thickness:
                row["Fine"] = thickness["fine grain"]
            else:
                row["Fine"] = 0
            if "mix grain" in thickness:
                row["Mix"] = thickness["mix grain"]
            else:
                row["Mix"] = 0
            if "coarse grain" in thickness:
                row["Coarse"] = thickness["coarse grain"]
            else:
                row["Coarse"] = 0
            row = row.to_frame().T
            concatlist.append(row)
        bar.next()
    df = pd.concat(concatlist).reset_index(drop=True)
    bar.finish()
    return df

def pre_bootstrap(dataframe,welllog, distance=500):
    matched_ttem, matched_well = select_closest(dataframe, welllog, distance=distance, showskip=False)
    stitched_ttem_well = ttem_well_connect(matched_ttem, matched_well)
    Resistivity = stitched_ttem_well["Resistivity"].to_numpy().astype('float64')
    Thickness_ratio = stitched_ttem_well[["Fine", "Mix", "Coarse"]].div(stitched_ttem_well["Thickness"],
                                                                   axis=0).to_numpy().astype('float64')
    return stitched_ttem_well, Resistivity, Thickness_ratio, matched_ttem, matched_well

#@jit(nopython=True)
def bootstrap(resistivity, thickness_ratio):
    """
    bootstrap method, randomly pick from pre_bootstrap dataset to create a new data set with same shape,
    use thenew data set as an over-determined problem to solve the equation. Repeat 1000 times and output the resistivity
    The linear algebra equation check https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/gwat.12656
    """
    print('Bootstraping...')
    fine_Resistivity = np.empty(1000)
    mix_Resistivity = np.empty(1000)
    coarse_Resistivity = np.empty(1000)
    for k in range(1000):
        random_index = np.random.choice(np.arange(len(resistivity)), len(resistivity), replace=True)
        resistivity_sample = resistivity[random_index]
        resistivity_reverse = 1 / resistivity_sample
        thickness_ratio_sample = thickness_ratio[random_index]
        lstsq_result = np.linalg.lstsq(thickness_ratio_sample, resistivity_reverse)
        if lstsq_result[0][0] == 0:
            fine_Resistivity[k] = 0
        else:
            fine_Resistivity[k] = 1/lstsq_result[0][0]
        if lstsq_result[0][1] == 0:
            mix_Resistivity[k] = 0
        else:
            mix_Resistivity[k] = 1/lstsq_result[0][1]
        if lstsq_result[0][2] == 0:
            coarse_Resistivity[k] = 0
        else:
            coarse_Resistivity[k] = 1/lstsq_result[0][2]
    print('Done!')
    return fine_Resistivity, mix_Resistivity, coarse_Resistivity

def confidence(bootstrap_result, confidence=95): #95% condifence interval
    confidence_index = [(100 - confidence)/2, confidence+(100 - confidence)/2]
    confidence_interval = [np.percentile(bootstrap_result,confidence_index[0]),
                           np.percentile(bootstrap_result, confidence_index[1])]
    return confidence_interval

def packup(Fine_Resistivity, Mix_Resistivity, Coarse_Resistivity):

    Resi_conf_df = pd.DataFrame({"Fine_conf": confidence(Fine_Resistivity),
                                 "Mix_conf": confidence(Mix_Resistivity),
                                 "Coarse_conf": confidence(Coarse_Resistivity)})
    return Resi_conf_df