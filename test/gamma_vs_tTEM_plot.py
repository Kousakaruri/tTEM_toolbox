# Parameters
import pandas as pd
import numpy as np
from pathlib import Path
from pyproj import Transformer
#from plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
workdir = Path(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test')
welllog = workdir.joinpath(r'Plot_with_well_log\Well_log.xlsx')
elevation = workdir.joinpath(r'well_Utah\usgs_water_elevation.csv')
ttemname_north = workdir.joinpath(r'Plot_with_well_log\PD1_I01_MOD.xyz')
ttemname_center = workdir.joinpath(r'Plot_with_well_log\PD22_I03_MOD.xyz')
ttem_lslake = workdir.joinpath(r'Plot_with_well_log\lsll_I05_MOD.xyz')
DOI = workdir.joinpath(r'Plot_with_well_log\DOID1_DOIStaE.xyz')
well_info = workdir.joinpath(r'well_Utah\USGSdownload\NWISMapperExport.xlsx')
gamma_file_path = r"C:\Users\jldz9\PycharmProjects\tTEM_toolbox\data\Parowan water table.xlsx"
location = r"C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\Gamma\location.csv"
colorRes = [[0, 'rgb(0,0,190)'],
            [1 / 16, 'rgb(0,75,220)'],
            [2 / 16, 'rgb(0,150,235)'],
            [3 / 16, 'rgb(0,200,255)'],
            [4 / 16, 'rgb(80,240,255)'],
            [5 / 16, 'rgb(30,210,0)'],
            [6 / 16, 'rgb(180,255,30)'],
            [7 / 16, 'rgb(255,255,0)'],
            [8 / 16, 'rgb(255,195,0)'],
            [9 / 16, 'rgb(255,115,0)'],
            [10 / 16, 'rgb(255,0,0)'],
            [11 / 16, 'rgb(255,0,120)'],
            [12 / 16, 'rgb(140,40,180)'],
            [13 / 16, 'rgb(165,70,220)'],
            [14 / 16, 'rgb(195,130,240)'],
            [15 / 16, 'rgb(230,155,240)'],
            [1, 'rgb(230,155,255)']]
colorrock = [[0, 'rgb(30,144,255)'],
                 [0.5, 'rgb(255,255,0)'],
                 [1,'rgb(255,0,0)']]
def get_distance(group1, group2):
    dis = np.sqrt((group1[0] - group2[0]) ** 2 + (group1[1] - group2[1]) ** 2)
    return dis
def gamma_search_close_ttem(ttem_data, gamma_data,distance=500):
    """
    Use single given gamma data to search the closest ttem data with given radius

    :param ttem_data: tTEM dataset from processed from Aarhus workbench
    :param gamma_data: single gamma data only use "X", "Y" location
    :param distance: metric
    :return: single tTEM trace that spatially close to the gamma data
    """
    UTMX_gamma = gamma_data["X"].iloc[0]
    UTMY_gamma = gamma_data["Y"].iloc[0]
    ttem_data["distance"] = np.sqrt((ttem_data["UTMX"] - UTMX_gamma) ** 2 +
                                    (ttem_data["UTMY"] - UTMY_gamma) ** 2)
    #match base on distance equation l = sqrt((x1-x2)**2 + (y1-y2)**2)
    df_closest = ttem_data[ttem_data["distance"] == ttem_data.distance.min()].reset_index(drop=True)
    df_closest = df_closest[df_closest["distance"] <= float(distance)]
    return df_closest
def ttem_search_close_welllog(ttem_data, well_log_data, distance=500):
    """
    Use single given tTEM data to search the closest well_log data with given radius
    :param ttem_data: tTEM dataset from processed from Aarhus workbench
    :param well_log_data: well log data processed by process_welll.load well
    :param distance: metric
    :return:sigle well data that spatially close to the tTEM trace
    """
    UTMX_ttem = ttem_data["UTMX"].iloc[0]
    UTMY_ttem = ttem_data["UTMY"].iloc[0]
    well_log_data['distance'] = np.sqrt((well_log_data["UTMX"] - UTMX_ttem) ** 2 +
                                    (well_log_data["UTMY"] - UTMY_ttem) ** 2)
    well_closest = well_log_data[well_log_data['distance'] == well_log_data.distance.min()].reset_index(drop=True)
    well_closest = well_closest[well_closest['distance'] <= float(distance)]
    return   well_closest
def well_search_close_ttem(well_log_data, ttem_data, distance=500):
    UTMX_ttem = well_log_data["UTMX"].iloc[0]
    UTMY_ttem = well_log_data["UTMY"].iloc[0]
    ttem_data['distance'] = np.sqrt((ttem_data["UTMX"] - UTMX_ttem) ** 2 + (ttem_data["UTMY"] - UTMY_ttem) ** 2)
    ttem_closest = ttem_data[ttem_data['distance'] == ttem_data.distance.min()].reset_index(drop=True)
    ttem_closest = ttem_closest[ttem_closest['distance'] <= float(distance)]
    return ttem_closest
def plot_rock_single(data):
    """
    Plot single trace of rock transform result for paper uses

    :param data: The input data should be rock physics transform result from Rock_trans.rock_transform
    :return:Plotly Figure
    """
    colorrock = [[0, 'rgb(30,144,255)'],
                 [0.5, 'rgb(255,255,0)'],
                 [1, 'rgb(255,0,0)']]

    y_shape = int(data.Depth_bottom.max()*10)
    empty_gird = np.full((y_shape,50), np.nan)
    for index,line in data.iterrows():
        y_start = int(line.Depth_top*10)
        y_stop = int(line.Depth_bottom*10)
        empty_gird[y_start:y_stop,:] = line.Identity_n
    fig_rock = px.imshow(empty_gird, range_color=(1, 3), color_continuous_scale=colorrock)
    return fig_rock
def plot_well_single(data):
    """
    Plot single trace of well log result for paper uses

    :param data: The input data should be well log from process_well.format_well
    :return: plotly fig
    """
    colorrock = [[0, 'rgb(30,144,255)'],
                 [0.5, 'rgb(255,255,0)'],
                 [1, 'rgb(255,0,0)']]
    y_shape = int(data.Depth2.max()*0.3048*10)
    empty_gird = np.full((y_shape,50),np.nan)
    for index,line in data.iterrows():
        y_start = int(line.Depth1*3.048) #0.3048*10
        y_stop = int(line.Depth2*3.048)
        empty_gird[y_start:y_stop,:] = line.Keyword_n
    fig_well = px.imshow(empty_gird, range_color=(1, 3), color_continuous_scale=colorrock)
    return fig_well
gamma_elevation = pd.read_csv(location)
bradshaw_farm_elevation = gamma_elevation[gamma_elevation['name'] == 'Bradshaw Farm']
halterman_farm_elevation = gamma_elevation[gamma_elevation['name'] == 'Halterman Farms']
adams_farm_elevation = gamma_elevation[gamma_elevation['name'] == 'Adams Farm']
bradshaw_farms = pd.read_excel(r'C:\Users\jldz9\PycharmProjects\tTEM_toolbox\data\Parowan water table.xlsx',sheet_name='bradshaw farms down',skiprows=[1])
bradshaw_farms.rename(columns=lambda x: str(x).strip(),inplace=True)
bradshaw_farms_clean = bradshaw_farms[~(bradshaw_farms['COUNT'] == -999)]
halterman_farms = pd.read_excel(r'C:\Users\jldz9\PycharmProjects\tTEM_toolbox\data\Parowan water table_corrected.xlsx',sheet_name='halterman down',skiprows=[1])
halterman_farms.rename(columns=lambda x: str(x).strip(),inplace=True)
halterman_farms_clean = halterman_farms[~(halterman_farms['COUNT'] == -999)]
adams_farms = pd.read_excel(r'C:\Users\jldz9\PycharmProjects\tTEM_toolbox\data\Parowan water table.xlsx',sheet_name='adams farm down',skiprows=[1])
adams_farms.rename(columns=lambda x: str(x).strip(),inplace=True)
adams_farms_clean = adams_farms[~(adams_farms['COUNT'] == -999)]
def transformation(gamma, locationdata):
    gammadata = gamma
    x_ori = locationdata.iloc[0]["X"]
    y_ori = locationdata.iloc[0]["Y"]
    transformer = Transformer.from_crs('epsg:4326', 'epsg:32612')  # WGS84-->UTM12N
    x, y = transformer.transform(y_ori, x_ori)
    gammadata.loc[:, "X"] = x
    gammadata.loc[:, "Y"] = y
    gammadata.loc[:, "Z"] = locationdata.loc[:, 'Z'].values[0]
    gammadata.loc[:, "Elevation"] = gammadata.loc[:, "Z"].sub(gammadata.loc[:, "Depth"])
    return gammadata
bradshaw_farms_gamma = transformation(bradshaw_farms_clean,bradshaw_farm_elevation)
halterman_farms_gamma = transformation(halterman_farms_clean,halterman_farm_elevation)
adams_farms_gamma = transformation(adams_farms_clean,adams_farm_elevation)
ttem_north = core.main.ProcessTTEM(ttem_path=[ttemname_north],
                                   welllog=welllog,
                                   DOI_path=DOI,
                                   layer_exclude=[],
                                   line_exclude=[])
ttem_north_data = ttem_north.data()
ttem_center = core.main.ProcessTTEM(ttem_path=[ttemname_center],
                                    welllog=welllog,
                                    DOI_path=DOI,
                                    layer_exclude=[],
                                    line_exclude=[])
ttem_center_data = ttem_center.data()

df_closest = gamma_search_close_ttem(ttem_north_data, bradshaw_farms_clean)
df_closest_halterman = gamma_search_close_ttem(ttem_center_data, halterman_farms_gamma)
df_closest_adams = gamma_search_close_ttem(ttem_center_data, adams_farms_gamma)
fig = utilities.plot.res_1d_plot(df_closest)
fig.update_xaxes(range=[1.06,1.97])
fig_halterman = utilities.plot.res_1d_plot(df_closest_halterman)
fig_halterman.update_xaxes(range=[1.06,1.97])
fig_adams = utilities.plot.res_1d_plot(df_closest_adams)
fig_adams.update_xaxes(range=[1.06,1.97])
bottom_limit = df_closest.Depth_bottom.max()
bottom_limit_halterman = df_closest_halterman.Depth_bottom.max()
bottom_limit_adams = df_closest_adams.Depth_bottom.max()
gamma_for_plot_bradshaw = bradshaw_farms_gamma[bradshaw_farms_gamma['Depth'] < bottom_limit]
gamma_for_plot_halterman = halterman_farms_gamma[halterman_farms_gamma['Depth']<bottom_limit_halterman]
gamma_for_plot_adams = adams_farms_gamma[adams_farms_gamma['Depth']<bottom_limit_adams]
fig_gamma_bradshaw = utilities.plot.well_test_plot(gamma_for_plot_bradshaw, window=30)
fig_gamma_bradshaw.update_layout(
    xaxis=dict(
        range = [25,100]
    ),
    yaxis=dict(
        rangemode = 'tozero'
    )
)
fig_gamma_halterman = utilities.plot.well_test_plot(gamma_for_plot_halterman, window=30)
fig_gamma_halterman.update_layout(
    xaxis=dict(
        range = [25,100]
    ),
    yaxis=dict(
        rangemode = 'tozero'
    )
)
fig_gamma_adams = utilities.plot.well_test_plot(gamma_for_plot_adams, window=30)
fig_gamma_adams.update_layout(
    xaxis=dict(
        range = [25,100]
    ),
    yaxis=dict(
        rangemode = 'tozero'
    )
)
### Rock physics transform for single column
Resi_conf_df_jason = pd.DataFrame({'Fine_conf':[6,23.5],'Mix_conf':[24.4,37],'Coarse_conf':[34.2,41.8]})
Resi_conf_df_center_interpreted = pd.DataFrame({'Fine_conf':[28.9,32],'Mix_conf':[32.3,40],'Coarse_conf':[36.1,42.7]})
rk_jason_bradshaw = core.Rock_trans.rock_transform(df_closest, Resi_conf_df_jason)
rk_central_halterman = core.Rock_trans.rock_transform(df_closest_halterman, Resi_conf_df_center_interpreted)
rk_central_adams = core.Rock_trans.rock_transform(df_closest_adams, Resi_conf_df_center_interpreted)
fig_rk_bardshaw = plot_rock_single(rk_jason_bradshaw)
fig_rk_halterman = plot_rock_single(rk_central_halterman)
fig_rk_adams = plot_rock_single(rk_central_adams)
### Plot well log that spatially close to the tTEM
ori_well = core.process_well.format_well(welllog, upscale=10)
well_closest_bardshaw = ttem_search_close_welllog(df_closest, ori_well)
fig_well_bardshaw = plot_well_single(well_closest_bardshaw)
well_closest_halterman = ttem_search_close_welllog(df_closest_halterman,ori_well)
fig_well_halterman = plot_well_single(well_closest_halterman)
well_closest_adams = ttem_search_close_welllog(df_closest_adams,ori_well,distance=2000)
fig_well_adams = plot_well_single(well_closest_adams)

well_26425 = ori_well[ori_well['Bore'] == '26425']
fig_well_26425_halterman = plot_well_single(well_26425)
well_438242 = ori_well[ori_well['Bore'] == '438242']
fig_well_438242_adams = plot_well_single(well_438242)
well_13870 = ori_well[ori_well['Bore'] == '13870']
ttem_close_13870 = well_search_close_ttem(well_13870, pd.concat([ttem_center_data,ttem_north_data]))
fig_ttem_13870 = utilities.plot.res_1d_plot(ttem_close_13870)
fig_ttem_13870.update_xaxes(range=[0.8,1.79])
well_27210 = ori_well[ori_well['Bore'] == '27210']
ttem_close_27210 = well_search_close_ttem(well_27210, pd.concat([ttem_center_data,ttem_north_data]))
fig_ttem_27210 = utilities.plot.res_1d_plot(ttem_close_27210)
fig_ttem_13870.update_xaxes(range=[0.8,1.79])
well_13870 = ori_well[ori_well['Bore'] == '13870']
ttem_close_13870 = well_search_close_ttem(well_13870, pd.concat([ttem_center_data,ttem_north_data]))
fig_ttem_13870 = utilities.plot.res_1d_plot(ttem_close_13870)
fig_ttem_13870.update_xaxes(range=[0.8,1.79])
ttem_all = pd.concat([ttem_center_data,ttem_north_data])
def ttem_closeby_plot(ttem_data, welllog, well_name):
    ori_well = core.process_well.format_well(welllog, upscale=10)
    well = ori_well[ori_well['Bore'] == str(well_name)]
    ttem_close = well_search_close_ttem(well, ttem_data)
    fig_ttem = utilities.plot.res_1d_plot(ttem_close)
    return fig_ttem