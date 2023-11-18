# This is the script for the 3rd manuscript.
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
import cv2 as cv
import plotly.express as px
from skimage import feature
# %% Parameters and Functions
workdir = Path(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test')
zonal_statistic_result = Path(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\2023\Paper\Image\Differential_subsidence')
welllog = workdir.joinpath(r'Plot_with_well_log\Well_log.xlsx')
elevation = workdir.joinpath(r'well_Utah\usgs_water_elevation.csv')
ttemname_north = workdir.joinpath(r'Plot_with_well_log\PD1_I01_MOD.xyz')
ttemname_center = workdir.joinpath(r'Plot_with_well_log\PD22_I03_MOD.xyz')
ttem_lslake = workdir.joinpath(r'Plot_with_well_log\lsll_I05_MOD.xyz')
DOI = workdir.joinpath(r'Plot_with_well_log\DOID1_DOIStaE.xyz')
well_info = workdir.joinpath(r'well_Utah\USGSdownload\NWISMapperExport.xlsx')
raster_insar = Path(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\2023\Paper\Raster\insar_tiff.tif')
demo_img = Path(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\2023\Paper\Raster\demo.jpg')
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
def write_raster(array, transform, name):
    with rasterio.open(
        raster_insar.parent.joinpath(name),
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=rasterio.CRS.from_epsg(32612),
        transform=transform,
    ) as dst:
        dst.write(array,1)
def turning_plot_ID_distance(ttem_df, turning_df, line_filter=None, fence=None, ID_filter=None, plot_ft=False):

    """
    This is the function that plot the given dataframe into gridded plot (for better looking)

    :param DataFrame ttem_df: a data frame contains tTEM data exported from Aarhus workbench
    :param int line_filter: In tTEM dataframe you can filter tTEM data by Line_No (defaults: None)
    :param list fence: receive a UTM coordinate to cut data, ex: [xmin, ymin, xmax, ymax] (defaults: None)
    :param plot_ft: True: plot all parameter in feet, False: plot in meter (defaults: True)
    :return:
        - fig - plotly fig of the block_plot
        - ttem_for_plot - the tTEM dataframe use for plot
        - empty_grid - the grid use to plot the figure
    """

    def distance_of_two_points(point1, point2):
        distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        return distance
    if line_filter is not None and fence is not None:
        ttem_preprocess_line = filter_line(ttem_df,line_filter)
        ttem_preprocess_fence = data_fence(ttem_preprocess_line,fence[0],fence[1],fence[2],fence[3])
        ttem_preprocessed = ttem_preprocess_fence.copy()
    elif line_filter is not None:
        ttem_preprocess_line = filter_line(ttem_df,line_filter)
        ttem_preprocessed = ttem_preprocess_line.copy()
    elif fence is not None:
        ttem_preprocess_fence = data_fence(ttem_df,fence[0],fence[1],fence[2],fence[3])
        ttem_preprocessed = ttem_preprocess_fence.copy()
    else:
        ttem_preprocessed = ttem_df
    if ID_filter is not None:
        ttem_preprocessed = ttem_preprocessed[(ttem_preprocessed['ID'] >= ID_filter[0]) &
                                              (ttem_preprocessed['ID'] <= ID_filter[1])]
        turning_df = turning_df[(turning_df['ID'] >= ID_filter[0]) & (turning_df['ID'] <= ID_filter[1])]
    if plot_ft is True:
        m_to_ft_factor = 3.28
        unit = 'ft'
    else:
        m_to_ft_factor = 1
        unit = 'm'
    ttem_preprocessed['Elevation_Cell'] = ttem_preprocessed['Elevation_Cell']*m_to_ft_factor
    turning_df['Elevation_Cell'] = turning_df['Elevation_Cell']*m_to_ft_factor
    ttem_preprocessed['Elevation_End'] = ttem_preprocessed['Elevation_End']*m_to_ft_factor
    turning_df['Elevation_End'] = turning_df['Elevation_End']*m_to_ft_factor
    elevation_max = ttem_preprocessed['Elevation_Cell'].max()
    elevation_end_min = ttem_preprocessed['Elevation_End'].min()
    elevation_range = int(elevation_max-elevation_end_min)
    ttem_preprocessed['Elevation_Cell_for_plot'] = abs(ttem_preprocessed['Elevation_Cell'] - elevation_max)
    turning_df['Elevation_Cell_for_plot'] = abs(turning_df['Elevation_Cell'] - elevation_max)
    ttem_preprocessed['Elevation_End_for_plot'] = abs(ttem_preprocessed['Elevation_End']-ttem_preprocessed['Elevation_End'].max())
    turning_df['Elevation_End_for_plot'] = abs(turning_df['Elevation_End']-turning_df['Elevation_End'].max())
    ID_groups = ttem_preprocessed.groupby('ID')
    UTM_groups = ID_groups[['UTMX','UTMY']].first().values.tolist()
    UTM_shift = ID_groups[['UTMX','UTMY']].first().shift(1).values.tolist()
    distance = list(map(distance_of_two_points, UTM_groups, UTM_shift))
    distance[0] = 0
    distance_marker = np.cumsum(distance)
    distance_range = sum(distance)
    y_distance = elevation_range*10
    x_distance = distance_range
    empty_grid = np.full((int((y_distance)),int(x_distance)),np.nan)
    turning_groups = turning_df.groupby('ID')
# Fill in the tTEM data by loop through the grid
    #colorscale = [[0, 'blue'], [0.5, 'yellow'], [1, 'red']]
    loop_count = 0
    for name, group in turning_groups:
        for index, line in group.iterrows():
            empty_grid[int(line['Elevation_Cell_for_plot']*10):int((line['Elevation_Cell_for_plot']+(line['Thickness'])*m_to_ft_factor)*10),
                        int(distance_marker[loop_count])-5:int(distance_marker[loop_count])+5] = np.log10(line['Resistivity'])
        loop_count += 1
    fig = px.imshow(empty_grid, range_color=(0,3),color_continuous_scale=[[0,'black'],[1,'black']])
    fig.update_layout(paper_bgcolor='white',plot_bgcolor='white',font_color='black')
    fig.update_layout(
        yaxis=dict(
        title='Elevation ({})'.format(unit),
        #tickmode='linear',
        #tick0=1774,
        #dtick=100
    ),
        xaxis=dict(
        title='Distance ({})'.format(unit)
    )
    )
    fig.update_layout(
        dict(
        xaxis=dict(
            titlefont=dict(
                family="Arial",
                size=50
            ),
            tickfont=dict(
                family="Arial",
                size=45
            ),
            tickmode = 'array',
            tickvals=list(np.arange(0, empty_grid.shape[1], 1000)),
            ticktext=[str(int(i)) for i in np.linspace(0, distance_range, 6)]
        ),
        yaxis=dict(
            titlefont=dict(
                family="Arial",
                size=50
            ),
            tickfont=dict(
                family="Arial",
                size=45
            ),
            tickmode='array',
            tickvals=list(np.arange(0, empty_grid.shape[0], 200)),
            ticktext=[str(int((elevation_max*10-i)/10)) for i in np.arange(0,empty_grid.shape[0],200)]
            #tickmode='array',
            #tickvals=list(np.arange(0,empty_grid.shape[0],500)),
            #TODO: this need more modification for m scale
            #ticktext=[str(int(elevation_max-i/10)) for i in np.arange(0,empty_grid.shape[0],500)]
        ),)
    )
    fig.update_coloraxes(
        colorbar=dict(
            ticks='outside',
            title='Resistivity',
            tickvals=[0, 1, 2, 3],
            tickfont=dict(
                size=30
            ),
            ticktext=['1', '10', '100', '1000'],
            tickmode='array',
            len=0.5))
    """fig.update_layout(
        title=dict(
            text='tTEM Line100, Northern Parowan Valley',
        ),
        titlefont=dict(
            size=50
        )
    )"""
    fig.show(renderer='browser')
    return fig, ttem_preprocessed,empty_grid
def Resistivity_plot_ID_distance(ttem_df, line_filter=None, fence=None, ID_filter=None, plot_ft=False):

    """
    This is the function that plot the given dataframe into gridded plot (for better looking)

    :param DataFrame ttem_df: a data frame contains tTEM data exported from Aarhus workbench
    :param int line_filter: In tTEM dataframe you can filter tTEM data by Line_No (defaults: None)
    :param list fence: receive a UTM coordinate to cut data, ex: [xmin, ymin, xmax, ymax] (defaults: None)
    :param plot_ft: True: plot all parameter in feet, False: plot in meter (defaults: True)
    :return:
        - fig - plotly fig of the block_plot
        - ttem_for_plot - the tTEM dataframe use for plot
        - empty_grid - the grid use to plot the figure
    """

    def distance_of_two_points(point1, point2):
        distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        return distance
    if line_filter is not None and fence is not None:
        ttem_preprocess_line = filter_line(ttem_df,line_filter)
        ttem_preprocess_fence = data_fence(ttem_preprocess_line,fence[0],fence[1],fence[2],fence[3])
        ttem_preprocessed = ttem_preprocess_fence.copy()
    elif line_filter is not None:
        ttem_preprocess_line = filter_line(ttem_df,line_filter)
        ttem_preprocessed = ttem_preprocess_line.copy()
    elif fence is not None:
        ttem_preprocess_fence = data_fence(ttem_df,fence[0],fence[1],fence[2],fence[3])
        ttem_preprocessed = ttem_preprocess_fence.copy()
    else:
        ttem_preprocessed = ttem_df
    if ID_filter is not None:
        ttem_preprocessed = ttem_preprocessed[(ttem_preprocessed['ID'] >= ID_filter[0]) &
                                              (ttem_preprocessed['ID'] <= ID_filter[1])]
    if plot_ft is True:
        m_to_ft_factor = 3.28
        unit = 'ft'
    else:
        m_to_ft_factor = 1
        unit = 'm'
    ttem_preprocessed['Elevation_Cell'] = ttem_preprocessed['Elevation_Cell']*m_to_ft_factor
    ttem_preprocessed['Elevation_End'] = ttem_preprocessed['Elevation_End']*m_to_ft_factor
    elevation_max = ttem_preprocessed['Elevation_Cell'].max()
    elevation_end_min = ttem_preprocessed['Elevation_End'].min()
    elevation_range = int(elevation_max-elevation_end_min)
    ttem_preprocessed['Elevation_Cell_for_plot'] = abs(ttem_preprocessed['Elevation_Cell'] - elevation_max)
    ttem_preprocessed['Elevation_End_for_plot'] = abs(ttem_preprocessed['Elevation_End']-ttem_preprocessed['Elevation_End'].max())
    ID_groups = ttem_preprocessed.groupby('ID')
    UTM_groups = ID_groups[['UTMX','UTMY']].first().values.tolist()
    UTM_shift = ID_groups[['UTMX','UTMY']].first().shift(1).values.tolist()
    distance = list(map(distance_of_two_points, UTM_groups, UTM_shift))
    distance[0] = 0
    distance_marker = np.cumsum(distance)
    distance_range = sum(distance)
    y_distance = elevation_range*10
    x_distance = distance_range
    empty_grid = np.full((int((y_distance)),int(x_distance)),np.nan)
# Fill in the tTEM data by loop through the grid
    #colorscale = [[0, 'blue'], [0.5, 'yellow'], [1, 'red']]

    loop_count = 0
    for name, group in ID_groups:
        for index, line in group.iterrows():
            empty_grid[int(line['Elevation_Cell_for_plot']*10):int((line['Elevation_End_for_plot']+6*m_to_ft_factor)*10),
                        int(distance_marker[loop_count])-5:int(distance_marker[loop_count])+5] = np.log10(line['Resistivity'])
        loop_count += 1
    fig = px.imshow(empty_grid, range_color=(0,3),color_continuous_scale=colorRes)
    fig.update_layout(paper_bgcolor='white',plot_bgcolor='white',font_color='black')
    fig.update_layout(
        yaxis=dict(
        title='Elevation ({})'.format(unit),
        #tickmode='linear',
        #tick0=1774,
        #dtick=100
    ),
        xaxis=dict(
        title='Distance ({})'.format(unit)
    )
    )
    fig.update_layout(
        dict(
        xaxis=dict(
            titlefont=dict(
                family="Arial",
                size=50
            ),
            tickfont=dict(
                family="Arial",
                size=45
            ),
            tickmode = 'array',
            tickvals=list(np.arange(0, empty_grid.shape[1], 1000)),
            ticktext=[str(int(i)) for i in np.linspace(0, distance_range, 6)]
        ),
        yaxis=dict(
            titlefont=dict(
                family="Arial",
                size=50
            ),
            tickfont=dict(
                family="Arial",
                size=45
            ),
            tickmode='array',
            tickvals=list(np.arange(0, empty_grid.shape[0], 200)),
            ticktext=[str(int((elevation_max*10-i)/10)) for i in np.arange(0,empty_grid.shape[0],200)]
            #tickmode='array',
            #tickvals=list(np.arange(0,empty_grid.shape[0],500)),
            #TODO: this need more modification for m scale
            #ticktext=[str(int(elevation_max-i/10)) for i in np.arange(0,empty_grid.shape[0],500)]
        ),)
    )
    fig.update_coloraxes(
        colorbar=dict(
            ticks='outside',
            title='Resistivity',
            tickvals=[0, 1, 2, 3],
            tickfont=dict(
                size=30
            ),
            ticktext=['1', '10', '100', '1000'],
            tickmode='array',
            len=0.5))
    """fig.update_layout(
        title=dict(
            text='tTEM Line100, Northern Parowan Valley',
        ),
        titlefont=dict(
            size=50
        )
    )"""
    fig.show(renderer='browser')
    return fig, ttem_preprocessed,empty_grid

def Resistivity_3d_plot(ttem_df, ID_filter=None):
    ttem_preprocessed = ttem_df.copy()
    if ID_filter is not None:
        ttem_preprocessed = ttem_preprocessed[(ttem_preprocessed['ID'] >= ID_filter[0]) &
                                              (ttem_preprocessed['ID'] <= ID_filter[1])]
    trace = utilities.plot.generate_trace(ttem_preprocessed, 'ttem')
    import plotly.graph_objects as go
    fig = go.Figure(trace)
    return fig.show()
# %% 1st attemp to plot canny edge detection
insar = rasterio.open(raster_insar)
insar_array = insar.read(1)
insar_new_array = np.where(insar_array>1, 0, insar_array)
insar_new_array_uint = (insar_new_array*255).astype(np.uint8)

img = cv.imread(str(demo_img.resolve()), cv.IMREAD_GRAYSCALE)
#canny
edges = cv.Canny(img,100,200)
edges2 = feature.canny(insar_new_array_uint, sigma=6)
edges_collection= ['']*20
for i in range(20):
    tmp = feature.canny(insar_new_array_uint, sigma=i)
    edges_collection[i] = tmp

# %%2nd attemt to plot Sobel edge detection
edges_collectionx = ['']*16
edges_collectiony = ['']*16
edges_collectionxy = ['']*16
for i in np.arange(1,32,2):
    edges_sobelx = cv.Sobel(insar_new_array, cv.CV_32F, 1, 0, ksize=i)
    edges_collectionx[i//2] = edges_sobelx
    edges_sobely = cv.Sobel(insar_new_array, cv.CV_32F, 0, 1, ksize=i)
    edges_collectiony[i//2] = edges_sobely
    edges_sobelxy = cv.Sobel(insar_new_array, cv.CV_32F, 1, 1, ksize=i)
    edges_collectionxy[i//2] = edges_sobelxy
for i in range(16):
    write_raster(edges_collectionx[i], insar.transform, 'sobelx{}.tif'.format(i))
    write_raster(edges_collectiony[i], insar.transform, 'sobely{}.tif'.format(i))
    write_raster(edges_collectionxy[i], insar.transform, 'sobelxy{}.tif'.format(i))

# %% tTEM processing
ttem_north = core.main.ProcessTTEM(ttem_path=[ttemname_north],
                                   welllog=welllog,
                                   DOI_path=DOI)
ttem_north_data = ttem_north.data()
ttem_center = core.main.ProcessTTEM(ttem_path=[ttemname_center],
                                    welllog=welllog,
                                    DOI_path=DOI)
ttem_center_data = ttem_center.data()

north_data_head = ttem_north_data.groupby('ID').head(1)
center_data_head = ttem_center_data.groupby('ID').head(1)
north_data_head.to_csv(workdir.joinpath('north_data_head.csv'))
center_data_head.to_csv(workdir.joinpath('center_data_head.csv'))

# %% plot some tTEM data
#fig,_,_ = Resistivity_plot_ID_distance(ttem_north_data,ID_filter = [3570, 3838])
fig2,_,_ = turning_plot_ID_distance(ttem_north_data , north_data_turning,ID_filter = [3570, 3838])


# %% try to seperate tTEM into different layers based on resistivity
def turning_points(array):
    """ turning_points(array) -> min_indices:list, max_indices:list
    Finds the turning points within an 1D array and returns the index of the minimum and
    maximum turning points in two separate lists.
    """
    idx_max, idx_min = [], []
    if (len(array) < 3):
        return idx_min, idx_max

    NEUTRAL, RISING, FALLING = range(3)
    def get_state(a, b):
        if a < b: return RISING
        if a > b: return FALLING
        return NEUTRAL

    ps = get_state(array[0], array[1])
    begin = 1
    for i in range(2, len(array)):
        s = get_state(array[i - 1], array[i])
        if s != NEUTRAL:
            if ps != NEUTRAL and ps != s:
                if s == FALLING:
                    idx_max.append((begin + i - 1) // 2)
                else:
                    idx_min.append((begin + i - 1) // 2)
            begin = i
            ps = s
    return idx_min, idx_max
def get_turning_rows(group: pd.DataFrame):
    index_min, index_max = turning_points(group['Resistivity'].values)
    index = index_min + index_max
    turning_rows = group.iloc[index]
    return turning_rows

north_data_ID_group = ttem_north_data.groupby('ID')
north_data_turning = north_data_ID_group.apply(get_turning_rows)
north_data_turning.reset_index(drop=True,inplace=True)
center_data_ID_group = ttem_center_data.groupby('ID')
center_data_turning = center_data_ID_group.apply(get_turning_rows)
center_data_turning.reset_index(drop=True,inplace=True)

# %% Plot buffer compare
zonal_files = list(zonal_statistic_result.glob('*.csv'))
zonal_left = pd.read_csv(zonal_files[2])
zonal_right = pd.read_csv(zonal_files[3])
gradient_left = pd.read_csv(zonal_files[1])
gradient_right = pd.read_csv(zonal_files[0])
def plot_zonal_hist(dataframe):



    fig = px.histogram(dataframe,x='MEAN',nbins=10)
    fig.update_xaxes(title = 'Subsidence Rate m/yr',
                     title_font=dict(
                         size=45
                     ),
                     tickfont=dict(
                         size=45
                     ),
    )
    fig.update_yaxes(range=[0,50],
                    title = 'Count',
                     title_font=dict(
                         size=45
                     ),
                     tickfont=dict(
                         size=45
                     ),
    ),
    fig.update_traces(marker=dict(line=dict(width=1, color='black')))


    return fig
fig_left = plot_zonal_hist(zonal_left)
fig_right = plot_zonal_hist(zonal_right)
fig_gradient_left = plot_zonal_hist(gradient_left)
fig_gradient_right = plot_zonal_hist(gradient_right)

# %% Try numpy gradient
def gradient_of_map(insar_array, cell_size, axis):

    new_gradient = np.gradient(insar_array, cell_size,axis=axis)
    conditions = [new_gradient>100000000, new_gradient<-100000000]
    choices = [0,0]
    gradient_clean = np.select(conditions, choices, default=new_gradient)
    return gradient_clean

write_raster(gradient_clean, insar.transform, 'Insar_gradient{}.tif'.format(50))

#%% Process of all rock transform for Julianne
from pathlib import Path
workdir = Path(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test')
ttemlsl_file = workdir.joinpath(r'Plot_with_well_log\lsll_I05_MOD.xyz')
ttemlsl = pd.read_fwf(ttemlsl_file, skiprows=34)
ttemlsl = ttemlsl.drop(columns="/")
ttemlsl = ttemlsl[ttemlsl.Thickness_STD != 9999]
ttemlsl["Elevation_End"] = ttemlsl["Elevation_Cell"].subtract(ttemlsl["Thickness"])
ttem_lsl = core.main.ProcessTTEM(ttem_path=ttemlsl,
                                 welllog=welllog,

                                 layer_exclude=[],
                                 line_exclude=[])

Resi_conf_df_lsl = pd.DataFrame({'Fine_conf':[6,36],'Mix_conf':[24.4,37],'Coarse_conf':[38.2,41.8]})

rk_trans_lsl = core.Rock_trans.rock_transform(ttem_lsl.data(), Resi_conf_df_lsl)
rk_trans_lsl.to_csv(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Old 2022\Paper\tTEM\rk_lsl.csv')
ttem_rk_trans_center, _ = ttem_center.ttem_well_connect()
ttem_rk_trans_center.to_csv(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Old 2022\Paper\tTEM\ttem_center.csv')
ttem_pure_north = ttem_north_data[~ttem_north_data['Line_No'].isin([180,190])]
Resi_conf_df_publish_wet = pd.DataFrame({'Fine_conf':[6,16],'Mix_conf':[12,22],'Coarse_conf':[23,43]})
rk_trans_north = core.Rock_trans.rock_transform(ttem_pure_north, Resi_conf_df_publish_wet)
rk_trans_north.to_csv(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Old 2022\Paper\tTEM\rk_north.csv')
jason_farm = ttem_north_data[(ttem_north_data['ID'] <2750)&(ttem_north_data['ID']>2484) ]
Resi_conf_df_jason = pd.DataFrame({'Fine_conf':[6,23.5],'Mix_conf':[24.4,37],'Coarse_conf':[34.2,41.8]})
rk_trans_jason = core.Rock_trans.rock_transform(jason_farm, Resi_conf_df_jason)
rk_trans_jason.to_csv(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Old 2022\Paper\tTEM\rk_jason.csv')