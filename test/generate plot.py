import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected = True)
import plotly.io as pio

pio.renderers.default = "browser"
pio.kaleido.scope.chromium_args = tuple([arg for arg in pio.kaleido.scope.chromium_args if arg != "--disable-dev-shm-usage"])
workdir = Path(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test')
welllog = workdir.joinpath(r'Plot_with_well_log\Well_log.xlsx')
elevation = workdir.joinpath(r'well_Utah\usgs_water_elevation.csv')
ttemname_north = workdir.joinpath(r'Plot_with_well_log\PD1_I01_MOD.xyz')
ttemname_center = workdir.joinpath(r'Plot_with_well_log\PD22_I03_MOD.xyz')
ttem_lslake = workdir.joinpath(r'Plot_with_well_log\lsll_I05_MOD.xyz')
DOI = workdir.joinpath(r'Plot_with_well_log\DOID1_DOIStaE.xyz')
well_info = workdir.joinpath(r'well_Utah\USGSdownload\NWISMapperExport.xlsx')
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
def distance_of_two_points(point1,point2):
    distance = np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return distance
def split_ttem(ttem_df, gwsurface_result):
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
def plot_bootstrap_result(dataframe):
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
        fig_hist.add_trace(go.Histogram(x=dataframe.mix, name='Mix', marker_color='Green', opacity=0.75))
    fig_hist.update_layout(paper_bgcolor='white',plot_bgcolor='white',font_color='black')
    fig_hist.update_layout(
        xaxis=dict(
            title='Resistivity (ohm-m)',
            title_font=dict(
                family='Arial',
                size=50,
                #weight='bold'
            ),
            tickfont=dict(
                family='Arial',
                size=45,
                #weight='bold'
            )
            #tickmode='linear',
            #tick0=1774,
            #dtick=100
        ),
        yaxis=dict(
            title='Counts',
            title_font=dict(
                family='Arial',
                size=50,
                #weight='bold'
            ),
            tickfont=dict(
                family='Arial',
                size=45,
                #weight='bold'

            )

            #tickmode='linear',
            #tick0=1774,
            #dtick=100
        ),
        legend=dict(
            font=dict(
                family='Arial',
                size=25
            )
        )
    )
    return fig_hist
def data_fence(ttem_df,xmin,ymin,xmax,ymax):
    if xmin>xmax:
        raise ValueError('xmin:{} is greater than xmax:{}'.format(xmin,xmax))
    if ymin>ymax:
        raise ValueError('ymin:{} is greater than ymax:{}'.format(xmin,xmax))
    new_ttem_df = ttem_df[(ttem_df['UTMX']>xmin)&(ttem_df['UTMX']<xmax)&(ttem_df['UTMY']>ymin)&(ttem_df['UTMY']<ymax)]
    return new_ttem_df
def filter_line(ttem_df, line_filter):
        ttem_preprocess_line = ttem_df[ttem_df['Line_No'] == int(line_filter)]
        if ttem_preprocess_line.empty:
            raise ValueError('Did not found any data under line_no {}, the line number suppose to be integer'.format(line_filter))
        return ttem_preprocess_line
def resistivity_avg(ttem_df):
    new_1d_df = []
    coordinate_group = ttem_df.groupby(['UTMX','UTMY'])
    for name, group in coordinate_group:
        mean_resistivity = group['Resistivity'].mean()
        output_df = {'UTMX':name[0],'UTMY':name[1],'Mean_Resistivity':mean_resistivity}
        tmp_df = pd.DataFrame([output_df],index=None)
        new_1d_df.append(tmp_df)
    export_df = pd.concat(new_1d_df)
    return  export_df
def lithology_pct(rock_transform_df):
    """
    # Plan to group by the coordinate and generate a new dataframe with only [UTMX, UTMY]
    """
    new_1d_df = []
    coordinate_group = rock_transform_df.groupby(['UTMX','UTMY'])
    for name, group in coordinate_group:
        total_thickness_for_point = sum(group['Thickness'])
        lithology_group = group.groupby('Identity')
        try:
            fine_grain_thickness = lithology_group.get_group('Fine_grain')['Thickness'].sum()
        except KeyError:
            fine_grain_thickness = 0
        try:
            mixed_grain_thickness = lithology_group.get_group('Mix_grain')['Thickness'].sum()
        except KeyError:
            mixed_grain_thickness = 0
        try:
            coarse_grain_thickness = lithology_group.get_group('Coarse_grain')['Thickness'].sum()
        except KeyError:
            coarse_grain_thickness = 0
        tmp_df = pd.DataFrame([{'UTMX':name[0],
                  'UTMY':name[1],
                  'Fine grained ratio':fine_grain_thickness/float(total_thickness_for_point),
                  'mixed_grain_thickness':mixed_grain_thickness/float(total_thickness_for_point),
                  'coarse_grain_thickness':coarse_grain_thickness/float(total_thickness_for_point)}])

        new_1d_df.append(tmp_df)
    export = pd.concat(new_1d_df)
    return export
def block_plot(ttem_df, line_filter=None, fence=None, plot_ft=True):

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
    if plot_ft is True:
        m_to_ft_factor = 3.28
        unit = 'ft'
    else:
        m_to_ft_factor = 1
        unit = 'm'
    distance=0
    ttem_coordinate_group = ttem_preprocessed.groupby(['UTMX','UTMY'])
    for i in range(1,len(list(ttem_coordinate_group.groups.keys()))):
        tmp =distance_of_two_points(list(ttem_coordinate_group.groups.keys())[i-1][0],
                                     list(ttem_coordinate_group.groups.keys())[i-1][1],
                                     list(ttem_coordinate_group.groups.keys())[i][0],
                                     list(ttem_coordinate_group.groups.keys())[i][1])
        distance=distance+tmp
    distance_range = int(distance*m_to_ft_factor)
    ttem_preprocessed['Elevation_Cell'] = ttem_preprocessed['Elevation_Cell']*m_to_ft_factor
    ttem_preprocessed['Elevation_End'] = ttem_preprocessed['Elevation_End']*m_to_ft_factor
    elevation_max = ttem_preprocessed['Elevation_Cell'].max()

    elevation_end_min = ttem_preprocessed['Elevation_End'].min()

    elevation_range = int(elevation_max-elevation_end_min)
    print('elevation_range',elevation_max,elevation_end_min,elevation_range)
    ttem_preprocessed['Elevation_Cell_for_plot'] = abs(ttem_preprocessed['Elevation_Cell'] - elevation_max)
    ttem_preprocessed['Elevation_End_for_plot'] = abs(ttem_preprocessed['Elevation_End']-ttem_preprocessed['Elevation_End'].max())
    y_distance = elevation_range
    x_distance = distance_range/10
    empty_grid = np.full((int((y_distance+500)),int(x_distance)),np.nan)
# Fill in the tTEM data by loop through the grid
    first_iteration = True
    for name, group in ttem_coordinate_group:
        if first_iteration is True:
            first_iteration = False
            initial_distance = 1
            tmp_coor = name
            for index, line in group.iterrows():
                elevation_cell_round = line['Elevation_Cell_for_plot']
                elevation_end_round = line['Elevation_End_for_plot']
                empty_grid[int((elevation_cell_round-6)*m_to_ft_factor):int((elevation_end_round+6)*m_to_ft_factor),int((initial_distance-10)*m_to_ft_factor/10):int(((initial_distance+10)*m_to_ft_factor/10))] = np.log10(line['Resistivity'])
        else:
            initial_distance = initial_distance + distance_of_two_points(tmp_coor[0],tmp_coor[1],name[0],name[1])
            tmp_coor = name
            for index, line in group.iterrows():
                elevation_cell_round = line['Elevation_Cell_for_plot']
                empty_grid[int((elevation_cell_round-6)*m_to_ft_factor):int((elevation_end_round+6)*m_to_ft_factor),int((initial_distance-10)*m_to_ft_factor/10):int(((initial_distance+10)*m_to_ft_factor/10))] = np.log10(line['Resistivity'])
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
            )
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
            tickvals=[0,1,2,3],
            tickfont=dict(
                size=30
            ),
            ticktext=['1','10','100','1000'],
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
def block_plot2(ttem_df, line_filter=None, fence=None, plot_ft=True):

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
    if plot_ft is True:
        m_to_ft_factor = 3.28
        unit = 'ft'
    else:
        m_to_ft_factor = 1
        unit = 'm'
    distance=0
    ttem_coordinate_group = ttem_preprocessed.groupby(['UTMX','UTMY'])
    for i in range(1,len(list(ttem_coordinate_group.groups.keys()))):
        tmp =distance_of_two_points(list(ttem_coordinate_group.groups.keys())[i-1][0],
                                     list(ttem_coordinate_group.groups.keys())[i-1][1],
                                     list(ttem_coordinate_group.groups.keys())[i][0],
                                     list(ttem_coordinate_group.groups.keys())[i][1])
        distance=distance+tmp
    distance_range = int(distance*m_to_ft_factor)
    ttem_preprocessed['Elevation_Cell'] = ttem_preprocessed['Elevation_Cell']*m_to_ft_factor
    ttem_preprocessed['Elevation_End'] = ttem_preprocessed['Elevation_End']*m_to_ft_factor
    elevation_max = ttem_preprocessed['Elevation_Cell'].max()

    elevation_end_min = ttem_preprocessed['Elevation_End'].min()

    elevation_range = int(elevation_max-elevation_end_min)
    print('elevation_range',elevation_max,elevation_end_min,elevation_range)
    ttem_preprocessed['Elevation_Cell_for_plot'] = abs(ttem_preprocessed['Elevation_Cell'] - elevation_max)
    ttem_preprocessed['Elevation_End_for_plot'] = abs(ttem_preprocessed['Elevation_End']-ttem_preprocessed['Elevation_End'].max())
    y_distance = elevation_range
    x_distance = distance_range/10
    empty_grid = np.full((int((y_distance+500)),int(x_distance)),np.nan)
# Fill in the tTEM data by loop through the grid
    first_iteration = True
    for name, group in ttem_coordinate_group:
        if first_iteration is True:
            first_iteration = False
            initial_distance = 1
            tmp_coor = name
            for index, line in group.iterrows():
                elevation_cell_round = line['Elevation_Cell_for_plot']
                elevation_end_round = line['Elevation_End_for_plot']
                empty_grid[int((elevation_cell_round)*m_to_ft_factor):int((elevation_end_round+6)*m_to_ft_factor),int((initial_distance-10)*m_to_ft_factor/10):int(((initial_distance+10)*m_to_ft_factor/10))] = np.log10(line['Resistivity'])
        else:
            initial_distance = initial_distance + distance_of_two_points(tmp_coor[0],tmp_coor[1],name[0],name[1])
            tmp_coor = name
            for index, line in group.iterrows():
                elevation_cell_round = line['Elevation_Cell_for_plot']
                empty_grid[int((elevation_cell_round-6)*m_to_ft_factor):int((elevation_end_round+6)*m_to_ft_factor),int((initial_distance-10)*m_to_ft_factor/10):int(((initial_distance+10)*m_to_ft_factor/10))] = np.log10(line['Resistivity'])
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
            )
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
            tickvals=[0,1,2,3],
            tickfont=dict(
                size=30
            ),
            ticktext=['1','10','100','1000'],
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

def Resistivity_plot_ID_distance(ttem_df, line_filter=None, fence=None, plot_ft=False):

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
def rock_plot_ID(ttem_df, line_filter=None, fence=None, plot_ft=False):

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
    survey_length = distance_of_two_points((ttem_preprocessed['UTMX'].min(),
                                            ttem_preprocessed['UTMY'].min()),
                                           (ttem_preprocessed['UTMX'].max(),
                                            ttem_preprocessed['UTMY'].max()))
    ttem_preprocessed['Elevation_Cell_for_plot'] = abs(ttem_preprocessed['Elevation_Cell'] - elevation_max)
    ttem_preprocessed['Elevation_End_for_plot'] = abs(ttem_preprocessed['Elevation_End']-ttem_preprocessed['Elevation_End'].max())
    distance_range = ttem_preprocessed['ID'].max() - ttem_preprocessed['ID'].min()
    ttem_preprocessed['ID_for_plot'] = ttem_preprocessed['ID'] - ttem_preprocessed['ID'].min()
    y_distance = elevation_range
    x_distance = distance_range
    empty_grid = np.full((int((y_distance)),int(x_distance)),np.nan)
# Fill in the tTEM data by loop through the grid
    colorscale = [[0, 'blue'], [0.5, 'yellow'], [1, 'red']]
    ttem_ID_group = ttem_preprocessed.groupby('ID_for_plot')
    for name, group in ttem_ID_group:
        for index, line in group.iterrows():
            empty_grid[int(line['Elevation_Cell_for_plot']):int(line['Elevation_End_for_plot']+6*m_to_ft_factor), int(name)-1:int(name)+1] = line['Identity_n']
    fig = px.imshow(empty_grid, range_color=(1,3),color_continuous_scale=colorscale)
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
            tickvals=list(np.arange(0, empty_grid.shape[1], 100)),
            ticktext=[str(int(i)) for i in np.linspace(0, survey_length, 6)]
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
            tickvals=list(np.arange(0, empty_grid.shape[0], 20)),
            ticktext=[str(int(elevation_max-i)) for i in np.arange(0,empty_grid.shape[0],20)]
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
            tickvals=[1,2,3],
            tickfont=dict(
                size=30
            ),
            ticktext=['Fine grained','Mix grained','Coarse grained'],
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
def rock_plot_ID_distance(ttem_df, welllog, WIN=None, line_filter=None, ID_filter=None ,fence=None, plot_ft=False):

    """
    This is the function that plot the given dataframe into gridded plot (for better looking)

    :param DataFrame ttem_df: a data frame contains tTEM data exported from Aarhus workbench
    :param int line_filter: In tTEM dataframe you can filter tTEM data by Line_No (defaults: None)
    :param list ID_filter: In tTEM dataframe you can filter tTEM data by ID (defaults: None) [start_ID, end_ID]
    :param list fence: receive a UTM coordinate to cut data, ex: [xmin, ymin, xmax, ymax] (defaults: None)
    :param plot_ft: True: plot all parameter in feet, False: plot in meter (defaults: True)

    :return:
        - fig - plotly fig of the block_plot
        - ttem_for_plot - the tTEM dataframe use for plot
        - empty_grid - the grid use to plot the figure
    """
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
    if isinstance(welllog, pathlib.PurePath):
        _, welllog_df = core.bootstrap.select_closest(ttem_preprocessed, welllog)
    elif isinstance(welllog, pd.DataFrame):
        welllog_df = welllog
    if WIN is not None:
        if isinstance(WIN, list):
            welllog_df = welllog_df[welllog_df['Bore'].isin(WIN)]
        elif isinstance(WIN, (int,str)):
            welllog_df = welllog_df[welllog_df['Bore'] == str(WIN)]

    well_groups = welllog_df.groupby('Bore')
    ttem_preprocessed['Elevation_Cell'] = ttem_preprocessed['Elevation_Cell']*m_to_ft_factor
    ttem_preprocessed['Elevation_End'] = ttem_preprocessed['Elevation_End']*m_to_ft_factor
    elevation_max = ttem_preprocessed['Elevation_Cell'].max()
    elevation_end_max = ttem_preprocessed['Elevation_End'].max()
    elevation_end_min = ttem_preprocessed['Elevation_End'].min()
    elevation_range = int(elevation_max-elevation_end_min)
    ttem_preprocessed['Elevation_Cell_for_plot'] = abs(ttem_preprocessed['Elevation_Cell'] - elevation_max)
    ttem_preprocessed['Elevation_End_for_plot'] = abs(ttem_preprocessed['Elevation_End']-ttem_preprocessed['Elevation_End'].max())
    ID_groups = ttem_preprocessed.groupby('ID')
    UTM_groups = ID_groups[['UTMX','UTMY']].first().values.tolist()
    UTM_shift = ID_groups[['UTMX','UTMY']].first().shift(1).values.tolist()
    well_name_n_distance = {}

    for name, group in well_groups:
        tmp = [[group['UTMX'].iloc[0],group['UTMY'].iloc[0]]]*len(UTM_groups)
        triangle_a = min(list(map(distance_of_two_points, UTM_groups, tmp)))
        triangle_c = distance_of_two_points(UTM_groups[0],[group['UTMX'].iloc[0],group['UTMY'].iloc[0]])
        triangle_b = np.sqrt(triangle_c**2-triangle_a**2)
        well_name_n_distance[name] = triangle_b
    welllog_df['Elevation_for_plot'] = well_groups['Elevation'].transform(lambda x: abs(x-elevation_max))
    distance = list(map(distance_of_two_points, UTM_groups, UTM_shift))
    distance[0] = 0
    distance_marker = np.cumsum(distance)
    distance_range = sum(distance)
    y_distance = elevation_range*10
    x_distance = distance_range
    empty_grid = np.full((int((y_distance)),int(x_distance)),np.nan)
# Fill in the tTEM data by loop through the grid
    colorscale = [[0, 'blue'], [0.5, 'yellow'], [1, 'red']]
    loop_count = 0
    for name, group in ID_groups:
        for index, line in group.iterrows():
            empty_grid[int(line['Elevation_Cell_for_plot']*10):int((line['Elevation_End_for_plot']+6*m_to_ft_factor)*10),
                        int(distance_marker[loop_count])-5:int(distance_marker[loop_count])+5] = line['Identity_n']
        loop_count += 1
    for name, group in well_groups:
        for index, line in group.iterrows():
            empty_grid[int(line['Elevation_for_plot']*10):int((line['Elevation_for_plot']+0.1*m_to_ft_factor)*10),
                        int(well_name_n_distance[name])-25:int(well_name_n_distance[name])+25] = line['Keyword_n']
        loop_count += 1
    fig = px.imshow(empty_grid, range_color=(1,3),color_continuous_scale=colorscale)
    for name, group in well_groups:
        fig.add_annotation(
            x=well_name_n_distance[name],
            y=5,
            text=name,
            showarrow=True,
            font=dict(
                size=30,
                color='Black'
            )

        )
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
                size=30
            ),
            tickfont=dict(
                family="Arial",
                size=25
            ),
            tickmode = 'array',
            tickvals=list(np.arange(0, empty_grid.shape[1], 1000)),
            ticktext=[str(int(i)) for i in np.linspace(0, distance_range, 6)]
        ),
        yaxis=dict(
            titlefont=dict(
                family="Arial",
                size=30
            ),
            tickfont=dict(
                family="Arial",
                size=25
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
            title='Rock Type',
            tickvals=[1,2,3],
            tickfont=dict(
                size=30
            ),
            ticktext=['Fine grained','Mix grained','Coarse grained'],
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
def rock_plot(rock_trans_df, line_filter=None, fence=None, plot_ft=True):

    """
    This is the function that plot the given dataframe into gridded plot (for better looking)

    :param ttem_df: a data frame contains tTEM data exported from Aarhus workbench
    :param line_filter: In tTEM dataframe you can filter tTEM data by Line_No (defaults: None)
    :param fence: receive a UTM coordinate to cut data, ex: [xmin, ymin, xmax, ymax] (defaults: None)
    :param plot_ft: True: plot all parameter in feet, False: plot in meter (defaults: True)
    :return:
    """
    if line_filter is not None and fence is not None:
        ttem_preprocess_line = filter_line(rock_trans_df,line_filter)
        ttem_preprocess_fence = data_fence(ttem_preprocess_line,fence[0],fence[1],fence[2],fence[3])
        ttem_preprocessed = ttem_preprocess_fence.copy()
    elif line_filter is not None:
        ttem_preprocess_line = filter_line(rock_trans_df,line_filter)
        ttem_preprocessed = ttem_preprocess_line.copy()
    elif fence is not None:
        ttem_preprocess_fence = data_fence(rock_trans_df,fence[0],fence[1],fence[2],fence[3])
        ttem_preprocessed = ttem_preprocess_fence.copy()
    else:
        ttem_preprocessed = rock_trans_df
    if plot_ft is True:
        m_to_ft_factor = 3.28
        unit = 'ft'
    else:
        m_to_ft_factor = 1
        unit = 'm'
    ttem_preprocessed['distance'] = np.sqrt(ttem_preprocessed['UTMX']**2 + ttem_preprocessed['UTMY']**2)*m_to_ft_factor
    ttem_preprocessed['Elevation_Cell'] = ttem_preprocessed['Elevation_Cell']*m_to_ft_factor
    ttem_preprocessed['Elevation_End'] = ttem_preprocessed['Elevation_End']*m_to_ft_factor
    distance_min = ttem_preprocessed['distance'].min()
    elevation_max = ttem_preprocessed['Elevation_Cell'].max()
    elevation_end_max = ttem_preprocessed['Elevation_End'].max()
    ttem_preprocessed['distance_for_plot'] = ttem_preprocessed['distance'] - distance_min
    ttem_preprocessed['Elevation_Cell_for_plot'] = ttem_preprocessed['Elevation_Cell'] - elevation_max
    ttem_preprocessed['Elevation_End_for_plot'] = ttem_preprocessed['Elevation_End'] - elevation_end_max
    ttem_for_plot = abs(ttem_preprocessed[['distance_for_plot', 'Identity_n','Elevation_Cell_for_plot','Elevation_End_for_plot']])
    x_distance = ttem_for_plot['distance_for_plot'].max()
    y_distance = ttem_for_plot['Elevation_End_for_plot'].max()
    empty_grid = np.full((int((y_distance+10)*10),int(x_distance)),np.nan)
# Fill in the tTEM data by loop through the grid
    for index, line in ttem_for_plot.iterrows():
        distance_round = int(line['distance_for_plot'])
        elevation_cell_round = int(line['Elevation_Cell_for_plot'])
        elevation_end_round = int(line['Elevation_End_for_plot'])
        empty_grid[int(elevation_cell_round*10-3*10*m_to_ft_factor):int(elevation_end_round*10+3*10*m_to_ft_factor),int(distance_round):int(distance_round+1)] = line['Identity_n']
    fig = px.imshow(empty_grid, range_color=(1,3))
    fig.update_layout(paper_bgcolor='white',plot_bgcolor='white',font_color='black')
    fig.update_layout(
        dict(
        xaxis=dict(
            title='Distance ({})'.format(unit),
            scaleanchor = 'y',
            scaleratio = 10,
            titlefont=dict(
                family="Arial",
                size=50
            ),
            tickfont=dict(
                family="Arial",
                size=45
            )
        ),
        yaxis=dict(
            title='Elevation ({})'.format(unit),
            titlefont=dict(
                family="Arial",
                size=50
            ),
            tickfont=dict(
                family="Arial",
                size=45
            ),
            tickmode='array',
            tickvals=list(np.arange(0,empty_grid.shape[0],500)),
            #TODO: this need more modification for m scale
            ticktext=[str(int(elevation_max-i/10)) for i in np.arange(0,empty_grid.shape[0],500)]
        ),
        )
    )
    fig.update_coloraxes(
        colorbar=dict(
            ticks='outside',
            title='Rock Type',
            tickvals=[1,2,3],
            tickfont=dict(
                size=30
            ),
            ticktext=['Fine grained','Mixed Grained','Coarse Grained'],
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
    return fig, ttem_for_plot, empty_grid
def plot_hist(dataframe):
    fig = px.histogram(dataframe,x='Resistivity',nbins=500)
    fig.update_xaxes(range=[0,100],
                     title = 'Resistivity',
                     title_font=dict(
                         size=45
                     ),
                     tickfont=dict(
                         size=45
                     ),
    )
    fig.update_yaxes(title = 'Count',
                     title_font=dict(
                         size=45
                     ),
                     tickfont=dict(
                         size=45
                     ),
    )

    return fig
ttem_north = core.main.ProcessTTEM(ttem_path=[ttemname_north],
                                   welllog=welllog,
                                   DOI_path=DOI)
ttem_north_data = ttem_north.data()
ttem_center = core.main.ProcessTTEM(ttem_path=[ttemname_center],
                                    welllog=welllog,
                                    DOI_path=DOI)
ttem_center_data = ttem_center.data()

ttem_north_jason = ttem_north_data[ttem_north_data['Line_No'].isin([180, 190])]
ttem_pure_north = ttem_north_data[~ttem_north_data['Line_No'].isin([180, 190])]
hist_north = go.Figure(go.Histogram(x=ttem_pure_north['Resistivity'],
                                    marker=dict(color='orange'),
                                    nbinsx=100,
                                    xbins=dict(start=1,end=100,size=(100)/100)))
hist_north.update_traces(marker=dict(line=dict(color='black', width=1)))
hist_north.update_layout(
    xaxis={
        'title':{
            'text': 'Resistivity ohm',
            'font': {
                'size': 50,
                'family': 'Arial',
                #'weight': 'bold'
            }
        },
        'tickfont': {
            'size': 45,
            'family': 'Arial',
            #'weight': 'bold'
        }
    },
    yaxis={
        'title': {
            'text': 'Count',
            'font': {
                'size': 50,
                'family': 'Arial',
                #'weight': 'bold'
            }
        },
        'tickfont': {
            'size': 45,
            'family': 'Arial',
            #'weight': 'bold'
        }
    }
)
#hist_north.update_xaxes(range=[1,100])
hist_jason = go.Figure(go.Histogram(x=ttem_north_jason['Resistivity'],
                                    marker=dict(color='orange'),
                                    nbinsx=100,
                                    xbins=dict(start=1,end=100,size=(100)/100)))
hist_jason.update_traces(marker=dict(line=dict(color='black', width=1)))
hist_jason.update_layout(
    xaxis={
        'title':{
            'text': 'Resistivity ohm',
            'font': {
                'size': 50,
                'family': 'Arial',
                #'weight': 'bold'
            }
        },
        'tickfont': {
            'size': 45,
            'family': 'Arial',
            #'weight': 'bold'
        }
    },
    yaxis={
        'title': {
            'text': 'Count',
            'font': {
                'size': 50,
                'family': 'Arial',
                #'weight': 'bold'
            }
        },
        'tickfont': {
            'size': 45,
            'family': 'Arial',
            #'weight': 'bold'
        }
    }
)
hist_center = go.Figure(go.Histogram(x=ttem_center_data['Resistivity'],
                                     marker=dict(color='orange'),
                                     nbinsx=100,
                                     xbins=dict(start=1,end=100,size=(100)/100)))
hist_center.update_traces(marker=dict(line=dict(color='black', width=1)))
hist_center.update_layout(
    xaxis={
        'title':{
            'text': 'Resistivity ohm',
            'font': {
                'size': 50,
                'family': 'Arial',
                #'weight': 'bold'
            }
        },
        'tickfont': {
            'size': 45,
            'family': 'Arial',
            #'weight': 'bold'
        }
    },
    yaxis={
        'title': {
            'text': 'Count',
            'font': {
                'size': 50,
                'family': 'Arial',
                #'weight': 'bold'
            }
        },
        'tickfont': {
            'size': 45,
            'family': 'Arial',
            #'weight': 'bold'
        }
    }
)
#hist_center.update_xaxes(range=[1,100])
#hist_north.show()
#time.sleep(2)
#hist_jason.show()
#time.sleep(2)
#hist_center.show()
Resi_conf_df_publish_dry = pd.DataFrame({'Fine_conf':[8,25],'Mix_conf':[25,31],'Coarse_conf':[31,150]})
Resi_conf_df_publish_wet = pd.DataFrame({'Fine_conf':[6,12],'Mix_conf':[12,22],'Coarse_conf':[22,43]})
Resi_conf_df_jason = pd.DataFrame({'Fine_conf':[6,23.5],'Mix_conf':[24.4,37],'Coarse_conf':[34.2,41.8]})
#ttem_test_data = ttem_center_data[ttem_center_data['Line_No'].isin([190])]
ttem_center_data_190 = ttem_center_data[ttem_center_data['Line_No'] == 190]
ttem_north_data_100 = ttem_north_data[ttem_north_data['Line_No'] == 100]
rk_trans_190 = core.Rock_trans.rock_transform(ttem_center_data_190, Resi_conf_df_publish_dry)
#rock_plot_ID(rk_trans_190)
rk_trans_100 =  core.Rock_trans.rock_transform(ttem_north_data_100, Resi_conf_df_publish_wet)
jason_farm = ttem_north_data[(ttem_north_data['ID'] <2750)&(ttem_north_data['ID']>2484) ]
rk_trans_jason = core.Rock_trans.rock_transform(jason_farm, Resi_conf_df_publish_dry)
rk_trans_jason_interpreted = core.Rock_trans.rock_transform(jason_farm, Resi_conf_df_jason)
#########rock trans for north jason and center
ttem_pure_north = ttem_north_data[~ttem_north_data['Line_No'].isin([180,190])]
Resi_conf_df_north_interpreted = pd.DataFrame({'Fine_conf':[5.7,15.2],'Mix_conf':[12,22],'Coarse_conf':[27.5,35.4]})
Resi_conf_df_center_interpreted = pd.DataFrame({'Fine_conf':[28.9,32],'Mix_conf':[32.3,40],'Coarse_conf':[36.1,42.7]})
Resi_conf_df_mid_manual = pd.DataFrame({'Fine_conf':[19.3,35.9],'Mix_conf':[29.7,41.3],'Coarse_conf':[37.6,122.4]})
rk_trans_north = core.Rock_trans.rock_transform(ttem_pure_north, Resi_conf_df_north_interpreted)
rk_trans_center = core.Rock_trans.rock_transform(ttem_center_data, Resi_conf_df_center_interpreted)
rk_trans_center_manual = core.Rock_trans.rock_transform(ttem_center_data, Resi_conf_df_mid_manual)
rk_trans_center_published= core.Rock_trans.rock_transform(ttem_center_data, Resi_conf_df_publish_dry)

#rock_plot_ID(rk_trans_center_manual,line_filter=190)
# plot hist for lithology
lithopath = Path(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\2022\2022SS\ArcGIS\ttem_specs\export')
north = pd.read_csv(lithopath.joinpath(r'northlitho.csv'))
jason = pd.read_csv(lithopath.joinpath(r'jason.csv'))
center = pd.read_csv(lithopath.joinpath(r'center.csv'))
ratio_north = core.Rock_trans.pct_count(rk_trans_north)
ratio_jason = core.Rock_trans.pct_count(rk_trans_jason_interpreted)
ratio_center = core.Rock_trans.pct_count(rk_trans_center)
ratio_north_fine = ratio_north.groupby('Identity').get_group('Fine_grain')
ratio_jason_fine = ratio_jason.groupby('Identity').get_group('Fine_grain')
ratio_center_fine = ratio_center.groupby('Identity').get_group('Fine_grain')


hist_litho_north = go.Figure(go.Histogram(x=ratio_north_fine['ratio']*100,
                                          nbinsx=50,
                                          xbins=dict(start=1,end=100,size=(100)/50)))
hist_litho_north.update_traces(marker=dict(line=dict(color='black', width=1)))
hist_litho_north.update_layout(
    xaxis={
        'range':[0,100],
        'title':{
            'text': 'Fine grained %',
            'font': {
                'size': 50,
                'family': 'Arial',
                #'weight': 'bold'
            }
        },
        'tickfont': {
            'size': 45,
            'family': 'Arial',
            #'weight': 'bold'
        },

    },
    yaxis={
        'title': {
            'text': 'Count',
            'font': {
                'size': 50,
                'family': 'Arial',
                #'weight': 'bold'
            }
        },
        'tickfont': {
            'size': 45,
            'family': 'Arial',
            #'weight': 'bold'
        }
    }
)
hist_litho_jason = go.Figure(go.Histogram(x=ratio_jason_fine['ratio']*100,nbinsx=50,xbins=dict(start=1,end=100,size=(100)/50)))
hist_litho_jason.update_traces(marker=dict(line=dict(color='black', width=1)))
hist_litho_jason.update_layout(
    xaxis={
        'range':[0,100],
        'title':{
            'text': 'Fine grained %',
            'font': {
                'size': 50,
                'family': 'Arial',
                #'weight': 'bold'
            }
        },
        'tickfont': {
            'size': 45,
            'family': 'Arial',
            #'weight': 'bold'
        },

    },
    yaxis={
        'title': {
            'text': 'Count',
            'font': {
                'size': 50,
                'family': 'Arial',
                #'weight': 'bold'
            }
        },
        'tickfont': {
            'size': 45,
            'family': 'Arial',
            #'weight': 'bold'
        }
    }
)
hist_litho_center = go.Figure(go.Histogram(x=ratio_center_fine['ratio']*100,nbinsx=50,xbins=dict(start=1,end=100,size=(100)/50)))
hist_litho_center.update_traces(marker=dict(line=dict(color='black', width=1)))
hist_litho_center.update_layout(
    xaxis={
        'range':[0,100],
        'title':{
            'text': 'Fine grained %',
            'font': {
                'size': 50,
                'family': 'Arial',
                #'weight': 'bold'
            }
        },
        'tickfont': {
            'size': 45,
            'family': 'Arial',
            #'weight': 'bold'
        },

    },
    yaxis={
        'title': {
            'text': 'Count',
            'font': {
                'size': 50,
                'family': 'Arial',
                #'weight': 'bold'
            }
        },
        'tickfont': {
            'size': 45,
            'family': 'Arial',
            #'weight': 'bold'
        }
    }
)
# Well vs rk trans scatter plot
def linear_fill(group, factor=100):
    group.reset_index(drop=True, inplace=True)
    newgroup = group.loc[group.index.repeat(group.Thickness * factor)]
    mul_per_gr = newgroup.groupby('Elevation_Cell').cumcount()
    newgroup['Elevation_Cell'] = newgroup['Elevation_Cell'].subtract(mul_per_gr * 1 / factor)
    newgroup['Depth_top'] = newgroup['Depth_top'].add(mul_per_gr * 1 / factor)
    newgroup['Depth_bottom'] = newgroup['Depth_top'].add(1 / factor)
    newgroup['Elevation_End'] = newgroup['Elevation_Cell'].subtract(1 / factor)
    newgroup['Thickness'] = 1 / factor
    newgroup.reset_index(drop=True, inplace=True)
    return newgroup
def closest_rock(rk_trans, welllog):
    rk_trans_group = rk_trans.groupby(['UTMX','UTMY'])
    group_length = len(list(rk_trans_group.groups.keys()))
    welllog_coor_tuple = (welllog['UTMX'].iloc[0],welllog['UTMY'].iloc[0])
    distance = list(map(distance_of_two_points,[welllog_coor_tuple for _ in range(group_length)],
                   list(rk_trans_group.groups.keys())))
    if min(distance) > 500:
        return print('No close well found')
    closest_coor=list(rk_trans_group.groups.keys())[distance.index(min(distance))]
    ttem_data = rk_trans_group.get_group(closest_coor)
    ttem_data = linear_fill(ttem_data, 100)
    return ttem_data
def levenshtein_distance(list1, list2):
    m = len(list1)
    n = len(list2)

    # Create a matrix to store the distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill in the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Return the Levenshtein distance
    return dp[m][n]
def rock_well_corr (rk_trans, welllog):
    from scipy import  spatial
    if isinstance(welllog,(str, pathlib.PurePath)):
        welllog = core.process_well.format_well(welllog, upscale=100)
        welllog = core.bootstrap.select_closest(rk_trans, welllog, 500)[1]
    elif isinstance(welllog, pd.DataFrame):
        pass
    welllog['Elevation'] = welllog['Elevation'].round(2)
    rk_trans['Elevation_Cell'] = rk_trans['Elevation_Cell'].round(2)
    well_group = welllog.groupby('Bore')
    corr_list = {}
    for name, group in well_group:
        close_rock = closest_rock(rk_trans, group)
        merge = pd.merge(group, close_rock, left_on=['Elevation'], right_on=['Elevation_Cell'])
        corr = merge['Keyword_n'].corr(merge['Identity_n'])
        cosin_similiar = 1- float(spatial.distance.cosine(merge['Keyword_n'], merge['Identity_n']))
        similiar = (merge['Keyword_n'] == merge['Identity_n']).sum()/len(merge['Keyword_n'])
        #leven_dist = levenshtein_distance(merge['Keyword_n'], merge['Identity_n'])
        corr_list[name] = {'Pearson':corr, 'Cosin':cosin_similiar, 'simple':similiar}
    return corr_list
def value_search(ttem_data_df, welllog, WIN, rho_fine=10, rho_coarse=25,step=1, loop_range=20,correct=False):
    #progress = Bar('Processing', max=100)
    import itertools
    if isinstance(welllog,(str, pathlib.PurePath)):
        welllog_df = core.process_well.format_well(welllog, upscale=100)
    elif isinstance(welllog, pd.DataFrame):
        welllog_df = welllog
    welllog_WIN = welllog_df[welllog_df['Bore']==str(WIN)]
    try:
        ttem_data = closest_rock(ttem_data_df, welllog_WIN)
    except TypeError:
         raise ('well log expty')
    fine_range = np.arange(rho_fine, rho_fine + (step * loop_range), step)
    coarse_range = np.arange(rho_coarse, rho_coarse + (step * loop_range), step)
    resistivity_list = list(itertools.product(fine_range, coarse_range))
    #Resi_conf_df = pd.DataFrame({'Fine_conf': [0, fine_rho], 'Mix_conf': [fine_rho, coarse_rho], 'Coarse_conf': [coarse_rho, 300]})
    welllog_WIN['Elevation'] = welllog_WIN['Elevation'].round(2)
    if correct is True:
        elevation_diff = welllog_WIN['Elevation'].iloc[0] - ttem_data['Elevation_Cell'].iloc[0]
        welllog_WIN['Elevation'] =welllog_WIN['Elevation'].subtract(elevation_diff)
        welllog_WIN['Elevation_End'] = welllog_WIN['Elevation_End'].subtract(elevation_diff)
    reslist = ['']*len(resistivity_list)
    corrlist = ['']*len(resistivity_list)
    i = 0
    for rho_fine, rho_coarse in resistivity_list:
        Resi_conf_df = pd.DataFrame(
            {'Fine_conf': [0, rho_fine],
             'Mix_conf': [rho_fine, rho_coarse],
             'Coarse_conf': [rho_coarse, 300]})
        reslist[i] = [rho_fine, rho_coarse]
        rk_trans = core.Rock_trans.rock_transform(ttem_data, Resi_conf_df)
        merge = pd.merge(welllog_WIN, rk_trans, left_on=['Elevation'], right_on=['Elevation_Cell'])
        #corr = merge['Keyword_n'].corr(merge['Identity_n'])
        corr2 = (merge['Keyword_n'] == merge['Identity_n']).sum()/len(merge['Keyword_n'])
        #corrlist.append(corr)
        corrlist[i]=corr2
        i += 1
    def fine_best_corr(reslist, corrlist):
        corrlist = [np.nan_to_num(x) for x in corrlist]
        best_corr = max(corrlist)
        match_list = [i for i, x in enumerate(corrlist) if x == best_corr]
        resistivity_list = [reslist[i] for i in match_list]
        resistivity_coarse_gt_fine = [i for i in resistivity_list if i[1]>i[0]]
        res_bkup_incase_empty = np.array(resistivity_list)
        resistivity_array = np.array(resistivity_coarse_gt_fine)
        if resistivity_array.size > 0:
            fine_grained_rho_mean = resistivity_array[:,0].mean()
            coarse_grained_rho_mean = resistivity_array[:,1].mean()
            export_result = {'similiarity':best_corr,'Fine_conf':fine_grained_rho_mean,'Coarse_conf':coarse_grained_rho_mean}
            return export_result
        else:
            coarse_grained_rho_mean = res_bkup_incase_empty[:, 1].mean()
            fine_grained_rho_mean = coarse_grained_rho_mean

            export_result = {'similiarity': best_corr, 'Fine_conf': fine_grained_rho_mean,
                             'Coarse_conf': coarse_grained_rho_mean}
            return export_result
    #resi_conf_df1, best1 = fine_best_corr(reslist, corrlist)
    best= fine_best_corr(reslist, corrlist)

    #progress.finish()
    return best
def value_search_res(ttem_data_df, welllog, WIN,
                     rho_fine:float=10,
                     rho_mix:float=15,
                     rho_coarse:float=25,
                     step:int=1,
                     loop_range:int=20,correct=False):
    """
    Assign each lithology type as corresponsing resistivity and run pearson correlation to fine the best resistiviry overall
    :param ttem_data_df: tTEM resistivity profile
    :param welllog: well log data
    :param WIN: The WIN number of the well log
    :param rho_fine: resistivity of fine-grained material
    :param rho_mix: resistivity of mix-grained material
    :param rho_coarse: resistivity of coarse-grained material
    :param step: loop of each step
    :param loop_range: the total range of the loop
    :return:
    """
    import itertools
    pd.options.mode.chained_assignment = None
    if isinstance(welllog,(str, pathlib.PurePath)):
        welllog_df = core.process_well.format_well(welllog, upscale=100)
    elif isinstance(welllog, pd.DataFrame):
        welllog_df = welllog
    welllog_WIN = welllog_df[welllog_df['Bore']==str(WIN)]
    welllog_WIN.fillna('',inplace=True)
    try:
        ttem_data = closest_rock(ttem_data_df, welllog_WIN)
    except TypeError:
         raise ('well log expty')
    welllog_WIN['Elevation'] = welllog_WIN['Elevation'].round(2)
    if correct is True:
        elevation_diff = welllog_WIN['Elevation'].iloc[0] - ttem_data['Elevation_Cell'].iloc[0]
        welllog_WIN['Elevation'] =welllog_WIN['Elevation'].subtract(elevation_diff)
        welllog_WIN['Elevation_End'] = welllog_WIN['Elevation_End'].subtract(elevation_diff)

    fine_range = np.arange(rho_fine, rho_fine+(step*loop_range), step)
    mix_range = np.arange(rho_mix, rho_mix+(step*loop_range), step)
    coarse_range = np.arange(rho_coarse, rho_coarse+(step*loop_range), step)
    resistivity_list = list(itertools.product(fine_range, mix_range, coarse_range))
    corr_list = ['']*len(resistivity_list)
    i=0
    #total = len(resistivity_list)
    #count = 0

    merge = pd.merge(welllog_WIN, ttem_data, left_on=['Elevation'], right_on=['Elevation_Cell'])
    choicelist = [merge['Keyword_n'] == 1, merge['Keyword_n'] == 2, merge['Keyword_n'] == 3]
    for rho_fine, rho_mix, rho_coarse in resistivity_list:

        choicelist2 = [rho_fine, rho_mix, rho_coarse]
        welllog_resistivity = np.select(choicelist, choicelist2)
        corr = np.corrcoef(welllog_resistivity, merge['Resistivity'])[0,1]
        corr_list[i]=corr
        i+=1
        #count = count + 1
        #print('{}/{}'.format(count, total))
    def best_corr(reslist, corrlist):
        corrlist = [np.nan_to_num(x) for x in corrlist]
        best = max(corrlist)
        match_list = [i for i, x in enumerate(corrlist) if x == best]
        resistivity_list = [reslist[i] for i in match_list]
        resistivity_coarse_gt_fine = [i for i in resistivity_list if i[2] > i[0]]
        res_bkup_incase_empty = np.array(resistivity_list)
        resistivity_array = np.array(resistivity_coarse_gt_fine)
        if resistivity_array.size > 0:
            fine_rho_avg = resistivity_array[:,0].mean()
            mix_rho_avg = resistivity_array[:,1].mean()
            coarse_rho_avg = resistivity_array[:,2].mean()
            export_result = {'pearson':best,'Fine_average':fine_rho_avg,'Mix_average':mix_rho_avg,'Coarse_average':coarse_rho_avg,}
            return export_result
        else:

            mix_rho_avg = res_bkup_incase_empty[:, 1].mean()
            coarse_rho_avg = res_bkup_incase_empty[:, 2].mean()
            fine_rho_avg = coarse_rho_avg
            export_result = {'pearson':best,'Fine_average':fine_rho_avg,'Mix_average':mix_rho_avg,'Coarse_average':coarse_rho_avg,}
            return export_result
    resi_conf_df = best_corr(resistivity_list, corr_list)

    return resi_conf_df
def find_matched_indexes(lst,value):
    return [ i for i, x in enumerate(lst) if x >= value]
def run_value_search(ttem_df, welllog, method = 'all',correct=False):
    if isinstance(welllog, (str, pathlib.PurePath)):
        welllog_df = core.process_well.format_well(welllog, upscale=100)
    elif isinstance(welllog, pd.DataFrame):
        welllog_df = welllog
    _, welllog_matched = core.bootstrap.select_closest(ttem_df, welllog_df)
    well_group = welllog_matched.groupby('Bore')
    result_dict = {}
    for name, group in well_group:
        ttem_single_match = closest_rock(ttem_df, group)
        if method == 1:
            result = value_search_res(ttem_single_match, group, name, rho_fine=5,
                                       rho_mix=10,
                                       rho_coarse=20,
                                       step=1,
                                       loop_range=40,correct=correct)
            result_dict[name] = result
        if method == 2:
            result2 = value_search(ttem_single_match, group, name,rho_fine=5, rho_coarse=20,step=1, loop_range=40,correct=correct)
            result_dict[name] = result2
        #result = value_search(ttem_df, group, name,rho_fine=5, rho_coarse=20,step=1, loop_range=40)
        if method == 'all':
            result = value_search_res(ttem_single_match, group, name, rho_fine=5,
                                       rho_mix=10,
                                       rho_coarse=20,
                                       step=1,
                                       loop_range=40,correct=correct)
            result2 = value_search(ttem_single_match, group, name,rho_fine=5, rho_coarse=20,step=1, loop_range=40,correct=correct)
            result_dict[name] = [result, result2]
        print({'{} is done'.format(name)})
    return result_dict
def plot_ttem_vs_well(ttem_data, welllog,WIN:int):
    result_dct = {}
    ori_welllog = core.process_well.format_well(welllog, upscale=100)
    welllog_WIN = ori_welllog[ori_welllog['Bore']==str(WIN)]
    result_dct['welllog_{}'.format(WIN)] = welllog_WIN
    ttem_close,_ = core.bootstrap.select_closest(ttem_data, welllog_WIN)
    ttem_fill = linear_fill(ttem_close,factor=100)
    welllog_WIN['Elevation'] = welllog_WIN['Elevation'].round(2)
    ttem_fill['Elevation_Cell'] = ttem_fill['Elevation_Cell'].round(2)
    merge = pd.merge(welllog_WIN, ttem_fill, left_on=['Elevation'], right_on=['Elevation_Cell'])
    result_dct['ttem_{}'.format(WIN)] = ttem_close
    fig_plot = utilities.plot.res_1d_plot(ttem_close)
    fig_plot.show()
    result_dct['plot_{}'.format(WIN)] = fig_plot
    well_plot = utilities.plot.plot_well_single(merge)
    well_plot.show()
    result_dct['well_plot_{}'.format(WIN)] = well_plot
    return result_dct

###for Profile 190
#result_value_search = run_value_search(ttem_center_data_190, welllog)
#rk_trans_190_value_search = tt.Rock_trans.rock_transform(ttem_center_data_190, result_value_search['431341'][0])

###072323 well log only use shallow DOI
ori_well = core.process_well.format_well(welllog, upscale=10)
well_group = ori_well.groupby('Bore')
def filter_above_60 (df):
    df = df[df['Elevation'] > (df['Elevation'].max()-60)]
    return df
result = well_group.apply(filter_above_60).reset_index(drop=True)
result_percent = result.groupby(['Bore'])
def calculate_well_lithology(group):
    total_thickness = group['Thickness'].sum()
    lithologypct = group.groupby('Keyword_n').agg({'Thickness':'sum'})/total_thickness
    result = group.iloc[0].to_frame().T
    try:
        result['fine_pct'] = lithologypct.loc[1,'Thickness']
    except KeyError:
        result['fine_pct'] = 0
    try:
        result['mix_pct'] = lithologypct.loc[2,'Thickness']
    except KeyError:
        result['mix_pct'] = 0
    try:
        result['coarse_pct'] = lithologypct.loc[3,'Thickness']
    except KeyError:
        result['coarse_pct'] = 0
    return result
well_60m = result_percent.apply(calculate_well_lithology)

#Rock physics transform comparsion of map and well log
Resi_conf_df_central_manual = pd.DataFrame({'Fine_conf':[5.7,36],'Mix_conf':[12,22],'Coarse_conf':[38,35.4]})
Resi_conf_df_north_grid_search = pd.DataFrame({'Fine_conf':[5.7,34],'Mix_conf':[12,22],'Coarse_conf':[35,35.4]})
Resi_conf_df_northeast_bootstrapping = pd.DataFrame({'Fine_conf':[5.7,32],'Mix_conf':[12,36],'Coarse_conf':[36,35.4]})
Resi_conf_df_north_grid_search = pd.DataFrame({'Fine_conf':[5.7,20],'Mix_conf':[12,22],'Coarse_conf':[21,35.4]})
rock_trans_manual = core.Rock_trans.rock_transform(ttem_center_data, Resi_conf_df_central_manual)
rock_trans_grid_search = core.Rock_trans.rock_transform(ttem_center_data, Resi_conf_df_north_grid_search)
rock_trans_bootstrapping = core.Rock_trans.rock_transform(ttem_center_data, Resi_conf_df_northeast_bootstrapping)
rock_ratio_manual = core.Rock_trans.pct_count(rock_trans_manual, grain='Fine_grain')
rock_ratio_grid = core.Rock_trans.pct_count(rock_trans_grid_search, grain='Fine_grain')
rock_ratio_bootstrapping = core.Rock_trans.pct_count(rock_trans_bootstrapping, grain='Fine_grain')
lithology_sample = pd.read_csv(workdir.joinpath('lithology_sample.csv'))
rock_manual_merge = pd.merge_asof(rock_ratio_manual,lithology_sample, left_on=['UTMX'], right_on=['X'])
rock_grid_merge = pd.merge_asof(rock_ratio_grid,lithology_sample, left_on=['UTMX'], right_on=['X'])
rock_bootstrapping_merge = pd.merge_asof(rock_ratio_bootstrapping,lithology_sample, left_on=['UTMX'], right_on=['X'])
scatter = rock_manual_merge.plot.scatter(x='ratio', y='GALAYERTORA10_CL')
import numpy
trendline = numpy.polyfit(rock_manual_merge['ratio'], rock_manual_merge['GALAYERTORA10_CL'], 1)
import statsmodels.api as sm
model_manual = sm.OLS(rock_manual_merge['GALAYERTORA10_CL'], rock_manual_merge['ratio'])
results = model_manual.fit()
corr_manual = np.corrcoef(rock_manual_merge['ratio'], rock_manual_merge['GALAYERTORA10_CL'])
def plot_lithology_ttem_ratio(rock_litho_merge):
    fig = px.scatter(rock_litho_merge, x="ratio", y="GALAYERTORA10_CL", trendline="ols", trendline_color_override="red")
    fig.update_layout(
        xaxis=dict(
            title='Rock Physics Transform Fine Grain Ratio',
            titlefont=dict(
                family='Arial',
                size=30
            ),
            tickfont=dict(
                family='Arial',
                size=25
            ),

        ),
        yaxis=dict(
            title='Lithology Fine Grain Ratio',
            titlefont=dict(
                family='Arial',
                size=30
            ),
            tickfont=dict(
                family='Arial',
                size=25
            ),

        ),
    )
    fig.update_traces(
        marker=dict(
            size=10
        ),
    )
    return fig
def correlation (rock_trans_result, lithology_result):
    rock_manual_merge = pd.merge_asof(rock_trans_result, lithology_result, left_on=['UTMX'], right_on=['X'])
    corr_manual = np.corrcoef(rock_manual_merge['ratio'], rock_manual_merge['GALAYERTORA10_CL'])
    #model_manual = sm.OLS(rock_manual_merge['GALAYERTORA10_CL'].values, rock_manual_merge['ratio'].values)
    #results = model_manual.fit()
    return corr_manual

def get_all_ttem_below_radius(rock_trans_ratio, welllog, wellname:int,radius=500):
    well = welllog[welllog['Bore'] == str(wellname)]
    def distance_of_two_points(group, point):
        distance = np.sqrt((group['UTMX'] - point[0]) ** 2 + (group['UTMY'] - point[1]) ** 2)
        group['distance'] = distance
        return group
    rock_trans_ratio = distance_of_two_points(rock_trans_ratio, point=(well['UTMX'].iloc[0], well['UTMY'].iloc[0]))
    close_sounding = rock_trans_ratio[rock_trans_ratio['distance'] < radius]
    ratio = close_sounding['ratio'].mean()
    max_thickness = close_sounding['T_sum'].mean()
    well_cut = well[well['Depth2_m']<=max_thickness]
    well_fine_percent = well_cut[well_cut['Keyword_n'] == 1]['Thickness'].sum()/well_cut['Thickness'].sum()
    return ratio,well_fine_percent
def well_rock_pearson_comparson(welllog, rock_ratio, radius=500):
    ori_well = core.process_well.format_well(welllog)
    well_group = ori_well.groupby('Bore')
    ttem_ratio = []
    well_ratio = []
    for name,group in well_group:
        tmp = get_all_ttem_below_radius(rock_ratio, ori_well, name, radius=radius)
        if pd.isna(tmp[0]):
            continue
        if pd.isna(tmp[1]):
            continue
        ttem_ratio.append(tmp[0])
        well_ratio.append(tmp[1])
    return ttem_ratio, well_ratio
result_ttem_manual, result_well_manual = well_rock_pearson_comparson(welllog,rock_ratio_manual)
result_ttem_grid, result_well_grid = well_rock_pearson_comparson(welllog,rock_ratio_grid)
result_ttem_bootstrapping, result_well_bootstrapping = well_rock_pearson_comparson(welllog,rock_ratio_bootstrapping)

corr_manual = np.corrcoef(result_ttem_manual, result_well_manual)
corr_grid = np.corrcoef(result_ttem_grid, result_well_grid)
corr_bootstrapping = np.corrcoef(result_ttem_bootstrapping, result_well_bootstrapping)
from scipy.stats import linregress
corr_manual_linear = linregress(result_ttem_manual, result_well_manual)
corr_grid_linear = linregress(result_ttem_grid, result_well_grid)
corr_bootstrapping_linear = linregress(result_ttem_bootstrapping, result_well_bootstrapping)

fig_manual = px.scatter(x=result_ttem_manual, y=result_well_manual, trendline="ols")
fig_manual.update_layout(
    xaxis=dict(
        title='Rock Physics Transform Fine Grain Ratio',
        titlefont=dict(
            family='Arial',
            size=30
        ),
        tickfont=dict(
            family='Arial',
            size=25
        )
    ),
    yaxis=dict(
        title='Lithology Fine Grain Ratio',
        titlefont=dict(
            family='Arial',
            size=30
        ),
        tickfont=dict(
            family='Arial',
            size=25
        )
    ),
)
fig_manual.update_traces(
    marker=dict(
        size=10
    ),
    line=dict(
        color='red',
        width=5,
    )
)
fig_grid = px.scatter(x=result_ttem_grid, y=result_well_grid, trendline="ols")
fig_grid.update_layout(
    xaxis=dict(
        title='Rock Physics Transform Fine Grain Ratio',
        titlefont=dict(
            family='Arial',
            size=30
        ),
        tickfont=dict(
            family='Arial',
            size=25
        )
    ),
    yaxis=dict(
        title='Lithology Fine Grain Ratio',
        titlefont=dict(
            family='Arial',
            size=30
        ),
        tickfont=dict(
            family='Arial',
            size=25
        )
    ),
)
fig_grid.update_traces(
    marker=dict(
        size=10
    ),
    line=dict(
        color='red',
        width=5,
    )
)
fig_bootstrapping = px.scatter(x=result_ttem_bootstrapping, y=result_well_bootstrapping, trendline="ols")
fig_bootstrapping.update_layout(
    xaxis=dict(
        title='Rock Physics Transform Fine Grain Ratio',
        titlefont=dict(
            family='Arial',
            size=30
        ),
        tickfont=dict(
            family='Arial',
            size=25
        )
    ),
    yaxis=dict(
        title='Lithology Fine Grain Ratio',
        titlefont=dict(
            family='Arial',
            size=30
        ),
        tickfont=dict(
            family='Arial',
            size=25
        )
    ),
)
fig_bootstrapping.update_traces(
    marker=dict(
        size=10
    ),
    line=dict(
        color='red',
        width=5,
    )
)
def export_grid_search(grid_search_dict):
    wells = list(grid_search_dict.keys())
    concat_list = []
    for i in wells:
        result = {'name':i,'Pearson':grid_search_dict[i][0],'Similiar':grid_search_dict[i][1][0]}
        concat_list.append(result)
    #result = pd.concat(concat_list)
    #result.to_csv(workdir.joinpath('grid_search_result.csv'))

    return concat_list
