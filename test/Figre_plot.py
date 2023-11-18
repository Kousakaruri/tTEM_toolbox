from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
def block_plot(ttem_df,
               well_WIN=None,
               welllog=Path(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\Well_log.xlsx'),
               line_filter=None,
               fence=None,
               plot_ft=True):

    """
    This is the function that plot the given dataframe into gridded plot (for better looking)

    :param ttem_df: a data frame contains tTEM data exported from Aarhus workbench
    :param line_filter: In tTEM dataframe you can filter tTEM data by Line_No (defaults: None)
    :param fence: receive a UTM coordinate to cut data, ex: [xmin, ymin, xmax, ymax] (defaults: None)
    :param plot_ft: True: plot all parameter in feet, False: plot in meter (defaults: True)
    :return:
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
    ttem_preprocessed['distance'] = np.sqrt(ttem_preprocessed['UTMX']**2 + ttem_preprocessed['UTMY']**2)*m_to_ft_factor
    ttem_preprocessed['Elevation_Cell'] = ttem_preprocessed['Elevation_Cell']*m_to_ft_factor
    ttem_preprocessed['Elevation_End'] = ttem_preprocessed['Elevation_End']*m_to_ft_factor
    distance_min = ttem_preprocessed['distance'].min()
    elevation_max = ttem_preprocessed['Elevation_Cell'].max()
    elevation_end_max = ttem_preprocessed['Elevation_End'].max()
    ttem_preprocessed['distance_for_plot'] = ttem_preprocessed['distance'] - distance_min
    ttem_preprocessed['Elevation_Cell_for_plot'] = ttem_preprocessed['Elevation_Cell'] - elevation_max
    ttem_preprocessed['Elevation_End_for_plot'] = ttem_preprocessed['Elevation_End'] - elevation_end_max
    ttem_for_plot = abs(ttem_preprocessed[['distance_for_plot', 'Resistivity','Elevation_Cell_for_plot','Elevation_End_for_plot','Elevation_Cell','Elevation_End']])
    x_distance = ttem_for_plot['distance_for_plot'].max()
    y_distance = ttem_for_plot['Elevation_End_for_plot'].max()
    empty_grid = np.full((int((y_distance+10)*10),int(x_distance)),np.nan)
# Fill in the tTEM data by loop through the grid
    for index, line in ttem_for_plot.iterrows():
        distance_round = int(line['distance_for_plot'])
        elevation_cell_round = int(line['Elevation_Cell_for_plot']*10)
        elevation_end_round = int(line['Elevation_End_for_plot']*10)
        empty_grid[elevation_cell_round-int(6*m_to_ft_factor):elevation_end_round+int(6*m_to_ft_factor),distance_round:distance_round+int(6*m_to_ft_factor)] = np.log10(line['Resistivity'])
    if well_WIN is not None:
        _, matched_well = core.bootstrap.select_closest(ttem_df, welllog, distance=1000)
        well_df = matched_well[matched_well['Bore'] == str(well_WIN)].copy()
        well_df['distance'] = np.sqrt(well_df['UTMX'] ** 2 + well_df['UTMY'] ** 2)*m_to_ft_factor
        well_df['distance_for_plot'] = well_df['distance'] - distance_min
        well_df['Elevation_for_plot'] = well_df['Elevation']*m_to_ft_factor - elevation_max
        well_df['Elevation_for_plot'] = abs(well_df['Elevation_for_plot'])
        for index, line in well_df.iterrows():
            distance = int(line['distance_for_plot'] )
            elevation = int(line['Elevation_for_plot'] * 10)
            empty_grid[elevation - int(6*m_to_ft_factor):elevation + int(7*m_to_ft_factor), distance - int(15*m_to_ft_factor):distance + int(15*m_to_ft_factor)] = line['Keyword_n']
    fig = px.imshow(empty_grid, range_color=(0,3),color_continuous_scale=colorRes)
    fig.update_layout(paper_bgcolor='white',plot_bgcolor='white',font_color='black')
    fig.update_layout(
        yaxis=dict(
        title='Elevation ({})'.format(unit),
        gridcolor='black',
        linewidth=2,
        showline=True,
        linecolor='black'
        #tickmode='linear',
        #tick0=1774,
        #dtick=100

    ),
        xaxis=dict(
        title='Distance ({})'.format(unit),
        linewidth=2,
        gridcolor='black',
        showline = True,
        linecolor ='black'
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
            )
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
    return fig, ttem_for_plot

if __name__=="__main__":
    ttem_north = core.main.ProcessTTEM(ttem_path=[ttemname_north],
                                       welllog=welllog,
                                       DOI_path=DOI)
    ttem_north_df = ttem_north.data()
    fig, ttem_for_plot = block_plot(ttem_north_df, well_WIN='0375002P00 27210',line_filter=110)
