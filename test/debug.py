import dash
from dash import dcc, html

from dash.dependencies import Input, Output, State

import plotly.graph_objs as go
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected = True)
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
        fig_hist.add_trace(go.Histogram(x=dataframe.mix, name='Mix', marker_color='Green', opacity=0.75))
    fig_hist.update_layout(paper_bgcolor='white',plot_bgcolor='white',font_color='black')

    return fig_hist
def data_fence(data_df,xmin,ymin,xmax,ymax):
    new_data_df = data_df[(data_df['UTMX']>xmin)&(data_df['UTMX']<xmax)&(data_df['UTMY']>ymin)&(data_df['UTMY']<ymax)]
    return new_data_df
#import importlib
#importlib.reload(tt.bootstrap)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})


import os
os.chdir(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah')
well_info = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah\USGSdownload\NWISMapperExport.xlsx'
location = r"C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\Gamma\location.csv"
welllog = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\Well_log.xlsx'
elevation = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah\usgs_water_elevation.csv'
ttemname = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\PD1_I01_MOD.xyz'
ttemname2 = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\PD22_I03_MOD.xyz'
DOI = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\DOID1_DOIStaE.xyz'
time = '2022-3'
ttem = core.main.ProcessTTEM(ttem_path=[ttemname],
                             welllog=welllog,
                             DOI_path=DOI,
                             layer_exclude=[])
data = ttem.data()
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
#water_corrected = data_fence(water_corrected, 349221.50,4203138,350539.30,4212763)
ttem_north = data[data['Line_No'] == 100]
ttem_north = data_fence(ttem_north,349221.50,4203138,350539.30,4212763)



def create_figure(skip_points=[]):
    dfs = ttem_north.drop(skip_points)
    return px.scatter_3d(dfs, x = 'UTMX', y = 'UTMY', z = 'Elevation_Cell')
f= create_figure()

app.layout = html.Div([html.Button('Delete', id='delete'),
                    html.Button('Clear Selection', id='clear'),
                    dcc.Graph(id = '3d_scat', figure=f),
                    html.Div('selected:'),
                    html.Div(id='selected_points'), #, style={'display': 'none'})),
                    html.Div('deleted:'),
                    html.Div(id='deleted_points') #, style={'display': 'none'}))
])

@app.callback(Output('deleted_points', 'children'),
            [Input('delete', 'n_clicks')],
            [State('selected_points', 'children'),
            State('deleted_points', 'children')])
def delete_points(n_clicks, selected_points, delete_points):
    print('n_clicks:',n_clicks)
    if selected_points:
        selected_points = json.loads(selected_points)
    else:
        selected_points = []

    if delete_points:
        deleted_points = json.loads(delete_points)
    else:
        deleted_points = []
    ns = [p['pointNumber'] for p in selected_points]
    new_indices = [df.index[n] for n in ns if df.index[n] not in deleted_points]
    print('new',new_indices)
    deleted_points.extend(new_indices)
    return json.dumps(deleted_points)



@app.callback(Output('selected_points', 'children'),
            [Input('3d_scat', 'clickData'),
                Input('deleted_points', 'children'),
                Input('clear', 'n_clicks')],
            [State('selected_points', 'children')])
def select_point(clickData, deleted_points, clear_clicked, selected_points):
    ctx = dash.callback_context
    ids = [c['prop_id'] for c in ctx.triggered]

    if selected_points:
        results = json.loads(selected_points)
    else:
        results = []


    if '3d_scat.clickData' in ids:
        if clickData:
            for p in clickData['points']:
                if p not in results:
                    results.append(p)
    if 'deleted_points.children' in ids or  'clear.n_clicks' in ids:
        results = []
    results = json.dumps(results)
    return results

@app.callback(Output('3d_scat', 'figure'),
            [Input('selected_points', 'children'),
            Input('deleted_points', 'children')],
            [State('deleted_points', 'children')])
def chart_3d( selected_points, deleted_points_input, deleted_points_state):
    global f
    deleted_points = json.loads(deleted_points_state) if deleted_points_state else []
    f = create_figure(deleted_points)

    selected_points = json.loads(selected_points) if selected_points else []
    if selected_points:
        f.add_trace(
            go.Scatter3d(
                mode='markers',
                x=[p['x'] for p in selected_points],
                y=[p['y'] for p in selected_points],
                z=[p['z'] for p in selected_points],
                marker=dict(
                    color='red',
                    size=5,
                    line=dict(
                        color='red',
                        width=2
                    )
                ),
                showlegend=False
            )
        )

    return f

if __name__ == '__main__':
    app.run_server(debug=True)