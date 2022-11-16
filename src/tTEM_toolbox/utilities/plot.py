import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import linregress
pd.options.plotting.backend = "plotly"
#pio.renderers.default = "browser"
def generate_trace(data, *args):
    # use plotly to add plot
    # define color bar
    # well log color bar fine, mix, coarse grained
    colorwell = [[0, 'rgb(0,204,255)'],
                 [0.5, 'rgb(204,204,0)'],
                 [1, 'rgb(255,0,0)']
                 ]
    # rock transform color bar fine, mix, coarse grained
    colorrock = [[0, 'rgb(30,144,255)'],
                 [0.5, 'rgb(255,255,0)'],
                 [1,'rgb(255,0,0)']]
    # tTEM data color bar, match color bar in aarhus workbench
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
                [1, 'rgb(230,155,255)']
                ]
    if np.isin("ttem", args):
        ttemfigdata = go.Scatter3d(x=data.UTMX.values, y=data.UTMY.values, z=data.Elevation_Cell.values,
                             customdata=data[["Layer_No", "Resistivity", "Line_No", "ID"]],
                             hovertemplate=
                             '<b>UTMX: %{x:.2f}</b><br>' +
                             '<b>UTMY: %{y:.2f}</b><br>' +
                             '<b>Elevation: %{z:.3f}</b><br>' +
                             '<b>ID: %{customdata[3]}</b><br>'
                             '<b>Layer_No: %{customdata[0]}</b><br>' +
                             '<b>Line_No: %{customdata[2]}</b><br>' +
                             '<b>Resistivity: %{customdata[1]}</b><br>',
                             mode='markers',
                             marker=dict(
                                 color=np.log10(data.Resistivity.values),
                                 cmin=0,
                                 cmax=3,
                                 showscale=True,
                                 size=8,
                                 colorscale=colorRes,
                                 colorbar=dict(
                                     ticks="outside",
                                     title="Resistivity",
                                     x=0.8,
                                     tickvals=[0, 1, 2, 3],
                                     ticktext=["1", "10", "100", "1000"],
                                     tickmode="array"
                                 ), ))
        layout = dict(scene=dict(xaxis=dict(title='UTMX'),
                                 yaxis=dict(title='UTMY'),
                                 zaxis=dict(title='Elevation (m)'),
                                 aspectmode='auto'),)
        trace = dict(data=[ttemfigdata], layout=layout)
        return trace
    elif np.isin("well", args):
        trace = go.Scatter3d(x=data["UTMX"], y=data["UTMY"], z=data["Elevation"], mode='markers',
                             text=data["Bore"],
                             hovertemplate=
                             '<b>WIN: %{text}</b><br>' +
                             '<b>Elevation: %{z:.2f}</b><br>',
                             marker=dict(
                                 color=data["Keyword_n"],
                                 cmin=1,
                                 cmax=3,
                                 showscale=True,
                                 size=10,
                                 colorscale=colorwell,
                                 colorbar=dict(
                                     ticks="outside",
                                     title="Lithology",
                                     x=0.1,
                                     tickvals=[1, 2, 3],
                                     ticktext=["Fine grain", "mix grain", "coarse grain"],
                                     tickmode="array"
                                 ),
                             ))
        return trace
    elif np.isin("rock",args):
        trace = go.Scatter3d(x=data.UTMX.values, y=data.UTMY.values, z=data.Elevation_Cell.values,
                             customdata=data[["Layer_No", "Resistivity", "Line_No","Identity","ID"]],
                             hovertemplate=
                             '<b>UTMX: %{x:.2f}</b><br>' +
                             '<b>UTMY: %{y:.2f}</b><br>' +
                             '<b>Elevation: %{z:.3f}</b><br>' +
                             '<b>ID: %{customdata[4]}</b><br>' +
                             '<b>Layer_No: %{customdata[0]}</b><br>' +
                             '<b>Line_No: %{customdata[2]}</b><br>' +
                             '<b>Resistivity: %{customdata[1]}</b><br>'+
                             '<b>Rocktype: %{customdata[3]}</b><br>',
                             mode="markers",
                             marker=dict(
                                 color=data.Identity_n.values,
                                 cmin=0,
                                 cmax=2,
                                 showscale=True,
                                 size=8,
                                 colorscale=colorrock,
                                 colorbar=dict(
                                     ticks="outside",
                                     title="Rock",
                                     x=0.8,
                                     tickvals=[0, 1, 2],
                                     ticktext=["Finegrain","Mixgrain","Coarsegrain"],
                                     tickmode="array"
                                 ),))
        return trace
    elif np.isin("geophysics", args):
        trace = data.plot(x=args, y="Depth")
        return trace
    elif np.isin('water', args):
        trace = go.Scatter3d(x=data['UTMX'].values,
                             y=data['UTMY'].values,
                             z=data['water_elevation'],
                             customdata=data[["wellname"]],
                             hovertemplate=
                             '<b>UTMX: %{x:.2f}</b><br>' +
                             '<b>UTMY: %{y:.2f}</b><br>' +
                             '<b>Water elevation: %{z:.3f}</b><br>' +
                             '<b>WellNo: %{customdata[0]}</b><br>'
                             #'<b>Well_elevation: %{customdata[1]}</b><br>'
                             )
        return trace
    else:
        raise TypeError('{} is not one of "ttem","well","rock",or,"geophysics"'.format(args))
def trendline(x,y):
    def length(n):
        import math
        if n > 0:
            digits = int(math.log10(n)) + 1
        elif n == 0:
            digits = 1
        else:
            digits = int(math.log10(-n)) + 2
        return digits
    trend_result = linregress(x, y) # result include slope, intercept, rvalue, pvalue, stderr, intercept_stderr
    x_limit = math.ceil(x.max()/10**(length(x.max())-1))*10**(length(x.max())-1)
    x_range = np.arange(x_limit)
    y_range = x_range*trend_result[0] + trend_result[1]
    slope = trend_result[0]; intercept = trend_result[1]; rvalue = trend_result[2]
    pvalue = trend_result[3];stderr = trend_result[4];intercept_stderr = trend_result.intercept_stderr
    rsquare = trend_result[2]**2
    df = pd.DataFrame({"x_range":x_range, "y_range":y_range})
    df[['slope','intercept','rvalue','pvalue','stderr','intercept_stderr','rsquare']] =[
        slope, intercept, rvalue, pvalue, stderr, intercept_stderr, rsquare
    ]
    trace = go.Scatter(x=x_range, y=y_range,
                       mode='lines',
                       name='trendline',
                       customdata=df[['slope','intercept','rvalue','pvalue','stderr','intercept_stderr','rsquare']],
                       hovertemplate=
                       '<b>y= %{customdata[0]:.3f}x+%{customdata[1]:.3f}</b><br>' +
                       '<b>r_value=: %{customdata[2]:.3f}</b><br>' +
                       '<b>p_value=: %{customdata[3]:.3f}</b><br>' +
                       '<b>r_square=: %{customdata[6]:.3f}</b><br>' +
                       '<b>stderr=: %{customdata[4]:.3f}</b><br>' +
                       '<b>intercept_stderr=: %{customdata[5]:.3f}</b><br>'
                       )
    return trace
def geophy_ttem_plot(df,**kwargs):
    if np.isin(["x","y"],list(kwargs.keys())).all():
        fig = df.plot.scatter(x=df[kwargs["x"]],
                              y=df[kwargs["y"]])
        trend_trace = trendline(x=df[kwargs["x"]],
                                y=df[kwargs["y"]])
        fig.add_trace(trend_trace)
        fig.update_layout(title=df.loc[0, "comment"])
        return fig
    else:
        print("Select two columns you want to plot scatter plot")

