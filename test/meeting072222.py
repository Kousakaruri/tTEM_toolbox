import tTEM_tool as tt
import pandas as pd
import glob
import re
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy.stats import linregress
gamma_file_path = r"B:\\Code\\Python\\Gamma\\Gamma_data"
gamma_file = glob.glob(gamma_file_path + '\*.csv')
exclude_keyword = re.compile(r'fluid')
filt_gamma_file = list(filter(lambda x: not exclude_keyword.search(x), gamma_file))
location = r"B:\\Code\\Python\\Gamma\\location.csv"
welllog = r'B:\Code\Python\tTEM_test\Plot_with_well_log\Well_log.xlsx'
ttemname = r'B:\Code\Python\tTEM_test\Plot_with_well_log\PD22_I03_MOD.xyz'
ttemname2 = r'B:\Code\Python\tTEM_test\Plot_with_well_log\PD1_I01_MOD.xyz'
DOI = r'B:\Code\Python\tTEM_test\Plot_with_well_log\DOID1_DOIStaE.xyz'
def plot_bst(dataframe):
    fig_hist = go.Figure()
    fig_hist.data = []
    fig_hist.add_trace(go.Histogram(x=dataframe.fine, name='Fine', marker_color='Blue', opacity=0.75))
    fig_hist.add_trace(go.Histogram(x=dataframe.coarse, name='Coarse', marker_color='Red', opacity=0.75))
    if dataframe.mix.sum() == 0:
        print("skip plot mix because there is no data")
    else:
        fig_hist.add_trace(go.Histogram(x=dataframe.mix, name='Mix', marker_color='Yellow', opacity=0.75))
    return fig_hist.show()
####
#1 process N and center Parowan seperately
#1.1 With manually selected wells
##center Parowan
welllog = r'B:\Code\Python\tTEM_test\Plot_with_well_log\tTEM.xlsx'
ttem = tt.main.ProcessTTEM(ttemname=ttemname,
                              welllog=welllog,
                              DOI=DOI,
                              layer_exclude=[1, 2])
pre_bootstrap,rk_trans, pack_bootstrap_result, Resi_conf_df = ttem.ttem_well_connect()
ratio = tt.Rock_trans.pct_count(rk_trans)
result_ct = {'pre_bootstrap':pre_bootstrap,
             'rk_trans':rk_trans,
             'pack_bootstrap_result':pack_bootstrap_result,
             'resi_conf_df':Resi_conf_df,
             'ratio':ratio}
##Nothern Parowan
ttem = tt.main.ProcessTTEM(ttemname=ttemname2,
                              welllog=welllog,
                              DOI=DOI,
                              layer_exclude=[1, 2])
pre_bootstrap,rk_trans, pack_bootstrap_result, Resi_conf_df = ttem.ttem_well_connect()
ratio = tt.Rock_trans.pct_count(rk_trans)
result_n = {'pre_bootstrap':pre_bootstrap,
            'rk_trans':rk_trans,
            'pack_bootstrap_result':pack_bootstrap_result,
            'resi_conf_df':Resi_conf_df,
            'ratio':ratio}
result1_1 = {'result_ct':result_ct,'result_n':result_n}
#1.2 With program selected wells (500m radius)
##center parowan
welllog = r'B:\Code\Python\tTEM_test\Plot_with_well_log\Well_log.xlsx'
ttem = tt.main.ProcessTTEM(ttemname=ttemname,
                              welllog=welllog,
                              DOI=DOI,
                              layer_exclude=[1, 2])
pre_bootstrap,rk_trans, pack_bootstrap_result, Resi_conf_df = ttem.ttem_well_connect()
ratio = tt.Rock_trans.pct_count(rk_trans)
result_ct = {'pre_bootstrap':pre_bootstrap,
            'rk_trans':rk_trans,
            'pack_bootstrap_result':pack_bootstrap_result,
            'resi_conf_df':Resi_conf_df,
            'ratio':ratio}
ttem = tt.main.ProcessTTEM(ttemname=ttemname2,
                              welllog=welllog,
                              DOI=DOI,
                              layer_exclude=[1, 2])
pre_bootstrap,rk_trans, pack_bootstrap_result, Resi_conf_df = ttem.ttem_well_connect()
ratio = tt.Rock_trans.pct_count(rk_trans)
result_n = {'pre_bootstrap':pre_bootstrap,
            'rk_trans':rk_trans,
            'pack_bootstrap_result':pack_bootstrap_result,
            'resi_conf_df':Resi_conf_df,
            'ratio':ratio}
result1_2 = {'result_ct':result_ct,'result_n':result_n}
##2.process N and center Parowan together

#2.1 with manually selected wells
welllog = r'B:\Code\Python\tTEM_test\Plot_with_well_log\tTEM.xlsx'
ttem = tt.main.ProcessTTEM(ttemname=[ttemname,ttemname2],
                              welllog=welllog,
                              DOI=DOI,
                              layer_exclude=[1, 2])
pre_bootstrap,rk_trans, pack_bootstrap_result, Resi_conf_df = ttem.ttem_well_connect()
ratio = tt.Rock_trans.pct_count(rk_trans)
result2_1 = {'pre_bootstrap':pre_bootstrap,
            'rk_trans':rk_trans,
            'pack_bootstrap_result':pack_bootstrap_result,
            'resi_conf_df':Resi_conf_df,
            'ratio':ratio}
#2.2 With program selected wells (500m radius)
welllog = r'B:\Code\Python\tTEM_test\Plot_with_well_log\Well_log.xlsx'
ttem = tt.main.ProcessTTEM(ttemname=[ttemname,ttemname2],
                              welllog=welllog,
                              DOI=DOI,
                              layer_exclude=[1, 2])
pre_bootstrap,rk_trans, pack_bootstrap_result, Resi_conf_df = ttem.ttem_well_connect()
ratio = tt.Rock_trans.pct_count(rk_trans)
result2_2 = {'pre_bootstrap':pre_bootstrap,
            'rk_trans':rk_trans,
            'pack_bootstrap_result':pack_bootstrap_result,
            'resi_conf_df':Resi_conf_df,
            'ratio':ratio}
del result_n, result_ct, ratio, pre_bootstrap, rk_trans, pack_bootstrap_result, Resi_conf_df, ttem

###gamma_result
gammavswell = pd.DataFrame()
gammavsttem = pd.DataFrame()
gamma_processed = pd.DataFrame()
for i in filt_gamma_file:
    print(i)
    gr = tt.main.ProcessGamma(gamma=i,
                              gammaloc=location,
                              welllog=welllog,
                              rolling=True,
                              window=7,
                              columns="GR",
                              )
    output = gr.process()
    gamma_processed = gamma_processed.append(output)
    result = gr.gr_wlg_combine(output)
    result2 = gr.gr_ttem_combine(output, ttem=result2_2['rk_trans'])
    gammavswell = gammavswell.append(result)
    gammavsttem = gammavsttem.append(result2)

import xarray as xr
ds = tt.process_well.format_usgs_water(tt.process_well.dl_usgs_water('375033112561101'))