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
####
#1 process N and center seperately and export the ratio
ttem = tt.main.ProcessTTEM(ttem_path=[ttemname2],
                                   welllog=welllog,
                                   DOI_path=DOI,
                                   layer_exclude=[],
                                   line_exclude=[190,180,170,160,150,140,130,120,0]
                           )
data = ttem.data()
fig = go.Figure(tt.plot.generate_trace(data,'ttem'))
fig.show(renderer='browser')