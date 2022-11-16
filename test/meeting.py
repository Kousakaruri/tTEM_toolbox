import tTEM_tool as tt
import pandas as pd
import glob
import re
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy.stats import linregress
pd.options.plotting.backend = "plotly"
pio.renderers.default = "browser"
#####
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
#point_exclude = pd.read_csv(r'B:\Code\Python\tTEM_test\Plot_with_well_log\ignore_point.csv')
ttem = tt.main.ProcessTTEM(ttemname=ttemname,
                           welllog=welllog,
                           DOI=DOI,
                           layer_exclude=[1, 2])
                           #point_exclude=point_exclude
ttem1_output = ttem.process()
rk_trans1 = ttem.ttem_well_connect(ttem1_output)
grainsum = rk_trans1.groupby(['UTMX','UTMY','Identity']).sum()
grainsum.reset_index(inplace=True)
thicksum = rk_trans1.groupby(['UTMX','UTMY']).agg({'Thickness': 'sum'})
thicksum.rename(columns={'Thickness': 'T_sum'}, inplace=True)
thicksum.reset_index(inplace=True)
ratio = pd.merge(grainsum,thicksum,on=['UTMX', 'UTMY'])
ratio['ratio'] = round(ratio['Thickness'].div(ratio['T_sum']),2)
ratio_fine = ratio[ratio['Identity']=='Fine_grain']
ratio_fine.to_csv('ratio_fine.csv')
ratio_mix = ratio[ratio['Identity']=='Mix_grain']
ratio_mix.to_csv('ratio_mix.csv')
ratio_coarse = ratio[ratio['Identity']=='Coarse_grain']
ratio_coarse.to_csv('ratio_coarse.csv')
ttem2 = tt.main.ProcessTTEM(ttemname=ttemname2,
                           welllog=welllog,
                           DOI=DOI,
                           layer_exclude=[1, 2])
ttem2_output = ttem2.process()
rk_trans2 = ttem2.ttem_well_connect(ttem2_output)
grainsum2 = rk_trans2.groupby(['UTMX','UTMY','Identity']).sum()
grainsum2.reset_index(inplace=True)
thicksum2 = rk_trans2.groupby(['UTMX','UTMY']).agg({'Thickness': 'sum'})
thicksum2.rename(columns={'Thickness': 'T_sum'}, inplace=True)
thicksum2.reset_index(inplace=True)
ratio2 = pd.merge(grainsum2,thicksum2,on=['UTMX', 'UTMY'])
ratio2['ratio'] = round(ratio2['Thickness'].div(ratio2['T_sum']),2)
ratio_fine2 = ratio2[ratio2['Identity']=='Fine_grain']
ratio_fine2.to_csv('ratio_fine2.csv')
ratio_mix2 = ratio2[ratio2['Identity']=='Mix_grain']
ratio_mix2.to_csv('ratio_mix2.csv')
ratio_coarse2 = ratio2[ratio2['Identity']=='Coarse_grain']
ratio_coarse2.to_csv('ratio_coarse2.csv')
ttem_output = ttem1_output.append(ttem2_output)
ratiooutput = ratio.append(ratio2)
ratiooutput.to_csv(r'B:\Code\Python\tTEM_test\Plot_with_well_log\ratio_cleaned.csv')
#welllog_upscale, ttem_DOI, pre_bootstrap,ttem_rk_trans, Resi_conf_df = ttem.process()
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
    result2 = gr.gr_ttem_combine(output, ttem=ttem_output)
    gammavswell = gammavswell.append(result)
    gammavsttem = gammavsttem.append(result2)
gammavswell.groupby("Keyword")["GRM"].mean()
##gamma vs well
##adams farm
adamsup = gammavswell[gammavswell["comment"] == "adams up"]
adamsdown = gammavswell[gammavswell["comment"] == "adams down"]
adams = adamsup.append(adamsdown)
adams_compare = adams.groupby("Keyword")["GRM"].mean()
##bradshaw farm
bradshawup = gammavswell[gammavswell["comment"] == "bradshaw farms up"]
bradshawdown = gammavswell[gammavswell["comment"] == "bradshaw farms down"]
bradshaw = bradshawup.append(bradshawdown)
bradshaw_compare = bradshaw.groupby("Keyword")["GRM"].mean()
##Stubbs farm
stubbsup = gammavswell[gammavswell["comment"] == "stubbs up"]
stubbsdown = gammavswell[gammavswell["comment"] == "stubbs down"]
stubbs = stubbsup.append(stubbsdown)
stubbs_compare = stubbs.groupby("Keyword")["GRM"].mean()
figstubbsup = px.bar(stubbsup, x='GRM', y='Elevation1',color='Keyword', text='Keyword',
             orientation='h', title=stubbsup.comment[0],
             color_discrete_map={
                "fine grain": "blue",
                "mix grain": "yellow",
                "coarse grain": "red"
                },)
figadamsup = px.bar(adamsup, x='GRM', y='Elevation1',color='Keyword', text='Keyword',
             orientation='h',title=adamsup.comment[0],
             color_discrete_map={
                "fine grain": "blue",
                "mix grain": "yellow",
                "coarse grain": "red"
                },)
figbradshawup = px.bar(bradshawup, x='GRM', y='Elevation1',color='Keyword', text='Keyword',
             orientation='h',title=bradshawup.comment[0],
             color_discrete_map={
                "fine grain": "blue",
                "mix grain": "yellow",
                "coarse grain": "red"
                },)
##### dont quite make scence


### gamma vs ttem
##adams up
adamsupttem = gammavsttem[gammavsttem["comment"] == "adams up"]
figadmup = tt.plot.geophy_ttem_plot(adamsupttem ,x="GRM", y="Resistivity")
#figadmup.show()
##adams down
adamsdownttem = gammavsttem[gammavsttem["comment"] == "adams down"]
figadmdown = tt.plot.geophy_ttem_plot(adamsdownttem, x="GRM", y="Resistivity")
#figadmdown.show()
##bradshaw farms down
bradshawdownttem = gammavsttem[gammavsttem["comment"] == "bradshaw farms down"]
figbrdown = tt.plot.geophy_ttem_plot(bradshawdownttem, x="GRM", y="Resistivity")
#figbrdown.show()
##halterman down fix
halterdownttem = gammavsttem[gammavsttem["comment"] == "halterman down fix"]
fighaldown = tt.plot.geophy_ttem_plot(halterdownttem, x="GRM", y="Resistivity")
#fighaldown.show()

## stubb farm
stubbttem = gammavsttem[gammavsttem["comment"] == "stubbs down"]
fighstubb = tt.plot.geophy_ttem_plot(stubbttem, x="GRM", y="Resistivity")
fighstubb.show()

### make more scence!
##new meeting content



#list(ds.merge(ds2).sel(time='2016',header='sl_lev_va').keys()) #get all variable name
#tmp = list(ds3.sel(time='2020',header='sl_lev_va').to_array().values.reshape(-1))
#elevation = [x for x in tmp if not pd.isna(x)]