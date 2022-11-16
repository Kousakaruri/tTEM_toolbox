#TODO generate a drawdown map base on the water well map
import tTEM_tool as tt
import os

os.chdir(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah')
well_info = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah\USGSdownload\NWISMapperExport.xlsx'
location = r"C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\Gamma\location.csv"
welllog = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\Well_log.xlsx'
elevation = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah\usgs_water_elevation.csv'
ttemname = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\PD1_I01_MOD.xyz'
ttemname2 = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\PD22_I03_MOD.xyz'
DOI = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\DOID1_DOIStaE.xyz'
water = tt.main.GWSurface(waterwell=well_info, elevation_type='NAVD88')
water_format = water.format(time=['2015-3','2016-3','2017-3','2018-3','2019-3','2020-3','2021-3'])
name = 'sl_lev_va'
year =2015

for i in range(6):
    water_format['drawdown'+str(year)+str(year+1)] = water_format[name+str(year)+'-3'].subtract(water_format[name+str(year+1)+'-3'])
    year +=1