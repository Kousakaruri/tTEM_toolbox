#TODO Excute the bootstrap result without selecting well logs
import tTEM_tool as tt
import os

os.chdir(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah')
well_info = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah\USGSdownload\NWISMapperExport.xls'
location = r"C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\Gamma\location.csv"
welllog = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\Well_log.xlsx'
elevation = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\well_Utah\usgs_water_elevation.csv'
ttemname = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\PD1_I01_MOD.xyz'
ttemname2 = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\PD22_I03_MOD.xyz'
DOI = r'C:\Users\jldz9\OneDrive - University of Missouri\MST\Code\Python\tTEM_test\Plot_with_well_log\DOID1_DOIStaE.xyz'
time = '2022-3'
water = tt.main.GWSurface(waterwell=well_info, elevation_type='depth')
water_format = water.format(elevation=elevation, time=time)


