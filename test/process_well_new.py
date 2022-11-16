import pandas as pd
from pyproj import Transformer
from progress.bar import Bar
def fill(group, factor=100):
    newgroup = group.loc[group.index.repeat(group.Thickness * factor)]
    mul_per_gr = newgroup.groupby('Elevation').cumcount()
    newgroup['Elevation'] = newgroup['Elevation'].subtract(mul_per_gr * 1 / factor)
    newgroup['Depth1_m'] = newgroup['Depth1_m'].subtract(mul_per_gr * 1 / factor)
    newgroup['Depth2_m'] = newgroup['Depth1_m'].add(1/factor)
    newgroup['Elevation_End'] = newgroup['Elevation'].subtract(1 / factor)
    newgroup['Thickness'] = 1 / factor
    return newgroup
def format_well(welllog,factor=1000):
    transformer = Transformer.from_crs('epsg:4326', 'epsg:32612')
    lithology = pd.read_excel(welllog, sheet_name='Lithology').drop_duplicates()
    lithology['Thickness'] = lithology['Depth2_m'].subtract(lithology['Depth1_m'])
    location = pd.read_excel(welllog, sheet_name='Location').drop_duplicates()
    lithology_group = lithology.groupby('Bore')
    concatlist = []
    bar = Bar('Processing', max=len(list(lithology_group.groups.keys())))
    bar.check_tty = False
    for name, group in lithology_group:
        group_location = location[location['Bore'] == name]
        if group_location.empty:
            group_location = location[location['Bore'] == name.strip()]
            if group_location.empty:
                continue
        long = group_location['X'].iloc[0]
        lat = group_location['Y'].iloc[0]
        x, y = transformer.transform(lat, long)
        group['Elevation'] = group_location['Z'].iloc[0]
        group['Elevation'] = group['Elevation'].subtract(group['Depth1_m'])
        group['UTMX'] = x
        group['UTMY'] = y
        group['Thickness'] = group['Depth2_m'].subtract(group['Depth1_m'])
        newgroup = fill(group, factor)
        concatlist.append(newgroup)
        bar.next()
    upscalled_well = pd.concat(concatlist)
    upscalled_well.reset_index(drop=True, inplace=True)
    bar.finish()
    return upscalled_well

if __name__ == '__main__':
    print('Demo data')
    welllog = r'C:\\Users\\jldz9\\OneDrive - University of Missouri\\MST\\Code\\Python\\tTEM_test\\Plot_with_well_log\\Well_log.xlsx'
    upwell = format_well(welllog)







