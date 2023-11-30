# constants.py
# Version: 11.17.2023
# Created: 2023-11-17
# Author: Jiawei Li

XYZ_FILE_PATTERN = 'ID'
DOI_FILE_PATTERN = 'UTMX'
CSV_EXTENSION = ('.csv',)
EXCEL_EXTENSION = ('.xlsx', '.xls', '.xlsm')
LITHOLOGY_SHEET_NAMES = ('Lithology', 'lithology', 'LITHOLOGY',
                         'litho', 'Litho', 'LITHO')
LITHOLOGY_COLUMN_NAMES_KEYWORD = ('Lithology', 'lithology', 'LITHOLOGY',
                          'litho', 'Litho', 'LITHO',
                          'KEYWORD', 'Keyword', 'keyword')
LITHOLOGY_COLUMN_NAMES_BORE = ('Bore', 'bore', 'BORE',)
LITHOLOGY_COLUMN_NAMES_DEPTH_TOP = ('Depth1', 'depth1', 'DEPTH1',
                       'Depth_1', 'depth_1', 'DEPTH_1',
                       'Depthtop', 'depthtop', 'DEPTHTOP',
                       'Depth_top', 'depth_top', 'DEPTH_TOP')
LITHOLOGY_COLUMN_NAMES_DEPTH_BOTTOM = ('Depth2', 'depth2', 'DEPTH2',
                          'Depth_2', 'depth_2', 'DEPTH_2',
                          'Depthbottom', 'depthbottom', 'DEPTHBOTTOM',
                          'Depth_bottom', 'depth_bottom', 'DEPTH_BOTTOM')
LOCATION_SHEET_NAMES = ('Location', 'location', 'LOCATION',
                        'coordinates', 'Coordinates', 'COORDINATES')
LOCATION_COLUMN_NAMES_UTMX = ('UTMX', 'utmx', 'Utmx', 'UTMx', 'utmX', 'UTMX')
LOCATION_COLUMN_NAMES_UTMY = ('UTMY', 'utmy', 'Utmy', 'UTMy', 'utmY', 'UTMY')
LOCATION_COLUMN_NAMES_LAT = ('LAT', 'lat', 'Lat',
                             'LATITUDE', 'Latitude', 'latitude',
                             'Y', 'y')
LOCATION_COLUMN_NAMES_LON = ('LON', 'lon', 'Lon',
                             'LONGITUDE', 'Longitude', 'longitude',
                             'X', 'x')