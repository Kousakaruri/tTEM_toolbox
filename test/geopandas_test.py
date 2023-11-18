import numpy as np
import geopandas
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from rasterio.plot import show
path = Path(r'C:\Users\jldz9\OneDrive - University of Missouri\MST\2022\2022SS\ArcGIS\ttem_specs\shapefile\Usgs water well')
df = pd.read_csv(path.joinpath('ParowanTable6.csv'))
gdf = gpd.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.long, df.lat))
gdf = gdf.set_crs(epsg=4326)
gdf = gdf.to_crs(epsg='32612')

xmin = gdf.geometry.x.min()
xmax = gdf.geometry.x.max()
ymin = gdf.geometry.y.min()
ymax = gdf.geometry.y.max()

x = np.arange(xmin,xmax,10000)
y = np.arange(ymin,ymax,10000)
gx,gy= np.meshgrid(x,y)
gz = np.full((y.shape[0], x.shape[0]),np.nan)

for
####write raster
from rasterio.transform import Affine
res = (xmax-xmin)/500
transform = Affine.translation(xmin-res/2, ymin-res/2)*Affine.scale(res,-res)

with rasterio.open(
    path.joinpath('demoraster.tif'),
    'w',
    driver='GTiff',
    height=500,
    width=500,
    count=1,
    dtype=gdf['Residue mg/L'].dtype,
    crs=rasterio.CRS.from_epsg(32612),
    transform=transform,

) as dst:
    dst.write(gz, 1)