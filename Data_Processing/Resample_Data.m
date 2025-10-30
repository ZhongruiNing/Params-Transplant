%% BFI
clear;clc;

BFI1 = ncread("F:\GSCD\GSCD.nc", "BFI1");
BFI2 = ncread("F:\GSCD\GSCD.nc", "BFI2");
BFI3 = ncread("F:\GSCD\GSCD.nc", "BFI3");
BFI4 = ncread("F:\GSCD\GSCD.nc", "BFI4");
temp_BFI = nan(3600, 7200, 4);
temp_BFI(:, :, 1) = BFI1;
temp_BFI(:, :, 2) = BFI2;
temp_BFI(:, :, 3) = BFI3;
temp_BFI(:, :, 4) = BFI4;
BFI = single(mean(temp_BFI, 3));

lon = ncread("F:\GSCD\GSCD.nc", "longitude");
lat = ncread("F:\GSCD\GSCD.nc", "latitude");
[lon1, lat1] = meshgrid(lon, lat);
lon = single(-179.975 : 0.15 : 179.975)';
lat = single(89.975 : -0.15 : -89.975)';
[lon2, lat2] = meshgrid(lon, lat);

new_BFI = interp2(lon1, lat1, BFI, lon2, lat2, "nearest");

write_nc('../../Raw_Data/BFI.nc', lon, lat, new_BFI, 'BFI', 'BFI')

%% Climate Zone
clear;clc;

Climate_Koppen = readgeoraster('E:/2024_01_Global_WBM_Validation/Raw_Data/Underlying/Koppen_resample.tif');
SpatialRef = geotiffinfo('E:/2024_01_Global_WBM_Validation/Raw_Data/Underlying/Koppen_resample.tif');

left_bound  = SpatialRef.BoundingBox(1, 1);
right_bound = SpatialRef.BoundingBox(2, 1);
top_bound   = SpatialRef.BoundingBox(2, 2);
botom_bound = SpatialRef.BoundingBox(1, 2);
Res = SpatialRef.PixelScale(1);
lon = left_bound + 0.5 * Res : Res : right_bound - 0.5 * Res;
lat = top_bound - 0.5 * Res : -Res : botom_bound + 0.5 * Res;
[lon1, lat1] = meshgrid(lon, lat);
lon = single(left_bound + 0.5 * Res : 0.15 : right_bound - 0.5 * Res)';
lat = single(top_bound - 0.5 * Res : -0.15 : botom_bound + 0.5 * Res)';
[lon2, lat2] = meshgrid(lon, lat);

New_Climate = interp2(lon1, lat1, Climate_Koppen, lon2, lat2, "nearest");

write_nc('../../Raw_Data/Climate.nc', lon, lat, New_Climate, 'Climate', 'Koppen Climate Classification')

clear Climate_Koppen SpatialRef left_bound right_bound top_bound botom_bound Res lon lat lon1 lat1 lon2 lat2 New_Climate

%% Soil Texture
clear;clc;

Clay_HWSD = readgeoraster('F:/Global_Soil/HWSD2/HWSD2_RASTER/CLAY_ratio.tif');
Sand_HWSD = readgeoraster('F:/Global_Soil/HWSD2/HWSD2_RASTER/SAND_ratio.tif');
Silt_HWSD = readgeoraster('F:/Global_Soil/HWSD2/HWSD2_RASTER/SILT_ratio.tif');
SpatialRef = geotiffinfo('F:/Global_Soil/HWSD2/HWSD2_RASTER/CLAY_ratio.tif');

left_bound  = SpatialRef.BoundingBox(1, 1);
right_bound = SpatialRef.BoundingBox(2, 1);
top_bound   = SpatialRef.BoundingBox(2, 2);
botom_bound = SpatialRef.BoundingBox(1, 2);
Res = SpatialRef.PixelScale(1);
lon = left_bound + 0.5 * Res : Res : right_bound - 0.5 * Res;
lat = top_bound - 0.5 * Res : -Res : botom_bound + 0.5 * Res;
[lon1, lat1] = meshgrid(lon, lat);
lon = single(left_bound + 0.5 * Res : 0.15 : right_bound - 0.5 * Res)';
lat = single(top_bound - 0.5 * Res : -0.15 : botom_bound + 0.5 * Res)';
[lon2, lat2] = meshgrid(lon, lat);

New_Clay = interp2(lon1, lat1, Clay_HWSD, lon2, lat2, "cubic");
New_Sand = interp2(lon1, lat1, Sand_HWSD, lon2, lat2, "cubic");
New_Silt = interp2(lon1, lat1, Silt_HWSD, lon2, lat2, "cubic");

write_nc('../../Raw_Data/Clay.nc', lon, lat, New_Clay, 'Clay', 'Clay')
write_nc('../../Raw_Data/Sand.nc', lon, lat, New_Sand, 'Sand', 'Sand')
write_nc('../../Raw_Data/Silt.nc', lon, lat, New_Silt, 'Silt', 'Silt')

%% Slope
clear;clc;

Slope = readgeoraster("../../Raw_Data/Slope.tif");
SpatialRef = geotiffinfo("../../Raw_Data/Slope.tif");
Slope(Slope > 100000) = nan;

left_bound  = SpatialRef.BoundingBox(1, 1);
right_bound = SpatialRef.BoundingBox(2, 1);
top_bound   = SpatialRef.BoundingBox(2, 2);
botom_bound = SpatialRef.BoundingBox(1, 2);
Res = SpatialRef.PixelScale(1);
lon = left_bound + 0.5 * Res : Res : right_bound - 0.5 * Res;
lat = top_bound - 0.5 * Res : -Res : botom_bound + 0.5 * Res;
[lon1, lat1] = meshgrid(lon, lat);
lon = single(left_bound + 0.5 * Res : 0.15 : right_bound - 0.5 * Res)';
lat = single(top_bound - 0.5 * Res : -0.15 : botom_bound + 0.5 * Res)';
[lon2, lat2] = meshgrid(lon, lat);

new_Slope = interp2(lon1, lat1, Slope, lon2, lat2, "cubic");

write_nc('../../Raw_Data/Slope.nc', lon, lat, double(new_Slope), 'Slope', 'Slope')

%% TI
clear;clc;

TI = readgeoraster("../../../2024_01_Global_WBM_Validation/Raw_Data/Underlying/TI.tif");
SpatialRef = geotiffinfo("../../../2024_01_Global_WBM_Validation/Raw_Data/Underlying/TI.tif");
TI(TI < -10000) = nan;

left_bound  = SpatialRef.BoundingBox(1, 1);
right_bound = SpatialRef.BoundingBox(2, 1);
top_bound   = SpatialRef.BoundingBox(2, 2);
botom_bound = SpatialRef.BoundingBox(1, 2);
Res = SpatialRef.PixelScale(1);
lon = left_bound + 0.5 * Res : Res : right_bound - 0.5 * Res;
lat = top_bound - 0.5 * Res : -Res : botom_bound + 0.5 * Res;
[lon1, lat1] = meshgrid(lon, lat);
lon = single(left_bound + 0.5 * Res : 0.15 : right_bound - 0.5 * Res)';
lat = single(top_bound - 0.5 * Res : -0.15 : botom_bound + 0.5 * Res)';
[lon2, lat2] = meshgrid(lon, lat);

new_TI = interp2(lon1, lat1, TI, lon2, lat2, "cubic");

write_nc('../../Raw_Data/TI.nc', lon, lat, double(new_TI), 'TI', 'Terrian Index')