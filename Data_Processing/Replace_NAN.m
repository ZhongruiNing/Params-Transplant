clear;clc;
%% Slope
slope = ncread("../../Data/Grids_Prop/Slope.nc", "Slope");
lon_slope = ncread("../../Data/Grids_Prop/Slope.nc", "longitude");
lat_slope = ncread("../../Data/Grids_Prop/Slope.nc", "latitude");
[lon1, lat1] = meshgrid(lon_slope, lat_slope);
[lon2, lat2] = meshgrid(lon_slope(1001:1004), lat_slope(220:224));
valid = ~isnan(slope);
new_slope = griddata(lon1(valid), lat1(valid), slope(valid), lon2, lat2, 'linear');
slope(220:224, 1001:1004) = new_slope;
write_nc("../../Data/Grids_Prop/Slope.nc", lon_slope, lat_slope, slope, 'Slope', 'Slope of grid center')

%% BFI
BFI = ncread("../../Data/Grids_Prop/BFI.nc", "BFI");
lon_BFI = ncread("../../Data/Grids_Prop/BFI.nc", "longitude");
lat_BFI = ncread("../../Data/Grids_Prop/BFI.nc", "latitude");
[lon1, lat1] = meshgrid(lon_slope, lat_slope);
[lon2, lat2] = meshgrid(lon_slope(1007:1110), lat_slope(202:251));
valid = ~isnan(BFI);
new_BFI = griddata(lon1(valid), lat1(valid), BFI(valid), lon2, lat2, 'linear');
BFI(202:251, 1007:1110) = new_BFI;
write_nc("../../Data/Grids_Prop/BFI.nc", lon_BFI, lat_BFI, BFI, 'BFI', 'Baseflow index')

%% TI
TI = ncread("../../Data/Grids_Prop/TI.nc", "TI");
lon_TI = ncread("../../Data/Grids_Prop/TI.nc", "longitude");
lat_TI = ncread("../../Data/Grids_Prop/TI.nc", "latitude");
[lon1, lat1] = meshgrid(lon_TI, lat_TI);
[lon2, lat2] = meshgrid(lon_TI(1001:1004), lat_TI(220:224));
valid = ~isnan(TI);
new_TI = griddata(lon1(valid), lat1(valid), TI(valid), lon2, lat2, 'linear');
TI(220:224, 1001:1004) = new_TI;
write_nc("../../Data/Grids_Prop/TI.nc", lon_TI, lat_TI, TI, 'TI', 'Terrian Index')