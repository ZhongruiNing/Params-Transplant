clear;clc;

%% 基准格点
res = 0.5;
lon = -180 + 0.5 * res : res : 180 - 0.5 * res;
lat = 90 - 0.5 * res : -res : -90 + 0.5 * res;
[lon_base, lat_base] = meshgrid(lon, lat);
% Global land boundary
border = shaperead("../../Raw_Data/Shape/Mainland.shp");
mask = nan(size(lon_base));
for b = 1 : length(border)
    disp(b)
    for ii = 1 : length(lat)
        for jj = 1 : length(lon)
            [in, on] = inpolygon(lon(jj), lat(ii), border(b).X, border(b).Y);
            if in
                mask(ii, jj) = 1;
            end
        end
    end
end
write_nc("../../Data/Grids_Prop/mask.nc", lon, lat, mask, 'mask', 'Boundary mask of global land')
mask = ncread("../../Data/Grids_Prop/mask.nc", 'mask');
%% 中心点经纬度
lon_masked = lon_base .* mask;
lat_masked = lat_base .* mask;
write_nc("../../Data/Grids_Prop/clon.nc", lon, lat, lon_masked, 'clon', 'longitude of grid center')
write_nc("../../Data/Grids_Prop/clat.nc", lon, lat, lat_masked, 'clat', 'latitude of grid center')

clear lon_masked lat_masked

%% 气候条件
pre_cru = ncread("F:/CRU_TS/pre_1901_2022.nc", "pre");
tem_cru = ncread("F:/CRU_TS/tmp_1901_2022.nc", "tmp");
pet_cru = ncread("F:/CRU_TS/pet_1901_2022.nc", "pet");
tmx_cru = ncread("F:/CRU_TS/tmx_1901_2022.nc", "tmx");
tmn_cru = ncread("F:/CRU_TS/tmn_1901_2022.nc", "tmn");

pre_ave = rot90(mean(pre_cru, 3));
tem_ave = rot90(mean(tem_cru, 3));
pet_ave = rot90(mean(pet_cru, 3));
tmx_ave = rot90(mean(tmx_cru, 3));
tmn_ave = rot90(mean(tmn_cru, 3));

lon_cru = ncread("F:/CRU_TS/pre_1901_2022.nc", "lon");
lat_cru = ncread("F:/CRU_TS/pre_1901_2022.nc", "lat");
lat_cru = flip(lat_cru);
[lon_cru, lat_cru] = meshgrid(lon_cru, lat_cru);

pre_interp = interp2(lon_cru, lat_cru, pre_ave, lon_base, lat_base, 'linear');
tem_interp = interp2(lon_cru, lat_cru, tem_ave, lon_base, lat_base, 'linear');
pet_interp = interp2(lon_cru, lat_cru, pet_ave, lon_base, lat_base, 'linear');
tmx_interp = interp2(lon_cru, lat_cru, tmx_ave, lon_base, lat_base, 'linear');
tmn_interp = interp2(lon_cru, lat_cru, tmn_ave, lon_base, lat_base, 'linear');

write_nc("../../Data/Grids_Prop/pre.nc", lon, lat, pre_interp, 'pre', 'Annual average precipitation')
write_nc("../../Data/Grids_Prop/tem.nc", lon, lat, tem_interp, 'tem', 'Annual average temperature')
write_nc("../../Data/Grids_Prop/pet.nc", lon, lat, pet_interp, 'pet', 'Annual average potential evapotranspiration')
write_nc("../../Data/Grids_Prop/tmx.nc", lon, lat, tmx_interp, 'tmx', 'Annual average maximum temperature')
write_nc("../../Data/Grids_Prop/tmn.nc", lon, lat, tmn_interp, 'tmn', 'Annual average minimum temperature')

clear pre_cru tem_cru pet_cru tmx_cru tmn_cru pre_ave tem_ave pet_ave tmx_ave tmn_ave pre_interp tem_interp pet_interp tmx_interp tmn_interp lon_cru lat_cru

%% 实际蒸散发
ae_gleam = ncread("F:/GLEAM/v42a/E/E_1980_GLEAM_v4.2a_MO.nc", "E");
lon_gleam = ncread("F:/GLEAM/v42a/E/E_1980_GLEAM_v4.2a_MO.nc", "lon");
lat_gleam = ncread("F:/GLEAM/v42a/E/E_1980_GLEAM_v4.2a_MO.nc", "lat");
ae_ave = fliplr(rot90(nanmean(ae_gleam, 3), 3));
for y = 1981 : 2023
    disp(y)
    temp_ae_gleam = ncread(strcat("F:/GLEAM/v42a/E/E_", num2str(y), "_GLEAM_v4.2a_MO.nc"), "E");
    temp_ae_ave = fliplr(rot90(nanmean(temp_ae_gleam, 3), 3));
    ae_ave = cat(3, ae_ave, temp_ae_ave);
end

ae_ave = mean(ae_ave, 3);
[lon_gleam, lat_gleam] = meshgrid(lon_gleam, lat_gleam);
ae_interp = interp2(lon_gleam, lat_gleam, ae_ave, lon_base, lat_base, 'cubic');
ae_interp = ae_interp .* mask;

write_nc("../../Data/Grids_Prop/ae.nc", lon, lat, ae_interp, 'ae', 'Annual average actual evapotranspiration')

clear ae_ave ae_interp lon_gleam lat_gleam temp_ae_gleam temp_ae_ave ae_gleam

%% NDVI
lon_NDVI = ncread("F:/NDVI/GIMMS/GIMMS_raw_data/ndvi3g_geo_v1_1982_0106.nc4", "lon");
lat_NDVI = ncread("F:/NDVI/GIMMS/GIMMS_raw_data/ndvi3g_geo_v1_1982_0106.nc4", "lat");
temp_NDVI = ncread("F:/NDVI/GIMMS/GIMMS_raw_data/ndvi3g_geo_v1_1982_0106.nc4", "ndvi");
NDVI_ave = fliplr(rot90(mean(temp_NDVI, 3), 3));
clear temp_NDVI

for y = 1982 : 2015
    disp(y)
    temp_NDVI1 = ncread(strcat("F:/NDVI/GIMMS/GIMMS_raw_data/ndvi3g_geo_v1_", num2str(y), "_0106.nc4"), "ndvi");
    temp_NDVI2 = ncread(strcat("F:/NDVI/GIMMS/GIMMS_raw_data/ndvi3g_geo_v1_", num2str(y), "_0712.nc4"), "ndvi");
    temp_NDVI = cat(3, temp_NDVI1, temp_NDVI2);
    temp_ave = fliplr(rot90(mean(temp_NDVI, 3), 3));
    NDVI_ave = cat(3, NDVI_ave, temp_ave);
    clear temp_NDVI1 temp_NDVI2 temp_NDVI temp_ave
end
NDVI_ave = mean(NDVI_ave(:, :, 2:end), 3);

[lon_NDVI, lat_NDVI] = meshgrid(lon_NDVI, lat_NDVI);
NDVI_interp = interp2(lon_NDVI, lat_NDVI, NDVI_ave, lon_base, lat_base, 'linear');
NDVI_interp = NDVI_interp .* mask / 10000;

write_nc("../../Data/Grids_Prop/NDVI.nc", lon, lat, NDVI_interp, 'NDVI', 'Annual average NDVI')
clear NDVI_ave NDVI_interp lon_NDVI lat_NDVI temp_NDVI temp_ave

%% Soil Texture
lon_Soil = ncread("../../Raw_Data/CLAY.nc", "longitude");
lat_Soil = ncread("../../Raw_Data/CLAY.nc", "latitude");
clay_Soil = ncread("../../Raw_Data/CLAY.nc", "CLAY");
silt_Soil = ncread("../../Raw_Data/SILT.nc", "SILT");
sand_Soil = ncread("../../Raw_Data/SAND.nc", "SAND");

[lon_Soil, lat_Soil] = meshgrid(lon_Soil, lat_Soil);
clay_interp = interp2(lon_Soil, lat_Soil, clay_Soil, lon_base, lat_base, 'linear');
silt_interp = interp2(lon_Soil, lat_Soil, silt_Soil, lon_base, lat_base, 'linear');
sand_interp = interp2(lon_Soil, lat_Soil, sand_Soil, lon_base, lat_base, 'linear');

write_nc("../../Data/Grids_Prop/clay.nc", lon, lat, clay_interp, 'clay', 'propertion of clay content')
write_nc("../../Data/Grids_Prop/silt.nc", lon, lat, silt_interp, 'silt', 'propertion of silt content')
write_nc("../../Data/Grids_Prop/sand.nc", lon, lat, sand_interp, 'sand', 'propertion of sand content')
clear clay_Soil silt_Soil sand_Soil clay_interp silt_interp sand_interp lon_Soil lat_Soil

%% BFI
lon_BFI = ncread("../../Raw_Data/BFI.nc", "longitude");
lat_BFI = ncread("../../Raw_Data/BFI.nc", "latitude");
BFI = ncread("../../Raw_Data/BFI.nc", "BFI");

[lon_BFI, lat_BFI] = meshgrid(lon_BFI, lat_BFI);
BFI_interp = interp2(lon_BFI, lat_BFI, BFI, lon_base, lat_base, 'linear');

BFI_interp = BFI_interp .* mask;
write_nc("../../Data/Grids_Prop/BFI.nc", lon, lat, BFI_interp, 'BFI', 'Baseflow index')
clear BFI lon_BFI lat_BFI BFI_interp

%% Climate Region
lon_CliReg = ncread("../../Raw_Data/Climate.nc", "longitude");
lat_CliReg = ncread("../../Raw_Data/Climate.nc", "latitude");
CliReg = ncread("../../Raw_Data/Climate.nc", "Climate");

[lon_CliReg, lat_CliReg] = meshgrid(lon_CliReg, lat_CliReg);
CliReg_interp = interp2(lon_CliReg, lat_CliReg, CliReg, lon_base, lat_base, 'nearest');

CliReg_interp = CliReg_interp .* mask;
write_nc("../../Data/Grids_Prop/Climate.nc", lon, lat, CliReg_interp, 'Climate', 'Climate region')
clear CliReg lon_CliReg lat_CliReg CliReg_interp

%% Slope
lon_Slope = ncread("../../Raw_Data/Slope.nc", "longitude");
lat_Slope = ncread("../../Raw_Data/Slope.nc", "latitude");
Slope = ncread("../../Raw_Data/Slope.nc", "Slope");

[lon_Slope, lat_Slope] = meshgrid(lon_Slope, lat_Slope);
Slope_interp = interp2(lon_Slope, lat_Slope, Slope, lon_base, lat_base, 'linear');
Slope_interp = Slope_interp .* mask;

write_nc("../../Data/Grids_Prop/Slope.nc", lon, lat, Slope_interp, 'Slope', 'Slope of grid center')
clear Slope lon_Slope lat_Slope Slope_interp

%% TI
lon_TI = ncread("../../Raw_Data/TI.nc", "longitude");
lat_TI = ncread("../../Raw_Data/TI.nc", "latitude");
TI = ncread("../../Raw_Data/TI.nc", "TI");

[lon_TI, lat_TI] = meshgrid(lon_TI, lat_TI);
TI_interp = interp2(lon_TI, lat_TI, TI, lon_base, lat_base, 'linear');
TI_interp = TI_interp .* mask;

write_nc("../../Data/Grids_Prop/TI.nc", lon, lat, TI_interp, 'TI', 'Terrian Index')
clear TI lon_TI lat_TI TI_interp

%% ISIMIP2a Input
elements_list = ["pr", "tas", "pet"];
datasets_list = ["gswp3"];
long_name_list = ["Precipitation", "Temperature", "Potential Evapotranspiration"];
for e = 1 : 3
    element = elements_list(e);
    if e == 2
        scaler = 1 / 30.4 / 86400;
        bias_offset = -273.15;
    else
        scaler = 1;
        bias_offset = 0;
    end
    for d = 1 : length(datasets_list)
        dataset = datasets_list(d);
        filepath = strcat("../../../Data/ISIMIP2a/Input/", dataset, "/", element, "_", dataset, "_1971_2010_m.nc4");
        lon_ISIMIP = single(ncread(filepath, "lon"));
        lat_ISIMIP = single(ncread(filepath, "lat"));
        time_ISIMIP = single(ncread(filepath, "time"));

        ISIMIP_data = single(ncread(filepath, element));
        ISIMIP_data = rot90(flipud(ISIMIP_data), 3) * scaler + bias_offset;

        [lon_ISIMIP_grid, lat_ISIMIP_grid, time_ISIMIP_grid] = meshgrid(lon_ISIMIP, lat_ISIMIP, time_ISIMIP);
        [lon_base3_grid, lat_base3_grid, time_base3_grid]    = meshgrid(single(lon), single(lat), time_ISIMIP);
        
        ISIMIP_data_Interped = interp3(lon_ISIMIP_grid, lat_ISIMIP_grid, time_ISIMIP_grid, ISIMIP_data, lon_base3_grid, lat_base3_grid, time_base3_grid);
        write_nc3(strcat("../../Data/forcing/", element, "_", dataset, "_1971_2010.nc"), lon, lat, time_ISIMIP, ISIMIP_data_Interped, element, long_name_list(e));
    end
end
clear lon_ISIMIP lat_ISIMIP time_ISIMIP ISIMIP_data lon_ISIMIP_grid lat_ISIMIP_grid time_ISIMIP_grid lon_base3_grid lat_base3_grid time_base3_grid ISIMIP_data_Interped
%% ISIMIP2a Outputs
element_list = ["qtot", "evap"];
models_list  = ["vic", "clm40", "dbh", "h08", "lpjml", "pcr_globwb"];
long_name_list = ["Simulation Natural Runoff", "Actural Evapotranspiration"];

for e= 1 : length(element_list)
    element = element_list(e);
    if e == 1
        scaler = 1;
        bias_offset = 0;
    elseif e == 2
        scalet = 86400 * 30.4;
        bias_offset = 0;
    end

    for m = 1 : length(models_list)
        model = models_list(m);

        filepath = strcat("../../../Data/ISIMIP2a/Output/", model, "/", model, "_gswp3_", element, "_1971_2010_m.nc");

        lon_ISIMIP = single(ncread(filepath, "lon"));
        lat_ISIMIP = single(ncread(filepath, "lat"));
        time_ISIMIP = single(ncread(filepath, "time"));

        ISIMIP_data = single(ncread(filepath, element));
        ISIMIP_data = rot90(flipud(ISIMIP_data), 3) * scaler + bias_offset;

        [lon_ISIMIP_grid, lat_ISIMIP_grid, time_ISIMIP_grid] = meshgrid(lon_ISIMIP, lat_ISIMIP, time_ISIMIP);
        [lon_base3_grid, lat_base3_grid, time_base3_grid]    = meshgrid(single(lon), single(lat), time_ISIMIP);

        ISIMIP_data_Interped = interp3(lon_ISIMIP_grid, lat_ISIMIP_grid, time_ISIMIP_grid, ISIMIP_data, lon_base3_grid, lat_base3_grid, time_base3_grid);
        write_nc3(strcat("../../Data/ISIMIP2a_outputs/", model, "_", element, "_1971_2010.nc"), lon, lat, time_ISIMIP, ISIMIP_data_Interped, element, long_name_list(e));
    end
end
%% CRU
filepath = "../../../Data/CRU_TS/pet_1901_2022.nc";
lon_CRU  = single(ncread(filepath, "lon"));
lat_CRU  = single(ncread(filepath, "lat"));
time_CRU = single(ncread(filepath, "time"));

CRU_data = single(rot90(ncread(filepath, "pet"))) * 30.4;
lat_CRU  = flipud(lat_CRU);

[lon_CRU_grid,   lat_CRU_grid,   time_CRU_grid]   = meshgrid(lon_CRU, lat_CRU, time_CRU);
[lon_base3_grid, lat_base3_grid, time_base3_grid] = meshgrid(single(lon), single(lat), single(time_CRU));

CRU_data_Interped = interp3(lon_CRU_grid, lat_CRU_grid, time_CRU_grid, CRU_data, lon_base3_grid, lat_base3_grid, time_base3_grid);
write_nc3(strcat("../../Data/forcing/pet_cru_1901_2022.nc"), lon, lat, time_CRU, CRU_data_Interped, "pet", "Potential Evapotranspiration");

filepath = "../../../Data/CRU_TS/pre_1901_2022.nc";
CRU_data = single(rot90(ncread(filepath, "pre")));
CRU_data_Interped = interp3(lon_CRU_grid, lat_CRU_grid, time_CRU_grid, CRU_data, lon_base3_grid, lat_base3_grid, time_base3_grid);
write_nc3(strcat("../../Data/forcing/pr_cru_1901_2022.nc"), lon, lat, time_CRU, CRU_data_Interped, "pr", "Precipitation");

filepath = "../../../Data/CRU_TS/tmp_1901_2022.nc";
CRU_data = single(rot90(ncread(filepath, "tmp")));
CRU_data_Interped = interp3(lon_CRU_grid, lat_CRU_grid, time_CRU_grid, CRU_data, lon_base3_grid, lat_base3_grid, time_base3_grid);
write_nc3(strcat("../../Data/forcing/tas_cru_1901_2022.nc"), lon, lat, time_CRU, CRU_data_Interped, "tas", "Temperature");

clear CRU_data CRU_data_Interped lon_CRU_grid lat_CRU_grid time_CRU_grid lon_base3_grid lat_base3_grid time_base3_grid

%% GRUN
filepath = "D:/Onedrive/Data/GRUN/GRUN.nc";

lon_GRUN  = single(ncread(filepath, "X"));
lat_GRUN  = single(ncread(filepath, "Y"));
time_GRUN = single(ncread(filepath, "time"));

GRUN_data = single(rot90(ncread(filepath, "Runoff") * 30.4, 1));
lat_GRUN = flip(lat_GRUN);

[lon_GRUN_grid, lat_GRUN_grid, time_GRUN_grid] = meshgrid(lon_GRUN, lat_GRUN, time_GRUN);
[lon_base3_grid, lat_base3_grid, time_base3_grid] = meshgrid(single(lon), single(lat), single(time_GRUN));

GRUN_data_interped = interp3(lon_GRUN_grid, lat_GRUN_grid, time_GRUN_grid, GRUN_data, lon_base3_grid, lat_base3_grid, time_base3_grid);

write_nc3(strcat("../../Data/GRUN/GRUN_1901_2014.nc"), lon, lat, time_GRUN, GRUN_data_interped, "Runoff", "Natural runoff by GRUN dataset");

%% GLEAM
for year = 1980 : 2023
    filepath = strcat("F:/GLEAM/v42a/E/E_", num2str(year), "_GLEAM_v4.2a_MO.nc");
    
    lon_GLEAM = single(ncread(filepath, "lon"));
    lat_GLEAM = single(ncread(filepath, "lat"));
    time_GLEAM = single(ncread(filepath, "time"));
    
    GLEAM_data = single(fliplr(rot90(ncread(filepath, "E"), 3)));
    lat_GLEAM  = flip(lat_GLEAM);
    
    [lon_GLEAM_grid, lat_GLEAM_grid, time_GLEAM_grid] = meshgrid(lon_GLEAM, lat_GLEAM, time_GLEAM);
    [lon_base3_grid, lat_base3_grid, time_base3_grid] = meshgrid(single(lon), single(lat), single(time_GLEAM));
    
    GELAM_data_interped = interp3(lon_GLEAM_grid, lat_GLEAM_grid, time_GLEAM_grid, GLEAM_data, lon_base3_grid, lat_base3_grid, time_base3_grid);
    
    outFile = strcat("F:/GLEAM/v42a/E/05/E_", num2str(year), "_05.nc");
    nccreate(outFile, 'lon', 'Dimensions', {'lon', length(lon)});
    nccreate(outFile, 'lat', 'Dimensions', {'lat', length(lat)});
    nccreate(outFile, 'time', 'Dimensions', {'time', length(time_GLEAM)});
    nccreate(outFile, 'E', 'Dimensions', {'lat', length(lat), 'lon', length(lon), 'time', length(time_GLEAM)}, 'Datatype', 'single');
    ncwrite(outFile, 'lon', lon);
    ncwrite(outFile, 'lat', lat);
    ncwrite(outFile, 'time', time_GLEAM);
    ncwrite(outFile, 'E', GELAM_data_interped);
    
    info = ncinfo(filepath);
    
    % 全局属性复制
    for i = 1:length(info.Attributes)
        name = info.Attributes(i).Name;
        value = info.Attributes(i).Value;
        ncwriteatt(outFile, '/', name, value);
    end
    
    % 变量属性复制（lat/lon/time/E 除外或部分修改）
    for v = 1:length(info.Variables)
        varname = info.Variables(v).Name;
        attrs = info.Variables(v).Attributes;
        for a = 1:length(attrs)
            if strcmp(attrs(a).Name, '_FillValue')
                continue
            end
        end
    end
end