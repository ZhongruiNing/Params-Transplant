function write_nc(out_filepath, lon, lat, data, variable_name, variable_long_name)
    % 创建NetCDF文件
    ncid = netcdf.create(out_filepath, 'CLOBBER');
    % 定义维度
    lon_dim = netcdf.defDim(ncid, 'longitude', length(lon));
    lat_dim = netcdf.defDim(ncid, 'latitude', length(lat));
    % 定义经度变量
    lon_var = netcdf.defVar(ncid, 'longitude', 'NC_DOUBLE', lon_dim);
    netcdf.putAtt(ncid, lon_var, 'units', 'degrees_east');
    netcdf.putAtt(ncid, lon_var, 'long_name', 'Longitude');
    % 定义纬度变量
    lat_var = netcdf.defVar(ncid, 'latitude', 'NC_DOUBLE', lat_dim);
    netcdf.putAtt(ncid, lat_var, 'units', 'degrees_north');
    netcdf.putAtt(ncid, lat_var, 'long_name', 'Latitude');
    % 定义BFI变量（注意维度顺序为 latitude × longitude）
    soil_var = netcdf.defVar(ncid, variable_name, 'NC_DOUBLE', [lat_dim, lon_dim]);
    netcdf.putAtt(ncid, soil_var, 'long_name', variable_long_name);
    netcdf.putAtt(ncid, soil_var, 'units', '1');
    % 结束定义模式
    netcdf.endDef(ncid);
    % 写入数据
    netcdf.putVar(ncid, lon_var, lon);
    netcdf.putVar(ncid, lat_var, lat);
    netcdf.putVar(ncid, soil_var, data);
    % 关闭文件
    netcdf.close(ncid);
end