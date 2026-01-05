#!/bin/sh
set -eu

mc alias set local http://minio:9000 minioadmin minioadmin
mc mb -p local/testdata || true

mc cp /seed/grib_data/gfs/2025/11/25/test.grib2 local/testdata/
mc cp /seed/model/regional/cordex/output/EUR-11/GERICS/NCC-NorESM1-M/rcp85/r1i1p1/GERICS-REMO2015/v1/3hr/pr/v20181212/pr_EUR-11_NCC-NorESM1-M_rcp85_r1i1p1_GERICS-REMO2015_v2_3hr_200701020130-200701020430.nc local/testdata/
mc cp /seed/geodata/TCD/2021/10m/districts/DE111/TCD_S2021_R10m_DE111.tif local/testdata/

mc anonymous set public local/testdata

mc ls local/testdata
