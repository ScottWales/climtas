#!/usr/bin/env python
# Copyright 2020 Scott Wales
# author: Scott Wales <scott.wales@unimelb.edu.au>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import xarray
import glob


def validate_vocab(name, value, valid_values):
    if value not in valid_values:
        raise Exception(f'{value} is not a valid {name} value. Choose one of {valid_values}')


def era5(variable, category):
    validate_vocab('category', category, ['pressure', 'surface', 'land', 'wave', 'static'])

    if category == 'static':
        return xarray.open_dataset('/g/data/ub4/era5/netcdf/static_era5.nc')[variable.lower()]

    paths = sorted(glob.glob(f'/g/data/ub4/era5/netcdf/{category}/{variable}/*/{variable}_era5_global_*.nc'))[:-4]

    ds = xarray.open_mfdataset(paths,
            combine='nested',
            concat_dim='time',
            compat='override',
            coords='minimal',
            chunks={'latitude': 91, 'longitude': 180},
            parallel=True)

    return ds[variable.lower()]

