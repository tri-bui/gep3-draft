

# Selected features
feats = ['building_id', 'primary_use', 'square_feet', 'year_built', 'country',
         'dayofyear', 'hour', 'is_weekend', 'is_holiday',
         'rel_humidity', 'dew_temperature', 'sea_level_pressure',
         'wind_speed', 'wind_direction_y']

# Site to country mapping
countries = {0: 'US', 1: 'UK', 2: 'US', 3: 'US',
             4: 'US', 5: 'UK', 6: 'US', 7: 'CA',
             8: 'US', 9: 'US', 10: 'US', 11: 'CA',
             12: 'IE', 13: 'US', 14: 'US', 15: 'US'}

# Timezone conversion offsets for sites 0 - 15
tz_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]

# Variables to be imputed using linear interpolation and cubic interpolation
lin_interp_vars = ['wind_direction', 'wind_speed']
cub_interp_vars = ['air_temperature', 'dew_temperature', 'sea_level_pressure']