Climatology Benchmark
=====================

Perform a climatology on ERA5 data, finding the 90th percentile temperature
over the period 1980 - 2019 for each gridpoint at each day of the year

Process:
1. Open the data from NCI archive
2. Resample the hourly dataset to daily mean
3. Smooth the data with a 5 day rolling average
4. Group the data by day of the year and calculate the 90th percentile
5. Save to file
