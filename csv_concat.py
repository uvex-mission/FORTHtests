# Concatenate multiple CSV files and add a column denoting temperature (TEMPK)

import pandas as pd

# Format: {T: 'path/to/file.csv', ...}

tdata = {111: 'PTC/260709_115200_ptc_hg.csv', 
        222: 'PTC/260710_133846_ptc_hg.csv'}

df = pd.concat(
    (pd.read_csv(fname).assign(TEMPK=temp) for temp, fname in tdata.items()),
    ignore_index=True
)

outfile = 'all_temps.csv'
df.to_csv(outfile)

print(df)
