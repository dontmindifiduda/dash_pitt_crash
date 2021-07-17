import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_rows", 100)


### Load file

crash_df = pd.read_csv('data/crash-data.csv', low_memory=False)

### Remove outlier points
crash_df = crash_df.loc[(crash_df['DEC_LONG'] < -79.7) & (crash_df['DEC_LONG'] > -80.4)].reset_index(drop=True)
crash_df = crash_df.loc[(crash_df['DEC_LAT'] < 40.7) & (crash_df['DEC_LAT'] > 40.2)].reset_index(drop=True)

### Filter for City of Pittsburgh
crash_df = crash_df.loc[crash_df['MUNICIPALITY'] == 2301].reset_index(drop=True)



drop_cols = []

### Remove columns with only one unique value
for col in crash_df.columns:
    if len(crash_df[col].unique()) == 1:
        drop_cols.append(col)

### Select additional columns to drop from dataset
additional_drop_cols = [
    'MUNICIPALITY', 'POLICE_AGCY', 'LATITUDE', 'LONGITUDE', 'ACCESS_CTRL', 'STREET_NAME',
    'FLAG_CRN', 'ROADWAY_CRN', 'RDWY_SEQ_NUM', 'ADJ_RDWY_SEQ', 'ROADWAY_COUNTY',
    'ROAD_OWNER', 'ROUTE', 'SEGMENT', 'OFFSET', 'LN_CLOSE_DIR', 'SCHOOL_BUS_UNIT',
    'RDWY_SURF_TYPE_CD', 'SPEC_JURIS_CD', 'WORK_ZONE_TYPE', 'WORK_ZONE_LOC', 'CONS_ZONE_SPD_LIM', 
    'WORKERS_PRES', 'WZ_CLOSE_DETOUR', 'WZ_FLAGGER', 'WZ_LAW_OFFCR_IND', 'WZ_LN_CLOSURE', 'WZ_MOVING',
    'WZ_OTHER', 'WZ_SHLDER_MDN', 'LIMIT_65MPH'
]

numeric_drop_cols = [
    'TIME_OF_DAY', 'TOTAL_UNITS', 'PERSON_COUNT', 'VEHICLE_COUNT', 'AUTOMOBILE_COUNT', 'MOTORCYCLE_COUNT',
    'BUS_COUNT', 'SMALL_TRUCK_COUNT', 'HEAVY_TRUCK_COUNT', 'SUV_COUNT', 'VAN_COUNT', 'BICYCLE_COUNT',
    'FATAL_COUNT', 'INJURY_COUNT', 'MAJ_INJ_COUNT', 'MOD_INJ_COUNT', 'MIN_INJ_COUNT', 'UNK_INJ_DEG_COUNT',
    'UNK_INJ_PER_COUNT', 'UNB_MAJ_INJ_COUNT', 'UNBELTED_OCC_COUNT', 'BELTED_MAJ_INJ_COUNT', 'BICYCLE_DEATH_COUNT',
    'BICYCLE_MAJ_INJ_COUNT', 'COMM_VEH_COUNT', 'DRIVER_COUNT_16YR', 'DRIVER_COUNT_17YR', 'DRIVER_COUNT_18YR',
    'DRIVER_COUNT_19YR', 'DRIVER_COUNT_20YR', 'DRIVER_COUNT_50_64YR', 'DRIVER_COUNT_65_74YR', 'DRIVER_COUNT_75PLUS',
    'LANE_CLOSED', 'LANE_COUNT', 'SPEED_LIMIT', 'UNB_DEATH_COUNT', 'BELTED_DEATH_COUNT', 'MCYCLE_DEATH_COUNT',
    'MCYCLE_MAJ_INJ_COUNT', 'PED_COUNT', 'PED_DEATH_COUNT', 'PED_MAJ_INJ_COUNT', 'MAX_SEVERITY_LEVEL',
    'EST_HRS_CLOSED', 'TOT_INJ_COUNT'
]

drop_cols += additional_drop_cols 
drop_cols += numeric_drop_cols

### Filter for crashes occurring betwen 2010 and 2019 + drop columns
cat_crash_df = crash_df[crash_df['CRASH_YEAR'] > 2009].drop(drop_cols, axis=1).reset_index(drop=True)


### Impute missing values

cat_crash_df['HOUR_OF_DAY'] = cat_crash_df['HOUR_OF_DAY'].fillna(99)
cat_crash_df['WEATHER'] = cat_crash_df['WEATHER'].fillna(1)
cat_crash_df['ROAD_CONDITION'] = cat_crash_df['ROAD_CONDITION'].fillna(1)
cat_crash_df['SCH_BUS_IND'] = cat_crash_df['SCH_BUS_IND'].fillna('N')
cat_crash_df['SCH_ZONE_IND'] = cat_crash_df['SCH_ZONE_IND'].fillna('N')
cat_crash_df['NTFY_HIWY_MAINT'] = cat_crash_df['NTFY_HIWY_MAINT'].fillna('N')
cat_crash_df['TFC_DETOUR_IND'] = cat_crash_df['TFC_DETOUR_IND'].fillna('N')
cat_crash_df['MODERATE_INJURY'] = cat_crash_df['MODERATE_INJURY'].fillna(0)
cat_crash_df['RDWY_ORIENT'] = cat_crash_df['RDWY_ORIENT'].fillna('U')
cat_crash_df['LOCAL_ROAD'] = cat_crash_df['LOCAL_ROAD'].fillna(cat_crash_df['LOCAL_ROAD_ONLY'])
cat_crash_df.loc[cat_crash_df['RDWY_ORIENT'] == 'B', 'RDWY_ORIENT'] = 'U'

cat_crash_df = cat_crash_df.dropna()


### Label encode non-numeric categorical variables
encode_columns = ['SCH_BUS_IND', 'SCH_ZONE_IND', 'NTFY_HIWY_MAINT', 'RDWY_ORIENT', 'WORK_ZONE_IND', 'TFC_DETOUR_IND']

for col in encode_columns:
    le = LabelEncoder()
    cat_crash_df[col] = le.fit_transform(cat_crash_df[col])


### Perform k-modes clustering (n_clusters and init optimized previously)

kmode = KModes(
        n_clusters=6, 
        init='random',
        n_jobs=-1,
        random_state=73
    )

cat_crash_df['KMODE_CLUSTER'] = kmode.fit_predict(cat_crash_df.drop(['CRASH_CRN', 'CRASH_YEAR', 'DEC_LAT', 'DEC_LONG',], axis=1))


### COLLISION_TYPE - reclassify 98 (other) and 99 (unknown) as 9 (other/unknown)
cat_crash_df.loc[cat_crash_df['COLLISION_TYPE'] == 98, 'COLLISION_TYPE'] = 9 
cat_crash_df.loc[cat_crash_df['COLLISION_TYPE'] == 99, 'COLLISION_TYPE'] = 9 

### ROAD_CONDITION - reclassify 98 (other) and 99 (unknown) as 9 (other/unknown)
cat_crash_df.loc[cat_crash_df['ROAD_CONDITION'] == 22.0, 'ROAD_CONDITION'] = 9.0 
cat_crash_df.loc[cat_crash_df['ROAD_CONDITION'] == 98.0, 'ROAD_CONDITION'] = 9.0 
cat_crash_df.loc[cat_crash_df['ROAD_CONDITION'] == 99.0, 'ROAD_CONDITION'] = 9.0 
cat_crash_df.loc[cat_crash_df['ROAD_CONDITION'] == 8.0, 'ROAD_CONDITION'] = 9.0 

### Create categorical 
cat_crash_df['MAX_INJURY_SEVERITY'] = 0
cat_crash_df.loc[cat_crash_df['MINOR_INJURY'] == 1, 'MAX_INJURY_SEVERITY'] = 1
cat_crash_df.loc[cat_crash_df['MODERATE_INJURY'] == 1, 'MAX_INJURY_SEVERITY'] = 2
cat_crash_df.loc[cat_crash_df['MAJOR_INJURY'] == 1, 'MAX_INJURY_SEVERITY'] = 3
cat_crash_df.loc[cat_crash_df['FATAL'] == 1, 'MAX_INJURY_SEVERITY'] = 4

### Save dataframe

cat_crash_df.to_csv('data/clean-crash-data.csv', index=False)











