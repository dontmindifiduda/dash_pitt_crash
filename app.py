import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

crash_df = pd.read_csv('data/clean-crash-data.csv')

### Define Constant Values

CENTER_LAT = (max(crash_df['DEC_LAT']) - min(crash_df['DEC_LAT'])) / 2 + min(crash_df['DEC_LAT'])
CENTER_LON = (max(crash_df['DEC_LONG']) - min(crash_df['DEC_LONG'])) / 2 + min(crash_df['DEC_LONG'])  

MAP_PANEL_HEIGHT = 700

DISCRETE_COLORS = px.colors.qualitative.G10

# Categorical varible label dictionaries
illum_dict = {
    1: 'Daylight',
    2: 'Dark - No Street Lights',
    3: 'Dark - Street Lights',
    4: 'Dusk',
    5: 'Dawn',
    6: 'Dark - Unknown Roadway Lighting',
    8: 'Other or Unknown'
}

collision_dict = {
    0: 'Non-Collision',
    1: 'Rear-End',
    2: 'Head-On',
    3: 'Rear-to-Rear (Backing)',
    4: 'Angle',
    5: 'Sideswipe (Same Direction)',
    6: 'Sideswipe (Opposite Direction)',
    7: 'Hit Fixed Object',
    8: 'Hit Pedestrian',
    9: 'Other or Unknown'
}

condition_dict = {
    0: 'Dry',
    1: 'Wet',
    2: 'Sand / Mud / Dirt / Oil / Gravel',
    3: 'Snow-Covered',
    4: 'Slush',
    5: 'Ice',
    6: 'Ice Patches',
    7: 'Water (Standing or Moving)',
    9: 'Other or Unknown'
}

relation_dict = {
    1: 'On Roadway',
    2: 'Shoulder',
    3: 'Median',
    4: 'Roadside ',
    5: 'Outside Trafficway ',
    6: 'In Parking Lane',
    7: 'Intersection of Ramp and Highway',
    9: 'Other or Unknown'
}

injury_dict = {
    0: 'No Injuries Reported',
    1: 'Minor Injury',
    2: 'Moderate Injury',
    3: 'Major Injury',
    4: 'Fatal ',
}

day_dict = {
    1: 'Sunday',
    2: 'Monday',
    3: 'Tuesday',
    4: 'Wednesday',
    5: 'Thursday',
    6: 'Friday',
    7: 'Saturday'
}

# Categorical variable color maps for bar plots
illum_color_map = dict(zip(illum_dict.values(), DISCRETE_COLORS[:len(illum_dict)]))
condition_color_map = dict(zip(condition_dict.values(), DISCRETE_COLORS[:len(condition_dict)]))
relation_color_map = dict(zip(relation_dict.values(), DISCRETE_COLORS[:len(relation_dict)]))
collision_color_map = dict(zip(collision_dict.values(), DISCRETE_COLORS[:len(collision_dict)]))
injury_color_map = dict(zip(injury_dict.values(), DISCRETE_COLORS[:len(injury_dict)]))

# Create blank figure to display when there is not enough data
FIG_NONE = go.Figure()
FIG_NONE = FIG_NONE.add_annotation(
    x=2, 
    y=3,
    text="Not Enough Data to Display",
    showarrow=False,
    yshift=10,
    font=dict(
        size=16
    )
)
FIG_NONE.update_layout(
    xaxis = dict(
        showgrid=False,
        ticks='',
        showticklabels=False,
        gridcolor='#FFFFFF',
        range=(0,4),
        zerolinecolor='#FFFFFF'),
    yaxis = dict(
        showgrid=False,
        ticks='',
        showticklabels=False,
        gridcolor='#FFFFFF',
        range=(0,6),
        zerolinecolor='#FFFFFF')
)

### Define Helper Functions

# Retrieve data with filters from user controls
def get_data(cluster_number, collision_type, road_condition, illumination, relation, injury, year_range, month_range, highlight):
    if highlight:
        df = crash_df.loc[crash_df[highlight] == 1].reset_index(drop=True) 
        
    else:
        df = crash_df
    df = df.loc[(df['CRASH_YEAR'] >= year_range[0]) & (df['CRASH_YEAR'] <= year_range[1])].reset_index(drop=True)
    df = df.loc[(df['CRASH_MONTH'] >= month_range[0]) & (df['CRASH_MONTH'] <= month_range[1])].reset_index(drop=True)
    df = df.loc[df['KMODE_CLUSTER'].isin(cluster_number)].reset_index(drop=True)
    df = df.loc[df['COLLISION_TYPE'].isin(collision_type)].reset_index(drop=True)
    df = df.loc[df['ROAD_CONDITION'].isin(road_condition)].reset_index(drop=True)
    df = df.loc[df['ILLUMINATION'].isin(illumination)].reset_index(drop=True)
    df = df.loc[df['RELATION_TO_ROAD'].isin(relation)].reset_index(drop=True)
    df = df.loc[df['MAX_INJURY_SEVERITY'].isin(injury)].reset_index(drop=True)
    
    if len(df) == 0:
        df.loc[0] = 0
        
    return df

# Create bar chart
def make_bar_chart(df, var_name, y_title, x_label_dict, color_map):
    data_group = df.groupby([var_name]).agg('count').reset_index()
    data_group = data_group.replace({var_name: x_label_dict}).sort_values(by='CRASH_CRN', ascending=False)
    
    return px.bar(data_group, 
                  x=var_name, 
                  y="CRASH_CRN", 
                  color=var_name,
                  color_discrete_map=color_map,
                  labels={var_name: y_title, 'CRASH_CRN': '# of Accidents'}).update_layout(showlegend=False)

# Create heatmap
def generate_heatmap(df):
    day_hour_group = df[df['HOUR_OF_DAY'] != 99.0].groupby(['DAY_OF_WEEK', 'HOUR_OF_DAY'],sort=False).agg(['count'])
    day_hour_heatmap = pd.pivot_table(day_hour_group, values='CRASH_CRN', index=['DAY_OF_WEEK'], columns='HOUR_OF_DAY')
    day_hour_heatmap = day_hour_heatmap.sort_values(by='DAY_OF_WEEK', ascending=True)
        
    return day_hour_heatmap

### Dash App
# Create app
app = dash.Dash(
    __name__,
    meta_tags=[{"name":"viewport", "content":"width=device=width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.LUMEN],
    suppress_callback_exceptions=True
)
server = app.server

app.title = 'Pittsbugh Car Accident Explorer (2010 - 2019)'


# Define controls
controls = dbc.Card(
    [
        html.H3(['Control Panel']),
        html.P(['Use the controls below to filter the data and change how it is visualized! If the Scatter Plot radio button below is selected, you may change how the points are colored on the map by changing tabs in the bar plot panel in the bottom right corner of the screen.']),
        dbc.FormGroup(
            [
                html.H5(['Hexbins or Heat Density Map']),
                dcc.RadioItems(
                    id='map-type',
                    options=[
                        {'label': 'Hexbins', 'value': 0},
                        {'label': 'Heat Density Map', 'value': 1},
                        {'label': 'Scatter Plot', 'value':2}
                    ],
                    value=0,
                    labelStyle={"margin-right": "20px"},
                    inputStyle={"margin-right": "5px"}
                ),
            ], className='selector-group'
        ),
        dbc.FormGroup(
            [
                html.H5(['Highlight Specific Characteristics:']),
                dcc.Dropdown(
                    id='highlight-dropdown',
                    options=[
                        {'label': 'None', 'value': 0},
                        {'label': 'Interstate', 'value': 'INTERSTATE'},
                        {'label': 'State Road', 'value': 'STATE_ROAD'},
                        {'label': 'Local Road', 'value': 'LOCAL_ROAD'},
                        {'label': 'Work Zone', 'value': 'WORK_ZONE_IND'},
                        {'label': 'School Zone', 'value': 'SCH_ZONE_IND'},
                        {'label': 'Bicycle', 'value': 'BICYCLE'},
                        {'label': 'Pedestrian', 'value': 'PEDESTRIAN'},
                        {'label': 'Motorcycle', 'value': 'MOTORCYCLE'},
                        {'label': 'Hazardous Truck', 'value': 'HAZARDOUS_TRUCK'},
                        {'label': 'Heavy Truck', 'value': 'HVY_TRUCK_RELATED'},
                        {'label': 'Deer', 'value': 'DEER_RELATED'},
                        {'label': 'Unbelted Passengers/Driver', 'value': 'UNBELTED'},
                        {'label': 'Unlicensed Driver', 'value': 'UNLICENSED'},
                        {'label': 'Alcohol Related', 'value': 'ALCOHOL_RELATED'},
                        {'label': 'Drug Related', 'value': 'DRUG_RELATED'},
                        {'label': 'Cell Phone', 'value': 'CELL_PHONE'},
                        {'label': 'Impaired Driver', 'value': 'IMPAIRED_DRIVER'},
                        {'label': 'Distracted Driver', 'value': 'DISTRACTED'},
                        {'label': 'Fatigue / Asleep', 'value': 'FATIGUE_ASLEEP'},
                        {'label': 'Tailgating', 'value': 'TAILGATING'},
                        {'label': 'Speeding', 'value': 'SPEEDING_RELATED'},
                        {'label': 'Aggressive Driving', 'value': 'AGGRESSIVE_DRIVING'},
                        {'label': 'Running a Red Light', 'value': 'RUNNING_RED_LT'},
                        {'label': 'Curved Road', 'value': 'CURVED_ROAD'},
                    ],
                    value=0,
                    multi=False
                ),      
            ], className='selector-group'
        ),
        dbc.FormGroup(
            [
                html.H5(['K-Modes Cluster:']),
                html.Div([
                    dcc.Dropdown(
                        id='cluster-dropdown',
                        options=[
                            {'label': '0 - Local Road Daytime Impairment / Inclement Weather', 'value': 0},
                            {'label': '1 - Local Road Aggressive Driving / Lack of Clearance ', 'value': 1},
                            {'label': '2 - Large Road Nighttime Impairment / Inclement Weather', 'value': 2},
                            {'label': '3 - Large Road Rear-End / Tailgating / Speeding - Injury-Causing', 'value': 3},
                            {'label': '4 - Pedestrian / Motorcycle / Bicycle - Injury-Causing', 'value': 4},
                            {'label': '5 - Local Road Intersection / Running a Red Light / Wet Roads', 'value': 5},
                        ],
                        value=[0,1,2,3,4,5],
                        multi=True
                    ),
                ])
            ], className='selector-group dropdown-kmodes'
        ),
        dbc.FormGroup(
            [
                html.H5(['Collision Type:']),
                dcc.Dropdown(
                    id='collision-type',
                    options=[
                        {'label': 'Non-Collision', 'value': 0},
                        {'label': 'Rear-End', 'value': 1},
                        {'label': 'Head-On', 'value': 2},
                        {'label': 'Rear-to-Rear (Backing)', 'value': 3},
                        {'label': 'Angle', 'value': 4},
                        {'label': 'Sideswipe (Same Direction)', 'value': 5},
                        {'label': 'Sideswipe (Opposite Direction)', 'value': 6},
                        {'label': 'Hit Fixed Object', 'value': 7},
                        {'label': 'Hit Pedestrian', 'value': 8},
                        {'label': 'Other or Unknown', 'value': 9},
                    ],
                    value=[0,1,2,3,4,5,6,7,8,9],
                    multi=True
                ),
            ], className='selector-group dropdown-collision'
        ),
        dbc.FormGroup(
            [
                html.H5(['Road Condition:']),
                dcc.Dropdown(
                    id='road-condition',
                    options=[
                        {'label': 'Dry', 'value': 0},
                        {'label': 'Wet', 'value': 1},
                        {'label': 'Sand/Mud/Dirt/Oil/Gravel', 'value': 2},
                        {'label': 'Snow Covered', 'value': 3},
                        {'label': 'Slush', 'value': 4},
                        {'label': 'Ice', 'value': 5},
                        {'label': 'Ice Patches', 'value': 6},
                        {'label': 'Water (Standing or Moving)', 'value': 7},
                        {'label': 'Other or Unknown', 'value': 9},
                    ],
                    value=[0,1,2,3,4,5,6,7,9],
                    multi=True
                ),
            ], className='selector-group dropdown-condition'
        ),
        dbc.FormGroup(
            [
                html.H5(['Illumination:']),
                dcc.Dropdown(
                    id='illumination',
                    options=[
                        {'label': 'Daylight', 'value': 1},
                        {'label': 'Dark - No Street Lights', 'value': 2},
                        {'label': 'Dark - Street Lights', 'value': 3},
                        {'label': 'Dusk', 'value': 4},
                        {'label': 'Dawn', 'value': 5},
                        {'label': 'Dark - Unknown Roadway Lighting', 'value': 6},
                        {'label': 'Other or Unknown', 'value': 8}
                    ],
                    value=[1,2,3,4,5,6,8],
                    multi=True
                ),
            ], className='selector-group dropdown-illumination'
        ),
        dbc.FormGroup(
            [
                html.H5(['Relation to Road:']),
                dcc.Dropdown(
                    id='relation',
                    options=[
                        {'label': 'On Roadway', 'value': 1},
                        {'label': 'Shoulder', 'value': 2},
                        {'label': 'Median', 'value': 3},
                        {'label': 'Roadside', 'value': 4},
                        {'label': 'Outside Traficway', 'value': 5},
                        {'label': 'In Parking Lane', 'value': 6},
                        {'label': 'Intersection of Ramp/Highway', 'value': 7},
                        {'label': 'Other or Unknown', 'value': 9}
                    ],
                    value=[1,2,3,4,5,6,7,9],
                    multi=True
                ),
            ], className='selector-group dropdown-relation'
        ),
        dbc.FormGroup(
            [
                html.H5(['Injuries / Deaths:']),
                dcc.Dropdown(
                    id='injury',
                    options=[
                        {'label': 'No Reported Injuries/Deaths', 'value': 0},
                        {'label': 'Minor Injury', 'value': 1},
                        {'label': 'Moderate Injury', 'value': 2},
                        {'label': 'Major Injury', 'value': 3},
                        {'label': 'Fatal', 'value': 4},
                    ],
                    value=[0,1,2,3,4],
                    multi=True
                ),
            ], className='selector-group dropdown-injury'
        ),
        dbc.FormGroup(
            [
                html.H5(['Year Range:']),
                dcc.RangeSlider(
                    id='year-slider',
                    min=2010,
                    max=2019,
                    step=1,
                    allowCross=False,
                    marks={
                        2010: '2010',
                        2011: '2011',
                        2012: '2012',
                        2013: '2013',
                        2014: '2014',
                        2015: '2015',
                        2016: '2016',
                        2017: '2017',
                        2018: '2018',
                        2019: '2019',
                    },
                    value=[2010,2019],
                ),
            ], className='selector-group'
        ),
        dbc.FormGroup(
            [
                html.H5(['Month Range:']),
                dcc.RangeSlider(
                    id='month-slider',
                    min=1,
                    max=12,
                    step=1,
                    allowCross=False,
                    marks={
                        1: '1',
                        2: '2',
                        3: '3',
                        4: '4',
                        5: '5',
                        6: '6',
                        7: '7',
                        8: '8',
                        9: '9',
                        10: '10',
                        11: '11',
                        12: '12',
                    },
                    value=[1,12],
                ),
            ], className='selector-group'
        ),
    ], className="control-card"
)

# Define app layout
app.layout = dbc.Container(
    [
        html.H2('Pittsburgh Car Accident Explorer (2010 - 2019)'),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col([
                    controls,
                ], md=4, align='start'),
                dbc.Col([
                    dbc.Card([dcc.Graph(id='crash-map')]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([dcc.Loading(children=dcc.Graph(id='crash-heat'))]),
                        ]),
                        dbc.Col([
                            dbc.Card([
                                dbc.Tabs(
                                    [
                                        dbc.Tab(label='Illumination', 
                                                tab_id='bar-illumination',
                                                children=[
                                                    dcc.Loading(children=dcc.Graph(id='bar-plot-illumination'))
                                                ]),
                                        dbc.Tab(label='Road Condition', 
                                                tab_id='bar-condition',
                                                children=[
                                                    dcc.Loading(children=dcc.Graph(id='bar-plot-condition'))
                                                ]),
                                        dbc.Tab(label='Relation to Road', 
                                                tab_id='bar-relation',
                                                children=[
                                                    dcc.Loading(children=dcc.Graph(id='bar-plot-relation'))
                                                ]),
                                        dbc.Tab(label='Collision Type', 
                                                tab_id='bar-collision',
                                                children=[
                                                    dcc.Loading(children=dcc.Graph(id='bar-plot-collision'))
                                                ]),
                                        dbc.Tab(label='Max Injury Severity', 
                                                tab_id='bar-injury',
                                                children=[
                                                    dcc.Loading(children=dcc.Graph(id='bar-plot-injury'))
                                                ]),
                                    ], 
                                    id='tabs',
                                    active_tab='bar-illumination'
                                ),
                                html.Div(id='tab-content', className='p-4')
                            ], style={'padding':'10px'})
                        ])
                    ])
                    
                ], md=8, align='start')
            ],
            align='center',
        ),
    ],
    fluid=True
)



# Define callback functions
# Update geo map
@app.callback(
    Output('crash-map', component_property='figure'),
    [
        Input('map-type', component_property='value'),
        Input('cluster-dropdown', component_property='value'),
        Input('collision-type', component_property='value'),
        Input('road-condition', component_property='value'),
        Input('illumination', component_property='value'),
        Input('relation', component_property='value'),
        Input('injury', component_property='value'),
        Input('year-slider', component_property='value'),
        Input('month-slider', component_property='value'),
        Input('highlight-dropdown', component_property='value'),
        Input('tabs', 'active_tab'),
        State('crash-map', component_property='relayoutData')
    ],
)
def update_geo_map(map_type, cluster_number, collision_type, 
                   road_condition, illumination, relation, 
                   injury, year_range, month_range, 
                   highlight, active_tab, map_figure):

    try:
        current_zoom = (map_figure['mapbox.zoom'])
        current_center_lat = (map_figure['mapbox.center']['lat'])
        current_center_lon = (map_figure['mapbox.center']['lon'])
    except:
        current_zoom = 10
        current_center_lat = CENTER_LAT
        current_center_lon = CENTER_LON

    df = get_data(cluster_number, collision_type, road_condition, illumination, relation, 
                  injury, year_range, month_range, highlight)

    if map_type == 2:
        if active_tab == 'bar-illumination':
            color_value = 'ILLUMINATION'
            df = df.replace({color_value:illum_dict})
            color_map = illum_color_map
        elif active_tab == 'bar-condition': 
            color_value = 'ROAD_CONDITION'
            df = df.replace({color_value:condition_dict})
            color_map = condition_color_map
        elif active_tab == 'bar-relation':
            color_value = 'RELATION_TO_ROAD'
            df = df.replace({color_value:relation_dict})
            color_map = relation_color_map
        elif active_tab == 'bar-injury':
            color_value = 'MAX_INJURY_SEVERITY'
            df = df.replace({color_value:injury_dict})
            color_map = injury_color_map
        else:
            color_value = 'COLLISION_TYPE'
            df = df.replace({color_value:collision_dict})
            color_map = collision_color_map
            
        fig = px.scatter_mapbox(df, lat='DEC_LAT', lon='DEC_LONG', 
                                color=color_value, mapbox_style='stamen-terrain',
                                color_discrete_map=color_map,
                                zoom=current_zoom, 
                                center=dict(lat=current_center_lat, lon=current_center_lon))
        fig.layout.height = MAP_PANEL_HEIGHT
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    
    elif map_type == 1:
        fig = px.density_mapbox(
            df, lat='DEC_LAT', lon='DEC_LONG', radius=5,
            center=dict(lat=current_center_lat, lon=current_center_lon), 
            zoom=current_zoom, mapbox_style='stamen-terrain')
        fig.layout.height = MAP_PANEL_HEIGHT
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
    elif map_type == 0:
        fig = ff.create_hexbin_mapbox(
            data_frame=df, lat="DEC_LAT", lon="DEC_LONG",
            nx_hexagon=300, opacity=0.5, labels={"color": "# of Accidents"},
            min_count=1, mapbox_style='stamen-terrain', show_original_data=True,
            original_data_marker=dict(size=3, opacity=0.6, color='black'),
            zoom=current_zoom, center=dict(lat=current_center_lat, lon=current_center_lon)
        )
        fig.layout.height = MAP_PANEL_HEIGHT
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
        )
    return fig

# Update bar plots 
@app.callback(
    Output('bar-plot-illumination', component_property='figure'),
    Output('bar-plot-collision', component_property='figure'),
    Output('bar-plot-condition', component_property='figure'),
    Output('bar-plot-relation', component_property='figure'),
    Output('bar-plot-injury', component_property='figure'),
    [
        Input('cluster-dropdown', component_property='value'),
        Input('collision-type', component_property='value'),
        Input('road-condition', component_property='value'),
        Input('illumination', component_property='value'),
        Input('relation', component_property='value'),
        Input('injury', component_property='value'),
        Input('year-slider', component_property='value'),
        Input('month-slider', component_property='value'),
        Input('highlight-dropdown', component_property='value'),
        Input('tabs', 'active_tab'),
    ],
)
def update_bar(cluster_number, collision_type, 
               road_condition, illumination, relation, 
               injury, year_range, month_range, 
               highlight, active_tab):

    df = get_data(cluster_number, collision_type, road_condition, illumination, relation, 
                  injury, year_range, month_range, highlight)
 
    if df.shape[0] < 2:
        bar_illum_fig = FIG_NONE
        bar_collision_fig = FIG_NONE
        bar_condition_fig = FIG_NONE
        bar_relation_fig = FIG_NONE
        bar_injury_fig = FIG_NONE
    else:
        bar_illum_fig = make_bar_chart(df=df, 
                                 var_name='ILLUMINATION', 
                                 y_title='Illumination',
                                 x_label_dict=illum_dict,
                                 color_map=illum_color_map)
        bar_collision_fig = make_bar_chart(df=df, 
                             var_name='COLLISION_TYPE', 
                             y_title='Collision Type',
                             x_label_dict=collision_dict,
                             color_map=collision_color_map)
        bar_condition_fig = make_bar_chart(df=df, 
                             var_name='ROAD_CONDITION', 
                             y_title='Road Condition',
                             x_label_dict=condition_dict,
                             color_map=condition_color_map)
        bar_relation_fig = make_bar_chart(df=df, 
                             var_name='RELATION_TO_ROAD', 
                             y_title='Relation to Road',
                             x_label_dict=relation_dict,
                             color_map=relation_color_map)
        bar_injury_fig = make_bar_chart(df=df, 
                             var_name='MAX_INJURY_SEVERITY', 
                             y_title='Maximum Injury Severity',
                             x_label_dict=injury_dict,
                             color_map=injury_color_map)
        
    bar_illum_fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(tickfont=dict(size=10))
    )
    bar_illum_fig.layout.height = MAP_PANEL_HEIGHT*1.5
    bar_collision_fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(tickfont=dict(size=10))
    )
    bar_collision_fig.layout.height = MAP_PANEL_HEIGHT*1.5
    bar_condition_fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(tickfont=dict(size=10)),
    )
    bar_condition_fig.layout.height = MAP_PANEL_HEIGHT*1.5
    bar_relation_fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(tickfont=dict(size=10)),
    )
    bar_relation_fig.layout.height = MAP_PANEL_HEIGHT*1.5
    bar_injury_fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(tickfont=dict(size=10))
    )
    bar_injury_fig.layout.height = MAP_PANEL_HEIGHT*1.5
    
    return bar_illum_fig, bar_collision_fig, bar_condition_fig, bar_relation_fig, bar_injury_fig


# Update heat map
@app.callback(
    Output('crash-heat', component_property='figure'),
    [
        Input('cluster-dropdown', component_property='value'),
        Input('collision-type', component_property='value'),
        Input('road-condition', component_property='value'),
        Input('illumination', component_property='value'),
        Input('relation', component_property='value'),
        Input('injury', component_property='value'),
        Input('year-slider', component_property='value'),
        Input('month-slider', component_property='value'),
        Input('highlight-dropdown', component_property='value')
    ],
)
def update_bar_and_heat(cluster_number, collision_type, 
                        road_condition, illumination, relation, 
                        injury, year_range, month_range, 
                        highlight):

    df = get_data(cluster_number, collision_type, road_condition, illumination, relation, 
                  injury, year_range, month_range, highlight)
    
    day_hour_heatmap = generate_heatmap(df)
    
    if day_hour_heatmap.shape[0] < 7 or day_hour_heatmap.shape[1] < 2:
        heat_fig = FIG_NONE 
    else: 
        heat_fig = px.imshow(day_hour_heatmap,
                             labels=dict(x="Time of Day", y="Day of Week", color='# of Accidents'),
                             x=[str(int(x[1])) for x in day_hour_heatmap.columns.values],
                             y=list(map(day_dict.get, day_hour_heatmap.index.values))
                            )

        heat_fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20)
        )
    
    return heat_fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)