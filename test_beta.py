import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.utils import check_array
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from datetime import datetime
from flask import Flask
from plotly import tools
from plotly.figure_factory import create_2d_density
from plotly.graph_objs import graph_objs

def mean_absolute_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_squared_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_array(y_true, y_pred)
    return np.mean((y_true - y_pred)**2 / y_true) * 100

def get_regression_metrics(rf, train, y_train, valid, y_valid):
    #rf.fit(train, y_train)
    y_pred = rf.predict(valid)
    
    validation = {}
    validation["msle"] = mean_squared_log_error(y_valid, y_pred)
    validation["mae"] = mean_absolute_error(y_valid, y_pred)
    validation["mse"] = mean_squared_error(y_valid, y_pred)
    validation["mape"] = mean_absolute_percentage_error(y_valid, y_pred)
    validation["r2"] = r2_score(y_valid, y_pred)
    validation["rmsle"] = np.sqrt(mean_squared_log_error(y_valid, y_pred))
    validation["rmse"] = np.sqrt(mean_squared_error(y_valid, y_pred))
    validation["preds"] = y_pred
    #validation["rmspe"] = mean_squared_percentage_error(y_valid, y_pred)
    
    y_pred_train = rf.predict(train)
    training = {}
    training["msle"] = mean_squared_log_error(y_train, y_pred_train)
    training["mae"] = mean_absolute_error(y_train, y_pred_train)
    training["mse"] = mean_squared_error(y_train, y_pred_train)
    training["mape"] = mean_absolute_percentage_error(y_train, y_pred_train)
    training["r2"] = r2_score(y_train, y_pred_train)
    training["rmsle"] = np.sqrt(mean_squared_log_error(y_train, y_pred_train))
    training["rmse"] = np.sqrt(mean_squared_error(y_train, y_pred_train))
    training["preds"] = y_pred_train
    
    return validation, training

########################
# Model + Data Loading #
########################
rf = joblib.load('./data/rf_trained.pkl')
train = pd.read_feather('./data/x_train.feather')
valid = pd.read_feather('./data/x_valid.feather')
y_train = pd.read_csv('./data/y_train.csv',header=None).iloc[:,0]
y_valid = pd.read_csv('./data/y_valid.csv',header=None).iloc[:,0]
print('Data Loaded!')

valid_metrics, train_metrics = get_regression_metrics(rf, train, y_train, valid, y_valid)
df_test = pd.DataFrame(valid_metrics).head(1).drop(["preds"], axis = 1).apply(lambda x:round(x, 4))
print('Metrics Loaded!')

################
# Server Stuff #
################
server = Flask(__name__)
app = dash.Dash(name='mav', sharing=True, server=server, csrf_protect=False)

################
# Layout Stuff #
################
def xy_sample(data, size):
    data_len = data.shape[0]
    if data_len > size:
        sub_val = np.random.choice(np.arange(data.shape[0]), size, replace=False)
    else:
        sub_val = np.arange(data.shape[0])
    return sub_val


# variables
columns = ["type", "mae", "mape", "mse", "msle", "rmse", "rmsle", "r2"]
df_test = pd.DataFrame(valid_metrics).head(1).drop(["preds"], axis = 1).apply(lambda x:round(x, 4))
df_test["type"] = "validation"
df_test = df_test[columns] 
df_train = pd.DataFrame(train_metrics).head(1).drop(["preds"], axis = 1).apply(lambda x:round(x, 4))
df_train["type"] = "training"
df_train = df_train[columns]

colorscale =[[0.0, 'rgb(50,50,50)'], [0.001, 'rgb(5, 57, 94)'],[1.0,'rgb(126, 247, 27)']]#[1.0, 'rgb(242, 59, 31)']]
train_rows = train.shape[0]
valid_rows = valid.shape[0]
train_mean = np.mean(y_train)
valid_mean = np.mean(y_valid)
color_bar_nums = [60, 5]
color_bar_list = ["max", "min"]
annot_font = {"family":'helvetica', "size":20,"color":"white"}#'rgb(2, 150, 255)'}
print('Plot Preprocessing Complete!')

# plot a subsample
sampSize = 2_000
iTrainSamp = xy_sample(train, sampSize)
iValidSamp = xy_sample(valid, sampSize)
y_train = y_train.loc[iTrainSamp]
y_valid = y_valid.loc[iValidSamp]

# traces
trace_hist = graph_objs.Histogram2dcontour(x = valid_metrics["preds"][iValidSamp], 
                                           y= y_valid, 
                                           name= "Validation set",
                                           ncontours=100,
                                           nbinsx=100,
                                           nbinsy=100, 
                                           autocontour= False,
                                           contours = {"showlines":False},
                                           yaxis='y2', 
                                           colorscale=colorscale,
                                           colorbar={"yanchor":"top", 
                                                     "len":.5,
                                                    'ticktext':color_bar_list,
                                                     'tickvals':color_bar_nums,
                                                    'tickmode':'array'}
                                                    #'tickfont':{'color':'white'}}
                                          )

trace_hist2 = graph_objs.Histogram2dcontour(x = train_metrics["preds"][iTrainSamp], 
                                            y= y_train, 
                                            name= "Training set", 
                                            xaxis='x2',
                                            yaxis='y2',
                                            ncontours=100,
                                            nbinsx=100,
                                            nbinsy=100, 
                                            autocontour= False,
                                            contours = {"showlines":False},
                                            colorscale=colorscale,
                                            showscale=False
                                            )

trace = go.Scatter(x = valid_metrics["preds"][iValidSamp], 
                   y= y_valid, 
                   mode='markers',
                   marker = {"opacity":0.7, 
                             "color":'rgb(22, 199, 229)'},
                   xaxis='x2', 
                   textfont=dict(family='helvetica', 
                                 size=14, 
                                 color='rgb(193, 192, 191)'),
                   showlegend=True, 
                   name = "Validation")

trace2 = go.Scatter(x = train_metrics["preds"][iTrainSamp], y= y_train, 
                          mode='markers',
                          marker = {"opacity":0.7, 
                                    "color":'rgb(2, 150, 255)'},
                          showlegend=True, 
                          name = "Training", 
                   )

annotations=[
    dict(x=0.19,
         y=1.04,
         xref='paper',
         yref='paper',
         text="Validation", 
         showarrow=False, 
         font= annot_font
        ), 
    dict(x=0.82, 
         y=1.04,
         xref='paper',
         yref='paper',
         text="Training", 
         showarrow=False,
         font=annot_font
        ),
    dict(x=0.14,
         y=.42,
         xref='paper',
         yref='paper',
         text="Density of Validation", 
         showarrow=False,
         font=annot_font
        ), 
    dict(x=0.87, 
         y=.42,
         xref='paper',
         yref='paper',
         text="Density of Training", 
         showarrow=False,
         font=annot_font
        )
]

layout = go.Layout(title='Predicted vs. True Values', titlefont={'color':"white"},
                   font=dict(family='helvetica', 
                             size=14, 
                             color='rgb(193, 192, 191)'),
                   xaxis={'title': 'Predicted', 
                          "domain":[0, 0.45]},
                   yaxis={'title': 'True', 
                          "domain":[0.6,1]},
                   xaxis2={'title': 'Predicted',
                           "domain":[0.55, 1]},
                   yaxis2={"title":"True", 
                           'domain':[0,.4]},
                   paper_bgcolor = "rgb(20,20,20)",
                   plot_bgcolor="rgb(50,50,50)",
                   autosize=False,
                   width=900,
                   height=900,
                   annotations=annotations)
print('Layout Complete!')
app.layout = html.Div(children=[
    html.H2(children='Beta Release'),


    html.Table(
               [html.Tr([html.Th(col) for col in df_test.columns])] +
               [html.Tr( [html.Td(df_test.iloc[i][col]) for col in df_test.columns] ) 
                                for i in range(min(len(df_test), 10))]+
               [html.Tr( [html.Td(df_train.iloc[i][col]) for col in df_train.columns] ) 
                                for i in range(min(len(df_train), 10))]
               ,
               style={'border': '4px solid',
                      'text-align': 'center',
                      'width':'900',
                      "background-color": "#141414",
                      "color": "#c1c0bf",
                      'border-color': "#c1c0bf" 
                     }
              ),
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [trace_hist, trace_hist2, trace, trace2],
            'layout': layout
        }
    )
])
print('Reticulating Splines!')