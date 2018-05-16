import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from flask import Flask
from plotly.figure_factory import create_2d_density
from plotly.graph_objs import graph_objs


def plot_metrics(model, train, valid, y_train, y_valid, port=9999):
	server = Flask('mav')
	app = dash.Dash(name='mav', sharing=True, server=server, csrf_protect=False)
	app.css.config.serve_locally = True
	app.scripts.config.serve_locally = True
	# from rfpimp
	def importances(model, X_valid, y_valid, features=None, n_samples=3500, sort=True):
	    """
	    Compute permutation feature importances for scikit-learn models using
	    a validation set.

	    Given a Classifier or Regressor in model
	    and validation X and y data, return a data frame with columns
	    Feature and Importance sorted in reverse order by importance.
	    The validation data is needed to compute model performance
	    measures (accuracy or R^2). The model is not retrained.

	    You can pass in a list with a subset of features interesting to you.
	    All unmentioned features will be grouped together into a single meta-feature
	    on the graph. You can also pass in a list that has sublists like:
	    [['latitude', 'longitude'], 'price', 'bedrooms']. Each string or sublist
	    will be permuted together as a feature or meta-feature; the drop in
	    overall accuracy of the model is the relative importance.

	    The model.score() method is called to measure accuracy drops.

	    This version that computes accuracy drops with the validation set
	    is much faster than the OOB, cross validation, or drop column
	    versions. The OOB version is a less vectorized because it needs to dig
	    into the trees to get out of examples. The cross validation and drop column
	    versions need to do retraining and are necessarily much slower.

	    This function used OOB not validation sets in 1.0.5; switched to faster
	    test set version for 1.0.6. (breaking API change)

	    :param model: The scikit model fit to training data
	    :param X_valid: Data frame with feature vectors of the validation set
	    :param y_valid: Series with target variable of validation set
	    :param features: The list of features to show in importance graph.
	                     These can be strings (column names) or lists of column
	                     names. E.g., features = ['bathrooms', ['latitude', 'longitude']].
	                     Feature groups can overlap, with features appearing in multiple.
	    :param n_samples: How many records of the validation set to use
	                      to compute permutation importance. The default is
	                      3500, which we arrived at by experiment over a few data sets.
	                      As we cannot be sure how all data sets will react,
	                      you can pass in whatever sample size you want. Pass in -1
	                      to mean entire validation set. Our experiments show that
	                      not too many records are needed to get an accurate picture of
	                      feature importance.
	    return: A data frame with Feature, Importance columns

	    SAMPLE CODE

	    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
	    X_train, y_train = ..., ...
	    X_valid, y_valid = ..., ...
	    rf.fit(X_train, y_train)
	    imp = importances(model, X_valid, y_valid)
	    """
	    def flatten(features):
	        all_features = set()
	        for sublist in features:
	            if isinstance(sublist, str):
	                all_features.add(sublist)
	            else:
	                for item in sublist:
	                    all_features.add(item)
	        return all_features

	    if not features:
	        # each feature in its own group
	        features = X_valid.columns.values
	    else:
	        req_feature_set = flatten(features)
	        model_feature_set = set(X_valid.columns.values)
	        # any features left over?
	        other_feature_set = model_feature_set.difference(req_feature_set)
	        if len(other_feature_set) > 0:
	            # if leftovers, we need group together as single new feature
	            features.append(list(other_feature_set))

	    if n_samples < 0: n_samples = len(X_valid)
	    n_samples = min(n_samples, len(X_valid))
	    if n_samples < len(X_valid):
	        ix = np.random.choice(len(X_valid), n_samples)
	        X_valid = X_valid.iloc[ix].copy(deep=False)  # shallow copy
	        y_valid = y_valid.iloc[ix].copy(deep=False)
	    else:
	        X_valid = X_valid.copy(deep=False)  # we're modifying columns

	    baseline = model.score(X_valid, y_valid)
	    imp = []
	    for group in features:
	        if isinstance(group, str):
	            save = X_valid[group].copy()
	            X_valid[group] = np.random.permutation(X_valid[group])
	            m = model.score(X_valid, y_valid)
	            X_valid[group] = save
	        else:
	            save = {}
	            for col in group:
	                save[col] = X_valid[col].copy()
	            for col in group:
	                X_valid[col] = np.random.permutation(X_valid[col])
	            m = model.score(X_valid, y_valid)
	            for col in group:
	                X_valid[col] = save[col]
	        imp.append(baseline - m)

	    # Convert and groups/lists into string column names
	    labels = []
	    for col in features:
	        if isinstance(col, list):
	            labels.append('\n'.join(col))
	        else:
	            labels.append(col)

	    I = pd.DataFrame(data={'Feature': labels, 'Importance': np.array(imp)})
	    I = I.set_index('Feature')
	    if sort:
	        I = I.sort_values('Importance', ascending=True)
	    return I


	########################
	# Classification Stuff #
	########################
	def do_classification(model=model, 
						  train=train, 
						  valid=valid, 
						  y_train=y_train,
						  y_valid=y_valid):
	################
	# Useful Funcs #
	################
		def pred_to_hard(val, arr):
		    arr[arr > val] = 1
		    arr[arr <= val] = 0
		    return arr


		def hard_results(preds, target, value):
		    hard = pred_to_hard(value, preds.copy())
		    tp_ = np.sum(np.logical_and((hard == 1), (y_valid.values == 1)))
		    tn_ = np.sum(np.logical_and((hard == 0), (y_valid.values == 0)))
		    fp_ = np.sum(np.logical_and((hard == 1), (y_valid.values == 0)))
		    fn_ = np.sum(np.logical_and((hard == 0), (y_valid.values == 1)))
		    return (tp_,tn_,fp_,fn_)


		def hard_pred_df(data, target, model):
		    df = pd.DataFrame(columns=['Break','TP','TN', 'FP', 'FN','F1'])
		    preds = model.predict_proba(data)[:,1]
		    for i,v in enumerate(np.arange(0,1.001,.001)):
		        v = round(v,3)
		        hard = pred_to_hard(v, preds.copy())
		        tp_ = np.sum(np.logical_and((hard == 1), (y_valid.values == 1)))
		        tn_ = np.sum(np.logical_and((hard == 0), (y_valid.values == 0)))
		        fp_ = np.sum(np.logical_and((hard == 1), (y_valid.values == 0)))
		        fn_ = np.sum(np.logical_and((hard == 0), (y_valid.values == 1)))
		        f1_ = f1_score(y_valid,hard)
		        df.loc[i] = {
		                     'Break':v,
		                     'TP':tp_,
		                     'TN':tn_,
		                     'FP':fp_,
		                     'FN':fn_,
		                     'F1':f1_
		                    }
		    return df


		hard_pred = hard_pred_df(valid,y_valid,model)
		breaks = list(hard_pred['Break'])
		imp = importances(model, valid,y_valid,n_samples=50_000)
		slider_labels = {}
		for i,x in enumerate(breaks):
		    if i%100 ==0:
		        slider_labels[round(x,1)]=round(x,1) 

		preds = model.predict_proba(valid)[:,1]
		bgcolor = 'rbg(255,255,255)'
		fontcolor = 'rgb(30,30,60)'
		colorscale = [[0,'rgb(200,200,200)'],[1,'rgb(50,50,200)']]

		print('Hard Predictions Loaded!')

		app.layout = html.Div([
		    html.Div([
		        dcc.Slider(id='pred-slider',
		                    min=0,
		                    max=1,
		                    step=0.001,
		                    value=hard_pred.loc[np.argmax(hard_pred.F1)].Break,
		                    marks = slider_labels,
		                   )],
		        style={'padding':'1em',
		              }
		        ),
		    html.Div([
		        dcc.Graph(id = 'metric-table')],
		        style={'padding':'1em'}
		    ),
		    html.Div([
		        dcc.Graph(id='auc-graph'),
		        dcc.Graph(id='precision-graph')
		        ],
		        style={'display':'inline-block',
		               'width':'49%'}
		    ),
		    html.Div([
		        dcc.Graph(id = 'heatmap-graph'),
		        dcc.Graph(id = 'rfpimp',
		          figure={
		              'data':[ go.Bar(
		              x=list(imp.Importance),
		              y=imp.index,
		              orientation = 'h'
		              )],
		          'layout':go.Layout(
		              title = 'Feature Importances',
		              xaxis = {
		                       'fixedrange':True,
		                       'range':[0,1]
		                      },
		              yaxis = {
		                       'fixedrange':True
		              },
		              paper_bgcolor = bgcolor,
		              plot_bgcolor=bgcolor,
		              font=dict(family='helvetica', 
		                                 size=14, 
		                                 color=fontcolor)
		              )
		          }
		        )
		    ],
		        style={'display':'inline-block',
		               'width':'49%',
		               'vertical-align':'top'
		              }
		    )
		],
		    style={}
		)


		def get_annotations(tp, tn, fp, fn):
		    annot_font = {"family":'helvetica', "size":16,"color":"white"}
		    annotations=[
		    dict(x=1.0,
		         y=1.0,
		         text= 'TN '+str(fn), 
		         showarrow=False, 
		         font= annot_font
		        ), 
		    dict(x=1.0, 
		         y=0,
		         text='FN '+str(tn), 
		         showarrow=False,
		         font= annot_font
		        ),
		    dict(x=0,
		         y=0,
		         text= 'TP '+str(tp), 
		         showarrow=False,
		         font= annot_font
		        ), 
		    dict(x=0, 
		         y=1.0,
		         text= 'FP '+str(fp), 
		         showarrow=False,
		         font= annot_font
		        )
		    ]
		    return annotations


		@app.callback(
		              dash.dependencies.Output('auc-graph','figure'),
		              [dash.dependencies.Input('pred-slider','value')]
		             )
		def update_auc(value):
		    res = hard_pred.query('Break == @value')
		    return {
		        'data':[go.Scatter(
		                    x=hard_pred.FP,
		                    y=hard_pred.TP,
		                    mode='markers',
		                    marker = {'opacity':.5,
		                              'color':'rgb(0,100,200)'},
		                    name = 'all TP/FP'
		                ),
		                go.Scatter(
		                    x = res.FP,
		                    y = res.TP*1.05,
		                    mode = 'markers',
		                    marker = {'symbol':'triangle-down',
		                              'size':15,
		                              'opacity':1, 
		                              'color':'rgb(22, 199, 229)'},
		                    name = value
		               )],
		        'layout': go.Layout(
		            title = 'AUC Curve',
		            xaxis = {'title' :'FP',
		                     'fixedrange':True,
		                     'range':[0,np.max(hard_pred.FP)*1.0]
		                    },
		            yaxis = {'title':'TP',
		                     'fixedrange':True,
		                     'range':[0,np.max(hard_pred.TP)*1.05]
		                    },
		            paper_bgcolor = bgcolor,
		            plot_bgcolor=bgcolor,
		            font=dict(family='helvetica', 
		                             size=14, 
		                             color=fontcolor)        
		        )
		    }


		@app.callback(
		              dash.dependencies.Output('metric-table','figure'),
		              [dash.dependencies.Input('pred-slider','value')]
		             )
		def update_table(value):
		    res = hard_pred.query('Break == @value')
		    return {
		        'data':[go.Table(
		            type = 'table',
		            header = dict(values = ['Value',
		            						'TP', 
		            						'TN', 
		            						'FP', 
		            						'FN',
		            						'F1-Score']),
		            cells = dict(values = [[value], 
		            					   [res.TP],
		            					   [res.TN], 
		            					   [res.FP], 
		            					   [res.FN], 
		            					   [round(res.F1,3)]]))
		               ],
		        'layout':go.Layout(
		                    height = 275,
		                    #width = 500,
		                    paper_bgcolor = bgcolor,
		                    plot_bgcolor=bgcolor,
		                    font=dict(family='helvetica', 
		                             size=14, 
		                             color=fontcolor),   
		        )
		    }


		@app.callback(
		              dash.dependencies.Output('precision-graph','figure'),
		              [dash.dependencies.Input('pred-slider','value')]
		             )
		def update_precision(value):
		    res = hard_pred.query('Break == @value')
		    return {
		        'data':[go.Scatter(
		            x = hard_pred.TP/(hard_pred.TP+hard_pred.FN),
		            y = hard_pred.TP/(hard_pred.TP+hard_pred.FP),
		            mode = 'markers',
		            marker = {'opacity':.5,
		                      'color':'rgb(0,100,200)'},
		            name = 'totals'),
		            go.Scatter(
		            # recall TP / (TP + FN)
		            x=res.TP/(res.TP+res.FN),
		            # precision TP / (TP + FP)
		            y=(res.TP/(res.TP+res.FP))*1.05,
		            mode='markers',
		            marker = {'symbol':'triangle-down',
		                      'size':15,
		                      "opacity":1, 
		                      "color":'rgb(22, 199, 229)'},
		            name = value
		            )
		            ],
		        'layout': go.Layout(
		            height=400,
		            title = 'Precision-Recall Graph',
		            xaxis = {'title' :'Recall',
		                     'range':[0,1],
		                     'fixedrange':True
		                    },
		            yaxis = {'title':'Precision',
		                     'range':[0,1],
		                     'fixedrange':True
		                    },
		            paper_bgcolor = bgcolor,
		            plot_bgcolor=bgcolor,
		            font=dict(family='helvetica', 
		                             size=14, 
		                             color=fontcolor)
		        )
		    }


		@app.callback(
		              dash.dependencies.Output('heatmap-graph','figure'),
		              [dash.dependencies.Input('pred-slider','value')]
		             )
		def update_heatmap(value):
		    res = hard_pred.query('Break == @value')
		    return {
		        'data':[go.Heatmap(

		            x=['Positive', 'Negative'],
		            y=['Positive', 'Negative'],
		            z = [[int(res.TP), int(res.FN)], [int(res.FP), int(res.TN)]],
		            colorscale = colorscale,
		        )],
		        'layout': go.Layout(
		            title = 'Confusion Matrix',
		            paper_bgcolor = bgcolor,
		            plot_bgcolor = bgcolor,
		            xaxis = {'fixedrange':True,
		                     'title':'Predicted'
		                    },
		            yaxis = {'fixedrange':True,
		                     'autorange':'reversed',
		                     'title':'TruValue'
		                    },
		            font=dict(family='helvetica', 
		                             size=14, 
		                             color=fontcolor),
		            annotations=get_annotations(int(res.TP), 
		                                        int(res.FN), 
		                                        int(res.FP), 
		                                        int(res.TN))
		            )
		        }

		app.run_server(debug=False, port=port)

	####################
	# Regression Stuff #
	####################
	def do_regression(model=model, 
					  train=train, 
					  valid=valid, 
					  y_train=y_train,
					  y_valid=y_valid):
		################
		# Useful Funcs #
		################
		def mean_absolute_percentage_error(y_true, y_pred): 
			return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


		def mean_squared_percentage_error(y_true, y_pred): 
		    return np.mean((y_true - y_pred)**2 / y_true) * 100


		def get_regression_metrics(model, train, y_train, valid, y_valid):
		    y_pred = model.predict(valid)

		    validation = {}

		    y_valid = y_valid.values

		    validation["msle"] = mean_squared_log_error(y_valid, y_pred)
		    validation["mae"] = mean_absolute_error(y_valid, y_pred)
		    validation["mse"] = mean_squared_error(y_valid, y_pred)
		    validation["mape"] = mean_absolute_percentage_error(y_valid, y_pred)
		    validation["r2"] = r2_score(y_valid, y_pred)
		    validation["rmsle"] = np.sqrt(mean_squared_log_error(y_valid, y_pred))
		    validation["rmse"] = np.sqrt(mean_squared_error(y_valid, y_pred))
		    validation["preds"] = y_pred
		    
		    y_pred_train = model.predict(train)
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


		valid_metrics, train_metrics = get_regression_metrics(model, train, y_train, valid, y_valid)
		df_test = pd.DataFrame(valid_metrics).head(1).drop(["preds"], axis = 1).apply(lambda x:round(x, 4))
		imp = importances(model, valid,y_valid,n_samples=50_000)
		print('Metrics Loaded!')

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

		bgcolor = 'rbg(255,255,255)'
		fontcolor = 'rgb(30,30,60)'
		colorscale =[[0.0, 'rgb(255,255,255)'],[0.0001, 'rgb(224,224,224)'], [1.0,'rgb(0, 0, 255)']]
		valid_rows = valid.shape[0]
		train_mean = np.mean(y_train)
		valid_mean = np.mean(y_valid)
		color_bar_nums = [0]
		color_bar_list = ["min"]
		annot_font = {"family":'helvetica', "size":20,"color":"white"}
		print('Plot Preprocessing Complete!')

		# plot a subsample
		sampSize = 5_000
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
		                                           colorscale=colorscale,
		                                           colorbar={"yanchor":"top", 
		                                                     "len":.5,
		                                                    #'ticktext':color_bar_list,
		                                                    #'tickvals':color_bar_nums,
		                                                    #'tickmode':'array'
		                                                    },
		                                           xaxis='x2',
		                                           yaxis='y2'
		                                           )

		trace_hist2 = graph_objs.Histogram2dcontour(x = train_metrics["preds"][iTrainSamp], 
		                                            y = y_train, 
		                                            name= "Training set", 
		                                            ncontours=100,
		                                            nbinsx=100,
		                                            nbinsy=100, 
		                                            autocontour= False,
		                                            contours = {"showlines":False},
		                                            colorscale=colorscale,
		                                            showscale=False,
		                                            yaxis='y2'
		                                            )

		trace = go.Scatter(x = valid_metrics["preds"][iValidSamp], 
		                   y= y_valid, 
		                   mode='markers',
		                   marker = {"opacity":0.7, 
		                             "color":'rgb(22, 199, 229)'},
		                   textfont=dict(family='helvetica', 
		                                 size=14, 
		                                 color='rgb(193, 192, 191)'),
		                   showlegend=True, 
		                   name = "Validation",
		                   xaxis='x2',
		                   )

		trace2 = go.Scatter(x = train_metrics["preds"][iTrainSamp], 
							y = y_train, 
		                    mode = 'markers',
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
		                             color=fontcolor),
		                   xaxis={'title': 'Predicted', 
		                          "domain":[0, 0.45]},
		                   yaxis={'title': 'True', 
		                          "domain":[.55,1]},
		                   xaxis2={'title': 'Predicted',
		                           "domain":[0.55, 1],
		                           'anchor':'y_den_v'},
		                   yaxis2={"title":"True", 
		                           'domain':[0,.45]},
		                   paper_bgcolor = bgcolor,
		                   plot_bgcolor = bgcolor,
		                   autosize=False,
		                   width=900,
		                   height=900,
		                   annotations=annotations)
		
		app.layout = html.Div(children=[
		    html.H2(children='Regression Metrics',
		            style={'font-family':'helvetica'}),
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
		                      "background-color": "white",
		                      "color": "black",
		                      'border-color': "gray" ,
		                      'font-family':'helvetica'
		                     }
		    ),
		    dcc.Graph(
		        id='example-graph',
		        figure={
		            'data': [trace_hist, trace_hist2, trace, trace2],
		            'layout': layout
		        }
		    ),
	    	dcc.Graph(id = 'rfpimp',
	        	figure={
		              'data':[ go.Bar(
			              x=list(imp.Importance),
			              y=imp.index,
			              orientation = 'h'
		              )],
			          'layout':go.Layout(
			              title = 'Feature Importances',
			              xaxis = {
			                       'fixedrange':True,
			                       'range':[0,1],
			                       'domain':[.1,1]
			                      },
			              yaxis = {
			                       'fixedrange':True
			              },
			              paper_bgcolor = bgcolor,
			              plot_bgcolor=bgcolor,
			              font=dict(family='helvetica', 
			                                 size=14, 
			                                 color=fontcolor),
			              autosize=False,
			              width=900
			              )
	          	}
	        )
		])
		print('Reticulating Splines!')

		app.run_server(debug=False, port=port)

	model_type = str(type(model)).lower()
	if 'randomforestclassifier' in model_type:
		do_classification()
	elif 'randomforestregressor' in model_type:
		do_regression()
	else:
		print('Unsupported Model Type!')
