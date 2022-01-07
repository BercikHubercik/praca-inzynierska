# -*- coding: utf-8 -*-
import dash
from dash import dcc, Output, Input, State
from dash import html
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn import svm
from dash import dash_table
from func import Classifier
import plotly.graph_objects as go
import spacy
nlp = spacy.load("pl_core_news_sm")



df = pd.read_pickle(r'..\data\data_df.pkl')
emotions = ['Sentyment', 'Lubienie', 'Wstręt', 'Złość', 'Strach', 'Zaskoczenie', 'Oczekiwanie', 'Radość', 'Smutek']

models = ['NaiveBayes', 'SVM']
features = ['Cały tekst', 'Tekst bez SW']
text_names = list(df.index.values)
text_names.insert(0, 'Własny tekst')
app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Aplikacja rozpoznająca emocje'),
    html.Div([
        html.H2(children='Wybierz parametry'),
        html.Div([
            html.H4(children='Wybierz klasyfikator'),
            dcc.Dropdown(
                id='clf_type',
                options=[{'label': i, 'value': i} for i in models]
            )
        ]),
        html.Div([
            html.H4(children='Czy chcesz użyc modyfikacji do regrecji porządkowej'),
            dcc.RadioItems(options=[
                                {'label': 'Tak', 'value': 'True'},
                                {'label': 'Nie', 'value': 'False'}
                            ],
                           value='False',
                           labelStyle={'display': 'inline-block'},
                           id='use_ordinal'
            )
        ]),
        html.Div([
            html.H4(children='Wybierz cechy tekstu użyte do predykcji'),
            dcc.Dropdown(
                id='features_type',
                options=[{'label': i, 'value': i} for i in features]
            )
        ]),
        html.Div([
            html.H2(children='Wybierz tekst'),
            dcc.Dropdown(
                id='text_name',
                options=[{'label': i, 'value': i} for i in text_names]
            ),
            html.H4(children='lub wpisz własny tekst poniżej'),
            dcc.Textarea(
                id='text_input'
            )
        ])
    ]),
    dcc.Tabs(id="tabs_performance_prediction", value='tab_2_evaluation', children=[
        dcc.Tab(label='Przewidziane metryki tekstu', value='tab_1_prediction'),
        dcc.Tab(label='Ocena działania modelu', value='tab_2_evaluation')
    ]),
    html.Div(id='tabs_content')
])


@app.callback(
    Output('tabs_content', 'children'),
    Input('tabs_performance_prediction', 'value'),
    Input('clf_type', 'value'),
    Input('text_name', 'value'),
    State('text_input', 'value'),
    Input('use_ordinal', 'value'),
    Input('features_type', 'value'))
def render_tabs(tab, clf_type, text_name, text_input, use_ordinal, features_type):

    if features_type == 'Cały tekst':
        features_type = 1
    else:
        features_type = 2

    # text choosing
    if text_name in list(df.index.values):
        text = df['OriginalText'][text_name]
    else:
        text = text_input
        if text is None:
            text = 'Wpisz własny tekst'

    # ordinal mod choosing
    if use_ordinal == 'True':
        use_ordinal = True
    else:
        use_ordinal = False
    # model choosing

    clf = MultinomialNB()
    if clf_type == 'NaiveBayes':
        pass
    elif clf_type == 'SVM':
        if use_ordinal:
            clf = svm.SVC(probability=True)
        else:
            clf = svm.SVC()

    # generate models metrics
    boss = Classifier(clf, features_type, df, use_ordinal)
    acc_dict, prec_dict, rec_dict, f1_dict, conf_matrix_dict = boss.calc_acc()

    # predict emotions for text
    pred_dict = boss.predict_single_val(text) #test value
    acc_table = go.Figure(data=[go.Table(header=dict(values=['Emotion', 'Accuracy', 'Precision', 'Recall', 'F1']),
                                   cells=dict(values=[list(acc_dict.keys()), list(acc_dict.values()), list(prec_dict.values()), list(rec_dict.values()), list(f1_dict.values())]))
                          ])

    # create figures for confusion matrix
    figures = []
    for emo in conf_matrix_dict.keys():
        fig = px.imshow(list(conf_matrix_dict[emo]),
                        labels=dict(x="Przewidziana wartość", y="Prawdziwa wartość", color="Liczba przewidzianych wartości"),
                        title=str(emo)
                        )

        figures.append(fig)

    # render tabs ########################################################################
    if tab == 'tab_1_prediction':
        return html.Div([
            html.Div([
                html.P(text)
            ]),
            html.Div([
                html.H4(children=str(list(pred_dict.keys())[0]) + ' - Nastrój narratora był'),
                dcc.Slider(min=-1,
                           max=1,
                           step=None,
                           marks={-1: 'Negatywny',
                                  0: 'Neutralny',
                                  1: 'Pozytywny'
                                  },
                           value=int(list(pred_dict.values())[0])),

                html.H4(children=str(list(pred_dict.keys())[1])),
                dcc.Slider(min=0,
                           max=4,
                           step=None,
                           marks={0: '0',
                                  1: '1',
                                  2: '2',
                                  3: '3',
                                  4: '4'
                                  },
                           value=int(list(pred_dict.values())[1])),

                html.H4(children=str(list(pred_dict.keys())[2])),
                dcc.Slider(min=0,
                           max=4,
                           step=None,
                           marks={0: '0',
                                  1: '1',
                                  2: '2',
                                  3: '3',
                                  4: '4'
                                  },
                           value=int(list(pred_dict.values())[2])),

                html.H4(children=str(list(pred_dict.keys())[3])),
                dcc.Slider(min=0,
                           max=4,
                           step=None,
                           marks={0: '0',
                                  1: '1',
                                  2: '2',
                                  3: '3',
                                  4: '4'
                                  },
                           value=int(list(pred_dict.values())[3])),

                html.H4(children=str(list(pred_dict.keys())[4])),
                dcc.Slider(min=0,
                           max=4,
                           step=None,
                           marks={0: '0',
                                  1: '1',
                                  2: '2',
                                  3: '3',
                                  4: '4'
                                  },
                           value=int(list(pred_dict.values())[4])),

                html.H4(children=str(list(pred_dict.keys())[5])),
                dcc.Slider(min=0,
                           max=4,
                           step=None,
                           marks={0: '0',
                                  1: '1',
                                  2: '2',
                                  3: '3',
                                  4: '4'
                                  },
                           value=int(list(pred_dict.values())[5])),

                html.H4(children=str(list(pred_dict.keys())[6])),
                dcc.Slider(min=0,
                           max=4,
                           step=None,
                           marks={0: '0',
                                  1: '1',
                                  2: '2',
                                  3: '3',
                                  4: '4'
                                  },
                           value=int(list(pred_dict.values())[6])),

                html.H4(children=str(list(pred_dict.keys())[7])),
                dcc.Slider(min=0,
                           max=4,
                           step=None,
                           marks={0: '0',
                                  1: '1',
                                  2: '2',
                                  3: '3',
                                  4: '4'
                                  },
                           value=int(list(pred_dict.values())[7])),

                html.H4(children=str(list(pred_dict.keys())[8])),
                dcc.Slider(min=0,
                           max=4,
                           step=None,
                           marks={0: '0',
                                  1: '1',
                                  2: '2',
                                  3: '3',
                                  4: '4'
                                  },
                           value=int(list(pred_dict.values())[8]))

            ])

        ])
    elif tab == 'tab_2_evaluation':
        return html.Div(children=[
            #dash_table.DataTable(data=acc_df),
            dcc.Graph(figure=acc_table),
            dcc.Graph(figure=figures[0]),
            dcc.Graph(figure=figures[1]),
            dcc.Graph(figure=figures[2]),
            dcc.Graph(figure=figures[3]),
            dcc.Graph(figure=figures[4]),
            dcc.Graph(figure=figures[5]),
            dcc.Graph(figure=figures[6]),
            dcc.Graph(figure=figures[7]),
            dcc.Graph(figure=figures[8])
        ])


if __name__ == '__main__':
    app.run_server(debug=True)
