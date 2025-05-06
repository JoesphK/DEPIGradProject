import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('DataSet/CSV/balanced.csv')

# Data cleaning
df = df.dropna(subset=['DiagnosedWithDiabetes'])  # Remove rows where diabetes status is missing
df['DiagnosedWithDiabetes'] = df['DiagnosedWithDiabetes'].astype(int)

# Initialize the app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Diabetes Prediction Dashboard", style={'text-align': 'center'}),
    
    # Data Overview Section
    html.Div([
        html.H2("Data Overview"),
        dcc.Markdown("""
        This dataset contains health metrics for **{} individuals** with **{}% diagnosed with diabetes**.
        Explore the relationships between different health factors and diabetes diagnosis.
        """.format(len(df), round(df['DiagnosedWithDiabetes'].mean()*100, 1))),
        
        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': col, 'value': col} for col in df.select_dtypes(include=np.number).columns],
                value='AgeInYears'
            ),
            dcc.RadioItems(
                id='xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                inline=True
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': col, 'value': col} for col in df.select_dtypes(include=np.number).columns],
                value='BodyMassIndex'
            ),
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                inline=True
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        
        dcc.Graph(id='indicator-graphic'),
        
        html.Div([
            dcc.Dropdown(
                id='hist-column',
                options=[{'label': col, 'value': col} for col in df.select_dtypes(include=np.number).columns],
                value='BodyMassIndex'
            ),
            dcc.Graph(id='hist-graphic')
        ])
    ], style={'padding': '20px', 'margin': '20px', 'border': '1px solid #ddd', 'border-radius': '5px'}),
    
    # Risk Factor Analysis
    html.Div([
        html.H2("Risk Factor Analysis"),
        dcc.Markdown("""
        Explore how different factors correlate with diabetes diagnosis.
        Select factors to compare their distributions (up to 3 recommended for clear visualization).
        """),
        
        dcc.Dropdown(
            id='risk-factor-selector',
            options=[{'label': col, 'value': col} for col in [
                'BodyMassIndex', 'AgeInYears', 'WaistCircumferenceCm', 
                'SerumGlucose', 'HighDensityLipoprotein', 'HighSensitivityCRP'
            ]],
            value=['BodyMassIndex', 'AgeInYears'],
            multi=True
        ),
        
        dcc.Graph(id='risk-factor-plot')
    ], style={'padding': '20px', 'margin': '20px', 'border': '1px solid #ddd', 'border-radius': '5px'}),
    
    # Predictive Model
    html.Div([
        html.H2("Diabetes Prediction Model"),
        dcc.Markdown("""
        This model predicts diabetes risk based on health metrics. 
        Adjust the feature importance threshold to see how it affects model performance.
        """),
        
        dcc.Slider(
            id='feature-threshold',
            min=0,
            max=0.2,
            step=0.01,
            value=0.05,
            marks={i/100: str(i/100) for i in range(0, 21, 5)}
        ),
        
        html.Div([
            dcc.Graph(id='feature-importance-plot'),
            dcc.Graph(id='confusion-matrix-plot')
        ], style={'display': 'flex', 'flex-direction': 'row'})
    ], style={'padding': '20px', 'margin': '20px', 'border': '1px solid #ddd', 'border-radius': '5px'}),
    
    # Individual Risk Assessment
    html.Div([
        html.H2("Individual Risk Assessment"),
        dcc.Markdown("""
        Enter your health metrics to get a personalized diabetes risk assessment.
        """),
        
        html.Div([
            html.Label('Age'),
            dcc.Input(id='age-input', type='number', value=50),
            
            html.Label('BMI'),
            dcc.Input(id='bmi-input', type='number', value=25),
            
            html.Label('Waist Circumference (cm)'),
            dcc.Input(id='waist-input', type='number', value=90),
            
            html.Label('Serum Glucose'),
            dcc.Input(id='glucose-input', type='number', value=100),
            
            html.Button('Calculate Risk', id='calculate-button', n_clicks=0)
        ], style={'columnCount': 2}),
        
        html.Div(id='risk-output')
    ], style={'padding': '20px', 'margin': '20px', 'border': '1px solid #ddd', 'border-radius': '5px'})
])

# Callbacks for interactive components
@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('xaxis-type', 'value'),
    Input('yaxis-type', 'value'))
def update_graph(xaxis_column_name, yaxis_column_name, xaxis_type, yaxis_type):
    fig = px.scatter(
        df, x=xaxis_column_name, y=yaxis_column_name,
        color='DiagnosedWithDiabetes',
        hover_data=['AgeInYears', 'BodyMassIndex', 'SerumGlucose'],
        title=f"{yaxis_column_name} vs {xaxis_column_name}"
    )
    
    fig.update_layout(
        xaxis_type='linear' if xaxis_type == 'Linear' else 'log',
        yaxis_type='linear' if yaxis_type == 'Linear' else 'log'
    )
    
    return fig

@app.callback(
    Output('hist-graphic', 'figure'),
    Input('hist-column', 'value'))
def update_hist(column_name):
    fig = px.histogram(
        df, x=column_name, color='DiagnosedWithDiabetes',
        marginal="box", barmode="overlay",
        title=f"Distribution of {column_name}"
    )
    return fig

@app.callback(
    Output('risk-factor-plot', 'figure'),
    Input('risk-factor-selector', 'value'))
def update_risk_factors(selected_factors):
    if not selected_factors or len(selected_factors) > 3:  # Limit to 3 factors for clarity
        return go.Figure()
    
    fig = px.box(
        df, 
        y=selected_factors, 
        color='DiagnosedWithDiabetes',
        facet_col='variable',
        facet_col_wrap=min(3, len(selected_factors)),
        boxmode="group"
    )
    
    fig.update_layout(
        title="Comparison of Selected Risk Factors",
        height=400 if len(selected_factors) <= 3 else 600
    )
    
    return fig

@app.callback(
    [Output('feature-importance-plot', 'figure'),
     Output('confusion-matrix-plot', 'figure')],
    Input('feature-threshold', 'value'))
def update_model(threshold):
    # Prepare data
    features = df.select_dtypes(include=np.number).columns.drop('DiagnosedWithDiabetes')
    X = df[features].fillna(df[features].median())
    y = df['DiagnosedWithDiabetes']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Feature importance
    importances = model.feature_importances_
    important_features = features[importances > threshold]
    
    # Retrain with important features only
    if len(important_features) > 0:
        X_train_imp = X_train_scaled[:, importances > threshold]
        X_test_imp = X_test_scaled[:, importances > threshold]
        
        model_imp = RandomForestClassifier(random_state=42)
        model_imp.fit(X_train_imp, y_train)
        
        y_pred = model_imp.predict(X_test_imp)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
    else:
        acc = 0
        cm = np.array([[0, 0], [0, 0]])
    
    # Feature importance plot
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    fig1 = px.bar(
        importance_df.head(10),
        x='Importance', y='Feature',
        title=f'Top Feature Importances (Threshold: {threshold})'
    )
    
    # Confusion matrix plot
    fig2 = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['No Diabetes', 'Diabetes'],
        y=['No Diabetes', 'Diabetes'],
        title=f'Confusion Matrix (Accuracy: {acc:.2f})',
        text_auto=True
    )
    
    return fig1, fig2

@app.callback(
    Output('risk-output', 'children'),
    Input('calculate-button', 'n_clicks'),
    Input('age-input', 'value'),
    Input('bmi-input', 'value'),
    Input('waist-input', 'value'),
    Input('glucose-input', 'value'))
def calculate_risk(n_clicks, age, bmi, waist, glucose):
    if n_clicks == 0:
        return ""
    
    # Prepare input data
    input_data = pd.DataFrame({
        'AgeInYears': [age],
        'BodyMassIndex': [bmi],
        'WaistCircumferenceCm': [waist],
        'SerumGlucose': [glucose]
    })
    
    # Get median values for other features
    features = df.select_dtypes(include=np.number).columns.drop('DiagnosedWithDiabetes')
    for col in features:
        if col not in input_data.columns:
            input_data[col] = df[col].median()
    
    # Reorder columns to match training data
    input_data = input_data[features]
    
    # Scale data
    scaler = StandardScaler()
    X = df[features].fillna(df[features].median())
    scaler.fit(X)
    input_scaled = scaler.transform(input_data)
    
    # Train model on full dataset for prediction
    model = RandomForestClassifier(random_state=42)
    model.fit(scaler.transform(X.fillna(X.median())), df['DiagnosedWithDiabetes'])
    
    # Predict probability
    proba = model.predict_proba(input_scaled)[0][1]
    
    # Risk interpretation
    if proba < 0.2:
        risk_level = "Low"
        color = "green"
    elif proba < 0.5:
        risk_level = "Moderate"
        color = "orange"
    else:
        risk_level = "High"
        color = "red"
    
    return html.Div([
        html.H3(f"Diabetes Risk: {risk_level}"),
        html.P(f"Probability: {proba:.1%}"),
        html.P("Risk Factors:"),
        html.Ul([
            html.Li(f"Age: {age} years"),
            html.Li(f"BMI: {bmi} ({'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'})"),
            html.Li(f"Waist Circumference: {waist} cm ({'Normal' if waist < 94 else 'High' if waist < 102 else 'Very High'})"),
            html.Li(f"Glucose Level: {glucose} ({'Normal' if glucose < 100 else 'Prediabetes' if glucose < 126 else 'Diabetes'})")
        ])
    ], style={'color': color, 'margin-top': '20px'})

if __name__ == '__main__':
    app.run(debug=True)