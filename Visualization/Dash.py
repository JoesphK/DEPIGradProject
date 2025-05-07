import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('Dataset/CSV/Diabetes.csv')

# Convert 'diabetic' to binary for analysis
df['diabetic_binary'] = df['diabetic'].apply(lambda x: 1 if x == 'Yes' else 0)

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Diabetes Risk Factors Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
    
    # Interactive controls
    html.Div([
        html.Label("Age Range Filter:", style={'fontWeight': 'bold'}),
        dcc.RangeSlider(
            id='age-slider',
            min=df['age'].min(),
            max=df['age'].max(),
            value=[df['age'].min(), df['age'].max()],
            marks={str(age): str(age) for age in range(df['age'].min(), df['age'].max()+1, 10)},
            step=1,
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'width': '90%', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': '#f8f9fa'}),
    
    # First row of visualizations
    html.Div([
        # Demographic Overview
        html.Div([
            dcc.Graph(id='gender-pie'),
            html.P("Gender distribution with diabetes status. Males in blue, females in pink. Diabetic portions in red.",
                  style={'textAlign': 'center', 'fontSize': '14px'})
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        # Age Distribution
        html.Div([
            dcc.Graph(id='age-hist'),
            html.P("Age distribution by diabetes status. Typically diabetes risk increases with age.",
                  style={'textAlign': 'center', 'fontSize': '14px'})
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
    ], style={'marginBottom': '20px'}),
    
    # Second row of visualizations
    html.Div([
        # BMI vs Glucose
        html.Div([
            dcc.Graph(id='bmi-glucose'),
            html.P("BMI vs glucose levels. Higher BMI often correlates with higher diabetes risk.",
                  style={'textAlign': 'center', 'fontSize': '14px'})
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        # Blood Pressure
        html.Div([
            dcc.Graph(id='bp-plot'),
            html.P("Blood pressure comparison. Hypertension often accompanies diabetes.",
                  style={'textAlign': 'center', 'fontSize': '14px'})
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
    ], style={'marginBottom': '20px'}),
    
    # Third row of visualizations
    html.Div([
        # Glucose Distribution
        html.Div([
            dcc.Graph(id='glucose-violin'),
            html.P("Glucose level distributions - the primary diagnostic marker for diabetes.",
                  style={'textAlign': 'center', 'fontSize': '14px'})
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        # Correlation Matrix
        html.Div([
            dcc.Graph(id='corr-matrix'),
            html.P("Correlation heatmap showing relationships between all numerical variables.",
                  style={'textAlign': 'center', 'fontSize': '14px'})
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
    ], style={'marginBottom': '20px'}),
    
    # Fourth row (single visualization)
    html.Div([
        # Family History Impact
        html.Div([
            dcc.Graph(id='family-history'),
            html.P("Family history of diabetes significantly increases individual risk.",
                  style={'textAlign': 'center', 'fontSize': '14px'})
        ], style={'width': '80%', 'margin': '0 auto', 'padding': '10px'})
    ])
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ffffff'})

# Callbacks for interactivity
@app.callback(
    [Output('gender-pie', 'figure'),
     Output('age-hist', 'figure'),
     Output('bmi-glucose', 'figure'),
     Output('bp-plot', 'figure'),
     Output('glucose-violin', 'figure'),
     Output('corr-matrix', 'figure'),
     Output('family-history', 'figure')],
    [Input('age-slider', 'value')]
)
def update_figures(age_range):
    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    
    # Gender pie chart with custom colors
    gender_pie = px.pie(filtered_df, 
                       names='gender', 
                       color='diabetic',
                       color_discrete_map={'Yes': '#d62728', 'No': '#2ca02c'},
                       title='Gender Distribution by Diabetes Status',
                       category_orders={'gender': ['Male', 'Female']},
                       hole=0.3)
    
    # Apply gender-specific colors and styling
    gender_pie.update_traces(
        marker=dict(colors=['#1f77b4', '#ff7f0e']),  # Blue for Male, Orange for Female
        textinfo='percent+label',
        pull=[0.1, 0],  # Slightly pull the slices apart
        opacity=0.9
    )
    gender_pie.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Age histogram
    age_hist = px.histogram(filtered_df, x='age', color='diabetic',
                           color_discrete_map={'Yes': '#d62728', 'No': '#2ca02c'},
                           barmode='overlay', 
                           title='Age Distribution by Diabetes Status',
                           nbins=20)
    age_hist.update_layout(
        yaxis_title="Count",
        xaxis_title="Age",
        legend_title="Diabetes Status"
    )
    
    # BMI vs Glucose
    bmi_glucose = px.scatter(filtered_df, x='bmi', y='glucose', color='diabetic',
                            color_discrete_map={'Yes': '#d62728', 'No': '#2ca02c'},
                            title='BMI vs Glucose Levels',
                            trendline="lowess",
                            hover_data=['age', 'gender'])
    bmi_glucose.update_layout(
        yaxis_title="Glucose Level",
        xaxis_title="Body Mass Index (BMI)",
        legend_title="Diabetes Status"
    )
    
    # Blood Pressure
    bp_plot = px.box(filtered_df, y=['systolic_bp', 'diastolic_bp'], color='diabetic',
                    color_discrete_map={'Yes': '#d62728', 'No': '#2ca02c'},
                    title='Blood Pressure by Diabetes Status')
    bp_plot.update_layout(
        yaxis_title="Blood Pressure (mmHg)",
        xaxis_title="Blood Pressure Type",
        legend_title="Diabetes Status"
    )
    
    # Glucose violin plot
    glucose_violin = px.violin(filtered_df, y='glucose', color='diabetic',
                              color_discrete_map={'Yes': '#d62728', 'No': '#2ca02c'},
                              box=True, points='all', 
                              title='Glucose Level Distribution')
    glucose_violin.update_layout(
        yaxis_title="Glucose Level",
        xaxis_title="Diabetes Status",
        showlegend=False
    )
    
    # Correlation matrix
    corr = filtered_df.select_dtypes(include=[np.number]).corr()
    corr_matrix = px.imshow(corr, 
                           text_auto=True, 
                           title='Correlation Matrix',
                           color_continuous_scale='RdBu',
                           zmin=-1, zmax=1)
    corr_matrix.update_layout(
        xaxis_title="Variables",
        yaxis_title="Variables"
    )
    
    # Family history
    family_history = px.bar(filtered_df, x='family_diabetes', color='diabetic',
                           color_discrete_map={'Yes': '#d62728', 'No': '#2ca02c'},
                           barmode='group', 
                           title='Diabetes Prevalence by Family History')
    family_history.update_layout(
        yaxis_title="Count",
        xaxis_title="Family History of Diabetes",
        legend_title="Diabetes Status",
        xaxis=dict(tickvals=[0, 1], ticktext=['No Family History', 'Family History']))
    
    return gender_pie, age_hist, bmi_glucose, bp_plot, glucose_violin, corr_matrix, family_history

if __name__ == '__main__':
    app.run(debug=True)