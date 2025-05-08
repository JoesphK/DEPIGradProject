import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Health Metrics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data


df = pd.read_csv('C:/DEPI_Project - 2025/DataSet/CSV/Diabetes.csv')

# Preprocess data
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['<30', '30-45', '45-60', '60+'])
df['bp_category'] = np.where((df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90), 'High', 'Normal')
df['glucose_category'] = np.where(df['glucose'] > 7, 'High', 'Normal')
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], 
                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Encode categorical variables
le = LabelEncoder()
df['diabetic_encoded'] = le.fit_transform(df['diabetic'])

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    
    age_groups = st.multiselect(
        "Age Groups",
        options=df['age_group'].unique(),
        default=df['age_group'].unique()
    )
    
    genders = st.multiselect(
        "Gender",
        options=df['gender'].unique(),
        default=df['gender'].unique()
    )
    
    diabetic_status = st.multiselect(
        "Diabetic Status",
        options=df['diabetic'].unique(),
        default=df['diabetic'].unique()
    )
    
    bp_filter = st.multiselect(
        "Blood Pressure Category",
        options=df['bp_category'].unique(),
        default=df['bp_category'].unique()
    )

# Apply filters
filtered_df = df[
    (df['age_group'].isin(age_groups)) &
    (df['gender'].isin(genders)) &
    (df['diabetic'].isin(diabetic_status)) &
    (df['bp_category'].isin(bp_filter))
]

# Main content
st.title("🏥 Health Metrics Dashboard")

# KPI cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Patients", len(filtered_df))
with col2:
    diabetic_percent = filtered_df['diabetic'].value_counts(normalize=True).get('Yes', 0) * 100
    st.metric("Diabetic Patients", f"{diabetic_percent:.1f}%")
with col3:
    hypertensive_percent = (filtered_df['hypertensive'].sum() / len(filtered_df)) * 100
    st.metric("Hypertensive Patients", f"{hypertensive_percent:.1f}%")
with col4:
    avg_bmi = filtered_df['bmi'].mean()
    st.metric("Average BMI", f"{avg_bmi:.1f}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visualizations", "Risk Factors", "Data"])

with tab1:
    st.subheader("Demographic Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        filtered_df['age_group'].value_counts().sort_index().plot(kind='bar', ax=ax)
        plt.title('Patients by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots()
        filtered_df['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        plt.title('Gender Distribution')
        st.pyplot(fig)
    
    st.subheader("Health Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_df, x='diabetic', y='glucose', ax=ax)
        plt.title('Glucose Levels by Diabetic Status')
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_df, x='diabetic', y='bmi', ax=ax)
        plt.title('BMI by Diabetic Status')
        st.pyplot(fig)
        
    with col3:
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_df, x='diabetic', y='age', ax=ax)
        plt.title('Age by Diabetic Status')
        st.pyplot(fig)

with tab2:
    st.subheader("Interactive Visualizations")
    
    chart_type = st.selectbox(
        "Select Chart Type",
        options=["Scatter Plot", "Histogram", "Violin Plot"]
    )
    
    if chart_type == "Scatter Plot":
        x_axis = st.selectbox("X-axis", options=filtered_df.select_dtypes(include=['int64', 'float64']).columns)
        y_axis = st.selectbox("Y-axis", options=filtered_df.select_dtypes(include=['int64', 'float64']).columns)
        color_by = st.selectbox("Color by", options=['None'] + list(filtered_df.select_dtypes(include=['object', 'category']).columns))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if color_by != 'None':
            sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, hue=color_by, ax=ax)
        else:
            sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)
        plt.title(f'{y_axis} vs {x_axis}')
        st.pyplot(fig)
        
    elif chart_type == "Histogram":
        column = st.selectbox("Select Column", options=filtered_df.select_dtypes(include=['int64', 'float64']).columns)
        bins = st.slider("Number of bins", 5, 50, 20)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=filtered_df, x=column, bins=bins, kde=True, ax=ax)
        plt.title(f'Distribution of {column}')
        st.pyplot(fig)
        
    elif chart_type == "Violin Plot":
        numeric_col = st.selectbox("Numeric Column", options=filtered_df.select_dtypes(include=['int64', 'float64']).columns)
        category_col = st.selectbox("Category Column", options=filtered_df.select_dtypes(include=['object', 'category']).columns)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=filtered_df, x=category_col, y=numeric_col, ax=ax)
        plt.title(f'Distribution of {numeric_col} by {category_col}')
        st.pyplot(fig)

with tab3:
    st.subheader("Risk Factor Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        sns.heatmap(filtered_df[['age', 'bmi', 'glucose', 'systolic_bp', 'diastolic_bp', 'diabetic_encoded']].corr(), 
                    annot=True, cmap='coolwarm', ax=ax)
        plt.title('Correlation Matrix')
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots()
        pd.crosstab(filtered_df['bmi_category'], filtered_df['diabetic']).plot(kind='bar', stacked=True, ax=ax)
        plt.title('Diabetes by BMI Category')
        plt.xlabel('BMI Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    st.subheader("Blood Pressure Analysis")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='systolic_bp', y='diastolic_bp', hue='diabetic', style='hypertensive', ax=ax)
    plt.axvline(x=140, color='r', linestyle='--')
    plt.axhline(y=90, color='r', linestyle='--')
    plt.title('Blood Pressure by Diabetic and Hypertensive Status')
    st.pyplot(fig)

with tab4:
    st.subheader("Raw Data")
    st.dataframe(filtered_df, use_container_width=True)
    
    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe(), use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name='filtered_health_data.csv',
        mime='text/csv'
    )