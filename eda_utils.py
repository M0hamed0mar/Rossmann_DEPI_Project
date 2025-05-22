# eda_utils.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
import plotly.io as pio
from io import StringIO

def data_overview_html(df, df_name="DataFrame"):
    if df.empty:
        return f"<h4>Overview of {df_name}</h4><p>DataFrame is empty.</p>"
    
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue().replace('\n', '<br>')
    
    html_output = f"<h4>Overview of {df_name}</h4>"
    html_output += f"<p><b>Shape:</b> {df.shape}</p>"
    html_output += f"<p><b>Columns:</b> {', '.join(list(df.columns))}</p>"
    html_output += "<h5>Data Info:</h5>"
    html_output += f"<div class='eda-info-box'>{info_str}</div>"
    html_output += "<h5>First 5 Rows:</h5>"
    try:
        table_html = df.head().to_html(classes=["table", "table-sm", "table-striped", "dataframe-table-inner"], border=0, escape=False, index=False) 
        html_output += f"<div class='table-responsive-wrapper'>{table_html}</div>"
    except Exception as e:
        html_output += f"<p>Error generating head: {e}</p>"
    return html_output

def data_statistics_html(df, df_name="DataFrame"):
    if df.empty:
        return f"<h4>Descriptive Statistics for {df_name}</h4><p>DataFrame is empty.</p>"
        
    html_output = f"<h4>Descriptive Statistics for {df_name}</h4>"
    try:
        numeric_desc = df.describe(include=[np.number]).transpose()
        object_desc = df.describe(include=['object', 'category', 'bool']).transpose()

        if not numeric_desc.empty:
            numeric_table = numeric_desc.to_html(classes=["table", "table-sm", "table-striped", "dataframe-table-inner"], border=0, float_format='{:,.2f}'.format)
            html_output += f"<h5>Numerical & Boolean-like Features:</h5><div class='table-responsive-wrapper'>{numeric_table}</div>"
        if not object_desc.empty:
             object_table = object_desc.to_html(classes=["table", "table-sm", "table-striped", "dataframe-table-inner"], border=0)
             html_output += f"<h5>Categorical/Object Features:</h5><div class='table-responsive-wrapper'>{object_table}</div>"
        
        html_output += "<h5>Unique Values per Column:</h5>"
        unique_counts_df = pd.DataFrame(df.nunique(), columns=['UniqueCount']).sort_values(by='UniqueCount')
        unique_counts_table = unique_counts_df.to_html(classes=["table", "table-sm", "table-striped", "dataframe-table-inner"], border=0)
        html_output += f"<div class='table-responsive-wrapper'>{unique_counts_table}</div>"
    except Exception as e:
        html_output += f"<p>Error generating statistics: {e}</p>"
    return html_output

def missing_values_report_html(df, df_name="DataFrame"):
    if df.empty:
        return f"<h4>Missing Values in {df_name}</h4><p>DataFrame is empty.</p>"
    html_output = f"<h4>Missing Values in {df_name}</h4>"
    try:
        missing_count = df.isnull().sum()
        if len(df) > 0:
            missing_percent = (missing_count / len(df)) * 100
        else:
            missing_percent = pd.Series([0.0] * len(df.columns), index=df.columns) 
            
        missing_data_df = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing %': missing_percent
        })
        missing_data_df = missing_data_df[missing_data_df['Missing Count'] > 0].sort_values(by='Missing %', ascending=False)
        
        if missing_data_df.empty:
            html_output += "<p>No missing values found.</p>"
        else:
            missing_table = missing_data_df.to_html(classes=["table", "table-sm", "table-striped", "dataframe-table-inner"], border=0, formatters={'Missing %': '{:,.2f}%'.format})
            html_output += f"<div class='table-responsive-wrapper'>{missing_table}</div>"
    except Exception as e:
        html_output += f"<p>Error generating missing values report: {e}</p>"
    return html_output

def plot_distribution_plotly_html(df, col, df_name=""):
    title_text = f'Distribution of {col}' + (f' in {df_name}' if df_name else '')
    if df.empty or col not in df.columns:
        return f"<div class='plot-container-placeholder'><h5>{title_text}</h5><p class='text-warning'>Cannot plot: Column '{col}' not found or DataFrame '{df_name}' is empty.</p></div>"
    try:
        if pd.api.types.is_numeric_dtype(df[col]):
            fig = px.histogram(df, x=col, title=title_text, nbins=50, marginal="box")
        else:
            # For categorical, show top N or all if few unique values
            top_n = df[col].nunique() if df[col].nunique() <= 20 else 20
            counts = df[col].value_counts(dropna=False).nlargest(top_n).reset_index()
            counts.columns = [col, 'count']
            counts[col] = counts[col].astype(str) # Ensure categorical treatment for plot
            fig = px.bar(counts, x=col, y='count', title=f'{title_text} (Top {top_n if df[col].nunique() > top_n else "All"})')
            fig.update_xaxes(type='category')
        
        fig.update_layout(margin=dict(l=40, r=20, t=60, b=40), height=450, title_x=0.5)
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        return f"<div class='plot-container-placeholder'><h5>{title_text}</h5><p class='text-danger'>Error plotting distribution for '{col}' in '{df_name}': {e}</p></div>"

def analyze_outliers_plotly_html(df, column, df_name=""):
    title_text = f"Outlier Analysis for '{column}'" + (f" in {df_name}" if df_name else "")
    if df.empty or column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return f"<div class='plot-container-placeholder'><h4>{title_text}</h4><p class='text-warning'>Cannot analyze: Column is not numeric, not found, or DataFrame is empty.</p></div>"

    html_output = f"<h4>{title_text}</h4>" # Keep heading for placeholder too
    try:
        numeric_column = pd.to_numeric(df[column], errors='coerce')
        if numeric_column.isnull().all():
            return f"<div class='plot-container-placeholder'><h4>{title_text}</h4><p class='text-warning'>Column '{column}' could not be treated as numeric for outlier analysis.</p></div>"

        Q1 = numeric_column.quantile(0.25)
        Q3 = numeric_column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_df_count = numeric_column[(numeric_column < lower_bound) | (numeric_column > upper_bound)].shape[0]
        
        stats_text = f"""
        <ul class='eda-stats-list'>
            <li><b>Total records:</b> {len(numeric_column)}</li>
            <li><b>Number of outliers (IQR method):</b> {outliers_df_count}</li>
            <li><b>Percentage of outliers:</b> {round(outliers_df_count / len(numeric_column) * 100, 2) if len(numeric_column) > 0 else 0}%</li>
            <li><b>IQR:</b> {IQR:.2f}</li>
            <li><b>Lower bound (for outliers):</b> {lower_bound:.2f}</li>
            <li><b>Upper bound (for outliers):</b> {upper_bound:.2f}</li>
            <li><b>Column Min:</b> {numeric_column.min():.2f}</li>
            <li><b>Column Max:</b> {numeric_column.max():.2f}</li>
            <li><b>Column Mean:</b> {numeric_column.mean():.2f}</li>
            <li><b>Column Median:</b> {numeric_column.median():.2f}</li>
        </ul>
        """
        html_output += stats_text
        
        fig = px.box(df, y=column, title=f"Box Plot for '{column}'", points="all") 
        fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=450, title_x=0.5)
        html_output += pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        html_output += f"<p class='text-danger'>Error analyzing outliers for '{column}': {e}</p>"
    return html_output

def show_value_counts_html(df, columns, df_name=""):
    if df.empty:
        return f"<h4>Value Counts for {df_name}</h4><p>DataFrame is empty.</p>"
    
    html_output = f"<h4>Value Counts (Selected Columns in {df_name})</h4>"
    found_any_relevant_columns = False
    for col in columns:
        if col in df.columns:
            found_any_relevant_columns = True
            html_output += f"<h5>Value Counts for '{col}':</h5>"
            try:
                counts_df = df[col].value_counts(dropna=False).reset_index()
                counts_df.columns = [col, 'Count']
                counts_df[col] = counts_df[col].astype(str) # Ensure string for mixed types
                
                value_counts_table = ""
                max_rows_display = 30
                if len(counts_df) > max_rows_display:
                    value_counts_table += f"<p>(Showing Top {max_rows_display} of {len(counts_df)} unique values for '{col}')</p>" + counts_df.head(max_rows_display).to_html(classes=["table", "table-sm", "table-striped", "dataframe-table-inner"], border=0, index=False)
                else:
                     value_counts_table += counts_df.to_html(classes=["table", "table-sm", "table-striped", "dataframe-table-inner"], border=0, index=False)
                html_output += f"<div class='table-responsive-wrapper'>{value_counts_table}</div>" 
            except Exception as e:
                html_output += f"<p class='text-danger'>Error generating value counts for '{col}': {e}</p>"
    if not found_any_relevant_columns:
        html_output += "<p>None of the specified columns for value counts were found in this dataset.</p>"
    return html_output