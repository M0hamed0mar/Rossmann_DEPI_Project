from flask import Flask, render_template, redirect, url_for, get_flashed_messages, flash, request, session, send_file
import os
import io 
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, RobustScaler 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib # For saving/loading models
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import eda_utils 

app = Flask(__name__)
app.secret_key = 'your_final_secret_key_with_top_nav_and_no_limits_v2!' 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CLEAN_DATA_PATH = os.path.join(DATA_DIR, 'clean_data.csv') # For cleaned training data
PROCESSED_TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv') # Cleaned test data, potentially overwriting original test.csv
OUTLIER_SCALERS_PATH = os.path.join(MODEL_DIR, 'outlier_scalers.pkl')
IMPUTATION_MEDIANS_PATH = os.path.join(MODEL_DIR, 'imputation_medians.pkl')
X_TRAIN_COLS_PATH = os.path.join(MODEL_DIR, 'x_train_columns.pkl')
STANDARD_SCALER_PATH = os.path.join(MODEL_DIR, 'standard_scaler.pkl')
PCA_PATH = os.path.join(MODEL_DIR, 'pca_transformer.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'XGB_Model.pkl')

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# --- Custom Jinja Test ---
def is_starting_with(value, prefix):
    return isinstance(value, str) and value.startswith(prefix)
app.jinja_env.tests['startingwith'] = is_starting_with

# --- Helper functions for preprocessing ---
def preprocess_outlier_columns(df, columns_to_process, fit_scalers_dict=None):
    """
    Applies log1p and RobustScaler to specified columns.
    If fit_scalers_dict is None, it fits new scalers and returns them.
    If fit_scalers_dict is provided, it uses them to transform.
    """
    df_copy = df.copy()
    fitted_scalers = fit_scalers_dict if fit_scalers_dict is not None else {}
    newly_fitted_scalers = {} 

    for col in columns_to_process:
        if col not in df_copy.columns:
            print(f"Warning: Column '{col}' not found for outlier preprocessing. Skipping.")
            continue
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            try:
                df_copy[col] = pd.to_numeric(df_copy[col])
            except ValueError:
                print(f"Warning: Column '{col}' is not numeric and couldn't be converted. Skipping outlier preprocessing.")
                continue
        
        df_copy[col] = df_copy[col].fillna(0) 
        df_copy[col] = np.log1p(df_copy[col].astype(float).clip(lower=0))
        
        if col not in fitted_scalers: 
            scaler = RobustScaler()
            scaled_values = scaler.fit_transform(df_copy[[col]])
            newly_fitted_scalers[col] = scaler 
        else: 
            scaler = fitted_scalers[col]
            scaled_values = scaler.transform(df_copy[[col]])
            
        df_copy[col] = scaled_values.flatten()
    
    if fit_scalers_dict is None: 
        return df_copy, newly_fitted_scalers
    return df_copy, fitted_scalers


def perform_rossmann_feature_engineering(df_input, is_training_phase=True):
    """
    Applies feature engineering (mappings, OHE, column drops) to the input DataFrame.
    This mirrors the steps in Rossmann_XGB_Model.ipynb that happen *after* loading clean_data.csv.
    """
    df = df_input.copy()

    columns_to_drop = ['Date', 'DayOfWeek'] 
    if is_training_phase and 'is_test' in df.columns: 
        columns_to_drop.append('is_test')
    
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df.drop(columns=existing_cols_to_drop, axis=1, inplace=True, errors='ignore')

    stateholiday_map = {'0': 0, '0.0':0, 'a': 1, 'b': 2, 'c': 3} 
    if 'StateHoliday' in df.columns:
        df['StateHoliday'] = df['StateHoliday'].astype(str).map(stateholiday_map).fillna(0)

    assortment_map = {'a': 0, 'b': 1, 'c': 2}
    if 'Assortment' in df.columns:
        df['Assortment'] = df['Assortment'].map(assortment_map).fillna(0)

    cols_to_one_hot = ['StoreType', 'PromoInterval', 'Day_name']
    existing_cols_to_one_hot = [col for col in cols_to_one_hot if col in df.columns]
    if existing_cols_to_one_hot:
        df = pd.get_dummies(df, columns=existing_cols_to_one_hot, dtype='int64', drop_first=True)
    
    if 'Customers' in df.columns and is_training_phase:
         df.drop(columns=['Customers'], inplace=True, errors='ignore')
    
    return df


def _perform_rossmann_data_cleaning_and_save_artifacts():
    """
    Performs full data cleaning based on the Rossmann_data_cleaning.ipynb notebook logic.
    """
    train_path = os.path.join(DATA_DIR, 'train.csv')
    store_path = os.path.join(DATA_DIR, 'store.csv')
    test_raw_input_path = os.path.join(DATA_DIR, 'test.csv') 

    if not all(os.path.exists(p) for p in [train_path, store_path, test_raw_input_path]):
        flash("`train.csv`, `store.csv`, or original `test.csv` not found in data directory.", "error")
        return

    try:
        df_train_raw = pd.read_csv(train_path, low_memory=False, parse_dates=['Date'])
        df_store_raw = pd.read_csv(store_path, low_memory=False)
        df_test_raw = pd.read_csv(test_raw_input_path, low_memory=False, parse_dates=['Date']) 
    except Exception as e:
        flash(f"Error reading CSVs: {e}", "error")
        return

    df_store = df_store_raw.copy()
    df_store.rename(columns={
        'CompetitionDistance': 'compdistance',
        'CompetitionOpenSinceMonth': 'compmonth',
        'CompetitionOpenSinceYear': 'compyear'
    }, inplace=True)

    train_merged = pd.merge(df_train_raw, df_store, on='Store', how='inner') 
    test_merged = pd.merge(df_test_raw, df_store, on='Store', how='inner')   

    train_merged['is_test'] = False
    test_merged['is_test'] = True
    if 'Sales' not in test_merged.columns: 
        test_merged['Sales'] = np.nan
    if 'Customers' not in test_merged.columns: 
        test_merged['Customers'] = np.nan
    
    train_cols = set(train_merged.columns)
    test_cols = set(test_merged.columns)
    for col in train_cols - test_cols:
        if col not in ['Sales', 'Customers']:
            test_merged[col] = np.nan 
    for col in test_cols - train_cols:
        if col not in ['Id']:
            train_merged[col] = np.nan

    combined_df = pd.concat([train_merged, test_merged], ignore_index=True, sort=False)
    
    if 'Id' in combined_df.columns: 
        combined_df.drop(columns=['Id'], inplace=True, errors='ignore')

    if 'StateHoliday' in combined_df.columns:
        combined_df['StateHoliday'] = combined_df['StateHoliday'].astype(str).replace('0.0', '0').fillna('0')

    if 'Open' in combined_df.columns:
        combined_df['Open'].fillna(1, inplace=True) 
        combined_df['Open'] = combined_df['Open'].astype(int)

    if 'Promo2' in combined_df.columns:
        combined_df.loc[combined_df['Promo2'] == 0, ['Promo2SinceWeek', 'Promo2SinceYear']] = 0
        combined_df.loc[combined_df['Promo2'] == 0, 'PromoInterval'] = 'noPromo'
        if 'Promo2SinceWeek' in combined_df.columns: combined_df['Promo2SinceWeek'].fillna(0, inplace=True)
        if 'Promo2SinceYear' in combined_df.columns: combined_df['Promo2SinceYear'].fillna(0, inplace=True)
        if 'PromoInterval' in combined_df.columns: combined_df['PromoInterval'].fillna('noPromo', inplace=True)
    
    imputation_medians = {}
    for col in ['compdistance', 'compmonth', 'compyear']:
        if col in combined_df.columns:
            if combined_df[col].isnull().any():
                median_val = combined_df[col].median()
                combined_df[col].fillna(median_val, inplace=True)
                imputation_medians[col] = median_val
            else: 
                imputation_medians[col] = combined_df[col].median() 

    if 'Date' in combined_df.columns:
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
        combined_df['Year'] = combined_df['Date'].dt.year
        combined_df['Month'] = combined_df['Date'].dt.month
        combined_df['Day'] = combined_df['Date'].dt.day
        combined_df['Day_name'] = combined_df['Date'].dt.day_name().astype(str) 

        date_components_to_impute = {'Year': 'median', 'Month': 'median', 'Day': 'median', 'Day_name': 'mode'}
        for comp_col, method in date_components_to_impute.items():
            if comp_col in combined_df.columns and combined_df[comp_col].isnull().any():
                val_to_impute = combined_df[comp_col].median() if method == 'median' else (combined_df[comp_col].mode()[0] if not combined_df[comp_col].mode().empty else 'Unknown')
                combined_df[comp_col].fillna(val_to_impute, inplace=True)
                imputation_medians[f'{comp_col}_dt_component'] = val_to_impute
            elif comp_col in combined_df.columns: # Save even if no NaNs for potential future use in prepare_test_data
                 imputation_medians[f'{comp_col}_dt_component'] = combined_df[comp_col].median() if method == 'median' else (combined_df[comp_col].mode()[0] if not combined_df[comp_col].mode().empty else ('0' if pd.api.types.is_numeric_dtype(combined_df[comp_col]) else 'Unknown') )
    
    joblib.dump(imputation_medians, IMPUTATION_MEDIANS_PATH)
    flash(f"Imputation medians (incl. date components) saved to {IMPUTATION_MEDIANS_PATH}", "info")

    outlier_cols_to_process = ['Customers', 'compdistance']
    actual_outlier_cols = [col for col in outlier_cols_to_process if col in combined_df.columns]
    
    if actual_outlier_cols:
        combined_df, fitted_outlier_scalers = preprocess_outlier_columns(combined_df, actual_outlier_cols) 
        joblib.dump(fitted_outlier_scalers, OUTLIER_SCALERS_PATH)
        flash(f"Outlier scalers (fitted on combined data) saved to {OUTLIER_SCALERS_PATH}", "info")
    else:
        flash("Skipped outlier processing for cleaning stage: 'Customers' or 'compdistance' not found.", "warning")
    
    clean_train_df = combined_df[combined_df['is_test'] == False].copy()
    if 'is_test' in clean_train_df.columns:
        clean_train_df.drop(columns=['is_test'], inplace=True, errors='ignore')
    clean_train_df.dropna(subset=['Sales'], inplace=True)
    clean_train_df.to_csv(CLEAN_DATA_PATH, index=False)
    flash(f"Cleaned training data saved to {CLEAN_DATA_PATH}", "success")

    processed_test_df = combined_df[combined_df['is_test'] == True].copy()
    if 'is_test' in processed_test_df.columns:
        processed_test_df.drop(columns=['is_test'], inplace=True, errors='ignore')
    processed_test_df.to_csv(PROCESSED_TEST_DATA_PATH, index=False) 
    flash(f"Cleaned test data (from combined cleaning) saved to {PROCESSED_TEST_DATA_PATH}", "success")


# --- Navigation Link Injector ---
@app.context_processor
def inject_nav_links():
    links = []
    excluded_endpoints = ['static', 'loading_screen', 'entry_point', 'explore_single_file_route', '_debug_toolbar.static', 'download_custom_plot']
    ordered_nav_items = [
        ('dashboard_view', 'Dataset Explorer', '📊'), ('list_datasets_route', 'Select Dataset', '📂'),
        ('data_cleaning', 'Data Cleaning', '🧹'), ('data_visualization', 'Data Visualization', '📈'),
        ('predict', 'Model Prediction', '🔮'),
    ]
    processed_endpoints = set()
    for endpoint, name, icon in ordered_nav_items:
        try:
            rule_found = any(not rule.arguments for rule in app.url_map.iter_rules(endpoint=endpoint))
            if rule_found: url_for(endpoint); links.append({'endpoint': endpoint, 'name': name, 'icon': icon}); processed_endpoints.add(endpoint)
        except Exception as e: print(f"Debug: Could not add ordered nav item '{endpoint}': {e}")

    for rule in app.url_map.iter_rules():
        endpoint = rule.endpoint
        if endpoint in processed_endpoints or endpoint in excluded_endpoints or rule.arguments or endpoint.startswith('_'): continue
        name = endpoint.replace('_', ' ').replace(' route', '').title(); icon = '🔗'
        if not any(link['endpoint'] == endpoint for link in links):
            try: url_for(endpoint); links.append({'endpoint': endpoint, 'name': name, 'icon': icon}); processed_endpoints.add(endpoint)
            except Exception as e: print(f"Debug: Could not auto-add nav item '{endpoint}': {e}")
    return dict(nav_links=links)

# --- Flask Routes ---
@app.route('/')
def entry_point(): return redirect(url_for('loading_screen'))
@app.route('/loading')
def loading_screen(): return render_template('loading.html')

@app.route('/dashboard')
def dashboard_view(): 
    return render_template('dashboard.html', active_page='dashboard', title_prefix="📊",
                           page_title="Dataset Exploration Dashboard", dataset_files=None, eda_report=None)    

@app.route('/list_datasets')
def list_datasets_route():
    dataset_files_list = []; page_title = "Select a Dataset to Explore"
    if not os.path.exists(DATA_DIR):
        try: os.makedirs(DATA_DIR); flash(f"Data directory '{DATA_DIR}' created. Add CSVs.", "info")
        except OSError as e: flash(f"Could not create data directory '{DATA_DIR}': {e}.", "error")
        return render_template('dashboard.html', dataset_files=[], page_title=page_title, title_prefix="📋", active_page='dashboard', eda_report=None)
    try:
        all_files_in_data_dir = os.listdir(DATA_DIR)
        csv_files = [f for f in all_files_in_data_dir if f.endswith('.csv') and not f.startswith('~$')]
        if not csv_files: flash("No CSV files in 'data/'. Add CSVs.", "warning")
        else: dataset_files_list = sorted(csv_files); flash(f"Found {len(dataset_files_list)} CSVs. Click to explore.", "success")
    except FileNotFoundError: flash(f"Data directory '{DATA_DIR}' not found.", "error")
    return render_template('dashboard.html', active_page='dashboard', title_prefix="📋", 
                           page_title=page_title, dataset_files=dataset_files_list, eda_report=None) 


@app.route('/explore_file/<path:filename>') 
def explore_single_file_route(filename):
    filepath = os.path.join(DATA_DIR, filename); available_files = []
    if os.path.exists(DATA_DIR):
        try: available_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and not f.startswith('~$')])
        except Exception as e: print(f"Error listing available files: {e}")
    if not os.path.exists(filepath) or not filename.endswith('.csv'):
        flash(f"Error: File '{filename}' not found or not CSV.", "error")
        return render_template('dashboard.html', dataset_files=available_files, eda_report=None, page_title="File Error", title_prefix="❓", active_page='dashboard')
    eda_item_report = {'filename': filename, 'overview_html': "<p>Generating EDA...</p>", 'statistics_html': "", 'missing_html': "", 'value_counts_html': "", 'plots': [] }
    try:
        df = pd.read_csv(filepath, low_memory=False)
        if df.empty:
            eda_item_report['overview_html'] = f"<p class='text-warning'>File '{filename}' is empty.</p>"
            eda_item_report['plots'].append(f"<p class='text-warning'>EDA cannot be performed on empty file: {filename}</p>")
        else:
            eda_item_report['overview_html'] = eda_utils.data_overview_html(df, filename)
            eda_item_report['statistics_html'] = eda_utils.data_statistics_html(df, filename)
            eda_item_report['missing_html'] = eda_utils.missing_values_report_html(df, filename)
            cols_for_dist_plots, cols_for_outlier_analysis = [], []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].nunique(dropna=False) > 1: cols_for_outlier_analysis.append(col)
                    if df[col].nunique(dropna=False) > 5: cols_for_dist_plots.append(col)
                elif df.shape[0] > 0 and df[col].nunique(dropna=False) > 1 and df[col].nunique(dropna=False) < (len(df)*0.95) and df[col].nunique(dropna=False) < 50:
                    cols_for_dist_plots.append(col)
            if cols_for_dist_plots:
                eda_item_report['plots'].append("<h4>Column Distributions:</h4>")
                for col_name in cols_for_dist_plots: eda_item_report['plots'].append(eda_utils.plot_distribution_plotly_html(df, col_name, filename))
            else: eda_item_report['plots'].append("<p>No suitable columns for distribution plots.</p>")
            if cols_for_outlier_analysis:
                eda_item_report['plots'].append("<hr class='eda-separator'><h4>Outlier Analysis (Box Plots):</h4>")
                for col_name in cols_for_outlier_analysis: eda_item_report['plots'].append(eda_utils.analyze_outliers_plotly_html(df, col_name, filename))
            elif cols_for_dist_plots: eda_item_report['plots'].append("<p>No suitable numeric columns for outlier analysis.</p>")
            notebook_cat_cols = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval', 'SchoolHoliday', 'Promo', 'Promo2', 'DayOfWeek', 'Year', 'Month', 'Day_name'] 
            relevant_cat_cols = [col for col in notebook_cat_cols if col in df.columns]
            if relevant_cat_cols: eda_item_report['value_counts_html'] = eda_utils.show_value_counts_html(df, relevant_cat_cols, filename)
        flash(f"EDA for '{filename}' generated.", "success")
    except pd.errors.EmptyDataError:
         error_msg = f"Error: File '{filename}' empty or not valid CSV."; eda_item_report['overview_html'] = f"<p class='text-danger'>{error_msg}</p>"
         eda_item_report['plots'].append(f"<p class='text-danger'>Cannot perform EDA: {filename}</p>"); flash(error_msg, "error")
    except Exception as e:
        error_message = f"Error processing '{filename}': {type(e).__name__} - {str(e)}"; print(f"CRITICAL ERROR for {filename}: {error_message}") 
        eda_item_report['overview_html'] = f"<p class='text-danger'>Error processing '{filename}'. Check logs.</p>"
        eda_item_report['plots'].append(f"<p class='text-danger'>Could not generate plots for '{filename}'.</p>"); flash(f"Error for {filename}. Check logs.", "error")
    return render_template('dashboard.html', active_page='dashboard', title_prefix="📄", 
                           page_title=f"EDA: {filename}", dataset_files=available_files, eda_report=eda_item_report)


@app.route('/data_cleaning', methods=['GET', 'POST'])
def data_cleaning():
    page_title="Data Cleaning (Rossmann)"; title_prefix="🧹"; table_html = None; missing_summary_dict = None
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'clean_data' or action == 'reset': 
            flash("Starting Rossmann data cleaning process (notebook logic)...", "info")
            if not os.path.exists(MODEL_DIR):
                try: os.makedirs(MODEL_DIR)
                except OSError as e: flash(f"Could not create models dir '{MODEL_DIR}': {e}.", "error"); return render_template('data_cleaning.html', page_title=page_title, title_prefix=title_prefix, table_html=None, missing_summary=None)
            
            if action == 'reset':
                if os.path.exists(CLEAN_DATA_PATH): os.remove(CLEAN_DATA_PATH)
                if os.path.exists(PROCESSED_TEST_DATA_PATH): os.remove(PROCESSED_TEST_DATA_PATH) 
                if os.path.exists(OUTLIER_SCALERS_PATH): os.remove(OUTLIER_SCALERS_PATH)
                if os.path.exists(IMPUTATION_MEDIANS_PATH): os.remove(IMPUTATION_MEDIANS_PATH)
                flash("Previous cleaned data files and artifacts removed for reset.", "info")

            _perform_rossmann_data_cleaning_and_save_artifacts() 
        else: flash(f"Unknown action: {action}", "warning")

    if os.path.exists(CLEAN_DATA_PATH):
        try:
            df_preview = pd.read_csv(CLEAN_DATA_PATH, low_memory=False)
            if not df_preview.empty:
                table_html = df_preview.head(10).to_html(classes=["table", "table-sm", "table-striped", "dataframe-table-inner"], border=0, escape=False, index=False)
                missing_summary_series = df_preview.isnull().sum()
                missing_summary_dict = missing_summary_series[missing_summary_series > 0].to_dict()
                if not missing_summary_dict: missing_summary_dict = {} 
            else: flash("'clean_data.csv' is empty.", "warning")
        except pd.errors.EmptyDataError: flash("'clean_data.csv' is empty. Click 'Clean Data'.", "warning")
        except Exception as e: flash(f"Error loading 'clean_data.csv' preview: {e}", "error"); print(f"Preview error: {e}")
    else: flash("'clean_data.csv' not found. Click 'Clean Data'.", "info")
    return render_template('data_cleaning.html', page_title=page_title, title_prefix=title_prefix,
                           table_html=table_html, missing_summary=missing_summary_dict)


# --- Data Visualization Routes ---
def plot_monthly_total_sales_plotly(df):
    if df.empty or not all(c in df.columns for c in ['Date', 'Sales']): return "<p class='text-warning'>Missing Date/Sales for Monthly Sales plot.</p>"
    try: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    except: pass 
    monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum().reset_index()
    monthly_sales['Date_str'] = monthly_sales['Date'].astype(str)
    fig = px.line(monthly_sales, x='Date_str', y='Sales', markers=True, title='Total Monthly Sales', labels={'Date_str': 'Month', 'Sales': 'Total Sales'})
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def plot_monthly_sales_by_store_type_plotly(df):
    if df.empty or not all(c in df.columns for c in ['Date', 'StoreType', 'Sales']): return "<p class='text-warning'>Missing Date/StoreType/Sales for this plot.</p>"
    try: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    except: pass
    monthly_sales = df.groupby([df['Date'].dt.to_period('M'), 'StoreType'])['Sales'].sum().reset_index()
    monthly_sales['Date_str'] = monthly_sales['Date'].astype(str)
    fig = px.line(monthly_sales, x='Date_str', y='Sales', color='StoreType', markers=True, title='Monthly Sales by Store Type', labels={'Date_str': 'Month', 'Sales': 'Total Sales'})
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def plot_avg_sales_by_day_of_week_plotly(df):
    if df.empty or not all(c in df.columns for c in ['Sales']): return "<p class='text-warning'>Missing Sales for this plot.</p>"
    
    day_col_for_x = None
    if 'Day_name' in df.columns: day_col_for_x = 'Day_name'
    elif 'DayOfWeek' in df.columns: day_col_for_x = 'DayOfWeek'
    else: return "<p class='text-warning'>Missing Day_name or DayOfWeek for this plot.</p>"

    avg_sales_day = df.groupby(day_col_for_x)['Sales'].mean().reset_index()
    
    if day_col_for_x == 'Day_name':
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        avg_sales_day[day_col_for_x] = pd.Categorical(avg_sales_day[day_col_for_x], categories=day_order, ordered=True)
    elif day_col_for_x == 'DayOfWeek': 
        day_order_num = sorted(avg_sales_day[day_col_for_x].unique())
        avg_sales_day[day_col_for_x] = pd.Categorical(avg_sales_day[day_col_for_x], categories=day_order_num, ordered=True)
    
    avg_sales_day.sort_values(day_col_for_x, inplace=True)
    fig = px.bar(avg_sales_day, x=day_col_for_x, y='Sales', color=day_col_for_x, title='Avg Sales by Day of Week', labels={day_col_for_x: 'Day of Week'})
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
def plot_sales_dist_by_store_type_plotly(df):
    if df.empty or not all(c in df.columns for c in ['StoreType', 'Sales']): return "<p class='text-warning'>Missing StoreType/Sales for this plot.</p>"
    fig = px.box(df, x='StoreType', y='Sales', color='StoreType', title='Sales Distribution by Store Type')
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def plot_customers_vs_sales_plotly(df):
    if df.empty or not all(c in df.columns for c in ['Customers', 'Sales']): return "<p class='text-warning'>Missing Customers/Sales for this plot.</p>"
    sample_df = df.sample(n=min(5000, len(df)), random_state=42) if len(df) > 5000 else df
    fig = px.scatter(sample_df, x='Customers', y='Sales', title='Customers (Processed) vs Sales (Sampled)', trendline="ols", trendline_color_override="red")
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def plot_correlation_heatmap_plotly(df):
    if df.empty: return "<p class='text-warning'>Data empty for Heatmap.</p>"
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty: return "<p class='text-warning'>No numeric columns for Heatmap.</p>"
    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmin=-1, zmax=1, text=np.around(corr.values,2), texttemplate="%{text}", hoverongaps=False))
    fig.update_layout(title='Correlation Heatmap', height=max(600, 50 * len(corr.columns)), margin=dict(l=100, t=80, b=100))
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

@app.route('/data_visualization', methods=['GET', 'POST'])
def data_visualization():
    plot_html, available_columns = None, []
    title_prefix, page_title = "📈", "Data Visualization"

    if os.path.exists(CLEAN_DATA_PATH):
        try: available_columns = pd.read_csv(CLEAN_DATA_PATH, nrows=0).columns.tolist()
        except Exception as e: flash(f"Could not read columns from clean_data.csv: {e}", "warning")
    else:
        if request.method == 'GET':
            flash("<code>clean_data.csv</code> not found. Please use the 'Data Cleaning' page first.", "error")

    if request.method == 'POST':
        action = request.form.get('action')

        if not os.path.exists(CLEAN_DATA_PATH):
            flash("<code>clean_data.csv</code> not found. Please clean data first.", "error")
            return render_template('data_visualization.html', page_title=page_title, title_prefix=title_prefix, 
                                   plot_html=None, available_columns=available_columns)
        
        df = pd.DataFrame()
        try: df = pd.read_csv(CLEAN_DATA_PATH, low_memory=False, parse_dates=['Date'])
        except Exception as e: flash(f"Error loading <code>clean_data.csv</code>: {e}", "error")
        
        if df.empty and os.path.exists(CLEAN_DATA_PATH):
            flash("<code>clean_data.csv</code> is empty.", "warning")
        elif not df.empty :
            plot_generated = False; fig_for_save = None 
            try:
                df_copy = df.copy()
                predefined_plot_functions = {
                    'monthly_total_sales': plot_monthly_total_sales_plotly,
                    'monthly_sales_by_store_type': plot_monthly_sales_by_store_type_plotly,
                    'avg_sales_by_day_of_week': plot_avg_sales_by_day_of_week_plotly,
                    'sales_dist_by_store_type': plot_sales_dist_by_store_type_plotly,
                    'cust_vs_sales': plot_customers_vs_sales_plotly,
                    'correlation_heatmap': plot_correlation_heatmap_plotly
                }
                if action in predefined_plot_functions:
                    plot_html_content = predefined_plot_functions[action](df_copy)
                    if isinstance(plot_html_content, str) and "<p class='text-warning'>" not in plot_html_content and "<p class='text-danger'>" not in plot_html_content :
                        plot_html = plot_html_content; plot_generated = True
                        session.pop('current_custom_plot_fig_json', None) 
                        flash(f"Predefined plot '{action.replace('_', ' ').title()}' generated.", "success")
                    elif isinstance(plot_html_content, str): plot_html = plot_html_content; plot_generated = False
                    else: flash(f"Plot '{action}' error.", "warning")
                elif action == 'generate_custom_plot':
                    plot_type = request.form.get('plot_type'); x_col = request.form.get('x_column')
                    y_col = request.form.get('y_column'); color_col = request.form.get('color_column')
                    agg_func = request.form.get('aggregation_func', 'mean')
                    page_title = f"Custom: {plot_type.title() if plot_type else 'Plot'}"
                    if x_col not in df.columns or \
                       (y_col and y_col != "" and y_col not in df.columns) or \
                       (color_col and color_col != "" and color_col not in df.columns):
                        flash("Invalid column selection.", "error")
                    elif not x_col: flash("X-axis column required.", "error")
                    else:
                        current_fig = None; plot_title_text = f"Custom {plot_type.replace('_',' ').title()}: {x_col}" + (f" by {y_col}" if y_col else "") + (f" (clr: {color_col})" if color_col else "")
                        try:
                            if plot_type == 'histogram':
                                current_fig = px.histogram(df_copy, x=x_col, color=color_col if color_col else None, title=plot_title_text, marginal="box" if pd.api.types.is_numeric_dtype(df_copy[x_col]) else None)
                            elif plot_type == 'boxplot':
                                if y_col and y_col!="" and pd.api.types.is_numeric_dtype(df_copy[y_col]):
                                    current_fig = px.box(df_copy, x=x_col if x_col else None, y=y_col, color=color_col if color_col else None, title=plot_title_text)
                                elif pd.api.types.is_numeric_dtype(df_copy[x_col]): 
                                    current_fig = px.box(df_copy, y=x_col, color=color_col if color_col else None, title=plot_title_text)
                                else: flash("Boxplot needs numeric Y.", "error")
                            elif plot_type == 'scatter':
                                if y_col and y_col!="" and pd.api.types.is_numeric_dtype(df_copy[x_col]) and pd.api.types.is_numeric_dtype(df_copy[y_col]):
                                    sample_df = df_copy.sample(n=min(5000, len(df_copy)), random_state=42)
                                    current_fig = px.scatter(sample_df, x=x_col, y=y_col, color=color_col if color_col else None, title=plot_title_text, trendline="ols")
                                else: flash("Scatter needs numeric X & Y.", "error")
                            elif plot_type in ['bar', 'line']:
                                if y_col and y_col!="" and pd.api.types.is_numeric_dtype(df_copy[y_col]):
                                    group_cols = [x_col]; 
                                    if color_col and color_col!="" and color_col!=x_col and color_col!=y_col: group_cols.append(color_col)
                                    df_agg = df_copy.groupby(group_cols, as_index=False)[y_col].agg(agg_func)
                                    target_color_col = color_col if color_col and color_col in df_agg.columns else None
                                    if plot_type == 'bar': current_fig = px.bar(df_agg, x=x_col, y=y_col, color=target_color_col, title=plot_title_text)
                                    elif plot_type == 'line': current_fig = px.line(df_agg, x=x_col, y=y_col, color=target_color_col, title=plot_title_text, markers=True)
                                else: flash(f"{plot_type.title()} needs numeric Y.", "error")
                            if current_fig:
                                current_fig.update_layout(title_x=0.5, margin=dict(l=40,r=20,t=80,b=40))
                                fig_for_save = current_fig 
                                plot_html = pio.to_html(current_fig, full_html=False, include_plotlyjs='cdn')
                                plot_generated = True; flash("Custom plot generated.", "success")
                                session['current_custom_plot_fig_json'] = current_fig.to_json()
                            elif not get_flashed_messages(category_filter=["error"]): flash("Could not generate custom plot.", "warning")
                        except Exception as e: flash(f"Custom plot error '{plot_type}': {e}", "error"); print(f"Custom plot err: {e}")
                elif action: flash(f"Action '{action}' not recognized.", "warning")
                if isinstance(fig_for_save, str): plot_html = fig_for_save; session.pop('current_custom_plot_fig_json', None)
                elif fig_for_save is not None : plot_html = pio.to_html(fig_for_save, full_html=False, include_plotlyjs='cdn'); session['current_custom_plot_fig_json'] = fig_for_save.to_json()
                elif plot_generated and not fig_for_save : flash(f"Plot for '{action}' problem.", "warning")
            except Exception as e: flash(f"Plot processing error: {e}", "error"); print(f"Plotting process err: {e}")
    custom_plot_generated_this_request = session.get('current_custom_plot_fig_json') is not None
    return render_template('data_visualization.html', page_title=page_title, title_prefix=title_prefix, 
                           plot_html=plot_html, available_columns=sorted(list(set(available_columns))),
                           custom_plot_generated=custom_plot_generated_this_request)

@app.route('/download_custom_plot/<filename_ext>')
def download_custom_plot(filename_ext):
    fig_json = session.get('current_custom_plot_fig_json')
    if not fig_json:
        flash("No custom plot in session to download.", "error")
        return redirect(url_for('data_visualization'))
    try:
        fig = pio.from_json(fig_json)
        img_format = filename_ext.lower() if filename_ext.lower() in ['png','jpeg','jpg','svg','pdf'] else 'png'
        filename = f"custom_plot.{img_format}"
        img_bytes = fig.to_image(format=img_format)
        return send_file(io.BytesIO(img_bytes), mimetype=f'image/{img_format if img_format != "jpg" else "jpeg"}', as_attachment=True, download_name=filename)
    except ImportError: flash("Kaleido not installed. `pip install kaleido`", "error"); return redirect(url_for('data_visualization'))
    except Exception as e: flash(f"Error generating image: {e}", "error"); print(f"Download err: {e}"); return redirect(url_for('data_visualization'))


# --- Model Prediction Route (and its helpers) ---
def prepare_test_data_for_prediction(raw_test_df_input, store_df_input, 
                                     fitted_outlier_scalers, imputation_medians,
                                     x_train_column_order, scaler_std, pca_transformer):
    """ 
    Prepares raw test data for prediction using fitted preprocessors and artifacts.
    """
    test_df = raw_test_df_input.copy() 
    store_df_copy = store_df_input.copy()
    
    if 'CompetitionDistance' in store_df_copy.columns:
        store_df_copy.rename(columns={
            'CompetitionDistance': 'compdistance',
            'CompetitionOpenSinceMonth': 'compmonth',
            'CompetitionOpenSinceYear': 'compyear'
        }, inplace=True, errors='ignore')
    
    test_merged = pd.merge(test_df, store_df_copy, how='left', on='Store')
    
    if 'StateHoliday' in test_merged.columns:
        test_merged['StateHoliday'] = test_merged['StateHoliday'].astype(str).replace('0.0', '0').fillna('0')

    if 'Open' in test_merged.columns:
        test_merged['Open'].fillna(1, inplace=True) 
        test_merged['Open'] = test_merged['Open'].astype(int)

    if 'Promo2' in test_merged.columns:
        test_merged.loc[test_merged['Promo2'] == 0, ['Promo2SinceWeek', 'Promo2SinceYear']] = 0
        test_merged.loc[test_merged['Promo2'] == 0, 'PromoInterval'] = 'noPromo'
        if 'Promo2SinceWeek' in test_merged.columns: test_merged['Promo2SinceWeek'].fillna(0, inplace=True)
        if 'Promo2SinceYear' in test_merged.columns: test_merged['Promo2SinceYear'].fillna(0, inplace=True)
        if 'PromoInterval' in test_merged.columns: test_merged['PromoInterval'].fillna('noPromo', inplace=True)

    for col in ['compdistance', 'compmonth', 'compyear']:
        if col in test_merged.columns and test_merged[col].isnull().any():
            median_to_use = imputation_medians.get(col, test_merged[col].median()) 
            test_merged[col].fillna(median_to_use, inplace=True)
    
    if 'Date' in test_merged.columns:
        test_merged['Date'] = pd.to_datetime(test_merged['Date'], errors='coerce')
        test_merged['Year'] = test_merged['Date'].dt.year
        test_merged['Month'] = test_merged['Date'].dt.month
        test_merged['Day'] = test_merged['Date'].dt.day
        test_merged['Day_name'] = test_merged['Date'].dt.day_name().astype(str)
        
        date_components_to_impute_keys = ['Year_dt_component', 'Month_dt_component', 'Day_dt_component', 'Day_name_dt_component']
        original_date_comp_names = ['Year', 'Month', 'Day', 'Day_name']

        for i, col in enumerate(original_date_comp_names): 
             if col in test_merged.columns and test_merged[col].isnull().any():
                artifact_key = date_components_to_impute_keys[i]
                fallback_val = test_merged[col].median() if pd.api.types.is_numeric_dtype(test_merged[col]) else (test_merged[col].mode()[0] if not test_merged[col].mode().empty else 'Unknown')
                val_to_impute = imputation_medians.get(artifact_key, fallback_val)
                test_merged[col].fillna(val_to_impute, inplace=True)

    outlier_cols_in_test = [col for col in ['compdistance'] if col in test_merged.columns] 
    if outlier_cols_in_test:
        test_merged_outlier_scaled, _ = preprocess_outlier_columns(test_merged, outlier_cols_in_test, fit_scalers_dict=fitted_outlier_scalers)
    else:
        test_merged_outlier_scaled = test_merged
        
    test_df_featured = perform_rossmann_feature_engineering(test_merged_outlier_scaled, is_training_phase=False)
    
    for col in x_train_column_order:
        if col not in test_df_featured.columns:
            test_df_featured[col] = 0 
    test_df_featured = test_df_featured[x_train_column_order] 
    
    if test_df_featured.isnull().any().any():
        print(f"NaNs found in test_df_featured before scaling (predict path): {test_df_featured.isnull().sum()[test_df_featured.isnull().sum() > 0]}")
        for col in test_df_featured.columns[test_df_featured.isnull().any()].tolist():
            if pd.api.types.is_numeric_dtype(test_df_featured[col]):
                 test_df_featured[col].fillna(test_df_featured[col].median(), inplace=True)
            else:
                 test_df_featured[col].fillna(test_df_featured[col].mode()[0] if not test_df_featured[col].mode().empty else 'Unknown', inplace=True)

    test_scaled = scaler_std.transform(test_df_featured)
    test_pca = pca_transformer.transform(test_scaled)
    
    return test_pca, test_df_featured


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    training_metrics, predictions_table, error_message = None, None, None
    page_title="Model Training & Prediction"
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == "train_model":
            if not os.path.exists(CLEAN_DATA_PATH):
                flash("<code>clean_data.csv</code> not found. Please run 'Data Cleaning' first.", "error")
            else:
                try:
                    df_clean = pd.read_csv(CLEAN_DATA_PATH, low_memory=False, parse_dates=['Date'])
                    if df_clean.empty: flash("'clean_data.csv' is empty.", "error"); raise ValueError("Empty clean data")

                    train_df_featured = perform_rossmann_feature_engineering(df_clean, is_training_phase=True)
                    
                    if 'Sales' not in train_df_featured.columns:
                        raise ValueError("'Sales' column not found in featured data for training.")

                    X = train_df_featured.drop('Sales', axis=1, errors='raise') 
                    y = train_df_featured['Sales']
                    
                    if X.isnull().any().any():
                        print("NaNs found in X before train/test split. Imputing with median/mode...")
                        flash("NaNs detected and imputed in feature set before training.", "warning")
                        for col in X.columns[X.isnull().any()].tolist():
                            if pd.api.types.is_numeric_dtype(X[col]):
                                X[col].fillna(X[col].median(), inplace=True)
                            else: 
                                X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
                    
                    joblib.dump(X.columns.tolist(), X_TRAIN_COLS_PATH) 

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

                    scaler_std = StandardScaler()
                    X_train_scaled = scaler_std.fit_transform(X_train) 
                    X_test_scaled = scaler_std.transform(X_test)     
                    joblib.dump(scaler_std, STANDARD_SCALER_PATH)

                    pca = PCA(n_components=0.90) 
                    X_train_pca = pca.fit_transform(X_train_scaled)
                    X_test_pca = pca.transform(X_test_scaled)
                    joblib.dump(pca, PCA_PATH)
                    
                    flash(f"Data split & preprocessed. X_train_pca shape: {X_train_pca.shape}", "info")

                    param_dist = {
                        'n_estimators': [500], 
                        'max_depth': [10],
                        'learning_rate': [0.1],
                        'subsample': [0.7],
                        'colsample_bytree': [0.8, 1.0],
                    }
                    flash("Starting XGBoost RandomizedSearchCV (this may take a moment)...", "info")
                    xgb_model_search = RandomizedSearchCV(
                        estimator=XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse', n_jobs=-1),
                        param_distributions=param_dist,
                        n_iter=5, 
                        scoring='neg_root_mean_squared_error',
                        cv=2, 
                        verbose=1, random_state=42, n_jobs=1 
                    )
                    xgb_model_search.fit(X_train_pca, y_train)
                    best_xgb_model = xgb_model_search.best_estimator_
                    
                    joblib.dump(best_xgb_model, MODEL_PATH)
                    
                    y_train_pred = best_xgb_model.predict(X_train_pca)
                    y_test_pred = best_xgb_model.predict(X_test_pca)
                    
                    training_metrics = {
                        'Train MAE': mean_absolute_error(y_train, y_train_pred),
                        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                        'Train R2': r2_score(y_train, y_train_pred),
                        'Test MAE': mean_absolute_error(y_test, y_test_pred),
                        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                        'Test R2': r2_score(y_test, y_test_pred),
                        'Best Params': xgb_model_search.best_params_
                    }
                    flash("XGBoost Model trained, evaluated, and all artifacts saved successfully.", "success")
                
                except ValueError as ve: 
                    if "Input X contains NaN" in str(ve):
                        error_message = f"Error during model training: Input data for PCA/Scaler contained NaNs. {ve}."
                        flash(error_message, "error")
                        print(f"Model training NaN error: {ve}")
                    else:
                        error_message = f"Error during model training: {type(ve).__name__} - {ve}"
                        flash(error_message, "error")
                        print(f"Model training error: {ve}")
                    import traceback
                    print(traceback.format_exc())
                except Exception as e:
                    error_message = f"Error during model training: {type(e).__name__} - {e}"
                    flash(error_message, "error")
                    import traceback
                    print(f"Model training error: {e}\n{traceback.format_exc()}")


        elif action == "predict_test":
            required_artifacts = [MODEL_PATH, STANDARD_SCALER_PATH, PCA_PATH, 
                                  X_TRAIN_COLS_PATH, OUTLIER_SCALERS_PATH, IMPUTATION_MEDIANS_PATH]
            if not all(os.path.exists(p) for p in required_artifacts):
                flash("One or more required model/preprocessor files not found. Train the model first or ensure all artifacts exist.", "error")
            else:
                raw_test_data_input_path = os.path.join(DATA_DIR, 'test.csv') 
                store_data_path = os.path.join(DATA_DIR, 'store.csv')
                if not os.path.exists(raw_test_data_input_path) or not os.path.exists(store_data_path):
                    flash(f"Original raw '{os.path.basename(raw_test_data_input_path)}' or 'store.csv' not found in data directory.", "error")
                else:
                    try:
                        model = joblib.load(MODEL_PATH)
                        scaler_std_loaded = joblib.load(STANDARD_SCALER_PATH)
                        pca_transformer_loaded = joblib.load(PCA_PATH)
                        x_train_cols_loaded = joblib.load(X_TRAIN_COLS_PATH)
                        outlier_scalers_loaded = joblib.load(OUTLIER_SCALERS_PATH)
                        imputation_medians_loaded = joblib.load(IMPUTATION_MEDIANS_PATH)

                        df_test_raw_input = pd.read_csv(raw_test_data_input_path, low_memory=False, parse_dates=['Date'])
                        df_store_raw_pred = pd.read_csv(store_data_path, low_memory=False)
                        
                        flash("Preparing test data (from raw test.csv) for prediction...", "info")
                        
                        X_test_final_pca, _ = prepare_test_data_for_prediction(
                            df_test_raw_input, df_store_raw_pred, 
                            outlier_scalers_loaded, imputation_medians_loaded,
                            x_train_cols_loaded, scaler_std_loaded, pca_transformer_loaded
                        )
                        
                        predictions_raw = model.predict(X_test_final_pca)
                        predictions = np.maximum(0, predictions_raw) 

                        test_ids = df_test_raw_input['Id'].copy() if 'Id' in df_test_raw_input.columns else pd.Series(range(len(predictions)))
                        predictions_df = pd.DataFrame({'Id': test_ids, 'PredictedSales': predictions})
                        predictions_table = predictions_df.head(20).to_html(classes="table table-sm table-striped dataframe-table-inner", border=0, escape=False, index=False)
                        flash("Predictions generated on test data successfully.", "success")

                    except Exception as e:
                        error_message = f"Error during prediction on test data: {type(e).__name__} - {e}"
                        flash(error_message, "error")
                        import traceback
                        print(f"Prediction error: {e}\n{traceback.format_exc()}")
        else:
            flash(f"Unknown action: {action}", "warning")

    return render_template('model.html', 
                           page_title=page_title, 
                           title_prefix="🔮",
                           training_metrics=training_metrics,
                           predictions_table=predictions_table, 
                           error_message=error_message)


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        print(f"Creating data directory '{DATA_DIR}'...")
        try: os.makedirs(DATA_DIR); print("Directory created. Add CSVs like train.csv, store.csv, test.csv.")
        except OSError as e: print(f"CRITICAL: Could not create data directory: {e}.")
    if not os.path.exists(MODEL_DIR):
        print(f"Creating models directory '{MODEL_DIR}'...")
        try: os.makedirs(MODEL_DIR); print("Directory created.")
        except OSError as e: print(f"CRITICAL: Could not create models directory: {e}.")
    app.run(debug=True)