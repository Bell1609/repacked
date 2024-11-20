import datetime as dt
import sys
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
# Add the parent directory to the system path
import os

import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fs.data_handling import Data_Handling
from fs.graph_drawing import Graph_Drawing
from authentication import make_sidebar
from db_connector import run_query

# Duong updated 4 Nov 2024 - start
def preprocess_data(df):
    # List of columns to keep
    columns_to_keep = [
        'Deal : id',
        'Deal : Name',
        'Deal : Total Deal Value',
        'Deal : Expected close date',
        'Deal : Closed date',
        'Deal : Owner',
        'Deal : Deal stage',
        'Deal : Project type',
        'Deal : Product 1',
        'Deal : Probability (%)',
        'Deal : Created at',
        'Deal : Age (in days)',
        'Deal : Account name',
        'Deal : Expected go live date/MED',
        'Deal : Tentative start date/MSD',
        'Deal : Project Code',
        'Deal : Type of Renewal',
        'Deal : Total Cost',
        'Deal : Gross Margin (GM)',
        'Deal : Software revenue: Product 1',
        'Deal : Software cost: Product 1',
        'Deal : Support revenue: Product 1',
        'Deal : Support cost: Product 1',
        'Deal : Service revenue: Product 1',
        'Deal : Service cost: Product 1',
        'Deal : Cons days: Product 1',
        'Deal : Technical days: Product 1',
        'Deal : PM days: Product 1',
        'Deal : PA days: Product 1',
        'Deal : Hosting revenue: Product 1',
        'Deal : Managed service revenue: Product 1',
        'Deal : Managed service cost: Product 1',
        'Deal : Product 2',
        'Deal : Software revenue: Product 2',
        'Deal : Software cost: Product 2',
        'Deal : Support revenue: Product 2',
        'Deal : Support cost: Product 2',
        'Deal : Service revenue: Product 2',
        'Deal : Service cost: Product 2',
        'Deal : Cons days: Product 2',
        'Deal : Technical days: Product 2',
        'Deal : PM days: Product 2',
        'Deal : PA days: Product 2',
        'Deal : Hosting revenue: Product 2',
        'Deal : Managed service revenue: Product 2',
        'Deal : Managed service cost: Product 2',
        'Deal : Product 3',
        'Deal : Software revenue: Product 3',
        'Deal : Software cost: Product 3',
        'Deal : Support revenue: Product 3',
        'Deal : Support cost: Product 3',
        'Deal : Service revenue: Product 3',
        'Deal : Service cost: Product 3',
        'Deal : Cons days: Product 3',
        'Deal : Technical days: Product 3',
        'Deal : PM days: Product 3',
        'Deal : PA days: Product 3',
        'Deal : Hosting revenue: Product 3',
        'Deal : Managed service revenue: Product 3',
        'Deal : Managed service cost: Product 3',
        'Deal : Product 4',
        'Deal : Software revenue: Product 4',
        'Deal : Software cost: Product 4',
        'Deal : Support revenue: Product 4',
        'Deal : Support cost: Product 4',
        'Deal : Service revenue: Product 4',
        'Deal : Service cost: Product 4',
        'Deal : Cons days: Product 4',
        'Deal : Technical days: Product 4',
        'Deal : PM days: Product 4',
        'Deal : PA days: Product 4',
        'Deal : Hosting revenue: Product 4',
        'Deal : Managed service revenue: Product 4',
        'Deal : Managed service cost: Product 4',
    ]

    # Select only the specified columns
    df = df[columns_to_keep]
    
    # Convert 'Deal : id' to string type
    df['Deal : id'] = df['Deal : id'].astype(str)

    # Clean and convert amount columns
    df = data_handling.clean_and_convert_amount_columns(df)

    # Define mixed columns to convert to strings (replace with actual columns if needed)
    df = data_handling.convert_mixed_columns_to_string(df)

    # Convert date columns to datetime format
    df = data_handling.convert_date_columns_to_date(df)
    
    # Ensure probability is numeric due to debugging error
    df['Deal : Probability (%)'] = pd.to_numeric(df['Deal : Probability (%)'], errors='coerce')

    return df

def filter_won_deals(dataframe, selected_year):
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Won")
    #st.write("Won Deals")
    #st.write(f"From month: {from_month} - To Month: {to_month}")
    won_deals = data_handling.filter_deals(dataframe, from_month, to_month, 'Deal : Closed date')

    # Filter the deals further to include only those with 'Deal : Deal stage' == 'Won'
    won_deals = won_deals[won_deals['Deal : Deal stage'] == 'Won']
    return won_deals

def filter_open_deals(dataframe, selected_year):
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Open")
    #st.write("Open Deals")
    #st.write(f"From month: {from_month} - To Month: {to_month}")
    open_deals = data_handling.filter_deals(dataframe, from_month, to_month, 'Deal : Expected close date')

    # Filter the deals further to include only those with 'Deal : Deal stage' == 'Won'
    open_deals = open_deals[
        (open_deals['Deal : Deal stage'] != "Lost")
        ]

    return open_deals

# Filter all new deals created within the selected_year
def filter_new_deals(dataframe, selected_year):
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Pipeline")
    new_deals = data_handling.filter_deals(dataframe, from_month, to_month, 'Deal : Created at')

    return new_deals

# Filter all pipeline deals expected closed within the selected_year with probability
def filter_pipeline_deals_probability(dataframe, selected_year, probability):
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Open")
    pipeline_deals = data_handling.filter_deals(dataframe, from_month, to_month, 'Deal : Expected close date')
    
     # Filter the deals further to include only those with 'Deal : Deal probability (%)'
    pipeline_deals = pipeline_deals[
        (pipeline_deals['Deal : Probability (%)'] >= probability) &
        (~pipeline_deals['Deal : Deal stage'].isin(['Won', 'Lost']))
    ]
    return pipeline_deals

def visual_funnel_chart_pipeline(funnel_data):
    
    # Create labels for each bar to display number of deals, total deal value, and probability
    funnel_data['label'] = (
        'Deals: ' + funnel_data['number_of_deals'].astype(str) +
        ', Value: $' + funnel_data['total_deal_value'].round(2).astype(str) +
        ', ' + funnel_data['Deal : Probability (%)'].astype(str) + '%'
    )
    #Duong review

    funnel_data['adjusted_value'] = (1 - funnel_data['Deal : Probability (%)'] / 100)

    # Create the funnel chart with custom x-axis and color mapping
    fig = px.funnel(
        funnel_data,
        x='adjusted_value',  # Use adjusted values for width simulation
        y='Deal : Deal stage',  # Set the y-axis to Deal stage
        color='Deal : Deal stage',  # Different color for each stage
        title='Deal Funnel by Probability'
        #labels={'adjusted_value': 'Adjusted Deal Value', 'Deal : Deal stage': 'Deal Stage'}
    )

    # Show only the custom label on each bar
    fig.update_traces(
        text=funnel_data['label'],
        textposition="inside",
        textinfo="text"
    )

    # Adjust layout for title and axes
    fig.update_layout(
        title_x=0.5,  # Center the title
        yaxis_title="Deal Stage",
        xaxis_title="Deal Count and Total Deal Value",
        showlegend=False
    )

    return fig

def filter_pipeline_backdate_deals(dataframe, selected_year):
    """Generate the trend of total deal value and deal count in the pipeline backdate grouped by month."""
    
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Pipeline")
    # st.write("Pipeline Deals")
    # st.write(f"From month: {from_month} - To Month: {to_month}")
    
    # Generate a range of month-ends from start to end
    date_range = pd.date_range(start=from_month, end=to_month, freq='ME')

    # Convert DatetimeIndex to a list to allow appending
    date_range_list = date_range.tolist()

    # Adjust the time to 23:59:59 for each date in the list
    date_range_list = [date.replace(hour=23, minute=59, second=59) for date in date_range_list]

    # Convert end_date to a pandas Timestamp if it is not already
    end_date_ts = pd.Timestamp(to_month).replace(hour=23, minute=59, second=59)

    # If the exact end_date is not already in the date range, add it
    if end_date_ts not in date_range_list:
        date_range_list.append(end_date_ts)

    # Sort the list of dates to maintain chronological order
    date_range_list = sorted(date_range_list)

    # Convert the list back to a DatetimeIndex
    date_range = pd.DatetimeIndex(date_range_list)

    
    def pipeline_value_and_count_at_month(df, month_end):
        """Calculate total deal value and count of deals in the pipeline as of the end of a given month."""
        # Extend end_date to include the full day
        #end_of_month = pd.to_datetime(month_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        
        # Calculate the start of the month based on month_end
        month_start = month_end.replace(day=1)

        # Filter deals that were in the pipeline during the given month
        pipeline_deals = df[
            (df['Deal : Created at'] <= month_end) &  # Deal was created on or before the month end
            ((df['Deal : Closed date'].isna()) | (df['Deal : Closed date'] > month_end))  # Deal is still open or closed after the month end
        ]
        # st.write(f'Start: {month_start} - End: {end_of_month}')
        # st.write(f'Rows: {pipeline_deals["Deal : id"].count()}')
        # st.dataframe(pipeline_deals[['Deal : Name','Deal : Total Deal Value','Deal : Owner','Deal : Project type']])
        # Sum the total deal value for the filtered deals
        total_value = pipeline_deals['Deal : Total Deal Value'].sum()

    
        # Count deals
        deal_count = pipeline_deals['Deal : id'].nunique()
        
        deals_closed = pipeline_deals[
            (pipeline_deals['Deal : Closed date'] >= month_start) &  
            (pipeline_deals['Deal : Closed date'] <= month_end) &
            (pipeline_deals['Deal : Deal stage'] == 'Won')
        ]
        deal_closed_count = deals_closed['Deal : id'].nunique()

        return total_value, deal_count, deal_closed_count

    # Initialize lists to store results
    months = []
    as_at_date = []
    total_values = []
    deal_counts = []
    deal_closed_counts = []

    # Calculate total deal value and deal count for each month in the date range
    for month_end in date_range:
        total_value, deal_count, deal_closed_count = pipeline_value_and_count_at_month(dataframe, month_end)
        months.append(month_end.strftime('%Y-%m'))
        as_at_date.append(month_end)
        total_values.append(total_value)  # Store total value
        deal_counts.append(deal_count)  # Store deal count
        deal_closed_counts.append(deal_closed_count)  # Store deal count

    
    # Create a DataFrame to return
    trend_df = pd.DataFrame({
        'Month': months,
        'As At Date': as_at_date,
        'Total Deal Value': total_values,
        'Deals Count': deal_counts,
        'Deals Closed Count': deal_closed_counts
    })

    return trend_df

# filter deals in pipeline expected to close within the selected year
def filter_expected_close_deals(dataframe, selected_year):
    """Generate the trend of total deal value and deal count in the pipeline backdate grouped by month."""
    
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Open")
    # st.write("Pipeline Deals")
    # st.write(f"From month: {from_month} - To Month: {to_month}")
    
    # Generate a range of month-ends from start to end
    date_range = pd.date_range(start=from_month, end=to_month, freq='ME')

    # Convert DatetimeIndex to a list to allow appending
    date_range_list = date_range.tolist()

    # Adjust the time to 23:59:59 for each date in the list
    date_range_list = [date.replace(hour=23, minute=59, second=59) for date in date_range_list]

    # Convert end_date to a pandas Timestamp if it is not already
    end_date_ts = pd.Timestamp(to_month).replace(hour=23, minute=59, second=59)

    # If the exact end_date is not already in the date range, add it
    if end_date_ts not in date_range_list:
        date_range_list.append(end_date_ts)

    # Sort the list of dates to maintain chronological order
    date_range_list = sorted(date_range_list)

    # Convert the list back to a DatetimeIndex
    date_range = pd.DatetimeIndex(date_range_list)

    
    def expected_to_close_value_and_count_at_month(df, month_end):
        """Calculate total deal value and count of deals in the pipeline as of the end of a given month."""
        # Extend end_date to include the full day
        #end_of_month = pd.to_datetime(month_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        
        # Calculate the start of the month based on month_end
        month_start = month_end.replace(day=1)

        # Filter deals expected to close within the month
        pipeline_deals = df[
            (df['Deal : Expected close date'] <= month_end) &  
            (df['Deal : Expected close date'] >= month_start) 
        ]
        # st.write(f'Start: {month_start} - End: {end_of_month}')
        # st.write(f'Rows: {pipeline_deals["Deal : id"].count()}')
        # st.dataframe(pipeline_deals[['Deal : Name','Deal : Total Deal Value','Deal : Owner','Deal : Project type']])
        # Sum the total deal value for the filtered deals
        total_value = pipeline_deals['Deal : Total Deal Value'].sum()

        # Count deals created in the current month (between month_start and month_end)

        deal_count = pipeline_deals['Deal : id'].nunique()
        
        deals_closed = pipeline_deals[
            (pipeline_deals['Deal : Closed date'] >= month_start) &  
            (pipeline_deals['Deal : Closed date'] <= month_end) &
            (pipeline_deals['Deal : Deal stage'] == 'Won')
        ]
        deal_closed_count = deals_closed['Deal : id'].nunique()

        return total_value, deal_count, deal_closed_count

    # Initialize lists to store results
    months = []
    as_at_date = []
    total_values = []
    deal_counts = []
    deal_closed_counts = []

    # Calculate total deal value and deal count for each month in the date range
    for month_end in date_range:
        total_value, deal_count, deal_closed_count = expected_to_close_value_and_count_at_month(dataframe, month_end)
        months.append(month_end.strftime('%Y-%m'))
        as_at_date.append(month_end)
        total_values.append(total_value)  # Store total value
        deal_counts.append(deal_count)  # Store deal count
        deal_closed_counts.append(deal_closed_count)  # Store deal count

    
    # Create a DataFrame to return
    trend_df = pd.DataFrame({
        'Month': months,
        'As At Date': as_at_date,
        'Total Deal Value': total_values,
        'Deals Count': deal_counts,
        'Deals Closed Count': deal_closed_counts
    })

    return trend_df

# Add last year columns to pipeline value backdate by months
def add_last_year_columns(pipeline_deals, pre_pipeline_deals):
    # Convert 'Month' columns to datetime format to make comparison easier
    pipeline_deals['Month'] = pd.to_datetime(pipeline_deals['Month'], format='%Y-%m')
    pre_pipeline_deals['Month'] = pd.to_datetime(pre_pipeline_deals['Month'], format='%Y-%m')

    # Initialize the new columns in pipeline_deals with NaN (numeric type, float)
    pre_pipeline_deals['Current Year Total Deal Value'] = pd.Series(dtype='float64')
    pre_pipeline_deals['Current Year Deals Count'] = pd.Series(dtype='float64')
    pre_pipeline_deals['Current Year Deals Closed Count'] = pd.Series(dtype='float64')

    # Loop through each row in pipeline_deals
    for index, row in pre_pipeline_deals.iterrows():
        # Calculate the current year month
        current_year_month = row['Month'] + pd.DateOffset(years=1)
        
        # Find the corresponding row in pipeline_deals with the same month as current_year_month
        current_year_data = pipeline_deals[pipeline_deals['Month'] == current_year_month]
        
        # If matching row exists in pre_pipeline_deals, update the columns
        if not current_year_data.empty:
            pre_pipeline_deals.at[index, 'Current Year Total Deal Value'] = current_year_data['Total Deal Value'].values[0]
            pre_pipeline_deals.at[index, 'Current Year Deals Count'] = current_year_data['Deals Count'].values[0]
            pre_pipeline_deals.at[index, 'Current Year Deals Closed Count'] = current_year_data['Deals Closed Count'].values[0]
    
    # Ensure the new columns are numeric
    pre_pipeline_deals['Current Year Total Deal Value'] = pd.to_numeric(pre_pipeline_deals['Current Year Total Deal Value'], errors='coerce')
    pre_pipeline_deals['Current Year Deals Count'] = pd.to_numeric(pre_pipeline_deals['Current Year Deals Count'], errors='coerce')
    pre_pipeline_deals['Current Year Deals Closed Count'] = pd.to_numeric(pre_pipeline_deals['Current Year Deals Closed Count'], errors='coerce')
    
    return pre_pipeline_deals

# Duong updated 04 Nov 2024 - start
def calculate_won_deal_product_breakdown_difference(input_df, input_year):
    """
    Calculate won deal overview difference with year comparison.

    Parameters:
    input_df (DataFrame): The input DataFrame containing deal data.
    input_year (int): The year for which to calculate the difference.

    Returns:
    DataFrame: A DataFrame with the won deal overview difference including year,
               won rate, lost rate, and aggregated totals for the specified columns.
    """

    # Define columns of interest
    columns_of_interest = [
        'Deal Software Revenue', 'Deal Software Cost', 
        'Deal Support Revenue', 'Deal Support Cost', 
        'Deal Service Revenue', 'Deal Service Cost', 
        'Deal Cons Days', 'Deal PM Days', 'Deal PA Days', 'Deal Technical Days', 
        'Deal Hosting Revenue', 'Deal Managed Service Revenue', 
        'Deal Managed Service Cost'
    ]

    # Step 1: Calculate date ranges for input_year and input_year - 1
    from_month, to_month = data_handling.calculate_date_range(input_year, input_df, "Won")
    from_month_prev, to_month_prev = data_handling.calculate_date_range(input_year - 1, input_df, "Won")

    # Step 2: Filter deals for the current and previous year within specified date ranges
    deals_current_year = data_handling.filter_deals(input_df, from_month, to_month, 'Deal : Closed date')
    deals_previous_year = data_handling.filter_deals(input_df, from_month_prev, to_month_prev, 'Deal : Closed date')

    # Step 3: Filter won deals by stage for each year
    won_deals_current = deals_current_year[deals_current_year['Deal : Deal stage'] == 'Won']
    won_deals_previous = deals_previous_year[deals_previous_year['Deal : Deal stage'] == 'Won']

    # Step 4: Convert relevant columns to numeric
    for col in columns_of_interest:
        if col in won_deals_current.columns:
            won_deals_current[col] = pd.to_numeric(won_deals_current[col].astype(str).str.replace(',', '.'), errors='coerce')
            won_deals_previous[col] = pd.to_numeric(won_deals_previous[col].astype(str).str.replace(',', '.'), errors='coerce')

    # Step 5: Aggregate totals by Deal Product Brand, Deal Product, and Project type for each year
    current_year_totals = won_deals_current.groupby(['Deal Product', 'Deal : Project type'])[columns_of_interest].sum().reset_index()
    previous_year_totals = won_deals_previous.groupby(['Deal Product', 'Deal : Project type'])[columns_of_interest].sum().reset_index()

    # Step 6: Add year information to DataFrames
    current_year_totals['Year'] = input_year
    previous_year_totals['Year'] = input_year - 1

    # Step 7: Concatenate DataFrames and rearrange columns to have 'Year' as the first column
    result_df = pd.concat([current_year_totals, previous_year_totals], ignore_index=True)
    result_df = result_df[['Year'] + [col for col in result_df.columns if col != 'Year']]

    # Step 8: Calculate Won and Lost counts for the current and previous year
    won_count_current = won_deals_current.shape[0]
    lost_count_current = deals_current_year[deals_current_year['Deal : Deal stage'] == 'Lost'].shape[0]

    won_count_previous = won_deals_previous.shape[0]
    lost_count_previous = deals_previous_year[deals_previous_year['Deal : Deal stage'] == 'Lost'].shape[0]

    # Step 9: Append won and lost counts to the DataFrame
    result_df['Won Count'] = result_df['Year'].apply(lambda x: won_count_current if x == input_year else won_count_previous)
    result_df['Lost Count'] = result_df['Year'].apply(lambda x: lost_count_current if x == input_year else lost_count_previous)

    return result_df

# Duong updated 18 Mov 2024 - start
def calculate_won_deal_overview_difference(input_df, input_year):
    """
    Calculate won deal overview difference with year comparison, grouped by 'Deal : Project type',
    'Deal Product Brand', and 'Deal Product'. Handles cases where grouping columns contain null values.
    
    Parameters:
    input_df (DataFrame): The input DataFrame containing deal data.
    input_year (int): The year for which to calculate the difference.
    
    Returns:
    DataFrame: A DataFrame with the won deal overview difference including year,
               won rate, lost rate, and aggregated totals for the specified columns.
    """

    # Define columns of interest
    columns_of_interest = [
        'Deal : Total Deal Value', 'Deal : Total Cost', 'Deal : Gross Margin (GM)'
    ]
    
    # Replace nulls in 'Deal : Project type' and 'Deal Product' with "Not Analysis"
    input_df['Deal : Project type'] = input_df['Deal : Project type'].fillna("Not Analysis")
    input_df['Deal Product'] = input_df['Deal Product'].fillna("Not Analysis")
    
    # Step 1: Calculate date ranges for input_year and input_year - 1
    from_month, to_month = data_handling.calculate_date_range(input_year, input_df, "Won")
    from_month_prev, to_month_prev = data_handling.calculate_date_range(input_year - 1, input_df, "Won")
    
    
    # Step 2: Filter deals for the current and previous year within specified date ranges
    deals_current_year = data_handling.filter_deals(input_df, from_month, to_month, 'Deal : Closed date')
    deals_previous_year = data_handling.filter_deals(input_df, from_month_prev, to_month_prev, 'Deal : Closed date')
    
    # Step 3: Filter won and lost deals by stage for each year
    won_deals_current = deals_current_year[deals_current_year['Deal : Deal stage'] == 'Won']
    won_deals_previous = deals_previous_year[deals_previous_year['Deal : Deal stage'] == 'Won']
    
    lost_deals_current = deals_current_year[deals_current_year['Deal : Deal stage'] == 'Lost']
    lost_deals_previous = deals_previous_year[deals_previous_year['Deal : Deal stage'] == 'Lost']
    
    # Step 4: Remove duplicates based on 'Deal : id' and all columns of interest
    won_deals_current = won_deals_current.drop_duplicates(subset=['Deal : id'] + columns_of_interest)
    won_deals_previous = won_deals_previous.drop_duplicates(subset=['Deal : id'] + columns_of_interest)
    lost_deals_current = lost_deals_current.drop_duplicates(subset=['Deal : id'] + columns_of_interest)
    lost_deals_previous = lost_deals_previous.drop_duplicates(subset=['Deal : id'] + columns_of_interest)
    
    # Step 5: Count won and lost deals for each year grouped by 'Deal : Project type' and 'Deal Product'
    won_current_counts = won_deals_current.groupby(['Deal : Project type', 'Deal Product']).size().reset_index(name='Won Count')
    lost_current_counts = lost_deals_current.groupby(['Deal : Project type', 'Deal Product']).size().reset_index(name='Lost Count')
    
    won_previous_counts = won_deals_previous.groupby(['Deal : Project type', 'Deal Product']).size().reset_index(name='Won Count')
    lost_previous_counts = lost_deals_previous.groupby(['Deal : Project type', 'Deal Product']).size().reset_index(name='Lost Count')
    
    # Step 6: Aggregate totals for specified columns grouped by 'Deal : Project type' and 'Deal Product'
    current_year_totals = (
        won_deals_current.groupby(['Deal : Project type', 'Deal Product'])[columns_of_interest]
        .sum()
        .reset_index()
    )
    previous_year_totals = (
        won_deals_previous.groupby(['Deal : Project type', 'Deal Product'])[columns_of_interest]
        .sum()
        .reset_index()
    )
    
    # Step 7: Add year information to DataFrames
    current_year_totals['Year'] = input_year
    previous_year_totals['Year'] = input_year - 1
    
    # Step 8: Merge counts with totals for the current and previous year
    current_year_totals = current_year_totals.merge(won_current_counts, on=['Deal : Project type', 'Deal Product'], how='left')
    current_year_totals = current_year_totals.merge(lost_current_counts, on=['Deal : Project type', 'Deal Product'], how='left')
    
    previous_year_totals = previous_year_totals.merge(won_previous_counts, on=['Deal : Project type', 'Deal Product'], how='left')
    previous_year_totals = previous_year_totals.merge(lost_previous_counts, on=['Deal : Project type', 'Deal Product'], how='left')
    
    # Step 9: Concatenate DataFrames and rearrange columns to have 'Year' as the first column
    overview_df = pd.concat([current_year_totals, previous_year_totals], ignore_index=True)
    overview_df = overview_df[['Year'] + [col for col in overview_df.columns if col != 'Year']]
    
    return overview_df

# Duong updated 18 Nov 2024 - end

def tab1_visualize_won_deal_revenue(won_deals_df, column_name, width=50, height=100, show_axes=False):
    """
    Generates a DataFrame grouped by year with the total values of the specified column for won deals,
    then visualizes it using a Plotly area chart with conditional color based on trend and a defined color palette.
    
    Parameters:
        won_deals_df (pd.DataFrame): DataFrame containing won deals with a 'Deal : Closed date' 
                                      and a specified column to be plotted.
        column_name (str): The name of the column to plot.
        width (int): Width of the plot in pixels.
        height (int): Height of the plot in pixels.
        show_axes (bool): Whether to display x and y axes (default is False).
    
    Returns:
        plotly.graph_objects.Figure: An area chart figure ready to be displayed in Streamlit.
    """
    # Ensure 'Deal : Closed date' is in datetime format
    won_deals_df['Deal : Closed date'] = pd.to_datetime(won_deals_df['Deal : Closed date'])
    
    # Group by year and sum the values in the specified column
    won_deals_by_year = (won_deals_df
                         .groupby(won_deals_df['Deal : Closed date'].dt.year)[column_name]
                         .sum()
                         .reset_index()
                         .rename(columns={'Deal : Closed date': 'Year'}))

    # Choose primary color for upward trend, supporting color for downward trend
    trend_color = POSITIVE_COLOR if won_deals_by_year[column_name].iloc[-1] >= won_deals_by_year[column_name].iloc[0] else NEGATIVE_COLOR
    
    # Create the area chart with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=won_deals_by_year['Year'],
        y=won_deals_by_year[column_name],
        mode='lines',
        fill='none',
        line=dict(color=trend_color, width=2)
    ))
    
    # Configure layout
    fig.update_layout(
        xaxis=dict(visible=show_axes),
        yaxis=dict(visible=show_axes),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',
        width=width,
        height=height
    )

    return fig

def tab1_visualize_metric_over_time(deals_data, metric, view_by):
    """
    Function to visualize a selected revenue/cost type over time (monthly, quarterly, or yearly) from the deals data.
    
    Args:
        deals_data (pd.DataFrame): The DataFrame containing the deals data.
        metric (str): The metric name (revenue/cost type) to visualize.
        view_by (str): The option to group data by ('Month', 'Quarter', or 'Year').
    """

    # Ensure that the 'Deal : Closed date' column is in datetime format
    deals_data['Deal : Closed date'] = pd.to_datetime(deals_data['Deal : Closed date'])
    
    # Group by Month, Quarter, or Year
    if view_by == "Month":
        # Extract month and year for grouping
        deals_data['Month'] = deals_data['Deal : Closed date'].dt.strftime('%Y-%m')  # Format as Year-Month (e.g., 2023-07)
        grouped_data = deals_data.groupby('Month').agg({metric: 'sum'}).reset_index()
        x_axis_label = 'Month'

    elif view_by == "Quarter":
        # Group by quarter with proper ordering
        deals_data['Quarter'] = deals_data['Deal : Closed date'].dt.to_period('Q')  # Create Period object for quarters
        deals_data['Year_Quarter'] = deals_data['Quarter'].dt.strftime('%Y - Q%q')  # Format as 'YYYY - QX'
        
        grouped_data = deals_data.groupby('Year_Quarter').agg({metric: 'sum'}).reset_index()
        x_axis_label = 'Quarter'

    else:  # view_by == "Year"
        # Group by year
        deals_data['Year'] = deals_data['Deal : Closed date'].dt.year  # Extract year
        grouped_data = deals_data.groupby('Year').agg({metric: 'sum'}).reset_index()
        x_axis_label = 'Year'

    # Calculate the average value of the metric
    average_value = grouped_data[metric].mean()

    # Create the bar chart
    fig = px.bar(
        grouped_data,
        x=grouped_data.columns[0],  # Either 'Month', 'Year_Quarter', or 'Year' depending on selection
        y=metric,
        #title=f'{metric} Over Time ({view_by})',
        labels={grouped_data.columns[0]: x_axis_label, metric: metric},
        color_discrete_sequence=[PRIMARY_COLORS[0]]  # Use primary color for visualization
    )

    # Add a line for the average value
    fig.add_trace(go.Scatter(
        x=grouped_data[grouped_data.columns[0]],  # X-axis values
        y=[average_value] * len(grouped_data),  # Repeat average value for all x-values
        mode='lines',
        name='Average',
        line=dict(color=PRIMARY_COLORS[1], dash='dash'),  # Use secondary color for the average line
    ))

    # Customize the layout to fit the width and height of the container
    fig.update_layout(
        autosize=True,
        height=380,
        width=900,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis_title=x_axis_label,
        yaxis_title=metric
    )

    return fig

def tab1_create_gauge_chart(target, progress, delta_value, title, color):
    """
    Create a gauge chart for revenue target achievement and progress using Plotly.
    
    Parameters:
    target (float): The target value to base steps on.
    progress (float): The percentage of progress towards the target (0 to 100).
    title (str): Title of the gauge.
    delta_value (float): The reference target value for comparison.
    color (list): List of colors for the gauge sections.
    
    Returns:
    Plotly figure showing the gauge chart for revenue progress.
    """
    
    # Ensure progress is within bounds
    #progress = min(max(progress, 0), 100)
    
    # Calculate steps based on target
    step_1 = 0.5 * target  # 50% of target
    step_2 = 0.75 * target # 75% of target
    

    # Create a gauge chart
    fig = go.Figure()

    # Define color ranges based on target steps
    steps = [
        {"range": [0, step_1], "color": "lightgray"},      # Step 1: 0 to 50% of target
        {"range": [step_1, step_2], "color": "gray"},      # Step 2: 50% to 75% of target
        {"range": [step_2, target], "color": "white"},      # Step 3: 75% to target
    ]

    # Add the gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=progress,
        delta={
            "reference": delta_value,          
            "position": "bottom",              # Position delta value just below the number
            "relative": False,                  # Make delta relative to indicate up/down movement
            "valueformat": ",.2f",              # Format delta value for consistency
            "increasing": {"color": "green"},  # Set green for positive delta (overachieving)
            "decreasing": {"color": "red"},    # Set red for negative delta (underachieving)
            "font": {"size": 20}               # Adjust delta font size
        },
        title={"text": title, "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, target]},       # Set the range of the gauge to the target
            "bar": {"color": color[0]},           # Color of the bar representing progress
            "bgcolor": "lightgray",                # Background color for the gauge
            "steps": steps,                        # Define steps with respective colors
            "threshold": {
                "line": {"color": "red", "width": 4},  # Threshold line properties
                "value": target,                   # Set threshold at target value
            },
        },
        number={"font": {"size": 30}},  # Adjust the font size to make it appear smaller
        domain={'x': [0, 1], 'y': [0, 1]}  # Full domain coverage
    ))


    # Customize the layout
    fig.update_layout(
        width=200,
        height=200,
        margin=dict(l=30, r=30, t=45, b=30)  # Adjust margins
    )

    return fig

@st.cache_data
def tab1_calculate_revenue_data_for_gauge_chart(df, sales_targets, current_year):
    """
    Calculate revenue and target data for all, recurring, and non-recurring deals,
    along with year-over-year deltas for each.

    Parameters:
    df: DataFrame containing deal data.
    sales_targets: Dictionary containing target values for different deal types.
    current_year: The current year to calculate deltas for.

    Returns:
    A dictionary containing target values, total deal values, progress percentages, 
    and year-over-year delta values for all, recurring, and non-recurring deals.
    """

    # Fetch targets from sales_targets dictionary
    target_total_revenue = sales_targets['deal_type']['AllDeal']
    target_non_recurring = sales_targets['deal_type']['NonRecurring']
    target_recurring = sales_targets['deal_type']['Recurring']
    
    # Define the recurring and non-recurring project types
    recurring_project_types = ['ARR']
    non_recurring_project_types = [ptype for ptype in df['Deal : Project type'].unique() if ptype not in recurring_project_types]
    
    # Filter data for current and previous year
    current_year_data = df[df['Year'] == current_year]
    previous_year_data = df[df['Year'] == current_year - 1]
    
    # Calculate achieved revenue for each deal type in the current year
    total_deal_value = current_year_data['Deal : Total Deal Value'].sum()
    recurring_revenue = current_year_data[current_year_data['Deal : Project type'].isin(recurring_project_types)]['Deal : Total Deal Value'].sum()
    non_recurring_revenue = current_year_data[current_year_data['Deal : Project type'].isin(non_recurring_project_types)]['Deal : Total Deal Value'].sum()
    
    # Calculate achieved revenue for each deal type in the previous year
    total_deal_value_previous = previous_year_data['Deal : Total Deal Value'].sum()
    recurring_revenue_previous = previous_year_data[previous_year_data['Deal : Project type'].isin(recurring_project_types)]['Deal : Total Deal Value'].sum()
    non_recurring_revenue_previous = previous_year_data[previous_year_data['Deal : Project type'].isin(non_recurring_project_types)]['Deal : Total Deal Value'].sum()



    # Create and return the result dictionary with all values
    return {
        'target_total_revenue': target_total_revenue,
        'target_recurring_revenue': target_recurring,
        'target_non_recurring_revenue': target_non_recurring,
        'total_deal_value': total_deal_value,
        'recurring_revenue': recurring_revenue,
        'non_recurring_revenue': non_recurring_revenue,
        'total_deal_value_previous': total_deal_value_previous,
        'recurring_revenue_previous': recurring_revenue_previous,
        'non_recurring_revenue_previous': non_recurring_revenue_previous
    }

# Duong added 4 Nov 2024 - start

def won_revenue_breakdown_tab(dataframe, selected_year):
    won_product_breakdown = calculate_won_deal_product_breakdown_difference(dataframe, selected_year)
    st.dataframe(won_product_breakdown)
# Duong added 4 Nov 2024 - end    

def sales_overview_tab(dataframe, selected_year):
    # generate won deals current and previous years with overview information "Total Deal Value", "Total Cost", "Total Gross Margin"
    # and grouped by "Deal Product" and "Deal : Project type"
    won_overview_df = calculate_won_deal_overview_difference(dataframe, selected_year)
   
    # generate won deals for all year from product and deals data
    # note that the total information "Total Deal Value", "Total Cost", "Total Gross Margin" are duplicate because 1 deals might have many product
    won_deals_all_year = dataframe[dataframe['Deal : Deal stage'] == 'Won']
    
    columns_of_interest =['Deal : Total Deal Value', 'Deal : Total Cost', 'Deal : Gross Margin (GM)']
    # Drop the duplicate total value row, use this new data frame to visualize the total value columns
    won_deals_all_year_total = won_deals_all_year.drop_duplicates(subset=['Deal : id'] + columns_of_interest)
   
    with st.container(border= True):
        #st.subheader(f"Sales Performance Dashboard {selected_year}")
        st.markdown(f"### Sales Overview {selected_year}")
        # Calculate the revenue data
        
        # Row 1: Overall metrics within a bordered container
        row1_cols = st.columns(4, gap='medium')
        with row1_cols[0]:
            with st.container(border=True):
                cols = st.columns(2)

                # Metric for Total Deal Value
                total_deal_value = won_overview_df.loc[won_overview_df['Year'] == selected_year, 'Deal : Total Deal Value'].sum()
                total_deal_value_previous = won_overview_df.loc[won_overview_df['Year'] == selected_year - 1, 'Deal : Total Deal Value'].sum()
                # total_deal_value = deals_current_year['Deal : Total Deal Value'].sum()
                # total_deal_value_previous = deals_previous_year['Deal : Total Deal Value'].sum()
                total_deal_value_delta = total_deal_value - total_deal_value_previous
                cols[0].metric('Total Deal Value', graph_drawing.format_number(total_deal_value), graph_drawing.format_number(total_deal_value_delta))

                # Plot the figure with custom width and height
                fig = tab1_visualize_won_deal_revenue(won_deals_all_year_total, column_name='Deal : Total Deal Value')
                cols[1].plotly_chart(fig)
        
        with row1_cols[1]:
            with st.container(border=True):
                cols = st.columns(2)
                total_cost = won_overview_df.loc[won_overview_df['Year'] == selected_year, 'Deal : Total Cost'].sum()
                total_cost_previous = won_overview_df.loc[won_overview_df['Year'] == selected_year-1, 'Deal : Total Cost'].sum()
                # total_cost = deals_current_year['Deal : Total Cost'].sum()
                # total_cost_previous = deals_previous_year['Deal : Total Cost'].sum()
                total_cost_delta = total_cost - total_cost_previous
                cols[0].metric('Total Cost', graph_drawing.format_number(total_cost), graph_drawing.format_number(total_cost_delta))
                
                # Plot the figure with custom width and height
                fig = tab1_visualize_won_deal_revenue(won_deals_all_year_total, column_name='Deal : Total Cost')
                cols[1].plotly_chart(fig)
        
        with row1_cols[2]:
            with st.container(border=True):
                cols = st.columns(2)
                total_GM = won_overview_df.loc[won_overview_df['Year'] == selected_year, 'Deal : Gross Margin (GM)'].sum()
                total_GM_previous = won_overview_df.loc[won_overview_df['Year'] == selected_year-1, 'Deal : Gross Margin (GM)'].sum()
                # total_GM = deals_current_year['Deal : Gross Margin (GM)'].sum()
                # total_GM_previous = deals_previous_year['Deal : Gross Margin (GM)'].sum()
                total_GM_delta = total_GM - total_GM_previous
                cols[0].metric('Total Gross Margin', graph_drawing.format_number(total_GM), graph_drawing.format_number(total_GM_delta))
                
                # Plot the figure with custom width and height
                fig = tab1_visualize_won_deal_revenue(won_deals_all_year_total, column_name='Deal : Gross Margin (GM)')
                cols[1].plotly_chart(fig)
                
        with row1_cols[3]:
            with st.container(border=True):
                cols = st.columns(2)
                won_count = won_overview_df.loc[won_overview_df['Year'] == selected_year, 'Won Count'].sum()
                lost_count = won_overview_df.loc[won_overview_df['Year'] == selected_year, 'Lost Count'].sum()
                total_count = (won_count + lost_count)
                won_rate = (won_count) / total_count
                won_count_previous = won_overview_df.loc[won_overview_df['Year'] == selected_year-1, 'Won Count'].sum()
                lost_count_previous = won_overview_df.loc[won_overview_df['Year'] == selected_year-1, 'Lost Count'].sum()
                total_previous_count = (won_count_previous + lost_count_previous)
                won_rate_previous = (won_count_previous) / total_previous_count
                # Assuming won_rate and won_rate_previous are calculated as decimal values (e.g., 0.75 for 75%)
                won_rate_display = f"{won_rate *100:.2f}%"
                won_rate_delta = f"{(won_rate - won_rate_previous) * 100:.2f}%"

                # Display metric with formatted values
                cols[0].metric("Won Deals", f"{won_count:.0f}", f"{(won_count - won_count_previous):.0f}")
                cols[1].metric('Conversion Rate', won_rate_display, won_rate_delta)
        
        
        row2_cols = st.columns((2, 4), gap='small', vertical_alignment="center")
        with row2_cols[0]:
            with st.container(height=500,border=True):
                # Calculate the revenue data
                revenue_data = tab1_calculate_revenue_data_for_gauge_chart(won_overview_df, sales_targets, selected_year)
                #st.dataframe(revenue_data)
                recurring_gauge = tab1_create_gauge_chart(revenue_data['target_recurring_revenue'], revenue_data['recurring_revenue'], revenue_data['recurring_revenue_previous'], 'Recurring Revenue Revenue',color=SUPPORTING_COLORS)
                st.markdown("")
                #st.markdown("")
                st.markdown("")
                st.plotly_chart(recurring_gauge, use_container_width=True)
                #st.markdown("")
                #st.markdown("")
                st.markdown("")
                non_recurring_gauge = tab1_create_gauge_chart(revenue_data['target_non_recurring_revenue'], revenue_data['non_recurring_revenue'], revenue_data['non_recurring_revenue_previous'],'Non-Recurring Revenue', color=SUPPORTING_COLORS)
                st.plotly_chart(non_recurring_gauge, use_container_width=True)
                 
        with row2_cols[1]:
            # Sales trend visualization
            metrics = [
                'Deal : Total Deal Value', 
                'Deal : Total Cost',
                'Deal : Gross Margin (GM)',
                'Deal Software Revenue',
                'Deal Software Cost',
                'Deal Support Revenue',
                'Deal Support Cost',
                'Deal Service Revenue',
                'Deal Service Cost',
                'Deal Cons Days',
                'Deal PM Days',
                'Deal PA Days',
                'Deal Technical Days',
                'Deal Hosting Revenue',
                'Deal Managed Service Revenue',
                'Deal Managed Service Cost'
            ]
            with st.container(height=500,border=True):
                cols = st.columns((4,2,1))
    
                with cols[0]:
                    st.markdown("#### Sales Trend")
                # Drop-down box to allow the user to select a revenue/cost type
                with cols[1]:
                    selected_metric = st.selectbox(
                        "Select Metric to Visualize",
                        options=metrics
                    )
                
                # Drop-down box to allow the user to select the view type (Month or Quarter)
                options = ["Month", "Quarter", "Year"]
                with cols[2]:
                    view_by = st.selectbox(
                    "Group By",
                    options=options,
                    index=options.index("Year")  # Set the default value to "Year"
                )
                if ":" in selected_metric:
                    fig = tab1_visualize_metric_over_time(won_deals_all_year_total, selected_metric, view_by)
                else:
                    fig = tab1_visualize_metric_over_time(won_deals_all_year, selected_metric, view_by)
                st.plotly_chart(fig, use_container_width=True)
                
        row3_cols = st.columns((2,1), gap='medium', vertical_alignment="center")
        with row3_cols[1]:
            with st.container(border=True):
                fig = graph_drawing.tab1_plot_total_deal_value_by_project_type(won_overview_df, selected_year, theme_colors)
                st.plotly_chart(fig, use_container_width=True)
                #st.write("test")
            
        with row3_cols[0]:
            with st.container(border=True):
                # fig = graph_drawing.tab1_plot_total_deal_value_by_product(won_overview_df, 2023, theme_colors)
                # st.plotly_chart(fig, use_container_width=True)
                # Filter the dataframe for the selected year
                filtered_df = won_overview_df[won_overview_df['Year'] == selected_year].copy()

                # Drop the "Year" column
                filtered_df = filtered_df.drop(columns=['Year'])

                # Sort by "Project type" and "Deal : Total Deal Value"
                sorted_df = filtered_df.sort_values(by=['Deal : Project type', 'Deal : Total Deal Value'], ascending=[True, False])
                # Reset the index to remove row indices and drop the old index
                sorted_df = sorted_df.reset_index(drop=True)
                # Display the dataframe without the index column
                st.dataframe(sorted_df, use_container_width=True)
    
#Duong added 05 Nov 2024 - start
def merge_products_deals_data(products_data, deals_data):
    """
    Merge products_data, which has multiple rows per Deal ID (each row per product), 
    with deals_data, which has a single row per Deal ID.
    This brings in additional deal details from deals_data.
    """
    
    # Columns to bring in from deals_data
    columns_to_include = [
        'Deal : id', 'Deal : Name', 'Deal : Total Deal Value', 'Deal : Expected close date',
        'Deal : Closed date', 'Deal : Owner', 'Deal : Deal stage', 'Deal : Project type',
        'Deal : Probability (%)', 'Deal : Created at', 'Deal : Age (in days)', 'Deal : Account name',
        'Deal : Expected go live date/MED', 'Deal : Tentative start date/MSD', 'Deal : Project Code',
        'Deal : Type of Renewal', 'Deal : Total Cost', 'Deal : Gross Margin (GM)'
    ]
    
    # Ensure 'Deal : id' is in the index for merging
    if 'Deal : id' not in products_data.columns or 'Deal : id' not in deals_data.columns:
        raise KeyError("Both DataFrames must contain the 'Deal : id' column for merging.")
    
    # Filter the columns from deals_data to include only the specified columns
    deals_data_filtered = deals_data[columns_to_include]
    
    # Perform a left join on 'Deal : id'
    merged_data = products_data.merge(deals_data_filtered, on='Deal : id', how='left')

    return merged_data
# Duong added 05 Nov 2024 - end

def revenue_breakdown_tab(dataframe, selected_year):
    from_month, to_month = data_handling.calculate_date_range(selected_year, dataframe, "Won")
    all_deals = data_handling.filter_deals(dataframe, from_month, to_month, 'Deal : Closed date')

    # Filter the deals further to include only those with 'Deal : Deal stage' == 'Won'
    won_deals = all_deals[all_deals['Deal : Deal stage'] == 'Won']
    lost_deals = all_deals[all_deals['Deal : Deal stage'] == 'Lost']
    
    # Calculate conversion rate (Won Deals / Open Deals * 100)
    won_deals_count = won_deals.shape[0]
    all_deals_count = all_deals.shape[0]
    lost_deals_count = lost_deals.shape[0]
    
    won_rate = (won_deals_count / all_deals_count * 100) if all_deals_count > 0 else 0
    lost_rate = (lost_deals_count / all_deals_count * 100) if all_deals_count > 0 else 0

    with st.container(border= True):
        #st.subheader(f"Sales Performance Dashboard {selected_year}")
        st.markdown(f"### Sales Overview {selected_year}")

        # Row 1: Overall metrics within a bordered container
        row1_cols = st.columns(6, gap='medium')
        with row1_cols[0]:
            with st.container(border=True):
                total_deal_value = graph_drawing.format_number(won_deals['Deal : Total Deal Value'].sum())
                st.metric('Total Deal Value', total_deal_value)
        with row1_cols[1]:
            with st.container(border=True):
                total_service_revenue = graph_drawing.format_number(won_deals['Deal Service Revenue'].sum())
                st.metric('Total Service Revenue', total_service_revenue)
        with row1_cols[2]:
            with st.container(border=True):
                total_software_revenue = graph_drawing.format_number(won_deals['Deal Software Revenue'].sum())
                st.metric('Total Software Revenue', total_software_revenue)
        with row1_cols[3]:
            with st.container(border=True):
                total_support_revenue = graph_drawing.format_number(won_deals['Deal Support Revenue'].sum())
                st.metric('Total Support Revenue', total_support_revenue)
        with row1_cols[4]:
            with st.container(border=True):
                total_managed_service_revenue = graph_drawing.format_number(won_deals['Deal Managed Service Revenue'].sum())
                st.metric('Total AMS Revenue', total_managed_service_revenue)
        with row1_cols[5]:
            with st.container(border=True):
                avg_deal_value = graph_drawing.format_number(won_deals['Deal : Total Deal Value'].mean())
                st.metric('Average Deal Value', avg_deal_value)
    
   
        # Row 2: Revenue progress donut charts and sales trend visualization within a bordered container
        
        row2_cols = st.columns(6, gap='medium')
        with row2_cols[0]:
            with st.container(border=True):
                total_deal_value = graph_drawing.format_number(won_deals['Deal : Total Deal Value'].sum())
                st.metric('Total Deal Value', total_deal_value)
        with row2_cols[1]:
            with st.container(border=True):
                total_service_revenue = graph_drawing.format_number(won_deals['Deal Service Revenue'].sum())
                st.metric('Total Service Revenue', total_service_revenue)
        with row2_cols[2]:
            with st.container(border=True):
                total_software_revenue = graph_drawing.format_number(won_deals['Deal Software Revenue'].sum())
                st.metric('Total Software Revenue', total_software_revenue)
        with row2_cols[3]:
            with st.container(border=True):
                total_support_revenue = graph_drawing.format_number(won_deals['Deal Support Revenue'].sum())
                st.metric('Total Support Revenue', total_support_revenue)
        with row2_cols[4]:
            with st.container(border=True):
                total_managed_service_revenue = graph_drawing.format_number(won_deals['Deal Managed Service Revenue'].sum())
                st.metric('Total AMS Revenue', total_managed_service_revenue)
        with row2_cols[5]:
            with st.container(border=True):
                avg_deal_value = graph_drawing.format_number(won_deals['Deal : Total Deal Value'].mean())
                st.metric('Average Deal Value', avg_deal_value)
        
        st.markdown('<div> </div>', unsafe_allow_html=True)
        row3_cols = st.columns((1, 4), gap='medium', vertical_alignment="center")
        with row3_cols[0]:
            with st.container(height=680,border=True):
                # Calculate the revenue data
                # revenue_data = graph_drawing.calculate_revenue_data(won_deals, sales_targets)
                
                # # Create individual donut charts for progress
                # total_chart = graph_drawing.create_donut_chart(revenue_data['total_progress'], 'Total Revenue Progress', ['#27AE60', '#12783D'])
                # recurring_chart = graph_drawing.create_donut_chart(revenue_data['recurring_progress'], 'Recurring Revenue Progress', ['#29b5e8', '#155F7A'])
                # non_recurring_chart = graph_drawing.create_donut_chart(revenue_data['non_recurring_progress'], 'Non-Recurring Revenue Progress', ['#E74C3C', '#781F16'])
                # st.markdown("##### Actual vs Target")
                # # Display charts aligned properly
                # st.write('Total Deals')
                # st.altair_chart(total_chart)
                # st.write('Recurring Deals')
                # st.altair_chart(recurring_chart)
                # st.write('Non-Recurring Deals')
                # st.altair_chart(non_recurring_chart)
                st.write("Hello")
            
        with row3_cols[1]:
            # Sales trend visualization
            metrics = [
                'Deal : Total Deal Value', 
                'Deal : Total Cost',
                'Deal : Total Gross Margin (GM)',
                'Deal Software Revenue',
                'Deal Software Cost',
                'Deal Retained Software Revenue',
                'Deal Support Revenue',
                'Deal Support Cost',
                'Deal Retained Support Revenue',
                'Deal Service Revenue',
                'Deal Service Cost',
                'Deal Retained Service Revenue',
                'Deal Cons Days',
                'Deal PM Days',
                'Deal PA Days',
                'Deal Technical Days',
                'Deal Hosting Revenue',
                'Deal Managed Service Revenue',
                'Deal Managed Service Cost'
            ]
            with st.container(height=680,border=True):
                st.markdown("### Sales Trend")
                # fig = graph_drawing.visualize_metric_over_time(won_deals, metrics)
                # st.plotly_chart(fig, use_container_width=True)

def main_process_deals_data_old_way(deals_data):
    """
    Processes the deals data by combining product columns into a single 'Deal Product' column
    and filtering based on vendor selection in the old format where each deal is represented by one row.

    Parameters:
    deals_data (DataFrame): The original DataFrame containing deals with multiple 'Deal : Product' columns.

    Returns:
    DataFrame: Processed DataFrame with combined 'Deal Product' column.
    list: Unique products available across 'Deal : Product' columns.
    """
    if not deals_data.empty:
        # Combine 'Deal : Product 1' to 'Deal : Product 4' into a single 'Deal Product' column
        deals_data['Deal Product'] = deals_data[
            ['Deal : Product 1', 'Deal : Product 2', 'Deal : Product 3', 'Deal : Product 4']
        ].fillna('').apply(
            lambda x: ', '.join([item for item in x if item != '']),
            axis=1
        )

        # Clean up any leading/trailing commas or spaces
        deals_data['Deal Product'] = deals_data['Deal Product'].str.strip(', ')
    else:
        st.error("No data available after filtering.")

    # Extract unique products from all 'Deal : Product' columns, excluding NaN values
    unique_products = pd.concat([
        deals_data['Deal : Product 1'],
        deals_data['Deal : Product 2'],
        deals_data['Deal : Product 3'],
        deals_data['Deal : Product 4']
    ]).dropna().unique()

    return deals_data, unique_products

def main_filter_deals_old_way(deals_data):
    """
    Filters the deals data based on selected vendors in the old format.

    Parameters:
    deals_data (DataFrame): The processed DataFrame containing the combined 'Deal Product' column.
    unique_products (list): List of unique products extracted from 'Deal : Product' columns.
    selected_vendors (list): List of selected vendor options.

    Returns:
    DataFrame: Filtered DataFrame based on selected vendors and products.
    """
    # # Define product filtering logic based on selected vendors
    # product_options = [
    #     product for product in unique_products
    #     if ('Infor' in product and 'Infor' in selected_vendors)
    #     or ('TRG' in product and 'TRG' in selected_vendors)
    #     or ('Infor' not in product and 'TRG' not in product and 'Others' in selected_vendors)
    # ]
    # product_options = sorted(product_options)

    # # # Filter the deals based on selected filters
    # deals_data_filtered = data_handling.filter_by_products(deals_data, product_options)

    # Initialize new columns in the dataframe with 0 for each row
    new_columns = [
        'Deal Software Revenue', 'Deal Software Cost', 'Deal Retained Software Revenue', 
        'Deal Support Revenue', 'Deal Support Cost', 'Deal Retained Support Revenue', 
        'Deal Service Revenue', 'Deal Service Cost', 'Deal Retained Service Revenue', 
        'Deal Cons Days', 'Deal PM Days', 'Deal PA Days', 'Deal Technical Days', 
        'Deal Hosting Revenue','Deal Managed Service Revenue', 'Deal Managed Service Cost'
    ]
    for col in new_columns:
        deals_data[col] = 0

    # Populate product values based on filtered options
    deals_data_filtered = data_handling.get_product_values(deals_data)

    return deals_data_filtered

def main_process_deals_data_new_way(deals_data):
    """
    Processes the deals data by creating a new DataFrame with each product in a separate row
    and combining it with the original deals data.

    Parameters:
    deals_data (DataFrame): The original DataFrame containing deals with multiple 'Deal : Product' columns.

    Returns:
    DataFrame: Merged DataFrame with each product in a separate row.
    list: Unique products available in the combined 'Deal Product' column.
    """
    
    # Create a new dataframe with each product in a separate row
    products_data = data_handling.create_product_value_dataframe_new(deals_data)
    # Merge products_data with deals_data
    deals_product_data = merge_products_deals_data(products_data, deals_data)

    return deals_product_data

def main_filter_deals_new_way(deals_product_data, unique_products, selected_vendors):
    """
    Filters the new deals data format based on selected vendors.

    Parameters:
    deals_product_data (DataFrame): The processed DataFrame with each product in a separate row.
    unique_products (list): List of unique products extracted from 'Deal Product' column.
    selected_vendors (list): List of selected vendor options.

    Returns:
    DataFrame: Filtered DataFrame based on selected vendors and products.
    """
    # Define product filtering logic based on selected vendors
    product_options = [
        product for product in unique_products
        if ('Infor' in product and 'Infor' in selected_vendors)
        or ('TRG' in product and 'TRG' in selected_vendors)
        or ('Infor' not in product and 'TRG' not in product and 'Others' in selected_vendors)
    ]
    product_options = sorted(product_options)
    st.write("Before filter product: ",deals_product_data.shape[0])
    # Filter the dataframe based on "Deal Product" column using product_options
    deals_data_filtered = deals_product_data[deals_product_data['Deal Product'].isin(product_options)]
    st.write("After filter product: ",deals_data_filtered.shape[0])
    return deals_data_filtered

    
st.set_page_config(
    page_title="TRG Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

make_sidebar()

# Load sales target data
with open("config/sales_targets.yaml", 'r') as file:
    sales_targets = yaml.safe_load(file)
    
with open("config/themes.yaml", "r") as file:
    themes = yaml.safe_load(file)
    
# Load column re-mapping to match logic after import
with open("config/col_mapping.yaml", 'r') as file:
    column_mapping = yaml.safe_load(file)

# Sidebar theme selection
selected_theme_name = st.sidebar.selectbox("Select Theme", list(themes.keys()))

# Get the colors for the selected theme
theme_colors = themes[selected_theme_name]

# Example usage of the theme colors
PRIMARY_COLORS = theme_colors["PRIMARY_COLORS"]
SUPPORTING_COLORS = theme_colors["SUPPORTING_COLORS"]
POSITIVE_COLOR = theme_colors["POSITIVE_COLOR"]
NEGATIVE_COLOR = theme_colors["NEGATIVE_COLOR"]
NEUTRAL_COLOR = theme_colors["NEUTRAL_COLOR"]

data_handling = Data_Handling()
graph_drawing = Graph_Drawing(theme_colors)

# Mandatory columns
required_columns = ['Deal : Name', 'Deal : Account name', 'Deal : Closed date','Deal : Expected close date', 'Deal : Total Deal Value', 'Deal : Probability (%)',
                          'Deal : Deal stage','Deal : Owner','Deal : Project type','Deal : Source','Deal : Total Cost','Deal : Gross Margin (GM)', 'Deal : Age (in days)',
                          'Deal : Tentative start date/MSD', 'Deal : Expected go live date/MED', 'Deal : Type of Renewal',
                          'Deal : Product 1', 'Deal : Product 2', 'Deal : Product 3', 'Deal : Product 4',
                          'Deal : Software revenue: Product 1','Deal : Software revenue: Product 2','Deal : Software revenue: Product 3','Deal : Software revenue: Product 4',
                          'Deal : Software cost: Product 1','Deal : Software cost: Product 2','Deal : Software cost: Product 3','Deal : Software cost: Product 4',
                          'Deal : Support revenue: Product 1','Deal : Support revenue: Product 2','Deal : Support revenue: Product 3','Deal : Support revenue: Product 4',
                          'Deal : Support cost: Product 1','Deal : Support cost: Product 2','Deal : Support cost: Product 3','Deal : Support cost: Product 4',
                          'Deal : Service revenue: Product 1','Deal : Service revenue: Product 2','Deal : Service revenue: Product 3','Deal : Service revenue: Product 4',
                          'Deal : Service cost: Product 1','Deal : Service cost: Product 2','Deal : Service cost: Product 3','Deal : Service cost: Product 4',
                          'Deal : Cons days: Product 1','Deal : Cons days: Product 2','Deal : Cons days: Product 3','Deal : Cons days: Product 4',
                          'Deal : Technical days: Product 1','Deal : Technical days: Product 2','Deal : Technical days: Product 3','Deal : Technical days: Product 4',
                          'Deal : PM days: Product 1','Deal : PM days: Product 2','Deal : PM days: Product 3','Deal : PM days: Product 4',
                          'Deal : PA days: Product 1','Deal : PA days: Product 2','Deal : PA days: Product 3','Deal : PA days: Product 4',
                          'Deal : Hosting revenue: Product 1','Deal : Hosting revenue: Product 2','Deal : Hosting revenue: Product 3','Deal : Hosting revenue: Product 4',
                          'Deal : Managed service revenue: Product 1','Deal : Managed service revenue: Product 2','Deal : Managed service revenue: Product 3','Deal : Managed service revenue: Product 4',
                          'Deal : Managed service cost: Product 1','Deal : Managed service cost: Product 2','Deal : Managed service cost: Product 3','Deal : Managed service cost: Product 4']

# File uploaderƒ
st.sidebar.header("Configuration")
#file_deals = st.sidebar.file_uploader("Upload Deals Data File", type=['csv', 'xlsx'])
#file_acct = st.sidebar.file_uploader("Upload Accounting Data File", type=['csv','xlsx'])
# Add filter View By on the sidebar
view_by = st.sidebar.selectbox(
    "View Trend By",
    options=["Monthly", "Quarterly"],
    help="Select option to view the chart over time by month or quarter"
)

probability = st.sidebar.number_input(
    "Deal Probability (%)",
    min_value=10,
    max_value=100,
    value=80,
    step=10,
    help="Enter deal probability % to view forecast deals"
)

# Queries to load Deals and Account data
deal_query = "SELECT * FROM FSAllDealsExtraction"

# Load data using SQL queries
deals_data = run_query(deal_query)

# Load and validate data
if deals_data is not None:
    # Add "Deal : " prefix to all deal columns
    deals_data.columns = ["Deal : " + col for col in deals_data.columns]
    deals_data.rename(columns=column_mapping, inplace=True)

    #Duong update 03 Nov 2024 - start
    deals_data = preprocess_data(deals_data)
    # Sidebar Filters
    st.sidebar.header("Filters")

    selected_year = data_handling.year_selection(deals_data)
    
    new_deals_data1 = main_process_deals_data_new_way(deals_data)
    
    # Get the current month
    current_month = dt.datetime.now().month
    current_year = dt.datetime.now().year
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sales Overview", "Pipeline Health", "Revenue Breakdown", "Sales Performance by Owner", "Product Performance"])
    
    
    # Styling for containers using Streamlit's "st.container" and "st.columns"
    with tab1: #Sales Overview
        sales_overview_tab(new_deals_data1, selected_year)

    with tab2: #Sales Actual dashboard
        #sale over view dashboard
        won_deals = filter_won_deals(new_deals_data1, selected_year)
        pre_won_deals = filter_won_deals(new_deals_data1, selected_year - 1)
        open_deals = filter_open_deals(new_deals_data1, selected_year)
        pre_open_deals = filter_open_deals(new_deals_data1, selected_year - 1)
        pipeline_deals = filter_pipeline_backdate_deals(new_deals_data1, selected_year)
        pre_pipeline_deals = filter_pipeline_backdate_deals(new_deals_data1, selected_year - 1)
        pre_pipeline_deals = add_last_year_columns(pipeline_deals, pre_pipeline_deals)
        new_deals_year = filter_new_deals(new_deals_data1, selected_year)
        opps_deals_probability = filter_pipeline_deals_probability(new_deals_data1, selected_year, probability)
        all_opps_deals = filter_pipeline_deals_probability(new_deals_data1, selected_year, 0)
        
        # filter deals in pipeline expected to close within the selected year
        expect_to_close_deals = filter_expected_close_deals(new_deals_data1, selected_year)
        pre_expect_to_close_deals = filter_expected_close_deals(new_deals_data1, selected_year-1)
        pre_expect_to_close_deals = add_last_year_columns(expect_to_close_deals, pre_expect_to_close_deals)
        
        # Calculate total deal values
        cur_won_value = won_deals['Deal : Total Deal Value'].sum()
        pre_won_value = pre_won_deals['Deal : Total Deal Value'].sum()
        delta_won = cur_won_value - pre_won_value

        cur_open_value = open_deals['Deal : Total Deal Value'].sum()
        pre_open_value = pre_open_deals['Deal : Total Deal Value'].sum()
        delta_open = cur_open_value - pre_open_value
        
        # Get the count of deals created in the specified range
        new_deals_year_count = new_deals_year.shape[0]
        # Get the total deal value for the specified range
        new_deals_year_value = new_deals_year['Deal : Total Deal Value'].sum()
        
        # Calculate conversion rate (Won Deals / Open Deals * 100)
        won_deals_count = won_deals.shape[0]
        open_deals_count = open_deals.shape[0]
        conversion_rate = (won_deals_count / new_deals_year_count * 100) if new_deals_year_count > 0 else 0
        
        # Combine them to form 'yyyy-mm' format
        current_year_month = f"{selected_year}-{current_month}"

        # Add a 'Created Month' column to the dataframe in 'yyyy-mm' format
        new_deals_year['Created Month'] = new_deals_year['Deal : Created at'].dt.strftime('%Y-%m')

        # Further filter new_deals_year by current year and month
        new_deals_month = new_deals_year[new_deals_year['Created Month'] == current_year_month]
        
        # # Add a 'Created Month' column to the dataframe in 'yyyy-mm' format
        # new_deals_year['Created Month'] = new_deals_year['Deal : Created at'].dt.strftime('%Y-%m')
        # # Further filter new_deals_year by current month and probability
        # new_deals_month = new_deals_year[new_deals_year['Created Month'] == current_month]

        # Get the count and total deal value for the filtered deals
        deals_created_month_count = new_deals_month.shape[0]
        total_deal_month_value = new_deals_month['Deal : Total Deal Value'].sum()
        
        # Further filter opps_deals_probability by probability
        opps_deals_probability_count = opps_deals_probability.shape[0]
        # Get the total deal value for the specified range
        opps_deals_probability_value = opps_deals_probability['Deal : Total Deal Value'].sum()
        
        # Create main row of three columns
        row1_cols = st.columns((2), gap="small", vertical_alignment="center")
        
        # Column [1]: Container for the chart
        with row1_cols[0]:
            with st.container(height=450,border=True):
                # Plot the graph                    
                fig2 = graph_drawing.visualize_pipeline_deal_values_current_last_year(pre_expect_to_close_deals, view_by, f"Pipeline Deals expected to close: {selected_year} vs {selected_year-1}")
                st.plotly_chart(fig2, use_container_width=True)

        # Column [2]: Container for three additional metrics
        with row1_cols[1]:
            with st.container(height=450,border=True):
                # Assuming your dataframe with 'Deal : Probability' and 'Deal : Total Deal Value' columns.
                funnel_deals_year = filter_pipeline_deals_probability(new_deals_data1, selected_year, 10)
                
                # Add a 'Created Month' column to the dataframe in 'yyyy-mm' format
                funnel_deals_year['Expected Close Month'] = funnel_deals_year['Deal : Expected close date'].dt.strftime('%Y-%m')

                # Further filter funnel_deals_year by current year and month
                funnel_deals_month = funnel_deals_year[funnel_deals_year['Expected Close Month'] == current_year_month]
                
                # Calculate number of deals and total deal value for each combination of probability and deal stage
                funnel_data = funnel_deals_month.groupby(['Deal : Probability (%)', 'Deal : Deal stage']).agg(
                    number_of_deals=('Deal : Probability (%)', 'size'),
                    total_deal_value=('Deal : Total Deal Value', 'sum')
                ).reset_index()


                # Sort the data for the funnel chart in descending probability order
                funnel_data = funnel_data.sort_values(by='Deal : Probability (%)', ascending=False)
    
                        
                fig = visual_funnel_chart_pipeline(funnel_data)
                st.plotly_chart(fig, use_container_width=True)
                    
        # Create main row of three columns
        row2_cols = st.columns((1, 3, 2), gap="small", vertical_alignment="center")

        # Custom CSS to add space between containers
        st.markdown(
            """
            <style>
            .metric-container {
                margin-bottom: 30px;  /* Adjust the margin as needed */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Use st.markdown to apply the custom CSS
        with row2_cols[0]:
            with st.container(height=450, border=True):  # No class_ argument here
                # Custom CSS for font size
                st.markdown(
                    """
                    <style>
                    .header {
                        font-size: 20px;
                        font-weight: bold;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Applying the custom CSS to the header
                st.markdown('<p class="header">Total Deal Value</p>', unsafe_allow_html=True)

                
                # Add metrics with a custom class applied via markdown
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("All Deals", f"$ {graph_drawing.format_number(cur_open_value)}", graph_drawing.format_number(delta_open), delta_color="normal",help="YTD total value of created deals which have probability > 0")
                #st.markdown('</div>', unsafe_allow_html=True)

                #st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Won Deals", f"$ {graph_drawing.format_number(cur_won_value)}", graph_drawing.format_number(delta_won), delta_color="normal", help="YTD won deals total value")
                #st.markdown('</div>', unsafe_allow_html=True)

                # Add Conversion Rate metric
                #st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Conversion Rate", f"{conversion_rate:.2f}%", help="Ratio of won deals vs all created deals within the year")
                st.markdown('</div>', unsafe_allow_html=True)


        # Column [1]: Container for the chart
        with row2_cols[1]:
            with st.container(height=450,border=True):
                # Plot the graph
                fig1 = graph_drawing.visualize_pipeline_deal_values_current_last_year(pre_pipeline_deals, view_by,f"New Deals Accumuated Value Backdate: {selected_year} vs {selected_year-1}")
                st.plotly_chart(fig1, use_container_width=True)
                
    

        # Column [2]: Container for three additional metrics
        with row2_cols[2]:
            with st.container(height=450,border=True):
                with st.container(height=130, border=True):
                    # visualize deals created count and total deal value in current month and by probability
                    cols =  st.columns(2)
                
                    cols[0].metric(f"Deals in pipeline from {probability}%", opps_deals_probability_count, help=f"Deals expected to close in the selected year which have probability >= {probability}")
                    cols[1].metric(" ", f"$ {graph_drawing.format_number(opps_deals_probability_value)}")

                with st.container(height=130, border=True):
                    cols =  st.columns(2)
                    # visualize deals created count and total deal value of current month
                    cols[0].metric("Deals created MTD", deals_created_month_count, help="MTD created deals, including lost deals")
                    cols[1].metric(" ", f"$ {graph_drawing.format_number(total_deal_month_value)}")
                

                with st.container(height=130, border=True):
                    # visualize deals created count and total deal value of selected year
                    cols =  st.columns(2)
                    cols[0].metric("Deals created this year", new_deals_year_count, help="All Deals created this year, including won and lost deals")
                    cols[1].metric(" ", f"$ {graph_drawing.format_number(new_deals_year_value)}")
    
    with tab3: #breakdown revenue
        won_revenue_breakdown_tab(new_deals_data1, selected_year)
        
    with tab4:
        # Sales Leaderboard
        st.header("Sales Leaderboard")
        graph_drawing.tab4_visualize_sales_leaderboard(won_deals, sales_targets)
        graph_drawing.tab4_visualize_sales_performance_by_owner(deals_data, selected_year)
else:
    st.info("Cannot get SQL Server connection")
