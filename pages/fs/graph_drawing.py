import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import plotly.express as px
import squarify
import streamlit as st
import plotly.graph_objects as go
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import yaml
import altair as alt



class Graph_Drawing():
    def format_number(self, num):
        if num > 1000000:
            if not num % 1000000:
                return f'{num // 1000000:,} M'
            return f'{round(num / 1000000, 1):,} M'
        return f'{num / 1000:,.0f} K'
    
    def __init__(self, theme_colors=None):
            """
            Initialize the Graph_drawing class with theme colors.

            Parameters:
            - theme_colors (dict, optional): Dictionary of theme colors with keys like PRIMARY_COLORS, SUPPORTING_COLORS, etc.
            """
            if theme_colors is None:
                # Nếu không cung cấp theme_colors, sử dụng giá trị mặc định
                theme_colors = {
                    "PRIMARY_COLORS": ["#FF5733", "#33FF57", "#3357FF"],
                    "SUPPORTING_COLORS": ["#FF5733", "#C70039", "#900C3F"]
                }
            
            # Gán theme_colors cho thuộc tính của lớp
            self.theme_colors = theme_colors

    def display_colors(self):
        """
        Display the stored theme colors.
        """
        print("Theme Colors:")
        for color_type, colors in self.theme_colors.items():
            print(f"{color_type}: {', '.join(colors)}")

    # Ví dụ sử dụng lớp Graph_Drawing:

    # Trường hợp 1: Cung cấp theme_colors
    theme_colors = {
        "PRIMARY_COLORS": ["#FF5733", "#33FF57", "#3357FF"],
        "SUPPORTING_COLORS": ["#FF5733", "#C70039", "#900C3F"]
    }
        
    @st.cache_data(show_spinner=False)
    def rfm_component_graph(_self, df_rfm, rfm_component, color):
        plt.figure()
        sns.histplot(df_rfm[rfm_component], bins=30, kde=True, color=color, edgecolor='pink')

        plt.xlabel(rfm_component)
        plt.ylabel('Number of Customers')
        plt.title(f"Number of Customers based on {rfm_component}")
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return plt.gcf()
    
    #Duong update the function treemap_drawing
    @st.cache_data(show_spinner=False)
    def treemap_drawing(_self, cluster_centers):
        plt.figure()
        total_customers = cluster_centers['Cluster Size'].sum()

        sns.set_style(style="whitegrid")  # Set Seaborn plot style

        sizes = cluster_centers['Cluster Size']  # Proportions of the categories

        # Generate random colors for each unique cluster
        unique_clusters = cluster_centers['Cluster'].unique()
        random.seed(50)  # Optional: Set seed for reproducibility
        colors = {cluster: f'#{random.randint(0, 0xFFFFFF):06x}' for cluster in unique_clusters}

        # Draw the treemap
        squarify.plot(
            sizes=sizes,
            alpha=0.6,
            color=[colors[cluster] for cluster in cluster_centers['Cluster']],
            label=cluster_centers['Cluster']
        ).axis('off')

        # Creating custom legend
        handles = []
        for i in cluster_centers.index:
            label = '{} \n{:.0f} days \n{:.0f} transactions \n${:,.0f} \n{:.0f} Customers ({:.1f}%)'.format(
                cluster_centers.loc[i, 'Cluster'], cluster_centers.loc[i, 'Recency'], cluster_centers.loc[i, 'Frequency'],
                cluster_centers.loc[i, 'Monetary'], cluster_centers.loc[i, 'Cluster Size'],
                cluster_centers.loc[i, 'Cluster Size'] / total_customers * 100
            )
            handles.append(Patch(facecolor=colors[cluster_centers.loc[i, 'Cluster']], label=label))

        
        
        plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')
        plt.title('RFM Segmentation Treemap', fontsize=20)

        return total_customers, plt.gcf()
        
    @st.cache_data(show_spinner=False)
    def scatter_3d_drawing(_self, df_kmeans):
        df_scatter = df_kmeans.copy()
        
        # Select relevant columns
        df_review = df_scatter[['Recency', 'Frequency', 'Monetary', 'Ranking']]
        
        # Ensure the columns are of type float
        df_scatter[['Recency', 'Frequency', 'Monetary']] = df_review[['Recency', 'Frequency', 'Monetary']].astype(float)
        
        # Define a custom color sequence
        custom_colors = ['#e60049', '#0bb4ff', '#9b19f5', '#00bfa0', '#e6d800', '#8D493A', '#55AD9B', '#7ED7C1', '#EA8FEA']
        
        # Create the 3D scatter plot
        fig = px.scatter_3d(
            df_scatter, 
            x='Recency', 
            y='Frequency', 
            z='Monetary', 
            color='Ranking', 
            opacity=0.7,
            width=600,
            height=500,
            color_discrete_sequence=custom_colors
        )
        
        # Update marker size and text position
        fig.update_traces(marker=dict(size=6), textposition='top center')
        
        # Update layout template
        fig.update_layout(template='plotly_white')
        
        return fig
    
    @st.cache_data(show_spinner=False)
    def pipeline_trend(_self, df, start_date, end_date):
        """Generate the trend of total deal value and deal count in the pipeline grouped by month."""
        
        # Ensure 'Deal : Created at' and 'Deal : Closed date' columns are in datetime format
        df['Deal : Created at'] = pd.to_datetime(df['Deal : Created at'], errors='coerce')
        df['Deal : Closed date'] = pd.to_datetime(df['Deal : Closed date'], errors='coerce')


        # Generate a range of month-ends from start to end
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')

        # Convert DatetimeIndex to a list to allow appending
        date_range_list = date_range.tolist()

        # Adjust the time to 23:59:59 for each date in the list
        date_range_list = [date.replace(hour=23, minute=59, second=59) for date in date_range_list]

        # Convert end_date to a pandas Timestamp if it is not already
        end_date_ts = pd.Timestamp(end_date).replace(hour=23, minute=59, second=59)

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

            # Count deals created in the current month (between month_start and month_end)
            deals_created = df[
                (df['Deal : Created at'] >= month_start) &  
                (df['Deal : Created at'] <= month_end)
            ]
            deal_created_count = deals_created['Deal : id'].nunique()
            
            deals_closed = df[
                (df['Deal : Closed date'] >= month_start) &  
                (df['Deal : Closed date'] <= month_end) &
                (df['Deal : Deal stage'] == 'Won')
            ]
            deal_closed_count = deals_closed['Deal : id'].nunique()

            return total_value, deal_created_count, deal_closed_count

        # Initialize lists to store results
        months = []
        as_at_date = []
        total_values = []
        deal_created_counts = []
        deal_closed_counts = []

        # Calculate total deal value and deal count for each month in the date range
        for month_end in date_range:
            total_value, deal_created_count, deal_closed_count = pipeline_value_and_count_at_month(df, month_end)
            months.append(month_end.strftime('%Y-%m'))
            as_at_date.append(month_end)
            total_values.append(total_value)  # Store total value
            deal_created_counts.append(deal_created_count)  # Store deal count
            deal_closed_counts.append(deal_closed_count)  # Store deal count

        
        # Create a DataFrame to return
        trend_df = pd.DataFrame({
            'Month': months,
            'As At Date': as_at_date,
            'Total Deal Value': total_values,
            'Deals Created Count': deal_created_counts,
            'Deals Closed Count': deal_closed_counts
        })
    
        return trend_df

    @st.cache_data(show_spinner=False)
    def plot_pipeline_trend(_self, trend_df, start_month, end_month):
        """Plots 'Deals Closed Count' and 'Deals Created Count' on a bar chart, and 'Total Deal Value' on a line chart."""
        
        # Ensure the 'Month' column is in datetime format
        #trend_df['As At Date'] = pd.to_datetime(trend_df['As At Date'])

        # Filter the DataFrame based on the selected month range
        filtered_trend_df = trend_df[
        (trend_df['Month'] >= start_month) & 
        (trend_df['Month'] <= end_month)
    ]
        st.write('Pipeline Data by Month')
        st.dataframe(filtered_trend_df)
        
        # Plot 1: Total Deal Value (line chart)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(filtered_trend_df['As At Date'], filtered_trend_df['Total Deal Value'], marker='o', linestyle='-', color='b', label='Total Deal Value')
        ax1.set_title('Total Deal Value in Pipeline by Month')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Total Deal Value')
        ax1.set_xticks(filtered_trend_df['As At Date'])
        ax1.set_xticklabels(filtered_trend_df['As At Date'].dt.strftime('%Y-%m'), rotation=45)
        ax1.grid(True)
        ax1.legend()

        # Display the first figure
        st.pyplot(fig1)

        # Plot 2: Deals Closed and Created Count (bar chart)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        bar_width = 10  # Width of the bars, reduced for better separation
        
        # Plot Deals Closed Count
        ax2.bar(filtered_trend_df['As At Date'] - pd.DateOffset(days=4),  # Offset bars to the left slightly
                filtered_trend_df['Deals Closed Count'], 
                width=bar_width, color='r', alpha=0.7, label='Deals Closed Count')

        # Plot Deals Created Count
        ax2.bar(filtered_trend_df['As At Date'] + pd.DateOffset(days=5),  # Offset bars to the right slightly
                filtered_trend_df['Deals Created Count'], 
                width=bar_width, color='g', alpha=0.7, label='Deals Created Count')

        ax2.set_title('Deals Closed and Created Count by Month')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Count')
        ax2.set_xticks(filtered_trend_df['As At Date'])
        ax2.set_xticklabels(filtered_trend_df['As At Date'].dt.strftime('%Y-%m'), rotation=45)
        ax2.grid(True)
        
        # Add a legend for the bar chart
        ax2.legend()

        # Display the second figure
        st.pyplot(fig2)

    #Duong updating
    @st.cache_data
    def visualize_actual_vs_target_sales_gauge(_self,won_deals, sales_targets):
        """
        Visualize actual 'Deal : Total Deal Value' of 'Won' Deal, grouped by Owner, compared to target revenue using gauge charts.

        Parameters:
        - won_deals (pd.DataFrame): The DataFrame containing won deal data, including 'Deal : Owner', 'Deal : Total Deal Value', and 'Deal : Deal stage'.
        - sales_targets (dict): A dictionary containing target sales for each owner.

        Returns:
        - fig: A Plotly figure object containing the gauge charts for each sales representative.
        """
        # Group the data by 'Deal : Owner' and sum the 'Deal : Total Deal Value'
        owner_deal_value = won_deals.groupby('Deal : Owner', as_index=False)['Deal : Total Deal Value'].sum()

        # Extract revenue targets from sales_targets
        revenue_targets = sales_targets.get('revenue_targets', {})
        
        # Ensure sales_targets is a Series for mapping
        sales_targets_series = pd.Series(revenue_targets)

        # Add a 'Target' column by mapping each owner's name to the corresponding target sales
        owner_deal_value['Target'] = owner_deal_value['Deal : Owner'].map(sales_targets_series)
        
        # Remove sales reps who do not have defined revenue targets
        owner_deal_value = owner_deal_value[owner_deal_value['Deal : Owner'].isin(revenue_targets.keys())]

        # Ensure all owners with targets are included (even if they have no won deals)
        for owner in revenue_targets.keys():
            if owner not in owner_deal_value['Deal : Owner'].values:
                # Create a new DataFrame for the missing owner with no won deals
                new_row = pd.DataFrame({
                    'Deal : Owner': [owner], 
                    'Deal : Total Deal Value': [0], 
                    'Target': [revenue_targets[owner]]
                })
                owner_deal_value = pd.concat([owner_deal_value, new_row], ignore_index=True)

        # Fill NaN targets with 0 to avoid division errors
        owner_deal_value['Target'].fillna(0, inplace=True)

        # Calculate the percentage of won deal value vs target
        owner_deal_value['Percentage_Won'] = owner_deal_value.apply(
            lambda row: (row["Deal : Total Deal Value"] / row['Target']) * 100 if row['Target'] > 0 else 0,
            axis=1
        )

        # Initialize a list to store each owner's gauge chart
        figures = []

        # Create a gauge chart for each owner
        for index, row in owner_deal_value.iterrows():
            owner = row['Deal : Owner']
            won_value = row['Deal : Total Deal Value']
            target_value = row['Target']
            percentage_won = row['Percentage_Won']

            # Create a gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=percentage_won,
                title={'text': f"{owner}<br>Progress to Target", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "steelblue"},
                    'steps': [
                        {'range': [0, 100], 'color': "lightgray"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

    def visualize_actual_vs_target_sales(_self, won_deals, sales_targets):
        """
        Visualize actual 'Deal : Total Deal Value' of 'Won' Deal, grouped by Owner, compared to target revenue.

        Parameters:
        - won_deals (pd.DataFrame): The DataFrame containing won deal data, including 'Deal : Owner', 'Deal : Total Deal Value', and 'Deal : Deal stage'.
        - sales_targets (dict): A dictionary containing target sales for each owner.

        Returns:
        - None: Displays the comparison visualization in Streamlit.
        """
        # Group the data by 'Deal : Owner' and sum the 'Deal : Total Deal Value'
        owner_deal_value = won_deals.groupby('Deal : Owner', as_index=False)['Deal : Total Deal Value'].sum()

        # Extract revenue targets from sales_targets
        revenue_targets = sales_targets.get('revenue_targets', {})
        
        # Ensure sales_targets is a Series for mapping
        sales_targets_series = pd.Series(revenue_targets)

        # Add a 'Target' column by mapping each owner's name to the corresponding target sales
        owner_deal_value['Target'] = owner_deal_value['Deal : Owner'].map(sales_targets_series)
        
        # Remove sales reps who do not have defined revenue targets
        owner_deal_value = owner_deal_value[owner_deal_value['Deal : Owner'].isin(revenue_targets.keys())]

        # Ensure all owners with targets are included (even if they have no won deals)
        for owner in revenue_targets.keys():
            if owner not in owner_deal_value['Deal : Owner'].values:
                # Create a new DataFrame for the missing owner with no won deals
                new_row = pd.DataFrame({
                    'Deal : Owner': [owner], 
                    'Deal : Total Deal Value': [0], 
                    'Target': [revenue_targets[owner]]
                })
                owner_deal_value = pd.concat([owner_deal_value, new_row], ignore_index=True)

        # Fill NaN targets with 0 to avoid division errors (though all should have targets by now)
        owner_deal_value['Target'] = owner_deal_value['Target'].fillna(0)
 


        # Convert the 'Target' column to numeric
        owner_deal_value['Target'] = pd.to_numeric(owner_deal_value['Target'], errors='coerce')

        # Calculate the percentage of won deal value vs target
        owner_deal_value['Percentage_Won'] = owner_deal_value.apply(
            lambda row: (row["Deal : Total Deal Value"] / row['Target']) * 100 if row['Target'] > 0 else 0,
            axis=1
        )

        # Sort the owners by 'Percentage_Won' in descending order
        owner_deal_value = owner_deal_value.sort_values('Percentage_Won', ascending=False)

        # Visualize the data in Streamlit
        for index, row in owner_deal_value.iterrows():
            owner = row['Deal : Owner']
            won_value = row['Deal : Total Deal Value']
            target_value = row['Target']
            percentage_won = row['Percentage_Won']

            # Display owner name, progress bar, and target sales in three columns
            col1, col2, col3 = st.columns([1, 5, 1])  # Adjust column width ratios
            
            with col1:
                # Left-aligned owner name with no wrapping and small space between rows
                st.write(f"<div style='text-align: left; white-space: nowrap; margin-bottom: 10px;'>{owner}</div>", unsafe_allow_html=True)

            with col2:
                # Display the progress bar with larger size and percentage on it
                bar_color = 'steelblue' if percentage_won > 0 else 'lightgray'
                progress_text = f"{percentage_won:.2f}%"

                # Display bar with white text on progress bar for better contrast
                st.write(f"""
                    <div style="background-color: lightgray; height: 30px; position: relative; border-radius: 4px;">
                        <div style="width: {min(percentage_won, 100)}%; background-color: {bar_color}; height: 100%; border-radius: 4px; text-align: center; color: white;">
                            {progress_text}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                # Right-aligned target sales value using format_number function
                formatted_target_value = _self.format_number(target_value)
                st.write(f"<div style='text-align: left;'>{formatted_target_value}</div>", unsafe_allow_html=True)

    @st.cache_data        
    def create_donut_chart(_self, progress, title, color):
        """
        Create donut charts for revenue target achievement and progress using Altair.
        
        Parameters:
        revenue_data: Dictionary containing calculated revenue percentages.
        
        Returns:
        Altair charts showing total revenue, recurring, and non-recurring progress as donut charts.
        """
        remaining_percentage = 100 - progress if progress <= 100 else 0
        chart_color = color

        source = pd.DataFrame({
            "Topic": ['', title],
            "% value": [remaining_percentage, progress]
        })

        source_bg = pd.DataFrame({
            "Topic": ['', title],
            "% value": [100, 0]
        })

        plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
            theta="% value",
            color=alt.Color("Topic:N",
                            scale=alt.Scale(
                                domain=[title, ''],
                                range=chart_color),
                            legend=None),
        ).properties(width=130, height=130)

        text = plot.mark_text(align='center', color=chart_color[0], font="Lato", fontSize=20, fontWeight=700).encode(
            text=alt.value(f'{progress:.1f}%')
        )

        plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
            theta="% value",
            color=alt.Color("Topic:N",
                            scale=alt.Scale(
                                domain=[title, ''],
                                range=chart_color),
                            legend=None),
        ).properties(width=130, height=130)

        return plot_bg + plot + text
    
    def plot_avg_days_to_close(self, df):
        # Filter out rows where 'Deal : Owner' contains 'Inactive'
        df_filtered = df[~df['Deal : Owner'].str.contains('Inactive', na=False)]
        
        # Calculate 'Days to Close'
        df_filtered['Days to Close'] = (df_filtered['Deal : Closed date'] - df_filtered['Deal : Created at']).dt.days
        
        # Calculate average 'Days to Close' by 'Deal : Owner'
        avg_days_to_close = df_filtered.groupby('Deal : Owner', as_index=False)['Days to Close'].mean()
        
        # Sort values from largest to smallest
        avg_days_to_close = avg_days_to_close.sort_values(by='Days to Close', ascending=False)
        
        # Plot the results
        fig = px.bar(avg_days_to_close, x='Deal : Owner', y='Days to Close')
        st.plotly_chart(fig)
        
    @st.cache_data
    def plot_deal_value_growth_rate(_self, df):       
        # Extract month from 'Deal : Closed date'
        df['Month'] = pd.to_datetime(df['Deal : Closed date']).dt.to_period('M')
        
        # Calculate total deal value growth by month
        deal_value_growth = df.groupby('Month', as_index=False)['Deal : Total Deal Value'].sum()
        
        # Format the month for better readability
        deal_value_growth['Month'] = deal_value_growth['Month'].dt.strftime('%Y-%m')
        
        # Plot the results
        fig = px.line(deal_value_growth, x='Month', y='Deal : Total Deal Value')
        st.plotly_chart(fig)
        
    #Function to visualize current year pipeline deal value compare to last year
    @st.cache_data
    def visualize_pipeline_deal_values_current_last_year(_self, df, view_by='Monthly', chart_title=''):
        # Convert 'Month' column to datetime if not already in datetime format
        df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')

        # Extract current and last year from the data
        last_year = df['Month'].dt.year.max()
        current_year = last_year + 1

        # Resample data based on the view_by selection (monthly or quarterly)
        if view_by == 'Monthly':
            # Resample by month and aggregate numeric columns
            data_resampled = df.set_index('Month').resample('M').sum(numeric_only=True).reset_index()

            # Generate a complete range of months from January to the max month in the data
            min_month = pd.Timestamp(f'{last_year}')
            max_month = data_resampled['Month'].max()

            # Generate the x-axis labels from January to max month of the current year
            x_axis = pd.date_range(start=min_month, end=max_month, freq='M')

            # Set the tick format for the x-axis
            xaxis_tickformat = '%b'
            xaxis_title = 'Month'

        elif view_by == 'Quarterly':
            # Resample by quarter and aggregate numeric columns
            data_resampled = df.set_index('Month').resample('Q').sum(numeric_only=True).reset_index()

            # Generate a complete range of quarters from Q1 to the max quarter in the data
            min_quarter = pd.Timestamp(f'{current_year}-01-01')
            max_quarter = data_resampled['Month'].max()

            # Generate the x-axis labels from Q1 to the max quarter of the current year
            x_axis = pd.date_range(start=min_quarter, end=max_quarter, freq='Q')

            # Set the tick format for the x-axis
            xaxis_tickformat = 'Q%q'
            xaxis_title = 'Quarter'

        else:
            raise ValueError("view_by must be either 'Monthly' or 'Quarterly'")

        # Create the bar chart using Plotly
        fig = go.Figure()

        # Add bars for 'Total Deal Value' (current year)
        fig.add_trace(go.Bar(
            x=x_axis,
            y=data_resampled['Current Year Total Deal Value'],
            name=f'{current_year}',
            marker_color='blue'
        ))

        # Add bars for 'Last Year Total Deal Value' (last year)
        fig.add_trace(go.Bar(
            x=x_axis,
            y=data_resampled['Total Deal Value'],
            name=f'{last_year}',
            marker_color='orange'
        ))

        # Update layout of the plot
        fig.update_layout(
            title={
                'text': f'{chart_title}',  # Add a title
                'font': {
                    'size': 20,  # Font size for the title (equivalent to st.markdown("#### Text value"))
                    'color': 'black'  # Optional: specify the color
                },
                'x': 0.7,  # Center the title
                'xanchor': 'right'  # Center anchor
            },
            xaxis_title=xaxis_title,
            yaxis_title='Total Deal Value',
            margin=dict(t=30),
            height=450,
            barmode='group',  # Group bars next to each other
            xaxis_tickformat=xaxis_tickformat,  # Format based on view_by
            template='plotly_white',
            xaxis_tickmode='array',
            xaxis_tickvals=x_axis,  # Set x-axis ticks manually
            xaxis_ticktext=[tick.strftime('%b') if view_by == 'Monthly' else f'Q{tick.quarter}' for tick in x_axis],  # Customize tick labels
            #legend_title="Year Comparison",
            #xaxis_tickangle=-45  # Rotate x-axis labels for better readability
        )

        return fig

    def tab1_plot_total_deal_value_by_project_type(_self, df, selected_year, theme_colors):
        """
        Plots a donut chart showing the 'Total Deal Value' by 'Deal : Project type' 
        for the selected year with each type's percentage of the total, using theme colors.

        Parameters:
        df (DataFrame): Data containing 'Year', 'Deal : Project type', and 'Total Deal Value'.
        selected_year (int): The year for which to display the data.
        theme_colors (dict): Dictionary containing color definitions.

        Returns:
        fig: Plotly Figure object.
        """
        
        # Define color variables
        PRIMARY_COLORS = theme_colors["PRIMARY_COLORS"]
        SUPPORTING_COLORS = theme_colors["SUPPORTING_COLORS"]
        POSITIVE_COLOR = theme_colors["POSITIVE_COLOR"]
        NEGATIVE_COLOR = theme_colors["NEGATIVE_COLOR"]
        NEUTRAL_COLOR = theme_colors["NEUTRAL_COLOR"]
        
        # Combine all colors into a single color sequence for the chart
        color_sequence = PRIMARY_COLORS + SUPPORTING_COLORS + [POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR]
        
        # Filter data for the selected year
        year_data = df[df['Year'] == selected_year]
        
        # Group by 'Deal : Project type' and sum the 'Total Deal Value'
        deal_value_by_type = year_data.groupby('Deal : Project type')['Deal : Total Deal Value'].sum().reset_index()
        
        # Create a donut chart using Plotly Express
        fig = px.pie(deal_value_by_type, 
                    names='Deal : Project type', 
                    values='Deal : Total Deal Value', 
                    hole=0.4,  # Creates the donut hole in the center
                    title=f'Won Deals by Project Type',
                    color_discrete_sequence=color_sequence)  # Apply custom color sequence

        # Update layout for better readability and presentation
        fig.update_traces(textinfo='percent+label')  # Show percentage and type on each slice
        fig.update_layout(showlegend=True)

        return fig

    def tab1_plot_total_deal_value_by_product(_self, df, selected_year, theme_colors):

        """ Plots a horizontal bar chart showing the 'Total Deal Value' by 'Deal Product Brand' 
        for the selected year, sorted from largest to smallest, using theme colors.

        Parameters:
        won_overview_df (DataFrame): Data containing 'Year', 'Deal Product Brand', and 'Total Deal Value'.
        selected_year (int): The year for which to display the data.
        theme_colors (dict): Dictionary containing color definitions.

        Returns:
        fig: Plotly Figure object. """
        
        
        # Define color variables
        SUPPORTING_COLORS = theme_colors["SUPPORTING_COLORS"]
        
        # Filter data for the selected year
        year_data = df[df['Year'] == selected_year]
        
        # Group by 'Deal Product Brand' and sum the 'Total Deal Value'
        deal_value_by_brand = year_data.groupby('Deal Product')['Deal : Total Deal Value'].sum().reset_index()
        
        # Sort the data by 'Total Deal Value' in descending order
        deal_value_by_brand = deal_value_by_brand.sort_values(by='Deal : Total Deal Value', ascending=True)
        
        # Create a horizontal bar chart using Plotly Express
        fig = px.bar(
            deal_value_by_brand, 
            y='Deal Product', 
            x='Deal : Total Deal Value', 
            title=f'Won Deals by Project Type',
            orientation='h',  # Display bars horizontally
            color_discrete_sequence=SUPPORTING_COLORS  # Apply primary colors from the theme
        )

        # Update layout for better readability and presentation
        fig.update_layout(
            xaxis_title="Total Deal Value",
            yaxis_title="Deal Product Brand",
            template="plotly_white",  # Optional: Use a clean background theme
            showlegend=False
        )
        
        # Optionally add hover formatting for better readability of large numbers
        fig.update_traces(hovertemplate='%{y}: %{x:$,.2f}')

        return fig
     
    def tab4_visualize_sales_leaderboard(self, won_deals, sales_targets):
        """
        Visualize actual 'Deal : Total Deal Value' of 'Won' Deal, grouped by Owner, compared to target revenue in a progress bar style with overachievement indicated.
        """
        # Access theme colors using self.theme_colors
        actual_sales_color = self.theme_colors["PRIMARY_COLORS"][0]
        overachievement_color = self.theme_colors["SUPPORTING_COLORS"][0]  # First supporting color for overachievement
        target_background_color = self.theme_colors["SUPPORTING_COLORS"][1]  # Second supporting color for target background
        background_color = self.theme_colors["NEUTRAL_COLOR"]

        # Name normalization dictionary
        name_mapping = {
            "Han Vo": "Vo Ly Ngoc Han",
            "Vo Ly Ngoc Han": "Vo Ly Ngoc Han"# Add more mappings if necessary
        }

        # Strip whitespace, apply name mapping, and sanitize owner names
        won_deals['Deal : Owner'] = won_deals['Deal : Owner'].str.strip().replace(name_mapping)

        # Group the data by 'Deal : Owner' and sum the 'Deal : Total Deal Value'
        owner_deal_value = won_deals.groupby('Deal : Owner', as_index=False)['Deal : Total Deal Value'].sum()

        # Extract revenue targets from sales_targets
        revenue_targets = sales_targets.get('revenue_targets', {})
        sales_targets_series = pd.Series(revenue_targets)

        # Add a 'Target' column by mapping each owner's name to the corresponding target sales
        owner_deal_value['Target'] = owner_deal_value['Deal : Owner'].map(sales_targets_series).fillna(0)

        # Calculate overachievement and target percentage
        owner_deal_value['Percentage of Target'] = (owner_deal_value['Deal : Total Deal Value'] / owner_deal_value['Target']).fillna(0)
        owner_deal_value['Actual Within Target'] = owner_deal_value[['Deal : Total Deal Value', 'Target']].min(axis=1)
        owner_deal_value['Overachievement'] = owner_deal_value['Deal : Total Deal Value'] - owner_deal_value['Target']
        owner_deal_value['Overachievement'] = owner_deal_value['Overachievement'].apply(lambda x: max(x, 0))

        # Sort data by actual sales (Deal : Total Deal Value)
        owner_deal_value = owner_deal_value.sort_values('Deal : Total Deal Value', ascending=False)

        # Prepare data for plotting
        owners = owner_deal_value['Deal : Owner']
        target_sales = owner_deal_value['Target']
        actual_within_target = owner_deal_value['Actual Within Target']
        overachievement = owner_deal_value['Overachievement']
        percentage_of_target = owner_deal_value['Percentage of Target']

        # Enclose the entire visualization in a bordered container with rounded corners and shadows
        with st.container(border=True):
            st.markdown("<div style='text-align:center; font-size: 1.5em; font-weight: bold;'>Sales Leaderboard: Progress Towards Target</div>", unsafe_allow_html=True)
            
            # Create the progress bar style chart
            fig = go.Figure()

            # Target Sales bar as the background (reference line for 100% target)
            fig.add_trace(go.Bar(
                x=target_sales,
                y=owners,
                name='Target Sales',
                orientation='h',
                marker_color=target_background_color,  # Color for target background
                hoverinfo='none',  # No hover info for background bar
                opacity=0.3,  # Slightly transparent to show as a background
                showlegend=True
            ))

            # Actual Sales within Target bar with formatted hover text
            fig.add_trace(go.Bar(
                x=actual_within_target,
                y=owners,
                name='Actual Sales (Progress)',
                orientation='h',
                marker_color=actual_sales_color,
                text=actual_within_target.apply(self.format_number),
                textposition="outside",
                hovertemplate=(
                    '<b>Owner:</b> %{y}<br>'
                    '<b>Actual Sales:</b> %{x}<br>'
                    '<b>Target Sales:</b> %{customdata[0]}<br>'
                    '<b>Percentage of Target:</b> %{customdata[1]:.0%}<extra></extra>'
                ),
                customdata=np.stack((target_sales.apply(self.format_number), percentage_of_target), axis=-1),
                showlegend=True
            ))

            # Overachievement bar that extends beyond the actual within target
            fig.add_trace(go.Bar(
                x=overachievement,
                y=owners,
                name='Overachievement',
                orientation='h',
                base=actual_within_target,  # Offset to start after the actual sales bar
                marker_color=overachievement_color,
                text=overachievement.apply(lambda x: f"+{self.format_number(x)}" if x > 0 else ""),
                textposition="outside",
                hovertemplate=(
                    '<b>Owner:</b> %{y}<br>'
                    '<b>Overachievement:</b> %{x}<extra></extra>'
                ),
                showlegend=True
            ))

            # Layout settings
            fig.update_layout(
                xaxis_title="Sales Value (USD)",
                yaxis_title="Sales Owner",
                barmode='overlay',  # Overlay bars to achieve progress bar effect
                plot_bgcolor=background_color,
                paper_bgcolor=background_color,
                font=dict(size=12, color=self.theme_colors["PRIMARY_COLORS"][1]),
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1, 
                    xanchor="right", 
                    x=1,
                    font=dict(color=self.theme_colors["PRIMARY_COLORS"][0])
                )
            )

            # Display chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)     
        
    def tab4_visualize_sales_performance_by_owner(self,df,selected_year,threshold = 300000):
        
        """
        Visualizes Sales Performance by Owner with key KPIs and relevant visualizations, including all products.

        Parameters:
        - df (pd.DataFrame): The filtered DataFrame for the selected year.
        - selected_year (int): The year selected by the user.
        - threshold (int): Threshold for outlier detection in deal value distribution.
        """
        st.subheader("Sales Performance by Owner Overview")

        # Filter DataFrame by selected year
        df_filtered = df[df['Deal : Expected close date'].dt.year == selected_year]

        # Owner Filter
        all_owners = df_filtered['Deal : Owner'].unique()
        selected_owners = st.multiselect(
            "Select Sales Owners", options=all_owners, default=all_owners
        )
        df_filtered = df_filtered[df_filtered['Deal : Owner'].isin(selected_owners)]

        # Calculate Total Revenue, Total Cost, and Gross Margin
        revenue_columns = [
            'Deal : Software revenue: Product 1', 'Deal : Support revenue: Product 1', 'Deal : Service revenue: Product 1',
            'Deal : Managed service revenue: Product 1',
            'Deal : Software revenue: Product 2', 'Deal : Support revenue: Product 2', 'Deal : Service revenue: Product 2',
            'Deal : Managed service revenue: Product 2',
            'Deal : Support revenue: Product 3', 'Deal : Service revenue: Product 3',
            'Deal : Managed service revenue: Product 3',
            'Deal : Software revenue: Product 4', 'Deal : Support revenue: Product 4', 'Deal : Service revenue: Product 4',
            'Deal : Managed service revenue: Product 4'
        ]
        cost_columns = [
            'Deal : Software cost: Product 1', 'Deal : Support cost: Product 1', 'Deal : Service cost: Product 1',
            'Deal : Managed service cost: Product 1',
            'Deal : Software cost: Product 2', 'Deal : Support cost: Product 2', 'Deal : Service cost: Product 2',
            'Deal : Managed service cost: Product 2',
            'Deal : Software cost: Product 3', 'Deal : Support cost: Product 3', 'Deal : Service cost: Product 3',
            'Deal : Managed service cost: Product 3',
            'Deal : Software cost: Product 4', 'Deal : Support cost: Product 4', 'Deal : Service cost: Product 4',
            'Deal : Managed service cost: Product 4'
        ]
        df_filtered['Total Revenue'] = df_filtered[revenue_columns].sum(axis=1)
        df_filtered['Total Cost'] = df_filtered[cost_columns].sum(axis=1)
        df_filtered['Gross Margin'] = df_filtered['Total Revenue'] - df_filtered['Total Cost']

        # KPIs
        total_revenue = df_filtered['Total Revenue'].sum()
        total_deals = df_filtered.shape[0]
        total_deals_by_owner = df_filtered.groupby('Deal : Owner').size()
        won_deals = df_filtered[df_filtered['Deal : Deal stage'] == 'Won']
        won_deals_by_owner = won_deals.groupby('Deal : Owner').size()
        win_rate_by_owner = (won_deals_by_owner / total_deals_by_owner * 100).fillna(0)
        avg_deal_value = df_filtered['Total Revenue'].mean()
        avg_win_rate = win_rate_by_owner.mean()

        # KPIs Display
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            with st.container(border=True):
                st.metric("Total Revenue", self.format_number(total_revenue))
        with kpi_cols[1]:
            with st.container(border=True):
                st.metric("Average Win Rate", f"{avg_win_rate:.2f}%")
        with kpi_cols[2]:
            with st.container(border=True):
                st.metric("Average Deal Value", self.format_number(avg_deal_value))
        with kpi_cols[3]:
            with st.container(border=True):
                st.metric("Total Number of Deals", total_deals)

        # Visualizations
        st.markdown("### Detailed Visualizations by Sales Owner")
        col1, col2 = st.columns(2)

        # 1. Total Revenue by Owner (Horizontal Bar Chart)
        with col1:
            with st.container(border=True):
                st.markdown("#### Total Revenue by Owner")
                total_revenue_by_owner = df_filtered.groupby('Deal : Owner')['Total Revenue'].sum().reset_index()
                fig_revenue_by_owner = px.bar(
                    total_revenue_by_owner,
                    x='Total Revenue',
                    y='Deal : Owner',
                    orientation='h',
                    text=total_revenue_by_owner['Total Revenue'].apply(self.format_number),
                    title="Total Revenue by Owner",
                    labels={'Total Revenue': 'Revenue (USD)', 'Deal : Owner': 'Sales Owner'}
                )
                fig_revenue_by_owner.update_traces(marker_color=self.theme_colors["PRIMARY_COLORS"][0], textposition='outside')
                fig_revenue_by_owner.update_layout(
                    plot_bgcolor=self.theme_colors["NEUTRAL_COLOR"],
                    paper_bgcolor=self.theme_colors["NEUTRAL_COLOR"],
                    font=dict(size=12)
                )
                st.plotly_chart(fig_revenue_by_owner, use_container_width=True)

        # 2. Win Rate by Owner (Horizontal Bar Chart)
        with col2:
            with st.container(border=True):
                st.markdown("#### Win Rate by Owner")
                win_rate_df = win_rate_by_owner.reset_index()
                win_rate_df.columns = ['Deal : Owner', 'Win Rate (%)']
                fig_win_rate = px.bar(
                    win_rate_df,
                    x='Win Rate (%)',
                    y='Deal : Owner',
                    orientation='h',
                    text=win_rate_df['Win Rate (%)'].apply(lambda x: f"{x:.2f}%"),
                    labels={'Win Rate (%)': 'Win Rate (%)', 'Deal : Owner': 'Sales Owner'}
                )
                fig_win_rate.update_traces(marker_color=self.theme_colors["SUPPORTING_COLORS"][0], textposition='outside')
                fig_win_rate.update_layout(
                    plot_bgcolor=self.theme_colors["NEUTRAL_COLOR"],
                    paper_bgcolor=self.theme_colors["NEUTRAL_COLOR"],
                    font=dict(size=12)
                )
                st.plotly_chart(fig_win_rate, use_container_width=True)

        # 3. Deal Value Distribution by Owner (Violin Plot)
        with col1:
            with st.container(border=True):
                st.markdown("#### Deal Value Distribution by Owner")
                core_data = df_filtered[df_filtered['Total Revenue'] <= threshold]
                outliers = df_filtered[df_filtered['Total Revenue'] > threshold]
                fig_violin = px.violin(
                    core_data,
                    x='Total Revenue',
                    y='Deal : Owner',
                    box=True,
                    points='all',
                    title="Deal Value Distribution by Owner",
                    labels={'Total Revenue': 'Deal Value (USD)', 'Deal : Owner': 'Sales Owner'}
                )
                fig_violin.add_trace(go.Scatter(
                    x=outliers['Total Revenue'],
                    y=outliers['Deal : Owner'],
                    mode='markers',
                    marker=dict(
                        color=self.theme_colors["SUPPORTING_COLORS"][1],
                        size=8,
                        symbol='cross'
                    ),
                    name="Outliers"
                ))
                fig_violin.update_layout(
                    plot_bgcolor=self.theme_colors["NEUTRAL_COLOR"],
                    paper_bgcolor=self.theme_colors["NEUTRAL_COLOR"],
                    font=dict(size=12)
                )
                st.plotly_chart(fig_violin, use_container_width=True)

        # 4. Total Number of Deals by Owner (Bubble Chart)
        with col2:
            with st.container(border=True):
                st.markdown("#### Total Number of Deals by Owner")
                fig_total_deals = px.scatter(
                    total_deals_by_owner.reset_index(),
                    x=0,
                    y='Deal : Owner',
                    size=total_deals_by_owner.values,
                    size_max=20,
                    labels={0: 'Number of Deals', 'Deal : Owner': 'Sales Owner'}
                )
                fig_total_deals.update_traces(marker=dict(color=self.theme_colors["PRIMARY_COLORS"][1]))
                fig_total_deals.update_layout(
                    plot_bgcolor=self.theme_colors["NEUTRAL_COLOR"],
                    paper_bgcolor=self.theme_colors["NEUTRAL_COLOR"],
                    font=dict(size=12)
                )
                st.plotly_chart(fig_total_deals, use_container_width=True)