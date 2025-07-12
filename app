import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from data_processor import DataProcessor
from threshold_analyzer import ThresholdAnalyzer
from alert_engine import AlertEngine
from api_client import APIClient
from utils import Utils

# Page configuration
st.set_page_config(
    page_title="Threshold Recalibration & Alert Impact Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'uat_data' not in st.session_state:
    st.session_state.uat_data = None
if 'prod_data' not in st.session_state:
    st.session_state.prod_data = None
if 'processed_trade_data' not in st.session_state:
    st.session_state.processed_trade_data = None
if 'exception_data' not in st.session_state:
    st.session_state.exception_data = None
if 'threshold_data' not in st.session_state:
    st.session_state.threshold_data = None
if 'adjusted_thresholds' not in st.session_state:
    st.session_state.adjusted_thresholds = None
if 'alert_results' not in st.session_state:
    st.session_state.alert_results = None

# Initialize components
data_processor = DataProcessor()
threshold_analyzer = ThresholdAnalyzer()
alert_engine = AlertEngine()
api_client = APIClient()
utils = Utils()

# Sidebar for all controls
st.sidebar.title("📊 Controls")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select Section",
    ["Data Input", "Threshold Configuration", "Alert Analysis", "Reporting"]
)

st.sidebar.markdown("---")

# Main title in main area
st.title("📊 Threshold Recalibration & Alert Impact Analysis")
st.markdown("---")

if page == "Data Input":
    # All controls in sidebar
    st.sidebar.subheader("📁 Data Input")
    
    # Data input method selection in sidebar
    input_method = st.sidebar.radio(
        "Input Method:",
        ["Upload Files", "API Download"]
    )
    
    if input_method == "Upload Files":
        # File uploaders in sidebar
        uat_file = st.sidebar.file_uploader(
            "UAT Trade Dataset",
            type=['csv', 'xlsx'],
            key="uat_upload"
        )
        
        prod_file = st.sidebar.file_uploader(
            "PROD Trade Dataset",
            type=['csv', 'xlsx'],
            key="prod_upload"
        )
        
        threshold_file = st.sidebar.file_uploader(
            "Threshold Configuration",
            type=['csv', 'xlsx'],
            key="threshold_upload"
        )
        
        exception_file = st.sidebar.file_uploader(
            "Exception Dataset (Optional)",
            type=['csv', 'xlsx'],
            key="exception_upload"
        )
        
        # Process data button in sidebar
        if st.session_state.uat_data is not None or st.session_state.prod_data is not None:
            if st.sidebar.button("🔄 Process Trade Data", type="primary"):
                try:
                    st.session_state.processed_trade_data = data_processor.process_combined_datasets(
                        uat_data=st.session_state.uat_data,
                        prod_data=st.session_state.prod_data
                    )
                except Exception as e:
                    st.error(f"Error processing trade data: {str(e)}")
        
        # Process uploaded files and show results in main area
        if uat_file is not None:
            try:
                st.session_state.uat_data = data_processor.load_file(uat_file)
                validation = data_processor.validate_trade_data(st.session_state.uat_data)
                
                if not validation['is_valid']:
                    st.error("❌ UAT data validation failed:")
                    for error in validation['errors']:
                        st.error(f"• {error}")
                else:
                    st.success(f"✅ UAT data loaded: {len(st.session_state.uat_data)} records")
                    if validation['warnings']:
                        for warning in validation['warnings']:
                            st.warning(f"⚠️ {warning}")
                            
            except Exception as e:
                st.error(f"❌ Error loading UAT file: {str(e)}")
        
        if prod_file is not None:
            try:
                st.session_state.prod_data = data_processor.load_file(prod_file)
                validation = data_processor.validate_trade_data(st.session_state.prod_data)
                
                if not validation['is_valid']:
                    st.error("❌ PROD data validation failed:")
                    for error in validation['errors']:
                        st.error(f"• {error}")
                else:
                    st.success(f"✅ PROD data loaded: {len(st.session_state.prod_data)} records")
                    if validation['warnings']:
                        for warning in validation['warnings']:
                            st.warning(f"⚠️ {warning}")
                            
            except Exception as e:
                st.error(f"❌ Error loading PROD file: {str(e)}")
        
        if threshold_file is not None:
            try:
                st.session_state.threshold_data = data_processor.load_file(threshold_file)
                validation = data_processor.validate_threshold_data(st.session_state.threshold_data)
                
                if not validation['is_valid']:
                    st.error("❌ Threshold data validation failed:")
                    for error in validation['errors']:
                        st.error(f"• {error}")
                else:
                    st.success(f"✅ Threshold data loaded: {len(st.session_state.threshold_data)} records")
                    if validation['warnings']:
                        for warning in validation['warnings']:
                            st.warning(f"⚠️ {warning}")
                            
            except Exception as e:
                st.error(f"❌ Error loading threshold file: {str(e)}")
        
        if exception_file is not None:
            try:
                st.session_state.exception_data = data_processor.load_file(exception_file)
                st.success(f"✅ Exception data loaded: {len(st.session_state.exception_data)} records")
            except Exception as e:
                st.error(f"❌ Error loading exception file: {str(e)}")
        
        # Show processing results in main area
        if st.session_state.processed_trade_data is not None:
            st.header("📊 Processed Trade Data")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(st.session_state.processed_trade_data))
            with col2:
                if st.session_state.uat_data is not None:
                    st.metric("Original UAT Records", len(st.session_state.uat_data))
            with col3:
                if st.session_state.prod_data is not None:
                    st.metric("Original PROD Records", len(st.session_state.prod_data))
            
            # Show processing summary
            if st.session_state.uat_data is not None and st.session_state.prod_data is not None:
                original_uat_count = len(st.session_state.uat_data)
                matching_count = len(st.session_state.processed_trade_data)
                st.info(f"📋 Matched {matching_count} UAT records with PROD trade_ids from {original_uat_count} original UAT records")
            
            # Display sample of processed data
            with st.expander("📄 View Sample Data", expanded=False):
                st.dataframe(st.session_state.processed_trade_data.head(10))
    
    else:  # API Download
        # API parameters in sidebar
        legal_entities_options = ["HBAP", "HBEU", "HBUS", "HBCA", "HBSG"]
        selected_legal_entities = st.sidebar.multiselect(
            "Legal Entities",
            legal_entities_options,
            default=["HBAP"]
        )
        
        source_systems_options = ["PTS1", "PTS2", "PTS3"]
        selected_source_systems = st.sidebar.multiselect(
            "Source Systems",
            source_systems_options,
            default=["PTS1"]
        )
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=[datetime.now() - timedelta(days=7), datetime.now()],
            max_value=datetime.now()
        )
        
        # Download summary in sidebar
        if selected_legal_entities and selected_source_systems and len(date_range) == 2:
            days_count = (date_range[1] - date_range[0]).days + 1
            total_combinations = len(selected_legal_entities) * len(selected_source_systems) * days_count
            st.sidebar.info(f"📊 {total_combinations} total files\n({len(selected_legal_entities)} × {len(selected_source_systems)} × {days_count})")
            
            # Download buttons in sidebar
            if st.sidebar.button("🚀 Download UAT & PROD", type="primary"):
                # Show progress in main area
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    def update_progress(progress, total, message=""):
                        if total > 0:
                            progress_bar.progress(progress / total)
                            status_text.text(f"{message} ({progress:.0f}%)")
                    
                    with st.spinner("Starting batch download..."):
                        # Download both UAT and PROD data
                        batch_results = api_client.download_batch_uat_and_prod(
                            legal_entities=selected_legal_entities,
                            source_systems=selected_source_systems,
                            start_date=date_range[0],
                            end_date=date_range[1],
                            progress_callback=update_progress
                        )
                        
                        # Store the results
                        st.session_state.uat_data = batch_results["uat_data"]
                        st.session_state.prod_data = batch_results["prod_data"]
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.rerun()
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Batch download failed: {str(e)}")
            
            if st.sidebar.button("📊 Download Exception Data"):
                try:
                    with st.spinner("Downloading exception data..."):
                        # For exception data, download for each combination
                        all_exception_data = []
                        
                        for legal_entity in selected_legal_entities:
                            for source_system in selected_source_systems:
                                try:
                                    exception_df = api_client.download_exception_data(
                                        legal_entity=legal_entity,
                                        source_system=source_system,
                                        start_date=date_range[0],
                                        end_date=date_range[1]
                                    )
                                    if not exception_df.empty:
                                        all_exception_data.append(exception_df)
                                except Exception as e:
                                    st.warning(f"Failed to download exception data for {legal_entity}-{source_system}: {str(e)}")
                        
                        if all_exception_data:
                            st.session_state.exception_data = pd.concat(all_exception_data, ignore_index=True)
                            st.success(f"Exception data downloaded: {len(st.session_state.exception_data)} records")
                        else:
                            st.info("No exception data found for the selected parameters.")
                            
                except Exception as e:
                    st.error(f"Error downloading exception data: {str(e)}")
        else:
            st.sidebar.warning("Complete all selections")
        
        # Show download results in main area
        if st.session_state.uat_data is not None or st.session_state.prod_data is not None:
            st.header("📊 Download Results")
            
            # Download statistics
            col1, col2 = st.columns(2)
            
            if st.session_state.uat_data is not None:
                with col1:
                    st.subheader("✅ UAT Data")
                    st.metric("Records", len(st.session_state.uat_data))
                    st.metric("Legal Entities", st.session_state.uat_data['LegalEntity'].nunique())
                    st.metric("Source Systems", st.session_state.uat_data['SourceSystem'].nunique())
            
            if st.session_state.prod_data is not None:
                with col2:
                    st.subheader("✅ PROD Data")
                    st.metric("Records", len(st.session_state.prod_data))
                    st.metric("Legal Entities", st.session_state.prod_data['LegalEntity'].nunique())
                    st.metric("Source Systems", st.session_state.prod_data['SourceSystem'].nunique())
            
            # Process data button in sidebar
            if st.session_state.uat_data is not None or st.session_state.prod_data is not None:
                if st.sidebar.button("🔄 Process Downloaded Data", type="primary"):
                    try:
                        with st.spinner("Processing trade data..."):
                            st.session_state.processed_trade_data = data_processor.process_combined_datasets(
                                uat_data=st.session_state.uat_data,
                                prod_data=st.session_state.prod_data
                            )
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error processing downloaded data: {str(e)}")
            
            # Show processed data results
            if st.session_state.processed_trade_data is not None:
                st.subheader("📄 Processed Trade Data")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Records", len(st.session_state.processed_trade_data))
                with col2:
                    if st.session_state.uat_data is not None:
                        st.metric("Original UAT", len(st.session_state.uat_data))
                with col3:
                    if st.session_state.prod_data is not None:
                        st.metric("Original PROD", len(st.session_state.prod_data))
                
                # Processing summary
                if st.session_state.uat_data is not None and st.session_state.prod_data is not None:
                    original_uat_count = len(st.session_state.uat_data)
                    matching_count = len(st.session_state.processed_trade_data)
                    st.info(f"📋 Matched {matching_count} UAT records with PROD trade_ids and filtered out 'out of scope' records")
                
                with st.expander("📄 View Sample Data", expanded=False):
                    st.dataframe(st.session_state.processed_trade_data.head(10))

    
    # Show general data status in main area regardless of input method
    if (st.session_state.uat_data is not None or st.session_state.prod_data is not None or 
        st.session_state.processed_trade_data is not None or st.session_state.exception_data is not None):
        
        st.header("📋 Data Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.uat_data is not None:
                st.metric("UAT Records", len(st.session_state.uat_data), "✅ Loaded")
            else:
                st.metric("UAT Records", "No data", "❌ Not loaded")
        
        with col2:
            if st.session_state.prod_data is not None:
                st.metric("PROD Records", len(st.session_state.prod_data), "✅ Loaded")
            else:
                st.metric("PROD Records", "No data", "❌ Not loaded")
        
        with col3:
            if st.session_state.processed_trade_data is not None:
                st.metric("Processed Records", len(st.session_state.processed_trade_data), "✅ Ready")
            else:
                st.metric("Processed Records", "No data", "❌ Not processed")
        
        with col4:
            if st.session_state.exception_data is not None:
                st.metric("Exception Records", len(st.session_state.exception_data), "✅ Loaded")
            else:
                st.metric("Exception Records", "No data", "⚪ Optional")

elif page == "Threshold Configuration":
    # Threshold controls in sidebar
    st.sidebar.subheader("⚙️ Threshold Settings")
    
    # Threshold file upload in sidebar
    threshold_file = st.sidebar.file_uploader(
        "Threshold File",
        type=['csv', 'xlsx'],
        key="threshold_config_upload"
    )
    
    if threshold_file is not None:
        try:
            st.session_state.threshold_data = data_processor.load_file(threshold_file)
            validation = data_processor.validate_threshold_data(st.session_state.threshold_data)
            
            if not validation['is_valid']:
                st.error("❌ Threshold data validation failed:")
                for error in validation['errors']:
                    st.error(f"• {error}")
            else:
                st.success(f"✅ Threshold data loaded: {len(st.session_state.threshold_data)} records")
                if validation['warnings']:
                    for warning in validation['warnings']:
                        st.warning(f"⚠️ {warning}")
                        
        except Exception as e:
            st.error(f"❌ Error loading threshold data: {str(e)}")
    
    if st.session_state.threshold_data is not None:
        # Threshold view selection in sidebar
        view_type = st.sidebar.radio(
            "View Type:",
            ["Group-wise", "Currency-wise"]
        )
        
        # Adjustment controls in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Adjust Thresholds")
        
        # Group thresholds and create adjustment controls in sidebar
        grouped_thresholds = threshold_analyzer.group_by_group(st.session_state.threshold_data)
        adjusted_data = []
        
        if view_type == "Group-wise":
            for group, group_data in grouped_thresholds.items():
                original_threshold = group_data['Threshold'].iloc[0]
                adjusted_threshold = st.sidebar.number_input(
                    f"Group {group}",
                    value=float(original_threshold),
                    key=f"group_{group}_threshold",
                    step=0.01,
                    format="%.3f",
                    help=f"Currencies: {', '.join(group_data['CURR'].unique())}"
                )
                
                # Update adjusted data for all currencies in this group
                for _, row in group_data.iterrows():
                    adjusted_row = row.copy()
                    adjusted_row['AdjustedThreshold'] = adjusted_threshold
                    adjusted_data.append(adjusted_row)
        
        else:  # Currency-wise
            for idx, row in st.session_state.threshold_data.iterrows():
                adjusted_threshold = st.sidebar.number_input(
                    f"{row['CURR']} (Group {row['GROUP']})",
                    value=float(row['Threshold']),
                    key=f"curr_{row['CURR']}_threshold",
                    step=0.01,
                    format="%.3f"
                )
                
                adjusted_row = row.copy()
                adjusted_row['AdjustedThreshold'] = adjusted_threshold
                adjusted_data.append(adjusted_row)
        
        # Store adjusted thresholds
        st.session_state.adjusted_thresholds = pd.DataFrame(adjusted_data)
        
        # Preview button in sidebar
        if st.sidebar.button("🔍 Preview Impact", type="primary"):
            if st.session_state.processed_trade_data is not None:
                try:
                    with st.spinner("Calculating impact..."):
                        impact_summary = threshold_analyzer.calculate_impact_preview(
                            st.session_state.processed_trade_data,
                            st.session_state.threshold_data,
                            st.session_state.adjusted_thresholds
                        )
                        st.session_state.impact_preview = impact_summary
                        st.rerun()
                except Exception as e:
                    st.error(f"Error calculating impact: {str(e)}")
            else:
                st.sidebar.error("Load trade data first")
        
        # Show threshold configuration and results in main area
        st.header("⚙️ Threshold Configuration")
        
        # Display current threshold configuration
        if view_type == "Group-wise":
            st.subheader("📊 Group-wise Threshold Configuration")
            
            for group, group_data in grouped_thresholds.items():
                with st.expander(f"Group {group} - {len(group_data)} currencies", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Currencies:**")
                        currencies_list = group_data['CURR'].unique().tolist()
                        st.write(", ".join(currencies_list))
                        
                    with col2:
                        original_threshold = group_data['Threshold'].iloc[0]
                        adjusted_threshold = st.session_state.adjusted_thresholds[
                            st.session_state.adjusted_thresholds['GROUP'] == group
                        ]['AdjustedThreshold'].iloc[0]
                        
                        col_orig, col_adj = st.columns(2)
                        with col_orig:
                            st.metric("Original", f"{original_threshold:.3f}")
                        with col_adj:
                            change = adjusted_threshold - original_threshold
                            st.metric("Adjusted", f"{adjusted_threshold:.3f}", delta=f"{change:+.3f}")
        
        else:  # Currency-wise
            st.subheader("💱 Currency-wise Threshold Configuration")
            
            # Create comparison table
            comparison_df = st.session_state.threshold_data.copy()
            comparison_df['Adjusted'] = st.session_state.adjusted_thresholds['AdjustedThreshold']
            comparison_df['Change'] = comparison_df['Adjusted'] - comparison_df['Threshold']
            
            # Display as table with styling
            st.dataframe(
                comparison_df[['CURR', 'GROUP', 'Threshold', 'Adjusted', 'Change']],
                column_config={
                    "CURR": "Currency",
                    "GROUP": "Group",
                    "Threshold": st.column_config.NumberColumn("Original", format="%.3f"),
                    "Adjusted": st.column_config.NumberColumn("Adjusted", format="%.3f"),
                    "Change": st.column_config.NumberColumn("Change", format="%+.3f")
                }
            )
        
        # Show impact preview results if available
        if hasattr(st.session_state, 'impact_preview') and st.session_state.impact_preview:
            st.subheader("📈 Impact Preview")
            
            impact = st.session_state.impact_preview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Original Alerts", impact.get('proposed_alerts', 0))
            with col2:
                st.metric("Adjusted Alerts", impact.get('adjusted_alerts', 0))
            with col3:
                change = impact.get('adjusted_alerts', 0) - impact.get('proposed_alerts', 0)
                percentage_change = (change / impact.get('proposed_alerts', 1)) * 100 if impact.get('proposed_alerts', 0) > 0 else 0
                st.metric("Change", f"{change:+d}", delta=f"{percentage_change:+.1f}%")
        
    else:
        st.header("⚙️ Threshold Configuration")
        st.info("👈 Upload a threshold file using the sidebar to begin configuration.")
        
        # Show column requirements
        st.subheader("📋 Required Threshold File Format")
        st.write("Your threshold file must contain these columns:")
        
        cols_df = pd.DataFrame({
            'Column': ['LegalEntity', 'CURR', 'GROUP', 'Threshold'],
            'Description': [
                'Legal entity code (e.g., HBAP, HBEU)', 
                'Currency code (e.g., EUR, USD, GBP)',
                'Threshold group identifier',
                'Threshold value (typically 0.01 to 1.0)'
            ],
            'Example': ['HBAP', 'EUR', '1', '0.030']
        })
        st.dataframe(cols_df, hide_index=True)

elif page == "Alert Analysis":
    st.header("🚨 Alert Analysis & Impact Assessment")
    
    if st.session_state.processed_trade_data is None:
        st.warning("Please load and process trade data first.")
    elif st.session_state.adjusted_thresholds is None:
        st.warning("Please configure thresholds first.")
    else:
        if st.button("Generate Alert Analysis", type="primary"):
            with st.spinner("Generating alerts and analyzing impact..."):
                try:
                    # Generate alerts
                    st.session_state.alert_results = alert_engine.generate_alerts(
                        st.session_state.processed_trade_data,
                        st.session_state.threshold_data,
                        st.session_state.adjusted_thresholds
                    )
                    
                    st.success("Alert analysis completed!")
                except Exception as e:
                    st.error(f"Error generating alerts: {str(e)}")
        
        if st.session_state.alert_results is not None:
            st.markdown("---")
            
            # Alert summary metrics
            st.subheader("📊 Alert Summary")
            
            results = st.session_state.alert_results
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Records",
                    len(results['processed_data'])
                )
            
            with col2:
                st.metric(
                    "Proposed Alerts",
                    results['summary']['proposed_alerts']
                )
            
            with col3:
                st.metric(
                    "Adjusted Alerts",
                    results['summary']['adjusted_alerts']
                )
            
            with col4:
                change = results['summary']['adjusted_alerts'] - results['summary']['proposed_alerts']
                st.metric(
                    "Alert Change",
                    change,
                    delta=change
                )
            
            # Currency-wise alert bucketing
            st.subheader("🪣 Currency-wise Alert Bucketing")
            
            bucket_analysis = alert_engine.generate_bucket_analysis(
                st.session_state.alert_results,
                st.session_state.adjusted_thresholds
            )
            
            # Display bucket analysis
            for currency, buckets in bucket_analysis.items():
                with st.expander(f"Currency: {currency}"):
                    bucket_df = pd.DataFrame(buckets)
                    
                    # Create bucket visualization
                    fig = px.bar(
                        bucket_df,
                        x='bucket_range',
                        y=['proposed_count', 'adjusted_count'],
                        title=f"Alert Distribution for {currency}",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display bucket table
                    st.dataframe(bucket_df, use_container_width=True)
            
            # Outlier analysis
            st.subheader("⚠️ Outlier Analysis")
            
            outliers = results['outliers']
            if len(outliers) > 0:
                st.warning(f"Found {len(outliers)} outliers (empty deviation percent)")
                st.dataframe(outliers)
            else:
                st.success("No outliers found")
            
            # Detailed alert data
            st.subheader("📋 Detailed Alert Data")
            
            # Filter options
            col1, col2 = st.columns(2)
            
            with col1:
                legal_entities = results['processed_data']['LegalEntity'].unique()
                selected_le = st.selectbox("Filter by Legal Entity", ['All'] + list(legal_entities))
            
            with col2:
                alert_types = ['All', 'Proposed Only', 'Adjusted Only', 'Both']
                selected_alert_type = st.selectbox("Filter by Alert Type", alert_types)
            
            # Apply filters
            filtered_data = results['processed_data'].copy()
            
            if selected_le != 'All':
                filtered_data = filtered_data[filtered_data['LegalEntity'] == selected_le]
            
            if selected_alert_type == 'Proposed Only':
                filtered_data = filtered_data[filtered_data['ProposedAlert'] == True]
            elif selected_alert_type == 'Adjusted Only':
                filtered_data = filtered_data[filtered_data['AdjustedAlert'] == True]
            elif selected_alert_type == 'Both':
                filtered_data = filtered_data[
                    (filtered_data['ProposedAlert'] == True) | 
                    (filtered_data['AdjustedAlert'] == True)
                ]
            
            st.dataframe(filtered_data, use_container_width=True)

elif page == "Reporting":
    st.header("📈 Comprehensive Reporting")
    
    if st.session_state.alert_results is None:
        st.warning("Please complete alert analysis first.")
    else:
        results = st.session_state.alert_results
        
        # Summary report by Legal Entity and Source System
        st.subheader("📊 Summary Report")
        
        summary_report = alert_engine.generate_summary_report(
            st.session_state.alert_results,
            st.session_state.trade_data
        )
        
        st.dataframe(summary_report, use_container_width=True)
        
        # Visualization dashboard
        st.subheader("📈 Visualization Dashboard")
        
        # Alert comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Alerts by Legal Entity
            le_summary = summary_report.groupby('LegalEntity').agg({
                'ProposedAlerts': 'sum',
                'AdjustedAlerts': 'sum'
            }).reset_index()
            
            fig1 = px.bar(
                le_summary,
                x='LegalEntity',
                y=['ProposedAlerts', 'AdjustedAlerts'],
                title="Alerts by Legal Entity",
                barmode='group'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Alert trend over time
            if 'Date' in results['processed_data'].columns:
                daily_alerts = results['processed_data'].groupby('Date').agg({
                    'ProposedAlert': 'sum',
                    'AdjustedAlert': 'sum'
                }).reset_index()
                
                fig2 = px.line(
                    daily_alerts,
                    x='Date',
                    y=['ProposedAlert', 'AdjustedAlert'],
                    title="Alert Trend Over Time"
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        # Threshold impact heatmap
        st.subheader("🔥 Threshold Impact Heatmap")
        
        # Create impact matrix
        impact_matrix = threshold_analyzer.create_impact_matrix(
            st.session_state.alert_results,
            st.session_state.adjusted_thresholds
        )
        
        fig3 = px.imshow(
            impact_matrix,
            title="Threshold Impact by Currency Group",
            color_continuous_scale="RdYlBu_r"
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Export options
        st.subheader("💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Summary Report"):
                csv = summary_report.to_csv(index=False)
                st.download_button(
                    label="Download Summary CSV",
                    data=csv,
                    file_name=f"threshold_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export Alert Details"):
                csv = results['processed_data'].to_csv(index=False)
                st.download_button(
                    label="Download Alert Details CSV",
                    data=csv,
                    file_name=f"alert_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("Export Bucket Analysis"):
                # Convert bucket analysis to DataFrame
                bucket_data = []
                bucket_analysis = alert_engine.generate_bucket_analysis(
                    st.session_state.alert_results,
                    st.session_state.adjusted_thresholds
                )
                
                for currency, buckets in bucket_analysis.items():
                    for bucket in buckets:
                        bucket['currency'] = currency
                        bucket_data.append(bucket)
                
                bucket_df = pd.DataFrame(bucket_data)
                csv = bucket_df.to_csv(index=False)
                st.download_button(
                    label="Download Bucket Analysis CSV",
                    data=csv,
                    file_name=f"bucket_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown(
    "💡 **Threshold Recalibration & Alert Impact Analysis** | "
    "Built with Streamlit | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
