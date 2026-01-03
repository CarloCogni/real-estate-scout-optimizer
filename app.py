"""
Real Estate Scout Optimizer - Streamlit Application
Main entry point for the web application

Developed for ZIGURAT Master in AI for Construction | M3U1&U2 Assignment
"""

import streamlit as st
import pandas as pd
import os

# Import all modules
from modules.utils import format_currency, format_number
from modules.data_prep import chunk_01_data_cleaning, chunk_02_market_context
from modules.analysis import (
    chunk_03_01_univariate_analysis,
    chunk_03_02_size_trap_analysis,
    chunk_03_03_density_analysis,
    chunk_03_04_value_hierarchy,
    chunk_03_05_condition_strategy,
    chunk_03_06_vintage_value,
    chunk_03_07_grade_value,
    chunk_03_08_correlation_analysis,
    chunk_03_09_chi_square_tests,
    chunk_03_10_anova_tests
)
from modules.scoring import (
    chunk_04_strategy_filter,
    chunk_05_smart_score,
    chunk_05b_validate_score,
    chunk_06_strategic_zones,
    prepare_export_files,
    get_top_deals
)
from modules.visualization import (
    plot_market_distribution,
    plot_size_trap_analysis,
    plot_density_analysis,
    plot_value_hierarchy,
    plot_condition_strategy,
    plot_vintage_value,
    plot_grade_value,
    plot_correlation_heatmaps,
    plot_anova_condition,
    plot_score_validation,
    plot_strategic_zones,
    create_interactive_map
)
from modules.visualization_plotly import plot_strategic_zones_interactive
from modules.data_validation import (
    validate_dataset,
    prepare_dataset_for_analysis,
    display_validation_report
)

# =============================================================================
# PAGE CONFIGURATION (Must be first Streamlit command)
# =============================================================================
st.set_page_config(
    page_title="Real Estate Scout Optimizer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
    st.session_state.df_clean = None
    st.session_state.df_context = None
    st.session_state.audit_report = None
    st.session_state.analysis_complete = False

    # Analysis results
    st.session_state.univariate_stats = None
    st.session_state.condition_results = None
    st.session_state.vintage_results = None
    st.session_state.grade_results = None
    st.session_state.correlation_results = None
    st.session_state.chi_square_results = None
    st.session_state.anova_results = None

    # Scoring results
    st.session_state.df_ranked = None
    st.session_state.scoring_audit = None
    st.session_state.validation_results = None
    st.session_state.strategic_zones = None
    st.session_state.action_list = None
    st.session_state.export_df = None

# =============================================================================
# HEADER
# =============================================================================
st.title("🏠 Real Estate Scout Optimizer")
st.markdown("### Data-Driven Investment Analysis Tool")
st.markdown("*Developed for ZIGURAT Master in AI for Architecture and Construction | M3U1&U2 Assignment | Group 4*")
st.divider()

# =============================================================================
# SIDEBAR - CONFIGURATION
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    # File source selection
    st.subheader("📁 Data Source")

    data_source = st.radio(
        "Choose data source:",
        ["📊 Use Demo Dataset (KC Housing)", "📤 Upload Your Own CSV"],
        help="Demo dataset shows full functionality. Upload your own for custom analysis."
    )

    uploaded_file = None

    if data_source == "📤 Upload Your Own CSV":
        uploaded_file = st.file_uploader(
            "Upload Housing Dataset",
            type=['csv'],
            help="CSV must include: id, price, bedrooms, bathrooms, sqft_living, sqft_lot, condition, grade, yr_built, zipcode, lat, long"
        )
    else:
        # Use demo dataset
        demo_path = os.path.join('data', 'MAICEN_1125_M3_U1&U2_Assignment input.csv')
        if os.path.exists(demo_path):
            uploaded_file = demo_path
            st.success("✅ Demo dataset loaded")
        else:
            st.error("❌ Demo dataset not found. Please upload a file.")
            uploaded_file = None

    # Reset analysis if file source changes
    if 'last_file_source' not in st.session_state:
        st.session_state.last_file_source = data_source

    if st.session_state.last_file_source != data_source:
        # File source changed - reset everything
        st.session_state.df_raw = None
        st.session_state.analysis_complete = False
        st.session_state.last_file_source = data_source
        st.rerun()

    st.divider()

    # Strategy parameters
    st.subheader("🎯 Strategy Parameters")

    max_budget = st.slider(
        "Maximum Acquisition Budget",
        min_value=100000,
        max_value=500000,
        value=300000,
        step=10000,
        help="Drag to adjust your maximum property budget"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Selected Budget",
            value=format_currency(max_budget, compact=True),
            label_visibility="collapsed"
        )

    st.info("💡 Additional parameters will be auto-detected from data")

    # Debug panel
    with st.expander("🐛 Debug Info", expanded=False):
        if 'df_raw' in st.session_state and st.session_state.df_raw is not None:
            st.success("✅ Raw data loaded")
        if 'df_clean' in st.session_state and st.session_state.df_clean is not None:
            st.success("✅ Data cleaned")
        if 'df_context' in st.session_state and st.session_state.df_context is not None:
            st.success("✅ Market context computed")
        if 'df_ranked' in st.session_state and st.session_state.df_ranked is not None:
            st.success(f"✅ {len(st.session_state.df_ranked)} properties scored")

# =============================================================================
# DATA LOADING & VALIDATION
# =============================================================================
if uploaded_file is not None:
    if st.session_state.df_raw is None:
        with st.spinner("📂 Loading and validating dataset..."):
            # Load CSV
            if isinstance(uploaded_file, str):
                df_raw_temp = pd.read_csv(uploaded_file)
            else:
                df_raw_temp = pd.read_csv(uploaded_file)

            # Validate
            is_valid, errors, report = validate_dataset(df_raw_temp)

            if not is_valid:
                st.error("❌ **Dataset Validation Failed**")
                for error in errors:
                    st.error(error)

                display_validation_report(report)

                st.info("""
                **Required Columns:**
                - `id`, `price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`
                - `condition`, `grade`, `yr_built`, `zipcode`, `lat`, `long`

                **Please upload a dataset with these columns to proceed.**
                """)

                st.stop()

            # Prepare
            st.session_state.df_raw = prepare_dataset_for_analysis(df_raw_temp)

            # Show validation report
            display_validation_report(report)


# =============================================================================
# INITIALIZE RUNNING FLAG
# =============================================================================
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# =============================================================================
# MAIN CONTENT - 3 CASES: RESULTS / RUNNING / LANDING PAGE
# =============================================================================

# CASE 1: Show results (after analysis complete)
if st.session_state.analysis_complete:
    st.markdown("## 📈 Analysis Complete!")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Properties Analyzed", format_number(len(st.session_state.df_context)))
    with col2:
        st.metric("Valid Targets Found", format_number(len(st.session_state.df_ranked)))
    with col3:
        st.metric("Avg Smart Score", f"{st.session_state.scoring_audit['mean_score']:.1f}")
    with col4:
        st.metric("Top 10 Avg Profit",
                  format_currency(st.session_state.scoring_audit['top_10_mean_gap'], compact=True))

    st.divider()
    st.markdown("## 📈 Analysis Results")

# CASE 2: Running analysis (show only progress)
elif st.session_state.is_running:
    st.markdown("## 🔄 Running Analysis...")
    st.info("Please wait while we process your data...")
    # Progress bar will appear below

# CASE 3: Landing page (before analysis)
else:
    # Show upload status
    if uploaded_file is not None:
        if isinstance(uploaded_file, str):
            filename = os.path.basename(uploaded_file)
            st.info(f"📊 **Demo Dataset Loaded**: `{filename}`")
            st.caption("Ready to analyze KC Housing data with default parameters")
        else:
            filename = uploaded_file.name
            st.success(f"✅ **Custom Dataset Loaded**: `{filename}`")
            st.caption("Ready to analyze your custom data")

        if st.session_state.df_raw is not None:
            with st.expander("📊 Data Preview (First 10 Rows)", expanded=False):
                st.dataframe(st.session_state.df_raw.head(10), width='stretch')
                st.caption(
                    f"**Shape**: {format_number(st.session_state.df_raw.shape[0])} rows × {st.session_state.df_raw.shape[1]} columns")
    else:
        st.info("👆 **Get Started**: Choose a data source in the sidebar to begin")

    st.markdown("---")
    st.markdown("### 📊 What This Tool Does")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔍 Smart Analysis")
        st.markdown("""
        - Market distribution analysis
        - Statistical validation (Chi-Square, ANOVA)
        - Correlation studies
        - Outlier detection with skewness/kurtosis
        """)

    with col2:
        st.markdown("#### 🎯 Data-Driven Strategy")
        st.markdown("""
        - Auto-detects optimal conditions
        - Auto-detects optimal grades
        - Auto-detects optimal vintages
        - Efficiency trap filtering
        """)

    with col3:
        st.markdown("#### 📥 Actionable Outputs")
        st.markdown("""
        - Prioritized action list (CSV)
        - Interactive geographic map
        - Full dataset export
        - Power BI ready files
        """)

    st.markdown("---")
    st.markdown("### 🎓 Methodology")

    st.markdown("""
    This tool implements the **Smart Scout Score** algorithm, a multi-factor ranking system that:

    1. **Cleans data** using zipcode-aware imputation (no data loss)
    2. **Calculates market context** (price gaps, efficiency metrics)
    3. **Performs statistical validation** (Chi-Square, ANOVA tests)
    4. **Auto-detects optimal parameters** from your specific dataset
    5. **Filters efficiency traps** (properties overpriced per sqft)
    6. **Ranks opportunities** using weighted scoring (80% financial, 10% structural, 10% vintage)

    **Key Innovation**: All thresholds are data-driven, making the algorithm portable to any market.
    """)

    # Show Run Analysis button
    if uploaded_file is not None and st.session_state.df_raw is not None:
        st.markdown("---")
        st.markdown("### 🚀 Ready to Analyze")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_analysis = st.button(
                "🚀 Run Complete Analysis",
                type="primary",
                width="stretch",
            )

        if run_analysis:
            st.session_state.is_running = True
            st.rerun()

# =============================================================================
# ANALYSIS PIPELINE (runs when button clicked)
# =============================================================================
if st.session_state.is_running and not st.session_state.analysis_complete:
    progress_bar = st.progress(0, text="Starting analysis pipeline...")

    try:
        df_raw = st.session_state.df_raw

        # CHUNK 1
        progress_bar.progress(5, text="Step 1/10: Cleaning data...")
        df_clean, audit_report = chunk_01_data_cleaning(df_raw)
        st.session_state.df_clean = df_clean
        st.session_state.audit_report = audit_report

        # CHUNK 2
        progress_bar.progress(10, text="Step 2/10: Computing market context...")
        df_context = chunk_02_market_context(df_clean)
        st.session_state.df_context = df_context

        # CHUNK 3.1
        progress_bar.progress(20, text="Step 3/10: Analyzing market distribution...")
        univariate_stats = chunk_03_01_univariate_analysis(df_context)
        st.session_state.univariate_stats = univariate_stats

        # CHUNK 3.5
        progress_bar.progress(30, text="Step 4/10: Detecting optimal conditions...")
        condition_results = chunk_03_05_condition_strategy(df_context, max_budget)
        st.session_state.condition_results = condition_results

        # CHUNK 3.6
        progress_bar.progress(40, text="Step 5/10: Detecting optimal decades...")
        vintage_results = chunk_03_06_vintage_value(df_context)
        st.session_state.vintage_results = vintage_results

        # CHUNK 3.7
        progress_bar.progress(50, text="Step 6/10: Detecting optimal grades...")
        grade_results = chunk_03_07_grade_value(df_context)
        st.session_state.grade_results = grade_results

        # CHUNK 3.8
        progress_bar.progress(60, text="Step 7/10: Computing correlations...")
        correlation_results = chunk_03_08_correlation_analysis(df_context)
        st.session_state.correlation_results = correlation_results

        # CHUNK 3.9-3.10
        progress_bar.progress(65, text="Step 8/10: Running statistical tests...")
        chi_square_results = chunk_03_09_chi_square_tests(df_context)
        anova_results = chunk_03_10_anova_tests(df_context)
        st.session_state.chi_square_results = chi_square_results
        st.session_state.anova_results = anova_results

        # CHUNK 4
        progress_bar.progress(70, text="Step 9/10: Applying strategy filters...")
        df_scout = chunk_04_strategy_filter(
            df_context,
            max_budget,
            condition_results['target']
        )

        # CHUNK 5
        progress_bar.progress(80, text="Step 10/10: Computing Smart Scores...")
        df_ranked, scoring_audit = chunk_05_smart_score(
            df_scout,
            df_clean,
            grade_results['optimal_grades'],
            vintage_results['optimal_decades']
        )
        st.session_state.df_ranked = df_ranked
        st.session_state.scoring_audit = scoring_audit

        # CHUNK 5B
        progress_bar.progress(85, text="Validating Smart Score...")
        validation_results = chunk_05b_validate_score(df_ranked)
        st.session_state.validation_results = validation_results

        # CHUNK 6
        progress_bar.progress(90, text="Identifying strategic zones...")
        strategic_zones = chunk_06_strategic_zones(df_ranked)
        st.session_state.strategic_zones = strategic_zones

        # EXPORT
        progress_bar.progress(95, text="Preparing export files...")
        action_list, export_df = prepare_export_files(
            df_clean,
            df_ranked,
            df_ranked['id'].unique()
        )
        st.session_state.action_list = action_list
        st.session_state.export_df = export_df

        progress_bar.progress(100, text="Analysis complete!")

        st.session_state.analysis_complete = True
        st.session_state.is_running = False
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        st.session_state.is_running = False
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)

# =============================================================================
# RESULTS TABS (shown only after analysis complete)
# =============================================================================
if st.session_state.analysis_complete:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Market Overview",
        "🔍 Deep Dive Analysis",
        "🎯 Strategy Detection",
        "🏆 Smart Score Results",
        "🗺️ Geographic Map",
        "📥 Downloads"
    ])

    # =====================================================================
    # TAB 1: MARKET OVERVIEW
    # =====================================================================
    with tab1:
        st.markdown("### 📊 Market Distribution Analysis")

        stats = st.session_state.univariate_stats
        df_context = st.session_state.df_context

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Price", format_currency(stats['mean'], compact=True))
        with col2:
            st.metric("Median Price", format_currency(stats['median'], compact=True))
        with col3:
            st.metric("Skewness", f"{stats['skewness']:.2f}")
        with col4:
            st.metric("Kurtosis", f"{stats['kurtosis']:.2f}")

        # Interpretation
        st.markdown("#### 📋 Statistical Diagnosis")

        if stats['is_skewed']:
            st.warning(f"⚠️ **High Positive Skew ({stats['skewness']:.2f})**: Luxury outliers distort city averages.")

        if stats['is_leptokurtic']:
            st.warning(
                f"⚠️ **Extreme Kurtosis ({stats['kurtosis']:.2f})**: Fat-tail distribution with frequent extreme outliers.")

        if stats['high_volatility']:
            st.warning(
                f"⚠️ **High Volatility (CV: {stats['cv']:.1f}%)**: Standard deviation unreliable, using robust medians.")

        st.info("""
        **Strategic Implication**: This extreme distribution validates our approach of:
        1. Using zipcode-specific medians instead of city averages
        2. Applying robust filtering (Efficiency Gate) to remove outliers
        3. Relying on Spearman (rank-based) correlations over Pearson
        """)

        # Distribution plots
        st.markdown("#### 📈 Distribution Plots")
        fig = plot_market_distribution(df_context, stats)
        st.pyplot(fig)

    # =====================================================================
    # TAB 2: DEEP DIVE ANALYSIS
    # =====================================================================
    with tab2:
        st.markdown("### 🔍 Deep Dive Analysis")

        df_context = st.session_state.df_context

        # Size Trap Analysis
        st.markdown("#### 🎯 Size vs. Value: The Truth Check")
        deep_dive_data = chunk_03_02_size_trap_analysis(df_context, max_budget)

        st.info(f"""
        **Analyzing {format_number(len(deep_dive_data))} properties** within budget and condition ≤ 3.
    
        **Key Question**: Are high profit-gap properties genuinely undervalued, or just small?
        - **Green dots** = True deals (cheaper per sqft than neighborhood)
        - **Red dots** = Size traps (expensive per sqft)
        """)

        fig = plot_size_trap_analysis(deep_dive_data)
        st.pyplot(fig)

        # Density Analysis
        st.markdown("#### 🏘️ Density Analysis")
        density_stats = chunk_03_03_density_analysis(df_context)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(density_stats, width='stretch', hide_index=True)
        with col2:
            fig = plot_density_analysis(df_context)
            st.pyplot(fig)

        # Value Hierarchy
        st.markdown("#### 🏆 Value Hierarchy: Premium Zones")
        data_viz = chunk_03_04_value_hierarchy(df_context)
        fig = plot_value_hierarchy(data_viz)
        st.pyplot(fig)

        # Correlation Analysis
        st.markdown("#### 📈 Correlation Analysis: Pearson vs Spearman")
        corr_results = st.session_state.correlation_results

        st.info(f"""
        **Recommended Method**: {corr_results['recommended_method'].upper()}
    
        Due to high skewness, Spearman (rank-based) correlations are more reliable.
        """)

        fig = plot_correlation_heatmaps(corr_results['pearson'], corr_results['spearman'])
        st.pyplot(fig)

        # Comparison table
        st.markdown("##### Method Comparison (Price Correlations)")
        st.dataframe(corr_results['comparison'], width='stretch', hide_index=True)

        # Statistical Tests
        st.markdown("#### 🧪 Statistical Validation")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Chi-Square Independence Tests")
            chi_results = st.session_state.chi_square_results

            chi_display = chi_results['results'].copy()
            chi_display['chi2'] = chi_display['chi2'].apply(lambda x: f"{x:,.2f}")
            chi_display['p_value'] = chi_display['p_value'].apply(lambda x: f"{x:.2e}")
            chi_display['dof'] = chi_display['dof'].apply(lambda x: format_number(x))
            chi_display['significant'] = chi_display['significant'].apply(lambda x: '✅ Yes' if x else '❌ No')
            chi_display.columns = ['Variable 1', 'Variable 2', 'Chi² Statistic', 'P-value', 'DOF', 'Significant']

            st.dataframe(chi_display, width='stretch', hide_index=True)

        with col2:
            st.markdown("##### ANOVA: Group Mean Comparisons")
            anova_results = st.session_state.anova_results

            anova_display = anova_results['results'].copy()
            anova_display['f_stat'] = anova_display['f_stat'].apply(lambda x: f"{x:,.2f}")
            anova_display['p_value'] = anova_display['p_value'].apply(lambda x: f"{x:.2e}")
            anova_display['significant'] = anova_display['significant'].apply(lambda x: '✅ Yes' if x else '❌ No')
            anova_display.columns = ['Grouping Variable', 'Measure', 'F-Statistic', 'P-value', 'Significant']

            st.dataframe(anova_display, width='stretch', hide_index=True)

        # ANOVA Visualization
        st.markdown("##### ANOVA Visualization: Condition Effect")
        fig = plot_anova_condition(df_context)
        st.pyplot(fig)

    # =====================================================================
    # TAB 3: STRATEGY DETECTION
    # =====================================================================
    with tab3:
        st.markdown("### 🎯 Data-Driven Strategy Detection")

        st.info("""
        **Methodology**: All optimal parameters are **auto-detected** from data using:
        - Median price gap comparisons
        - Profitability thresholds
        - Business logic constraints
    
        This makes the algorithm **portable** to any market without manual recalibration.
        """)

        # Condition Strategy
        st.markdown("#### 🔧 Optimal Conditions")
        condition_results = st.session_state.condition_results

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Detection Results:**")
            st.success(f"🥇 **Ideal**: {condition_results['ideal']}")
            st.warning(f"🥈 **Acceptable**: {condition_results['acceptable']}")
            st.info(f"🎯 **Target List**: {condition_results['target']}")
            st.caption(f"Max Condition: {condition_results['max_condition']}")

        with col2:
            condition_display = condition_results['analysis_table'].copy()
            condition_display['median_price'] = condition_display['median_price'].apply(lambda x: format_currency(x))
            condition_display['median_gap'] = condition_display['median_gap'].apply(lambda x: format_currency(x))
            condition_display['count'] = condition_display['count'].apply(lambda x: format_number(int(x)))
            condition_display.columns = ['Median Price', 'Median Gap', 'Count']
            st.dataframe(condition_display, width='stretch')

        fig = plot_condition_strategy(st.session_state.df_context, max_budget)
        st.pyplot(fig)

        # Vintage Strategy
        st.markdown("#### 📅 Optimal Construction Decades")
        vintage_results = st.session_state.vintage_results

        col1, col2 = st.columns([1, 2])
        with col1:
            st.success(f"**Optimal Decades**: {vintage_results['optimal_decades']}")
            st.caption(f"Market Median: {format_currency(vintage_results['market_median'])}")

        with col2:
            perf_df = pd.DataFrame([
                {'Decade': int(k), 'Median Gap': format_currency(v)}
                for k, v in vintage_results['decade_performance'].items()
            ])
            perf_df = perf_df.sort_values('Decade')
            st.dataframe(perf_df, width='stretch', hide_index=True)

        fig = plot_vintage_value(
            st.session_state.df_context,
            vintage_results['optimal_decades'],
            vintage_results['market_median']
        )
        st.pyplot(fig)

        # Grade Strategy
        st.markdown("#### 🏗️ Optimal Building Grades")
        grade_results = st.session_state.grade_results

        col1, col2 = st.columns([1, 2])
        with col1:
            st.success(f"**Optimal Grades**: {grade_results['optimal_grades']}")
            st.caption(f"Market Median: {format_currency(grade_results['market_median'])}")
            st.caption(f"Min Grade (Business Rule): {grade_results['min_grade']}")

        with col2:
            perf_df = pd.DataFrame([
                {'Grade': int(k), 'Median Gap': format_currency(v)}
                for k, v in grade_results['grade_performance'].items()
            ])
            perf_df = perf_df.sort_values('Grade')
            st.dataframe(perf_df, width='stretch', hide_index=True)

        fig = plot_grade_value(
            st.session_state.df_context,
            grade_results['optimal_grades'],
            grade_results['market_median']
        )
        st.pyplot(fig)

    # =====================================================================
    # TAB 4: SMART SCORE RESULTS
    # =====================================================================
    with tab4:
        st.markdown("### 🏆 Smart Score Results")

        df_ranked = st.session_state.df_ranked
        scoring_audit = st.session_state.scoring_audit
        validation_results = st.session_state.validation_results

        # Scoring Audit
        st.markdown("#### 📋 Scoring Pipeline Audit")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Size Traps Blocked", format_number(scoring_audit['size_traps_blocked']))
        with col2:
            st.metric("Final Candidates", format_number(scoring_audit['candidates_after_trap_filter']))
        with col3:
            st.metric("Mean Score", f"{scoring_audit['mean_score']:.1f}")
        with col4:
            st.metric("Max Score", f"{scoring_audit['max_score']:.1f}")

        # Top Deals
        st.markdown("#### 🥇 Top 10 Investment Opportunities")
        top_10 = get_top_deals(df_ranked, 10)

        display_cols = ['id', 'SMART_SCORE', 'price', 'price_gap', 'sqft_gap',
                        'zipcode', 'grade', 'condition', 'yr_built', 'sqft_living']

        top_10_display = top_10[display_cols].copy()
        top_10_display['price'] = top_10_display['price'].apply(lambda x: format_currency(x))
        top_10_display['price_gap'] = top_10_display['price_gap'].apply(lambda x: format_currency(x))
        top_10_display['SMART_SCORE'] = top_10_display['SMART_SCORE'].apply(lambda x: f"{x:.1f}")

        st.dataframe(top_10_display, width='stretch', hide_index=True)

        # Validation
        st.markdown("#### ✅ Smart Score Validation")

        col1, col2 = st.columns([1, 2])
        with col1:
            validation_display = validation_results['validation_table'].copy()
            validation_display['median_price_gap'] = validation_display['median_price_gap'].apply(
                lambda x: format_currency(x))
            validation_display['median_sqft_gap'] = validation_display['median_sqft_gap'].apply(
                lambda x: f"${x:.2f}/sqft")
            validation_display['median_price_per_sqft'] = validation_display['median_price_per_sqft'].apply(
                lambda x: f"${x:.2f}/sqft")
            validation_display['count'] = validation_display['count'].apply(lambda x: format_number(x))

            st.dataframe(validation_display, width='stretch')

            if validation_results['improvement_pct'] > 0:
                st.success(f"""
                **Validation Passed**: High-score properties have **{validation_results['improvement_pct']:.0f}% higher** 
                median profit gap than low-score properties.
                """)

        with col2:
            df_validation = df_ranked.copy()
            df_validation['score_tier'] = pd.cut(
                df_validation['SMART_SCORE'],
                bins=[0, 40, 60, 100],
                labels=['Low (<40)', 'Mid (40-60)', 'High (>60)']
            )
            fig = plot_score_validation(df_validation)
            st.pyplot(fig)

    # =====================================================================
    # TAB 5: GEOGRAPHIC MAP
    # =====================================================================
    with tab5:
        st.markdown("### 🗺️ Geographic Intelligence")

        strategic_zones = st.session_state.strategic_zones
        df_ranked = st.session_state.df_ranked

        # Strategic Zones Scatter
        st.markdown("#### 📍 Strategic Zones: Profit vs. Volume")

        col1, col2 = st.columns([3, 1])
        with col2:
            use_interactive = st.checkbox("🎯 Interactive Version", value=True,
                                          help="Use Plotly for interactive chart")

        if use_interactive:
            fig = plot_strategic_zones_interactive(strategic_zones)
            st.plotly_chart(fig, width="stretch")
        else:
            fig = plot_strategic_zones(strategic_zones)
            st.pyplot(fig)

        # Top Zones Table
        st.markdown("#### 🏆 Top 10 Target Zipcodes")
        top_zones = strategic_zones.head(10).copy()
        top_zones['avg_potential_profit'] = top_zones['avg_potential_profit'].apply(lambda x: format_currency(x))
        top_zones['zip_avg_price'] = top_zones['zip_avg_price'].apply(lambda x: format_currency(x))
        st.dataframe(top_zones, width='stretch', hide_index=True)

        # Interactive Map
        st.markdown("#### 🗺️ Interactive Property Map")
        st.info("Click on markers for detailed property information. Use layer controls to toggle views.")

        top_500 = df_ranked.head(500)
        folium_map = create_interactive_map(top_500)

        st.components.v1.html(
            folium_map._repr_html_(),
            height=600,
            scrolling=True
        )

    # =====================================================================
    # TAB 6: DOWNLOADS
    # =====================================================================
    with tab6:
        st.markdown("### 📥 Export Results")

        action_list = st.session_state.action_list
        export_df = st.session_state.export_df

        # Action List
        st.markdown("#### 🎯 Action List (Prioritized Opportunities)")
        st.caption(f"{len(action_list)} properties ranked by Smart Score")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(action_list.head(20), width='stretch', hide_index=True)
        with col2:
            csv_action = action_list.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Action List",
                data=csv_action,
                file_name="action_list_optimized.csv",
                mime="text/csv",
                width="stretch"
            )

        st.divider()

        # Full Dataset
        st.markdown("#### 📊 Full Dataset (Power BI Ready)")
        st.caption(f"{len(export_df)} properties with target flags and scores")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(export_df.head(20), width='stretch', hide_index=True)
        with col2:
            csv_full = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Dataset",
                data=csv_full,
                file_name="kc_housing_smart_strategy.csv",
                mime="text/csv",
                width="stretch"
            )

        st.divider()

        # Data Quality Report
        st.markdown("#### 📋 Data Quality Report")
        audit_report = st.session_state.audit_report

        report_df = pd.DataFrame([
            {"Metric": k.replace('_', ' ').title(), "Value": format_number(v)}
            for k, v in audit_report.items()
        ])
        st.dataframe(report_df, width='stretch', hide_index=True)

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("🎓 Developed for ZIGURAT Master in AI for Architecture and Construction | M3U1&U2 Assignment")
    st.caption("📧 Questions? Contact FMP Group 4")