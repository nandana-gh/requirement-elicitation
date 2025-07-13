import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Import our modules
from core.proposition_extractor import PropositionExtractor
from core.relation_analyzer import RelationAnalyzer
from core.negation_generator import NegationGenerator
from core.quality_analyzer import QualityAnalyzer
from core.database_manager import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="Requirement Elicitation Tool (RET)",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    return {
        'proposition_extractor': PropositionExtractor(),
        'relation_analyzer': RelationAnalyzer(),
        'negation_generator': NegationGenerator(),
        'quality_analyzer': QualityAnalyzer(),
        'database_manager': DatabaseManager()
    }

# Main function
def main():
    st.title("📋 Requirement Elicitation Tool (RET)")
    st.markdown("---")
    
    # Initialize components
    components = initialize_components()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Analysis Options")
        
        # Analysis options
        st.subheader("Analysis Settings")
        include_negations = st.checkbox("Include Negation Analysis", value=True)
        include_quality = st.checkbox("Include Quality Analysis", value=True)
        include_relations = st.checkbox("Include Relation Analysis", value=True)
        
        # Export options
        st.subheader("Export Options")
        export_format = st.selectbox(
            "Export Format",
            ["Excel", "CSV", "Both"]
        )
        
        st.markdown("---")
        st.markdown("### 📈 Statistics")
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            if 'propositions' in results:
                st.metric("Propositions", len(results['propositions']))
            if 'relations' in results:
                st.metric("Relations", len(results['relations']))
            if 'quality_analysis' in results:
                st.metric("Quality Score", f"{results['quality_analysis']['overall_score']:.2f}")
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Input Requirements", 
        "🔍 Analysis Results", 
        "📊 Quality Report", 
        "📋 Data Tables", 
        "💾 Export Results"
    ])
    
    with tab1:
        input_requirements(components)
    
    with tab2:
        if 'analysis_results' in st.session_state:
            display_analysis_results(st.session_state.analysis_results)
        else:
            st.info("Please input requirements and run analysis first.")
    
    with tab3:
        if 'analysis_results' in st.session_state:
            display_quality_report(st.session_state.analysis_results)
        else:
            st.info("Please input requirements and run analysis first.")
    
    with tab4:
        if 'analysis_results' in st.session_state:
            display_data_tables(st.session_state.analysis_results, components['database_manager'])
        else:
            st.info("Please input requirements and run analysis first.")
    
    with tab5:
        if 'analysis_results' in st.session_state:
            export_results(st.session_state.analysis_results, components['database_manager'], export_format)
        else:
            st.info("Please input requirements and run analysis first.")

def input_requirements(components):
    st.header("📝 Input Requirements")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload", "Sample Requirements"]
    )
    
    requirements = []
    
    if input_method == "Text Input":
        st.subheader("Enter Requirements")
        
        # Multiple text inputs
        num_requirements = st.number_input(
            "Number of requirements:", 
            min_value=1, 
            max_value=50, 
            value=3
        )
        
        for i in range(num_requirements):
            req = st.text_area(
                f"Requirement {i+1}:",
                key=f"req_{i}",
                height=100,
                placeholder="Enter your requirement statement here..."
            )
            if req.strip():
                requirements.append(req.strip())
    
    elif input_method == "File Upload":
        st.subheader("Upload Requirements File")
        
        uploaded_file = st.file_uploader(
            "Choose a text file with requirements (one per line):",
            type=['txt', 'csv']
        )
        
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.txt'):
                content = uploaded_file.read().decode('utf-8')
                requirements = [line.strip() for line in content.split('\n') if line.strip()]
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'requirement' in df.columns:
                    requirements = df['requirement'].tolist()
                else:
                    requirements = df.iloc[:, 0].tolist()
    
    elif input_method == "Sample Requirements":
        st.subheader("Sample Requirements")
        
        sample_requirements = [
            "The system shall accept user input and validate it before processing.",
            "The software must provide real-time monitoring of system performance.",
            "The application shall enable users to export data in multiple formats.",
            "The system must handle concurrent user sessions without conflicts.",
            "The software shall provide backup and recovery functionality."
        ]
        
        st.info("Using sample requirements for demonstration.")
        requirements = sample_requirements
    
    # Display requirements
    if requirements:
        st.subheader("📋 Requirements Summary")
        
        for i, req in enumerate(requirements, 1):
            st.write(f"**R{i}:** {req}")
        
        # Analysis button
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Analyzing requirements..."):
                run_analysis(requirements, components)
                st.success("Analysis completed!")
                st.rerun()

def run_analysis(requirements, components):
    """Run the complete analysis pipeline."""
    results = {
        'requirements': requirements,
        'timestamp': datetime.now().isoformat()
    }
    
    # Always create new instances for stateful classes
    from core.proposition_extractor import PropositionExtractor
    from core.relation_analyzer import RelationAnalyzer
    from core.negation_generator import NegationGenerator

    # Step 1: Extract propositions
    st.info("Extracting propositions...")
    extraction_results = PropositionExtractor().extract_propositions(requirements)
    results['propositions'] = extraction_results['all_propositions']
    results['requirement_propositions'] = extraction_results['requirement_propositions']
    
    # Step 2: Analyze relations
    if st.session_state.get('include_relations', True):
        st.info("Analyzing relations...")
        relation_results = RelationAnalyzer().analyze_relations(results['propositions'])
        results['relations'] = relation_results['relations']
        results['relation_matrix'] = relation_results['relation_matrix']
        results['relation_statistics'] = relation_results['statistics']
    
    # Step 3: Generate negations
    if st.session_state.get('include_negations', True):
        st.info("Generating negations...")
        negation_results = NegationGenerator().generate_negations(results['propositions'])
        results['negations'] = negation_results['negations']
        results['negation_inferences'] = negation_results['inferences']
        results['negation_statistics'] = negation_results['statistics']
    
    # Step 4: Quality analysis
    if st.session_state.get('include_quality', True):
        st.info("Analyzing quality...")
        quality_results = components['quality_analyzer'].analyze_requirement_quality(
            requirements, results['propositions']
        )
        per_prop = components['quality_analyzer'].analyze_per_proposition(
            results['propositions'], results['propositions']
        )
        results['quality_analysis'] = quality_results
        results['per_proposition_analysis'] = per_prop
    
    # Store results in session state
    st.session_state.analysis_results = results

def display_analysis_results(results):
    st.header("🔍 Analysis Results")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Requirements", len(results['requirements']))
    
    with col2:
        st.metric("Propositions", len(results['propositions']))
    
    with col3:
        if 'relations' in results:
            st.metric("Relations", len(results['relations']))
        else:
            st.metric("Relations", 0)
    
    with col4:
        if 'quality_analysis' in results:
            st.metric("Quality Score", f"{results['quality_analysis']['overall_score']:.2f}")
        else:
            st.metric("Quality Score", "N/A")
    
    # Proposition analysis
    st.subheader("📝 Proposition Analysis")
    
    if 'propositions' in results:
        prop_df = pd.DataFrame(results['propositions'])
        
        # Proposition type distribution
        if 'proposition_type' in prop_df.columns:
            type_counts = prop_df['proposition_type'].value_counts()
            fig = px.pie(
                values=type_counts.values, 
                names=type_counts.index,
                title="Proposition Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display propositions table
        st.dataframe(
            prop_df[['proposition_id', 'proposition_statement', 'proposition_type']],
            use_container_width=True
        )
    
    # Relation analysis
    if 'relations' in results:
        st.subheader("🔗 Relation Analysis")
        rel_df = pd.DataFrame(results['relations'])
        if not rel_df.empty and all(col in rel_df.columns for col in ['relation_id', 'proposition_id_1', 'proposition_id_2', 'type', 'confidence']):
            # Relation type distribution
            if 'type' in rel_df.columns:
                rel_counts = rel_df['type'].value_counts()
                fig = px.bar(
                    x=rel_counts.index, 
                    y=rel_counts.values,
                    title="Relation Type Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            # Display relations table
            st.dataframe(
                rel_df[['relation_id', 'proposition_id_1', 'proposition_id_2', 'type', 'confidence']],
                use_container_width=True
            )
        else:
            st.info("No relation data available or columns missing.")

def display_quality_report(results):
    st.header("📊 Quality Report")
    
    if 'quality_analysis' not in results:
        st.warning("Quality analysis not available.")
        return
    
    quality = results['quality_analysis']
    
    # Overall quality score
    st.subheader("Overall Quality Score")
    
    score = quality['overall_score']
    if score >= 0.8:
        st.success(f"Quality Score: {score:.2f} (Good)")
    elif score >= 0.6:
        st.warning(f"Quality Score: {score:.2f} (Fair)")
    else:
        st.error(f"Quality Score: {score:.2f} (Poor)")
    
    # Quality metrics radar chart
    metrics = ['completeness', 'consistency', 'clarity', 'ambiguity']
    scores = [quality[metric]['score'] for metric in metrics]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=metrics,
        fill='toself',
        name='Quality Metrics'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Quality Metrics Radar Chart"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Issues Found")
        
        for metric, data in quality.items():
            if isinstance(data, dict) and 'issues' in data and data['issues']:
                st.write(f"**{metric.title()}:**")
                for issue in data['issues'][:3]:  # Show first 3 issues
                    st.write(f"• {issue}")
    
    with col2:
        st.subheader("Recommendations")
        for rec in quality.get('recommendations', []):
            st.write(f"• {rec}")

    if 'contradictions' in quality and quality['contradictions']:
        st.subheader("Contradictions Detected")
        for c in quality['contradictions']:
            st.write(f"**{c['proposition1_id']}**: {c['proposition1']}")
            st.write(f"**{c['proposition2_id']}**: {c['proposition2']}")
            st.write(f"Reason: {c['reason']}")
            st.markdown("---")

    if 'per_proposition_analysis' in results:
        st.subheader("Per-Proposition Analysis")
        df = pd.DataFrame(results['per_proposition_analysis'])
        st.dataframe(df, use_container_width=True)

def display_data_tables(results, db_manager):
    st.header("📋 Data Tables")
    
    # Create tables
    if 'requirements' in results:
        req_df = db_manager.create_requirement_table(results['requirements'])
        st.subheader("Requirements Table")
        st.dataframe(req_df, use_container_width=True)
    
    if 'propositions' in results:
        prop_df = db_manager.create_proposition_table(results['propositions'])
        st.subheader("Propositions Table")
        st.dataframe(prop_df, use_container_width=True)
    
    if 'relations' in results:
        rel_df = db_manager.create_relation_table(results['relations'])
        st.subheader("Relations Table")
        st.dataframe(rel_df, use_container_width=True)
    
    if 'negation_inferences' in results:
        inf_df = db_manager.create_inference_table(results['negation_inferences'])
        st.subheader("Inferences Table")
        st.dataframe(inf_df, use_container_width=True)

def export_results(results, db_manager, export_format):
    st.header("💾 Export Results")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Options")
        
        if export_format == "Excel":
            if st.button("📊 Export to Excel"):
                with st.spinner("Exporting to Excel..."):
                    filepath = db_manager.export_to_excel(results)
                    st.success(f"Exported to: {filepath}")
                    
                    # Download button
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label="📥 Download Excel File",
                            data=f.read(),
                            file_name=os.path.basename(filepath),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        
        elif export_format == "CSV":
            if st.button("📄 Export to CSV"):
                with st.spinner("Exporting to CSV..."):
                    filepaths = db_manager.export_to_csv(results)
                    st.success(f"Exported {len(filepaths)} CSV files")
                    
                    # Download buttons
                    for filepath in filepaths:
                        with open(filepath, 'rb') as f:
                            st.download_button(
                                label=f"📥 Download {os.path.basename(filepath)}",
                                data=f.read(),
                                file_name=os.path.basename(filepath),
                                mime="text/csv"
                            )
        
        else:  # Both
            if st.button("📊 Export All Formats"):
                with st.spinner("Exporting..."):
                    # Excel
                    excel_path = db_manager.export_to_excel(results)
                    st.success(f"Excel exported to: {excel_path}")
                    
                    # CSV
                    csv_paths = db_manager.export_to_csv(results)
                    st.success(f"CSV files exported: {len(csv_paths)} files")
                    
                    # Download buttons
                    with open(excel_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Excel File",
                            data=f.read(),
                            file_name=os.path.basename(excel_path),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
    
    with col2:
        st.subheader("Export Summary")
        st.write(f"**Requirements:** {len(results['requirements'])}")
        st.write(f"**Propositions:** {len(results['propositions'])}")
        if 'relations' in results:
            st.write(f"**Relations:** {len(results['relations'])}")
        if 'negation_inferences' in results:
            st.write(f"**Inferences:** {len(results['negation_inferences'])}")

if __name__ == "__main__":
    main() 