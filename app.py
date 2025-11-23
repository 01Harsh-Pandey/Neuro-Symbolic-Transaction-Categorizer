import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import sys
import os
import yaml
from datetime import datetime
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure page
st.set_page_config(
    page_title="Neuro-Symbolic Transaction Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode and professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-header {
        color: #8898aa;
        font-size: 1.2rem;
        margin-top: 0;
    }
    .tier-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .result-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid #4a5568;
    }
    .metric-card {
        background: #1a202c;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #2d3748;
    }
    .success-tier {
        border-left: 4px solid #48bb78 !important;
        background: #1e2e1e !important;
    }
    .failed-tier {
        border-left: 4px solid #e53e3e !important;
        background: #2e1e1e !important;
    }
    .bypassed-tier {
        border-left: 4px solid #718096 !important;
        background: #2d3748 !important;
        opacity: 0.6;
    }
    .pipeline-container {
        background: #0f1419;
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid #2d3748;
    }
</style>
""", unsafe_allow_html=True)

# Cache the engine to avoid reloading on every interaction
@st.cache_resource
def load_engine():
    """Load the transaction engine with error handling"""
    try:
        from engine import TransactionEngine
        engine = TransactionEngine()
        return engine, None
    except Exception as e:
        return None, str(e)

def initialize_session_state():
    """Initialize session state variables"""
    if 'inference_history' not in st.session_state:
        st.session_state.inference_history = []
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    if 'taxonomy_content' not in st.session_state:
        st.session_state.taxonomy_content = load_taxonomy_file()

def load_taxonomy_file():
    """Load current taxonomy.yaml content"""
    try:
        with open('data/taxonomy.yaml', 'r') as f:
            return f.read()
    except Exception as e:
        return f"# Error loading taxonomy file: {str(e)}"

def save_taxonomy_file(content):
    """Save taxonomy.yaml and clear cache"""
    try:
        with open('data/taxonomy.yaml', 'w') as f:
            f.write(content)
        
        # Clear caches to force reload
        st.cache_resource.clear()
        if 'engine' in st.session_state:
            del st.session_state.engine
        
        return True, "Taxonomy updated successfully! Engine reloaded."
    except Exception as e:
        return False, f"Error saving taxonomy: {str(e)}"

def render_pipeline_visualization(tier_results):
    """Render the three-tier pipeline visualization"""
    st.markdown("### 🚀 Classification Pipeline")
    
    with st.container():
        st.markdown('<div class="pipeline-container">', unsafe_allow_html=True)
        
        # Create three columns for the tiers
        col1, col2, col3 = st.columns(3)
        
        # Tier 1: Rules
        with col1:
            tier1_status = tier_results['tier1']
            css_class = "success-tier" if tier1_status['active'] else "failed-tier" if tier1_status['reached'] else "bypassed-tier"
            st.markdown(f'<div class="tier-card {css_class}">', unsafe_allow_html=True)
            st.markdown("**🎯 Tier 1: Rules Engine**")
            st.markdown(f"**Status:** {'✅ MATCH' if tier1_status['active'] else '❌ NO MATCH' if tier1_status['reached'] else '⚡ BYPASSED'}")
            if tier1_status.get('latency'):
                st.markdown(f"**Latency:** {tier1_status['latency']}ms")
            if tier1_status.get('reason'):
                st.markdown(f"*{tier1_status['reason']}*")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tier 2: AI Model
        with col2:
            tier2_status = tier_results['tier2']
            css_class = "success-tier" if tier2_status['active'] else "failed-tier" if tier2_status['reached'] else "bypassed-tier"
            st.markdown(f'<div class="tier-card {css_class}">', unsafe_allow_html=True)
            st.markdown("**🧠 Tier 2: AI Model**")
            st.markdown(f"**Status:** {'✅ CONFIDENT' if tier2_status['active'] else '❌ LOW CONFIDENCE' if tier2_status['reached'] else '⚡ BYPASSED'}")
            if tier2_status.get('confidence'):
                st.markdown(f"**Confidence:** {tier2_status['confidence']:.1%}")
            if tier2_status.get('latency'):
                st.markdown(f"**Latency:** {tier2_status['latency']}ms")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tier 3: Semantic Memory
        with col3:
            tier3_status = tier_results['tier3']
            css_class = "success-tier" if tier3_status['active'] else "bypassed-tier"
            st.markdown(f'<div class="tier-card {css_class}">', unsafe_allow_html=True)
            st.markdown("**💾 Tier 3: Vector Memory**")
            st.markdown(f"**Status:** {'✅ FALLBACK' if tier3_status['active'] else '⚡ BYPASSED'}")
            if tier3_status.get('similarity'):
                st.markdown(f"**Similarity:** {tier3_status['similarity']:.3f}")
            if tier3_status.get('latency'):
                st.markdown(f"**Latency:** {tier3_status['latency']}ms")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_result_card(result):
    """Render the classification result card"""
    st.markdown("### 📊 Classification Result")
    
    with st.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**🎯 Category**")
            st.markdown(f"# {result['category']} > {result['subcategory']}")
            
        with col2:
            st.markdown(f"**📈 Confidence**")
            confidence_color = "normal" if result['confidence'] > 0.7 else "off" if result['confidence'] > 0.4 else "inverse"
            st.metric(
                label="Confidence Score",
                value=f"{result['confidence']:.1%}",
                delta=None,
                delta_color=confidence_color
            )
            
        with col3:
            st.markdown(f"**⚡ Performance**")
            avg_latency = np.mean([r['latency_ms'] for r in st.session_state.inference_history[-10:]]) if st.session_state.inference_history else 0
            latency_delta = avg_latency - result['latency_ms'] if avg_latency > 0 else None
            st.metric(
                label="Latency",
                value=f"{result['latency_ms']}ms",
                delta=f"{(latency_delta or 0):.1f}ms" if latency_delta is not None else None,
                delta_color="normal" if (latency_delta or 0) > 0 else "inverse"
            )
        
        # Reason box
        st.markdown("**💡 Decision Reasoning**")
        st.info(result['reason'])
        
        st.markdown('</div>', unsafe_allow_html=True)

def mode_live_inference(engine):
    """Live inference mode - the main demo"""
    st.markdown('<h1 class="main-header">Live Inference</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time transaction classification with three-tier architecture</p>', unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([3, 1])
    with col1:
        transaction_input = st.text_input(
            "**📥 Enter Transaction String**",
            placeholder="e.g., UBER *TRIP 84920 CA or Starbucks 0229...",
            value=""
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        classify_clicked = st.button("🚀 Classify Transaction", type="primary", use_container_width=True)
    
    # Example transactions
    st.markdown("**💡 Quick Examples**")
    example_cols = st.columns(4)
    examples = [
        ("UBER *TRIP", "Rule Match"),
        ("Starbucks 0229", "AI Classification"), 
        ("SQ *MERCHANT 231", "Semantic Fallback"),
        ("UNKNOWN TXN", "Edge Case")
    ]
    
    for i, (example, desc) in enumerate(examples):
        with example_cols[i]:
            if st.button(f"`{example}`", use_container_width=True, help=desc):
                st.session_state.example_input = example
                st.rerun()
    
    # Use example if clicked
    if hasattr(st.session_state, 'example_input'):
        transaction_input = st.session_state.example_input
        del st.session_state.example_input
        classify_clicked = True
    
    if classify_clicked and transaction_input:
        with st.spinner("🔄 Processing through neuro-symbolic pipeline..."):
            # Run inference
            start_time = time.time()
            result = engine.predict(transaction_input)
            processing_time = (time.time() - start_time) * 1000
            
            # Add to history
            history_entry = {
                'timestamp': datetime.now(),
                'input': transaction_input,
                'result': result,
                'processing_time': processing_time
            }
            st.session_state.inference_history.append(history_entry)
            
            # Determine tier results for visualization
            tier_results = {
                'tier1': {
                    'active': result['tier'] == 1,
                    'reached': True,  # Always check tier 1 first
                    'latency': result['latency_ms'] if result['tier'] == 1 else None,
                    'reason': result['reason'] if result['tier'] == 1 else None
                },
                'tier2': {
                    'active': result['tier'] == 2,
                    'reached': result['tier'] >= 2,
                    'confidence': result['confidence'] if result['tier'] == 2 else None,
                    'latency': result['latency_ms'] if result['tier'] == 2 else None
                },
                'tier3': {
                    'active': result['tier'] == 3,
                    'reached': True,  # Always the last resort
                    'similarity': result.get('similarity', 0),
                    'latency': result['latency_ms'] if result['tier'] == 3 else None
                }
            }
            
            # Render pipeline visualization
            render_pipeline_visualization(tier_results)
            
            # Render result card
            render_result_card(result)
            
            # Show success toast
            st.toast(f"✅ Classification complete! Tier {result['tier']} - {result['latency_ms']}ms", icon="🎯")
    
    # Recent inferences
    if st.session_state.inference_history:
        st.markdown("### 📜 Recent Inferences")
        recent_df = pd.DataFrame([
            {
                'Time': entry['timestamp'].strftime('%H:%M:%S'),
                'Input': entry['input'][:50] + '...' if len(entry['input']) > 50 else entry['input'],
                'Category': entry['result']['category'],
                'Subcategory': entry['result']['subcategory'],
                'Tier': entry['result']['tier'],
                'Confidence': f"{entry['result']['confidence']:.1%}",
                'Latency': f"{entry['result']['latency_ms']}ms"
            }
            for entry in st.session_state.inference_history[-5:]
        ])
        st.dataframe(recent_df, use_container_width=True, hide_index=True)

def mode_taxonomy_operations():
    """Taxonomy editing mode"""
    st.markdown('<h1 class="main-header">Taxonomy Operations</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dynamic rule management and engine hot-reload</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📝 Taxonomy Configuration")
        st.markdown("Edit the YAML taxonomy below. Changes take effect immediately after saving.")
        
        # Taxonomy editor
        taxonomy_content = st.text_area(
            "Taxonomy YAML",
            value=st.session_state.taxonomy_content,
            height=600,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### ⚡ Actions")
        
        if st.button("💾 Save & Apply", type="primary", use_container_width=True):
            success, message = save_taxonomy_file(taxonomy_content)
            if success:
                st.session_state.taxonomy_content = taxonomy_content
                st.toast("✅ " + message, icon="💾")
                st.rerun()
            else:
                st.error("❌ " + message)
        
        st.markdown("---")
        st.markdown("**🔧 Engine Status**")
        
        # Check if engine is loaded
        engine, error = load_engine()
        if engine:
            st.success("✅ Engine Loaded")
            info = engine.get_engine_info()
            st.metric("Rules", f"{info['rules_loaded']}")
            st.metric("Categories", f"{info['total_categories']}")
            st.metric("Memory Ref", f"{info['semantic_references']}")
        else:
            st.error("❌ Engine Error")
            st.code(error)
        
        st.markdown("---")
        st.markdown("**📚 YAML Guide**")
        st.markdown("""
        - `taxonomy`: Root list
        - `id`: Machine-friendly ID
        - `name`: Human-readable name  
        - `subcategories`: List of sub-items
        - `keywords`: Regex patterns
        """)
    
    # Show current rules summary
    try:
        data = yaml.safe_load(taxonomy_content)
        if data and 'taxonomy' in data:
            st.markdown("### 📊 Current Taxonomy Summary")
            
            categories = data['taxonomy']
            total_keywords = sum(
                len(sub['keywords']) 
                for cat in categories 
                for sub in cat['subcategories']
            )
            total_subcategories = sum(len(cat['subcategories']) for cat in categories)
            
            metric_cols = st.columns(4)
            metric_cols[0].metric("Categories", len(categories))
            metric_cols[1].metric("Subcategories", total_subcategories)
            metric_cols[2].metric("Keywords", total_keywords)
            metric_cols[3].metric("Total Rules", total_keywords)
            
    except yaml.YAMLError as e:
        st.error(f"❌ YAML Syntax Error: {str(e)}")

def mode_analytics(engine):
    """Analytics and batch processing mode"""
    st.markdown('<h1 class="main-header">System Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Performance metrics and batch classification insights</p>', unsafe_allow_html=True)
    
    # Batch processing section
    st.markdown("### 🚀 Batch Processing")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("📊 Process 100 Sample Transactions", type="primary", use_container_width=True):
            with st.spinner("Processing batch transactions..."):
                process_batch_transactions(engine)
    
    with col2:
        sample_size = st.selectbox("Sample Size", [50, 100, 200], index=1)
    
    with col3:
        if st.session_state.batch_results is not None:
            if st.button("🔄 Clear Results", use_container_width=True):
                st.session_state.batch_results = None
                st.rerun()
    
    # Display batch results if available
    if st.session_state.batch_results:
        results = st.session_state.batch_results
        df = results['dataframe']
        
        # Key metrics
        st.markdown("### 📈 Performance Metrics")
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            avg_latency = df['latency_ms'].mean()
            st.metric(
                "Avg Latency", 
                f"{avg_latency:.1f}ms",
                delta="- Fast" if avg_latency < 20 else "+ Slow" if avg_latency > 50 else None,
                delta_color="inverse" if avg_latency > 50 else "normal"
            )
        
        with metric_cols[1]:
            rule_coverage = (df['tier'] == 1).mean() * 100
            st.metric(
                "Rule Coverage", 
                f"{rule_coverage:.1f}%",
                delta="+ Efficient" if rule_coverage > 60 else "- Needs Rules" if rule_coverage < 30 else None
            )
        
        with metric_cols[2]:
            ai_confidence = df[df['tier'] == 2]['confidence'].mean() if (df['tier'] == 2).any() else 0
            st.metric(
                "AI Confidence", 
                f"{ai_confidence:.1%}",
                delta="+ Reliable" if ai_confidence > 0.7 else "- Uncertain" if ai_confidence < 0.5 else None
            )
        
        with metric_cols[3]:
            fallback_rate = (df['tier'] == 3).mean() * 100
            st.metric(
                "Fallback Rate", 
                f"{fallback_rate:.1f}%",
                delta="+ Robust" if fallback_rate < 10 else "- High Ambiguity" if fallback_rate > 30 else None,
                delta_color="inverse" if fallback_rate > 30 else "normal"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Source Distribution")
            source_counts = df['source'].value_counts()
            fig_source = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_source.update_layout(showlegend=True, height=300)
            st.plotly_chart(fig_source, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Confidence Distribution")
            fig_confidence = px.histogram(
                df, 
                x='confidence',
                nbins=20,
                title="Confidence Scores Distribution",
                color_discrete_sequence=['#6366f1']
            )
            fig_confidence.update_layout(showlegend=False, height=300)
            fig_confidence.add_vline(x=0.6, line_dash="dash", line_color="red", annotation_text="60% Threshold")
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Detailed results table
        st.markdown("#### 📋 Detailed Results")
        display_df = df[['input', 'category', 'subcategory', 'source', 'confidence', 'latency_ms']].copy()
        display_df['input'] = display_df['input'].str[:40] + '...'
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df['latency_ms'] = display_df['latency_ms'].apply(lambda x: f"{x:.1f}ms")
        
        st.dataframe(display_df, use_container_width=True, height=300)
    
    else:
        # Placeholder when no batch results
        st.info("👆 Click 'Process Sample Transactions' to run batch analysis and see performance metrics.")

def process_batch_transactions(engine):
    """Process a batch of transactions for analytics"""
    try:
        # Load synthetic dataset
        df = pd.read_csv('data/synthetic_dataset.csv')
        sample_df = df.head(100)  # First 100 rows
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, row in enumerate(sample_df.itertuples()):
            status_text.text(f"Processing {i+1}/{len(sample_df)}: {row.raw_text[:30]}...")
            
            result = engine.predict(row.raw_text)
            results.append({
                'input': row.raw_text,
                'category': result['category'],
                'subcategory': result['subcategory'],
                'source': result['source'],
                'confidence': result['confidence'],
                'tier': result['tier'],
                'latency_ms': result['latency_ms']
            })
            
            progress_bar.progress((i + 1) / len(sample_df))
        
        status_text.text("✅ Batch processing complete!")
        
        # Store results
        st.session_state.batch_results = {
            'dataframe': pd.DataFrame(results),
            'processed_at': datetime.now()
        }
        
    except Exception as e:
        st.error(f"❌ Batch processing failed: {str(e)}")

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Load engine
    engine, error = load_engine()
    
    # Sidebar navigation
    st.sidebar.markdown("# 🧠 Neuro-Symbolic Engine")
    st.sidebar.markdown("---")
    
    # Engine status in sidebar
    if engine:
        st.sidebar.success("✅ Engine Ready")
        info = engine.get_engine_info()
        st.sidebar.metric("Rules", info['rules_loaded'])
        st.sidebar.metric("Memory", f"{info['semantic_references']} refs")
    else:
        st.sidebar.error("❌ Engine Offline")
        st.sidebar.error(f"Error: {error}")
    
    st.sidebar.markdown("---")
    
    # Navigation
    mode = st.sidebar.radio(
        "**Navigation**",
        ["Live Inference", "Taxonomy Operations", "System Analytics"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 System Info**")
    
    if st.session_state.inference_history:
        recent_latency = np.mean([r['result']['latency_ms'] for r in st.session_state.inference_history[-5:]])
        st.sidebar.metric("Avg Latency", f"{recent_latency:.1f}ms")
    
    st.sidebar.metric("Inferences", len(st.session_state.inference_history))
    
    # Render selected mode
    if mode == "Live Inference" and engine:
        mode_live_inference(engine)
    elif mode == "Taxonomy Operations":
        mode_taxonomy_operations()
    elif mode == "System Analytics" and engine:
        mode_analytics(engine)
    else:
        if not engine:
            st.error(f"🚫 Engine not available: {error}")
            st.info("Please check that all model files are properly generated and try restarting the app.")

if __name__ == "__main__":
    main()