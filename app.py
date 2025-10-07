import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from hh_model import HodgkinHuxleySimulator
from ode_analysis import ODEAnalyzer
from plot_styling import setup_plot_theme, get_theme_colors

# Page configuration
st.set_page_config(
    page_title="Hodgkin-Huxley & ODE Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
def inject_custom_css():
    with open("attached_assets/globals_1759862306955.css", "r") as f:
        css = f.read()
    
    # Additional Streamlit-specific styling combined with imported CSS
    streamlit_css = f"""
    <style>
    {css}
    
    .main {{
        background: var(--color-bg);
        color: var(--color-text);
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--color-surface);
        border-radius: var(--radius-lg);
        padding: 0.5rem;
        margin-bottom: 2rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: var(--radius-md);
        color: var(--color-muted);
        font-family: var(--font-aclonica);
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        margin: 0 0.25rem;
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: var(--gradient-light-to-dark-blue) !important;
        color: white !important;
        box-shadow: var(--shadow-md);
    }}
    
    .glass-card {{
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-lg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1.5rem;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .glass-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 255, 238, 0.15), var(--shadow-lg);
        border-color: var(--color-primary);
    }}
    
    .stSelectbox > div > div {{
        background: var(--input-background);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-md);
        color: var(--color-text);
    }}
    
    .stSlider > div > div > div {{
        background: var(--gradient-light-to-dark-blue);
    }}
    
    .stButton > button {{
        background: var(--gradient-light-to-dark-blue);
        border: none;
        border-radius: var(--radius-xl);
        color: white;
        font-family: var(--font-aclonica);
        font-weight: 700;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(56, 152, 255, 0.5);
    }}
    
    .stButton > button:hover {{
        filter: brightness(1.1);
        transform: translateY(-1px);
    }}
    
    .gradient-text-primary {{
        background: var(--gradient-light-to-dark-blue);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: var(--font-aclonica);
        font-weight: 700;
    }}
    
    .content-placeholder {{
        background: var(--glass-bg);
        border: 2px dashed var(--glass-border);
        border-radius: var(--radius-lg);
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        color: var(--color-muted);
        font-style: italic;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        font-family: var(--font-aclonica) !important;
        color: var(--color-text);
    }}
    
    .math-equation {{
        font-family: "Times New Roman", "Computer Modern", serif !important;
        color: var(--color-math);
    }}
    </style>
    """
    
    st.markdown(streamlit_css, unsafe_allow_html=True)

# Initialize CSS
inject_custom_css()

def main():
    # App header
    st.markdown('<h1 class="gradient-text-primary">Hodgkin-Huxley & ODE Analysis Platform</h1>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["🧠 Hodgkin-Huxley Model", "📊 ODE Phase Analysis"])
    
    with tab1:
        hodgkin_huxley_interface()
    
    with tab2:
        ode_analysis_interface()

def hodgkin_huxley_interface():
    st.markdown('<h2 class="gradient-text-secondary">Hodgkin-Huxley Neuron Model</h2>', unsafe_allow_html=True)
    
    # Content placeholder
    st.markdown("""
    <div class="content-placeholder">
        <h3>Educational Content Area</h3>
        <p>This section will contain information about the Hodgkin-Huxley model, its biological significance, and mathematical formulation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Model Parameters")
        
        # HH Model Parameters
        st.markdown("**Membrane Properties**")
        C_m = st.slider("Membrane Capacitance (μF/cm²)", 0.5, 2.0, 1.0, 0.1)
        
        st.markdown("**Maximum Conductances**")
        g_Na = st.slider("Sodium Conductance (mS/cm²)", 50, 200, 120, 5)
        g_K = st.slider("Potassium Conductance (mS/cm²)", 10, 50, 36, 2)
        g_L = st.slider("Leak Conductance (mS/cm²)", 0.1, 1.0, 0.3, 0.05)
        
        st.markdown("**Reversal Potentials**")
        E_Na = st.slider("Sodium Reversal Potential (mV)", 40, 70, 50, 1)
        E_K = st.slider("Potassium Reversal Potential (mV)", -100, -60, -77, 1)
        E_L = st.slider("Leak Reversal Potential (mV)", -70.0, -50.0, -54.387, 0.1)
        
        st.markdown("**Stimulation**")
        I_ext = st.slider("Applied Current (μA/cm²)", -50, 100, 10, 1)
        duration = st.slider("Simulation Duration (ms)", 10, 200, 50, 5)
        
        st.markdown("**Threshold Detection**")
        V_threshold = st.slider("Firing Threshold (mV)", -70.0, -40.0, -55.0, 0.5)
        show_threshold = st.checkbox("Show threshold line", value=True)
        
        simulate_button = st.button("🚀 Run Simulation", key="hh_simulate")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        if simulate_button or 'hh_results' not in st.session_state:
            # Initialize HH simulator
            hh_sim = HodgkinHuxleySimulator(C_m, g_Na, g_K, g_L, E_Na, E_K, E_L)
            
            # Run simulation
            with st.spinner("Running Hodgkin-Huxley simulation..."):
                results = hh_sim.simulate(I_ext, duration)
                st.session_state.hh_results = results
                st.session_state.V_threshold = V_threshold
                st.session_state.show_threshold = show_threshold
        
        if 'hh_results' in st.session_state:
            results = st.session_state.hh_results
            
            # Setup plot theme
            setup_plot_theme()
            
            # Voltage trace
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(results['t'], results['V'], color=get_theme_colors()['primary'], linewidth=2, label='Membrane Potential')
            
            # Add threshold line if enabled
            if st.session_state.get('show_threshold', True):
                threshold = st.session_state.get('V_threshold', -55.0)
                ax1.axhline(y=threshold, color=get_theme_colors()['secondary'], 
                           linestyle='--', linewidth=2, alpha=0.8, label=f'Threshold ({threshold} mV)')
                
                # Detect spikes (threshold crossings)
                V = results['V']
                spike_times = []
                for i in range(1, len(V)):
                    if V[i-1] < threshold and V[i] >= threshold:
                        spike_times.append(results['t'][i])
                
                # Mark spike times
                for spike_t in spike_times:
                    ax1.axvline(x=spike_t, color=get_theme_colors()['accent'], 
                               linestyle=':', linewidth=1, alpha=0.5)
                
                # Display spike count
                if spike_times:
                    ax1.text(0.02, 0.98, f'Spikes: {len(spike_times)}', 
                            transform=ax1.transAxes, fontfamily='Aclonica',
                            verticalalignment='top', fontsize=11,
                            bbox=dict(boxstyle='round', facecolor=get_theme_colors()['surface'], 
                                    alpha=0.8, edgecolor=get_theme_colors()['primary']))
            
            ax1.set_xlabel('Time (ms)', fontfamily='Aclonica')
            ax1.set_ylabel('Membrane Potential (mV)', fontfamily='Aclonica')
            ax1.set_title('Membrane Potential vs Time', fontfamily='Aclonica', fontweight='bold')
            ax1.legend(prop={'family': 'Aclonica', 'size': 9}, loc='upper right')
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            plt.close()
            
            # Gating variables
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            colors = get_theme_colors()
            ax2.plot(results['t'], results['n'], label='n (K⁺ activation)', color=colors['secondary'], linewidth=2)
            ax2.plot(results['t'], results['m'], label='m (Na⁺ activation)', color=colors['accent'], linewidth=2)
            ax2.plot(results['t'], results['h'], label='h (Na⁺ inactivation)', color=colors['chart_3'], linewidth=2)
            ax2.set_xlabel('Time (ms)', fontfamily='Aclonica')
            ax2.set_ylabel('Gating Variable', fontfamily='Aclonica')
            ax2.set_title('Gating Variables vs Time', fontfamily='Aclonica', fontweight='bold')
            ax2.legend(prop={'family': 'Aclonica', 'size': 10})
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close()
            
            # Phase space plot
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.plot(results['V'], results['n'], color=colors['primary'], linewidth=2, alpha=0.8)
            ax3.set_xlabel('Membrane Potential V (mV)', fontfamily='Aclonica')
            ax3.set_ylabel('K⁺ Activation n', fontfamily='Aclonica')
            ax3.set_title('Phase Space: V-n Plane', fontfamily='Aclonica', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            plt.close()
        
        st.markdown('</div>', unsafe_allow_html=True)

def ode_analysis_interface():
    st.markdown('<h2 class="gradient-text-accent">ODE Phase Portraits & Bifurcation Analysis</h2>', unsafe_allow_html=True)
    
    # Content placeholder
    st.markdown("""
    <div class="content-placeholder">
        <h3>Educational Content Area</h3>
        <p>This section will contain theory about phase portraits, bifurcation diagrams, and dynamical systems analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    analysis_type = st.selectbox("Select Analysis Type", 
                                ["Phase Portrait", "Bifurcation Diagram"])
    
    if analysis_type == "Phase Portrait":
        phase_portrait_interface()
    else:
        bifurcation_interface()

def phase_portrait_interface():
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("System Selection")
        
        system_type = st.selectbox("ODE System", [
            "Van der Pol Oscillator", 
            "Lotka-Volterra", 
            "Duffing Oscillator",
            "Pendulum",
            "Custom 2D System"
        ])
        
        st.subheader("Parameters")
        if system_type == "Van der Pol Oscillator":
            mu = st.slider("μ (damping parameter)", 0.1, 5.0, 1.0, 0.1)
            params = {'mu': mu}
        elif system_type == "Lotka-Volterra":
            alpha = st.slider("α (prey growth)", 0.5, 2.0, 1.0, 0.1)
            beta = st.slider("β (predation rate)", 0.5, 2.0, 1.0, 0.1)
            gamma = st.slider("γ (predator death)", 0.5, 2.0, 1.0, 0.1)
            delta = st.slider("δ (predator efficiency)", 0.5, 2.0, 1.0, 0.1)
            params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta}
        elif system_type == "Duffing Oscillator":
            a = st.slider("a (linear stiffness)", -2.0, 2.0, -1.0, 0.1)
            b = st.slider("b (nonlinear stiffness)", 0.1, 2.0, 1.0, 0.1)
            c = st.slider("c (damping)", 0.1, 1.0, 0.3, 0.05)
            params = {'a': a, 'b': b, 'c': c}
        elif system_type == "Pendulum":
            b = st.slider("b (damping)", 0.0, 1.0, 0.25, 0.05)
            g = st.slider("g/l (gravity/length)", 0.5, 2.0, 1.0, 0.1)
            params = {'b': b, 'g': g}
        else:  # Custom system
            st.text_area("dx/dt =", "x + y", key="dx_dt")
            st.text_area("dy/dt =", "-x + y", key="dy_dt")
            params = {}
        
        st.subheader("Plot Settings")
        x_range = st.slider("X range", 1, 10, 5)
        y_range = st.slider("Y range", 1, 10, 5)
        n_trajectories = st.slider("Number of trajectories", 1, 20, 5)
        
        plot_button = st.button("📈 Generate Phase Portrait", key="phase_plot")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        if plot_button or f'phase_results_{system_type}' not in st.session_state:
            ode_analyzer = ODEAnalyzer()
            
            with st.spinner("Generating phase portrait..."):
                fig = ode_analyzer.plot_phase_portrait(
                    system_type, params, x_range, y_range, n_trajectories
                )
                st.session_state[f'phase_results_{system_type}'] = fig
        
        if f'phase_results_{system_type}' in st.session_state:
            fig = st.session_state[f'phase_results_{system_type}']
            st.pyplot(fig)
            plt.close()
        
        st.markdown('</div>', unsafe_allow_html=True)

def bifurcation_interface():
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Bifurcation Analysis")
        
        bifurc_type = st.selectbox("Bifurcation Type", [
            "Logistic Map",
            "Pitchfork Bifurcation", 
            "Hopf Bifurcation",
            "Saddle-Node Bifurcation"
        ])
        
        st.subheader("Parameter Ranges")
        if bifurc_type == "Logistic Map":
            r_min = st.slider("r min", 0.0, 2.0, 0.5, 0.1)
            r_max = st.slider("r max", 2.5, 4.0, 4.0, 0.1)
            param_range = (r_min, r_max)
        elif bifurc_type == "Pitchfork Bifurcation":
            r_min = st.slider("r min", -2.0, 0.0, -1.0, 0.1)
            r_max = st.slider("r max", 0.0, 2.0, 1.0, 0.1)
            param_range = (r_min, r_max)
        elif bifurc_type == "Hopf Bifurcation":
            mu_min = st.slider("μ min", -1.0, 0.0, -0.5, 0.1)
            mu_max = st.slider("μ max", 0.0, 1.0, 0.5, 0.1)
            param_range = (mu_min, mu_max)
        else:  # Saddle-Node
            r_min = st.slider("r min", -1.0, 0.0, -0.5, 0.1)
            r_max = st.slider("r max", 0.0, 1.0, 0.5, 0.1)
            param_range = (r_min, r_max)
        
        resolution = st.slider("Resolution", 100, 2000, 1000, 100)
        
        bifurc_button = st.button("🔄 Generate Bifurcation Diagram", key="bifurc_plot")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        if bifurc_button or f'bifurc_results_{bifurc_type}' not in st.session_state:
            ode_analyzer = ODEAnalyzer()
            
            with st.spinner("Computing bifurcation diagram..."):
                fig = ode_analyzer.plot_bifurcation_diagram(
                    bifurc_type, param_range, resolution
                )
                st.session_state[f'bifurc_results_{bifurc_type}'] = fig
        
        if f'bifurc_results_{bifurc_type}' in st.session_state:
            fig = st.session_state[f'bifurc_results_{bifurc_type}']
            st.pyplot(fig)
            plt.close()
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
