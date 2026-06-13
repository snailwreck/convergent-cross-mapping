import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pyEDM

# --- Page Configuration ---
st.set_page_config(page_title="Lorenz Attractor CCM", layout="wide")

# --- Functions ---
@st.cache_data
def generate_lorenz_data(sigma, rho, beta, t_max, num_points):
    """Generates time series data for the Lorenz attractor."""
    def lorenz_system(t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    # Initial conditions
    initial_state = [1.0, 1.0, 1.0]
    t_eval = np.linspace(0, t_max, num_points)
    
    # Solve the differential equations
    solution = solve_ivp(lorenz_system, [0, t_max], initial_state, t_eval=t_eval)
    
    # Create DataFrame suitable for pyEDM
    df = pd.DataFrame({
        'Time': np.arange(1, num_points + 1),
        'X': solution.y[0],
        'Y': solution.y[1],
        'Z': solution.y[2]
    })
    return df

# --- Sidebar UI ---
st.sidebar.title("Configuration")

st.sidebar.header("Lorenz Parameters")
sigma = st.sidebar.slider("Sigma (σ)", 1.0, 20.0, 10.0)
rho = st.sidebar.slider("Rho (ρ)", 10.0, 40.0, 28.0)
beta = st.sidebar.slider("Beta (β)", 1.0, 5.0, 2.666)

st.sidebar.header("Simulation Settings")
num_points = st.sidebar.number_input("Number of Data Points", 500, 5000, 2000, step=100)
t_max = st.sidebar.number_input("Max Time (t)", 10.0, 100.0, 40.0)

st.sidebar.header("CCM Parameters")
max_lib_size = st.sidebar.slider("Max Library Size", 100, num_points, 500, step=100)
lib_step = st.sidebar.number_input("Library Step Size", 10, 200, 20)

# --- Main Application ---
st.title("Convergent Cross Mapping (CCM) on the Lorenz Attractor")
st.markdown(r"$$ \frac{dx}{dt} = \sigma(y-x) $$")
st.markdown(r"$$ \frac{dy}{dt} = x(\rho-z)-y $$")
st.markdown(r"$$ \frac{dz}{dt} = xy-\beta z $$")

# Generate Data
df_lorenz = generate_lorenz_data(sigma, rho, beta, t_max, num_points)

# Layout for Data Visualization
col1, col2 = st.columns(2)

if st.button("Generate Lorenz attractor graph and time series (X vs. Y)"):
    with col1:
        st.subheader("Lorenz Attractor (X vs Y)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(df_lorenz['X'], df_lorenz['Y'], lw=0.5, color='royalblue')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title("Phase Space")
        st.pyplot(fig1)

    with col2:
        st.subheader("Time Series (X and Y)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(df_lorenz['Time'][:500], df_lorenz['X'][:500], label="X", lw=1)
        ax2.plot(df_lorenz['Time'][:500], df_lorenz['Y'][:500], label="Y", lw=1)
        ax2.set_xlabel("Time (Index)")
        ax2.set_ylabel("Value")
        ax2.set_title("First 500 Points")
        ax2.legend()
        st.pyplot(fig2)

# --- pyEDM CCM Execution ---
st.markdown("---")
st.subheader("CCM Analysis: X and Y Coupling")

if st.button("Run CCM Analysis"):
    with st.spinner("Running Convergent Cross Mapping via pyEDM..."):
        # Format library sizes for pyEDM: "start stop step"
        lib_sizes_str = f"10 {max_lib_size} {lib_step}"
        
        # Run CCM (X cross-maps Y, Y cross-maps X)
        ccm_result = pyEDM.CCM(
            dataFrame=df_lorenz,
            E=3,
            columns="X", 
            target="Y", 
            libSizes=lib_sizes_str, 
            sample=100,      
            showPlot=False   
        )
        
        # Plotting the CCM results
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        
        if 'X:Y' in ccm_result.columns and 'Y:X' in ccm_result.columns:
            ax3.plot(ccm_result['LibSize'], ccm_result['X:Y'], marker='o', label='X cross-maps Y (Y causes X)')
            ax3.plot(ccm_result['LibSize'], ccm_result['Y:X'], marker='s', label='Y cross-maps X (X causes Y)')
        else:
            for col in ccm_result.columns:
                if col != 'LibSize':
                    ax3.plot(ccm_result['LibSize'], ccm_result[col], marker='o', label=col)
                    
        ax3.set_xlabel("Library Size (L)")
        ax3.set_ylabel("Correlation (ρ)")
        ax3.set_title("Correlation vs. Library Size")
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.set_ylim([-0.1, 1.1])
        
        st.pyplot(fig3)
        st.success("Analysis complete! As $L$ increases, the correlation $\\rho$ should converge towards 1.0, indicating strong bidirectional causation.")

        # --- Prediction Performance Visualization using Simplex ---
        st.markdown("---")
        st.subheader("Prediction Performance Visualization")
        st.markdown(f"Using the maximum library size (**L = {max_lib_size}**), we extract the predicted values to see how well the shadow manifold of $X$ reconstructs $Y$.")

        # Extract Raw Predictions using Simplex
        simplex_result = pyEDM.Simplex(
            dataFrame=df_lorenz,
            lib=f"1 {max_lib_size}",
            pred=f"1 {max_lib_size}",
            E=3,
            columns="X",
            target="Y"
        )
        
        # Drop NaNs resulting from time-lag embedding
        simplex_result = simplex_result.dropna()

        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Observed vs. Predicted**")
            fig4, ax4 = plt.subplots(figsize=(5, 5))
            
            # Scatter plot of predictions vs true values
            ax4.scatter(simplex_result['Observations'], simplex_result['Predictions'], 
                        alpha=0.5, s=10, color='purple')
            
            # Perfect prediction reference line (y=x)
            min_val = min(simplex_result['Observations'].min(), simplex_result['Predictions'].min())
            max_val = max(simplex_result['Observations'].max(), simplex_result['Predictions'].max())
            ax4.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Prediction (y=x)")
            
            ax4.set_xlabel("True Observed Y")
            ax4.set_ylabel("Predicted Y (from X's history)")
            ax4.set_title(f"Accuracy (L={max_lib_size})")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            st.pyplot(fig4)

        with col4:
            st.markdown("**Time Series Tracking**")
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            
            # Time series overlay (first 150 points for visibility)
            plot_slice = 150 
            time_idx = np.arange(len(simplex_result))[:plot_slice]
            
            ax5.plot(time_idx, simplex_result['Observations'].iloc[:plot_slice], 
                     label="True Y", color='black', lw=2)
            ax5.plot(time_idx, simplex_result['Predictions'].iloc[:plot_slice], 
                     label="Predicted Y", color='orange', linestyle='--', lw=2)
            
            ax5.set_xlabel("Time Step")
            ax5.set_ylabel("Value of Y")
            ax5.set_title("First 150 Predictions")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            st.pyplot(fig5)