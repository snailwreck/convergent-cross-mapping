import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pyEDM

# --- Page Configuration ---
st.set_page_config(page_title="CCM", layout="wide")

# --- Functions ---
@st.cache_data
def generate_lorenz_data(sigma, rho, beta, t_max, num_points):
    """Generates time series data for the continuous Lorenz attractor."""
    def lorenz_system(t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    initial_state = [1.0, 1.0, 1.0]
    t_eval = np.linspace(0, t_max, num_points)
    solution = solve_ivp(lorenz_system, [0, t_max], initial_state, t_eval=t_eval)
    
    df = pd.DataFrame({
        'Time': np.arange(1, num_points + 1),
        'X': solution.y[0],
        'Y': solution.y[1],
        'Z': solution.y[2]
    })
    return df

@st.cache_data
def generate_logistic_data(rx, ry, beta_xy, beta_yx, num_points):
    """Generates discrete time series data for the coupled logistic map."""
    x = np.zeros(num_points)
    y = np.zeros(num_points)
    
    x[0] = 0.4
    y[0] = 0.2
    
    for t in range(num_points - 1):
        x[t+1] = x[t] * (rx - rx * x[t] - beta_xy * y[t])
        y[t+1] = y[t] * (ry - ry * y[t] - beta_yx * x[t])
        
        # Cap to prevent math explosion if sliders pushed too far
        x[t+1] = max(0, min(1, x[t+1]))
        y[t+1] = max(0, min(1, y[t+1]))
        
    df = pd.DataFrame({
        'Time': np.arange(1, num_points + 1),
        'X': x,
        'Y': y
    })
    return df

# --- Sidebar UI ---
st.sidebar.title("Configuration")

with st.sidebar.expander("Lorenz Parameters (Tab 1)", expanded=True):
    sigma = st.slider("Sigma (σ)", 1.0, 20.0, 10.0)
    rho = st.slider("Rho (ρ)", 10.0, 40.0, 28.0)
    beta = st.slider("Beta (β)", 1.0, 5.0, 2.666)
    t_max = st.number_input("Max Time (t)", 10.0, 100.0, 40.0)

with st.sidebar.expander("Logistic Map Parameters (Tab 2)", expanded=True):
    rx = st.slider("Growth Rate rx", 3.5, 4.0, 3.8, step=0.01)
    ry = st.slider("Growth Rate ry", 3.5, 4.0, 3.5, step=0.01)
    beta_xy = st.slider("Effect of Y on X (β_xy)", 0.0, 0.5, 0.02, step=0.01)
    beta_yx = st.slider("Effect of X on Y (β_yx)", 0.0, 0.5, 0.10, step=0.01)

st.sidebar.header("Global Settings")
num_points = st.sidebar.number_input("Number of Data Points", 500, 5000, 2000, step=100)
max_lib_size = st.sidebar.slider("Max Library Size", 100, num_points - 10, 500, step=100)
lib_step = st.sidebar.number_input("Library Step Size", 10, 200, 20)

# --- Main Application ---
st.title("Convergent Cross Mapping (CCM)")

tab1, tab2 = st.tabs(["Lorenz Attractor (Continuous)", "Coupled Logistic Map (Discrete)"])

# ==========================================
# TAB 1: LORENZ ATTRACTOR
# ==========================================
with tab1:
    st.header("Lorenz Attractor")
    st.markdown(r"$$ \frac{dx}{dt} = \sigma(y-x)$$")
    st.markdown(r"$$\frac{dy}{dt} = x(\rho-z)-y $$")
    st.mardkwon(r"$$\frac{dz}{dt} = xy-\beta z $$")

    df_lorenz = generate_lorenz_data(sigma, rho, beta, t_max, num_points)

    col1, col2 = st.columns(2)

    if st.button("Generate Lorenz Graph & Time Series"):
        with col1:
            st.subheader("Lorenz Attractor (X vs Y)")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(df_lorenz['X'], df_lorenz['Y'], lw=0.5, color='royalblue')
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            st.pyplot(fig1)

        with col2:
            st.subheader("Time Series (X and Y)")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(df_lorenz['Time'][:500], df_lorenz['X'][:500], label="X", lw=1)
            ax2.plot(df_lorenz['Time'][:500], df_lorenz['Y'][:500], label="Y", lw=1)
            ax2.set_xlabel("Time (Index)")
            ax2.set_ylabel("Value")
            ax2.legend()
            st.pyplot(fig2)

    st.markdown("---")
    st.subheader("CCM Analysis: X and Y Coupling")

    if st.button("Run Lorenz CCM Analysis"):
        with st.spinner("Running Convergent Cross Mapping on Lorenz..."):
            lib_sizes_str = f"10 {max_lib_size} {lib_step}"
            
            ccm_result = pyEDM.CCM(
                dataFrame=df_lorenz, E=3, columns="X", target="Y", 
                libSizes=lib_sizes_str, sample=100, showPlot=False
            )
            
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            if 'X:Y' in ccm_result.columns and 'Y:X' in ccm_result.columns:
                ax3.plot(ccm_result['LibSize'], ccm_result['X:Y'], marker='o', label='X cross-maps Y (Y causes X)')
                ax3.plot(ccm_result['LibSize'], ccm_result['Y:X'], marker='s', label='Y cross-maps X (X causes Y)')
            
            ax3.set_xlabel("Library Size (L)")
            ax3.set_ylabel("Correlation (ρ)")
            ax3.set_title("Correlation vs. Library Size")
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.set_ylim([-0.1, 1.1])
            st.pyplot(fig3)

            # Prediction Performance Visualization 
            st.markdown("---")
            st.subheader("Prediction Performance")

            simplex_XY = pyEDM.Simplex(dataFrame=df_lorenz, lib=f"1 {max_lib_size}", pred=f"1 {max_lib_size}", E=3, columns="X", target="Y").dropna()
            simplex_YX = pyEDM.Simplex(dataFrame=df_lorenz, lib=f"1 {max_lib_size}", pred=f"1 {max_lib_size}", E=3, columns="Y", target="X").dropna()

            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("### Predicting Y from X")
                fig4, ax4 = plt.subplots(figsize=(5, 5))
                ax4.scatter(simplex_XY['Observations'], simplex_XY['Predictions'], alpha=0.5, s=10, color='purple')
                min_val = min(simplex_XY['Observations'].min(), simplex_XY['Predictions'].min())
                max_val = max(simplex_XY['Observations'].max(), simplex_XY['Predictions'].max())
                ax4.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Prediction")
                ax4.set_xlabel("True Observed Y")
                ax4.set_ylabel("Predicted Y (from X)")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                st.pyplot(fig4)

            with col4:
                st.markdown("### Predicting X from Y")
                fig5, ax5 = plt.subplots(figsize=(5, 5))
                ax5.scatter(simplex_YX['Observations'], simplex_YX['Predictions'], alpha=0.5, s=10, color='teal')
                min_val = min(simplex_YX['Observations'].min(), simplex_YX['Predictions'].min())
                max_val = max(simplex_YX['Observations'].max(), simplex_YX['Predictions'].max())
                ax5.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Prediction")
                ax5.set_xlabel("True Observed X")
                ax5.set_ylabel("Predicted X (from Y)")
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                st.pyplot(fig5)


# ==========================================
# TAB 2: COUPLED LOGISTIC MAP
# ==========================================
with tab2:
    st.header("Coupled Logistic Map")
    st.markdown(r"$$ X_{t+1} = X_t [r_x - r_x X_t - \beta_{xy} Y_t]$$") 
    st.markdown(r"$$Y_{t+1} = Y_t [r_y - r_y Y_t - \beta_{yx} X_t] $$")

    df_map = generate_logistic_data(rx, ry, beta_xy, beta_yx, num_points)

    col1_m, col2_m = st.columns(2)

    if st.button("Generate Logistic Map & Time Series"):
        with col1_m:
            st.subheader("Phase Space (X vs Y)")
            fig1_m, ax1_m = plt.subplots(figsize=(6, 4))
            ax1_m.scatter(df_map['X'], df_map['Y'], s=2, alpha=0.5, color='purple')
            ax1_m.set_xlabel("X")
            ax1_m.set_ylabel("Y")
            st.pyplot(fig1_m)

        with col2_m:
            st.subheader("Time Series (First 100 Points)")
            fig2_m, ax2_m = plt.subplots(figsize=(6, 4))
            ax2_m.plot(df_map['Time'][:100], df_map['X'][:100], label="X", marker='.', lw=1)
            ax2_m.plot(df_map['Time'][:100], df_map['Y'][:100], label="Y", marker='.', lw=1)
            ax2_m.set_xlabel("Time (Step)")
            ax2_m.set_ylabel("Value")
            ax2_m.legend()
            st.pyplot(fig2_m)

    st.markdown("---")
    st.subheader("CCM Analysis: Asymmetric Causality")

    if st.button("Run Logistic Map CCM Analysis"):
        with st.spinner("Running Convergent Cross Mapping on Logistic Map..."):
            lib_sizes_str = f"10 {max_lib_size} {lib_step}"
            
            ccm_result_m = pyEDM.CCM(
                dataFrame=df_map, E=2, columns="X", target="Y", 
                libSizes=lib_sizes_str, sample=100, showPlot=False
            )
            
            fig3_m, ax3_m = plt.subplots(figsize=(8, 5))
            if 'X:Y' in ccm_result_m.columns and 'Y:X' in ccm_result_m.columns:
                ax3_m.plot(ccm_result_m['LibSize'], ccm_result_m['X:Y'], marker='o', color='teal', label='X cross-maps Y (Y causes X)')
                ax3_m.plot(ccm_result_m['LibSize'], ccm_result_m['Y:X'], marker='s', color='orange', label='Y cross-maps X (X causes Y)')
            
            ax3_m.set_xlabel("Library Size (L)")
            ax3_m.set_ylabel("Correlation (ρ)")
            ax3_m.set_title("CCM Convergence: Asymmetric Coupling")
            ax3_m.legend()
            ax3_m.grid(True, linestyle='--', alpha=0.7)
            ax3_m.set_ylim([-0.1, 1.1])
            st.pyplot(fig3_m)

            # Prediction Performance Visualization 
            st.markdown("---")
            st.subheader("Prediction Performance")

            simplex_XY_m = pyEDM.Simplex(dataFrame=df_map, lib=f"1 {max_lib_size}", pred=f"1 {max_lib_size}", E=2, columns="X", target="Y").dropna()
            simplex_YX_m = pyEDM.Simplex(dataFrame=df_map, lib=f"1 {max_lib_size}", pred=f"1 {max_lib_size}", E=2, columns="Y", target="X").dropna()

            col3_m, col4_m = st.columns(2)
            
            with col3_m:
                st.markdown("### Predicting Y from X")
                fig4_m, ax4_m = plt.subplots(figsize=(5, 5))
                ax4_m.scatter(simplex_XY_m['Observations'], simplex_XY_m['Predictions'], alpha=0.5, s=10, color='teal')
                min_val = min(simplex_XY_m['Observations'].min(), simplex_XY_m['Predictions'].min())
                max_val = max(simplex_XY_m['Observations'].max(), simplex_XY_m['Predictions'].max())
                ax4_m.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Prediction")
                ax4_m.set_xlabel("True Observed Y")
                ax4_m.set_ylabel("Predicted Y (from X)")
                ax4_m.legend()
                ax4_m.grid(True, alpha=0.3)
                st.pyplot(fig4_m)

            with col4_m:
                st.markdown("### Predicting X from Y")
                fig5_m, ax5_m = plt.subplots(figsize=(5, 5))
                ax5_m.scatter(simplex_YX_m['Observations'], simplex_YX_m['Predictions'], alpha=0.5, s=10, color='orange')
                min_val = min(simplex_YX_m['Observations'].min(), simplex_YX_m['Predictions'].min())
                max_val = max(simplex_YX_m['Observations'].max(), simplex_YX_m['Predictions'].max())
                ax5_m.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Prediction")
                ax5_m.set_xlabel("True Observed X")
                ax5_m.set_ylabel("Predicted X (from Y)")
                ax5_m.legend()
                ax5_m.grid(True, alpha=0.3)
                st.pyplot(fig5_m)