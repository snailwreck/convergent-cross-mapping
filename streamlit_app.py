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

# CHANGED: Number of data points is now a slider
num_points = st.sidebar.slider("Number of Data Points (Length)", 500, 5000, 2000, step=100)

# FIX: Dynamic bounds calculation to prevent slider constraint crashes
absolute_max_lib = num_points - 50
default_lib_value = min(500, absolute_max_lib)

max_lib_size = st.sidebar.slider("Max Library Size", 100, absolute_max_lib, default_lib_value, step=50)
lib_step = st.sidebar.number_input("Library Step Size", 10, 200, 20)

# --- Main Application ---
st.title("Convergent Cross Mapping (CCM)")

tab1, tab2 = st.tabs(["Lorenz Attractor", "Coupled Logistic Map"])

# ==========================================
# TAB 1: LORENZ ATTRACTOR
# ==========================================
with tab1:
    st.header("Lorenz Attractor")
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
            # Plot dynamic slice based on generated length
            plot_limit = min(500, num_points)
            ax2.plot(df_lorenz['Time'][:plot_limit], df_lorenz['X'][:plot_limit], label="X", lw=1)
            ax2.plot(df_lorenz['Time'][:plot_limit], df_lorenz['Y'][:plot_limit], label="Y", lw=1)
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

            # --- Prediction Performance via pyEDM.Simplex ---
            st.markdown("---")
            st.subheader("Prediction Performance")

            col3, col4 = st.columns(2)
            lib_range = [1, int(max_lib_size)]
            pred_range = [1, int(num_points)]
            
            with col3:
                st.markdown("### Does X cause Y?")
                simplex_XY = pyEDM.Simplex(
                    dataFrame=df_lorenz, lib=lib_range, pred=pred_range,
                    columns="Y", target="X", E=3, Tp=0, tau=-1
                )
                
                fig_xy, ax_xy = plt.subplots(figsize=(5, 5))
                ax_xy.scatter(simplex_XY['Observations'], simplex_XY['Predictions'], alpha=0.4, edgecolors='none', color='C0')
                ax_xy.plot([simplex_XY['Observations'].min(), simplex_XY['Observations'].max()],
                            [simplex_XY['Observations'].min(), simplex_XY['Observations'].max()], 'r--', lw=2)
                ax_xy.set_xlabel("Observed X")
                ax_xy.set_ylabel("Predicted X from M_Y")
                ax_xy.set_title("Cross-mapping Performance")
                st.pyplot(fig_xy)
                plt.close()

            with col4:
                st.markdown("### Does Y cause X?")
                simplex_YX = pyEDM.Simplex(
                    dataFrame=df_lorenz, lib=lib_range, pred=pred_range,
                    columns="X", target="Y", E=3, Tp=0, tau=-1
                )
                
                fig_yx, ax_yx = plt.subplots(figsize=(5, 5))
                ax_yx.scatter(simplex_YX['Observations'], simplex_YX['Predictions'], alpha=0.4, edgecolors='none', color='C1')
                ax_yx.plot([simplex_YX['Observations'].min(), simplex_YX['Observations'].max()],
                            [simplex_YX['Observations'].min(), simplex_YX['Observations'].max()], 'r--', lw=2)
                ax_yx.set_xlabel("Observed Y")
                ax_yx.set_ylabel("Predicted Y from M_X")
                ax_yx.set_title("Cross-mapping Performance")
                st.pyplot(fig_yx)
                plt.close()


# ==========================================
# TAB 2: COUPLED LOGISTIC MAP
# ==========================================
with tab2:
    st.header("Coupled Logistic Map")
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
            st.subheader("Time Series")
            fig2_m, ax2_m = plt.subplots(figsize=(6, 4))
            plot_limit_m = min(100, num_points)
            ax2_m.plot(df_map['Time'][:plot_limit_m], df_map['X'][:plot_limit_m], label="X", marker='.', lw=1)
            ax2_m.plot(df_map['Time'][:plot_limit_m], df_map['Y'][:plot_limit_m], label="Y", marker='.', lw=1)
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

            # --- Prediction Performance via pyEDM.Simplex ---
            st.markdown("---")
            st.subheader("Prediction Performance")

            col3_m, col4_m = st.columns(2)
            lib_range_m = [1, int(max_lib_size)]
            pred_range_m = [1, int(num_points)]
            
            with col3_m:
                st.markdown("### Does X cause Y? ")
                simplex_XY_m = pyEDM.Simplex(
                    dataFrame=df_map, lib=lib_range_m, pred=pred_range_m,
                    columns="Y", target="X", E=2, Tp=0, tau=-1
                )
                
                fig_xy_m, ax_xy_m = plt.subplots(figsize=(5, 5))
                ax_xy_m.scatter(simplex_XY_m['Observations'], simplex_XY_m['Predictions'], alpha=0.4, edgecolors='none', color='teal')
                ax_xy_m.plot([simplex_XY_m['Observations'].min(), simplex_XY_m['Observations'].max()],
                              [simplex_XY_m['Observations'].min(), simplex_XY_m['Observations'].max()], 'r--', lw=2)
                ax_xy_m.set_xlabel("Observed X")
                ax_xy_m.set_ylabel("Predicted X from M_Y")
                ax_xy_m.set_title("Cross-mapping Performance")
                st.pyplot(fig_xy_m)
                plt.close()

            with col4_m:
                st.markdown("### Does Y cause X?")
                simplex_YX_m = pyEDM.Simplex(
                    dataFrame=df_map, lib=lib_range_m, pred=pred_range_m,
                    columns="X", target="Y", E=2, Tp=0, tau=-1
                )
                
                fig_yx_m, ax_yx_m = plt.subplots(figsize=(5, 5))
                ax_yx_m.scatter(simplex_YX_m['Observations'], simplex_YX_m['Predictions'], alpha=0.4, edgecolors='none', color='orange')
                ax_yx_m.plot([simplex_YX_m['Observations'].min(), simplex_YX_m['Observations'].max()],
                              [simplex_YX_m['Observations'].min(), simplex_YX_m['Observations'].max()], 'r--', lw=2)
                ax_yx_m.set_xlabel("Observed Y")
                ax_yx_m.set_ylabel("Predicted Y from M_X")
                ax_yx_m.set_title("Cross-mapping Performance")
                st.pyplot(fig_yx_m)
                plt.close()