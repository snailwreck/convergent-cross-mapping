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

num_points = st.sidebar.slider("Number of Data Points (Length)", 500, 5000, 2000, step=100)

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
    st.markdown(r"$$ \frac{dx}{dt} = \sigma(y-x)$$")
    st.markdown(r"$$\frac{dy}{dt} = x(\rho-z)-y $$")
    st.markdown(r"$$\frac{dz}{dt} = xy-\beta z $$")
    df_lorenz = generate_lorenz_data(sigma, rho, beta, t_max, num_points)

    st.markdown("---")
    st.subheader("Data Selection & Visualization")
    var_pair = st.radio("Select Variable Pair to Analyze:", ["X and Y", "X and Z", "Y and Z"], horizontal=True)
    v1, v2 = var_pair.split(" and ")

    # ADDED: Sliding Window
    window_start, window_end = st.slider(
        "Select Time Window for Analysis",
        min_value=1,
        max_value=num_points,
        value=(1, num_points),
        step=10
    )
    
    # Slice the dataframe based on the window
    df_lorenz_window = df_lorenz[(df_lorenz['Time'] >= window_start) & (df_lorenz['Time'] <= window_end)].copy()
    df_lorenz_window.reset_index(drop=True, inplace=True)
    window_len = len(df_lorenz_window)

    st.subheader("Variable Correlations (Selected Window)")

    corr_matrix = df_lorenz_window[['X', 'Y', 'Z']].corr()
    corr_xy = corr_matrix.loc['X', 'Y']
    corr_yz = corr_matrix.loc['Y', 'Z']
    corr_xz = corr_matrix.loc['X', 'Z']

    corr_col1, corr_col2, corr_col3 = st.columns(3)
    corr_col1.metric("Corr(X, Y)", f"{corr_xy:.3f}")
    corr_col2.metric("Corr(Y, Z)", f"{corr_yz:.3f}")
    corr_col3.metric("Corr(X, Z)", f"{corr_xz:.3f}")

    with st.expander("Full Correlation Matrix"):
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format("{:.3f}"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Phase Space ({v1} vs {v2})")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(df_lorenz_window[v1], df_lorenz_window[v2], lw=0.8, color='royalblue')
        ax1.set_xlabel(v1)
        ax1.set_ylabel(v2)
        ax1.set_title("Phase Space (Current Window)")
        st.pyplot(fig1)

    with col2:
        st.subheader(f"Time Series ({v1} and {v2})")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(df_lorenz['Time'], df_lorenz[v1], label=v1, lw=1, alpha=0.8)
        ax2.plot(df_lorenz['Time'], df_lorenz[v2], label=v2, lw=1, alpha=0.8)
        ax2.axvspan(window_start, window_end, color='yellow', alpha=0.2, label='Selected Window')
        ax2.set_xlabel("Time (Index)")
        ax2.set_ylabel("Value")
        ax2.legend()
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader(f"CCM Analysis: {v1} and {v2} Coupling")

    if st.button(f"Run Lorenz CCM Analysis ({v1} & {v2})"):
        if window_len < 100:
            st.error("The selected time window is too small for meaningful CCM analysis. Please select a wider range.")
        else:
            with st.spinner(f"Running Convergent Cross Mapping on {v1} and {v2} for t={window_start} to t={window_end}..."):
                
                # Dynamically adjust max library size so pyEDM doesn't crash on small windows
                dynamic_max_lib = max(10, window_len - 50)
                actual_max_lib = min(max_lib_size, dynamic_max_lib)
                lib_sizes_str = f"10 {actual_max_lib} {lib_step}"
                
                ccm_result = pyEDM.CCM(
                    dataFrame=df_lorenz_window, E=3, columns=v1, target=v2, 
                    libSizes=lib_sizes_str, sample=100, showPlot=False
                )
                
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                col_v1_v2 = f"{v1}:{v2}"
                col_v2_v1 = f"{v2}:{v1}"
                
                if col_v1_v2 in ccm_result.columns and col_v2_v1 in ccm_result.columns:
                    ax3.plot(ccm_result['LibSize'], ccm_result[col_v1_v2], marker='o', label=f'{v1} cross-maps {v2} ({v2} causes {v1})')
                    ax3.plot(ccm_result['LibSize'], ccm_result[col_v2_v1], marker='s', label=f'{v2} cross-maps {v1} ({v1} causes {v2})')
                
                ax3.set_xlabel("Library Size (L)")
                ax3.set_ylabel("Correlation (ρ)")
                ax3.set_title(f"CCM Convergence ({v1} vs {v2})\nWindow: t={window_start} to t={window_end}")
                ax3.legend()
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.set_ylim([-0.1, 1.1])
                st.pyplot(fig3)

                # --- Prediction Performance via pyEDM.Simplex ---
                st.markdown("---")
                st.subheader(f"Prediction Performance ({v1} vs {v2})")

                col3, col4 = st.columns(2)
                lib_range = [1, int(actual_max_lib)]
                pred_range = [1, int(window_len)]
                
                with col3:
                    st.markdown(f"### Does {v1} cause {v2}?")
                    simplex_12 = pyEDM.Simplex(
                        dataFrame=df_lorenz_window, lib=lib_range, pred=pred_range,
                        columns=v2, target=v1, E=3, Tp=0, tau=-1
                    )
                    
                    corr_12 = simplex_12['Observations'].corr(simplex_12['Predictions'])
                    
                    fig_12, ax_12 = plt.subplots(figsize=(5, 5))
                    ax_12.scatter(simplex_12['Observations'], simplex_12['Predictions'], alpha=0.4, edgecolors='none', color='C1')
                    ax_12.plot([simplex_12['Observations'].min(), simplex_12['Observations'].max()],
                                [simplex_12['Observations'].min(), simplex_12['Observations'].max()], 'r--', lw=2)
                    ax_12.set_xlabel(f"Observed {v1}")
                    ax_12.set_ylabel(f"Predicted {v1} from M_{v2}")
                    ax_12.set_title(f"Cross-mapping Performance\nρ = {corr_12:.3f}")
                    st.pyplot(fig_12)
                    plt.close()

                with col4:
                    st.markdown(f"### Does {v2} cause {v1}?")
                    simplex_21 = pyEDM.Simplex(
                        dataFrame=df_lorenz_window, lib=lib_range, pred=pred_range,
                        columns=v1, target=v2, E=3, Tp=0, tau=-1
                    )
                    
                    corr_21 = simplex_21['Observations'].corr(simplex_21['Predictions'])
                    
                    fig_21, ax_21 = plt.subplots(figsize=(5, 5))
                    ax_21.scatter(simplex_21['Observations'], simplex_21['Predictions'], alpha=0.4, edgecolors='none', color='C0')
                    ax_21.plot([simplex_21['Observations'].min(), simplex_21['Observations'].max()],
                                [simplex_21['Observations'].min(), simplex_21['Observations'].max()], 'r--', lw=2)
                    ax_21.set_xlabel(f"Observed {v2}")
                    ax_21.set_ylabel(f"Predicted {v2} from M_{v1}")
                    ax_21.set_title(f"Cross-mapping Performance\nρ = {corr_21:.3f}")
                    st.pyplot(fig_21)
                    plt.close()


# ==========================================
# TAB 2: COUPLED LOGISTIC MAP
# ==========================================
with tab2:
    st.header("Coupled Logistic Map")
    st.markdown(r"$$ X_{t+1} = X_t [r_x - r_x X_t - \beta_{xy} Y_t]$$") 
    st.markdown(r"$$Y_{t+1} = Y_t [r_y - r_y Y_t - \beta_{yx} X_t] $$")
    df_map = generate_logistic_data(rx, ry, beta_xy, beta_yx, num_points)

    col1_m, col2_m = st.columns(2)
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
        ax2_m.plot(df_map['Time'], df_map['X'], label="X", marker='.', lw=1)
        ax2_m.plot(df_map['Time'], df_map['Y'], label="Y", marker='.', lw=1)
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
                ax3_m.plot(ccm_result_m['LibSize'], ccm_result_m['X:Y'], marker='o', color='orange', label='X cross-maps Y (Y causes X)')
                ax3_m.plot(ccm_result_m['LibSize'], ccm_result_m['Y:X'], marker='s', color='teal', label='Y cross-maps X (X causes Y)')
            
            ax3_m.set_xlabel("Library Size (L)")
            ax3_m.set_ylabel("Correlation (ρ)")
            ax3_m.set_title("CCM Convergence")
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
                st.markdown("### Does X cause Y?")
                simplex_XY_m = pyEDM.Simplex(
                    dataFrame=df_map, lib=lib_range_m, pred=pred_range_m,
                    columns="Y", target="X", E=2, Tp=0, tau=-1
                )
                
                corr_xy_m = simplex_XY_m['Observations'].corr(simplex_XY_m['Predictions'])

                fig_xy_m, ax_xy_m = plt.subplots(figsize=(5, 5))
                ax_xy_m.scatter(simplex_XY_m['Observations'], simplex_XY_m['Predictions'], alpha=0.4, edgecolors='none', color='orange')
                ax_xy_m.plot([simplex_XY_m['Observations'].min(), simplex_XY_m['Observations'].max()],
                              [simplex_XY_m['Observations'].min(), simplex_XY_m['Observations'].max()], 'r--', lw=2)
                ax_xy_m.set_xlabel("Observed X")
                ax_xy_m.set_ylabel("Predicted X from M_Y")
                ax_xy_m.set_title(f"Cross-mapping Performance\nρ = {corr_xy_m:.3f}")
                st.pyplot(fig_xy_m)
                plt.close()

            with col4_m:
                st.markdown("### Does Y cause X?")
                simplex_YX_m = pyEDM.Simplex(
                    dataFrame=df_map, lib=lib_range_m, pred=pred_range_m,
                    columns="X", target="Y", E=2, Tp=0, tau=-1
                )
                
                corr_yx_m = simplex_YX_m['Observations'].corr(simplex_YX_m['Predictions'])

                fig_yx_m, ax_yx_m = plt.subplots(figsize=(5, 5))
                ax_yx_m.scatter(simplex_YX_m['Observations'], simplex_YX_m['Predictions'], alpha=0.4, edgecolors='none', color='teal')
                ax_yx_m.plot([simplex_YX_m['Observations'].min(), simplex_YX_m['Observations'].max()],
                              [simplex_YX_m['Observations'].min(), simplex_YX_m['Observations'].max()], 'r--', lw=2)
                ax_yx_m.set_xlabel("Observed Y")
                ax_yx_m.set_ylabel("Predicted Y from M_X")
                ax_yx_m.set_title(f"Cross-mapping Performance\nρ = {corr_yx_m:.3f}")
                st.pyplot(fig_yx_m)
                plt.close()