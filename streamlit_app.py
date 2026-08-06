import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pyEDM 
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mutual_info_score

# --- Page Configuration ---
st.set_page_config(page_title="CCM", layout="wide")

# --- Information Theory Functions ---
@st.cache_data
def calc_mutual_information(x, y, bins=20):
    """Calculates Mutual Information between two continuous variables using binning."""
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)

@st.cache_data
def calc_transfer_entropy(x, y, lag=1, bins=10):
    """
    Calculates Transfer Entropy from x to y: TE(X -> Y).
    Measures how much knowing the past of X reduces uncertainty about the future of Y,
    beyond what is already known from the past of Y.
    """
    y_t = y[lag:]
    y_past = y[:-lag]
    x_past = x[:-lag]
    
    # 3D histogram for the joint distribution (y_t, y_past, x_past)
    c_3d, _ = np.histogramdd(np.vstack([y_t, y_past, x_past]).T, bins=bins)
    p_3d = c_3d / np.sum(c_3d)
    
    # Calculate marginal distributions
    p_y_past_x_past = np.sum(p_3d, axis=0)
    p_y_t_y_past = np.sum(p_3d, axis=2)
    p_y_past = np.sum(p_y_past_x_past, axis=1)
    
    # Mask zeros to avoid log(0) errors
    p_3d_safe = np.where(p_3d > 0, p_3d, 1e-10)
    p_y_past_x_past_safe = np.where(p_y_past_x_past > 0, p_y_past_x_past, 1e-10)
    p_y_t_y_past_safe = np.where(p_y_t_y_past > 0, p_y_t_y_past, 1e-10)
    p_y_past_safe = np.where(p_y_past > 0, p_y_past, 1e-10)
    
    # Reshape for broadcasting
    term1 = p_3d_safe * p_y_past_safe.reshape(1, bins, 1)
    term2 = p_y_t_y_past_safe.reshape(bins, bins, 1) * p_y_past_x_past_safe.reshape(1, bins, bins)
    
    # Sum over the joint probability distribution
    te = np.sum(p_3d * np.log2(term1 / term2))
    # Ensure non-negative due to potential floating point errors
    return max(0.0, te)

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
    sigma = st.sidebar.slider("Sigma (σ)", 1.0, 20.0, 10.0)
    rho = st.sidebar.slider("Rho (ρ)", 10.0, 40.0, 28.0)
    beta = st.sidebar.slider("Beta (β)", 1.0, 5.0, 2.666)
    t_max = st.sidebar.number_input("Max Time (t)", 10.0, 100.0, 40.0)

with st.sidebar.expander("Logistic Map Parameters (Tab 2)", expanded=True):
    rx = st.sidebar.slider("Growth Rate rx", 3.5, 4.0, 3.8, step=0.01)
    ry = st.sidebar.slider("Growth Rate ry", 3.5, 4.0, 3.5, step=0.01)
    beta_xy = st.sidebar.slider("Effect of Y on X (β_xy)", 0.0, 0.5, 0.02, step=0.01)
    beta_yx = st.sidebar.slider("Effect of X on Y (β_yx)", 0.0, 0.5, 0.10, step=0.01)

st.sidebar.header("Global Settings")

num_points = st.sidebar.slider("Number of Data Points (Length)", 500, 5000, 2000, step=100)

absolute_max_lib = num_points - 50
default_lib_value = min(500, absolute_max_lib)

max_lib_size = st.sidebar.slider("Max Library Size", 100, absolute_max_lib, default_lib_value, step=50)
lib_step = st.sidebar.number_input("Library Step Size", 10, 200, 20)
max_lag = st.sidebar.slider("Max Lag for Analysis", min_value=1, max_value=30, value=10, step=1)

# ADDED: Bin Range for Graphs
bin_min, bin_max = st.sidebar.slider(
    "Bin Range for Info Theory Graphs", 
    min_value=5, 
    max_value=250, 
    value=(5, 50), 
    step=5
)
bin_steps = np.arange(bin_min, bin_max + 1, 5)

# --- Main Application ---
st.title("Convergent Cross Mapping (CCM) & Information Theory")

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

    window_start, window_end = st.slider(
        "Select Time Window for Analysis",
        min_value=1,
        max_value=num_points,
        value=(1, num_points),
        step=10
    )
    
    df_lorenz_window = df_lorenz[(df_lorenz['Time'] >= window_start) & (df_lorenz['Time'] <= window_end)].copy()
    df_lorenz_window.reset_index(drop=True, inplace=True)
    window_len = len(df_lorenz_window)

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

    st.subheader("Statistical & Information Theory Metrics (Selected Window)")

    corr_matrix = df_lorenz_window[['X', 'Y', 'Z']].corr()
    st.metric(f"Pearson Correlation ({v1}, {v2})", f"{corr_matrix.loc[v1, v2]:.3f}")
    
    # Calculate Info Theory metrics over bins
    v1_data = df_lorenz_window[v1].values
    v2_data = df_lorenz_window[v2].values
    
    mi_vals = []
    te_12_vals = []
    te_21_vals = []
    
    with st.spinner(f"Calculating Information Theory metrics over {len(bin_steps)} bin settings..."):
        for b in bin_steps:
            mi_vals.append(calc_mutual_information(v1_data, v2_data, bins=b))
            te_12_vals.append(calc_transfer_entropy(v1_data, v2_data, lag=1, bins=b))
            te_21_vals.append(calc_transfer_entropy(v2_data, v1_data, lag=1, bins=b))

    fig_info, ax_info = plt.subplots(figsize=(10, 4))
    ax_info.plot(bin_steps, mi_vals, marker='o', label='Mutual Info (MI)', color='green')
    ax_info.plot(bin_steps, te_12_vals, marker='s', label=f'TE: {v1} -> {v2}', color='orange')
    ax_info.plot(bin_steps, te_21_vals, marker='^', label=f'TE: {v2} -> {v1}', color='purple')
    
    ax_info.set_xlabel("Number of Bins")
    ax_info.set_ylabel("Information (Bits)")
    ax_info.set_title(f"Information Theory Metrics vs. Histogram Bins ({v1} and {v2})")
    ax_info.legend()
    ax_info.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_info)
    plt.close()

    st.markdown("---")
    st.subheader(f"CCM Analysis: {v1} and {v2} Coupling")

    if st.button(f"Run Lorenz Analysis ({v1} & {v2})"):
        if window_len < 100:
            st.error("The selected time window is too small for meaningful analysis. Please select a wider range.")
        else:
            with st.spinner(f"Running Analysis on {v1} and {v2} for t={window_start} to t={window_end}..."):
                
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

                st.markdown("---")
                st.subheader("Granger Causality Comparison")
                
                try:
                    st.markdown("---")
                    st.subheader(f"Univariate and Bivariate Autoregressive Models (Predicting {v2})")

                    gc_12 = grangercausalitytests(df_lorenz_window[[v2, v1]], maxlag=max_lag)
                    p_values_12 = [gc_12[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
                    
                    gc_21 = grangercausalitytests(df_lorenz_window[[v1, v2]], maxlag=max_lag)
                    p_values_21 = [gc_21[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
                    
                    models_tuple = gc_12[max_lag][1]
                    restricted_model = models_tuple[0]    # Univariate AR (Past Y only)
                    unrestricted_model = models_tuple[1]  # Bivariate AR (Past Y + Past X)

                    actual_target = unrestricted_model.model.endog
                    pred_restricted = restricted_model.fittedvalues
                    pred_unrestricted = unrestricted_model.fittedvalues

                    time_axis = df_lorenz_window['Time'].iloc[max_lag:].values

                    fig_fit, ax_fit = plt.subplots(figsize=(10, 4))

                    ax_fit.plot(time_axis, actual_target, label=f"Actual {v2}", color='black', lw=1.5, alpha=0.6)
                    ax_fit.plot(time_axis, pred_restricted, label=f"Univariate AR (Uses past {v2} only)", color='red', linestyle='dashed', alpha=0.7)
                    ax_fit.plot(time_axis, pred_unrestricted, label=f"Bivariate AR (Uses past {v2} & {v1})", color='dodgerblue', linestyle='dotted', lw=2)

                    ax_fit.set_xlabel("Time")
                    ax_fit.set_ylabel(v2)
                    ax_fit.set_title(f"Granger Linear Fits at Lag {max_lag} (Full Time Series)")
                    ax_fit.legend()
                    ax_fit.grid(True, linestyle='--', alpha=0.5)

                    st.pyplot(fig_fit)
                    plt.close()

                    fig_gc, ax_gc = plt.subplots(figsize=(8, 4))
                    lags = np.arange(1, max_lag + 1)
                    ax_gc.plot(lags, p_values_12, marker='o', color='C1', label=f'{v1} Granger-causes {v2}')
                    ax_gc.plot(lags, p_values_21, marker='s', color='C0', label=f'{v2} Granger-causes {v1}')
                    ax_gc.axhline(y=0.05, color='r', linestyle='--', label='α = 0.05 Significance Threshold')
                    ax_gc.set_xlabel("Lag Step")
                    ax_gc.set_ylabel("p-value")
                    ax_gc.set_title(f"Granger Causality Significance vs. Lags (Up to Lag {max_lag})")
                    ax_gc.set_ylim([-0.05, 1.05])
                    ax_gc.legend()
                    ax_gc.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig_gc)
                    plt.close()

                    
                except Exception as e:
                    st.error(f"Could not compute Granger Causality (likely due to data alignment/stationarity limitations): {e}")


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

    st.subheader("Information Theory Metrics")
    
    mi_vals_map = []
    te_x_y_vals = []
    te_y_x_vals = []
    
    x_data_map = df_map['X'].values
    y_data_map = df_map['Y'].values

    with st.spinner(f"Calculating Information Theory metrics over {len(bin_steps)} bin settings..."):
        for b in bin_steps:
            mi_vals_map.append(calc_mutual_information(x_data_map, y_data_map, bins=b))
            te_x_y_vals.append(calc_transfer_entropy(x_data_map, y_data_map, lag=1, bins=b))
            te_y_x_vals.append(calc_transfer_entropy(y_data_map, x_data_map, lag=1, bins=b))

    fig_info_map, ax_info_map = plt.subplots(figsize=(10, 4))
    ax_info_map.plot(bin_steps, mi_vals_map, marker='o', label='Mutual Info (MI)', color='green')
    ax_info_map.plot(bin_steps, te_x_y_vals, marker='s', label='TE: X -> Y', color='orange')
    ax_info_map.plot(bin_steps, te_y_x_vals, marker='^', label='TE: Y -> X', color='purple')
    
    ax_info_map.set_xlabel("Number of Bins")
    ax_info_map.set_ylabel("Information (Bits)")
    ax_info_map.set_title("Information Theory Metrics vs. Histogram Bins (Logistic Map)")
    ax_info_map.legend()
    ax_info_map.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_info_map)
    plt.close()

    st.markdown("---")
    st.subheader("CCM Analysis: Asymmetric Causality")

    if st.button("Run Logistic Map Analysis"):
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