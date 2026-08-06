import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pyEDM 
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mutual_info_score

# --- Page Configuration ---
st.set_page_config(page_title="CCM", layout="wide") #[cite: 1]

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
def generate_lorenz_data(sigma, rho, beta, t_max, num_points): #[cite: 1]
    """Generates time series data for the continuous Lorenz attractor.""" #[cite: 1]
    def lorenz_system(t, state): #[cite: 1]
        x, y, z = state #[cite: 1]
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z] #[cite: 1]

    initial_state = [1.0, 1.0, 1.0] #[cite: 1]
    t_eval = np.linspace(0, t_max, num_points) #[cite: 1]
    solution = solve_ivp(lorenz_system, [0, t_max], initial_state, t_eval=t_eval) #[cite: 1]
    
    df = pd.DataFrame({ #[cite: 1]
        'Time': np.arange(1, num_points + 1), #[cite: 1]
        'X': solution.y[0], #[cite: 1]
        'Y': solution.y[1], #[cite: 1]
        'Z': solution.y[2] #[cite: 1]
    }) #[cite: 1]
    return df #[cite: 1]

@st.cache_data
def generate_logistic_data(rx, ry, beta_xy, beta_yx, num_points): #[cite: 1]
    """Generates discrete time series data for the coupled logistic map.""" #[cite: 1]
    x = np.zeros(num_points) #[cite: 1]
    y = np.zeros(num_points) #[cite: 1]
    
    x[0] = 0.4 #[cite: 1]
    y[0] = 0.2 #[cite: 1]
    
    for t in range(num_points - 1): #[cite: 1]
        x[t+1] = x[t] * (rx - rx * x[t] - beta_xy * y[t]) #[cite: 1]
        y[t+1] = y[t] * (ry - ry * y[t] - beta_yx * x[t]) #[cite: 1]
        
        # Cap to prevent math explosion if sliders pushed too far
        x[t+1] = max(0, min(1, x[t+1])) #[cite: 1]
        y[t+1] = max(0, min(1, y[t+1])) #[cite: 1]
        
    df = pd.DataFrame({ #[cite: 1]
        'Time': np.arange(1, num_points + 1), #[cite: 1]
        'X': x, #[cite: 1]
        'Y': y #[cite: 1]
    }) #[cite: 1]
    return df #[cite: 1]

# --- Sidebar UI ---
st.sidebar.title("Configuration") #[cite: 1]

with st.sidebar.expander("Lorenz Parameters (Tab 1)", expanded=True): #[cite: 1]
    sigma = st.slider("Sigma (σ)", 1.0, 20.0, 10.0) #[cite: 1]
    rho = st.slider("Rho (ρ)", 10.0, 40.0, 28.0) #[cite: 1]
    beta = st.slider("Beta (β)", 1.0, 5.0, 2.666) #[cite: 1]
    t_max = st.number_input("Max Time (t)", 10.0, 100.0, 40.0) #[cite: 1]

with st.sidebar.expander("Logistic Map Parameters (Tab 2)", expanded=True): #[cite: 1]
    rx = st.slider("Growth Rate rx", 3.5, 4.0, 3.8, step=0.01) #[cite: 1]
    ry = st.slider("Growth Rate ry", 3.5, 4.0, 3.5, step=0.01) #[cite: 1]
    beta_xy = st.slider("Effect of Y on X (β_xy)", 0.0, 0.5, 0.02, step=0.01) #[cite: 1]
    beta_yx = st.slider("Effect of X on Y (β_yx)", 0.0, 0.5, 0.10, step=0.01) #[cite: 1]

st.sidebar.header("Global Settings") #[cite: 1]

num_points = st.sidebar.slider("Number of Data Points (Length)", 500, 5000, 2000, step=100) #[cite: 1]

absolute_max_lib = num_points - 50 #[cite: 1]
default_lib_value = min(500, absolute_max_lib) #[cite: 1]

max_lib_size = st.sidebar.slider("Max Library Size", 100, absolute_max_lib, default_lib_value, step=50) #[cite: 1]
lib_step = st.sidebar.number_input("Library Step Size", 10, 200, 20) #[cite: 1]
max_lag = st.sidebar.slider("Max Lag for Analysis", min_value=1, max_value=30, value=10, step=1) #[cite: 1]

info_bins = st.sidebar.slider("Bins for Information Theory", min_value=5, max_value=50, value=20, step=5)


# --- Main Application ---
st.title("Convergent Cross Mapping (CCM) & Information Theory") #[cite: 1]

tab1, tab2 = st.tabs(["Lorenz Attractor", "Coupled Logistic Map"]) #[cite: 1]

# ==========================================
# TAB 1: LORENZ ATTRACTOR
# ==========================================
with tab1:
    st.header("Lorenz Attractor") #[cite: 1]
    st.markdown(r"$$ \frac{dx}{dt} = \sigma(y-x)$$") #[cite: 1]
    st.markdown(r"$$\frac{dy}{dt} = x(\rho-z)-y $$") #[cite: 1]
    st.markdown(r"$$\frac{dz}{dt} = xy-\beta z $$") #[cite: 1]
    df_lorenz = generate_lorenz_data(sigma, rho, beta, t_max, num_points) #[cite: 1]

    st.markdown("---") #[cite: 1]
    st.subheader("Data Selection & Visualization") #[cite: 1]
    var_pair = st.radio("Select Variable Pair to Analyze:", ["X and Y", "X and Z", "Y and Z"], horizontal=True) #[cite: 1]
    v1, v2 = var_pair.split(" and ") #[cite: 1]

    # ADDED: Sliding Window
    window_start, window_end = st.slider( #[cite: 1]
        "Select Time Window for Analysis", #[cite: 1]
        min_value=1, #[cite: 1]
        max_value=num_points, #[cite: 1]
        value=(1, num_points), #[cite: 1]
        step=10 #[cite: 1]
    )
    
    # Slice the dataframe based on the window
    df_lorenz_window = df_lorenz[(df_lorenz['Time'] >= window_start) & (df_lorenz['Time'] <= window_end)].copy() #[cite: 1]
    df_lorenz_window.reset_index(drop=True, inplace=True) #[cite: 1]
    window_len = len(df_lorenz_window) #[cite: 1]

    col1, col2 = st.columns(2) #[cite: 1]
    with col1: #[cite: 1]
        st.subheader(f"Phase Space ({v1} vs {v2})") #[cite: 1]
        fig1, ax1 = plt.subplots(figsize=(6, 4)) #[cite: 1]
        ax1.plot(df_lorenz_window[v1], df_lorenz_window[v2], lw=0.8, color='royalblue') #[cite: 1]
        ax1.set_xlabel(v1) #[cite: 1]
        ax1.set_ylabel(v2) #[cite: 1]
        ax1.set_title("Phase Space (Current Window)") #[cite: 1]
        st.pyplot(fig1) #[cite: 1]

    with col2: #[cite: 1]
        st.subheader(f"Time Series ({v1} and {v2})") #[cite: 1]
        fig2, ax2 = plt.subplots(figsize=(6, 4)) #[cite: 1]
        ax2.plot(df_lorenz['Time'], df_lorenz[v1], label=v1, lw=1, alpha=0.8) #[cite: 1]
        ax2.plot(df_lorenz['Time'], df_lorenz[v2], label=v2, lw=1, alpha=0.8) #[cite: 1]
        ax2.axvspan(window_start, window_end, color='yellow', alpha=0.2, label='Selected Window') #[cite: 1]
        ax2.set_xlabel("Time (Index)") #[cite: 1]
        ax2.set_ylabel("Value") #[cite: 1]
        ax2.legend() #[cite: 1]
        st.pyplot(fig2) #[cite: 1]

    
    st.subheader("Statistical & Information Theory Metrics (Selected Window)")

    # Core Stats
    corr_matrix = df_lorenz_window[['X', 'Y', 'Z']].corr() #[cite: 1]
    corr_xy = corr_matrix.loc['X', 'Y'] #[cite: 1]
    corr_yz = corr_matrix.loc['Y', 'Z'] #[cite: 1]
    corr_xz = corr_matrix.loc['X', 'Z'] #[cite: 1]
    
    # Calculate Info Theory metrics
    v1_data = df_lorenz_window[v1].values
    v2_data = df_lorenz_window[v2].values
    
    mi_val = calc_mutual_information(v1_data, v2_data, bins=info_bins)
    te_1_to_2 = calc_transfer_entropy(v1_data, v2_data, lag=1, bins=info_bins)
    te_2_to_1 = calc_transfer_entropy(v2_data, v1_data, lag=1, bins=info_bins)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric(f"Corr({v1}, {v2})", f"{corr_matrix.loc[v1, v2]:.3f}")
    mcol2.metric(f"Mutual Info (MI)", f"{mi_val:.3f} bits")
    mcol3.metric(f"TE: {v1} -> {v2}", f"{te_1_to_2:.3f} bits")
    mcol4.metric(f"TE: {v2} -> {v1}", f"{te_2_to_1:.3f} bits")

    st.markdown("---") #[cite: 1]
    st.subheader(f"CCM Analysis: {v1} and {v2} Coupling") #[cite: 1]

    if st.button(f"Run Lorenz Analysis ({v1} & {v2})"):
        if window_len < 100: #[cite: 1]
            st.error("The selected time window is too small for meaningful analysis. Please select a wider range.") #[cite: 1]
        else: #[cite: 1]
            with st.spinner(f"Running Analysis on {v1} and {v2} for t={window_start} to t={window_end}..."): #[cite: 1]
                
                # Dynamically adjust max library size so pyEDM doesn't crash on small windows
                dynamic_max_lib = max(10, window_len - 50) #[cite: 1]
                actual_max_lib = min(max_lib_size, dynamic_max_lib) #[cite: 1]
                lib_sizes_str = f"10 {actual_max_lib} {lib_step}" #[cite: 1]
                
                ccm_result = pyEDM.CCM( #[cite: 1]
                    dataFrame=df_lorenz_window, E=3, columns=v1, target=v2, #[cite: 1]
                    libSizes=lib_sizes_str, sample=100, showPlot=False #[cite: 1]
                ) #[cite: 1]
                
                fig3, ax3 = plt.subplots(figsize=(8, 5)) #[cite: 1]
                col_v1_v2 = f"{v1}:{v2}" #[cite: 1]
                col_v2_v1 = f"{v2}:{v1}" #[cite: 1]
                
                if col_v1_v2 in ccm_result.columns and col_v2_v1 in ccm_result.columns: #[cite: 1]
                    ax3.plot(ccm_result['LibSize'], ccm_result[col_v1_v2], marker='o', label=f'{v1} cross-maps {v2} ({v2} causes {v1})') #[cite: 1]
                    ax3.plot(ccm_result['LibSize'], ccm_result[col_v2_v1], marker='s', label=f'{v2} cross-maps {v1} ({v1} causes {v2})') #[cite: 1]
                
                ax3.set_xlabel("Library Size (L)") #[cite: 1]
                ax3.set_ylabel("Correlation (ρ)") #[cite: 1]
                ax3.set_title(f"CCM Convergence ({v1} vs {v2})\nWindow: t={window_start} to t={window_end}") #[cite: 1]
                ax3.legend() #[cite: 1]
                ax3.grid(True, linestyle='--', alpha=0.7) #[cite: 1]
                ax3.set_ylim([-0.1, 1.1]) #[cite: 1]
                st.pyplot(fig3) #[cite: 1]

                # --- Prediction Performance via pyEDM.Simplex ---
                st.markdown("---") #[cite: 1]
                st.subheader(f"Prediction Performance ({v1} vs {v2})") #[cite: 1]

                col3, col4 = st.columns(2) #[cite: 1]
                lib_range = [1, int(actual_max_lib)] #[cite: 1]
                pred_range = [1, int(window_len)] #[cite: 1]
                
                with col3: #[cite: 1]
                    st.markdown(f"### Does {v1} cause {v2}?") #[cite: 1]
                    simplex_12 = pyEDM.Simplex( #[cite: 1]
                        dataFrame=df_lorenz_window, lib=lib_range, pred=pred_range, #[cite: 1]
                        columns=v2, target=v1, E=3, Tp=0, tau=-1 #[cite: 1]
                    ) #[cite: 1]
                    
                    corr_12 = simplex_12['Observations'].corr(simplex_12['Predictions']) #[cite: 1]
                    
                    fig_12, ax_12 = plt.subplots(figsize=(5, 5)) #[cite: 1]
                    ax_12.scatter(simplex_12['Observations'], simplex_12['Predictions'], alpha=0.4, edgecolors='none', color='C1') #[cite: 1]
                    ax_12.plot([simplex_12['Observations'].min(), simplex_12['Observations'].max()], #[cite: 1]
                               [simplex_12['Observations'].min(), simplex_12['Observations'].max()], 'r--', lw=2) #[cite: 1]
                    ax_12.set_xlabel(f"Observed {v1}") #[cite: 1]
                    ax_12.set_ylabel(f"Predicted {v1} from M_{v2}") #[cite: 1]
                    ax_12.set_title(f"Cross-mapping Performance\nρ = {corr_12:.3f}") #[cite: 1]
                    st.pyplot(fig_12) #[cite: 1]
                    plt.close() #[cite: 1]

                with col4: #[cite: 1]
                    st.markdown(f"### Does {v2} cause {v1}?") #[cite: 1]
                    simplex_21 = pyEDM.Simplex( #[cite: 1]
                        dataFrame=df_lorenz_window, lib=lib_range, pred=pred_range, #[cite: 1]
                        columns=v1, target=v2, E=3, Tp=0, tau=-1 #[cite: 1]
                    ) #[cite: 1]
                    
                    corr_21 = simplex_21['Observations'].corr(simplex_21['Predictions']) #[cite: 1]
                    
                    fig_21, ax_21 = plt.subplots(figsize=(5, 5)) #[cite: 1]
                    ax_21.scatter(simplex_21['Observations'], simplex_21['Predictions'], alpha=0.4, edgecolors='none', color='C0') #[cite: 1]
                    ax_21.plot([simplex_21['Observations'].min(), simplex_21['Observations'].max()], #[cite: 1]
                               [simplex_21['Observations'].min(), simplex_21['Observations'].max()], 'r--', lw=2) #[cite: 1]
                    ax_21.set_xlabel(f"Observed {v2}") #[cite: 1]
                    ax_21.set_ylabel(f"Predicted {v2} from M_{v1}") #[cite: 1]
                    ax_21.set_title(f"Cross-mapping Performance\nρ = {corr_21:.3f}") #[cite: 1]
                    st.pyplot(fig_21) #[cite: 1]
                    plt.close() #[cite: 1]

                # --- ADDED: Granger Causality Comparison Section ---
                st.markdown("---") #[cite: 1]
                st.subheader("Granger Causality Comparison") #[cite: 1]
                
                try: #[cite: 1]
                    # --- ADDED: Visualize the Underlying Granger Models ---
                    st.markdown("---") #[cite: 1]
                    st.subheader(f"Univariate and Bivariate Autoregressive Models (Predicting {v2})") #[cite: 1]

                    gc_12 = grangercausalitytests(df_lorenz_window[[v2, v1]], maxlag=max_lag, verbose=False) #[cite: 1]
                    p_values_12 = [gc_12[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)] #[cite: 1]
                    
                    gc_21 = grangercausalitytests(df_lorenz_window[[v1, v2]], maxlag=max_lag, verbose=False) #[cite: 1]
                    p_values_21 = [gc_21[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)] #[cite: 1]
                    
                    models_tuple = gc_12[max_lag][1] #[cite: 1]
                    restricted_model = models_tuple[0]    # Univariate AR (Past Y only) #[cite: 1]
                    unrestricted_model = models_tuple[1]  # Bivariate AR (Past Y + Past X) #[cite: 1]

                    # The actual target values used in the regression
                    actual_target = unrestricted_model.model.endog #[cite: 1]
                    pred_restricted = restricted_model.fittedvalues #[cite: 1]
                    pred_unrestricted = unrestricted_model.fittedvalues #[cite: 1]

                    # Time axis shifted by the lag
                    time_axis = df_lorenz_window['Time'].iloc[max_lag:].values #[cite: 1]

                    fig_fit, ax_fit = plt.subplots(figsize=(10, 4)) #[cite: 1]

                    # Plotting all available points
                    ax_fit.plot(time_axis, actual_target, label=f"Actual {v2}", color='black', lw=1.5, alpha=0.6) #[cite: 1]
                    ax_fit.plot(time_axis, pred_restricted, label=f"Univariate AR (Uses past {v2} only)", color='red', linestyle='dashed', alpha=0.7) #[cite: 1]
                    ax_fit.plot(time_axis, pred_unrestricted, label=f"Bivariate AR (Uses past {v2} & {v1})", color='dodgerblue', linestyle='dotted', lw=2) #[cite: 1]

                    ax_fit.set_xlabel("Time") #[cite: 1]
                    ax_fit.set_ylabel(v2) #[cite: 1]
                    ax_fit.set_title(f"Granger Linear Fits at Lag {max_lag} (Full Time Series)") #[cite: 1]
                    ax_fit.legend() #[cite: 1]
                    ax_fit.grid(True, linestyle='--', alpha=0.5) #[cite: 1]

                    st.pyplot(fig_fit) #[cite: 1]
                    plt.close() #[cite: 1]

                    fig_gc, ax_gc = plt.subplots(figsize=(8, 4)) #[cite: 1]
                    lags = np.arange(1, max_lag + 1) #[cite: 1]
                    ax_gc.plot(lags, p_values_12, marker='o', color='C1', label=f'{v1} Granger-causes {v2}') #[cite: 1]
                    ax_gc.plot(lags, p_values_21, marker='s', color='C0', label=f'{v2} Granger-causes {v1}') #[cite: 1]
                    ax_gc.axhline(y=0.05, color='r', linestyle='--', label='α = 0.05 Significance Threshold') #[cite: 1]
                    ax_gc.set_xlabel("Lag Step") #[cite: 1]
                    ax_gc.set_ylabel("p-value") #[cite: 1]
                    ax_gc.set_title(f"Granger Causality Significance vs. Lags (Up to Lag {max_lag})") #[cite: 1]
                    ax_gc.set_ylim([-0.05, 1.05]) #[cite: 1]
                    ax_gc.legend() #[cite: 1]
                    ax_gc.grid(True, linestyle='--', alpha=0.5) #[cite: 1]
                    st.pyplot(fig_gc) #[cite: 1]
                    plt.close() #[cite: 1]

                    
                except Exception as e: #[cite: 1]
                    st.error(f"Could not compute Granger Causality (likely due to data alignment/stationarity limitations): {e}") #[cite: 1]


# ==========================================
# TAB 2: COUPLED LOGISTIC MAP
# ==========================================
with tab2:
    st.header("Coupled Logistic Map") #[cite: 1]
    st.markdown(r"$$ X_{t+1} = X_t [r_x - r_x X_t - \beta_{xy} Y_t]$$") #[cite: 1]
    st.markdown(r"$$Y_{t+1} = Y_t [r_y - r_y Y_t - \beta_{yx} X_t] $$") #[cite: 1]
    df_map = generate_logistic_data(rx, ry, beta_xy, beta_yx, num_points) #[cite: 1]

    col1_m, col2_m = st.columns(2) #[cite: 1]
    with col1_m: #[cite: 1]
        st.subheader("Phase Space (X vs Y)") #[cite: 1]
        fig1_m, ax1_m = plt.subplots(figsize=(6, 4)) #[cite: 1]
        ax1_m.scatter(df_map['X'], df_map['Y'], s=2, alpha=0.5, color='purple') #[cite: 1]
        ax1_m.set_xlabel("X") #[cite: 1]
        ax1_m.set_ylabel("Y") #[cite: 1]
        st.pyplot(fig1_m) #[cite: 1]

    with col2_m: #[cite: 1]
        st.subheader("Time Series") #[cite: 1]
        fig2_m, ax2_m = plt.subplots(figsize=(6, 4)) #[cite: 1]
        ax2_m.plot(df_map['Time'], df_map['X'], label="X", marker='.', lw=1) #[cite: 1]
        ax2_m.plot(df_map['Time'], df_map['Y'], label="Y", marker='.', lw=1) #[cite: 1]
        ax2_m.set_xlabel("Time (Step)") #[cite: 1]
        ax2_m.set_ylabel("Value") #[cite: 1]
        ax2_m.legend() #[cite: 1]
        st.pyplot(fig2_m) #[cite: 1]

    st.subheader("Information Theory Metrics")
    mi_val_map = calc_mutual_information(df_map['X'].values, df_map['Y'].values, bins=info_bins)
    te_x_to_y = calc_transfer_entropy(df_map['X'].values, df_map['Y'].values, lag=1, bins=info_bins)
    te_y_to_x = calc_transfer_entropy(df_map['Y'].values, df_map['X'].values, lag=1, bins=info_bins)
    
    mcol1_map, mcol2_map, mcol3_map = st.columns(3)
    mcol1_map.metric("Mutual Info (MI)", f"{mi_val_map:.3f} bits")
    mcol2_map.metric("TE: X -> Y", f"{te_x_to_y:.3f} bits")
    mcol3_map.metric("TE: Y -> X", f"{te_y_to_x:.3f} bits")

    st.markdown("---") #[cite: 1]
    st.subheader("CCM Analysis: Asymmetric Causality") #[cite: 1]

    if st.button("Run Logistic Map Analysis"):
        with st.spinner("Running Convergent Cross Mapping on Logistic Map..."): #[cite: 1]
            lib_sizes_str = f"10 {max_lib_size} {lib_step}" #[cite: 1]
            
            ccm_result_m = pyEDM.CCM( #[cite: 1]
                dataFrame=df_map, E=2, columns="X", target="Y", #[cite: 1]
                libSizes=lib_sizes_str, sample=100, showPlot=False #[cite: 1]
            ) #[cite: 1]
            
            fig3_m, ax3_m = plt.subplots(figsize=(8, 5)) #[cite: 1]
            if 'X:Y' in ccm_result_m.columns and 'Y:X' in ccm_result_m.columns: #[cite: 1]
                ax3_m.plot(ccm_result_m['LibSize'], ccm_result_m['X:Y'], marker='o', color='orange', label='X cross-maps Y (Y causes X)') #[cite: 1]
                ax3_m.plot(ccm_result_m['LibSize'], ccm_result_m['Y:X'], marker='s', color='teal', label='Y cross-maps X (X causes Y)') #[cite: 1]
            
            ax3_m.set_xlabel("Library Size (L)") #[cite: 1]
            ax3_m.set_ylabel("Correlation (ρ)") #[cite: 1]
            ax3_m.set_title("CCM Convergence") #[cite: 1]
            ax3_m.legend() #[cite: 1]
            ax3_m.grid(True, linestyle='--', alpha=0.7) #[cite: 1]
            ax3_m.set_ylim([-0.1, 1.1]) #[cite: 1]
            st.pyplot(fig3_m) #[cite: 1]

            # --- Prediction Performance via pyEDM.Simplex ---
            st.markdown("---") #[cite: 1]
            st.subheader("Prediction Performance") #[cite: 1]

            col3_m, col4_m = st.columns(2) #[cite: 1]
            lib_range_m = [1, int(max_lib_size)] #[cite: 1]
            pred_range_m = [1, int(num_points)] #[cite: 1]
            
            with col3_m: #[cite: 1]
                st.markdown("### Does X cause Y?") #[cite: 1]
                simplex_XY_m = pyEDM.Simplex( #[cite: 1]
                    dataFrame=df_map, lib=lib_range_m, pred=pred_range_m, #[cite: 1]
                    columns="Y", target="X", E=2, Tp=0, tau=-1 #[cite: 1]
                ) #[cite: 1]
                
                corr_xy_m = simplex_XY_m['Observations'].corr(simplex_XY_m['Predictions']) #[cite: 1]

                fig_xy_m, ax_xy_m = plt.subplots(figsize=(5, 5)) #[cite: 1]
                ax_xy_m.scatter(simplex_XY_m['Observations'], simplex_XY_m['Predictions'], alpha=0.4, edgecolors='none', color='orange') #[cite: 1]
                ax_xy_m.plot([simplex_XY_m['Observations'].min(), simplex_XY_m['Observations'].max()], #[cite: 1]
                              [simplex_XY_m['Observations'].min(), simplex_XY_m['Observations'].max()], 'r--', lw=2) #[cite: 1]
                ax_xy_m.set_xlabel("Observed X") #[cite: 1]
                ax_xy_m.set_ylabel("Predicted X from M_Y") #[cite: 1]
                ax_xy_m.set_title(f"Cross-mapping Performance\nρ = {corr_xy_m:.3f}") #[cite: 1]
                st.pyplot(fig_xy_m) #[cite: 1]
                plt.close() #[cite: 1]

            with col4_m: #[cite: 1]
                st.markdown("### Does Y cause X?") #[cite: 1]
                simplex_YX_m = pyEDM.Simplex( #[cite: 1]
                    dataFrame=df_map, lib=lib_range_m, pred=pred_range_m, #[cite: 1]
                    columns="X", target="Y", E=2, Tp=0, tau=-1 #[cite: 1]
                ) #[cite: 1]
                
                corr_yx_m = simplex_YX_m['Observations'].corr(simplex_YX_m['Predictions']) #[cite: 1]

                fig_yx_m, ax_yx_m = plt.subplots(figsize=(5, 5)) #[cite: 1]
                ax_yx_m.scatter(simplex_YX_m['Observations'], simplex_YX_m['Predictions'], alpha=0.4, edgecolors='none', color='teal') #[cite: 1]
                ax_yx_m.plot([simplex_YX_m['Observations'].min(), simplex_YX_m['Observations'].max()], #[cite: 1]
                              [simplex_YX_m['Observations'].min(), simplex_YX_m['Observations'].max()], 'r--', lw=2) #[cite: 1]
                ax_yx_m.set_xlabel("Observed Y") #[cite: 1]
                ax_yx_m.set_ylabel("Predicted Y from M_X") #[cite: 1]
                ax_yx_m.set_title(f"Cross-mapping Performance\nρ = {corr_yx_m:.3f}") #[cite: 1]
                st.pyplot(fig_yx_m) #[cite: 1]
                plt.close() #[cite: 1]