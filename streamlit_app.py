import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pyEDM 
from statsmodels.tsa.stattools import grangercausalitytests  

# --- Page Configuration ---
st.set_page_config(page_title="CCM & Rolling Causality", layout="wide")

# --- Helper Functions ---
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

st.sidebar.header("Global & Rolling Settings")

num_points = st.sidebar.slider("Number of Data Points (Length)", 500, 5000, 2000, step=100)
roll_window_size = st.sidebar.slider("Rolling Window Size (Points)", 100, 1000, 300, step=50)
roll_step_size = st.sidebar.slider("Rolling Evaluation Step Size", 5, 100, 20, step=5)

absolute_max_lib = num_points - 50
default_lib_value = min(500, absolute_max_lib)

max_lib_size = st.sidebar.slider("Max Library Size", 100, absolute_max_lib, default_lib_value, step=50)
lib_step = st.sidebar.number_input("Library Step Size", 10, 200, 20)
max_lag = st.sidebar.slider("Max Lag for Granger Causality", min_value=1, max_value=30, value=5, step=1)

# --- Main Application ---
st.title("Convergent Cross Mapping & Causality Analysis")

tab1, tab2 = st.tabs(["Lorenz Attractor", "Coupled Logistic Map"])

# ==========================================
# TAB 1: LORENZ ATTRACTOR
# ==========================================
with tab1:
    st.header("Lorenz Attractor Analysis")
    df_lorenz = generate_lorenz_data(sigma, rho, beta, t_max, num_points)

    # ---------------------------------------------------------
    # 1. Rolling Pairwise Correlation
    # ---------------------------------------------------------
    st.subheader("1. Rolling Pairwise Correlations")
    st.markdown(f"Computes Pearson correlation over a rolling window of **{roll_window_size}** points.")

    df_lorenz['Corr_XY'] = df_lorenz['X'].rolling(window=roll_window_size).corr(df_lorenz['Y'])
    df_lorenz['Corr_YZ'] = df_lorenz['Y'].rolling(window=roll_window_size).corr(df_lorenz['Z'])
    df_lorenz['Corr_XZ'] = df_lorenz['X'].rolling(window=roll_window_size).corr(df_lorenz['Z'])

    fig_corr, ax_corr = plt.subplots(figsize=(10, 4))
    ax_corr.plot(df_lorenz['Time'], df_lorenz['Corr_XY'], label='Corr(X, Y)', color='royalblue', lw=1.5)
    ax_corr.plot(df_lorenz['Time'], df_lorenz['Corr_YZ'], label='Corr(Y, Z)', color='darkorange', lw=1.5)
    ax_corr.plot(df_lorenz['Time'], df_lorenz['Corr_XZ'], label='Corr(X, Z)', color='forestgreen', lw=1.5)
    ax_corr.set_xlabel("Time Step")
    ax_corr.set_ylabel("Correlation Coefficient (r)")
    ax_corr.set_title(f"Rolling Pairwise Correlations (Window Size = {roll_window_size})")
    ax_corr.set_ylim([-1.05, 1.05])
    ax_corr.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax_corr.legend()
    ax_corr.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_corr)
    plt.close()

    st.markdown("---")

    # ---------------------------------------------------------
    # 2. Rolling Granger Causality (p-values & Linear Fits)
    # ---------------------------------------------------------
    st.subheader("2. Rolling Granger Causality Analysis")
    st.markdown(f"Evaluates Granger causality at lag **{max_lag}** over rolling windows of **{roll_window_size}** points.")

    pairs = [('X', 'Y'), ('Y', 'X'), ('Y', 'Z'), ('Z', 'Y'), ('X', 'Z'), ('Z', 'X')]
    
    if st.button("Compute Rolling Granger Causality & Fits"):
        with st.spinner("Computing rolling Granger causality tests and model fits..."):
            
            # Data containers for rolling p-values
            time_indices = []
            gc_results = {f"{src}->{tgt}": [] for src, tgt in pairs}

            for start in range(0, len(df_lorenz) - roll_window_size + 1, roll_step_size):
                end = start + roll_window_size
                sub_df = df_lorenz.iloc[start:end]
                mid_time = sub_df['Time'].iloc[-1]
                time_indices.append(mid_time)

                for src, tgt in pairs:
                    try:
                        # Format: [target, predictor]
                        res = grangercausalitytests(sub_df[[tgt, src]], maxlag=max_lag, verbose=False)
                        p_val = res[max_lag][0]['ssr_ftest'][1]
                    except Exception:
                        p_val = np.nan
                    gc_results[f"{src}->{tgt}"].append(p_val)

            df_gc_rolling = pd.DataFrame(gc_results, index=time_indices)

            # --- Rolling p-values Plot ---
            fig_gc_roll, ax_gc_roll = plt.subplots(figsize=(10, 4.5))
            for col in df_gc_rolling.columns:
                ax_gc_roll.plot(df_gc_rolling.index, df_gc_rolling[col], label=f"{col}", lw=1.2)

            ax_gc_roll.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05 Threshold')
            ax_gc_roll.set_xlabel("Time Step (Window End)")
            ax_gc_roll.set_ylabel("p-value")
            ax_gc_roll.set_title("Rolling Granger Causality p-values Across Pairwise Directions")
            ax_gc_roll.set_ylim([-0.05, 1.05])
            ax_gc_roll.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            ax_gc_roll.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig_gc_roll)
            plt.close()

            # --- Granger Model Linear Fits Time-Series ---
            st.markdown("#### Granger Linear Fit Time-Series (Unrestricted Bivariate Model)")
            
            fit_cols = st.columns(3)
            pair_combos = [('X', 'Y'), ('Y', 'Z'), ('X', 'Z')]
            
            for idx, (v1, v2) in enumerate(pair_combos):
                with fit_cols[idx]:
                    st.markdown(f"**Target: {v2} | Predictors: Past {v2} + Past {v1}**")
                    try:
                        res_fit = grangercausalitytests(df_lorenz[[v2, v1]], maxlag=max_lag, verbose=False)
                        unrestricted_model = res_fit[max_lag][1][1]
                        
                        actuals = unrestricted_model.model.endog
                        fitted = unrestricted_model.fittedvalues
                        time_axis = df_lorenz['Time'].iloc[max_lag:].values

                        fig_f, ax_f = plt.subplots(figsize=(5, 3.5))
                        ax_f.plot(time_axis, actuals, label=f"Actual {v2}", color='black', alpha=0.6, lw=1)
                        ax_f.plot(time_axis, fitted, label=f"Fit from ({v2}, {v1})", color='dodgerblue', linestyle='--', lw=1)
                        ax_f.set_xlabel("Time")
                        ax_f.set_ylabel(v2)
                        ax_f.set_title(f"Linear Fit ({v1} → {v2})")
                        ax_f.legend()
                        ax_f.grid(True, linestyle='--', alpha=0.4)
                        st.pyplot(fig_f)
                        plt.close()
                    except Exception as e:
                        st.error(f"Fit error for {v1}->{v2}: {e}")

    st.markdown("---")

    # ---------------------------------------------------------
    # 3. Rolling CCM Correlation Values (ρ)
    # ---------------------------------------------------------
    st.subheader("3. Rolling CCM Correlation Values (ρ)")
    st.markdown("Evaluates Convergent Cross Mapping correlation ($\rho$) on a moving window across all directional pairs.")

    if st.button("Run Rolling CCM Analysis"):
        with st.spinner("Computing rolling CCM correlation values..."):

            time_ccm = []
            ccm_rho_results = {f"{src}->{tgt}": [] for src, tgt in pairs}
            
            lib_size_eval = min(max_lib_size, roll_window_size - 20)
            lib_str = f"{lib_size_eval} {lib_size_eval} 10"

            for start in range(0, len(df_lorenz) - roll_window_size + 1, roll_step_size):
                end = start + roll_window_size
                sub_df = df_lorenz.iloc[start:end].copy().reset_index(drop=True)
                mid_time = sub_df['Time'].iloc[-1]
                time_ccm.append(mid_time)

                for src, tgt in pairs:
                    try:
                        # src causes tgt implies tgt cross-maps src
                        ccm_out = pyEDM.CCM(
                            dataFrame=sub_df, E=3, columns=tgt, target=src,
                            libSizes=lib_str, sample=20, showPlot=False
                        )
                        col_name = f"{tgt}:{src}"
                        if col_name in ccm_out.columns:
                            rho_val = ccm_out[col_name].iloc[-1]
                        else:
                            rho_val = np.nan
                    except Exception:
                        rho_val = np.nan
                    ccm_rho_results[f"{src}->{tgt}"].append(rho_val)

            df_ccm_roll = pd.DataFrame(ccm_rho_results, index=time_ccm)

            fig_ccm_roll, ax_ccm_roll = plt.subplots(figsize=(10, 4.5))
            for col in df_ccm_roll.columns:
                ax_ccm_roll.plot(df_ccm_roll.index, df_ccm_roll[col], label=f"CCM {col}", lw=1.2)

            ax_ccm_roll.set_xlabel("Time Step (Window End)")
            ax_ccm_roll.set_ylabel("Cross-Mapping Correlation (ρ)")
            ax_ccm_roll.set_title(f"Rolling CCM Correlation (Window Size = {roll_window_size}, L = {lib_size_eval})")
            ax_ccm_roll.set_ylim([-0.1, 1.05])
            ax_ccm_roll.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
            ax_ccm_roll.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig_ccm_roll)
            plt.close()

# ==========================================
# TAB 2: COUPLED LOGISTIC MAP
# ==========================================
with tab2:
    st.header("Coupled Logistic Map Analysis")
    st.markdown(r"$$ X_{t+1} = X_t [r_x - r_x X_t - \beta_{xy} Y_t]$$") 
    st.markdown(r"$$ Y_{t+1} = Y_t [r_y - r_y Y_t - \beta_{yx} X_t] $$")
    df_map = generate_logistic_data(rx, ry, beta_xy, beta_yx, num_points)

    col1_m, col2_m = st.columns(2)
    with col1_m:
        st.subheader("Phase Space (X vs Y)")
        fig1_m, ax1_m = plt.subplots(figsize=(6, 4))
        ax1_m.scatter(df_map['X'], df_map['Y'], s=2, alpha=0.5, color='purple')
        ax1_m.set_xlabel("X")
        ax1_m.set_ylabel("Y")
        st.pyplot(fig1_m)
        plt.close()

    with col2_m:
        st.subheader("Time Series")
        fig2_m, ax2_m = plt.subplots(figsize=(6, 4))
        ax2_m.plot(df_map['Time'], df_map['X'], label="X", marker='.', lw=1)
        ax2_m.plot(df_map['Time'], df_map['Y'], label="Y", marker='.', lw=1)
        ax2_m.set_xlabel("Time (Step)")
        ax2_m.set_ylabel("Value")
        ax2_m.legend()
        st.pyplot(fig2_m)
        plt.close()

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
            plt.close()