import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pyEDM 
from statsmodels.tsa.stattools import grangercausalitytests  

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

# --- Rolling Window Analysis Functions ---
@st.cache_data(show_spinner=False)
def compute_rolling_correlation(df, v1, v2, window_size, step):
    """Computes Pearson correlation between v1 and v2 over a rolling window."""
    times, corrs = [], []
    n = len(df)
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window = df.iloc[start:end]
        corrs.append(window[v1].corr(window[v2]))
        times.append(df['Time'].iloc[end - 1])
    return pd.DataFrame({'Time': times, 'Correlation': corrs})


@st.cache_data(show_spinner=False)
def compute_rolling_granger(df, target, predictor, window_size, step, lag):
    """
    Rolling Granger causality test: does `predictor` Granger-cause `target`?
    Returns, for each window, the F-test p-value plus the R^2 of the
    restricted (target's own past only) and unrestricted (target's past +
    predictor's past) linear models -- the "linear fit" underlying the test.
    """
    times, pvals, r2_restricted, r2_unrestricted = [], [], [], []
    n = len(df)
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window = df.iloc[start:end][[target, predictor]].reset_index(drop=True)
        try:
            gc = grangercausalitytests(window, maxlag=lag, verbose=False)
            pval = gc[lag][0]['ssr_ftest'][1]
            models = gc[lag][1]
            restricted_model, unrestricted_model = models[0], models[1]
            times.append(df['Time'].iloc[end - 1])
            pvals.append(pval)
            r2_restricted.append(restricted_model.rsquared)
            r2_unrestricted.append(unrestricted_model.rsquared)
        except Exception:
            continue
    result = pd.DataFrame({
        'Time': times,
        'PValue': pvals,
        'R2_Restricted': r2_restricted,
        'R2_Unrestricted': r2_unrestricted
    })
    if not result.empty:
        result['DeltaR2'] = result['R2_Unrestricted'] - result['R2_Restricted']
    return result


@st.cache_data(show_spinner=False)
def compute_rolling_ccm(df, v1, v2, window_size, step, E, sample):
    """Rolling CCM cross-map skill (rho) between v1 and v2 over a rolling window."""
    times, rho_12, rho_21 = [], [], []
    n = len(df)
    lib_size = min(window_size, max(E * 3 + 10, window_size - 10))
    lib_str = str(lib_size)
    col_12 = f"{v1}:{v2}"
    col_21 = f"{v2}:{v1}"
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window = df.iloc[start:end][['Time', v1, v2]].reset_index(drop=True)
        try:
            res = pyEDM.CCM(
                dataFrame=window, E=E, columns=v1, target=v2,
                libSizes=lib_str, sample=sample, showPlot=False
            )
            times.append(df['Time'].iloc[end - 1])
            rho_12.append(res[col_12].iloc[-1])
            rho_21.append(res[col_21].iloc[-1])
        except Exception:
            continue
    return pd.DataFrame({'Time': times, col_12: rho_12, col_21: rho_21})

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
max_lag = st.sidebar.slider("Max Lag for Granger Causality", min_value=1, max_value=30, value=10, step=1)

with st.sidebar.expander("Rolling Window Analysis Settings (Tab 1)", expanded=False):
    st.caption("Controls the rolling-window correlation / Granger causality / CCM analysis on the full Lorenz series.")
    roll_window_max = max(60, num_points // 2)
    roll_window_default = min(300, roll_window_max)
    roll_window_size = st.slider("Rolling Window Size", min_value=30, max_value=roll_window_max, value=roll_window_default, step=10)
    roll_step = st.slider("Rolling Step Size", min_value=10, max_value=500, value=50, step=10)
    granger_lag_roll = st.slider("Granger Lag (Rolling)", min_value=1, max_value=20, value=5, step=1)
    ccm_sample_roll = st.slider("CCM Sample Size (Rolling, speed vs. stability)", min_value=10, max_value=200, value=30, step=10)


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

    
    st.subheader("Variable Correlations (Selected Window)")

    corr_matrix = df_lorenz_window[['X', 'Y', 'Z']].corr()
    corr_xy = corr_matrix.loc['X', 'Y']
    corr_yz = corr_matrix.loc['Y', 'Z']
    corr_xz = corr_matrix.loc['X', 'Z']

    corr_col1, corr_col2, corr_col3 = st.columns(3)
    corr_col1.metric("Corr(X, Y)", f"{corr_xy:.3f}")
    corr_col2.metric("Corr(Y, Z)", f"{corr_yz:.3f}")
    corr_col3.metric("Corr(X, Z)", f"{corr_xz:.3f}")

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

                # --- ADDED: Granger Causality Comparison Section ---
                st.markdown("---")
                st.subheader("Granger Causality Comparison")
                
                try:
                    
                    # --- ADDED: Visualize the Underlying Granger Models ---
                    st.markdown("---")
                    st.subheader(f"Univariate and Bivariate Autoregressive Models (Predicting {v2})")

                    gc_12 = grangercausalitytests(df_lorenz_window[[v2, v1]], maxlag=max_lag, verbose=False)
                    p_values_12 = [gc_12[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
                    
                    gc_21 = grangercausalitytests(df_lorenz_window[[v1, v2]], maxlag=max_lag, verbose=False)
                    p_values_21 = [gc_21[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
                    
                    models_tuple = gc_12[max_lag][1]
                    restricted_model = models_tuple[0]    # Univariate AR (Past Y only)
                    unrestricted_model = models_tuple[1]  # Bivariate AR (Past Y + Past X)

                    # The actual target values used in the regression
                    actual_target = unrestricted_model.model.endog
                    pred_restricted = restricted_model.fittedvalues
                    pred_unrestricted = unrestricted_model.fittedvalues

                    # Time axis shifted by the lag
                    time_axis = df_lorenz_window['Time'].iloc[max_lag:].values

                    fig_fit, ax_fit = plt.subplots(figsize=(10, 4))

                    # Plotting all available points
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

                    # df[[target, predictor]] -> checks if predictor Granger-causes target

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
    # ROLLING WINDOW ANALYSIS (FULL TIME SERIES)
    # ==========================================
    st.markdown("---")
    st.header("Rolling Window Analysis (Full Time Series)")
    st.markdown(
        "Slides a fixed-size window across the **entire** Lorenz series (independent of the "
        "window selected above) and recomputes pairwise correlation, Granger causality, and "
        "CCM skill at each window position for all three variable pairs — showing how these "
        "relationships evolve over time, e.g. as the trajectory switches between the two lobes "
        "of the attractor."
    )

    n_windows_est = max(0, (num_points - roll_window_size) // roll_step + 1)
    st.caption(
        f"Current settings produce **{n_windows_est} windows** per pair "
        f"(window size = {roll_window_size}, step = {roll_step}, Granger lag = {granger_lag_roll}). "
        f"CCM is the slowest step (~{n_windows_est * 3} total CCM calls) — this may take a minute or two."
    )

    run_rolling = st.button("Run Rolling Window Analysis", type="primary")

    if run_rolling:
        if roll_window_size >= num_points:
            st.error("Rolling window size must be smaller than the total number of data points.")
        elif n_windows_est < 2:
            st.error("Current settings produce fewer than 2 windows. Reduce the window size or step.")
        else:
            pairs = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
            progress_bar = st.progress(0.0, text="Starting rolling window analysis...")
            total_steps = len(pairs) * 3  # correlation, granger, ccm per pair
            step_count = 0
            rolling_results = {}

            for pa, pb in pairs:
                progress_bar.progress(step_count / total_steps, text=f"Rolling correlation: {pa}-{pb}...")
                corr_df = compute_rolling_correlation(df_lorenz, pa, pb, roll_window_size, roll_step)
                step_count += 1

                progress_bar.progress(step_count / total_steps, text=f"Rolling Granger causality: {pa}-{pb}...")
                # granger_ab: does pa Granger-cause pb? (target=pb, predictor=pa)
                granger_ab = compute_rolling_granger(df_lorenz, pb, pa, roll_window_size, roll_step, granger_lag_roll)
                # granger_ba: does pb Granger-cause pa? (target=pa, predictor=pb)
                granger_ba = compute_rolling_granger(df_lorenz, pa, pb, roll_window_size, roll_step, granger_lag_roll)
                step_count += 1

                progress_bar.progress(step_count / total_steps, text=f"Rolling CCM: {pa}-{pb}...")
                ccm_df = compute_rolling_ccm(df_lorenz, pa, pb, roll_window_size, roll_step, 3, ccm_sample_roll)
                step_count += 1

                rolling_results[(pa, pb)] = {
                    'corr': corr_df,
                    'granger_ab': granger_ab,
                    'granger_ba': granger_ba,
                    'ccm': ccm_df
                }

            progress_bar.progress(1.0, text="Done!")
            progress_bar.empty()

            st.session_state['rolling_results'] = rolling_results
            st.session_state['rolling_pairs'] = pairs
            st.session_state['rolling_lag'] = granger_lag_roll

    if 'rolling_results' in st.session_state:
        rolling_results = st.session_state['rolling_results']
        pairs = st.session_state['rolling_pairs']
        lag_used = st.session_state['rolling_lag']

        for idx, (pa, pb) in enumerate(pairs):
            data = rolling_results[(pa, pb)]
            with st.expander(f"{pa} ↔ {pb}", expanded=(idx == 0)):

                row1_col1, row1_col2 = st.columns(2)

                with row1_col1:
                    st.markdown(f"**Rolling Pearson Correlation ({pa}, {pb})**")
                    corr_df = data['corr']
                    if not corr_df.empty:
                        fig_rc, ax_rc = plt.subplots(figsize=(6, 4))
                        ax_rc.plot(corr_df['Time'], corr_df['Correlation'], color='royalblue', lw=1.5)
                        ax_rc.axhline(0, color='gray', linestyle=':', lw=1)
                        ax_rc.set_xlabel("Time (Window End)")
                        ax_rc.set_ylabel(f"Corr({pa}, {pb})")
                        ax_rc.set_ylim([-1.05, 1.05])
                        ax_rc.grid(True, linestyle='--', alpha=0.4)
                        st.pyplot(fig_rc)
                        plt.close(fig_rc)
                    else:
                        st.info("No windows computed.")

                with row1_col2:
                    st.markdown(f"**Rolling CCM Cross-Map Skill ({pa}, {pb})**")
                    ccm_df = data['ccm']
                    col_ab = f"{pa}:{pb}"
                    col_ba = f"{pb}:{pa}"
                    if not ccm_df.empty and col_ab in ccm_df.columns and col_ba in ccm_df.columns:
                        fig_rccm, ax_rccm = plt.subplots(figsize=(6, 4))
                        ax_rccm.plot(ccm_df['Time'], ccm_df[col_ab], marker='o', ms=3,
                                     label=f'{pa} cross-maps {pb} ({pb} causes {pa})')
                        ax_rccm.plot(ccm_df['Time'], ccm_df[col_ba], marker='s', ms=3,
                                     label=f'{pb} cross-maps {pa} ({pa} causes {pb})')
                        ax_rccm.set_xlabel("Time (Window End)")
                        ax_rccm.set_ylabel("CCM ρ")
                        ax_rccm.set_ylim([-0.1, 1.1])
                        ax_rccm.legend(fontsize=8)
                        ax_rccm.grid(True, linestyle='--', alpha=0.4)
                        st.pyplot(fig_rccm)
                        plt.close(fig_rccm)
                    else:
                        st.info("CCM could not be computed for these windows (try a larger window size).")

                g_ab, g_ba = data['granger_ab'], data['granger_ba']
                row2_col1, row2_col2 = st.columns(2)

                with row2_col1:
                    st.markdown(f"**Rolling Granger Causality: p-values (lag={lag_used})**")
                    if not g_ab.empty and not g_ba.empty:
                        fig_gp, ax_gp = plt.subplots(figsize=(6, 4))
                        ax_gp.plot(g_ab['Time'], g_ab['PValue'], marker='o', ms=3, color='C1',
                                   label=f'{pa} causes {pb}')
                        ax_gp.plot(g_ba['Time'], g_ba['PValue'], marker='s', ms=3, color='C0',
                                   label=f'{pb} causes {pa}')
                        ax_gp.axhline(0.05, color='r', linestyle='--', lw=1, label='α = 0.05')
                        ax_gp.set_xlabel("Time (Window End)")
                        ax_gp.set_ylabel("p-value")
                        ax_gp.set_ylim([-0.05, 1.05])
                        ax_gp.legend(fontsize=8)
                        ax_gp.grid(True, linestyle='--', alpha=0.4)
                        st.pyplot(fig_gp)
                        plt.close(fig_gp)
                    else:
                        st.info("Granger causality could not be computed for these windows.")

                with row2_col2:
                    st.markdown("**Rolling Granger Causality: Linear Fit (ΔR²)**")
                    if not g_ab.empty and not g_ba.empty:
                        fig_gf, ax_gf = plt.subplots(figsize=(6, 4))
                        ax_gf.plot(g_ab['Time'], g_ab['DeltaR2'], marker='o', ms=3, color='C1',
                                   label=f'{pa} causes {pb}')
                        ax_gf.plot(g_ba['Time'], g_ba['DeltaR2'], marker='s', ms=3, color='C0',
                                   label=f'{pb} causes {pa}')
                        ax_gf.axhline(0, color='gray', linestyle=':', lw=1)
                        ax_gf.set_xlabel("Time (Window End)")
                        ax_gf.set_ylabel("ΔR² (Unrestricted − Restricted)")
                        ax_gf.legend(fontsize=8)
                        ax_gf.grid(True, linestyle='--', alpha=0.4)
                        st.pyplot(fig_gf)
                        plt.close(fig_gf)
                    else:
                        st.info("Linear fit could not be computed for these windows.")

                st.caption(
                    f"ΔR² is the increase in R² when {pb}'s (or {pa}'s) own-lag model also gets the "
                    f"other variable's lagged values — the same nested-model comparison the Granger "
                    f"F-test (left) is built on, so the two panels are two views of one test: "
                    f"significance (p-value) vs. magnitude (ΔR²)."
                )


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