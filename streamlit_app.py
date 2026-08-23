import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import pyEDM 
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mutual_info_score

# --- Page Configuration ---
st.set_page_config(page_title="CCM & Causality", layout="wide")

# --- Information Theory Functions ---
@st.cache_data
def calc_mutual_information(x, y, bins=2):
    x_edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    y_edges = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))
    c_xy = np.histogram2d(x, y, bins=[x_edges, y_edges])[0]
    return mutual_info_score(None, None, contingency=c_xy)

@st.cache_data
def calc_transfer_entropy(x, y, lag=1, bins=2):
    y_t, y_past, x_past = y[lag:], y[:-lag], x[:-lag]
    
    yt_edges = np.unique(np.quantile(y_t, np.linspace(0, 1, bins + 1)))
    yp_edges = np.unique(np.quantile(y_past, np.linspace(0, 1, bins + 1)))
    xp_edges = np.unique(np.quantile(x_past, np.linspace(0, 1, bins + 1)))
    
    c_3d, _ = np.histogramdd(np.vstack([y_t, y_past, x_past]).T, bins=[yt_edges, yp_edges, xp_edges])
    p_3d = c_3d / np.sum(c_3d)
    
    p_y_past_x_past = np.sum(p_3d, axis=0)
    p_y_t_y_past = np.sum(p_3d, axis=2)
    p_y_past = np.sum(p_y_past_x_past, axis=1)
    
    p_3d_safe = np.where(p_3d > 0, p_3d, 1e-10)
    p_y_past_x_past_safe = np.where(p_y_past_x_past > 0, p_y_past_x_past, 1e-10)
    p_y_t_y_past_safe = np.where(p_y_t_y_past > 0, p_y_t_y_past, 1e-10)
    p_y_past_safe = np.where(p_y_past > 0, p_y_past, 1e-10)
    
    b_yt, b_yp, b_xp = c_3d.shape
    term1 = p_3d_safe * p_y_past_safe.reshape(1, b_yp, 1)
    term2 = p_y_t_y_past_safe.reshape(b_yt, b_yp, 1) * p_y_past_x_past_safe.reshape(1, b_yp, b_xp)
    
    te = np.sum(p_3d * np.log2(term1 / term2))
    return max(0.0, te)

# --- Data Generation Functions ---
@st.cache_data
def generate_lorenz_ensemble(sigma, rho, beta, t_max, num_points, perturbation_scale=0.1, phase_offset=0.0):
    def lorenz_system(t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    transient_sol = solve_ivp(lorenz_system, [0, 50.0], [1.0, 1.0, 1.0])
    state_on_attractor = transient_sol.y[:, -1]
    
    if phase_offset > 0:
        offset_sol = solve_ivp(lorenz_system, [0, phase_offset], state_on_attractor)
        base_state = offset_sol.y[:, -1]
    else:
        base_state = state_on_attractor
    
    states = [base_state]
    np.random.seed(42)
    for _ in range(10):
        states.append(base_state + np.random.normal(0, perturbation_scale, 3))

    t_eval = np.linspace(0, t_max, num_points)
    dfs = []
    for i, start_state in enumerate(states):
        sol = solve_ivp(lorenz_system, [0, t_max], start_state, t_eval=t_eval)
        dfs.append(pd.DataFrame({'Time': np.arange(1, num_points + 1), 'X': sol.y[0], 'Y': sol.y[1], 'Z': sol.y[2], 'RunID': i}))
    return dfs

@st.cache_data
def generate_logistic_data(rx, ry, beta_xy, beta_yx, num_points):
    x, y = np.zeros(num_points), np.zeros(num_points)
    x[0], y[0] = 0.4, 0.2
    for t in range(num_points - 1):
        x[t+1] = max(0, min(1, x[t] * (rx - rx * x[t] - beta_xy * y[t])))
        y[t+1] = max(0, min(1, y[t] * (ry - ry * y[t] - beta_yx * x[t])))
    return pd.DataFrame({'Time': np.arange(1, num_points + 1), 'X': x, 'Y': y})

@st.cache_data
def generate_sinusoids(num_points, noise_level=0.1):
    t = np.linspace(0, 10 * np.pi, num_points)
    np.random.seed(42)
    x = np.sin(t) + np.random.normal(0, noise_level, num_points)
    y = np.sin(t + np.pi/4) + np.random.normal(0, noise_level, num_points)
    z = np.sin(t + np.pi/2) + np.random.normal(0, noise_level, num_points)
    return [pd.DataFrame({'Time': np.arange(1, num_points + 1), 'X': x, 'Y': y, 'Z': z})]

@st.cache_data
def generate_common_driver(num_points, noise_level=0.1):
    t = np.linspace(0, 20 * np.pi, num_points)
    np.random.seed(42)
    driver = np.sin(t) + np.cos(2*t)
    x = driver + np.random.normal(0, noise_level, num_points)
    y = -driver + np.random.normal(0, noise_level, num_points)
    z = driver**2 + np.random.normal(0, noise_level, num_points)
    return [pd.DataFrame({'Time': np.arange(1, num_points + 1), 'X': x, 'Y': y, 'Z': z})]

# --- Reusable UI Component for Analysis ---
def render_pairwise_analysis(dfs_window, df_base_win, pairs, window_len, max_lib_size, lib_step, max_lag, bin_steps, E_dim=3):
    for v1, v2 in pairs:
        st.markdown(f"---")
        st.header(f"Analysis: {v1} and {v2}")
        
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            for i, df in enumerate(dfs_window):
                ax1.plot(df[v1], df[v2], lw=1.0 if i==0 else 0.5, color='royalblue' if i==0 else 'gray', alpha=1.0 if i==0 else 0.3)
                ax1.scatter(df[v1].iloc[0], df[v2].iloc[0], color='red', s=5, zorder=5)
            ax1.set_xlabel(v1)
            ax1.set_ylabel(v2)
            ax1.set_title(f"Phase Space ({v1} vs {v2})")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            for i, df in enumerate(dfs_window):
                ax2.plot(df['Time'], df[v1], color='dodgerblue', lw=1.0 if i==0 else 0.5, alpha=1.0 if i==0 else 0.3, label=v1 if i==0 else "")
                ax2.plot(df['Time'], df[v2], color='orange', lw=1.0 if i==0 else 0.5, alpha=1.0 if i==0 else 0.3, label=v2 if i==0 else "")
            ax2.set_xlabel("Time (Index)")
            ax2.set_ylabel("Value")
            ax2.legend()
            ax2.set_title(f"Time Series ({v1} and {v2})")
            st.pyplot(fig2)

        mi_matrix = np.zeros((len(dfs_window), len(bin_steps)))
        te_12_matrix = np.zeros((len(dfs_window), len(bin_steps)))
        te_21_matrix = np.zeros((len(dfs_window), len(bin_steps)))
        
        for i, df in enumerate(dfs_window):
            for j, b in enumerate(bin_steps):
                mi_matrix[i, j] = calc_mutual_information(df[v1].values, df[v2].values, bins=b)
                te_12_matrix[i, j] = calc_transfer_entropy(df[v1].values, df[v2].values, lag=1, bins=b)
                te_21_matrix[i, j] = calc_transfer_entropy(df[v2].values, df[v1].values, lag=1, bins=b)

        fig_info, ax_info = plt.subplots(figsize=(10, 4))
        x_b = [str(b) for b in bin_steps]
        ax_info.plot(x_b, np.mean(mi_matrix, axis=0), marker='o', label='MI', color='green')
        ax_info.plot(x_b, np.mean(te_12_matrix, axis=0), marker='s', label=f'TE: {v1}->{v2}', color='orange')
        ax_info.plot(x_b, np.mean(te_21_matrix, axis=0), marker='^', label=f'TE: {v2}->{v1}', color='purple')
        ax_info.set_xlabel("Equal-Mass Bins")
        ax_info.set_ylabel("Bits")
        ax_info.set_title("Information Theory Metrics")
        ax_info.legend()
        ax_info.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_info)
        plt.close()

    st.markdown("---")
    st.header("Comprehensive Causality Analysis (All Pairs)")
    if st.button(f"Run CCM & Granger Analysis", key=f"btn_{pairs[0]}"):
        if window_len < 100:
            st.error("Time window too small.")
        else:
            with st.spinner("Processing..."):
                for v1, v2 in pairs:
                    st.subheader(f"Coupling Results: {v1} and {v2}")
                    actual_max_lib = min(max_lib_size, max(10, window_len - 50))
                    lib_sizes_str = f"10 {actual_max_lib} {lib_step}"
                    
                    ccm_12_all, ccm_21_all = [], []
                    lib_sizes = None

                    for df in dfs_window:
                        ccm_res = pyEDM.CCM(dataFrame=df, E=E_dim, columns=v1, target=v2, libSizes=lib_sizes_str, sample=100, showPlot=False)
                        if lib_sizes is None: lib_sizes = ccm_res['LibSize'].values
                        if f"{v1}:{v2}" in ccm_res: ccm_12_all.append(ccm_res[f"{v1}:{v2}"].values)
                        if f"{v2}:{v1}" in ccm_res: ccm_21_all.append(ccm_res[f"{v2}:{v1}"].values)

                    fig3, ax3 = plt.subplots(figsize=(8, 4))
                    if len(ccm_12_all) > 0:
                        ax3.plot(lib_sizes, np.mean(ccm_12_all, axis=0), marker='o', label=f'{v1} cross-maps {v2}')
                        ax3.plot(lib_sizes, np.mean(ccm_21_all, axis=0), marker='s', label=f'{v2} cross-maps {v1}')
                    ax3.set_xlabel("Library Size (L)")
                    ax3.set_ylabel("Correlation (ρ)")
                    ax3.set_title(f"CCM Convergence ({v1} vs {v2})")
                    ax3.legend()
                    ax3.grid(True, linestyle='--')
                    ax3.set_ylim([-0.1, 1.1])
                    st.pyplot(fig3)
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        simp12 = pyEDM.Simplex(dataFrame=df_base_win, lib=[1, int(actual_max_lib)], pred=[1, int(window_len)], columns=v2, target=v1, E=E_dim, Tp=0, tau=-1)
                        fig12, ax12 = plt.subplots(figsize=(5,5))
                        ax12.scatter(simp12['Observations'], simp12['Predictions'], alpha=0.4)
                        ax12.plot([simp12['Observations'].min(), simp12['Observations'].max()], [simp12['Observations'].min(), simp12['Observations'].max()], 'r--')
                        ax12.set_title(f"M_{v2} predicts {v1} (ρ={simp12['Observations'].corr(simp12['Predictions']):.3f})")
                        st.pyplot(fig12)
                    with col4:
                        simp21 = pyEDM.Simplex(dataFrame=df_base_win, lib=[1, int(actual_max_lib)], pred=[1, int(window_len)], columns=v1, target=v2, E=E_dim, Tp=0, tau=-1)
                        fig21, ax21 = plt.subplots(figsize=(5,5))
                        ax21.scatter(simp21['Observations'], simp21['Predictions'], alpha=0.4, color='orange')
                        ax21.plot([simp21['Observations'].min(), simp21['Observations'].max()], [simp21['Observations'].min(), simp21['Observations'].max()], 'r--')
                        ax21.set_title(f"M_{v1} predicts {v2} (ρ={simp21['Observations'].corr(simp21['Predictions']):.3f})")
                        st.pyplot(fig21)

                    try:
                        gc_12 = grangercausalitytests(df_base_win[[v2, v1]], maxlag=max_lag, verbose=False)
                        gc_21 = grangercausalitytests(df_base_win[[v1, v2]], maxlag=max_lag, verbose=False)
                        fig_gc, ax_gc = plt.subplots(figsize=(8, 4))
                        ax_gc.plot(range(1, max_lag+1), [gc_12[l][0]['ssr_ftest'][1] for l in range(1, max_lag+1)], marker='o', label=f'{v1} Granger-causes {v2}')
                        ax_gc.plot(range(1, max_lag+1), [gc_21[l][0]['ssr_ftest'][1] for l in range(1, max_lag+1)], marker='s', label=f'{v2} Granger-causes {v1}')
                        ax_gc.axhline(0.05, color='r', linestyle='--', label='α = 0.05')
                        ax_gc.set_title(f"Granger Causality (p-values)")
                        ax_gc.legend()
                        st.pyplot(fig_gc)
                    except Exception as e:
                        st.warning(f"Granger Causality failed: {e}")

# --- Sidebar UI ---
st.sidebar.title("Configuration")
num_points = st.sidebar.slider("Data Points", 500, 10000, 2000, step=100)
max_lib_size = st.sidebar.slider("Max Library Size", 100, num_points-50, min(500, num_points-50), step=50)
lib_step = st.sidebar.number_input("Library Step", 10, 200, 20)
max_lag = st.sidebar.slider("Max Lag", 1, 30, 10)
bin_steps = [2, 4, 8, 16]
noise_level = st.sidebar.slider("Noise Level (Sinusoids)", 0.0, 1.0, 0.1, step=0.05)
window_start, window_end = st.sidebar.slider("Time Window", 1, num_points, (1, num_points), step=10)

# --- Main Application ---
st.title("CCM & Information Theory Explorer")
tab1, tab2, tab3, tab4 = st.tabs(["Lorenz", "Logistic Map", "Phase-Shifted Sinusoids", "Common Driver"])

pairs_xyz = [("X", "Y"), ("X", "Z"), ("Y", "Z")]

with tab1:
    st.header("Lorenz Attractor")
    dfs = generate_lorenz_ensemble(10.0, 28.0, 2.666, 40.0, num_points)
    dfs_win = [df[(df['Time'] >= window_start) & (df['Time'] <= window_end)].reset_index(drop=True) for df in dfs]
    if len(dfs_win[0]) > 50:
        render_pairwise_analysis(dfs_win, dfs_win[0], pairs_xyz, len(dfs_win[0]), max_lib_size, lib_step, max_lag, bin_steps, E_dim=3)

with tab2:
    st.header("Coupled Logistic Map")
    df_map = generate_logistic_data(3.8, 3.5, 0.02, 0.10, num_points)
    df_win = [df_map[(df_map['Time'] >= window_start) & (df_map['Time'] <= window_end)].reset_index(drop=True)]
    if len(df_win[0]) > 50:
        render_pairwise_analysis(df_win, df_win[0], [("X", "Y")], len(df_win[0]), max_lib_size, lib_step, max_lag, bin_steps, E_dim=2)

with tab3:
    st.header("Phase-Shifted Sinusoids")
    st.write("Identical sine waves offset in phase. They are highly correlated but have no causal linkage.")
    dfs_sin = generate_sinusoids(num_points, noise_level)
    dfs_sin_win = [df[(df['Time'] >= window_start) & (df['Time'] <= window_end)].reset_index(drop=True) for df in dfs_sin]
    if len(dfs_sin_win[0]) > 50:
        render_pairwise_analysis(dfs_sin_win, dfs_sin_win[0], pairs_xyz, len(dfs_sin_win[0]), max_lib_size, lib_step, max_lag, bin_steps, E_dim=2)

with tab4:
    st.header("Common Driver System")
    st.write("A shared driver generates X, Y, and Z. They move together but do not influence each other directly.")
    dfs_com = generate_common_driver(num_points, noise_level)
    dfs_com_win = [df[(df['Time'] >= window_start) & (df['Time'] <= window_end)].reset_index(drop=True) for df in dfs_com]
    if len(dfs_com_win[0]) > 50:
        render_pairwise_analysis(dfs_com_win, dfs_com_win[0], pairs_xyz, len(dfs_com_win[0]), max_lib_size, lib_step, max_lag, bin_steps, E_dim=3)