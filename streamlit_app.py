import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import pyEDM 
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mutual_info_score
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Measures of Causality and Dynamical Systems", layout="wide")

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
def generate_lorenz_ensemble(sigma, rho, beta, t_max, num_points):
    def lorenz_system(t, state): return [sigma*(state[1]-state[0]), state[0]*(rho-state[2])-state[1], state[0]*state[1]-beta*state[2]]
    transient = solve_ivp(lorenz_system, [0, 50.0], [1.0, 1.0, 1.0]).y[:, -1]
    t_eval = np.linspace(0, t_max, num_points)
    sol = solve_ivp(lorenz_system, [0, t_max], transient, t_eval=t_eval)
    return [pd.DataFrame({'Time': np.arange(1, num_points + 1), 'X': sol.y[0], 'Y': sol.y[1], 'Z': sol.y[2]})]

@st.cache_data
def generate_logistic_data(rx, ry, beta_xy, beta_yx, num_points):
    x, y = np.zeros(num_points), np.zeros(num_points)
    x[0], y[0] = 0.4, 0.2
    for t in range(num_points - 1):
        x[t+1] = max(0, min(1, x[t] * (rx - rx * x[t] - beta_xy * y[t])))
        y[t+1] = max(0, min(1, y[t] * (ry - ry * y[t] - beta_yx * x[t])))
    return [pd.DataFrame({'Time': np.arange(1, num_points + 1), 'X': x, 'Y': y})]

@st.cache_data
def generate_phase_sinusoids(num_points, noise, phase_y, phase_z):
    t = np.linspace(0, 10 * np.pi, num_points)
    np.random.seed(42)
    return [pd.DataFrame({
        'Time': np.arange(1, num_points + 1), 
        'X': np.sin(t) + np.random.normal(0, noise, num_points),
        'Y': np.sin(t + phase_y) + np.random.normal(0, noise, num_points),
        'Z': np.sin(t + phase_z) + np.random.normal(0, noise, num_points)
    })]

@st.cache_data
def generate_frequency_sinusoids(num_points, noise, freq_x, freq_y, freq_z):
    t = np.linspace(0, 10 * np.pi, num_points)
    np.random.seed(42)
    return [pd.DataFrame({
        'Time': np.arange(1, num_points + 1), 
        'X': np.sin(freq_x * t) + np.random.normal(0, noise, num_points),
        'Y': np.sin(freq_y * t) + np.random.normal(0, noise, num_points),
        'Z': np.sin(freq_z * t) + np.random.normal(0, noise, num_points)
    })]

# --- Reusable UI Component for Analysis ---
def render_pairwise_analysis(dfs_window, df_base_win, pairs, window_len, max_lib_size, lib_step, max_lag, bin_steps, E_dim=3, key_suffix=""):
    for v1, v2 in pairs:
        st.markdown(f"---")
        st.header(f"Analysis: {v1} and {v2}")
        
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(df_base_win[v1], df_base_win[v2], lw=1.0, color='royalblue')
            ax1.set_xlabel(f"{v1} Value")
            ax1.set_ylabel(f"{v2} Value")
            ax1.set_title(f"Phase Space ({v1} vs {v2})")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(df_base_win['Time'], df_base_win[v1], color='dodgerblue', lw=1.0, label=v1)
            ax2.plot(df_base_win['Time'], df_base_win[v2], color='orange', lw=1.0, label=v2)
            ax2.set_xlabel("Time (Index)")
            ax2.set_ylabel("Amplitude")
            ax2.set_title(f"Time Series")
            ax2.legend()
            st.pyplot(fig2)

        pearson_r = df_base_win[v1].corr(df_base_win[v2])
        st.write(f"**Linear Pearson Correlation (r):** {pearson_r:.3f}")

        mi_vals, te12_vals, te21_vals = [], [], []
        for b in bin_steps:
            mi_vals.append(calc_mutual_information(df_base_win[v1].values, df_base_win[v2].values, bins=b))
            te12_vals.append(calc_transfer_entropy(df_base_win[v1].values, df_base_win[v2].values, lag=1, bins=b))
            te21_vals.append(calc_transfer_entropy(df_base_win[v2].values, df_base_win[v1].values, lag=1, bins=b))

        fig_info, ax_info = plt.subplots(figsize=(10, 3))
        x_b = [str(b) for b in bin_steps]
        ax_info.plot(x_b, mi_vals, marker='o', label='MI', color='green')
        ax_info.plot(x_b, te12_vals, marker='s', label=f'TE: {v1}->{v2}', color='orange')
        ax_info.plot(x_b, te21_vals, marker='^', label=f'TE: {v2}->{v1}', color='purple')
        ax_info.set_title("Information Theory Metrics vs Equal-Mass Bins")
        ax_info.set_xlabel("Number of Equal-Mass Bins")
        ax_info.set_ylabel("Bits")
        ax_info.legend()
        st.pyplot(fig_info)
        plt.close()

    st.markdown("---")
    st.header("Causality Analysis")
    
    # Tooltip added to the execution button
    if st.button(f"Run CCM & Granger", key=f"btn_{pairs[0]}_{E_dim}_{key_suffix}", 
                 help="Executes convergent cross-mapping and Granger causality routines."):
        with st.spinner("Processing Causality Metrics..."):
            for v1, v2 in pairs:
                st.subheader(f"Coupling Results: {v1} & {v2}")
                actual_max_lib = min(max_lib_size, max(10, window_len - 50))
                ccm_res = pyEDM.CCM(dataFrame=df_base_win, E=E_dim, columns=v1, target=v2, libSizes=f"10 {actual_max_lib} {lib_step}", sample=100, showPlot=False)
                
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                if f"{v1}:{v2}" in ccm_res: ax3.plot(ccm_res['LibSize'], ccm_res[f"{v1}:{v2}"], marker='o', label=f'{v1} cross-maps {v2}')
                if f"{v2}:{v1}" in ccm_res: ax3.plot(ccm_res['LibSize'], ccm_res[f"{v2}:{v1}"], marker='s', label=f'{v2} cross-maps {v1}')
                ax3.set_title("CCM Convergence")
                ax3.set_xlabel("Library Size (L)")
                ax3.set_ylabel("Correlation (ρ)")
                ax3.legend()
                ax3.set_ylim([-0.1, 1.1])
                st.pyplot(fig3)
                
                col3, col4 = st.columns(2)
                with col3:
                    simp12 = pyEDM.Simplex(dataFrame=df_base_win, lib=[1, actual_max_lib], pred=[1, window_len], columns=v2, target=v1, E=E_dim, Tp=0, tau=-1)
                    fig12, ax12 = plt.subplots()
                    ax12.scatter(simp12['Observations'], simp12['Predictions'], alpha=0.4, color='C1')
                    ax12.plot([simp12['Observations'].min(), simp12['Observations'].max()], [simp12['Observations'].min(), simp12['Observations'].max()], 'r--')
                    ax12.set_title(f"M_{v2} predicts {v1} (ρ={simp12['Observations'].corr(simp12['Predictions']):.3f})")
                    ax12.set_xlabel("Observed Values")
                    ax12.set_ylabel("Predicted Values")
                    st.pyplot(fig12)
                with col4:
                    simp21 = pyEDM.Simplex(dataFrame=df_base_win, lib=[1, actual_max_lib], pred=[1, window_len], columns=v1, target=v2, E=E_dim, Tp=0, tau=-1)
                    fig21, ax21 = plt.subplots()
                    ax21.scatter(simp21['Observations'], simp21['Predictions'], alpha=0.4, color='C0')
                    ax21.plot([simp21['Observations'].min(), simp21['Observations'].max()], [simp21['Observations'].min(), simp21['Observations'].max()], 'r--')
                    ax21.set_title(f"M_{v1} predicts {v2} (ρ={simp21['Observations'].corr(simp21['Predictions']):.3f})")
                    ax21.set_xlabel("Observed Values")
                    ax21.set_ylabel("Predicted Values")
                    st.pyplot(fig21)
                    
                try:
                    gc_12 = grangercausalitytests(df_base_win[[v2, v1]], maxlag=max_lag)
                    gc_21 = grangercausalitytests(df_base_win[[v1, v2]], maxlag=max_lag)
                    fig_gc, ax_gc = plt.subplots(figsize=(8, 4))
                    ax_gc.plot(range(1, max_lag+1), [gc_12[l][0]['ssr_ftest'][1] for l in range(1, max_lag+1)], marker='o', label=f'{v1} Granger-causes {v2}')
                    ax_gc.plot(range(1, max_lag+1), [gc_21[l][0]['ssr_ftest'][1] for l in range(1, max_lag+1)], marker='s', label=f'{v2} Granger-causes {v1}')
                    ax_gc.axhline(0.05, color='r', linestyle='--', label='α = 0.05')
                    ax_gc.set_title("Granger Causality Significance vs. Lags")
                    ax_gc.set_xlabel("Lag Step")
                    ax_gc.set_ylabel("p-value")
                    ax_gc.set_ylim([-0.05, 1.05])
                    ax_gc.legend()
                    st.pyplot(fig_gc)
                except Exception as e:
                    st.warning(f"Could not compute Granger Causality for {v1} & {v2}: {e}")

# --- Sidebar UI with Tooltips ---
st.sidebar.title("Configuration")
num_points = st.sidebar.slider("Data Points", 500, 10000, 2000, step=100, 
                               help="Total number of discrete time steps generated for the modeled system.")
max_lib_size = st.sidebar.slider("Max Library Size", 100, num_points-50, step=50, 
                                 help="Maximum number of historical observations used to reconstruct the state space manifold in CCM. Larger libraries improve prediction if causality exists.")
lib_step = st.sidebar.number_input("Library Step", 10, 200, 20, 
                                   help="The increment by which the library size increases to measure convergence in cross-mapping.")
max_lag = st.sidebar.slider("Max Lag", 1, 30, 10, 
                            help="The maximum number of past time steps (lags) evaluated by the autoregressive model to determine Granger causality.")
bin_steps = [2, 4, 8, 16]
noise = st.sidebar.slider("Noise Level (Sinusoids)", 0.0, 1.0, 0.1, 
                          help="The amount of Gaussian noise added to the sinusoid signals.")

# --- Main App ---
st.title("Measures of Causality and Dynamical Systems")
intro, tab0, tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Key Systems and Techniques", "Lorenz", "Logistic", "Phase Shifts", "Freq Shifts"])
pairs_xyz = [("X", "Y"), ("X", "Z"), ("Y", "Z")]

with intro: 
    st.header("Introduction")
    st.markdown(""" 
    In chaotic systems, minute effects have large cumulative impacts, resulting in a significant divergence of trajectories, 
    even if their initial starting points are close. An example of a chaotic system is the Lorenz system, 
    which transitions through various regimes, including bifurcations, transient chaos, and strange attractors. 
    Understanding the causal links governing chaotic systems is crucial for prediction and modelling. 
    Causality can be determined from a variety of methods, including convergent cross-mapping (CCM), information-theoretic methods, 
    and more traditional statistical methods like Granger causality, all with varying effectiveness. 
    The goal of this project is to determine which method is the best at determining causality: Pearson correlation, 
    convergent cross-mapping, Granger causality, or information-theoretic measures. In particular, 
    the goal is to understand if convergent cross-mapping is indeed the best way to determine causality. 
    The other three methods are common tools to determine relationships between random variables, used here to compare with CCM. 
    The performance of these four approaches are compared on four different systems: the Lorenz system, the coupled logistic map, 
    a sinusoidal system with a phase-shift difference, and a sinusoidal system with a frequency difference. 
    The first system is an example of causation without correlation, the second system is an example of asymmetric causation, 
    and the last two are examples of correlation without causation.
    """)

with tab0:
    st.header("The Four Dynamical Systems")
    st.markdown("""
    * **Lorenz System**: The Lorenz system is a continuous-time, three-dimensional nonlinear deterministic system. It exhibits chaotic behavior characterized by the famous "butterfly" attractor shape. A good causality predictor should detect a causal relationship between the three variables, despite the lack of obvious correlation and linearity.
    * **Coupled Logistic Map**: The coupled logistic map is a discrete-time demographic model used to simulate chaotic population dynamics and featured in the 2012 Sugihara paper in Science analyzing CCM. As a coupled system, the system is characterized by two parameters that control the strength of each variable on the other. Adjusting the beta parameters can give one variable a disproportionate effect on the other, meaning a good causality predictor should forecast an asymmetric causal relationship.
    * **Phase-Shifted Sinusoids**: This is a periodic sinusoidal system with variables shifted in time with additional Gaussian noise, illustrating an environment where variables are correlated but not causally related. A good causality predictor should not forecast a causal relationship since the variables are only correlated.
    * **Frequency-Shifted Sinusoids**: This is a periodic sinusoidal system with variables in different frequencies with additional Gaussian noise, illustrating an environment where variables are correlated but not causally related. A good causality predictor should not forecast a causal relationship since the variables are only correlated.
    """)
    
    st.markdown("---")
    st.header("The Four Analytical Approaches")
    st.markdown("""
    * **Pearson Correlation**: A symmetric measure of the linear relationship between two variables.
    * **Information-theoretical Measures**: Mutual information is a symmetric measure of the statistical dependence between two variables without assuming linearity. Transfer entropy extends this to measure the directed exchange of information over time, identifying which variable has more predictive power than the other.
    * **Granger Causality**: A statistical hypothesis test asserting that if a signal $X$ causes $Y$, past values of $X$ should help forecast $Y$ better than using past values of $Y$ alone, assuming the underlying relationships are linear.
    * **Convergent Cross-Mapping (CCM)**: A methodology developed by Sugihara et al. designed specifically for non-linear, dynamic environments where Granger causality fails. It relies on the theoretical underpinnings of Takens' embedding theorem to reconstruct state spaces to see if the historical states of a variable $X$ can reliably estimate the historical states of another variable $Y$. If $X$ controls $Y$, the history of $Y$ cross-maps (i.e. predicts) the states of $X$.
    """)

with tab1:
    st.header("Lorenz Attractor")
    st.latex(r"""
    \begin{align*}
    \frac{dx}{dt} &= \sigma (y - x) \\
    \frac{dy}{dt} &= x (\rho - z) - y \\
    \frac{dz}{dt} &= x y - \beta z
    \end{align*}
    """)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        sigma_val = st.slider(r"$\sigma$ (Sigma)", 1.0, 20.0, 10.0, step=0.1, help="Prandtl number: controls the rate of heat transfer.")
    with col_p2:
        rho_val = st.slider(r"$\rho$ (Rho)", 10.0, 50.0, 28.0, step=0.5, help="Rayleigh number: controls the strength of convection.")
    with col_p3:
        beta_val = st.slider(r"$\beta$ (Beta)", 0.5, 5.0, 2.666, step=0.001, help="Physical proportion of the convection space.")

    dfs1 = generate_lorenz_ensemble(sigma_val, rho_val, beta_val, 40.0, num_points)
    df_base1 = dfs1[0]
    
    st.markdown("---")
    st.subheader("Global Attractor Dynamics & Distributions")
    
    z_vals, x_vals = df_base1['Z'].values, df_base1['X'].values
    peaks, _ = find_peaks(z_vals)
    num_cycles = len(peaks)
    avg_cycle_steps = np.mean(np.diff(peaks)) if num_cycles > 1 else 0
    zero_crossings = np.where(np.diff(np.sign(x_vals)))[0]
    
    if len(zero_crossings) > 0:
        loops_per_lobe = []
        start_idx = 0
        for zc in zero_crossings:
            loops_per_lobe.append(np.sum((peaks >= start_idx) & (peaks < zc)))
            start_idx = zc
        loops_per_lobe.append(np.sum(peaks >= start_idx))
        avg_loops_per_lobe = np.mean(loops_per_lobe)
    else:
        avg_loops_per_lobe = num_cycles

    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Total Cycles (Loops)", f"{num_cycles}", help="Number of distinct orbits around either of the attractor's lobes.")
    col_d2.metric("Avg. Cycle Time (Steps)", f"{avg_cycle_steps:.1f}", help="Average number of discrete time steps to complete one orbit.")
    col_d3.metric("Avg. Loops Before Switch", f"{avg_loops_per_lobe:.2f}", help="Mean consecutive orbits on a single side before crossing to the opposite lobe.")

    fig_hist, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, var in enumerate(['X', 'Y', 'Z']):
        edges = np.unique(np.quantile(df_base1[var], np.linspace(0, 1, 9))) 
        axes[i].hist(df_base1[var], bins=edges, edgecolor='black', color=['skyblue', 'lightgreen', 'salmon'][i])
        axes[i].set_title(f"{var} Distribution (Equal Mass, Bins=8)")
        axes[i].set_xlabel(f"{var} Value")
        axes[i].set_ylabel("Count")
    st.pyplot(fig_hist)
    plt.close()

    st.markdown("---")
    st.subheader("Delay Embedding Reconstruction (X-Variable)")
    tau = st.slider(r"Time Delay ($\tau$)", 1, 50, 15, help="Time lag used for Takens' state space reconstruction.")
    
    # Generate delayed vectors for 3D embedding
    x_vals = df_base1['Y'].values
    x_t = x_vals[:-2*tau]
    x_t_tau = x_vals[tau:-tau]
    x_t_2tau = x_vals[2*tau:]
    
    # Plot the reconstructed attractor
    fig_embed = plt.figure(figsize=(8, 6))
    ax_embed = fig_embed.add_subplot(111, projection='3d')
    ax_embed.plot(x_t, x_t_tau, x_t_2tau, lw=0.7, color='purple')
    ax_embed.set_xlabel('X(t)')
    ax_embed.set_ylabel(f'X(t + {tau})')
    ax_embed.set_zlabel(f'X(t + {2*tau})')
    ax_embed.set_title("3D Delay Reconstructed Phase Space from X")
    st.pyplot(fig_embed)
    plt.close(fig_embed)
    
    render_pairwise_analysis(dfs1, df_base1, pairs_xyz, num_points, max_lib_size, lib_step, max_lag, bin_steps, E_dim=3, key_suffix="lorenz")

with tab2:
    st.header("Coupled Logistic Map")
    st.latex(r"""
    \begin{align*}
    X_{t+1} &= X_t (r_x - r_x X_t - \beta_{xy} Y_t) \\
    Y_{t+1} &= Y_t (r_y - r_y Y_t - \beta_{yx} X_t)
    \end{align*}
    """)
    
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    with col_l1:
        rx_val = st.slider(r"$r_x$", 3.0, 4.0, 3.8, step=0.01, help="Growth rate for population X.")
    with col_l2:
        ry_val = st.slider(r"$r_y$", 3.0, 4.0, 3.5, step=0.01, help="Growth rate for population Y.")
    with col_l3:
        bxy_val = st.slider(r"$\beta_{xy}$ (Y->X)", 0.0, 0.5, 0.02, step=0.01, help="Strength of the influence Y has on X's growth.")
    with col_l4:
        byx_val = st.slider(r"$\beta_{yx}$ (X->Y)", 0.0, 0.5, 0.10, step=0.01, help="Strength of the influence X has on Y's growth.")

    dfs2 = generate_logistic_data(rx_val, ry_val, bxy_val, byx_val, num_points)
    render_pairwise_analysis(dfs2, dfs2[0], [("X", "Y")], num_points, max_lib_size, lib_step, max_lag, bin_steps, E_dim=2, key_suffix="logistic")

with tab3:
    st.header("Phase-Shifted Sinusoids")
    st.latex(r"""
    \begin{align*}
    X_t &= \sin(t) + \epsilon_x \\
    Y_t &= \sin(t + \phi_y) + \epsilon_y \\
    Z_t &= \sin(t + \phi_z) + \epsilon_z 
    \end{align*}
    """)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        py = st.slider(r"$\phi_y$ Phase Offset Y (Radians)", 0.0, 2*np.pi, np.pi/4, help="Amount to shift the Y sine wave horizontally.")
    with col_p2:
        pz = st.slider(r"$\phi_z$ Phase Offset Z (Radians)", 0.0, 2*np.pi, np.pi/2, help="Amount to shift the Z sine wave horizontally.")
        
    dfs3 = generate_phase_sinusoids(num_points, noise, py, pz)
    render_pairwise_analysis(dfs3, dfs3[0], pairs_xyz, num_points, max_lib_size, lib_step, max_lag, bin_steps, E_dim=2, key_suffix="phase")

with tab4:
    st.header("Frequency-Shifted Sinusoids")
    st.latex(r"""
    \begin{align*}
    X_t &= \sin(f_x t) + \epsilon_x \\
    Y_t &= \sin(f_y t) + \epsilon_y \\
    Z_t &= \sin(f_z t) + \epsilon_z 
    \end{align*}
    """)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        fx = st.slider(r"$f_x$ Frequency Multiplier X", 0.5, 3.0, 1.0, help="Cycles per time unit for X.")
    with col_f2:
        fy = st.slider(r"$f_y$ Frequency Multiplier Y", 0.5, 3.0, 1.5, help="Cycles per time unit for Y.")
    with col_f3:
        fz = st.slider(r"$f_z$ Frequency Multiplier Z", 0.5, 3.0, 2.0, help="Cycles per time unit for Z.")
        
    dfs4 = generate_frequency_sinusoids(num_points, noise, fx, fy, fz)
    render_pairwise_analysis(dfs4, dfs4[0], pairs_xyz, num_points, max_lib_size, lib_step, max_lag, bin_steps, E_dim=2, key_suffix="freq")