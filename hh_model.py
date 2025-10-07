import numpy as np
from scipy.integrate import solve_ivp

class HodgkinHuxleySimulator:
    """Hodgkin-Huxley neuron model simulator"""
    
    def __init__(self, C_m=1.0, g_Na=120, g_K=36, g_L=0.3, E_Na=50, E_K=-77, E_L=-54.387):
        """
        Initialize HH model parameters
        
        Parameters:
        - C_m: membrane capacitance (μF/cm²)
        - g_Na, g_K, g_L: maximum conductances (mS/cm²)
        - E_Na, E_K, E_L: reversal potentials (mV)
        """
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
    
    def alpha_n(self, V):
        """Potassium activation rate constant"""
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    
    def beta_n(self, V):
        """Potassium deactivation rate constant"""
        return 0.125 * np.exp(-(V + 65) / 80)
    
    def alpha_m(self, V):
        """Sodium activation rate constant"""
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    
    def beta_m(self, V):
        """Sodium deactivation rate constant"""
        return 4 * np.exp(-(V + 65) / 18)
    
    def alpha_h(self, V):
        """Sodium inactivation rate constant"""
        return 0.07 * np.exp(-(V + 65) / 20)
    
    def beta_h(self, V):
        """Sodium deinactivation rate constant"""
        return 1 / (1 + np.exp(-(V + 35) / 10))
    
    def hh_derivatives(self, t, y, I_ext):
        """
        HH model differential equations
        
        State variables: [V, n, m, h]
        """
        V, n, m, h = y
        
        # Rate constants
        an = self.alpha_n(V)
        bn = self.beta_n(V)
        am = self.alpha_m(V)
        bm = self.beta_m(V)
        ah = self.alpha_h(V)
        bh = self.beta_h(V)
        
        # Ionic currents
        I_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        I_K = self.g_K * n**4 * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)
        
        # Derivatives
        dV_dt = (I_ext - I_Na - I_K - I_L) / self.C_m
        dn_dt = an * (1 - n) - bn * n
        dm_dt = am * (1 - m) - bm * m
        dh_dt = ah * (1 - h) - bh * h
        
        return [dV_dt, dn_dt, dm_dt, dh_dt]
    
    def simulate(self, I_ext, duration, dt=0.01):
        """
        Run HH simulation
        
        Parameters:
        - I_ext: applied current (μA/cm²)
        - duration: simulation time (ms)
        - dt: time step (ms)
        
        Returns:
        - Dictionary with time series data
        """
        # Time vector
        t_span = (0, duration)
        t_eval = np.arange(0, duration + dt, dt)
        
        # Initial conditions (resting state)
        V_rest = -65  # mV
        n_inf = self.alpha_n(V_rest) / (self.alpha_n(V_rest) + self.beta_n(V_rest))
        m_inf = self.alpha_m(V_rest) / (self.alpha_m(V_rest) + self.beta_m(V_rest))
        h_inf = self.alpha_h(V_rest) / (self.alpha_h(V_rest) + self.beta_h(V_rest))
        
        y0 = [V_rest, n_inf, m_inf, h_inf]
        
        # Solve ODE
        sol = solve_ivp(
            lambda t, y: self.hh_derivatives(t, y, I_ext),
            t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6
        )
        
        return {
            't': sol.t,
            'V': sol.y[0],
            'n': sol.y[1],
            'm': sol.y[2],
            'h': sol.y[3]
        }
    
    def compute_currents(self, results):
        """Compute ionic currents from simulation results"""
        V = results['V']
        n = results['n']
        m = results['m']
        h = results['h']
        
        I_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        I_K = self.g_K * n**4 * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)
        
        return {
            'I_Na': I_Na,
            'I_K': I_K,
            'I_L': I_L
        }
    
    def simulate_pulse_train(self, pulse_params, total_duration, dt=0.01):
        """
        Run HH simulation with pulse train stimulus
        
        Parameters:
        - pulse_params: dict with 'n_pulses', 'start', 'duration', 'interval', 'amplitude'
        - total_duration: total simulation time (ms)
        - dt: time step (ms)
        
        Returns:
        - Dictionary with time series data and pulse timing info
        """
        # Time vector
        t_span = (0, total_duration)
        t_eval = np.arange(0, total_duration + dt, dt)
        
        # Initial conditions (resting state)
        V_rest = -65  # mV
        n_inf = self.alpha_n(V_rest) / (self.alpha_n(V_rest) + self.beta_n(V_rest))
        m_inf = self.alpha_m(V_rest) / (self.alpha_m(V_rest) + self.beta_m(V_rest))
        h_inf = self.alpha_h(V_rest) / (self.alpha_h(V_rest) + self.beta_h(V_rest))
        
        y0 = [V_rest, n_inf, m_inf, h_inf]
        
        # Create pulse train current function
        def pulse_train_current(t):
            I = 0
            for i in range(pulse_params['n_pulses']):
                pulse_start = pulse_params['start'] + i * pulse_params['interval']
                pulse_end = pulse_start + pulse_params['duration']
                if pulse_start <= t <= pulse_end:
                    I = pulse_params['amplitude']
                    break
            return I
        
        # Solve ODE with pulse train
        sol = solve_ivp(
            lambda t, y: self.hh_derivatives(t, y, pulse_train_current(t)),
            t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6
        )
        
        # Store pulse timing for visualization
        pulse_times = []
        for i in range(pulse_params['n_pulses']):
            pulse_start = pulse_params['start'] + i * pulse_params['interval']
            pulse_end = pulse_start + pulse_params['duration']
            pulse_times.append((pulse_start, pulse_end))
        
        return {
            't': sol.t,
            'V': sol.y[0],
            'n': sol.y[1],
            'm': sol.y[2],
            'h': sol.y[3],
            'pulse_times': pulse_times,
            'pulse_amplitude': pulse_params['amplitude']
        }
    
    def nullclines(self, V_range=(-100, 50), n_points=1000):
        """Compute V and n nullclines for phase plane analysis"""
        V = np.linspace(V_range[0], V_range[1], n_points)
        
        # V-nullcline (dV/dt = 0)
        # This requires solving for the equilibrium condition
        # For simplicity, we'll compute it numerically
        
        # n-nullcline (dn/dt = 0)
        n_null = self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))
        
        return V, n_null
