import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from plot_styling import setup_plot_theme, get_theme_colors

class ODEAnalyzer:
    """ODE phase portrait and bifurcation analysis tools"""
    
    def __init__(self):
        self.colors = get_theme_colors()
    
    def van_der_pol(self, t, y, mu):
        """Van der Pol oscillator: d²x/dt² - μ(1-x²)dx/dt + x = 0"""
        x, dx_dt = y
        d2x_dt2 = mu * (1 - x**2) * dx_dt - x
        return [dx_dt, d2x_dt2]
    
    def lotka_volterra(self, t, y, alpha, beta, gamma, delta):
        """Lotka-Volterra predator-prey model"""
        x, y_prey = y  # x = prey, y = predator
        dx_dt = alpha * x - beta * x * y_prey
        dy_dt = delta * x * y_prey - gamma * y_prey
        return [dx_dt, dy_dt]
    
    def duffing_oscillator(self, t, y, a, b, c):
        """Duffing oscillator: d²x/dt² + c*dx/dt + a*x + b*x³ = 0"""
        x, dx_dt = y
        d2x_dt2 = -c * dx_dt - a * x - b * x**3
        return [dx_dt, d2x_dt2]
    
    def pendulum(self, t, y, b, g):
        """Damped pendulum: d²θ/dt² + b*dθ/dt + (g/l)*sin(θ) = 0"""
        theta, dtheta_dt = y
        d2theta_dt2 = -b * dtheta_dt - g * np.sin(theta)
        return [dtheta_dt, d2theta_dt2]
    
    def get_system_function(self, system_type, params):
        """Get the appropriate ODE system function"""
        if system_type == "Van der Pol Oscillator":
            return lambda t, y: self.van_der_pol(t, y, params['mu'])
        elif system_type == "Lotka-Volterra":
            return lambda t, y: self.lotka_volterra(
                t, y, params['alpha'], params['beta'], params['gamma'], params['delta']
            )
        elif system_type == "Duffing Oscillator":
            return lambda t, y: self.duffing_oscillator(
                t, y, params['a'], params['b'], params['c']
            )
        elif system_type == "Pendulum":
            return lambda t, y: self.pendulum(t, y, params['b'], params['g'])
        else:
            # Default system for demonstration
            return lambda t, y: [y[1], -y[0]]
    
    def plot_phase_portrait(self, system_type, params, x_range, y_range, n_trajectories):
        """Generate phase portrait for 2D ODE system"""
        setup_plot_theme()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get system function
        system_func = self.get_system_function(system_type, params)
        
        # Create meshgrid for vector field
        x_grid = np.linspace(-x_range, x_range, 20)
        y_grid = np.linspace(-y_range, y_range, 20)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Compute vector field
        DX = np.zeros_like(X)
        DY = np.zeros_like(Y)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                derivatives = system_func(0, [X[i,j], Y[i,j]])
                DX[i,j] = derivatives[0]
                DY[i,j] = derivatives[1]
        
        # Normalize vectors for better visualization
        M = np.sqrt(DX**2 + DY**2)
        M[M == 0] = 1  # Avoid division by zero
        DX_norm = DX / M
        DY_norm = DY / M
        
        # Plot vector field
        ax.quiver(X, Y, DX_norm, DY_norm, 
                 alpha=0.6, color=self.colors['muted'], width=0.003)
        
        # Generate trajectories
        colors_cycle = [
            self.colors['primary'], self.colors['secondary'], 
            self.colors['accent'], self.colors['chart_3'], 
            self.colors['chart_5']
        ]
        
        for i in range(n_trajectories):
            # Random initial conditions
            x0 = np.random.uniform(-x_range*0.8, x_range*0.8)
            y0 = np.random.uniform(-y_range*0.8, y_range*0.8)
            
            # Solve ODE
            sol = solve_ivp(
                system_func, 
                [0, 20], 
                [x0, y0], 
                dense_output=True, 
                rtol=1e-8
            )
            
            if sol.success:
                t_plot = np.linspace(0, sol.t[-1], 1000)
                trajectory = sol.sol(t_plot)
                
                color = colors_cycle[i % len(colors_cycle)]
                ax.plot(trajectory[0], trajectory[1], 
                       color=color, linewidth=2, alpha=0.8)
                
                # Mark starting point
                ax.plot(x0, y0, 'o', color=color, markersize=6, 
                       markeredgecolor='white', markeredgewidth=1)
        
        # Styling
        ax.set_xlim(-x_range, x_range)
        ax.set_ylim(-y_range, y_range)
        ax.set_xlabel('x', fontfamily='Aclonica', fontsize=14)
        ax.set_ylabel('y', fontfamily='Aclonica', fontsize=14)
        ax.set_title(f'Phase Portrait: {system_type}', 
                    fontfamily='Aclonica', fontweight='bold', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='white', linestyle='-', alpha=0.3)
        
        return fig
    
    def logistic_map_bifurcation(self, r_range, resolution):
        """Compute bifurcation diagram for logistic map"""
        r_min, r_max = r_range
        r_values = np.linspace(r_min, r_max, resolution)
        
        # Parameters for iteration
        n_skip = 1000  # Skip transient behavior
        n_plot = 100   # Points to plot after transient
        
        r_plot = []
        x_plot = []
        
        for r in r_values:
            # Initial condition
            x = 0.5
            
            # Skip transients
            for _ in range(n_skip):
                x = r * x * (1 - x)
            
            # Collect points
            for _ in range(n_plot):
                x = r * x * (1 - x)
                r_plot.append(r)
                x_plot.append(x)
        
        return np.array(r_plot), np.array(x_plot)
    
    def pitchfork_bifurcation(self, r_range, resolution):
        """Compute pitchfork bifurcation: dx/dt = rx - x³"""
        r_values = np.linspace(r_range[0], r_range[1], resolution)
        
        # Analytical solution for equilibrium points
        r_plot = []
        x_plot = []
        
        for r in r_values:
            if r <= 0:
                # Only stable equilibrium at x = 0
                r_plot.append(r)
                x_plot.append(0)
            else:
                # Three equilibria: x = 0, ±√r
                equilibria = [0, np.sqrt(r), -np.sqrt(r)]
                for eq in equilibria:
                    r_plot.append(r)
                    x_plot.append(eq)
        
        return np.array(r_plot), np.array(x_plot)
    
    def hopf_bifurcation_system(self, mu_range, resolution):
        """Compute Hopf bifurcation for system: dx/dt = μx - y - x(x²+y²), dy/dt = x + μy - y(x²+y²)"""
        mu_values = np.linspace(mu_range[0], mu_range[1], resolution)
        
        mu_plot = []
        amplitude_plot = []
        
        for mu in mu_values:
            if mu <= 0:
                # Stable equilibrium at origin
                mu_plot.append(mu)
                amplitude_plot.append(0)
            else:
                # Limit cycle with amplitude √μ
                mu_plot.append(mu)
                amplitude_plot.append(np.sqrt(mu))
                mu_plot.append(mu)
                amplitude_plot.append(-np.sqrt(mu))
        
        return np.array(mu_plot), np.array(amplitude_plot)
    
    def saddle_node_bifurcation(self, r_range, resolution):
        """Compute saddle-node bifurcation: dx/dt = r + x²"""
        r_values = np.linspace(r_range[0], r_range[1], resolution)
        
        r_plot = []
        x_plot = []
        
        for r in r_values:
            if r <= 0:
                # Two equilibria: x = ±√(-r)
                equilibria = [np.sqrt(-r), -np.sqrt(-r)]
                for eq in equilibria:
                    r_plot.append(r)
                    x_plot.append(eq)
            # For r > 0, no real equilibria
        
        return np.array(r_plot), np.array(x_plot)
    
    def plot_bifurcation_diagram(self, bifurc_type, param_range, resolution):
        """Generate bifurcation diagram"""
        setup_plot_theme()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if bifurc_type == "Logistic Map":
            param_vals, var_vals = self.logistic_map_bifurcation(param_range, resolution)
            param_label = 'r'
            var_label = 'x'
            title = 'Logistic Map Bifurcation Diagram'
            ax.scatter(param_vals, var_vals, s=0.1, c=self.colors['primary'], alpha=0.6)
            
        elif bifurc_type == "Pitchfork Bifurcation":
            param_vals, var_vals = self.pitchfork_bifurcation(param_range, resolution)
            param_label = 'r'
            var_label = 'x*'
            title = 'Pitchfork Bifurcation Diagram'
            ax.plot(param_vals, var_vals, '.', color=self.colors['secondary'], markersize=2)
            
        elif bifurc_type == "Hopf Bifurcation":
            param_vals, var_vals = self.hopf_bifurcation_system(param_range, resolution)
            param_label = 'μ'
            var_label = 'Amplitude'
            title = 'Hopf Bifurcation Diagram'
            ax.plot(param_vals, var_vals, '.', color=self.colors['accent'], markersize=2)
            
        elif bifurc_type == "Saddle-Node Bifurcation":
            param_vals, var_vals = self.saddle_node_bifurcation(param_range, resolution)
            param_label = 'r'
            var_label = 'x*'
            title = 'Saddle-Node Bifurcation Diagram'
            ax.plot(param_vals, var_vals, '.', color=self.colors['chart_3'], markersize=2)
        
        # Styling
        ax.set_xlabel(param_label, fontfamily='Aclonica', fontsize=14)
        ax.set_ylabel(var_label, fontfamily='Aclonica', fontsize=14)
        ax.set_title(title, fontfamily='Aclonica', fontweight='bold', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at bifurcation point if applicable
        if bifurc_type in ["Pitchfork Bifurcation", "Hopf Bifurcation", "Saddle-Node Bifurcation"]:
            ax.axvline(x=0, color=self.colors['chart_5'], linestyle='--', alpha=0.8, linewidth=2)
        
        return fig
