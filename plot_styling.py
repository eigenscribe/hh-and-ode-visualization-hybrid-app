import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import numpy as np

def get_theme_colors():
    """Get color palette matching the CSS theme"""
    return {
        'bg': '#0d1117',
        'surface': '#161b22',
        'primary': '#14b5ff',
        'secondary': '#f78166',
        'text': '#f5f5f5',  # whitesmoke
        'muted': '#8b949e',
        'link': '#7652f5',
        'link_hover': '#00e8ff',
        'math': '#00f5db',
        'accent': '#22f792',
        'chart_3': '#22f792',
        'chart_4': '#7652f5',
        'chart_5': '#00f5db'
    }

def get_gradients():
    """Get gradient definitions matching the CSS theme"""
    return {
        'light_to_dark_blue': ['#00e8ff', '#14b5ff', '#3a98ff', '#0070eb'],
        'cyan_to_blue': ['#00ffee', '#0a95eb'],
        'green_to_blue': ['#22f792', '#27aaff'],
        'blue_to_purple': ['#00e8ff', '#14b5ff', '#5280ff', '#7952f5']
    }

def setup_plot_theme():
    """Configure matplotlib to match the theme"""
    colors = get_theme_colors()
    
    # Set the style parameters
    plt.style.use('dark_background')
    
    # Custom rcParams
    mpl.rcParams.update({
        # Figure settings
        'figure.facecolor': colors['bg'],
        'figure.edgecolor': colors['bg'],
        'savefig.facecolor': colors['bg'],
        'savefig.edgecolor': colors['bg'],
        
        # Axes settings
        'axes.facecolor': colors['bg'],
        'axes.edgecolor': colors['muted'],
        'axes.labelcolor': colors['text'],
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'axes.grid.alpha': 0.3,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        
        # Grid settings
        'grid.color': colors['muted'],
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        
        # Tick settings
        'xtick.color': colors['text'],
        'ytick.color': colors['text'],
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Font settings - Use system fonts since we can't load custom fonts easily
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.size': 11,
        
        # Line settings
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        
        # Legend settings
        'legend.facecolor': colors['surface'],
        'legend.edgecolor': colors['muted'],
        'legend.fontsize': 10,
        'legend.framealpha': 0.9,
        
        # Text settings
        'text.color': colors['text'],
        
        # Math text settings (for equations)
        'mathtext.default': 'regular',
        'mathtext.fontset': 'cm',  # Computer Modern for math
    })

def create_gradient_colormap(gradient_name='light_to_dark_blue', n_colors=256):
    """Create a matplotlib colormap from theme gradients"""
    gradients = get_gradients()
    
    if gradient_name not in gradients:
        gradient_name = 'light_to_dark_blue'
    
    colors = gradients[gradient_name]
    
    # Convert hex colors to RGB
    rgb_colors = []
    for color in colors:
        # Remove # if present
        color = color.lstrip('#')
        # Convert to RGB tuple (0-1 range)
        rgb = tuple(int(color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        rgb_colors.append(rgb)
    
    # Create colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(gradient_name, rgb_colors, N=n_colors)
    
    return cmap

def apply_theme_to_axis(ax, title=None, xlabel=None, ylabel=None):
    """Apply theme styling to a matplotlib axis"""
    colors = get_theme_colors()
    
    # Set face and edge colors
    ax.set_facecolor(colors['bg'])
    
    # Configure spines
    for spine in ax.spines.values():
        spine.set_color(colors['muted'])
        spine.set_linewidth(1.5)
    
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Configure ticks
    ax.tick_params(colors=colors['text'], which='both')
    
    # Configure grid
    ax.grid(True, alpha=0.3, color=colors['muted'])
    
    # Set labels and title
    if title:
        ax.set_title(title, color=colors['text'], fontweight='bold', fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, color=colors['text'], fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=colors['text'], fontsize=12)

def get_color_cycle():
    """Get a color cycle that matches the theme"""
    colors = get_theme_colors()
    return [
        colors['primary'],
        colors['secondary'],
        colors['accent'],
        colors['chart_3'],
        colors['chart_4'],
        colors['chart_5'],
        colors['link'],
        colors['math']
    ]

def style_math_text(text, color='math'):
    """Style mathematical text with appropriate font and color"""
    colors = get_theme_colors()
    color_val = colors.get(color, colors['math'])
    
    # For matplotlib, we can use mathtext
    return f'${text}$'

def create_themed_figure(figsize=(10, 6), nrows=1, ncols=1):
    """Create a figure with theme styling applied"""
    setup_plot_theme()
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # Apply theme to axes
    if nrows * ncols == 1:
        axes = [axes]  # Make it a list for consistency
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for ax in axes:
        apply_theme_to_axis(ax)
    
    # Return single axis if only one subplot
    if len(axes) == 1:
        return fig, axes[0]
    else:
        return fig, axes
