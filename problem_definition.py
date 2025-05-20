import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.patches import Circle, Arrow
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
import os

def create_problem_illustration():
    # Use a robust style
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        plt.style.use('ggplot')
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    
    # Create a 2x2 grid for different aspects of the problem
    gs = fig.add_gridspec(2, 2)
    
    # 1. Basic TSP Illustration
    ax1 = fig.add_subplot(gs[0, 0])
    G = nx.Graph()
    cities = {'A': (0.2, 0.8), 'B': (0.8, 0.8), 'C': (0.2, 0.2), 'D': (0.8, 0.2)}
    for city, pos in cities.items():
        G.add_node(city, pos=pos)
    edges = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]
    G.add_edges_from(edges)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_nodes(G, pos, node_color='#4ECDC4', node_size=1200, ax=ax1, edgecolors='black', linewidths=2)
    nx.draw_networkx_edges(G, pos, edge_color='#888', width=2.5, alpha=0.7, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', ax=ax1)
    ax1.set_title('Basic TSP Structure', fontsize=20, fontweight='bold', pad=18)
    ax1.axis('off')
    
    # 2. NP-Hard Complexity
    ax2 = fig.add_subplot(gs[0, 1])
    n_values = np.array([5, 10, 15, 20, 25, 30, 40, 50, 75, 100])
    factorial = np.array([120, 3628800, 1.3e12, 2.4e18, 1.6e25, 2.7e32, 8.2e47, 3.0e64, 2.5e109, 9.3e157])
    ax2.set_yscale('log')
    ax2.plot(n_values, factorial, 'o-', color='#FF6B6B', linewidth=3, markersize=10)
    ax2.set_xlabel('Number of Cities (n)', fontsize=16)
    ax2.set_ylabel('Possible Routes (n!)', fontsize=16)
    ax2.set_title('Combinatorial Explosion\n(Logarithmic Scale)', fontsize=20, fontweight='bold', pad=18)
    ax2.grid(True, linestyle='--', alpha=0.7)
    for i, (n, f) in enumerate(zip(n_values, factorial)):
        if i in [0, 4, 9]:
            ax2.annotate(f'n={n}\n{n}! ≈ {f:.1e}', xy=(n, f), xytext=(10, 10), textcoords='offset points', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))
    
    # 3. Real-world Application
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(-125, -65)
    ax3.set_ylim(25, 50)
    ax3.set_title('Real-world Application', fontsize=20, fontweight='bold', pad=18)
    cities = {'Boston': (-71.0589, 42.3601), 'NYC': (-74.0060, 40.7128), 'Chicago': (-87.6298, 41.8781), 'LA': (-118.2437, 34.0522)}
    for city, (lon, lat) in cities.items():
        ax3.plot(lon, lat, 'o', color='#4ECDC4', markersize=14, markeredgecolor='black', markeredgewidth=2)
        ax3.text(lon, lat, city, fontsize=14, ha='right', va='bottom', fontweight='bold', color='#222')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.set_xlabel('Longitude', fontsize=16)
    ax3.set_ylabel('Latitude', fontsize=16)
    
    # 4. Optimization Challenge (Local vs Global Optima)
    ax4 = fig.add_subplot(gs[1, 1])
    x = np.linspace(-5, 5, 500)
    y = np.sin(x) + 0.5 * np.sin(2*x) + 0.25 * np.sin(3*x) + 0.1 * np.cos(5*x)
    ax4.plot(x, y, color='#45B7D1', linewidth=3, label='Cost Function')
    # Find local minima
    from scipy.signal import argrelextrema
    local_min_idx = argrelextrema(y, np.less)[0]
    global_min_idx = np.argmin(y)
    ax4.plot(x[local_min_idx], y[local_min_idx], 'ro', label='Local Minima', markersize=8)
    ax4.plot(x[global_min_idx], y[global_min_idx], 'g*', label='Global Minimum', markersize=18)
    ax4.set_title('Local vs Global Optima', fontsize=20, fontweight='bold', pad=18)
    ax4.set_xlabel('Solution Space', fontsize=16)
    ax4.set_ylabel('Cost Function', fontsize=16)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(fontsize=13, loc='upper right', frameon=True, facecolor='white', edgecolor='gray')
    # Add annotation arrows
    ax4.annotate('Local Minimum', xy=(x[local_min_idx[0]], y[local_min_idx[0]]), xytext=(x[local_min_idx[0]]-2, y[local_min_idx[0]]+1), arrowprops=dict(facecolor='red', arrowstyle='->'), fontsize=13, color='red', fontweight='bold')
    ax4.annotate('Global Minimum', xy=(x[global_min_idx], y[global_min_idx]), xytext=(x[global_min_idx]+1, y[global_min_idx]-1), arrowprops=dict(facecolor='green', arrowstyle='->'), fontsize=13, color='green', fontweight='bold')
    # Main title
    fig.suptitle('The Traveling Salesman Problem: A Complex Optimization Challenge', fontsize=24, y=1.04, fontweight='bold')
    # Save the figure
    plt.savefig('problem_definition.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_problem_text():
    # Create a text file with the problem definition
    text = """The Traveling Salesman Problem (TSP) is a fundamental challenge in combinatorial optimization that has fascinated mathematicians and computer scientists for decades. At its core, the problem is deceptively simple: given a set of cities and the distances between them, find the shortest possible route that visits each city exactly once and returns to the starting point.

Key Characteristics:
1. NP-Hard Complexity: The number of possible routes grows factorially with the number of cities, making exact solutions computationally infeasible for large instances.
2. Real-world Applications: From logistics and delivery routing to circuit board drilling and DNA sequencing, TSP has numerous practical applications.
3. Optimization Challenge: The problem involves finding the global optimum in a solution space with many local optima, making it a perfect testbed for heuristic algorithms.

In our implementation, we focus on a real-world instance of TSP involving 19 major U.S. cities, with Boston as both the starting and ending point. This practical application demonstrates the challenges and solutions in modern optimization problems."""
    
    with open('problem_definition.txt', 'w') as f:
        f.write(text)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Generate the problem definition visualization
    create_problem_illustration()
    create_problem_text() 