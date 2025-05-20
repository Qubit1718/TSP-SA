import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import math
import os
from scipy.signal import argrelextrema
from matplotlib.colors import LinearSegmentedColormap
import imageio
import io
from scipy.spatial.distance import cdist

# 1. Simulated Annealing Flowchart

def plot_simulated_annealing_flowchart():
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis('off')
    # Define box positions and text (vertical layout)
    boxes = [
        (0.5, 0.93, 'Start: Initial Solution', '#b3e5fc'),
        (0.5, 0.80, 'Repeat for each iteration', '#b3e5fc'),
        (0.5, 0.68, 'Generate new solution', '#b2dfdb'),
        (0.5, 0.56, 'Compute ΔE', '#b2dfdb'),
        (0.5, 0.44, 'ΔE < 0?', '#ffe082'),
        (0.25, 0.32, 'Accept new solution', '#c5e1a5'),
        (0.75, 0.32, 'Accept with probability\nexp(-ΔE/T)', '#ffccbc'),
        (0.5, 0.20, 'Update best solution', '#b2dfdb'),
        (0.5, 0.08, 'Decrease temperature', '#b3e5fc')
    ]
    # Draw boxes
    for (x, y, text, color) in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x-0.18, y-0.06), 0.36, 0.12,
                                             boxstyle="round,pad=0.04", fc=color, ec='#1976d2', lw=2))
        ax.text(x, y, text, ha='center', va='center', fontsize=13, fontweight='bold', color='#263238')
    # Draw arrows (vertical flow)
    arrowprops = dict(arrowstyle='->', color='#1976d2', lw=2)
    # Down arrows
    for i in range(1, 4):
        ax.annotate('', xy=(0.5, boxes[i][1]+0.06), xytext=(0.5, boxes[i-1][1]-0.06), arrowprops=arrowprops)
    # To decision
    ax.annotate('', xy=(0.5, boxes[4][1]+0.06), xytext=(0.5, boxes[3][1]-0.06), arrowprops=arrowprops)
    # Decision split
    ax.annotate('', xy=(0.32, boxes[5][1]+0.06), xytext=(0.5, boxes[4][1]-0.06), arrowprops=arrowprops)
    ax.annotate('', xy=(0.68, boxes[6][1]+0.06), xytext=(0.5, boxes[4][1]-0.06), arrowprops=arrowprops)
    # Both merge to update
    ax.annotate('', xy=(0.5, boxes[7][1]+0.06), xytext=(0.32, boxes[5][1]-0.06), arrowprops=arrowprops)
    ax.annotate('', xy=(0.5, boxes[7][1]+0.06), xytext=(0.68, boxes[6][1]-0.06), arrowprops=arrowprops)
    # Down to decrease temp
    ax.annotate('', xy=(0.5, boxes[8][1]+0.06), xytext=(0.5, boxes[7][1]-0.06), arrowprops=arrowprops)
    # Loop back to repeat
    ax.annotate('', xy=(0.5, boxes[1][1]+0.06), xytext=(0.5, boxes[8][1]-0.06), arrowprops=dict(arrowstyle='-[,widthB=5.0', color='#1976d2', lw=1.5, linestyle='dashed'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Simulated Annealing Algorithm Flow', fontsize=20, fontweight='bold', color='#1976d2', pad=20)
    plt.savefig('sa_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Metropolis Acceptance Plot

def plot_metropolis_acceptance():
    delta_E = np.linspace(-5, 5, 200)
    T_values = [0.5, 1.0, 2.0]
    plt.figure(figsize=(7, 5))
    for T in T_values:
        prob = np.where(delta_E < 0, 1, np.exp(-delta_E / T))
        plt.plot(delta_E, prob, label=f'T = {T}')
    plt.xlabel(r'Energy Difference $\Delta E$', fontsize=13)
    plt.ylabel('Acceptance Probability', fontsize=13)
    plt.title('Metropolis Acceptance Criterion', fontsize=15, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('metropolis_acceptance.png', dpi=300)
    plt.close()

# 3. Haversine Formula Globe Illustration

def plot_haversine_globe():
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(theta, np.ones_like(theta), color='black', lw=1)
    # Two points (cities)
    lat1, lon1 = 0.8*np.pi, 0.2*np.pi
    lat2, lon2 = 1.2*np.pi, 1.5*np.pi
    ax.plot([lon1, lon2], [1, 1], 'ro', markersize=10)
    # Only plot the great-circle arc (orange)
    arc_theta = np.linspace(lon1, lon2, 100)
    ax.plot(arc_theta, np.ones_like(arc_theta), color='orange', lw=4, label='Haversine (Great-circle) Path')
    ax.text(lon1, 1.12, 'City 1', ha='center', fontsize=12, color='red', fontweight='bold')
    ax.text(lon2, 1.12, 'City 2', ha='center', fontsize=12, color='red', fontweight='bold')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title('Haversine Formula: Great-circle Distance', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=12, frameon=True)
    plt.tight_layout()
    plt.savefig('haversine_globe.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Workflow Block Diagram

def plot_workflow_diagram():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    steps = [
        ('City Data', '#b2ebf2'),
        ('Distance Matrix\n(Haversine)', '#b2dfdb'),
        ('Simulated Annealing', '#ffe082'),
        ('Best Route', '#c5e1a5')
    ]
    box_width = 2.5
    box_height = 1.2
    spacing = 1.2
    y = 1.5
    for i, (label, color) in enumerate(steps):
        x = i * (box_width + spacing)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), box_width, box_height,
            boxstyle="round,pad=0.08", fc=color, ec='#006064', lw=2, mutation_scale=0.05))
        ax.text(x + box_width/2, y + box_height/2, label, ha='center', va='center', fontsize=15, fontweight='bold', color='#263238')
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + box_width, y + box_height/2), xytext=(x + box_width + spacing - 0.2, y + box_height/2),
                        arrowprops=dict(arrowstyle='-|>', lw=3, color='#00838f'))
    ax.set_xlim(-0.5, x + box_width + 0.5)
    ax.set_ylim(0, 4)
    ax.set_title('Workflow Overview', fontsize=20, fontweight='bold', color='#006064', pad=20)
    plt.savefig('workflow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sa_energy_landscape():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.linspace(0, 10, 400)
    y = 2 * np.sin(1.5 * x) + np.sin(0.5 * x + 1) + 0.3 * np.cos(3 * x)
    ax.plot(x, y, color='black', lw=2)
    # Mark local minima
    local_min_idx = argrelextrema(y, np.less)[0]
    ax.plot(x[local_min_idx], y[local_min_idx], 'go', markersize=8, label='Local Minima')
    # Mark global minimum
    global_min_idx = np.argmin(y)
    ax.plot(x[global_min_idx], y[global_min_idx], 'r*', markersize=18, label='Global Minimum')
    # Draw walker paths/arrows
    # High T: big jumps
    ax.annotate('', xy=(x[10], y[10]), xytext=(x[60], y[60]), arrowprops=dict(arrowstyle='->', lw=3, color='#1976d2'))
    ax.annotate('', xy=(x[60], y[60]), xytext=(x[120], y[120]), arrowprops=dict(arrowstyle='->', lw=3, color='#1976d2'))
    ax.annotate('', xy=(x[120], y[120]), xytext=(x[180], y[180]), arrowprops=dict(arrowstyle='->', lw=3, color='#1976d2'))
    # Low T: small steps
    ax.annotate('', xy=(x[250], y[250]), xytext=(x[260], y[260]), arrowprops=dict(arrowstyle='->', lw=2, color='#ff9800'))
    ax.annotate('', xy=(x[260], y[260]), xytext=(x[270], y[270]), arrowprops=dict(arrowstyle='->', lw=2, color='#ff9800'))
    ax.annotate('', xy=(x[270], y[270]), xytext=(x[global_min_idx], y[global_min_idx]), arrowprops=dict(arrowstyle='->', lw=2, color='#ff9800'))
    # Add temperature bar
    cmap = LinearSegmentedColormap.from_list('temp', ['#1976d2', '#ffeb3b', '#ff9800'])
    temp_bar = np.linspace(0, 1, 100).reshape(1, -1)
    ax.imshow(temp_bar, extent=[0, 10, min(y)-1.5, min(y)-1.2], aspect='auto', cmap=cmap)
    ax.text(0, min(y)-1.7, r'$T \gg 1$', fontsize=13, color='#1976d2', fontweight='bold', ha='left', va='center')
    ax.text(10, min(y)-1.7, r'$T = 0$', fontsize=13, color='#ff9800', fontweight='bold', ha='right', va='center')
    # Labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Simulated Annealing: Escaping Local Minima at High $T$', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig('sa_energy_landscape.png', dpi=300, bbox_inches='tight')
    plt.close()

def make_sa_animation():
    # Energy landscape
    x = np.linspace(0, 10, 400)
    y = 2 * np.sin(1.5 * x) + np.sin(0.5 * x + 1) + 0.3 * np.cos(3 * x)
    # Find local minima and global minimum
    local_min_idx = argrelextrema(y, np.less)[0]
    global_min_idx = np.argmin(y)
    # Simulated annealing parameters
    n_frames = 120
    T0 = 5.0
    Tmin = 0.001
    alpha = (Tmin / T0) ** (1 / n_frames)
    T = T0
    # Start at a random position
    np.random.seed(42)
    pos_idx = np.random.randint(0, len(x))
    path_x = [x[pos_idx]]
    path_y = [y[pos_idx]]
    frames = []
    if not os.path.exists('sa_gif'):
        os.makedirs('sa_gif')
    for frame in range(n_frames):
        fig, ax = plt.subplots(figsize=(12, 5.5), dpi=200)  # Higher resolution
        ax.plot(x, y, color='#263238', lw=2)
        # Mark local minima
        ax.plot(x[local_min_idx], y[local_min_idx], 'o', color='#e57373', markersize=8, label='Local Minimum')
        # Mark global minimum
        ax.plot(x[global_min_idx], y[global_min_idx], 'o', color='#1976d2', markersize=10, label='Global Minimum')
        # Path history
        if len(path_x) > 1:
            ax.plot(path_x, path_y, color='#90caf9', lw=2, alpha=0.7, zorder=1, label='Path History' if frame == 0 else None)
        # Propose a new position (allow large jumps at high T)
        jump_scale = int(10 + 100 * (T / T0))
        step = np.random.randint(-jump_scale, jump_scale+1)
        new_idx = np.clip(pos_idx + step, 0, len(x) - 1)
        # Show attempted move
        ax.plot([x[pos_idx], x[new_idx]], [y[pos_idx], y[new_idx]], '--', color='#ffb300', lw=2, alpha=0.5, zorder=2)
        ax.plot(x[new_idx], y[new_idx], 'o', color='#ffb300', markersize=10, alpha=0.5, zorder=2, label='Attempted Move' if frame == 0 else None)
        delta_E = y[new_idx] - y[pos_idx]
        accepted = False
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            pos_idx = new_idx
            accepted = True
            path_x.append(x[pos_idx])
            path_y.append(y[pos_idx])
        # Mark current position
        # On the last frame, force the walker to the global minimum for illustration
        if frame == n_frames - 1:
            pos_idx = global_min_idx
            path_x.append(x[pos_idx])
            path_y.append(y[pos_idx])
        ax.plot(x[pos_idx], y[pos_idx], 'o', color='#43a047', markersize=13, label='Current Solution' if frame == 0 else None, zorder=5)
        # Temperature display
        ax.text(0.98, 0.98, f'T = {T:.2f}', ha='right', va='top', fontsize=18, color='#1976d2', fontweight='bold', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='#1976d2', boxstyle='round,pad=0.2'))
        # Legend only on first frame
        if frame == 0:
            ax.legend(loc='upper right', fontsize=15, frameon=True)
        # Labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Simulated Annealing: Exploration and Convergence', fontsize=22, fontweight='bold', pad=16)
        plt.tight_layout()
        # Save frame as PNG for Overleaf
        png_path = f'sa_gif/frame_{frame:03d}.png'
        plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
        # Also save to GIF buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        frames.append(imageio.v2.imread(buf))
        plt.close()
        T *= alpha
    # Save GIF
    imageio.mimsave('sa_animation.gif', frames, fps=8)

def plot_metropolis_acceptance_viz():
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    x = np.linspace(-3, 3, 400)
    y = np.sin(x) + 0.2 * x
    ax.plot(x, y, color='black', lw=3)
    # Current and proposed points
    x_current, y_current = -1, np.sin(-1) + 0.2 * -1
    x_proposed, y_proposed = 0.3, np.sin(0.3) + 0.2 * 0.3
    ax.scatter([x_current], [y_current], color='#43a047', s=180, zorder=5, label='Current')
    ax.scatter([x_proposed], [y_proposed], color='#ffb300', s=180, zorder=5, label='Proposed')
    # Draw arrow
    ax.annotate('', xy=(x_proposed, y_proposed), xytext=(x_current, y_current), arrowprops=dict(arrowstyle='->', lw=3, color='#ffb300'))
    # Delta E label
    ax.text((x_current + x_proposed)/2, (y_current + y_proposed)/2 + 0.2, r'$\Delta E > 0$', fontsize=22, ha='center', color='black')
    ax.legend(loc='upper right', fontsize=18, frameon=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Metropolis Acceptance Visualization', fontsize=28, pad=16)
    plt.tight_layout()
    plt.savefig('metropolis_acceptance_viz.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_sa_exploration_convergence():
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    x = np.linspace(-3, 3, 400)
    y = np.sin(x) + 0.2 * x
    # Find local minima and global minimum
    local_min_idx = argrelextrema(y, np.less)[0]
    global_min_idx = np.argmin(y)
    ax.plot(x, y, color='#263238', lw=3)
    # Local minima
    ax.scatter(x[local_min_idx], y[local_min_idx], color='#e57373', s=120, zorder=5, label='Local Minimum')
    # Global minimum
    ax.scatter(x[global_min_idx], y[global_min_idx], color='#1976d2', s=140, zorder=5, label='Global Minimum')
    # Current and attempted
    x_current, y_current = -1, np.sin(-1) + 0.2 * -1
    x_attempt, y_attempt = 0.3, np.sin(0.3) + 0.2 * 0.3
    ax.scatter([x_current], [y_current], color='#43a047', s=180, zorder=6, label='Current Solution')
    ax.scatter([x_attempt], [y_attempt], color='#ffb300', s=180, zorder=6, label='Attempted Move')
    # Dashed arrow
    ax.plot([x_current, x_attempt], [y_current, y_attempt], '--', color='#ffb300', lw=3, alpha=0.7)
    ax.legend(loc='upper right', fontsize=18, frameon=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Simulated Annealing: Exploration and Convergence', fontsize=28, pad=16)
    plt.tight_layout()
    plt.savefig('sa_exploration_convergence.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_tsp_heuristic_vs_sa():
    np.random.seed(123)
    n_cities = 8
    # Random city coordinates in a unit square
    coords = np.random.rand(n_cities, 2)
    city_labels = [chr(65+i) for i in range(n_cities)]
    dist_matrix = cdist(coords, coords)

    # --- Nearest Neighbor Heuristic ---
    nn_tour = [0]
    unvisited = set(range(1, n_cities))
    while unvisited:
        last = nn_tour[-1]
        next_city = min(unvisited, key=lambda j: dist_matrix[last, j])
        nn_tour.append(next_city)
        unvisited.remove(next_city)
    nn_tour.append(0)  # return to start
    nn_dist = sum(dist_matrix[nn_tour[i], nn_tour[i+1]] for i in range(n_cities))

    # --- Simulated Annealing (simple version) ---
    def total_dist(tour):
        return sum(dist_matrix[tour[i], tour[i+1]] for i in range(n_cities))
    sa_tour = list(range(n_cities))
    np.random.shuffle(sa_tour)
    sa_tour.append(sa_tour[0])
    best_tour = sa_tour.copy()
    best_dist = total_dist(sa_tour)
    T = 1.0
    Tmin = 1e-4
    alpha = 0.995
    for _ in range(2000):
        i, j = np.sort(np.random.choice(n_cities, 2, replace=False))
        new_tour = sa_tour[:i] + sa_tour[i:j+1][::-1] + sa_tour[j+1:]
        new_tour[-1] = new_tour[0]
        d_new = total_dist(new_tour)
        d_old = total_dist(sa_tour)
        if d_new < d_old or np.random.rand() < np.exp(-(d_new-d_old)/T):
            sa_tour = new_tour
            if d_new < best_dist:
                best_tour = new_tour.copy()
                best_dist = d_new
        T *= alpha
    # --- Plotting function ---
    def plot_tour(tour, coords, city_labels, total_dist, title, filename, color):
        fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
        ax.scatter(coords[:,0], coords[:,1], s=180, color='#1976d2', zorder=5)
        for i, (x, y) in enumerate(coords):
            ax.text(x, y+0.025, city_labels[i], fontsize=18, ha='center', va='bottom', fontweight='bold', color='#263238')
        # Draw arrows for the tour
        for i in range(len(tour)-1):
            ax.annotate('', xy=coords[tour[i+1]], xytext=coords[tour[i]],
                        arrowprops=dict(arrowstyle='->', lw=3, color=color, alpha=0.8), zorder=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(title, fontsize=22, fontweight='bold', pad=16)
        # Place total distance inside the plot (top left)
        ax.text(0.02, 0.98, f'Total Distance: {total_dist:.2f}',
                fontsize=18, ha='left', va='top', color=color,
                transform=ax.transAxes, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))
        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
    plot_tour(nn_tour, coords, city_labels, nn_dist, 'Nearest Neighbor Solution', 'tsp_nn_example.png', '#ff7043')
    plot_tour(best_tour, coords, city_labels, best_dist, 'Simulated Annealing Solution', 'tsp_sa_example.png', '#43a047')

if __name__ == "__main__":
    if not os.path.exists('output'):
        os.makedirs('output')
    plot_metropolis_acceptance_viz()
    plot_sa_exploration_convergence()
    plot_tsp_heuristic_vs_sa()
    plot_sa_energy_landscape()
    plot_simulated_annealing_flowchart()
    plot_metropolis_acceptance()
    plot_haversine_globe()
    plot_workflow_diagram()
    make_sa_animation() 