import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import json
import os
import imageio.v2 as imageio

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Custom CSS to ensure full-width display
st.markdown("""
    <style>
        .stTitle {
            font-size: 2.5rem !important;
            padding-bottom: 2rem !important;
            text-align: center !important;
        }
        .stNumberInput > div > div > input {
            text-align: center;
        }
        .stButton > button {
            width: 100%;
            margin-top: 1rem;
            margin-bottom: 2rem;
        }
        /* Ensure images take full width */
        .element-container img {
            width: 100% !important;
            max-width: 1500px !important;
            margin: 0 auto !important;
            display: block !important;
        }
        /* Adjust container width */
        .block-container {
            max-width: 95% !important;
            padding-top: 1rem !important;
        }
        /* Custom styling for the figure */
        .main-svg {
            width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)

# Set the style for the plots
plt.style.use('default')  # Use default style instead of bmh for better contrast

# Load the console styles 
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('styles/console.css')

def build_logs_html(logs):
    lines_html = []
    for i, log in enumerate(logs):
        if "Current Distance =" in log:
            parts = log.split(":")
            iteration = parts[0]
            values = parts[1].split(",")
            
            current_dist = values[0].split("=")[1].strip()
            best_dist = values[1].split("=")[1].strip()
            
            formatted_log = (
                f"{iteration}: "
                f"Current Distance = <span class='distance-value'>{current_dist}</span>, "
                f"Best Distance = <span class='distance-value'>{best_dist}</span>"
            )
        else:
            formatted_log = log

        if i == len(logs) - 1:
            lines_html.append(f"<div class='log-line highlight'>{formatted_log}</div>")
        else:
            lines_html.append(f"<div class='log-line'>{formatted_log}</div>")

    return "\n".join(lines_html)

def create_log_box(logs):
    """
    This version keeps the view at the top, showing oldest logs first
    """
    logs_html = build_logs_html(logs)

    log_box_html = f"""
    <div class="log-container">
        <div class="log-header">Logs</div>
        <div class="scrollable-log-box" id="log-box">
            <div class="log-content" style="display: flex; flex-direction: column-reverse;">
                {logs_html}
            </div>
        </div>
    </div>
    <script>
        (function() {{
            const logBox = document.getElementById('log-box');
            
            // Keep scroll at top
            function keepAtTop() {{
                if (logBox) {{
                    logBox.scrollTop = 0;
                }}
            }}

            // Handle content changes
            const observer = new MutationObserver(() => {{
                keepAtTop();
            }});

            observer.observe(logBox, {{ 
                childList: true, 
                subtree: true 
            }});

            // Initial position
            keepAtTop();
        }})();
    </script>
    """
    return log_box_html


# Load city data from the uploaded JSON file
def load_data(file_path="data.json"):
    with open(file_path, "r") as file:
        return json.load(file)

# Haversine formula to calculate distance between two points on the Earth (in miles)
def distance(city1, city2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(city1[1]), np.radians(city1[0])  # city1[1] is latitude, city1[0] is longitude
    lat2, lon2 = np.radians(city2[1]), np.radians(city2[0])  # city2[1] is latitude, city2[0] is longitude

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Radius of Earth in miles (instead of kilometers)
    R = 3958.8  # Radius of Earth in miles
    return R * c  # Distance in miles

# Calculate total distance for a given tour
def total_distance(tour, cities):
    return sum(distance(cities[tour[i]], cities[tour[i-1]]) for i in range(len(tour)))

# Simulated Annealing algorithm to solve TSP
def simulated_annealing(cities_data, initial_temp, cooling_rate, num_iterations):
    # Extract the coordinates (longitude, latitude) of each city
    cities = np.array([(city['longitude'], city['latitude']) for city in cities_data])
    
    num_cities = len(cities)
    # Ensure Boston is the starting city
    boston_idx = next(i for i, city in enumerate(cities_data) if city['city'] == 'Boston')
    current_tour = [boston_idx] + [i for i in range(num_cities) if i != boston_idx]
    np.random.shuffle(current_tour[1:])  # Shuffle the rest of the cities, keeping Boston first
    current_dist = total_distance(current_tour, cities)
    best_tour = current_tour.copy()
    best_dist = current_dist
    temp = initial_temp
    history = []

    for _ in range(num_iterations):
        new_tour = current_tour.copy()
        i, j = np.random.randint(1, num_cities, 2)  # Avoid swapping Boston
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_dist = total_distance(new_tour, cities)

        # Accept new tour if it's better or with some probability
        if new_dist < current_dist or np.random.random() < np.exp((current_dist - new_dist) / temp):
            current_tour = new_tour
            current_dist = new_dist

        # Update best solution found so far
        if current_dist < best_dist:
            best_tour = current_tour.copy()
            best_dist = current_dist

        # Keep track of progress
        history.append((current_tour.copy(), current_dist, best_tour.copy(), best_dist))
        temp *= cooling_rate

    return best_tour, best_dist, history

def plot_final_state(tour, distance, cities, city_names, color):
    """Helper function to plot final state with consistent styling"""
    plt.scatter(cities[:, 0], cities[:, 1], 
               c='red', s=100, zorder=5,
               marker='o', edgecolor='darkred', linewidth=1.5)
    
    # Plot the tour
    tour_coords = cities[tour + [tour[0]]]
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], 
             color=color, linewidth=2.5, alpha=0.7, zorder=4)
    
    # Add city labels
    for i, (x, y) in enumerate(cities):
        plt.annotate(city_names[i], 
                    (x, y),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5',
                            fc='white',
                            ec='black',
                            alpha=0.9))
    
    # Add distance text
    plt.text(0.02, 1.02, f'Distance: {distance:.2f} miles',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, 
                      boxstyle='round,pad=0.5'),
             fontsize=11,
             fontweight='bold')
    
    # Customize axes
    plt.grid(True, linestyle='--', alpha=0.6, color='gray')
    plt.xlabel('Longitude', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Latitude', fontsize=12, fontweight='bold', labelpad=10)
    plt.gca().set_facecolor('#f0f0f0')
    
    # Set axis limits with padding
    lon_min, lon_max = np.min(cities[:, 0]), np.max(cities[:, 0])
    lat_min, lat_max = np.min(cities[:, 1]), np.max(cities[:, 1])
    lon_pad = (lon_max - lon_min) * 0.05
    lat_pad = (lat_max - lat_min) * 0.05
    plt.xlim([lon_min - lon_pad, lon_max + lon_pad])
    plt.ylim([lat_min - lat_pad, lat_max + lat_pad])

# -- Streamlit App --

st.title("Traveling Salesman Problem Solver")

# Load cities data from the uploaded JSON file
cities_data = load_data("data.json")

# Extract the coordinates (latitude, longitude) of each city
cities = np.array([(city['longitude'], city['latitude']) for city in cities_data])
city_names = [city['city'] for city in cities_data]  # Extract city names

col1, col2, col3 = st.columns(3)
with col1:
    initial_temp = st.number_input("Initial temperature", 1.0, 1500.0, 100.0)
with col2:
    cooling_rate = st.number_input("Cooling rate", 0.8, 0.9999, 0.995, format="%.4f")
with col3:
    num_iterations = st.number_input("Number of iterations", 100, 10000, 500)

if st.button("Solve TSP"):
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Lists to store frames for GIF
    frames = []
    
    # Prepare lists for live plotting
    costs = []
    accept_probs = []
    temps = []
    
    # Run simulated annealing
    best_tour, best_dist, history = simulated_annealing(
        cities_data, initial_temp, cooling_rate, num_iterations
    )

    # Plot the animation
    animation_placeholder = st.empty()
    log_placeholder = st.empty()

    logs = []

    # Create Streamlit placeholders for live plots
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    path_placeholder = col1.empty()
    cost_placeholder = col2.empty()
    accept_placeholder = col3.empty()
    temp_placeholder = col4.empty()

    # Reset all matplotlib parameters to default
    plt.rcdefaults()
    plt.style.use('default')
    
    # Create figure with larger size and more height for titles
    fig = plt.figure(figsize=(15, 9))  # Increased height further
    
    # Create gridspec with space for titles
    gs = plt.GridSpec(2, 2, height_ratios=[1, 15], width_ratios=[1, 1])
    gs.update(hspace=0.5, wspace=0.3)  # Increased spacing
    
    # Create a special subplot for the main title
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')
    title_ax.text(0.5, 0.5, "Traveling Salesman Problem - Simulated Annealing",
                 fontsize=16, fontweight='bold', ha='center', va='center')
    
    # Create main plot axes
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    
    # Set background colors
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#f0f0f0')
    ax2.set_facecolor('#f0f0f0')
    
    # Calculate axis limits with padding
    lon_min, lon_max = np.min(cities[:, 0]), np.max(cities[:, 0])
    lat_min, lat_max = np.min(cities[:, 1]), np.max(cities[:, 1])
    
    # Add 5% padding to the limits
    lon_pad = (lon_max - lon_min) * 0.05
    lat_pad = (lat_max - lat_min) * 0.05
    x_limits = [lon_min - lon_pad, lon_max + lon_pad]
    y_limits = [lat_min - lat_pad, lat_max + lat_pad]

    for idx, (main_ax, title) in enumerate(zip([ax1, ax2], ["Current Tour", "Best Tour"])):
        main_ax.set_xlim(x_limits)
        main_ax.set_ylim(y_limits)
        
        # Draw grid with light gray lines
        main_ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        
        # Plot cities with larger markers
        main_ax.scatter(cities[:, 0], cities[:, 1], 
                       c='red', s=100, zorder=5,
                       marker='o', edgecolor='darkred', linewidth=1.5)
        
        # Customize axes
        main_ax.set_xlabel('Longitude', fontsize=12, fontweight='bold', labelpad=10)
        main_ax.set_ylabel('Latitude' if idx == 0 else '', fontsize=12, fontweight='bold', labelpad=10)
        
        # Customize ticks
        main_ax.tick_params(axis='both', which='major', labelsize=10, 
                          width=1, length=6, colors='black')
        main_ax.tick_params(axis='x', rotation=45)
        
        # Add city labels with offset positions
        for i, (x, y) in enumerate(cities):
            main_ax.annotate(city_names[i], 
                           (x, y),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           va='bottom',
                           fontsize=9,
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5',
                                   fc='white',
                                   ec='black',
                                   alpha=0.9))
        
        # Set subplot titles
        main_ax.text(0.5, 1.1, title,
                    transform=main_ax.transAxes,
                    fontsize=14, fontweight='bold',
                    ha='center', va='bottom')
    
    # Create tour lines with thicker width
    line1, = ax1.plot([], [], color='navy', linewidth=2.5, alpha=0.7, zorder=4)
    line2, = ax2.plot([], [], color='darkgreen', linewidth=2.5, alpha=0.7, zorder=4)
    
    # Add distance text with better visibility and positioning
    text_props = dict(
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, 
                 boxstyle='round,pad=0.5'),
        fontsize=11,
        fontweight='bold',
        ha='left',
        va='top'
    )
    
    # Position distance text at the top left of each subplot
    current_distance_text = ax1.text(-0.12, 1.15, '', transform=ax1.transAxes, **text_props)
    best_distance_text = ax2.text(-0.12, 1.15, '', transform=ax2.transAxes, **text_props)
    
    # Update plot function
    def update_plot(frame):
        current_tour, current_dist, best_tour_frame, best_dist_frame = history[frame]
        
        # Update tour lines
        x1 = cities[current_tour + [current_tour[0]], 0]
        y1 = cities[current_tour + [current_tour[0]], 1]
        line1.set_data(x1, y1)
        current_distance_text.set_text(f'Distance: {current_dist:.2f} miles')
        
        x2 = cities[best_tour_frame + [best_tour_frame[0]], 0]
        y2 = cities[best_tour_frame + [best_tour_frame[0]], 1]
        line2.set_data(x2, y2)
        best_distance_text.set_text(f'Distance: {best_dist_frame:.2f} miles')

    # Save figure with high contrast
    def save_figure():
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                   facecolor='white', edgecolor='none',
                   pad_inches=0.3)
        buf.seek(0)
        return buf

    # Simulate "real-time" updates with progress bar
    progress_bar = st.progress(0)
    
    for i, (curr_tour, curr_dist, best_t, best_dist_iteration) in enumerate(history):
        update_plot(i)
        progress = (i + 1) / len(history)
        progress_bar.progress(progress)
        
        # Save the current frame
        buf = save_figure()
        
        # For GIF creation
        img_byte_arr = buf.getvalue()
        img = imageio.imread(img_byte_arr)
        frames.append(img)
        
        # Display in Streamlit
        animation_placeholder.image(buf, caption=f"Iteration: {i + 1}/{num_iterations}", use_container_width=True)
        
        log_message = f"Iteration {i+1}: Current Distance = {curr_dist:.2f} miles, Best Distance = {best_dist_iteration:.2f} miles"
        logs.append(log_message)
        
        log_box_html = create_log_box(logs)
        log_placeholder.markdown(log_box_html, unsafe_allow_html=True)

        # --- Live plots ---
        costs.append(curr_dist)
        temps.append(initial_temp * (cooling_rate ** i))
        if i > 0:
            prev_dist = history[i-1][1]
            delta_E = curr_dist - prev_dist
            if delta_E < 0:
                accept_probs.append(1.0)
            else:
                accept_probs.append(np.exp(-delta_E / (initial_temp * (cooling_rate ** i))))
        else:
            accept_probs.append(1.0)

        # Path plot (current tour)
        fig_path, ax_path = plt.subplots()
        ax_path.plot(cities[curr_tour + [curr_tour[0]], 0], cities[curr_tour + [curr_tour[0]], 1], '-o', color='blue')
        ax_path.set_title("Path")
        ax_path.set_xlabel("x")
        ax_path.set_ylabel("y")
        path_placeholder.pyplot(fig_path)
        plt.close(fig_path)

        # Cost plot
        fig_cost, ax_cost = plt.subplots()
        ax_cost.plot(costs, color='blue')
        ax_cost.set_title("Total length (cost)")
        ax_cost.set_xlabel("Step")
        ax_cost.set_ylabel("Length")
        cost_placeholder.pyplot(fig_cost)
        plt.close(fig_cost)

        # Acceptance probability plot
        fig_accept, ax_accept = plt.subplots()
        ax_accept.plot(accept_probs, '.', color='blue')
        ax_accept.set_title("Acceptance probability")
        ax_accept.set_xlabel("Step")
        ax_accept.set_ylabel("Acceptance probability")
        accept_placeholder.pyplot(fig_accept)
        plt.close(fig_accept)

        # Temperature plot
        fig_temp, ax_temp = plt.subplots()
        ax_temp.plot(temps, color='blue')
        ax_temp.set_title("Temperature")
        ax_temp.set_xlabel("Step")
        ax_temp.set_ylabel("Temperature")
        temp_placeholder.pyplot(fig_temp)
        plt.close(fig_temp)

    # Save the final state as separate images
    plt.figure(figsize=(15, 9))
    
    # Create final current tour plot
    plt.subplot(121)
    plt.title("Final Current Tour", fontsize=14, fontweight='bold')
    plot_final_state(curr_tour, curr_dist, cities, city_names, "blue")
    
    # Create final best tour plot
    plt.subplot(122)
    plt.title("Final Best Tour", fontsize=14, fontweight='bold')
    plot_final_state(best_t, best_dist_iteration, cities, city_names, "green")
    
    # Save final state
    plt.savefig(os.path.join(output_dir, 'final_state.png'), 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Save individual final plots
    # Current Tour
    plt.figure(figsize=(10, 8))
    plot_final_state(curr_tour, curr_dist, cities, city_names, "blue")
    plt.title("Final Current Tour", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'final_current_tour.png'), 
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Best Tour
    plt.figure(figsize=(10, 8))
    plot_final_state(best_t, best_dist_iteration, cities, city_names, "green")
    plt.title("Final Best Tour", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'final_best_tour.png'), 
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Save the animation as GIF
    imageio.mimsave(os.path.join(output_dir, 'tsp_animation.gif'), 
                   frames, fps=10)
    
    # Display success message for saved files
    st.success("""
    ✅ Successfully saved visualization files:
    - 🎬 Animation GIF: output/tsp_animation.gif
    - 🖼️ Final state (both tours): output/final_state.png
    - 🗺️ Final current tour: output/final_current_tour.png
    - 🎯 Final best tour: output/final_best_tour.png
    """)
    
    # Add a separator
    st.markdown("<hr style='margin: 2rem 0; background: linear-gradient(to right, #ff4b4b, #7e56ff); height: 2px; border: none;'>", unsafe_allow_html=True)
    
    # Display final results in an attractive format
    st.markdown("""
    <h2 style='text-align: center; color: #1f1f1f; margin-bottom: 2rem;'>
        🏆 Final Results
    </h2>
    """, unsafe_allow_html=True)

    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    # Initial distance
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
                    padding: 1rem; border-radius: 10px; border: 1px solid #ff4b4b'>
            <h3 style='margin: 0; color: #ff4b4b; font-size: 1.1rem; text-align: center;'>
                🚗 Initial Distance
            </h3>
            <p style='font-size: 1.8rem; margin: 0.5rem 0; text-align: center; font-weight: bold;'>
                {:.2f} miles
            </p>
        </div>
        """.format(history[0][1]), unsafe_allow_html=True)

    # Best distance
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #00994422, #00994444);
                    padding: 1rem; border-radius: 10px; border: 1px solid #009944'>
            <h3 style='margin: 0; color: #009944; font-size: 1.1rem; text-align: center;'>
                ✨ Best Distance
            </h3>
            <p style='font-size: 1.8rem; margin: 0.5rem 0; text-align: center; font-weight: bold;'>
                {:.2f} miles
            </p>
        </div>
        """.format(best_dist), unsafe_allow_html=True)

    # Improvement percentage
    with col3:
        improvement = ((history[0][1] - best_dist) / history[0][1] * 100)
        st.markdown("""
        <div style='background: linear-gradient(135deg, #7e56ff22, #7e56ff44);
                    padding: 1rem; border-radius: 10px; border: 1px solid #7e56ff'>
            <h3 style='margin: 0; color: #7e56ff; font-size: 1.1rem; text-align: center;'>
                📈 Improvement
            </h3>
            <p style='font-size: 1.8rem; margin: 0.5rem 0; text-align: center; font-weight: bold;'>
                {:.2f}%
            </p>
        </div>
        """.format(improvement), unsafe_allow_html=True)

    # Add space
    st.markdown("<br>", unsafe_allow_html=True)

    # Display the best route found
    st.markdown("""
    <div style='
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        max-width: 600px;
        margin: 0 auto;
    '>
        <div style='display: flex; flex-direction: column; align-items: stretch; gap: 0.5rem;'>
    """, unsafe_allow_html=True)

    # Create route steps with city names and arrows
    route_steps = [city_names[i] for i in best_tour]
    for i, city in enumerate(route_steps):
        # Add city bubble with modern styling
        st.markdown(f"""
        <div style='
            background: white;
            padding: 1rem 2rem;
            border-radius: 50px;
            border: 1px solid #e0e0e0;
            display: flex;
            align-items: center;
            margin: 0.2rem 0;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        '>
            <div style='
                background: #f8f9fa;
                color: #495057;
                font-weight: 600;
                padding: 0.4rem 0.8rem;
                border-radius: 20px;
                margin-right: 1.2rem;
                min-width: 2.5rem;
                text-align: center;
                font-size: 0.9rem;
            '>
                {i+1}
            </div>
            <span style='
                color: #212529;
                font-weight: 500;
                font-size: 1.1rem;
                letter-spacing: 0.01em;
            '>{city}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Add arrow if not the last city
        if i < len(route_steps) - 1:
            st.markdown("""
            <div style='
                display: flex;
                justify-content: center;
                padding: 0.3rem 0;
            '>
                <svg width="20" height="12" viewBox="0 0 20 12" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M10 12L0 2L2 0L10 8L18 0L20 2L10 12Z" fill="#CED4DA"/>
                </svg>
            </div>
            """, unsafe_allow_html=True)

    # Add return to start note with modern styling
    st.markdown("""
        </div>
        <div style='
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 1.5rem;
            padding: 0.8rem;
            background: #f8f9fa;
            border-radius: 25px;
            color: #6c757d;
            font-size: 0.9rem;
            font-weight: 500;
        '>
            <svg width="16" height="16" viewBox="0 0 16 16" style="margin-right: 8px;">
                <path d="M8 3L4 7H7V13H9V7H12L8 3Z" fill="currentColor"/>
                <path d="M1 9L0 8C0.75 3.5 4.5 0 9 0C13.5 0 17.25 3.5 18 8L17 9C16.25 4.5 13 1 9 1C5 1 1.75 4.5 1 9Z" fill="currentColor"/>
            </svg>
            Returns to starting city (Boston)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add space at the bottom
    st.markdown("<br>", unsafe_allow_html=True)

    # After the main loop, save the final versions of the four plots to the output folder
    # Path plot (final best tour)
    fig_path, ax_path = plt.subplots()
    ax_path.plot(cities[best_tour + [best_tour[0]], 0], cities[best_tour + [best_tour[0]], 1], '-o', color='blue')
    ax_path.set_title("Path")
    ax_path.set_xlabel("x")
    ax_path.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_path.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig_path)

    # Cost plot
    fig_cost, ax_cost = plt.subplots()
    ax_cost.plot(costs, color='blue')
    ax_cost.set_title("Total length (cost)")
    ax_cost.set_xlabel("Step")
    ax_cost.set_ylabel("Length")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_cost.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig_cost)

    # Acceptance probability plot
    fig_accept, ax_accept = plt.subplots()
    ax_accept.plot(accept_probs, '.', color='blue')
    ax_accept.set_title("Acceptance probability")
    ax_accept.set_xlabel("Step")
    ax_accept.set_ylabel("Acceptance probability")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_acceptance_probability.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig_accept)

    # Temperature plot
    fig_temp, ax_temp = plt.subplots()
    ax_temp.plot(temps, color='blue')
    ax_temp.set_title("Temperature")
    ax_temp.set_xlabel("Step")
    ax_temp.set_ylabel("Temperature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_temperature.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig_temp)


