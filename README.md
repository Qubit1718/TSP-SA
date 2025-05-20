# Traveling Salesman Problem (TSP) Solver with Simulated Annealing

This project is an interactive web application for visualizing and solving the Traveling Salesman Problem (TSP) using the Simulated Annealing algorithm. Built with Streamlit, it provides real-time visualizations, live metrics, and exportable results for educational and research purposes.

## Features
- **Interactive Streamlit web app** for TSP
- **Simulated Annealing** with tunable parameters (initial temperature, cooling rate, iterations)
- **Live-updating visualizations**:
  - Current and best tour
  - Path plot
  - Cost (total length) over time
  - Acceptance probability over time
  - Temperature cooling curve
- **Exportable results**:
  - Final tour images
  - Animation GIF
  - All live plots as PNGs
- **Modern, poster-ready design**
- **Customizable city data** (via `data.json`)

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install the main dependencies:
```bash
pip install streamlit numpy matplotlib imageio
```

### 3. Run the app
```bash
streamlit run streamlit.py
```

## Usage
- Adjust the initial temperature, cooling rate, and number of iterations in the sidebar.
- Click **Solve TSP** to run the algorithm.
- Watch the live-updating plots and tour animation.
- Download/export results from the `output/` folder:
  - `final_path.png`, `final_cost.png`, `final_acceptance_probability.png`, `final_temperature.png`
  - `final_state.png`, `final_current_tour.png`, `final_best_tour.png`, `tsp_animation.gif`

## Customizing Cities
- Edit `data.json` to change the set of cities and their coordinates.
- The app expects a list of city objects with `city`, `latitude`, and `longitude` fields.

## Project Structure
```
├── streamlit.py                # Main Streamlit app
├── data.json                   # City data
├── output/                     # Saved results and images
├── styles/                     # Custom CSS
├── methodology_figures.py      # Scripts for generating poster figures
├── problem_definition.py       # Scripts for generating problem definition figures
├── README.md                   # This file
```

## Example Visualizations
- Live path, cost, acceptance probability, and temperature plots
- Animated GIF of the optimization process
- Poster-ready comparison of Nearest Neighbor vs. Simulated Annealing

## Credits
- Developed by [Your Name]
- Inspired by classic TSP and Simulated Annealing literature
- Visualization style inspired by scientific posters and educational resources

## License
[MIT License](LICENSE) 