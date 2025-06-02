import numpy as np
import plotly.graph_objects as go
from django.shortcuts import render
from .forms import KMCSimulationForm

def generate_lattice(size):
    return np.zeros((size, size, size))

def simulate_kmc(size, steps, temperature):
    lattice = generate_lattice(size)
    for _ in range(steps):
        x, y, z = np.random.randint(0, size, 3)
        lattice[x][y][z] = 1
    return lattice

def plot_lattice(lattice):
    coords = np.argwhere(lattice == 1)
    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker=dict(size=3, color='blue')
    )])
    fig.update_layout(
        title='KMC Lattice',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        )
    )
    return fig.to_html(full_html=False)
def index(request):
    html_plot = None
    result_message = ""
    if request.method == 'POST':
        form = KMCSimulationForm(request.POST)
        if form.is_valid():
            size = form.cleaned_data['lattice_size']
            steps = form.cleaned_data['steps']
            temperature = form.cleaned_data['temperature']
            lattice = simulate_kmc(size, steps, temperature)
            html_plot = plot_lattice(lattice)

            # Simple "AI-style" interpretation
            occupied_sites = np.sum(lattice)
            total_sites = size**3
            occupancy_pct = (occupied_sites / total_sites) * 100

            result_message = (
                f"Simulation completed: lattice size={size}, steps={steps}, temperature={temperature}°C.\n"
                f"Occupied sites: {occupied_sites} out of {total_sites} ({occupancy_pct:.2f}%).\n"
                f"The graph visualizes occupied sites in blue markers in the 3D lattice.\n"
                f"Higher occupancy may indicate increased reaction or diffusion activity."
            )
    else:
        form = KMCSimulationForm()

    return render(request, 'simulation/index.html', {
        'form': form,
        'plot': html_plot,
        'result': result_message,
    })
