import numpy as np
import plotly.graph_objects as go
from django.shortcuts import render
from plotly.subplots import make_subplots
import time
from .forms import KMCSimulationForm


def generate_lattice(size):
    return np.zeros((size, size, size))


def insert_impurities(lattice, impurity_concentration):
    total_sites = np.prod(lattice.shape)
    impurity_sites = int(total_sites * impurity_concentration)
    indices = np.random.choice(total_sites, impurity_sites, replace=False)
    flat_lattice = lattice.flatten()
    flat_lattice[indices] = 2  # 2 represents impurity atoms
    return flat_lattice.reshape(lattice.shape)


def activation_energy(event_type, temperature):
    base_energies = {
        'surface_diffusion': 0.5,
        'bulk_diffusion': 1.1,
        'impurity_migration': 0.8,
        'atom_hop': 0.6
    }
    return base_energies.get(event_type, 1.0)


def simulate_kmc(size, steps, temperature, impurity_concentration):
    lattice = generate_lattice(size)
    lattice = insert_impurities(lattice, impurity_concentration)

    history = {'coverage': [], 'time_steps': [], 'impurity_sites': []}

    for step in range(steps):
        x, y, z = np.random.randint(0, size, 3)
        site = lattice[x, y, z]

        if site == 0:
            event = 'surface_diffusion'
        elif site == 2:
            event = 'impurity_migration'
        else:
            event = 'atom_hop'

        Ea = activation_energy(event, temperature)
        prob = np.exp(-Ea / (8.617e-5 * temperature))  # Boltzmann constant in eV/K

        if np.random.random() < prob:
            lattice[x, y, z] = 1

        if step % 100 == 0:
            coverage = np.sum(lattice == 1) / size ** 3
            history['coverage'].append(coverage)
            history['impurity_sites'].append(np.sum(lattice == 2))
            history['time_steps'].append(step)

    return lattice, history


def plot_lattice(lattice, history):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'xy'}]],
        subplot_titles=('3D Lattice', 'Coverage Over Time')
    )

    occupied = np.argwhere(lattice == 1)
    impurity = np.argwhere(lattice == 2)

    if len(occupied) > 0:
        fig.add_trace(go.Scatter3d(
            x=occupied[:, 0], y=occupied[:, 1], z=occupied[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.7),
            name='Tungsten Atoms'
        ), row=1, col=1)

    if len(impurity) > 0:
        fig.add_trace(go.Scatter3d(
            x=impurity[:, 0], y=impurity[:, 1], z=impurity[:, 2],
            mode='markers',
            marker=dict(size=3, color='red', opacity=0.5),
            name='Impurities'
        ), row=1, col=1)

    if history['time_steps']:
        fig.add_trace(go.Scatter(
            x=history['time_steps'],
            y=history['coverage'],
            mode='lines+markers',
            name='Coverage',
            line=dict(color='green')
        ), row=1, col=2)

    fig.update_layout(
        height=500,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig.to_html(full_html=False)


def index(request):
    html_plot = None
    result_message = ""
    metrics = {}

    if request.method == 'POST':
        form = KMCSimulationForm(request.POST)
        if form.is_valid():
            start_time = time.time()

            try:
                size = form.cleaned_data['lattice_size']
                steps = form.cleaned_data['steps']
                temperature = form.cleaned_data['temperature']
                impurity_concentration = form.cleaned_data['impurity_concentration']

                lattice, history = simulate_kmc(size, steps, temperature, impurity_concentration)
                html_plot = plot_lattice(lattice, history)

                execution_time = time.time() - start_time
                occupied = np.sum(lattice == 1)
                impurities = np.sum(lattice == 2)
                total_sites = size ** 3

                result_message = (
                    f"Simulation completed in {execution_time:.2f} seconds\n"
                    f"Occupied sites: {occupied}/{total_sites} ({occupied / total_sites * 100:.1f}%)\n"
                    f"Impurity sites: {impurities} ({impurities / total_sites * 100:.2f}%)\n"
                    f"Final coverage: {history['coverage'][-1] * 100:.1f}%"
                )

                metrics = {
                    'execution_time': f"{execution_time:.2f}s",
                    'lattice_size': f"{size}",
                    'steps': steps,
                    'temperature': f"{temperature}",
                    'coverage_A': f"{history['coverage'][-1] * 100:.1f}%",
                    'coverage_B': f"{impurities / total_sites * 100:.2f}%",
                    'reactions': f"{steps}"
                }

            except Exception as e:
                result_message = f"Error: {str(e)}"
    else:
        form = KMCSimulationForm()

    return render(request, 'simulation/index.html', {
        'form': form,
        'plot': html_plot,
        'result': result_message,
        'metrics': metrics
    })
