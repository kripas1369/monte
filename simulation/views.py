import numpy as np
import plotly.graph_objects as go
from django.shortcuts import render
from django.http import JsonResponse
from plotly.subplots import make_subplots
import time
from .forms import KMCSimulationForm  # Make sure you have this form defined

def generate_lattice(size):
    """Create a 3D lattice of given size"""
    return np.zeros((size, size, size))

def simulate_kmc(size, steps, temperature):
    """Basic KMC simulation"""
    lattice = generate_lattice(size)
    history = {'coverage': [], 'time_steps': []}
    
    for step in range(steps):
        # Random site selection
        x, y, z = np.random.randint(0, size, 3)
        
        # Simple temperature-dependent probability
        prob = 1 / (1 + np.exp(-temperature/100))
        
        if lattice[x, y, z] == 0 and np.random.random() < prob:
            lattice[x, y, z] = 1
            
        # Record every 100 steps
        if step % 100 == 0:
            history['coverage'].append(np.sum(lattice == 1) / size**3)
            history['time_steps'].append(step)
    
    return lattice, history

def plot_lattice(lattice, history):
    """Create visualization"""
    fig = make_subplots(rows=1, cols=2,
                       specs=[[{'type': 'scatter3d'}, {'type': 'xy'}]],
                       subplot_titles=('3D Lattice', 'Coverage Over Time'))
    
    # 3D plot
    occupied = np.argwhere(lattice == 1)
    if len(occupied) > 0:
        fig.add_trace(go.Scatter3d(
            x=occupied[:, 0], y=occupied[:, 1], z=occupied[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.7),
            name='Occupied Sites'
        ), row=1, col=1)
    
    # Coverage plot
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
                size = int(form.cleaned_data['lattice_size'])
                steps = int(form.cleaned_data['steps'])
                temperature = float(form.cleaned_data['temperature'])
                
                # Validate inputs
                if size <= 0 or steps <= 0 or temperature < 0:
                    raise ValueError("All values must be positive")
                
                lattice, history = simulate_kmc(size, steps, temperature)
                html_plot = plot_lattice(lattice, history)
                
                execution_time = time.time() - start_time
                occupied = np.sum(lattice == 1)
                total_sites = size**3
                
                result_message = (
                    f"Simulation completed in {execution_time:.2f} seconds\n"
                    f"Occupied sites: {occupied}/{total_sites} ({occupied/total_sites*100:.1f}%)\n"
                    f"Final coverage: {history['coverage'][-1]*100:.1f}%"
                )
                
                metrics = {
                    'time': f"{execution_time:.2f}s",
                    'size': f"{size}×{size}×{size}",
                    'steps': steps,
                    'temperature': f"{temperature}K",
                    'coverage': f"{history['coverage'][-1]*100:.1f}%"
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