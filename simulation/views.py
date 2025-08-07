import time
import pickle
import logging
import zlib
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseServerError, JsonResponse
from django.contrib import messages
import csv
from io import StringIO
import uuid
from .config import SIMULATION_PARAMS, STATES, VISUALIZATION
from .models import CrystalGrowthSimulation, SavedSimulation
from .forms import KMCSimulationForm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import types

logger = logging.getLogger(__name__)

def plot_lattice(occupied_sites, history, cluster_analyzer, lattice_size):
    try:
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}]],
            subplot_titles=('3D Island Formation', '2D Top View', 'Island Size Distribution', ''),
            row_heights=[0.6, 0.4],
            column_widths=[0.5, 0.5]
        )

        frames = []
        for i, snapshot in enumerate(history['lattices']):
            frame_data = []
            mobile = [(s[0], s[1], s[2]) for s in snapshot if s[3] == STATES['MOBILE']]
            stable = [(s[0], s[1], s[2]) for s in snapshot if s[3] == STATES['STABLE']]
            substrate = [(s[0], s[1], s[2]) for s in snapshot if s[3] == STATES['SUBSTRATE']]
            defect = [(s[0], s[1], s[2]) for s in snapshot if s[3] == STATES['DEFECT']]
            critical_clusters = cluster_analyzer.get_critical_clusters() if i == len(history['lattices']) - 1 else []

            if substrate:
                frame_data.append(go.Scatter3d(
                    x=[p[0] for p in substrate], y=[p[1] for p in substrate], z=[p[2] for p in substrate],
                    mode='markers', marker=dict(size=6, color=VISUALIZATION['colors'][STATES['SUBSTRATE']], opacity=0.7),
                    name='Substrate', hovertemplate='Substrate<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
                ))
            if mobile:
                frame_data.append(go.Scatter3d(
                    x=[p[0] for p in mobile], y=[p[1] for p in mobile], z=[p[2] for p in mobile],
                    mode='markers', marker=dict(size=6, color=VISUALIZATION['colors'][STATES['MOBILE']], opacity=0.7),
                    name='Mobile Atoms', hovertemplate='Mobile Atom<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
                ))
            if stable:
                frame_data.append(go.Scatter3d(
                    x=[p[0] for p in stable], y=[p[1] for p in stable], z=[p[2] for p in stable],
                    mode='markers', marker=dict(size=6, color=VISUALIZATION['colors'][STATES['STABLE']], opacity=0.7),
                    name='Stable Atoms', hovertemplate='Stable Atom<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
                ))
            if defect:
                frame_data.append(go.Scatter3d(
                    x=[p[0] for p in defect], y=[p[1] for p in defect], z=[p[2] for p in defect],
                    mode='markers', marker=dict(size=6, color=VISUALIZATION['colors'][STATES['DEFECT']], opacity=0.5),
                    name='Impurities', hovertemplate='Impurity<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
                ))
            for cluster in critical_clusters:
                indices = cluster['indices'].tolist()
                frame_data.append(go.Scatter3d(
                    x=[idx[0] for idx in indices], y=[idx[1] for idx in indices], z=[idx[2] for idx in indices],
                    mode='markers', marker=dict(size=7, color=VISUALIZATION['colors'][STATES['NUCLEATION']], opacity=0.9),
                    name=f'Island {cluster["id"]} (Size: {cluster["size"]})',
                    hovertemplate='Island ID: %{text}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>',
                    text=[f'{cluster["id"]}, Size: {cluster["size"]}, x/y: {cluster["aspect_ratios"]["x/y"]:.2f}' for _ in indices]
                ))

            if substrate:
                frame_data.append(go.Scatter(
                    x=[p[0] for p in substrate], y=[p[1] for p in substrate],
                    mode='markers', marker=dict(size=8, color=VISUALIZATION['colors'][STATES['SUBSTRATE']], opacity=0.7),
                    name='Substrate', hovertemplate='Substrate<br>x: %{x}<br>y: %{y}<extra></extra>',
                    xaxis='x2', yaxis='y2'
                ))
            if mobile:
                frame_data.append(go.Scatter(
                    x=[p[0] for p in mobile], y=[p[1] for p in mobile],
                    mode='markers', marker=dict(size=8, color=VISUALIZATION['colors'][STATES['MOBILE']], opacity=0.7),
                    name='Mobile Atoms', hovertemplate='Mobile Atom<br>x: %{x}<br>y: %{y}<extra></extra>',
                    xaxis='x2', yaxis='y2'
                ))
            if stable:
                frame_data.append(go.Scatter(
                    x=[p[0] for p in stable], y=[p[1] for p in stable],
                    mode='markers', marker=dict(size=8, color=VISUALIZATION['colors'][STATES['STABLE']], opacity=0.7),
                    name='Stable Atoms', hovertemplate='Stable Atom<br>x: %{x}<br>y: %{y}<extra></extra>',
                    xaxis='x2', yaxis='y2'
                ))
            if defect:
                frame_data.append(go.Scatter(
                    x=[p[0] for p in defect], y=[p[1] for p in defect],
                    mode='markers', marker=dict(size=8, color=VISUALIZATION['colors'][STATES['DEFECT']], opacity=0.5),
                    name='Impurities', hovertemplate='Impurity<br>x: %{x}<br>y: %{y}<extra></extra>',
                    xaxis='x2', yaxis='y2'
                ))
            for cluster in critical_clusters:
                indices = cluster['indices'].tolist()
                frame_data.append(go.Scatter(
                    x=[idx[0] for idx in indices], y=[idx[1] for idx in indices],
                    mode='markers', marker=dict(size=9, color=VISUALIZATION['colors'][STATES['NUCLEATION']], opacity=0.9),
                    name=f'Island {cluster["id"]} (Size: {cluster["size"]})',
                    hovertemplate='Island ID: %{text}<br>x: %{x}<br>y: %{y}<extra></extra>',
                    text=[f'{cluster["id"]}, Size: {cluster["size"]}, x/y: {cluster["aspect_ratios"]["x/y"]:.2f}' for _ in indices],
                    xaxis='x2', yaxis='y2'
                ))

            frames.append(go.Frame(data=frame_data, name=f'step_{i}'))

        mobile = [(s[0], s[1], s[2]) for s in occupied_sites if s[3] == STATES['MOBILE']]
        stable = [(s[0], s[1], s[2]) for s in occupied_sites if s[3] == STATES['STABLE']]
        substrate = [(s[0], s[1], s[2]) for s in occupied_sites if s[3] == STATES['SUBSTRATE']]
        defect = [(s[0], s[1], s[2]) for s in occupied_sites if s[3] == STATES['DEFECT']]
        critical_clusters = cluster_analyzer.get_critical_clusters()

        if substrate:
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in substrate], y=[p[1] for p in substrate], z=[p[2] for p in substrate],
                mode='markers', marker=dict(size=6, color=VISUALIZATION['colors'][STATES['SUBSTRATE']], opacity=0.7),
                name='Substrate', hovertemplate='Substrate<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
            ), row=1, col=1)
        if mobile:
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in mobile], y=[p[1] for p in mobile], z=[p[2] for p in mobile],
                mode='markers', marker=dict(size=6, color=VISUALIZATION['colors'][STATES['MOBILE']], opacity=0.7),
                name='Mobile Atoms', hovertemplate='Mobile Atom<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
            ), row=1, col=1)
        if stable:
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in stable], y=[p[1] for p in stable], z=[p[2] for p in stable],
                mode='markers', marker=dict(size=6, color=VISUALIZATION['colors'][STATES['STABLE']], opacity=0.7),
                name='Stable Atoms', hovertemplate='Stable Atom<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
            ), row=1, col=1)
        if defect:
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in defect], y=[p[1] for p in defect], z=[p[2] for p in defect],
                mode='markers', marker=dict(size=6, color=VISUALIZATION['colors'][STATES['DEFECT']], opacity=0.5),
                name='Impurities', hovertemplate='Impurity<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
            ), row=1, col=1)
        for cluster in critical_clusters:
            indices = cluster['indices'].tolist()
            fig.add_trace(go.Scatter3d(
                x=[idx[0] for idx in indices], y=[idx[1] for idx in indices], z=[idx[2] for idx in indices],
                mode='markers', marker=dict(size=7, color=VISUALIZATION['colors'][STATES['NUCLEATION']], opacity=0.9),
                name=f'Island {cluster["id"]} (Size: {cluster["size"]})',
                hovertemplate='Island ID: %{text}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>',
                text=[f'{cluster["id"]}, Size: {cluster["size"]}, x/y: {cluster["aspect_ratios"]["x/y"]:.2f}' for _ in indices]
            ), row=1, col=1)

        if substrate:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in substrate], y=[p[1] for p in substrate],
                mode='markers', marker=dict(size=8, color=VISUALIZATION['colors'][STATES['SUBSTRATE']], opacity=0.7),
                name='Substrate', hovertemplate='Substrate<br>x: %{x}<br>y: %{y}<extra></extra>',
                xaxis='x2', yaxis='y2'
            ), row=1, col=2)
        if mobile:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in mobile], y=[p[1] for p in mobile],
                mode='markers', marker=dict(size=8, color=VISUALIZATION['colors'][STATES['MOBILE']], opacity=0.7),
                name='Mobile Atoms', hovertemplate='Mobile Atom<br>x: %{x}<br>y: %{y}<extra></extra>',
                xaxis='x2', yaxis='y2'
            ), row=1, col=2)
        if stable:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in stable], y=[p[1] for p in stable],
                mode='markers', marker=dict(size=8, color=VISUALIZATION['colors'][STATES['STABLE']], opacity=0.7),
                name='Stable Atoms', hovertemplate='Stable Atom<br>x: %{x}<br>y: %{y}<extra></extra>',
                xaxis='x2', yaxis='y2'
            ), row=1, col=2)
        if defect:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in defect], y=[p[1] for p in defect],
                mode='markers', marker=dict(size=8, color=VISUALIZATION['colors'][STATES['DEFECT']], opacity=0.5),
                name='Impurities', hovertemplate='Impurity<br>x: %{x}<br>y: %{y}<extra></extra>',
                xaxis='x2', yaxis='y2'
            ), row=1, col=2)
        for cluster in critical_clusters:
            indices = cluster['indices'].tolist()
            fig.add_trace(go.Scatter(
                x=[idx[0] for idx in indices], y=[idx[1] for idx in indices],
                mode='markers', marker=dict(size=9, color=VISUALIZATION['colors'][STATES['NUCLEATION']], opacity=0.9),
                name=f'Island {cluster["id"]} (Size: {cluster["size"]})',
                hovertemplate='Island ID: %{text}<br>x: %{x}<br>y: %{y}<extra></extra>',
                text=[f'{cluster["id"]}, Size: {cluster["size"]}, x/y: {cluster["aspect_ratios"]["x/y"]:.2f}' for _ in indices],
                xaxis='x2', yaxis='y2'
            ), row=1, col=2)

        cluster_stats = cluster_analyzer.get_cluster_statistics()
        sizes = list(cluster_stats['size_distribution'].keys())
        counts = list(cluster_stats['size_distribution'].values())
        fig.add_trace(go.Bar(
            x=sizes, y=counts, name='Island Sizes', marker_color='#3498DB',
            hovertemplate='Size: %{x}<br>Count: %{y}<extra></extra>'
        ), row=2, col=1)
        fig.update_xaxes(title_text='Island Size', row=2, col=1)
        fig.update_yaxes(title_text='Count', row=2, col=1)

        fig.frames = frames
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                        dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                    ],
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="right",
                    y=1.1,
                    yanchor="top"
                )
            ],
            sliders=[dict(
                steps=[dict(method="animate", args=[[f"step_{i}"], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}], label=f"Step {i}") for i in range(len(frames))],
                active=0,
                currentvalue={"prefix": "Step: "},
                pad={"t": 50}
            )],
            height=800, width=1000, showlegend=True,
            margin=dict(l=20, r=20, t=80, b=20),
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                camera_eye=dict(x=1.2, y=1.2, z=0.8),
                camera_up=dict(x=0, y=0, z=1),
                dragmode='orbit',
                aspectmode='cube',
                xaxis=dict(range=[-1, lattice_size]),
                yaxis=dict(range=[-1, lattice_size]),
                zaxis=dict(range=[-1, lattice_size])
            ),
            xaxis2=dict(title='X', range=[-1, lattice_size]),
            yaxis2=dict(title='Y', range=[-1, lattice_size]),
            title=dict(text='3D Island Formation Simulation', x=0.5, font=dict(size=24, color='#1F2937')),
            paper_bgcolor='rgba(255,255,255,0.9)',
            plot_bgcolor='rgba(255,255,255,0.9)'
        )
        return fig.to_html(full_html=False)
    except Exception as e:
        logger.error(f"Error in plot_lattice: {str(e)}")
        raise

def generate_comparison_plots(simulations):
    try:
        logger.debug(f"Generating comparison plots with {len(simulations)} simulations")
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Average Island Density vs Temperature (Fig. 4.2.3)',
                'Average Island Size vs Temperature (Fig. 4.2.4)',
                'Relative Width Distribution vs Flux (Fig. 4.2.5)',
                'Island Size Distribution (Fig. 4.2.7)'
            ),
            specs=[[{'type': 'xy'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}]],
            row_heights=[0.5, 0.5],
            column_widths=[0.5, 0.5]
        )

        temperatures = [220, 240, 260, 280, 300, 320]
        fluxes = [0.004, 0.006, 0.02, 0.04, 0.08, 0.3]
        flux_colors = {0.004: '#1f77b4', 0.02: '#ff7f0e', 0.3: '#2ca02c'}

        # Plot 1: Average Island Density vs Temperature (Fig. 4.2.3)
        for flux in [0.004, 0.02, 0.3]:
            sims = [s for s in simulations if abs(s.flux - flux) < 0.001]
            if sims:
                temps = []
                density_values = []
                for s in sims:
                    try:
                        sim_data = s.get_simulation_data()
                        cluster_stats = sim_data['cluster_stats']
                        logger.debug(f"Simulation {s.simulation_id} cluster_stats: {cluster_stats}")
                        total_clusters = cluster_stats.get('total_clusters', 0)
                        lattice_size = s.lattice_size
                        density = total_clusters / (lattice_size ** 3) if lattice_size > 0 else 0
                        if 'density' in cluster_stats:
                            if isinstance(cluster_stats['density'], types.FunctionType):
                                logger.error(f"Found function in cluster_stats['density'] for sim {s.simulation_id}: {cluster_stats['density']}")
                            else:
                                density = cluster_stats['density']
                        temps.append(s.temperature)
                        density_values.append(density)
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Invalid cluster stats for simulation {s.simulation_id}: {str(e)}")
                        temps.append(s.temperature)
                        density_values.append(0)
                logger.debug(f"Flux {flux}: temps={temps}, density_values={density_values}")
                fig.add_trace(
                    go.Scatter(
                        x=temps, y=density_values, mode='lines+markers', name=f'Flux {flux} ML/s',
                        line=dict(color=flux_colors[flux]), marker=dict(size=8)
                    ), row=1, col=1
                )
        fig.update_xaxes(title_text='Temperature (K)', row=1, col=1)
        fig.update_yaxes(title_text='Island Density (islands/site)', row=1, col=1, type='log')

        # Plot 2: Average Island Size vs Temperature (Fig. 4.2.4)
        for flux in [0.004, 0.02, 0.3]:
            sims = [s for s in simulations if abs(s.flux - flux) < 0.001]
            if sims:
                temps = []
                sizes = []
                for s in sims:
                    try:
                        sizes.append(s.get_simulation_data()['cluster_stats'].get('avg_island_size', 0))
                        temps.append(s.temperature)
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Invalid avg_island_size for simulation {s.simulation_id}: {str(e)}")
                        temps.append(s.temperature)
                        sizes.append(0)
                logger.debug(f"Flux {flux}: temps={temps}, sizes={sizes}")
                fig.add_trace(
                    go.Scatter(
                        x=temps, y=sizes, mode='lines+markers', name=f'Flux {flux} ML/s',
                        line=dict(color=flux_colors[flux]), marker=dict(size=8)
                    ), row=1, col=2
                )
        fig.update_xaxes(title_text='Temperature (K)', row=1, col=2)
        fig.update_yaxes(title_text='Average Island Size (sites)', row=1, col=2)

        # Plot 3: Relative Width Distribution vs Flux (Fig. 4.2.5)
        for temp in [220, 240, 260, 280, 300, 320]:
            sims = [s for s in simulations if abs(s.temperature - temp) < 5]
            if sims:
                flux_vals = []
                rwd = []
                for s in sims:
                    try:
                        rwd.append(s.get_simulation_data()['cluster_stats'].get('relative_width_distribution', 1.0))
                        flux_vals.append(s.flux)
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Invalid relative_width_distribution for simulation {s.simulation_id}: {str(e)}")
                        flux_vals.append(s.flux)
                        rwd.append(1.0)
                logger.debug(f"Temp {temp}: fluxes={flux_vals}, rwd={rwd}")
                fig.add_trace(
                    go.Scatter(
                        x=flux_vals, y=rwd, mode='lines+markers', name=f'Temperature {temp} K',
                        marker=dict(size=8)
                    ), row=2, col=1
                )
        fig.update_xaxes(title_text='Flux (ML/s)', row=2, col=1, type='log')
        fig.update_yaxes(title_text='Relative Width Distribution', row=2, col=1)

        # Plot 4: Island Size Distribution (Fig. 4.2.7)
        flux = 0.3
        for temp in [220, 240, 260, 280, 300]:
            sims = [s for s in simulations if abs(s.flux - flux) < 0.001 and abs(s.temperature - temp) < 5]
            if sims:
                sim = sims[0]
                try:
                    size_dist = sim.get_simulation_data()['cluster_stats'].get('size_distribution', {})
                    sizes = list(size_dist.keys())
                    counts = list(size_dist.values())
                except (KeyError, TypeError) as e:
                    logger.warning(f"Invalid size_distribution for simulation {sim.simulation_id}: {str(e)}")
                    sizes = [0]
                    counts = [0]
                logger.debug(f"Temp {temp}, Flux {flux}: sizes={sizes}, counts={counts}")
                fig.add_trace(
                    go.Scatter(
                        x=sizes, y=counts, mode='lines', name=f'Temperature {temp} K',
                        line=dict(width=2)
                    ), row=2, col=2
                )
        fig.update_xaxes(title_text='Island Size (sites)', row=2, col=2)
        fig.update_yaxes(title_text='Count', row=2, col=2)

        fig.update_layout(
            height=800, width=1000, showlegend=True,
            title=dict(text='Simulation Comparison Plots', x=0.5, font=dict(size=24)),
            paper_bgcolor='rgba(255,255,255,0.9)',
            plot_bgcolor='rgba(255,255,255,0.9)'
        )
        return fig.to_html(full_html=False) if simulations else "<p>No comparison plots available: insufficient simulation data.</p>"
    except Exception as e:
        logger.error(f"Error in generate_comparison_plots: {str(e)}")
        return "<p>Error generating comparison plots: insufficient or invalid data.</p>"

def save_simulation_data(data, parameters, html_plot):
    try:
        sim_id = uuid.uuid4()
        serialized_data = pickle.dumps(data)
        sim = SavedSimulation.objects.create(
            simulation_id=sim_id,
            name=parameters.get('name', None),
            lattice_size=parameters['lattice_size'],
            steps=parameters['steps'],
            temperature=parameters['temperature'],
            impurity_concentration=parameters['impurity_concentration'],
            diffusion_x=parameters['diffusion_x'],
            diffusion_y=parameters['diffusion_y'],
            diffusion_z=parameters['diffusion_z'],
            impurity_effect=parameters['impurity_effect'],
            flux=parameters['flux'],
            run_sweep=False,
            simulation_data=serialized_data,
            html_plot=html_plot,
            comparison_plot="<p>Comparison plots will be generated after saving.</p>"
        )
        return sim_id
    except Exception as e:
        logger.error(f"Error saving simulation data: {str(e)}")
        raise

def index(request):
    html_plot = None
    result_message = ""
    metrics = {}
    comparison_result = ""

    if request.method == 'POST':
        form = KMCSimulationForm(request.POST)
        if form.is_valid():
            start_time = time.time()
            try:
                size = form.cleaned_data['lattice_size']
                steps = form.cleaned_data['steps']
                temperature = form.cleaned_data['temperature']
                impurity_concentration = form.cleaned_data['impurity_concentration']
                diffusion_x = form.cleaned_data['diffusion_x']
                diffusion_y = form.cleaned_data['diffusion_y']
                diffusion_z = form.cleaned_data['diffusion_z']
                impurity_effect = form.cleaned_data['impurity_effect']
                flux = form.cleaned_data['flux']
                name = form.cleaned_data['name']
                diffusion_barriers = {'x': diffusion_x, 'y': diffusion_y, 'z': diffusion_z}

                parameters = {
                    'lattice_size': size,
                    'steps': steps,
                    'temperature': temperature,
                    'impurity_concentration': impurity_concentration,
                    'diffusion_x': diffusion_x,
                    'diffusion_y': diffusion_y,
                    'diffusion_z': diffusion_z,
                    'impurity_effect': impurity_effect,
                    'flux': flux,
                    'name': name
                }

                sim = CrystalGrowthSimulation(size, temperature, impurity_concentration, diffusion_barriers, impurity_effect, flux)
                sim.run_simulation(steps, update_interval=50)
                html_plot = plot_lattice(sim.occupied_sites, sim.history, sim.cluster_analyzer, size)

                execution_time = time.time() - start_time
                occupied = len(sim.occupied_sites)
                impurities = sum(1 for s in sim.occupied_sites if s[3] == STATES['DEFECT'])
                total_sites = size ** 3
                cluster_stats = sim.cluster_analyzer.get_cluster_statistics()
                logger.debug(f"New simulation cluster_stats: {cluster_stats}")

                # Remove any function references from cluster_stats
                if 'density' in cluster_stats and isinstance(cluster_stats['density'], types.FunctionType):
                    logger.warning(f"Removing density function from cluster_stats for sim {sim.simulation_id}")
                    del cluster_stats['density']
                    cluster_stats['density'] = cluster_stats.get('total_clusters', 0) / (size ** 3) if size > 0 else 0

                result_message = (
                    f"Simulation completed in {execution_time:.2f} seconds\n"
                    f"Occupied sites: {occupied}/{total_sites} ({occupied / total_sites * 100:.1f}%)\n"
                    f"Impurity sites: {impurities} ({impurities / total_sites * 100:.2f}%)\n"
                    f"Final coverage: {sim.history['coverage'][-1] * 100:.1f}%\n"
                    f"Nucleation events: {sim.nucleation_count}\n"
                    f"Total islands: {cluster_stats['total_clusters']}\n"
                    f"Critical islands: {len(cluster_stats['critical_clusters'])}\n"
                    f"Average aspect ratios: x/y={cluster_stats['avg_aspect_ratios']['x/y']:.2f}, "
                    f"x/z={cluster_stats['avg_aspect_ratios']['x/z']:.2f}, y/z={cluster_stats['avg_aspect_ratios']['y/z']:.2f}"
                )

                metrics = {
                    'execution_time': f"{execution_time:.2f}s",
                    'lattice_size': f"{size}",
                    'steps': steps,
                    'temperature': f"{temperature}K",
                    'flux': f"{flux} ML/s",
                    'coverage': f"{sim.history['coverage'][-1] * 100:.1f}%",
                    'impurity_coverage': f"{impurities / total_sites * 100:.2f}%",
                    'nucleation_events': f"{sim.nucleation_count}",
                    'total_clusters': f"{cluster_stats['total_clusters']}",
                    'critical_clusters': f"{len(cluster_stats['critical_clusters'])}",
                    'aspect_xy': f"{cluster_stats['avg_aspect_ratios']['x/y']:.2f}",
                    'aspect_xz': f"{cluster_stats['avg_aspect_ratios']['x/z']:.2f}",
                    'aspect_yz': f"{cluster_stats['avg_aspect_ratios']['y/z']:.2f}",
                    'relative_width_distribution': f"{cluster_stats.get('relative_width_distribution', 1.0):.2f}"
                }

                optimal_conditions = [
                    {'temperature': 260, 'flux': 0.006, 'desc': 'Uniform islands at low flux'},
                    {'temperature': 280, 'flux': 0.02, 'desc': 'Square-shaped, regular islands'},
                    {'temperature': 300, 'flux': 0.04, 'desc': 'Optimal patterning at low flux'},
                    {'temperature': 320, 'flux': 0.08, 'desc': 'Large, uniform islands at high flux'}
                ]
                is_optimal = False
                optimal_desc = ""
                for condition in optimal_conditions:
                    if abs(temperature - condition['temperature']) < 5 and abs(flux - condition['flux']) < 0.001:
                        is_optimal = True
                        optimal_desc = condition['desc']
                        break

                rwd = cluster_stats.get('relative_width_distribution', 1.0)
                island_density = cluster_stats.get('density', cluster_stats.get('total_clusters', 0) / (size ** 3))
                avg_island_size = cluster_stats.get('avg_island_size', 0)
                comparison_result = (
                    f"Simulation Evaluation:\n"
                    f"- Relative Width Distribution: {rwd:.2f} (lower indicates more uniform sizes)\n"
                    f"- Island Density: {island_density:.6f} islands/site (moderate is optimal)\n"
                    f"- Average Island Size: {avg_island_size:.2f} sites (larger is better at optimal conditions)\n"
                )
                if is_optimal:
                    comparison_result += f"- This simulation matches optimal conditions: {optimal_desc}\n"
                else:
                    comparison_result += "- Non-optimal conditions. Try adjusting temperature or flux for better patterning.\n"
                if rwd < 0.5 and 0.0001 < island_density < 0.001 and avg_island_size > 50:
                    comparison_result += "- **Good Simulation**: Likely produces uniform, well-patterned islands.\n"
                else:
                    comparison_result += "- **Suboptimal Simulation**: May have uneven sizes or irregular patterning.\n"

                sim_data = {
                    'occupied_sites': list(sim.occupied_sites),
                    'history': {
                        'time_steps': sim.history['time_steps'],
                        'coverage': sim.history['coverage'],
                        'nucleation_events': sim.history['nucleation_events'],
                        'cluster_counts': sim.history['cluster_counts'],
                        'event_counts': sim.history['event_counts'],
                        'lattices': sim.history['lattices']
                    },
                    'cluster_stats': {
                        'total_clusters': cluster_stats['total_clusters'],
                        'critical_clusters': cluster_stats['critical_clusters'],
                        'largest_size': cluster_stats['largest_size'],
                        'size_distribution': cluster_stats['size_distribution'],
                        'avg_aspect_ratios': cluster_stats['avg_aspect_ratios'],
                        'avg_island_size': cluster_stats['avg_island_size'],
                        'relative_width_distribution': cluster_stats['relative_width_distribution'],
                        'density': island_density
                    },
                    'comparison_result': comparison_result
                }
                sim_id = save_simulation_data(sim_data, parameters, html_plot)
                request.session['simulation_id'] = str(sim_id)
                request.session['comparison_result'] = comparison_result

            except Exception as e:
                logger.error(f"Error in index view: {str(e)}")
                result_message = f"Simulation failed: {str(e)}"
                return HttpResponseServerError(f"Error: {str(e)}. Please check inputs or contact support.")
    else:
        form = KMCSimulationForm()

    try:
        return render(request, 'simulation/index.html', {
            'form': form,
            'plot': html_plot,
            'result': result_message,
            'metrics': metrics,
            'comparison_result': comparison_result
        })
    except Exception as e:
        logger.error(f"Template rendering error: {str(e)}")
        return HttpResponseServerError("Failed to render template.")

def save_simulation(request):
    if request.method == 'POST':
        try:
            sim_id = request.session.get('simulation_id')
            if not sim_id:
                messages.error(request, "No simulation data available to save.")
                return JsonResponse({'status': 'error', 'message': 'No simulation data available.'}, status=400)

            sim = SavedSimulation.objects.get(simulation_id=sim_id)
            name = request.POST.get('name')
            if name:
                sim.name = name
            sim.save()  # Save simulation first to ensure it persists

            # Generate comparison plots after initial save
            simulations = SavedSimulation.objects.all()
            try:
                comparison_plot = generate_comparison_plots(simulations)
            except Exception as e:
                logger.warning(f"Failed to generate comparison plots: {str(e)}. Using fallback message.")
                comparison_plot = "<p>Error generating comparison plots: insufficient or invalid data.</p>"

            sim.comparison_plot = comparison_plot
            sim.save()  # Save again with comparison plots

            if 'simulation_id' in request.session:
                del request.session['simulation_id']
            if 'comparison_result' in request.session:
                comparison_result = request.session['comparison_result']
                del request.session['comparison_result']
            else:
                comparison_result = "No comparison result available."

            messages.success(request, f"Simulation {sim_id} saved successfully. View comparison at /compare/{sim_id}/")
            return JsonResponse({
                'status': 'success',
                'message': f'Simulation {sim_id} saved successfully.',
                'redirect_url': f"/compare/{sim_id}/"
            })
        except SavedSimulation.DoesNotExist:
            logger.error(f"Simulation {sim_id} not found.")
            messages.error(request, "Error: Simulation not found.")
            return JsonResponse({'status': 'error', 'message': 'Simulation not found.'}, status=404)
        except Exception as e:
            logger.error(f"Error in save_simulation: {str(e)}")
            messages.error(request, f"Error saving simulation: {str(e)}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=400)

def compare_simulation(request, sim_id):
    try:
        simulation = SavedSimulation.objects.get(simulation_id=sim_id)
        comparison_result = simulation.get_simulation_data().get('comparison_result', 'No comparison result available.')
        return render(request, 'simulation/comparison.html', {
            'simulation': simulation,
            'comparison_result': comparison_result
        })
    except SavedSimulation.DoesNotExist:
        logger.error(f"Simulation {sim_id} not found.")
        messages.error(request, "Error: Simulation not found.")
        return redirect('saved_simulations')
    except Exception as e:
        logger.error(f"Error in compare_simulation: {str(e)}")
        messages.error(request, f"Error loading comparison: {str(e)}")
        return redirect('saved_simulations')

def download_results(request):
    try:
        sim_id = request.session.get('simulation_id')
        if not sim_id:
            return HttpResponse("No simulation data available.", status=400)
        sim = SavedSimulation.objects.get(simulation_id=sim_id)
        sim_data = pickle.loads(sim.simulation_data)

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Step', 'Coverage', 'Nucleation Events', 'Island Count', 'Attach Events', 'Diffuse X', 'Diffuse Y', 'Diffuse Z'])
        for i, step in enumerate(sim_data['history']['time_steps']):
            writer.writerow([
                step,
                sim_data['history']['coverage'][i] * 100,
                sim_data['history']['nucleation_events'][i],
                sim_data['history']['cluster_counts'][i],
                sim_data['history']['event_counts'][i].get('attach', 0),
                sim_data['history']['event_counts'][i].get('diffuse_x', 0),
                sim_data['history']['event_counts'][i].get('diffuse_y', 0),
                sim_data['history']['event_counts'][i].get('diffuse_z', 0)
            ])

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="simulation_{sim_id}.csv"'
        response.write(output.getvalue())
        return response
    except Exception as e:
        logger.error(f"Error in download_results: {str(e)}")
        return HttpResponseServerError(f"Error generating CSV: {str(e)}")

def documentation(request):
    try:
        return render(request, 'simulation/documentation.html', {})
    except Exception as e:
        logger.error(f"Documentation rendering error: {str(e)}")
        return HttpResponseServerError("Failed to render documentation.")

def saved_simulations(request):
    try:
        simulations = SavedSimulation.objects.all().order_by('-created_at')
        return render(request, 'simulation/saved_simulations.html', {
            'simulations': simulations
        })
    except Exception as e:
        logger.error(f"Error in saved_simulations view: {str(e)}")
        return HttpResponseServerError("Failed to load saved simulations.")

def delete_simulation(request):
    if request.method == 'POST':
        sim_id = request.POST.get('simulation_id')
        try:
            sim = SavedSimulation.objects.get(simulation_id=sim_id)
            sim.delete()
            messages.success(request, f"Simulation {sim_id} deleted successfully.")
            return redirect('saved_simulations')
        except SavedSimulation.DoesNotExist:
            logger.error(f"Simulation {sim_id} not found.")
            messages.error(request, "Error: Simulation not found.")
            return redirect('saved_simulations')
        except Exception as e:
            logger.error(f"Error in delete_simulation: {str(e)}")
            messages.error(request, f"Error deleting simulation: {str(e)}")
            return redirect('saved_simulations')
    return redirect('saved_simulations')