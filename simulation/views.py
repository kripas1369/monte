import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseServerError
from scipy.ndimage import label, center_of_mass
import random
import math
import time
import logging
import csv
from io import StringIO
from .forms import KMCSimulationForm
from typing import Dict, List, Set, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Constants
SIMULATION_PARAMS = {
    'A': 1e10, 'E_a': 1.0, 'k_B': 8.617e-5, 'base_temp': 800, 'critical_size': 4
}

STATES = {
    'EMPTY': 0, 'SUBSTRATE': 1, 'MOBILE': 2, 'STABLE': 3, 'DEFECT': 4,
    'NUCLEATION': 5, 'CLUSTER': 6
}

VISUALIZATION = {
    'colors': ['#FFFFFF', '#4E79A7', '#E15759', '#59A14F', '#F28E2B', '#EDC948', '#B07AA1'],
    'view_angle': (30, 45), 'voxel_alpha': 0.85
}

DIFFUSION = {'x': 0.75, 'y': 0.95, 'z': 1.2}

NUCLEATION = {
    "T_m": 1700.0, "L": 1.0e9, "gamma": 0.3, "theta_deg": 60.0, "A": 1e10
}

STRUCTURE_3D = np.ones((3,3,3), dtype=bool)

class ClusterAnalyzer:
    def __init__(self, critical_size: int):
        self.critical_size = critical_size
        self.cluster_labels = None
        self.num_clusters = 0
        self.cluster_sizes: Dict[int, int] = {}
        self.cluster_properties: Dict[int, dict] = {}

    def update_cluster_info(self, lattice: np.ndarray):
        try:
            binary = (lattice == STATES['MOBILE']).astype(int)
            self.cluster_labels, self.num_clusters = label(binary, structure=STRUCTURE_3D)
            self.cluster_sizes.clear()
            self.cluster_properties.clear()
            for cluster_id in range(1, self.num_clusters + 1):
                mask = (self.cluster_labels == cluster_id)
                indices = np.argwhere(mask)
                self.cluster_sizes[cluster_id] = int(np.sum(mask))
                self.cluster_properties[cluster_id] = {
                    'size': self.cluster_sizes[cluster_id],
                    'center': center_of_mass(mask),
                    'extent': np.ptp(indices, axis=0),
                    'indices': indices
                }
        except Exception as e:
            logger.error(f"Error in ClusterAnalyzer.update_cluster_info: {str(e)}")
            raise

    def get_critical_clusters(self) -> List[dict]:
        return [
            {'id': cid, **props}
            for cid, props in self.cluster_properties.items()
            if props['size'] >= self.critical_size
        ]

    def get_cluster_statistics(self) -> Dict:
        critical = self.get_critical_clusters()
        sizes = list(self.cluster_sizes.values())
        return {
            'total_clusters': self.num_clusters,
            'critical_clusters': critical,
            'largest_size': max(sizes) if sizes else 0,
            'size_distribution': {s: sizes.count(s) for s in set(sizes)}
        }

class RateCalculator:
    def __init__(self, lattice_size: int):
        self.size = lattice_size
        self.neighbor_offsets = {
            'x': [(1,0,0), (-1,0,0)], 'y': [(0,1,0), (0,-1,0)], 'z': [(0,0,1), (0,0,-1)]
        }

    def calculate_total_rates(self, lattice: np.ndarray, temperature: float) -> Dict[str, float]:
        try:
            rates = {'diffuse_x': 0.0, 'diffuse_y': 0.0, 'diffuse_z': 0.0, 'attach': 0.0}
            mobile_atoms = np.argwhere(lattice == STATES['MOBILE'])
            for pos in mobile_atoms:
                for direction in ['x', 'y', 'z']:
                    rates[f'diffuse_{direction}'] += self._calculate_diffusion_rate(
                        tuple(pos), lattice, direction, temperature
                    )
            empty_sites = np.sum(lattice == STATES['EMPTY'])
            rates['attach'] = self._arrhenius_rate(SIMULATION_PARAMS['E_a'], temperature) * empty_sites
            return rates
        except Exception as e:
            logger.error(f"Error in RateCalculator.calculate_total_rates: {str(e)}")
            raise

    def _calculate_diffusion_rate(self, pos: Tuple[int,int,int], lattice: np.ndarray, direction: str, temperature: float) -> float:
        vacant = 0
        for dx, dy, dz in self.neighbor_offsets[direction]:
            nx = (pos[0] + dx) % self.size
            ny = (pos[1] + dy) % self.size
            nz = (pos[2] + dz) % self.size
            if lattice[nx, ny, nz] == STATES['EMPTY']:
                vacant += 1
        if vacant == 0:
            return 0.0
        return self._arrhenius_rate(DIFFUSION[direction], temperature)

    def get_periodic_neighbors(self, pos: Tuple[int,int,int]) -> Dict[str,Tuple[int,int,int]]:
        x, y, z = pos
        return {
            'x+': ((x+1)%self.size, y, z), 'x-': ((x-1)%self.size, y, z),
            'y+': (x, (y+1)%self.size, z), 'y-': (x, (y-1)%self.size, z),
            'z+': (x, y, (z+1)%self.size), 'z-': (x, y, (z-1)%self.size)
        }

    @staticmethod
    def _arrhenius_rate(barrier: float, temperature: float) -> float:
        try:
            return SIMULATION_PARAMS['A'] * np.exp(-barrier / (SIMULATION_PARAMS['k_B'] * temperature))
        except ZeroDivisionError:
            logger.error("Temperature is zero in arrhenius_rate")
            return 0.0

class NucleationCalculator:
    def __init__(self, T_m: float, L: float, gamma: float, theta_deg: float, k_B: float):
        self.T_m = T_m
        self.L = L
        self.gamma = gamma
        self.theta_deg = theta_deg
        self.k_B = k_B

    def compute_undercooling(self, T: float) -> float:
        return self.T_m - T

    def compute_volume_energy(self, delta_T: float) -> float:
        return (self.L * delta_T) / self.T_m

    def compute_critical_radius(self, delta_Gv: float) -> float:
        try:
            return (2 * self.gamma) / delta_Gv
        except ZeroDivisionError:
            logger.error("Zero volume energy in compute_critical_radius")
            return float('inf')

    def compute_nucleation_barriers(self, delta_T: float) -> Tuple[float, float]:
        try:
            delta_Gv = self.compute_volume_energy(delta_T)
            delta_G_homo = (16 * math.pi * self.gamma**3) / (3 * delta_Gv**2)
            f_theta = self._compute_hetero_factor()
            delta_G_hetero = f_theta * delta_G_homo
            return delta_G_homo, delta_G_hetero
        except Exception as e:
            logger.error(f"Error in compute_nucleation_barriers: {str(e)}")
            raise

    def compute_nucleation_probability(self, delta_G: float, T: float) -> float:
        try:
            return math.exp(-delta_G / (self.k_B * T))
        except ZeroDivisionError:
            logger.error("Temperature is zero in compute_nucleation_probability")
            return 0.0

    def _compute_hetero_factor(self) -> float:
        theta = math.radians(self.theta_deg)
        return (2 + math.cos(theta)) * (1 - math.cos(theta))**2 / 4

class CrystalGrowthSimulation:
    def __init__(self, lattice_size: int, temperature: float, impurity_concentration: float):
        self.lattice_size = lattice_size
        self.temperature = temperature
        self.impurity_concentration = impurity_concentration
        self.lattice = np.zeros((lattice_size,)*3, dtype=np.int8)
        self.empty_sites: Set[Tuple[int, int, int]] = set()
        self.occupied_sites: Set[Tuple[int, int, int]] = set()
        self.rate_calc = RateCalculator(lattice_size)
        self.cluster_analyzer = ClusterAnalyzer(SIMULATION_PARAMS['critical_size'])
        self.nucleation_calc = NucleationCalculator(
            NUCLEATION['T_m'], NUCLEATION['L'], NUCLEATION['gamma'], 
            NUCLEATION['theta_deg'], SIMULATION_PARAMS['k_B']
        )
        self.time = 0.0
        self.step_count = 0
        self.nucleation_count = 0
        self.event_counts = {
            'attach': 0, 'diffuse_x': 0, 'diffuse_y': 0, 'diffuse_z': 0, 'nucleation': 0
        }
        self.history = {
            'lattices': [], 'coverage': [], 'time_steps': [], 'impurity_sites': [], 
            'nucleation_events': [], 'cluster_counts': [], 'event_counts': []
        }
        try:
            self._initialize_lattice()
            self.cluster_analyzer.update_cluster_info(self.lattice)
            self.history['lattices'].append(self.lattice.copy())
        except Exception as e:
            logger.error(f"Error initializing CrystalGrowthSimulation: {str(e)}")
            raise

    def _initialize_lattice(self):
        size = self.lattice_size
        self.empty_sites = set(np.ndindex(self.lattice.shape))
        self.occupied_sites.clear()
        for x, y in np.ndindex((size, size)):
            pos = (x, y, 0)
            self.lattice[pos] = STATES['SUBSTRATE']
            self.occupied_sites.add(pos)
            self.empty_sites.discard(pos)
        seed_pos = (size//2, size//2, 1)
        self.lattice[seed_pos] = STATES['MOBILE']
        self.occupied_sites.add(seed_pos)
        self.empty_sites.discard(seed_pos)
        total_sites = size ** 3
        impurity_sites = int(total_sites * self.impurity_concentration)
        empty_list = list(self.empty_sites)
        try:
            indices = random.sample(empty_list, min(impurity_sites, len(empty_list)))
            for pos in indices:
                self.lattice[pos] = STATES['DEFECT']
                self.occupied_sites.add(pos)
                self.empty_sites.discard(pos)
        except ValueError:
            logger.warning("Not enough empty sites for impurities")

    def execute_simulation_step(self) -> Tuple[np.ndarray, float, str]:
        try:
            rates = self.rate_calc.calculate_total_rates(self.lattice, self.temperature)
            if self.cluster_analyzer.get_critical_clusters():
                rates['nucleation'] = self._calculate_nucleation_rate()
            event_type = self._select_and_execute_event(rates)
            self.cluster_analyzer.update_cluster_info(self.lattice)
            total_rate = sum(rates.values())
            dt = -np.log(random.random()) / total_rate if total_rate > 0 else 0
            self.time += dt
            self.step_count += 1
            return self.lattice, dt, event_type
        except Exception as e:
            logger.error(f"Error in execute_simulation_step: {str(e)}")
            raise

    def _calculate_nucleation_rate(self) -> float:
        try:
            delta_T = self.nucleation_calc.compute_undercooling(self.temperature)
            delta_Gv = self.nucleation_calc.compute_volume_energy(delta_T)
            _, delta_G_hetero = self.nucleation_calc.compute_nucleation_barriers(delta_T)
            total_rate = 0.0
            for cluster in self.cluster_analyzer.get_critical_clusters():
                prob = self.nucleation_calc.compute_nucleation_probability(delta_G_hetero, self.temperature)
                total_rate += NUCLEATION['A'] * prob * cluster['size']
            return total_rate
        except Exception as e:
            logger.error(f"Error in _calculate_nucleation_rate: {str(e)}")
            return 0.0

    def _select_and_execute_event(self, rates: Dict[str, float]) -> str:
        event_groups = [g for g in rates if rates[g] > 0]
        weights = [rates[g] for g in event_groups]
        if not event_groups:
            return 'no_event'
        selected = random.choices(event_groups, weights=weights)[0]
        if selected.startswith('diffuse'):
            return self._execute_diffusion(selected.split('_')[1])
        elif selected == 'attach':
            return self._execute_attachment()
        elif selected == 'nucleation':
            return self._execute_nucleation()
        return 'no_event'

    def _execute_diffusion(self, direction: str) -> str:
        mobile_atoms = [p for p in self.occupied_sites if self.lattice[p] == STATES['MOBILE']]
        if not mobile_atoms:
            return 'no_mobile_atoms'
        pos = random.choice(mobile_atoms)
        possible_moves = []
        for delta in [-1, 1]:
            x, y, z = pos
            if direction == 'x':
                new_pos = ((x + delta) % self.lattice_size, y, z)
            elif direction == 'y':
                new_pos = (x, (y + delta) % self.lattice_size, z)
            else:
                new_pos = (x, y, (z + delta) % self.lattice_size)
            if self.lattice[new_pos] == STATES['EMPTY']:
                possible_moves.append(new_pos)
        if not possible_moves:
            return 'no_available_moves'
        new_pos = random.choice(possible_moves)
        self._move_atom(pos, new_pos)
        event_type = f'diffuse_{direction}'
        self.event_counts[event_type] += 1
        return event_type

    def _execute_attachment(self) -> str:
        candidates = [p for p in self.empty_sites 
                     if any(self.lattice[n] in (STATES['SUBSTRATE'], STATES['STABLE'])
                            for n in self.rate_calc.get_periodic_neighbors(p).values())]
        if not candidates:
            return 'no_attachment_sites'
        pos = random.choice(candidates)
        self.lattice[pos] = STATES['MOBILE']
        self.empty_sites.remove(pos)
        self.occupied_sites.add(pos)
        self.event_counts['attach'] += 1
        return 'attach'

    def _execute_nucleation(self) -> str:
        critical_clusters = self.cluster_analyzer.get_critical_clusters()
        if not critical_clusters:
            return 'no_critical_clusters'
        selected = random.choices(critical_clusters, weights=[c['size'] for c in critical_clusters])[0]
        for idx in selected['indices']:
            self.lattice[tuple(idx)] = STATES['STABLE']
        self.event_counts['nucleation'] += 1
        self.nucleation_count += 1
        return 'nucleation'

    def _move_atom(self, old_pos: Tuple[int, int, int], new_pos: Tuple[int, int, int]):
        self.lattice[new_pos] = self.lattice[old_pos]
        self.lattice[old_pos] = STATES['EMPTY']
        self.occupied_sites.remove(old_pos)
        self.occupied_sites.add(new_pos)
        self.empty_sites.add(old_pos)
        self.empty_sites.remove(new_pos)

    def calculate_coverage(self) -> float:
        return len(self.occupied_sites) / self.lattice.size if self.lattice.size > 0 else 0.0

    def run_simulation(self, steps: int, update_interval: int):
        try:
            for step in range(steps):
                self.execute_simulation_step()
                if step % update_interval == 0:
                    coverage = self.calculate_coverage()
                    cluster_stats = self.cluster_analyzer.get_cluster_statistics()
                    self.history['lattices'].append(self.lattice.copy())
                    self.history['coverage'].append(coverage)
                    self.history['time_steps'].append(step)
                    self.history['impurity_sites'].append(np.sum(self.lattice == STATES['DEFECT']))
                    self.history['nucleation_events'].append(self.nucleation_count)
                    self.history['cluster_counts'].append(len(cluster_stats['critical_clusters']))
                    self.history['event_counts'].append(self.event_counts.copy())
        except Exception as e:
            logger.error(f"Error in run_simulation: {str(e)}")
            raise

def theoretical_nucleation_rate(nucleation_calc, temp):
    try:
        delta_T = nucleation_calc.compute_undercooling(temp)
        _, delta_G_hetero = nucleation_calc.compute_nucleation_barriers(delta_T)
        return NUCLEATION['A'] * nucleation_calc.compute_nucleation_probability(delta_G_hetero, temp)
    except Exception as e:
        logger.error(f"Error in theoretical_nucleation_rate: {str(e)}")
        return 0.0

def run_temperature_sweep(size, steps, temp_range, imp_conc):
    try:
        nucleation_rates = []
        for temp in temp_range:
            sim = CrystalGrowthSimulation(size, temp, imp_conc)
            sim.run_simulation(steps, update_interval=100)
            rate = sim.nucleation_count / sim.time if sim.time > 0 else 0
            nucleation_rates.append(rate)
        return temp_range, nucleation_rates
    except Exception as e:
        logger.error(f"Error in run_temperature_sweep: {str(e)}")
        raise

def plot_lattice(lattice, history, cluster_analyzer, run_sweep=False, temp_range=None, nucleation_rates=None):
    try:
        # Updated subplot layout: 2 rows, 2 cols, with 3D plot larger
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d', 'colspan': 2}, None],
                   [{'type': 'xy'}, {'type': 'xy'}]],
            subplot_titles=('3D Crystal Lattice', 'Cluster Size Distribution', 'Event & Nucleation Rates'),
            row_heights=[0.6, 0.4]
        )

        # 3D Lattice Plot with animation frames
        frames = []
        for i, lat in enumerate(history['lattices']):
            frame_data = []
            mobile = np.argwhere(lat == STATES['MOBILE'])
            stable = np.argwhere(lat == STATES['STABLE'])
            substrate = np.argwhere(lat == STATES['SUBSTRATE'])
            defect = np.argwhere(lat == STATES['DEFECT'])
            if i == len(history['lattices']) - 1:
                temp_analyzer = ClusterAnalyzer(SIMULATION_PARAMS['critical_size'])
                temp_analyzer.update_cluster_info(lat)
                critical_clusters = temp_analyzer.get_critical_clusters()
            else:
                critical_clusters = []

            if len(substrate) > 0:
                frame_data.append(go.Scatter3d(
                    x=substrate[:, 0], y=substrate[:, 1], z=substrate[:, 2],
                    mode='markers', 
                    marker=dict(size=6, color=VISUALIZATION['colors'][STATES['SUBSTRATE']], opacity=0.7),
                    name='Substrate',
                    hovertemplate='Substrate<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
                ))
            if len(mobile) > 0:
                frame_data.append(go.Scatter3d(
                    x=mobile[:, 0], y=mobile[:, 1], z=mobile[:, 2],
                    mode='markers', 
                    marker=dict(size=6, color=VISUALIZATION['colors'][STATES['MOBILE']], opacity=0.7),
                    name='Mobile Atoms',
                    hovertemplate='Mobile Atom<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
                ))
            if len(stable) > 0:
                frame_data.append(go.Scatter3d(
                    x=stable[:, 0], y=stable[:, 1], z=stable[:, 2],
                    mode='markers', 
                    marker=dict(size=6, color=VISUALIZATION['colors'][STATES['STABLE']], opacity=0.7),
                    name='Stable Atoms',
                    hovertemplate='Stable Atom<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
                ))
            if len(defect) > 0:
                frame_data.append(go.Scatter3d(
                    x=defect[:, 0], y=defect[:, 1], z=defect[:, 2],
                    mode='markers', 
                    marker=dict(size=6, color=VISUALIZATION['colors'][STATES['DEFECT']], opacity=0.5),
                    name='Impurities',
                    hovertemplate='Impurity<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
                ))
            for cluster in critical_clusters:
                indices = cluster['indices']
                frame_data.append(go.Scatter3d(
                    x=indices[:, 0], y=indices[:, 1], z=indices[:, 2],
                    mode='markers', 
                    marker=dict(size=7, color=VISUALIZATION['colors'][STATES['NUCLEATION']], opacity=0.9),
                    name=f'Cluster {cluster["id"]} (Size: {cluster["size"]})',
                    hovertemplate='Cluster ID: %{text}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>',
                    text=[f'{cluster["id"]}, Size: {cluster["size"]}' for _ in indices]
                ))
            frames.append(go.Frame(data=frame_data, name=f'step_{i}'))

        # Add final lattice as initial plot
        mobile = np.argwhere(lattice == STATES['MOBILE'])
        stable = np.argwhere(lattice == STATES['STABLE'])
        substrate = np.argwhere(lattice == STATES['SUBSTRATE'])
        defect = np.argwhere(lattice == STATES['DEFECT'])
        critical_clusters = cluster_analyzer.get_critical_clusters()

        if len(substrate) > 0:
            fig.add_trace(go.Scatter3d(
                x=substrate[:, 0], y=substrate[:, 1], z=substrate[:, 2],
                mode='markers', 
                marker=dict(size=6, color=VISUALIZATION['colors'][STATES['SUBSTRATE']], opacity=0.7),
                name='Substrate',
                hovertemplate='Substrate<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
            ), row=1, col=1)
        if len(mobile) > 0:
            fig.add_trace(go.Scatter3d(
                x=mobile[:, 0], y=mobile[:, 1], z=mobile[:, 2],
                mode='markers', 
                marker=dict(size=6, color=VISUALIZATION['colors'][STATES['MOBILE']], opacity=0.7),
                name='Mobile Atoms',
                hovertemplate='Mobile Atom<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
            ), row=1, col=1)
        if len(stable) > 0:
            fig.add_trace(go.Scatter3d(
                x=stable[:, 0], y=stable[:, 1], z=stable[:, 2],
                mode='markers', 
                marker=dict(size=6, color=VISUALIZATION['colors'][STATES['STABLE']], opacity=0.7),
                name='Stable Atoms',
                hovertemplate='Stable Atom<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
            ), row=1, col=1)
        if len(defect) > 0:
            fig.add_trace(go.Scatter3d(
                x=defect[:, 0], y=defect[:, 1], z=defect[:, 2],
                mode='markers', 
                marker=dict(size=6, color=VISUALIZATION['colors'][STATES['DEFECT']], opacity=0.5),
                name='Impurities',
                hovertemplate='Impurity<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>'
            ), row=1, col=1)
        for cluster in critical_clusters:
            indices = cluster['indices']
            fig.add_trace(go.Scatter3d(
                x=indices[:, 0], y=indices[:, 1], z=indices[:, 2],
                mode='markers', 
                marker=dict(size=7, color=VISUALIZATION['colors'][STATES['NUCLEATION']], opacity=0.9),
                name=f'Cluster {cluster["id"]} (Size: {cluster["size"]})',
                hovertemplate='Cluster ID: %{text}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>',
                text=[f'{cluster["id"]}, Size: {cluster["size"]}' for _ in indices]
            ), row=1, col=1)

        # Add animation frames
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
            )]
        )

        # Cluster Size Distribution
        cluster_stats = cluster_analyzer.get_cluster_statistics()
        sizes = list(cluster_stats['size_distribution'].keys())
        counts = list(cluster_stats['size_distribution'].values())
        fig.add_trace(go.Bar(
            x=sizes, y=counts, name='Cluster Sizes', marker_color='#3498DB',
            hovertemplate='Size: %{x}<br>Count: %{y}<extra></extra>'
        ), row=2, col=1)
        fig.update_xaxes(title_text='Cluster Size', row=2, col=1)
        fig.update_yaxes(title_text='Count', row=2, col=1)

        # Event Distribution and Nucleation Rate
        if history['time_steps']:
            event_types = ['attach', 'diffuse_x', 'diffuse_y', 'diffuse_z', 'nucleation']
            colors = ['#E74C3C', '#1ABC9C', '#3498DB', '#9B59B6', '#F1C40F']
            for i, event in enumerate(event_types):
                counts = [history['event_counts'][t].get(event, 0) for t in range(len(history['time_steps']))]
                fig.add_trace(go.Scatter(
                    x=history['time_steps'], y=counts,
                    mode='lines+markers', name=event.replace('_', ' ').title(),
                    line=dict(color=colors[i]),
                    hovertemplate='Step: %{x}<br>Events: %{y}<extra></extra>'
                ), row=2, col=2)
            if run_sweep and temp_range is not None and nucleation_rates is not None:
                fig.add_trace(go.Scatter(
                    x=temp_range, y=nucleation_rates,
                    mode='lines+markers', name='Simulated Nucleation Rate', line=dict(color='#E67E22'),
                    hovertemplate='Temp: %{x}K<br>Rate: %{y:.2e}/s<extra></extra>'
                ), row=2, col=2)
                theoretical_rates = [theoretical_nucleation_rate(NucleationCalculator(
                    NUCLEATION['T_m'], NUCLEATION['L'], NUCLEATION['gamma'], 
                    NUCLEATION['theta_deg'], SIMULATION_PARAMS['k_B']
                ), t) for t in temp_range]
                fig.add_trace(go.Scatter(
                    x=temp_range, y=theoretical_rates,
                    mode='lines', name='Theoretical Nucleation Rate', line=dict(color='#7F8C8D', dash='dash'),
                    hovertemplate='Temp: %{x}K<br>Rate: %{y:.2e}/s<extra></extra>'
                ), row=2, col=2)
            fig.update_xaxes(title_text='Step / Temperature (K)', row=2, col=2)
            fig.update_yaxes(title_text='Event Count / Rate (1/s)', row=2, col=2)

        fig.update_layout(
            height=1000,  # Increased height for larger 3D plot
            width=1200,   # Increased width
            showlegend=True, 
            margin=dict(l=20, r=20, t=80, b=20),
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                camera_eye=dict(x=1.2, y=1.2, z=0.8),  # Closer initial view
                camera_up=dict(x=0, y=0, z=1),
                dragmode='orbit',  # Smooth orbiting for zoom
                aspectmode='cube',
                xaxis=dict(range=[-1, lattice.shape[0]]),
                yaxis=dict(range=[-1, lattice.shape[1]]),
                zaxis=dict(range=[-1, lattice.shape[2]])
            ),
            title=dict(text='Crystal Growth Simulation Results', x=0.5, font=dict(size=24, color='#1F2937')),
            paper_bgcolor='rgba(255,255,255,0.9)',
            plot_bgcolor='rgba(255,255,255,0.9)'
        )
        return fig.to_html(full_html=False)
    except Exception as e:
        logger.error(f"Error in plot_lattice: {str(e)}")
        raise

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
                run_sweep = form.cleaned_data['run_sweep']

                if run_sweep:
                    temp_range = np.linspace(600, 1200, 5)
                    temp_range, nucleation_rates = run_temperature_sweep(size, steps, temp_range, impurity_concentration)
                    sim = CrystalGrowthSimulation(size, temperature, impurity_concentration)
                    sim.run_simulation(steps, update_interval=100)
                    html_plot = plot_lattice(sim.lattice, sim.history, sim.cluster_analyzer, run_sweep, temp_range, nucleation_rates)
                else:
                    sim = CrystalGrowthSimulation(size, temperature, impurity_concentration)
                    sim.run_simulation(steps, update_interval=100)
                    html_plot = plot_lattice(sim.lattice, sim.history, sim.cluster_analyzer)

                execution_time = time.time() - start_time
                occupied = len(sim.occupied_sites)
                impurities = np.sum(sim.lattice == STATES['DEFECT'])
                total_sites = size ** 3
                cluster_stats = sim.cluster_analyzer.get_cluster_statistics()

                result_message = (
                    f"Simulation completed in {execution_time:.2f} seconds\n"
                    f"Occupied sites: {occupied}/{total_sites} ({occupied / total_sites * 100:.1f}%)\n"
                    f"Impurity sites: {impurities} ({impurities / total_sites * 100:.2f}%)\n"
                    f"Final coverage: {sim.history['coverage'][-1] * 100:.1f}%\n"
                    f"Nucleation events: {sim.nucleation_count}\n"
                    f"Total clusters: {cluster_stats['total_clusters']}\n"
                    f"Critical clusters: {len(cluster_stats['critical_clusters'])}"
                )

                metrics = {
                    'execution_time': f"{execution_time:.2f}s",
                    'lattice_size': f"{size}",
                    'steps': steps,
                    'temperature': f"{temperature}K",
                    'coverage_A': f"{sim.history['coverage'][-1] * 100:.1f}%",
                    'coverage_B': f"{impurities / total_sites * 100:.2f}%",
                    'reactions': f"{steps}",
                    'nucleation_events': f"{sim.nucleation_count}",
                    'total_clusters': f"{cluster_stats['total_clusters']}",
                    'critical_clusters': f"{len(cluster_stats['critical_clusters'])}"
                }

                # Store simulation data for download
                request.session['simulation_data'] = {
                    'lattice': sim.lattice.tolist(),
                    'history': {
                        'time_steps': sim.history['time_steps'],
                        'coverage': sim.history['coverage'],
                        'nucleation_events': sim.history['nucleation_events'],
                        'cluster_counts': sim.history['cluster_counts'],
                        'event_counts': sim.history['event_counts']
                    },
                    'cluster_stats': cluster_stats
                }

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
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Template rendering error: {str(e)}")
        return HttpResponseServerError("Failed to render template. Please ensure all dependencies are installed.")

def download_results(request):
    try:
        sim_data = request.session.get('simulation_data', {})
        if not sim_data:
            return HttpResponse("No simulation data available.", status=400)

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Step', 'Coverage', 'Nucleation Events', 'Cluster Count', 'Attach Events', 'Diffuse X', 'Diffuse Y', 'Diffuse Z'])
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
        response['Content-Disposition'] = 'attachment; filename="simulation_results.csv"'
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