import uuid
import numpy as np
from scipy.ndimage import label
import random
import math
from typing import Dict, List, Set, Tuple
from django.db import models
from .config import SIMULATION_PARAMS, STATES, NUCLEATION, STRUCTURE_3D
import logging
import pickle

logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    def __init__(self, critical_size: int):
        self.critical_size = critical_size
        self.cluster_labels = None
        self.num_clusters = 0
        self.cluster_sizes: Dict[int, int] = {}
        self.cluster_properties: Dict[int, dict] = {}

    def update_cluster_info(self, occupied_sites: Set[Tuple[int,int,int]], lattice_size: int):
        try:
            binary = np.zeros((lattice_size, lattice_size, lattice_size), dtype=np.int8)
            for pos in occupied_sites:
                if pos[3] == STATES['MOBILE']:
                    binary[pos[0], pos[1], pos[2]] = 1
            self.cluster_labels, self.num_clusters = label(binary, structure=STRUCTURE_3D)
            self.cluster_sizes.clear()
            self.cluster_properties.clear()
            for cluster_id in range(1, self.num_clusters + 1):
                mask = (self.cluster_labels == cluster_id)
                indices = np.argwhere(mask)
                if len(indices) == 0:
                    continue
                extent = np.ptp(indices, axis=0)
                aspect_ratios = {
                    'x/y': extent[0] / extent[1] if extent[1] != 0 else float('inf'),
                    'x/z': extent[0] / extent[2] if extent[2] != 0 else float('inf'),
                    'y/z': extent[1] / extent[2] if extent[2] != 0 else float('inf'),
                }
                self.cluster_sizes[cluster_id] = int(np.sum(mask))
                self.cluster_properties[cluster_id] = {
                    'size': self.cluster_sizes[cluster_id],
                    'center': np.mean(indices, axis=0),
                    'extent': extent,
                    'aspect_ratios': aspect_ratios,
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
        aspect_ratios = [props['aspect_ratios'] for props in self.cluster_properties.values()]
        avg_aspect = {
            'x/y': np.mean([ar['x/y'] for ar in aspect_ratios if ar['x/y'] != float('inf')] or [0]),
            'x/z': np.mean([ar['x/z'] for ar in aspect_ratios if ar['x/z'] != float('inf')] or [0]),
            'y/z': np.mean([ar['y/z'] for ar in aspect_ratios if ar['y/z'] != float('inf')] or [0]),
        }
        # Calculate average island size and relative width distribution
        avg_island_size = np.mean(sizes) if sizes else 0
        if sizes:
            std_dev = np.std(sizes) if len(sizes) > 1 else 0
            relative_width_distribution = std_dev / avg_island_size if avg_island_size > 0 else 1.0
        else:
            relative_width_distribution = 1.0
        return {
            'total_clusters': self.num_clusters,
            'critical_clusters': critical,
            'largest_size': max(sizes) if sizes else 0,
            'size_distribution': {s: sizes.count(s) for s in set(sizes)},
            'avg_aspect_ratios': avg_aspect,
            'avg_island_size': avg_island_size,
            'relative_width_distribution': relative_width_distribution
        }

class RateCalculator:
    # Unchanged from original code
    def __init__(self, lattice_size: int, diffusion_barriers: Dict[str, float], impurity_effect: float, flux: float):
        self.size = lattice_size
        self.diffusion_barriers = diffusion_barriers
        self.impurity_effect = impurity_effect
        self.flux = flux
        self.neighbor_offsets = {
            'x': [(1,0,0), (-1,0,0)], 'y': [(0,1,0), (0,-1,0)], 'z': [(0,0,1), (0,0,-1)]
        }

    def calculate_total_rates(self, occupied_sites: Set[Tuple[int,int,int]], empty_sites: Set[Tuple[int,int,int]], temperature: float) -> Dict[str, float]:
        try:
            rates = {'diffuse_x': 0.0, 'diffuse_y': 0.0, 'diffuse_z': 0.0, 'attach': 0.0}
            for pos in occupied_sites:
                if pos[3] != STATES['MOBILE']:
                    continue
                for direction in ['x', 'y', 'z']:
                    rates[f'diffuse_{direction}'] += self._calculate_diffusion_rate(
                        pos[:3], occupied_sites, direction, temperature
                    )
            rates['attach'] = self.flux * len(empty_sites)
            return rates
        except Exception as e:
            logger.error(f"Error in RateCalculator.calculate_total_rates: {str(e)}")
            raise

    def _calculate_diffusion_rate(self, pos: Tuple[int,int,int], occupied_sites: Set[Tuple[int,int,int]], direction: str, temperature: float) -> float:
        vacant = 0
        impurity_near = False
        for dx, dy, dz in self.neighbor_offsets[direction]:
            nx = (pos[0] + dx) % self.size
            ny = (pos[1] + dy) % self.size
            nz = (pos[2] + dz) % self.size
            neighbor = (nx, ny, nz)
            if not any(s[:3] == neighbor and s[3] != STATES['EMPTY'] for s in occupied_sites):
                vacant += 1
            if any(s[:3] == neighbor and s[3] == STATES['DEFECT'] for s in occupied_sites):
                impurity_near = True
        if vacant == 0:
            return 0.0
        barrier = self.diffusion_barriers[direction]
        if impurity_near:
            barrier += self.impurity_effect
        return self._arrhenius_rate(barrier, temperature)

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
    # Unchanged from original code
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
    # Unchanged from original code
    def __init__(self, lattice_size: int, temperature: float, impurity_concentration: float, diffusion_barriers: Dict[str, float], impurity_effect: float, flux: float):
        self.lattice_size = lattice_size
        self.temperature = temperature
        self.impurity_concentration = impurity_concentration
        self.diffusion_barriers = diffusion_barriers
        self.impurity_effect = impurity_effect
        self.flux = flux
        self.empty_sites: Set[Tuple[int, int, int]] = set()
        self.occupied_sites: Set[Tuple[int, int, int, int]] = set()
        self.rate_calc = RateCalculator(lattice_size, diffusion_barriers, impurity_effect, flux)
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
            'changes': [], 'coverage': [], 'time_steps': [], 'impurity_sites': [], 
            'nucleation_events': [], 'cluster_counts': [], 'event_counts': [],
            'critical_cluster_properties': [], 'lattices': []
        }
        self.cluster_update_interval = 50
        try:
            self._initialize_lattice()
            self.cluster_analyzer.update_cluster_info(self.occupied_sites, lattice_size)
            self._save_lattice_snapshot()
        except Exception as e:
            logger.error(f"Error initializing CrystalGrowthSimulation: {str(e)}")
            raise

    def _initialize_lattice(self):
        size = self.lattice_size
        self.empty_sites = {(x, y, z) for x in range(size) for y in range(size) for z in range(size)}
        self.occupied_sites.clear()
        for x in range(size):
            for y in range(size):
                pos = (x, y, 0)
                self.occupied_sites.add((x, y, 0, STATES['SUBSTRATE']))
                self.empty_sites.discard(pos)
        seed_pos = (size//2, size//2, 1)
        self.occupied_sites.add((seed_pos[0], seed_pos[1], seed_pos[2], STATES['MOBILE']))
        self.empty_sites.discard(seed_pos)
        total_sites = size ** 3
        impurity_sites = int(total_sites * self.impurity_concentration)
        empty_list = list(self.empty_sites)
        try:
            indices = random.sample(empty_list, min(impurity_sites, len(empty_list)))
            for pos in indices:
                self.occupied_sites.add((pos[0], pos[1], pos[2], STATES['DEFECT']))
                self.empty_sites.discard(pos)
        except ValueError:
            logger.warning("Not enough empty sites for impurities")

    def _save_lattice_snapshot(self):
        snapshot = list(self.occupied_sites)
        self.history['lattices'].append(snapshot)
        if len(self.history['lattices']) > 5:
            self.history['lattices'].pop(0)

    def execute_simulation_step(self) -> Tuple[List[Tuple[int,int,int,int]], float, str]:
        try:
            rates = self.rate_calc.calculate_total_rates(self.occupied_sites, self.empty_sites, self.temperature)
            if self.cluster_analyzer.get_critical_clusters():
                rates['nucleation'] = self._calculate_nucleation_rate()
            event_type = self._select_and_execute_event(rates)
            if event_type == 'nucleation' or self.step_count % self.cluster_update_interval == 0:
                self.cluster_analyzer.update_cluster_info(self.occupied_sites, self.lattice_size)
            total_rate = sum(rates.values())
            dt = -np.log(random.random()) / total_rate if total_rate > 0 else 0
            self.time += dt
            self.step_count += 1
            return list(self.occupied_sites), dt, event_type
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
        mobile_atoms = [p for p in self.occupied_sites if p[3] == STATES['MOBILE']]
        if not mobile_atoms:
            return 'no_mobile_atoms'
        pos = random.choice(mobile_atoms)
        possible_moves = []
        for delta in [-1, 1]:
            x, y, z = pos[:3]
            if direction == 'x':
                new_pos = ((x + delta) % self.lattice_size, y, z)
            elif direction == 'y':
                new_pos = (x, (y + delta) % self.lattice_size, z)
            else:
                new_pos = (x, y, (z + delta) % self.lattice_size)
            if new_pos in self.empty_sites:
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
                     if any(self.rate_calc.get_periodic_neighbors(p)[n] in [s[:3] for s in self.occupied_sites if s[3] in (STATES['SUBSTRATE'], STATES['STABLE'])]
                            for n in self.rate_calc.get_periodic_neighbors(p))]
        if not candidates:
            return 'no_attachment_sites'
        pos = random.choice(candidates)
        self.occupied_sites.add((pos[0], pos[1], pos[2], STATES['MOBILE']))
        self.empty_sites.remove(pos)
        self.event_counts['attach'] += 1
        return 'attach'

    def _execute_nucleation(self) -> str:
        critical_clusters = self.cluster_analyzer.get_critical_clusters()
        if not critical_clusters:
            return 'no_critical_clusters'
        selected = random.choices(critical_clusters, weights=[c['size'] for c in critical_clusters])[0]
        for idx in selected['indices']:
            pos = (idx[0], idx[1], idx[2])
            old = next(s for s in self.occupied_sites if s[:3] == pos)
            self.occupied_sites.remove(old)
            self.occupied_sites.add((pos[0], pos[1], pos[2], STATES['STABLE']))
        self.event_counts['nucleation'] += 1
        self.nucleation_count += 1
        return 'nucleation'

    def _move_atom(self, old_pos: Tuple[int, int, int, int], new_pos: Tuple[int, int, int]):
        self.occupied_sites.remove(old_pos)
        self.occupied_sites.add((new_pos[0], new_pos[1], new_pos[2], old_pos[3]))
        self.empty_sites.add(old_pos[:3])
        self.empty_sites.remove(new_pos)

    def calculate_coverage(self) -> float:
        total_sites = self.lattice_size ** 3
        return len(self.occupied_sites) / total_sites if total_sites > 0 else 0.0

    def run_simulation(self, steps: int, update_interval: int):
        try:
            for step in range(steps):
                self.execute_simulation_step()
                if step % update_interval == 0:
                    coverage = self.calculate_coverage()
                    cluster_stats = self.cluster_analyzer.get_cluster_statistics()
                    self.history['coverage'].append(coverage)
                    self.history['time_steps'].append(step)
                    self.history['impurity_sites'].append(sum(1 for s in self.occupied_sites if s[3] == STATES['DEFECT']))
                    self.history['nucleation_events'].append(self.nucleation_count)
                    self.history['cluster_counts'].append(len(cluster_stats['critical_clusters']))
                    self.history['event_counts'].append(self.event_counts.copy())
                    self._save_lattice_snapshot()
        except Exception as e:
            logger.error(f"Error in run_simulation: {str(e)}")
            raise
class SavedSimulation(models.Model):
    simulation_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    lattice_size = models.IntegerField()
    steps = models.IntegerField()
    temperature = models.FloatField()
    impurity_concentration = models.FloatField()
    diffusion_x = models.FloatField()
    diffusion_y = models.FloatField()
    diffusion_z = models.FloatField()
    impurity_effect = models.FloatField()
    flux = models.FloatField()
    run_sweep = models.BooleanField(default=False)
    simulation_data = models.BinaryField()
    html_plot = models.TextField(blank=True, null=True)
    comparison_plot = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def get_simulation_data(self):
        return pickle.loads(self.simulation_data)

    class Meta:
        ordering = ['-created_at']