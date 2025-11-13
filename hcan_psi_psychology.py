"""
HCAN-Ψ (Psi): Level 5 - Collective Psychology Layer

This module implements collective market psychology, swarm intelligence, and market consciousness.
Markets are not just physical systems - they have emergent collective behavior.

Components:
1. Swarm Intelligence - Agent-based collective behavior
2. Opinion Dynamics - How beliefs propagate through markets
3. Market Consciousness - Integrated Information Theory (IIT) for markets
4. Herding Behavior - Imitation cascades
5. Sentiment Contagion - Emotional spread dynamics

Author: RD-Agent Research Team
Date: 2025-11-13
Level: 5 (Meta-dynamics + Physics + Psychology)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import pdist, squareform


class SwarmIntelligence:
    """
    Agent-based swarm dynamics for market participants.

    Models:
    - Flocking behavior (alignment, cohesion, separation)
    - Leader-follower dynamics
    - Collective decision-making

    Reference: Reynolds (1987), Boids algorithm
    """

    def __init__(self, n_agents: int = 100, perception_radius: float = 0.1):
        """
        Args:
            n_agents: Number of market agents
            perception_radius: Interaction radius for agent influence
        """
        self.n_agents = n_agents
        self.perception_radius = perception_radius

        # Agent states: [position, velocity, sentiment]
        self.positions = np.random.randn(n_agents, 2) * 0.1  # Price-volume space
        self.velocities = np.random.randn(n_agents, 2) * 0.01
        self.sentiments = np.random.randn(n_agents)  # -1 (bearish) to +1 (bullish)

    def alignment(self, agent_idx: int, neighbors: np.ndarray) -> np.ndarray:
        """
        Alignment: steer towards average velocity of neighbors.

        Args:
            agent_idx: Index of current agent
            neighbors: Indices of neighboring agents

        Returns:
            Alignment force vector [2]
        """
        if len(neighbors) == 0:
            return np.zeros(2)

        avg_velocity = np.mean(self.velocities[neighbors], axis=0)
        return avg_velocity - self.velocities[agent_idx]

    def cohesion(self, agent_idx: int, neighbors: np.ndarray) -> np.ndarray:
        """
        Cohesion: steer towards average position of neighbors.

        Args:
            agent_idx: Index of current agent
            neighbors: Indices of neighboring agents

        Returns:
            Cohesion force vector [2]
        """
        if len(neighbors) == 0:
            return np.zeros(2)

        avg_position = np.mean(self.positions[neighbors], axis=0)
        return avg_position - self.positions[agent_idx]

    def separation(self, agent_idx: int, neighbors: np.ndarray) -> np.ndarray:
        """
        Separation: avoid crowding neighbors.

        Args:
            agent_idx: Index of current agent
            neighbors: Indices of neighboring agents

        Returns:
            Separation force vector [2]
        """
        if len(neighbors) == 0:
            return np.zeros(2)

        repulsion = np.zeros(2)
        for neighbor in neighbors:
            diff = self.positions[agent_idx] - self.positions[neighbor]
            dist = np.linalg.norm(diff)
            if dist > 0:
                repulsion += diff / (dist ** 2)

        return repulsion

    def update(self, dt: float = 0.01, weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Update swarm state using flocking rules.

        Args:
            dt: Time step
            weights: (alignment_weight, cohesion_weight, separation_weight)

        Returns:
            Swarm metrics: {polarization, clustering, fragmentation}
        """
        w_align, w_cohesion, w_separation = weights

        # Find neighbors for each agent
        distances = squareform(pdist(self.positions))

        new_velocities = np.copy(self.velocities)

        for i in range(self.n_agents):
            neighbors = np.where((distances[i] < self.perception_radius) & (distances[i] > 0))[0]

            # Compute forces
            align_force = self.alignment(i, neighbors)
            cohesion_force = self.cohesion(i, neighbors)
            separation_force = self.separation(i, neighbors)

            # Update velocity
            acceleration = (
                w_align * align_force +
                w_cohesion * cohesion_force +
                w_separation * separation_force
            )
            new_velocities[i] += acceleration * dt

            # Speed limit
            speed = np.linalg.norm(new_velocities[i])
            if speed > 0.1:
                new_velocities[i] = new_velocities[i] / speed * 0.1

        # Update positions
        self.velocities = new_velocities
        self.positions += self.velocities * dt

        # Compute swarm metrics
        metrics = self._compute_swarm_metrics()
        return metrics

    def _compute_swarm_metrics(self) -> Dict[str, float]:
        """
        Compute collective behavior metrics.

        Returns:
            Dictionary with:
            - polarization: Alignment of velocities (0=random, 1=aligned)
            - clustering: Spatial concentration (0=dispersed, 1=clustered)
            - fragmentation: Number of disconnected groups
        """
        # Polarization: average normalized velocity
        if self.n_agents == 0:
            return {'polarization': 0.0, 'clustering': 0.0, 'fragmentation': 0.0}

        avg_velocity = np.mean(self.velocities, axis=0)
        polarization = np.linalg.norm(avg_velocity) / 0.1  # Normalize by max speed

        # Clustering: inverse of average distance
        distances = squareform(pdist(self.positions))
        avg_distance = np.mean(distances)
        clustering = 1.0 / (1.0 + avg_distance)

        # Fragmentation: count connected components
        adjacency = (distances < self.perception_radius).astype(int)
        n_components = self._count_connected_components(adjacency)
        fragmentation = n_components / self.n_agents

        return {
            'polarization': polarization,
            'clustering': clustering,
            'fragmentation': fragmentation
        }

    def _count_connected_components(self, adjacency: np.ndarray) -> int:
        """Count connected components using BFS."""
        visited = np.zeros(self.n_agents, dtype=bool)
        n_components = 0

        for i in range(self.n_agents):
            if not visited[i]:
                # BFS from i
                queue = [i]
                visited[i] = True
                while queue:
                    node = queue.pop(0)
                    neighbors = np.where(adjacency[node] > 0)[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                n_components += 1

        return n_components

    def sentiment_dynamics(self, returns: np.ndarray, influence_strength: float = 0.1):
        """
        Update agent sentiments based on returns and peer influence.

        Args:
            returns: Market returns
            influence_strength: Strength of peer influence

        Returns:
            Average sentiment [-1, 1]
        """
        # Returns influence sentiment
        returns_effect = np.tanh(returns * 10)  # Bounded response

        # Peer influence
        distances = squareform(pdist(self.positions))
        adjacency = (distances < self.perception_radius).astype(float)
        adjacency = adjacency / (adjacency.sum(axis=1, keepdims=True) + 1e-6)  # Normalize

        peer_sentiments = adjacency @ self.sentiments

        # Update sentiments
        self.sentiments = (
            (1 - influence_strength) * self.sentiments +
            influence_strength * peer_sentiments +
            0.1 * returns_effect
        )
        self.sentiments = np.clip(self.sentiments, -1, 1)

        return np.mean(self.sentiments)


class OpinionDynamics:
    """
    Opinion dynamics models for market beliefs.

    Models:
    - Voter model (discrete opinions)
    - DeGroot model (continuous opinions)
    - Bounded confidence (agents only influenced by similar opinions)

    Reference: Acemoglu & Ozdaglar (2011)
    """

    def __init__(self, n_agents: int = 100, confidence_threshold: float = 0.3):
        """
        Args:
            n_agents: Number of agents
            confidence_threshold: Max opinion difference for influence
        """
        self.n_agents = n_agents
        self.confidence_threshold = confidence_threshold
        self.opinions = np.random.randn(n_agents)  # Continuous opinions

    def degroot_update(self, adjacency_matrix: np.ndarray, n_iterations: int = 10):
        """
        DeGroot model: opinions converge to weighted average.

        x(t+1) = A @ x(t)

        Args:
            adjacency_matrix: Influence weights [n_agents, n_agents]
            n_iterations: Number of update steps

        Returns:
            Final consensus opinion
        """
        # Normalize adjacency to stochastic matrix
        row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
        A = adjacency_matrix / (row_sums + 1e-6)

        opinions = np.copy(self.opinions)
        for _ in range(n_iterations):
            opinions = A @ opinions

        self.opinions = opinions
        return np.mean(opinions)

    def bounded_confidence_update(self):
        """
        Hegselmann-Krause model: agents only influenced by similar opinions.

        Returns:
            Number of opinion clusters formed
        """
        new_opinions = np.copy(self.opinions)

        for i in range(self.n_agents):
            # Find agents within confidence threshold
            similar_agents = np.where(
                np.abs(self.opinions - self.opinions[i]) < self.confidence_threshold
            )[0]

            # Update to average of similar opinions
            new_opinions[i] = np.mean(self.opinions[similar_agents])

        self.opinions = new_opinions

        # Count clusters
        n_clusters = self._count_opinion_clusters()
        return n_clusters

    def _count_opinion_clusters(self, tolerance: float = 0.1) -> int:
        """Count number of distinct opinion clusters."""
        sorted_opinions = np.sort(self.opinions)
        gaps = np.diff(sorted_opinions)
        n_clusters = 1 + np.sum(gaps > tolerance)
        return n_clusters

    def polarization_index(self) -> float:
        """
        Measure opinion polarization.

        Returns:
            Polarization score (0=consensus, 1=maximum disagreement)
        """
        variance = np.var(self.opinions)
        max_variance = 4.0  # Max variance for opinions in [-2, 2]
        polarization = variance / max_variance
        return polarization


class MarketConsciousness:
    """
    Integrated Information Theory (IIT) applied to markets.

    Market consciousness measured by:
    - Φ (Phi): Integrated information
    - Causal density: How much each part influences the whole
    - Differentiation: Number of distinguishable states

    Reference: Tononi (2004), IIT 3.0
    """

    def __init__(self, n_components: int = 10):
        """
        Args:
            n_components: Number of market components (e.g., sectors)
        """
        self.n_components = n_components

    def integrated_information(self, state: np.ndarray, transition_matrix: np.ndarray) -> float:
        """
        Compute Φ (Phi) - integrated information.

        Φ = Effective Information - Sum of partitioned information

        Args:
            state: Current state vector [n_components]
            transition_matrix: State transition probabilities [n_states, n_states]

        Returns:
            Φ value (higher = more integrated/conscious)
        """
        # Simplified Φ calculation
        # Full IIT Φ is computationally expensive

        # Effective information: entropy reduction by transitions
        state_entropy = self._entropy(state)

        # Mutual information between past and future
        # MI(past; future) = H(future) - H(future|past)
        mi_full = self._mutual_information_full_system(state, transition_matrix)

        # Sum of partitioned mutual information
        mi_partitioned = self._mutual_information_partitioned(state, transition_matrix)

        # Φ = integrated - partitioned
        phi = max(0, mi_full - mi_partitioned)

        return phi

    def _entropy(self, state: np.ndarray) -> float:
        """Shannon entropy of state."""
        # Discretize state
        hist, _ = np.histogram(state, bins=10, density=True)
        hist = hist[hist > 0]
        return scipy_entropy(hist)

    def _mutual_information_full_system(self, state: np.ndarray, transition_matrix: np.ndarray) -> float:
        """Mutual information for full system."""
        # Approximate MI using correlation
        # True MI requires joint distributions
        correlation = np.corrcoef(state, transition_matrix[0])[0, 1]
        mi = -0.5 * np.log(1 - correlation**2 + 1e-6)
        return mi

    def _mutual_information_partitioned(self, state: np.ndarray, transition_matrix: np.ndarray) -> float:
        """Sum of MI for all bipartitions."""
        # Simplified: use independence approximation
        # True calculation requires all possible bipartitions
        n = len(state)

        # Just compute for one bipartition (middle split)
        mid = n // 2
        state_a = state[:mid]
        state_b = state[mid:]

        mi_a = self._entropy(state_a)
        mi_b = self._entropy(state_b)

        return (mi_a + mi_b) / 2

    def causal_density(self, correlation_matrix: np.ndarray) -> float:
        """
        Measure how densely interconnected the system is.

        Args:
            correlation_matrix: Correlations between components [n, n]

        Returns:
            Causal density [0, 1]
        """
        # Density = average absolute correlation
        n = correlation_matrix.shape[0]
        upper_triangle = np.triu(np.abs(correlation_matrix), k=1)
        density = 2 * np.sum(upper_triangle) / (n * (n - 1))
        return density


class HerdingBehavior:
    """
    Herding and imitation cascades.

    Models:
    - Informational cascades
    - Threshold models of collective action
    - Social proof dynamics

    Reference: Bikhchandani et al. (1992)
    """

    def __init__(self, n_agents: int = 100):
        """
        Args:
            n_agents: Number of agents
        """
        self.n_agents = n_agents
        self.private_signals = np.random.randn(n_agents)  # Private information
        self.actions = np.zeros(n_agents)  # 0=sell, 1=buy
        self.thresholds = np.random.uniform(0.3, 0.7, n_agents)  # Herding threshold

    def information_cascade(self) -> Tuple[float, bool]:
        """
        Simulate information cascade.

        Agents observe:
        - Private signal (quality)
        - Actions of previous agents

        Returns:
            (herding_ratio, cascade_occurred)
        """
        for i in range(self.n_agents):
            # Private signal
            private_vote = 1 if self.private_signals[i] > 0 else 0

            # Observe previous actions
            if i > 0:
                previous_actions = self.actions[:i]
                public_vote = 1 if np.mean(previous_actions) > 0.5 else 0

                # Weight between private and public
                # If enough others have acted, ignore private signal
                public_weight = i / (i + 5)  # Increases with more observations

                # Decision
                if np.random.rand() < public_weight:
                    self.actions[i] = public_vote  # Follow the crowd
                else:
                    self.actions[i] = private_vote  # Use private info
            else:
                self.actions[i] = private_vote

        # Measure herding
        # True aggregate should be based on private signals
        true_aggregate = np.mean(self.private_signals > 0)
        observed_aggregate = np.mean(self.actions)

        herding_ratio = np.abs(observed_aggregate - true_aggregate)
        cascade_occurred = herding_ratio > 0.2

        return herding_ratio, cascade_occurred

    def threshold_model(self, initial_adopters_fraction: float = 0.1) -> int:
        """
        Threshold model: agents act when enough neighbors have acted.

        Args:
            initial_adopters_fraction: Initial fraction of adopters

        Returns:
            Final number of adopters
        """
        # Initialize
        n_initial = int(self.n_agents * initial_adopters_fraction)
        adopters = np.zeros(self.n_agents, dtype=bool)
        adopters[:n_initial] = True

        # Iterate until convergence
        changed = True
        iterations = 0
        max_iterations = 100

        while changed and iterations < max_iterations:
            changed = False
            current_fraction = np.mean(adopters)

            for i in range(self.n_agents):
                if not adopters[i]:
                    # Adopt if fraction exceeds threshold
                    if current_fraction >= self.thresholds[i]:
                        adopters[i] = True
                        changed = True

            iterations += 1

        return np.sum(adopters)


class SentimentContagion:
    """
    Sentiment spread via social networks.

    Models:
    - SIS (Susceptible-Infected-Susceptible) dynamics
    - Emotional contagion
    - Viral spread of fear/greed

    Reference: Dodds & Watts (2004)
    """

    def __init__(self, n_agents: int = 100, infection_rate: float = 0.3, recovery_rate: float = 0.1):
        """
        Args:
            n_agents: Number of agents
            infection_rate: Probability of sentiment transmission
            recovery_rate: Probability of returning to neutral
        """
        self.n_agents = n_agents
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate

        # States: 0=neutral, 1=fearful, 2=greedy
        self.states = np.zeros(n_agents, dtype=int)

    def spread(self, network: np.ndarray, initial_infected: int = 5, n_steps: int = 10) -> Dict[str, float]:
        """
        Simulate sentiment contagion.

        Args:
            network: Adjacency matrix [n_agents, n_agents]
            initial_infected: Number of initially infected agents
            n_steps: Simulation steps

        Returns:
            Final state statistics
        """
        # Initialize with random infected agents
        infected_indices = np.random.choice(self.n_agents, initial_infected, replace=False)
        self.states[infected_indices] = np.random.choice([1, 2], size=initial_infected)

        for step in range(n_steps):
            new_states = np.copy(self.states)

            for i in range(self.n_agents):
                if self.states[i] == 0:  # Susceptible
                    # Check infected neighbors
                    neighbors = np.where(network[i] > 0)[0]
                    infected_neighbors = neighbors[self.states[neighbors] > 0]

                    if len(infected_neighbors) > 0:
                        # Probability of infection increases with infected neighbors
                        p_infection = 1 - (1 - self.infection_rate) ** len(infected_neighbors)

                        if np.random.rand() < p_infection:
                            # Adopt most common sentiment among neighbors
                            neighbor_sentiments = self.states[infected_neighbors]
                            new_states[i] = np.random.choice(neighbor_sentiments)

                elif self.states[i] > 0:  # Infected
                    # Recover to neutral
                    if np.random.rand() < self.recovery_rate:
                        new_states[i] = 0

            self.states = new_states

        # Compute statistics
        n_neutral = np.sum(self.states == 0)
        n_fearful = np.sum(self.states == 1)
        n_greedy = np.sum(self.states == 2)

        return {
            'neutral_fraction': n_neutral / self.n_agents,
            'fear_fraction': n_fearful / self.n_agents,
            'greed_fraction': n_greedy / self.n_agents,
            'infected_fraction': (n_fearful + n_greedy) / self.n_agents
        }


# ====================================================================================================
# PYTORCH NEURAL NETWORK LAYERS
# ====================================================================================================


class CollectiveBehaviorLayer(nn.Module):
    """
    Neural network layer extracting collective psychology features.

    Inputs: Agent positions/velocities/sentiments
    Outputs: Collective behavior features (polarization, clustering, etc.)
    """

    def __init__(self, n_agents: int = 100, feature_dim: int = 32):
        """
        Args:
            n_agents: Number of simulated agents
            feature_dim: Output feature dimension
        """
        super().__init__()
        self.n_agents = n_agents

        # Swarm simulator
        self.swarm = SwarmIntelligence(n_agents=n_agents)

        # Encoder: collective metrics → features
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),  # 3 swarm metrics
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract collective behavior features from market returns.

        Args:
            returns: Market returns [batch_size, seq_len] or [batch_size]

        Returns:
            Dictionary with:
            - polarization: [batch_size]
            - clustering: [batch_size]
            - fragmentation: [batch_size]
            - encoded: [batch_size, feature_dim]
        """
        batch_size = returns.shape[0]

        polarizations = []
        clusterings = []
        fragmentations = []

        for i in range(batch_size):
            # Get returns for this sample
            ret = returns[i].detach().cpu().numpy()
            if ret.ndim > 0:
                ret = ret[-1] if len(ret) > 0 else 0.0  # Use last return

            # Update swarm sentiment based on returns
            self.swarm.sentiment_dynamics(ret)

            # Update swarm positions
            metrics = self.swarm.update()

            polarizations.append(metrics['polarization'])
            clusterings.append(metrics['clustering'])
            fragmentations.append(metrics['fragmentation'])

        # Convert to tensors
        polarization = torch.tensor(polarizations, dtype=torch.float32, device=returns.device).unsqueeze(1)
        clustering = torch.tensor(clusterings, dtype=torch.float32, device=returns.device).unsqueeze(1)
        fragmentation = torch.tensor(fragmentations, dtype=torch.float32, device=returns.device).unsqueeze(1)

        # Encode
        collective_features = torch.cat([polarization, clustering, fragmentation], dim=1)
        encoded = self.encoder(collective_features)

        return {
            'polarization': polarization.squeeze(1),
            'clustering': clustering.squeeze(1),
            'fragmentation': fragmentation.squeeze(1),
            'encoded': encoded
        }


class ConsciousnessLayer(nn.Module):
    """
    Market consciousness measurement layer.

    Computes Φ (integrated information) as a measure of market "awareness".
    High Φ = market is highly integrated and responsive.
    """

    def __init__(self, n_components: int = 10, feature_dim: int = 32):
        """
        Args:
            n_components: Number of market components
            feature_dim: Output feature dimension
        """
        super().__init__()
        self.n_components = n_components

        # Consciousness calculator
        self.consciousness = MarketConsciousness(n_components=n_components)

        # Encoder: consciousness metrics → features
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),  # phi, causal_density
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, correlations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute market consciousness.

        Args:
            correlations: Correlation matrix [batch_size, n_components, n_components]

        Returns:
            Dictionary with:
            - phi: Integrated information [batch_size]
            - causal_density: [batch_size]
            - encoded: [batch_size, feature_dim]
        """
        batch_size = correlations.shape[0]

        phis = []
        densities = []

        for i in range(batch_size):
            corr_matrix = correlations[i].detach().cpu().numpy()

            # Simplified Φ calculation using correlation structure
            # Full IIT requires transition matrices
            state = np.random.randn(self.n_components)  # Placeholder
            transition = corr_matrix  # Approximate

            phi = self.consciousness.integrated_information(state, transition)
            density = self.consciousness.causal_density(corr_matrix)

            phis.append(phi)
            densities.append(density)

        # Convert to tensors
        phi_tensor = torch.tensor(phis, dtype=torch.float32, device=correlations.device).unsqueeze(1)
        density_tensor = torch.tensor(densities, dtype=torch.float32, device=correlations.device).unsqueeze(1)

        # Encode
        consciousness_features = torch.cat([phi_tensor, density_tensor], dim=1)
        encoded = self.encoder(consciousness_features)

        return {
            'phi': phi_tensor.squeeze(1),
            'causal_density': density_tensor.squeeze(1),
            'encoded': encoded
        }


class HerdingLayer(nn.Module):
    """
    Herding behavior detection layer.

    Detects when market is in herding/cascade mode.
    """

    def __init__(self, n_agents: int = 100, feature_dim: int = 32):
        """
        Args:
            n_agents: Number of simulated agents
            feature_dim: Output feature dimension
        """
        super().__init__()
        self.n_agents = n_agents

        # Herding simulator
        self.herding = HerdingBehavior(n_agents=n_agents)

        # Encoder: herding metrics → features
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),  # herding_ratio
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect herding from returns.

        Args:
            returns: Market returns [batch_size, seq_len]

        Returns:
            Dictionary with:
            - herding_ratio: [batch_size]
            - cascade_detected: [batch_size] (boolean)
            - encoded: [batch_size, feature_dim]
        """
        batch_size = returns.shape[0]

        herding_ratios = []
        cascades = []

        for i in range(batch_size):
            ret = returns[i].detach().cpu().numpy()

            # Set private signals based on returns
            self.herding.private_signals = ret[:self.n_agents] if len(ret) >= self.n_agents else np.random.randn(self.n_agents)

            # Simulate cascade
            herding_ratio, cascade = self.herding.information_cascade()

            herding_ratios.append(herding_ratio)
            cascades.append(float(cascade))

        # Convert to tensors
        herding_tensor = torch.tensor(herding_ratios, dtype=torch.float32, device=returns.device).unsqueeze(1)
        cascade_tensor = torch.tensor(cascades, dtype=torch.float32, device=returns.device)

        # Encode
        encoded = self.encoder(herding_tensor)

        return {
            'herding_ratio': herding_tensor.squeeze(1),
            'cascade_detected': cascade_tensor,
            'encoded': encoded
        }


# ====================================================================================================
# VALIDATION
# ====================================================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("COLLECTIVE PSYCHOLOGY LAYER VALIDATION")
    print("=" * 80)

    # 1. Swarm Intelligence
    print("\n1. SWARM INTELLIGENCE")
    print("-" * 80)
    swarm = SwarmIntelligence(n_agents=50)
    for t in range(5):
        metrics = swarm.update()
        print(f"   t={t}: Polarization={metrics['polarization']:.4f}, "
              f"Clustering={metrics['clustering']:.4f}, "
              f"Fragmentation={metrics['fragmentation']:.4f}")

    # 2. Opinion Dynamics
    print("\n2. OPINION DYNAMICS")
    print("-" * 80)
    opinions = OpinionDynamics(n_agents=50)

    # Random adjacency
    adj = np.random.rand(50, 50)
    adj = (adj + adj.T) / 2  # Symmetric
    consensus = opinions.degroot_update(adj, n_iterations=20)
    print(f"   DeGroot consensus: {consensus:.4f}")

    n_clusters = opinions.bounded_confidence_update()
    polarization = opinions.polarization_index()
    print(f"   Opinion clusters: {n_clusters}")
    print(f"   Polarization: {polarization:.4f}")

    # 3. Market Consciousness
    print("\n3. MARKET CONSCIOUSNESS")
    print("-" * 80)
    consciousness = MarketConsciousness(n_components=10)

    state = np.random.randn(10)
    transition = np.random.rand(10, 10)
    transition = transition / transition.sum(axis=1, keepdims=True)

    phi = consciousness.integrated_information(state, transition)

    corr_matrix = np.corrcoef(np.random.randn(10, 100))
    density = consciousness.causal_density(corr_matrix)

    print(f"   Φ (Integrated Information): {phi:.4f}")
    print(f"   Causal Density: {density:.4f}")

    # 4. Herding Behavior
    print("\n4. HERDING BEHAVIOR")
    print("-" * 80)
    herding = HerdingBehavior(n_agents=50)

    herding_ratio, cascade = herding.information_cascade()
    print(f"   Herding ratio: {herding_ratio:.4f}")
    print(f"   Cascade occurred: {cascade}")

    n_adopters = herding.threshold_model(initial_adopters_fraction=0.1)
    print(f"   Final adopters: {n_adopters}/50")

    # 5. Sentiment Contagion
    print("\n5. SENTIMENT CONTAGION")
    print("-" * 80)
    contagion = SentimentContagion(n_agents=50, infection_rate=0.3, recovery_rate=0.1)

    # Random network
    network = (np.random.rand(50, 50) > 0.8).astype(float)
    stats = contagion.spread(network, initial_infected=5, n_steps=10)

    print(f"   Neutral: {stats['neutral_fraction']:.2%}")
    print(f"   Fear: {stats['fear_fraction']:.2%}")
    print(f"   Greed: {stats['greed_fraction']:.2%}")
    print(f"   Total infected: {stats['infected_fraction']:.2%}")

    # 6. PyTorch Layers
    print("\n6. PYTORCH LAYERS")
    print("-" * 80)

    # Collective behavior layer
    collective_layer = CollectiveBehaviorLayer(n_agents=50, feature_dim=32)
    returns = torch.randn(4, 100) * 0.01
    collective_output = collective_layer(returns)
    print(f"   Collective behavior output shape: {collective_output['encoded'].shape}")
    print(f"   Mean polarization: {collective_output['polarization'].mean():.4f}")

    # Consciousness layer
    consciousness_layer = ConsciousnessLayer(n_components=10, feature_dim=32)
    correlations = torch.randn(4, 10, 10)
    consciousness_output = consciousness_layer(correlations)
    print(f"   Consciousness output shape: {consciousness_output['encoded'].shape}")
    print(f"   Mean Φ: {consciousness_output['phi'].mean():.4f}")

    # Herding layer
    herding_layer = HerdingLayer(n_agents=50, feature_dim=32)
    herding_output = herding_layer(returns)
    print(f"   Herding output shape: {herding_output['encoded'].shape}")
    print(f"   Mean herding ratio: {herding_output['herding_ratio'].mean():.4f}")

    print("\n" + "=" * 80)
    print("COLLECTIVE PSYCHOLOGY LAYER VALIDATION COMPLETE")
    print("=" * 80)
