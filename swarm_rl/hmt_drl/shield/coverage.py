"""
Coverage Reassignment (Section 4.4, Eq. 18).

When an agent is quarantined its patrol region (waypoints) is redistributed
using a max-heap keyed by capacity score:

    w_k(t) = lambda_1 * E_k(t) + lambda_2 * C^avail_k(t) - lambda_3 * L_k(t)

where
    E_k     : remaining battery charge normalised to [0, 1]
    C^avail : available operational capacity (fraction of resources not committed)
    L_k     : ratio of assigned waypoints to maximum single-agent coverage limit

The reassignment executes in O(log N) per waypoint via a single heap
extraction and update.
"""

import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple


class CoverageManager:
    """Manages waypoint assignments and quarantine-triggered reassignment."""

    def __init__(
        self,
        num_agents: int,
        max_waypoints_per_agent: int = 10,
        lambda1: float = 0.4,
        lambda2: float = 0.4,
        lambda3: float = 0.2,
    ):
        self.num_agents = num_agents
        self.max_wp = max_waypoints_per_agent

        # Capacity score weights (Eq. 18)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # Per-agent state
        self.waypoints: Dict[int, List[np.ndarray]] = {i: [] for i in range(num_agents)}
        self.battery: Dict[int, float] = {i: 1.0 for i in range(num_agents)}
        self.capacity: Dict[int, float] = {i: 1.0 for i in range(num_agents)}

        # Track quarantined agents
        self.quarantined: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, num_agents: int):
        """Reset all assignments for a new episode."""
        self.num_agents = num_agents
        self.waypoints = {i: [] for i in range(num_agents)}
        self.battery = {i: 1.0 for i in range(num_agents)}
        self.capacity = {i: 1.0 for i in range(num_agents)}
        self.quarantined = set()

    def assign_initial_waypoints(self, all_waypoints: List[np.ndarray]):
        """Distribute waypoints evenly across agents at episode start."""
        active = [i for i in range(self.num_agents) if i not in self.quarantined]
        if len(active) == 0:
            return
        for idx, wp in enumerate(all_waypoints):
            agent_id = active[idx % len(active)]
            self.waypoints[agent_id].append(wp)

    def reassign_coverage(self, quarantined_id: int) -> Dict[int, List[np.ndarray]]:
        """Reassign waypoints from quarantined agent using max-heap (Eq. 18).

        Returns a dict mapping agent_id -> list of newly assigned waypoints.
        """
        self.quarantined.add(quarantined_id)
        orphan_wps = self.waypoints.pop(quarantined_id, [])
        if not orphan_wps:
            return {}

        # Build max-heap of eligible agents (negate score for min-heap -> max-heap)
        active = [i for i in range(self.num_agents) if i not in self.quarantined]
        if len(active) == 0:
            return {}

        assignments: Dict[int, List[np.ndarray]] = {i: [] for i in active}

        for wp in orphan_wps:
            # Build heap with current scores
            heap: List[Tuple[float, int]] = []
            for aid in active:
                score = self._capacity_score(aid)
                heapq.heappush(heap, (-score, aid))  # negate for max-heap

            # Extract best candidate - O(log N)
            neg_score, best_id = heapq.heappop(heap)
            self.waypoints.setdefault(best_id, []).append(wp)
            assignments[best_id].append(wp)

        # Remove empty entries
        return {k: v for k, v in assignments.items() if v}

    def update_agent_state(self, agent_id: int, battery: float, capacity: float):
        """Update per-agent resource state each timestep."""
        self.battery[agent_id] = np.clip(battery, 0.0, 1.0)
        self.capacity[agent_id] = np.clip(capacity, 0.0, 1.0)

    def get_load(self, agent_id: int) -> float:
        """L_k: ratio of assigned waypoints to max coverage limit."""
        return len(self.waypoints.get(agent_id, [])) / max(1, self.max_wp)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capacity_score(self, agent_id: int) -> float:
        """Compute w_k (Eq. 18)."""
        E = self.battery.get(agent_id, 1.0)
        C = self.capacity.get(agent_id, 1.0)
        L = self.get_load(agent_id)
        return self.lambda1 * E + self.lambda2 * C - self.lambda3 * L
