"""
Message Router for SHIELD inter-agent communication.

Models the communication constraints from Section 3.1:
  - Bidirectional links constrained by range and line-of-sight.
  - Messages can be dropped under jamming (A2, Eq. 2).
  - Byzantine agents may broadcast fabricated state info (A3, Eq. 3).
  - Network intrusion may inject crafted messages (A6).

The router maintains the communication graph and delivers messages
between agents each timestep.
"""

import numpy as np
from typing import Dict, List, Optional, Set
from .node import Message, ShieldNode


class MessageRouter:
    """Routes messages between ShieldNodes with range-limited communication."""

    def __init__(
        self,
        comm_range: float = 15.0,    # metres; arena is 20x20x10
        nodes: Optional[Dict[int, ShieldNode]] = None,
    ):
        self.comm_range = comm_range
        self.nodes: Dict[int, ShieldNode] = nodes or {}

        # Active attack state (set externally by AttackInjector)
        self.jammed_links: Set[tuple] = set()           # (i,j) pairs being jammed
        self.byzantine_ids: Set[int] = set()             # compromised sender IDs
        self.intrusion_segment: Optional[Set[int]] = None  # network segment under A6

    def set_nodes(self, nodes: Dict[int, ShieldNode]):
        self.nodes = nodes

    # ------------------------------------------------------------------
    # Per-timestep message delivery
    # ------------------------------------------------------------------

    def deliver_messages(self, positions: Dict[int, np.ndarray]) -> Dict[int, List[Message]]:
        """Broadcast and deliver messages for all active nodes.

        Parameters
        ----------
        positions : dict mapping node_id -> 3-D position

        Returns
        -------
        received : dict mapping node_id -> list of Message objects received
        """
        # 1. Collect broadcasts from all active (non-quarantined) nodes
        broadcasts: Dict[int, Message] = {}
        for nid, node in self.nodes.items():
            if node.is_quarantined:
                continue
            msg = node.create_broadcast()

            # A3: Byzantine fault - replace state with fabricated data (Eq. 3)
            if nid in self.byzantine_ids:
                if np.random.random() < 0.10:  # p = 0.10 per paper
                    msg = self._fabricate_byzantine_message(msg)

            broadcasts[nid] = msg

        # 2. Deliver to neighbours within comm range
        received: Dict[int, List[Message]] = {nid: [] for nid in self.nodes}

        for sender_id, msg in broadcasts.items():
            sender_pos = positions.get(sender_id)
            if sender_pos is None:
                continue

            for receiver_id, node in self.nodes.items():
                if receiver_id == sender_id or node.is_quarantined:
                    continue
                receiver_pos = positions.get(receiver_id)
                if receiver_pos is None:
                    continue

                # Range check
                dist = float(np.linalg.norm(sender_pos - receiver_pos))
                if dist > self.comm_range:
                    continue

                # A2: Jamming - drop with probability 0.5 (Eq. 2)
                link = (sender_id, receiver_id)
                if link in self.jammed_links or (receiver_id, sender_id) in self.jammed_links:
                    if np.random.random() < 0.5:
                        continue  # message dropped

                # A6: Network intrusion - inject crafted messages
                delivered_msg = msg
                if self.intrusion_segment is not None:
                    if sender_id in self.intrusion_segment:
                        if np.random.random() < 0.05:  # selective injection
                            delivered_msg = self._craft_intrusion_message(msg, receiver_id)

                received[receiver_id].append(delivered_msg)

        return received

    # ------------------------------------------------------------------
    # Attack helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fabricate_byzantine_message(original: Message) -> Message:
        """A3: Replace state with random sample from state distribution (Eq. 3)."""
        fake_state = np.random.uniform(-10.0, 10.0, size=original.state.shape)
        return Message(
            sender_id=original.sender_id,
            state=fake_state,
            belief=original.belief.copy(),
            action_dist=original.action_dist.copy(),
            timestamp=original.timestamp,
            vacuity=original.vacuity,
        )

    @staticmethod
    def _craft_intrusion_message(original: Message, target_id: int) -> Message:
        """A6: Inject crafted state message mimicking legitimate agent."""
        # Slight perturbation to avoid trivial detection
        crafted_state = original.state + np.random.normal(0, 0.5, size=original.state.shape)
        return Message(
            sender_id=original.sender_id,
            state=crafted_state,
            belief=original.belief.copy(),
            action_dist=original.action_dist.copy(),
            timestamp=original.timestamp,
            vacuity=original.vacuity,
        )

    # ------------------------------------------------------------------
    # Attack injection controls (called by AttackInjector)
    # ------------------------------------------------------------------

    def set_jammed_links(self, links: Set[tuple]):
        """Set currently jammed (i,j) pairs for A2."""
        self.jammed_links = links

    def set_byzantine_agents(self, agent_ids: Set[int]):
        """Set compromised agents for A3."""
        self.byzantine_ids = agent_ids

    def set_intrusion_segment(self, segment: Optional[Set[int]]):
        """Set network segment under A6 intrusion."""
        self.intrusion_segment = segment

    def clear_attacks(self):
        """Remove all active attack state."""
        self.jammed_links = set()
        self.byzantine_ids = set()
        self.intrusion_segment = None
