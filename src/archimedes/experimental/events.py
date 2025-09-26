#
# Copyright (c) 2025 Pine Tree Labs, LLC.
#
# This file is part of Archimedes
# (see github.com/pinetreelabs/archimedes).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.#
import numpy as np

import dataclasses
from typing import List, Set, Tuple, Dict, Callable


@dataclasses.dataclass
class PeriodicEvent:
    name: str
    period: float
    update: Callable
    offset: float = 0
    priority: int = 0  # Higher priority means earlier execution


@dataclasses.dataclass
class EventScheduler:
    """tree-compatible event scheduler for multirate events"""

    events: List[PeriodicEvent]
    eps: float = 1e-10  # Tolerance for floating-point comparisons

    def next_events(self, time):
        """Compute next event time and events to be executed"""
        next_times = {}

        for i, event in enumerate(self.events):
            # Calculate number of periods since offset
            periods_elapsed = (time - event.offset) / event.period

            # Round up to next period
            next_period = np.ceil(periods_elapsed)
            if np.isclose(periods_elapsed, next_period, atol=self.eps):
                next_period += 1

            # Calculate actual time of next occurrence
            next_time = event.offset + next_period * event.period

            # Group events by time, accounting for floating-point precision
            # by quantizing to the nearest multiple of epsilon
            quantized_time = np.round(next_time / self.eps) * self.eps

            if quantized_time not in next_times:
                next_times[quantized_time] = []
            next_times[quantized_time].append(event)

        if not next_times:
            return None, []

        next_time = min(next_times.keys())
        next_events = next_times[next_time]

        # Sort events by priority
        next_events = sorted(next_events, key=lambda e: e.priority, reverse=True)

        return next_time, next_events
