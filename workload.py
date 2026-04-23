"""
workload.py
Synthetic workload generator for scheduler simulation.

Produces realistic task mixes representative of mobile / embedded use-cases:
  - Interactive UI tasks   (short, latency-sensitive)
  - Background sync tasks  (long, deferrable)
  - Media decode tasks     (periodic, deadline-driven)
  - Sensor fusion tasks    (real-time, short deadline)
  - Network I/O tasks      (bursty, mixed intensity)
"""

import random
import math
from dataclasses import dataclass
from typing import List, Tuple
from scheduler_core import Task, TaskPriority


# ─────────────────────────────────────────────────────────────────────────────
# Task Profiles
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskProfile:
    name_prefix: str
    burst_mean_ms: float
    burst_std_ms: float
    priority: TaskPriority
    has_deadline: bool
    deadline_slack_ms: float      # time after arrival before deadline
    cpu_intensity_mean: float
    cpu_intensity_std: float
    memory_mb_mean: float
    inter_arrival_mean_ms: float  # Poisson rate


PROFILES: List[TaskProfile] = [
    TaskProfile(
        name_prefix="UI",
        burst_mean_ms=5,  burst_std_ms=2,
        priority=TaskPriority.HIGH,
        has_deadline=True, deadline_slack_ms=16,    # 60 fps target
        cpu_intensity_mean=0.6, cpu_intensity_std=0.15,
        memory_mb_mean=2.0,
        inter_arrival_mean_ms=16,
    ),
    TaskProfile(
        name_prefix="BG_SYNC",
        burst_mean_ms=50, burst_std_ms=20,
        priority=TaskPriority.LOW,
        has_deadline=False, deadline_slack_ms=0,
        cpu_intensity_mean=0.3, cpu_intensity_std=0.1,
        memory_mb_mean=5.0,
        inter_arrival_mean_ms=200,
    ),
    TaskProfile(
        name_prefix="MEDIA",
        burst_mean_ms=8, burst_std_ms=2,
        priority=TaskPriority.MEDIUM,
        has_deadline=True, deadline_slack_ms=33,    # 30 fps
        cpu_intensity_mean=0.8, cpu_intensity_std=0.1,
        memory_mb_mean=10.0,
        inter_arrival_mean_ms=33,
    ),
    TaskProfile(
        name_prefix="SENSOR",
        burst_mean_ms=2, burst_std_ms=0.5,
        priority=TaskPriority.CRITICAL,
        has_deadline=True, deadline_slack_ms=5,
        cpu_intensity_mean=0.4, cpu_intensity_std=0.1,
        memory_mb_mean=0.5,
        inter_arrival_mean_ms=20,
    ),
    TaskProfile(
        name_prefix="NET_IO",
        burst_mean_ms=15, burst_std_ms=8,
        priority=TaskPriority.MEDIUM,
        has_deadline=False, deadline_slack_ms=0,
        cpu_intensity_mean=0.5, cpu_intensity_std=0.2,
        memory_mb_mean=3.0,
        inter_arrival_mean_ms=100,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class WorkloadGenerator:
    """Generates a synthetic task stream for a given simulation duration."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._id_counter = 1

    def _clamp(self, val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    def _sample_task(self, profile: TaskProfile, arrival_time: float) -> Task:
        burst = abs(self.rng.gauss(profile.burst_mean_ms, profile.burst_std_ms))
        burst = self._clamp(burst, 1.0, profile.burst_mean_ms * 3)

        intensity = self.rng.gauss(profile.cpu_intensity_mean,
                                   profile.cpu_intensity_std)
        intensity = self._clamp(intensity, 0.05, 1.0)

        deadline = (arrival_time + profile.deadline_slack_ms
                    if profile.has_deadline else None)

        task = Task(
            task_id=self._id_counter,
            name=f"{profile.name_prefix}_{self._id_counter:03d}",
            burst_time=round(burst, 2),
            arrival_time=round(arrival_time, 2),
            priority=profile.priority,
            deadline=deadline,
            cpu_intensity=round(intensity, 3),
            memory_footprint=round(
                self.rng.gauss(profile.memory_mb_mean, profile.memory_mb_mean * 0.2), 2
            ),
        )
        self._id_counter += 1
        return task

    def generate(self, duration_ms: float = 500.0,
                 profile_mix: List[Tuple[TaskProfile, float]] = None
                 ) -> List[Task]:
        """
        Generate tasks over `duration_ms`.

        profile_mix: list of (profile, weight) — defaults to equal weights
                     across all built-in profiles.
        """
        if profile_mix is None:
            profile_mix = [(p, 1.0) for p in PROFILES]

        tasks: List[Task] = []
        # For each profile, simulate a Poisson arrival process
        for profile, weight in profile_mix:
            t = 0.0
            lam = profile.inter_arrival_mean_ms / weight
            while t < duration_ms:
                inter = self.rng.expovariate(1.0 / lam)
                t += inter
                if t < duration_ms:
                    tasks.append(self._sample_task(profile, arrival_time=t))

        tasks.sort(key=lambda x: x.arrival_time)
        return tasks

    def generate_stress(self, duration_ms: float = 300.0) -> List[Task]:
        """Heavy workload: mostly MEDIA + UI, high intensity."""
        mix = [
            (PROFILES[0], 3.0),   # UI  — 3× normal rate
            (PROFILES[2], 2.0),   # MEDIA
            (PROFILES[3], 1.5),   # SENSOR
        ]
        return self.generate(duration_ms, mix)

    def generate_idle(self, duration_ms: float = 300.0) -> List[Task]:
        """Light workload: only background tasks."""
        mix = [
            (PROFILES[1], 0.5),   # BG_SYNC  — half rate
            (PROFILES[4], 0.5),   # NET_IO
        ]
        return self.generate(duration_ms, mix)


def summarise_workload(tasks: List[Task]) -> dict:
    """Return a quick summary of a generated task list."""
    if not tasks:
        return {}
    types = {}
    for t in tasks:
        prefix = t.name.rsplit("_", 1)[0]
        types[prefix] = types.get(prefix, 0) + 1
    return {
        "total_tasks": len(tasks),
        "type_counts": types,
        "avg_burst_ms": round(sum(t.burst_time for t in tasks) / len(tasks), 2),
        "avg_intensity": round(sum(t.cpu_intensity for t in tasks) / len(tasks), 3),
        "tasks_with_deadline": sum(1 for t in tasks if t.deadline is not None),
    }
