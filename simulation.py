"""
simulation.py
Top-level simulation runner and benchmarking utilities.

Ties together WorkloadGenerator, EnergyAwareScheduler, and the
comparison baseline (Round-Robin, FCFS) to produce metrics
suitable for reporting and the UI dashboard.
"""

from scheduler_core import EnergyAwareScheduler, Task, TaskPriority
from workload import WorkloadGenerator, summarise_workload
from dvfs import Governor, OPP_TABLE, compare_governors
from thermal import ThermalManager
from typing import List, Dict, Any
import copy
import math


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: Round-Robin scheduler (no DVFS, fixed max frequency)
# ─────────────────────────────────────────────────────────────────────────────

class RoundRobinBaseline:
    """Simple RR scheduler at maximum frequency for comparison."""

    def __init__(self, time_quantum_ms: float = 10.0):
        self.tq = time_quantum_ms
        self.current_time = 0.0
        self.completed: List[Task] = []
        self.total_energy = 0.0
        # Always run at max OPP
        self.opp = OPP_TABLE[-1]

    def run(self, tasks: List[Task]) -> dict:
        queue = sorted(tasks, key=lambda t: t.arrival_time)
        rr_queue = []
        idx = 0

        while rr_queue or idx < len(queue):
            # Admit newly arrived tasks
            while idx < len(queue) and queue[idx].arrival_time <= self.current_time:
                rr_queue.append(queue[idx])
                idx += 1

            if not rr_queue:
                # Advance to next task arrival
                if idx < len(queue):
                    self.current_time = queue[idx].arrival_time
                continue

            task = rr_queue.pop(0)
            if task.start_time is None:
                task.start_time = self.current_time

            work = min(self.tq, task.remaining_time)
            task.remaining_time -= work
            self.current_time += self.tq

            # Energy: always full power
            power_mw = self.opp.total_power_mw(activity=task.cpu_intensity)
            energy = power_mw * (self.tq / 1000.0)
            self.total_energy += energy

            if task.remaining_time <= 0:
                task.finish_time = self.current_time
                self.completed.append(task)
            else:
                rr_queue.append(task)

        tats = [t.turnaround_time for t in self.completed if t.turnaround_time]
        wts  = [t.waiting_time    for t in self.completed if t.waiting_time is not None]
        return {
            "scheduler": "RoundRobin (baseline)",
            "total_tasks": len(self.completed),
            "avg_turnaround_ms": round(sum(tats) / max(len(tats), 1), 2),
            "avg_waiting_ms":    round(sum(wts)  / max(len(wts),  1), 2),
            "total_energy_mj":   round(self.total_energy, 4),
            "max_temp_c":        90.0,   # always at max — estimate
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(scenario: str = "mixed",
                   duration_ms: float = 500.0,
                   seed: int = 42,
                   time_quantum_ms: float = 10.0) -> Dict[str, Any]:
    """
    Run the full simulation for a given scenario.

    Scenarios: 'mixed' | 'stress' | 'idle'
    Returns a dict with results for both schedulers + governor comparison.
    """
    gen = WorkloadGenerator(seed=seed)

    if scenario == "stress":
        tasks = gen.generate_stress(duration_ms)
    elif scenario == "idle":
        tasks = gen.generate_idle(duration_ms)
    else:
        tasks = gen.generate(duration_ms)

    workload_summary = summarise_workload(tasks)

    # --- Energy-aware scheduler ---
    ea_tasks = copy.deepcopy(tasks)
    ea_sched = EnergyAwareScheduler(time_quantum_ms=time_quantum_ms)
    for t in ea_tasks:
        ea_sched.add_task(t)
    ea_stats = ea_sched.run(max_ticks=1000)
    ea_stats["scheduler"] = "EnergyAware (EDF+DVFS)"

    # --- Round-Robin baseline ---
    rr_tasks = copy.deepcopy(tasks)
    rr_sched = RoundRobinBaseline(time_quantum_ms=time_quantum_ms)
    rr_stats = rr_sched.run(rr_tasks)

    # --- Governor comparison ---
    util_trace = [s.get("energy_mj", 0) / max(s.get("energy_mj", 1), 0.0001)
                  for s in ea_stats.get("tick_data", [])]
    util_trace = [min(1.0, abs(u)) for u in util_trace]
    thermal_trace = [1.0] * len(util_trace)  # simplified
    governor_cmp = compare_governors(util_trace or [0.5] * 20, thermal_trace or [1.0] * 20)

    # --- Energy savings ---
    ea_energy = ea_stats.get("total_energy_mj", 0)
    rr_energy = rr_stats.get("total_energy_mj", 1)
    savings_pct = round((1 - ea_energy / max(rr_energy, 1)) * 100, 1)

    return {
        "scenario": scenario,
        "duration_ms": duration_ms,
        "workload": workload_summary,
        "energy_aware": ea_stats,
        "round_robin": rr_stats,
        "governor_comparison": governor_cmp,
        "energy_savings_pct": savings_pct,
        "tick_data": ea_stats.get("tick_data", []),
        "thermal_history": ea_sched.thermal.history,
    }


def run_all_scenarios(duration_ms: float = 400.0) -> Dict[str, Any]:
    """Run all three scenarios and return aggregated results."""
    results = {}
    for scenario in ("mixed", "stress", "idle"):
        results[scenario] = run_simulation(scenario=scenario,
                                           duration_ms=duration_ms)
    return results


if __name__ == "__main__":
    import json
    result = run_simulation(scenario="mixed", duration_ms=300)
    print(json.dumps({
        "workload":       result["workload"],
        "energy_aware":  {k: v for k, v in result["energy_aware"].items()
                          if k != "tick_data"},
        "round_robin":   result["round_robin"],
        "energy_savings": f"{result['energy_savings_pct']}%",
    }, indent=2))
