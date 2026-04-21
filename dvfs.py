"""
dvfs.py
Dynamic Voltage and Frequency Scaling (DVFS) subsystem.

Implements the governor logic that decides which operating point to use
based on CPU utilization, queue depth, and thermal budget.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import math


@dataclass
class OPP:
    """Operating Performance Point."""
    freq_mhz: float
    voltage_mv: float   # millivolts
    static_power_mw: float   # leakage at this voltage

    @property
    def dynamic_power_coefficient(self) -> float:
        """α in P_dyn = α * V^2 * f  (normalised units)."""
        return (self.voltage_mv / 1000.0) ** 2 * (self.freq_mhz / 1800.0)

    def total_power_mw(self, activity: float = 1.0) -> float:
        """Total power at given activity factor (0..1)."""
        return self.static_power_mw + activity * self.dynamic_power_coefficient * 100

    def __str__(self):
        return f"{self.freq_mhz:.0f} MHz @ {self.voltage_mv} mV"


# ARM big.LITTLE-inspired OPP table
OPP_TABLE: List[OPP] = [
    OPP(300,   800,  2.5),
    OPP(600,   900,  5.0),
    OPP(900,  1000,  8.5),
    OPP(1200, 1100, 13.0),
    OPP(1500, 1200, 19.0),
    OPP(1800, 1300, 27.0),
]


class Governor:
    """
    Software CPU frequency governor.

    Supports three policies:
      - 'performance'  : always max frequency
      - 'powersave'    : always min frequency
      - 'ondemand'     : ramp up quickly, ramp down slowly
      - 'schedutil'    : our custom energy-aware policy (default)
    """

    def __init__(self, policy: str = "schedutil"):
        self.policy = policy
        self._idx = 0
        self._util_history: List[float] = []
        self._up_threshold = 0.70
        self._down_threshold = 0.30
        self._history_window = 5

    @property
    def current_opp(self) -> OPP:
        return OPP_TABLE[self._idx]

    def _clamp(self, idx: int) -> int:
        return max(0, min(len(OPP_TABLE) - 1, idx))

    def update(self, utilization: float, thermal_headroom: float,
               deadline_urgency: float) -> OPP:
        """
        Decide next OPP.

        utilization      : 0..1  (fraction of time CPU was busy last quantum)
        thermal_headroom : 0..1  (1 = cool, 0 = at throttle limit)
        deadline_urgency : 0..1  (1 = imminent deadline, 0 = slack)
        """
        self._util_history.append(utilization)
        if len(self._util_history) > self._history_window:
            self._util_history.pop(0)

        if self.policy == "performance":
            self._idx = len(OPP_TABLE) - 1

        elif self.policy == "powersave":
            self._idx = 0

        elif self.policy == "ondemand":
            avg_util = sum(self._util_history) / len(self._util_history)
            if avg_util >= self._up_threshold:
                self._idx = len(OPP_TABLE) - 1   # jump to max
            elif avg_util < self._down_threshold:
                self._idx = self._clamp(self._idx - 1)

        else:  # schedutil (default)
            avg_util = sum(self._util_history) / len(self._util_history)
            # Boost for deadline urgency
            effective_util = min(1.0, avg_util + deadline_urgency * 0.3)
            # Throttle if thermal headroom is low
            thermal_cap = thermal_headroom  # 0..1
            target_util = effective_util * thermal_cap

            # Map to OPP index
            target_idx = math.ceil(target_util * (len(OPP_TABLE) - 1))
            # Smooth: ramp up fast, ramp down slow
            if target_idx > self._idx:
                self._idx = target_idx
            else:
                self._idx = self._clamp(self._idx - 1)

        return self.current_opp

    def energy_estimate_mj(self, opp: OPP, duration_ms: float,
                           activity: float = 1.0) -> float:
        """Estimate energy in milli-joules for executing at `opp` for `duration_ms`."""
        power_mw = opp.total_power_mw(activity)
        return power_mw * (duration_ms / 1000.0)

    def efficiency_score(self) -> float:
        """
        Return a 0..100 score representing energy efficiency.
        Higher = more efficient.  Based on ratio of work done to energy spent.
        """
        if not self._util_history:
            return 100.0
        avg_util = sum(self._util_history) / len(self._util_history)
        opp = self.current_opp
        perf = opp.freq_mhz / OPP_TABLE[-1].freq_mhz
        power_norm = opp.total_power_mw() / OPP_TABLE[-1].total_power_mw()
        if power_norm == 0:
            return 100.0
        return round(min(100.0, (avg_util * perf / power_norm) * 100), 1)


def compare_governors(utilization_trace: List[float],
                      thermal_trace: List[float]) -> dict:
    """
    Run all governors over the same trace and return energy comparison.
    Useful for benchmarking.
    """
    results = {}
    for policy in ("performance", "powersave", "ondemand", "schedutil"):
        gov = Governor(policy=policy)
        total_energy = 0.0
        total_perf = 0.0
        for util, thermal in zip(utilization_trace, thermal_trace):
            opp = gov.update(util, thermal, deadline_urgency=0.0)
            e = gov.energy_estimate_mj(opp, duration_ms=10, activity=util)
            total_energy += e
            total_perf += opp.freq_mhz * util
        results[policy] = {
            "total_energy_mj": round(total_energy, 2),
            "avg_perf_score": round(total_perf / max(len(utilization_trace), 1), 1),
        }
    return results
