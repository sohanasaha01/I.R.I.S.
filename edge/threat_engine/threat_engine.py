"""
Rule-based, explainable threat scoring engine for border surveillance.

Inputs are evidence already produced by the vision pipeline:
- restricted-zone dwell time
- tripwire crossings
- day/night
- object type
- speed / trajectory anomalies
- loitering

Output:
    0-100 threat score
    severity
    plain-English reasons
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence, Tuple
import math


# -----------------------------
# Configurable scoring weights
# -----------------------------
DEFAULT_WEIGHTS = {
    "dwell": 25.0,
    "tripwire": 20.0,
    "night": 15.0,
    "object": 10.0,
    "trajectory": 20.0,
    "loitering": 10.0,
}

# All weights add up to 100.
assert sum(DEFAULT_WEIGHTS.values()) == 100.0


@dataclass
class ThreatResult:
    score: float
    severity: str
    reasons: List[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return " | ".join(self.reasons) if self.reasons else "No significant threat indicators."


@dataclass
class ThreatEvidence:
    track_id: Optional[int] = None
    object_type: str = "unknown"
    person_name: str = "Unknown"

    # Restricted-zone evidence
    inside_zone: bool = False
    dwell_seconds: float = 0.0
    zone_name: str = "Restricted Zone"

    # Tripwire evidence
    tripwire_crossings: int = 0

    # Time evidence
    timestamp: Optional[datetime] = None
    is_night: Optional[bool] = None

    # Movement evidence
    speed_px_s: float = 0.0
    previous_speed_px_s: float = 0.0
    trajectory: Sequence[Tuple[float, float]] = field(default_factory=list)

    # Loitering evidence
    loitering: bool = False
    loiter_threshold_px: float = 15.0
    loiter_seconds: float = 0.0


def is_night_time(dt: Optional[datetime]) -> bool:
    """Prototype night rule: 18:00-06:00 local time."""
    if dt is None:
        return False
    return dt.hour >= 18 or dt.hour < 6


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def dwell_score(dwell_seconds: float, max_dwell_seconds: float = 60.0) -> float:
    """
    0 at no dwell, 1 at max_dwell_seconds or more.
    """
    if dwell_seconds <= 0:
        return 0.0
    return _clamp(dwell_seconds / max_dwell_seconds)


def tripwire_score(crossings: int) -> float:
    """
    First crossing is the strongest signal; repeated crossings remain suspicious
    but are capped at the category maximum.
    """
    if crossings <= 0:
        return 0.0
    return _clamp(0.65 + 0.15 * (crossings - 1))


def object_score(object_type: str) -> float:
    """
    Person is more relevant to this prototype's intrusion logic than a vehicle,
    while explicit weapon classes are treated as high-risk objects.
    """
    label = (object_type or "").lower()

    if label in {"gun", "bomb", "knife"}:
        return 1.0
    if label == "person":
        return 0.75
    if label in {"car", "tank", "helicopter"}:
        return 0.60
    if label == "number_plate":
        return 0.20
    return 0.30


def trajectory_anomaly_score(
    speed_px_s: float,
    previous_speed_px_s: float,
    trajectory: Sequence[Tuple[float, float]],
    sudden_stop_threshold: float = 2.0,
) -> Tuple[float, List[str]]:
    """
    Detect simple, explainable movement anomalies:
    - sudden stop
    - direction reversal
    - abrupt speed change

    Returns normalized score and reasons.
    """
    reasons = []
    components = []

    # Sudden stop: was moving, now almost stationary.
    if previous_speed_px_s > 8.0 and speed_px_s <= sudden_stop_threshold:
        components.append(1.0)
        reasons.append(
            f"sudden stop ({previous_speed_px_s:.1f}→{speed_px_s:.1f} px/s)"
        )

    # Abrupt speed change.
    if previous_speed_px_s > 0:
        ratio = abs(speed_px_s - previous_speed_px_s) / max(previous_speed_px_s, 1.0)
        if ratio >= 0.75:
            components.append(min(1.0, ratio))
            reasons.append(f"abrupt speed change ({ratio * 100:.0f}%)")

    # Direction reversal from the last 4 movement vectors.
    if len(trajectory) >= 5:
        p1 = trajectory[-5]
        p2 = trajectory[-4]
        p3 = trajectory[-2]
        p4 = trajectory[-1]

        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p4[0] - p3[0], p4[1] - p3[1])

        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)

        if mag1 > 1.0 and mag2 > 1.0:
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            cosine = dot / (mag1 * mag2)
            if cosine < -0.5:
                components.append(0.9)
                reasons.append("trajectory reversal detected")

    if not components:
        return 0.0, []

    return max(components), reasons


def compute_threat(evidence: ThreatEvidence,
                   weights: Optional[dict] = None) -> ThreatResult:
    """
    Compute an explainable 0-100 threat score.

    The score is NOT a machine-learning probability. It is a deterministic
    rule-based risk score built from observable evidence.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    reasons: List[str] = []

    # 1) Dwell
    dwell_norm = dwell_score(evidence.dwell_seconds)
    if evidence.inside_zone and evidence.dwell_seconds > 0:
        reasons.append(
            f"{evidence.object_type.title()} dwelling "
            f"{evidence.dwell_seconds:.0f}s in {evidence.zone_name}"
        )

    # 2) Tripwire
    trip_norm = tripwire_score(evidence.tripwire_crossings)
    if evidence.tripwire_crossings > 0:
        reasons.append(
            f"{evidence.tripwire_crossings} tripwire crossing"
            f"{'s' if evidence.tripwire_crossings != 1 else ''}"
        )

    # 3) Night
    night = evidence.is_night
    if night is None:
        night = is_night_time(evidence.timestamp)
    night_norm = 1.0 if night else 0.0
    if night:
        reasons.append("activity detected during night hours")

    # 4) Object type
    object_norm = object_score(evidence.object_type)
    if evidence.object_type.lower() in {"gun", "bomb", "knife"}:
        reasons.append(f"high-risk object detected: {evidence.object_type}")

    # 5) Trajectory anomaly
    trajectory_norm, trajectory_reasons = trajectory_anomaly_score(
        evidence.speed_px_s,
        evidence.previous_speed_px_s,
        evidence.trajectory,
    )
    reasons.extend(trajectory_reasons)

    # 6) Loitering
    loiter_norm = 1.0 if evidence.loitering else 0.0
    if evidence.loitering:
        reasons.append(
            f"loitering pattern: movement within "
            f"{evidence.loiter_threshold_px:.0f}px for "
            f"{evidence.loiter_seconds:.0f}s"
        )

    raw_score = (
        weights["dwell"] * dwell_norm
        + weights["tripwire"] * trip_norm
        + weights["night"] * night_norm
        + weights["object"] * object_norm
        + weights["trajectory"] * trajectory_norm
        + weights["loitering"] * loiter_norm
    )

    score = round(_clamp(raw_score, 0.0, 100.0), 1)

    # Recognition is deliberately a reason/context field, not an automatic
    # reduction of the threat score. Authorization policy should be handled
    # separately by the pipeline's known/unknown logic.
    identity = evidence.person_name or "Unknown"
    if identity != "Unknown":
        reasons.insert(0, f"enrollment match: {identity}")
    else:
        reasons.insert(0, "no prior enrollment match")

    if score >= 70:
        severity = "CRITICAL"
    elif score >= 50:
        severity = "HIGH"
    elif score >= 25:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    if evidence.track_id is not None:
        reasons.insert(0, f"Track {evidence.track_id}")

    return ThreatResult(
        score=score,
        severity=severity,
        reasons=reasons,
    )


# -----------------------------------------------------------
# Optional tripwire helper
# -----------------------------------------------------------

def _orientation(a, b, c):
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def _on_segment(a, b, p):
    return (
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def segments_intersect(a, b, c, d):
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and \
       (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True

    eps = 1e-9
    if abs(o1) < eps and _on_segment(a, b, c):
        return True
    if abs(o2) < eps and _on_segment(a, b, d):
        return True
    if abs(o3) < eps and _on_segment(c, d, a):
        return True
    if abs(o4) < eps and _on_segment(c, d, b):
        return True

    return False


def crossed_tripwire(
    previous_center: Tuple[float, float],
    current_center: Tuple[float, float],
    tripwire_start: Tuple[float, float],
    tripwire_end: Tuple[float, float],
) -> bool:
    """Return True when the track segment crosses the tripwire line segment."""
    return segments_intersect(
        previous_center,
        current_center,
        tripwire_start,
        tripwire_end,
    )


if __name__ == "__main__":
    # Small standalone demonstration.
    demo = ThreatEvidence(
        track_id=7,
        object_type="person",
        person_name="Unknown",
        inside_zone=True,
        dwell_seconds=45,
        zone_name="Zone A",
        tripwire_crossings=1,
        is_night=True,
        speed_px_s=0.5,
        previous_speed_px_s=12.0,
        trajectory=[(100,100),(110,100),(120,100),(118,100),(117,100)],
        loitering=True,
        loiter_threshold_px=15,
        loiter_seconds=20,
    )

    result = compute_threat(demo)
    print(f"Threat score: {result.score}/100")
    print(f"Severity: {result.severity}")
    print(f"Reason: {result.reason}")
