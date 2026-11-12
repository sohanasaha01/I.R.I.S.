from datetime import datetime

from threat_engine import (
    ThreatEvidence,
    compute_threat,
    crossed_tripwire,
)


def test_low_risk_daytime_vehicle():
    evidence = ThreatEvidence(
        track_id=1,
        object_type="car",
        person_name="Unknown",
        inside_zone=False,
        is_night=False,
        speed_px_s=12,
        previous_speed_px_s=11,
    )
    result = compute_threat(evidence)
    print("TEST 1")
    print(result.score, result.severity, result.reason)
    print()


def test_high_risk_unknown_person():
    evidence = ThreatEvidence(
        track_id=4,
        object_type="person",
        person_name="Unknown",
        inside_zone=True,
        dwell_seconds=45,
        zone_name="Zone A",
        tripwire_crossings=1,
        timestamp=datetime(2026, 9, 11, 2, 14),
        is_night=True,
        speed_px_s=0.5,
        previous_speed_px_s=12,
        trajectory=[
            (100, 100),
            (110, 100),
            (120, 100),
            (118, 100),
            (117, 100),
        ],
        loitering=True,
        loiter_threshold_px=15,
        loiter_seconds=20,
    )
    result = compute_threat(evidence)
    print("TEST 2")
    print(result.score, result.severity)
    print(result.reason)
    print()


def test_tripwire():
    crossed = crossed_tripwire(
        (100, 100),
        (100, 300),
        (50, 200),
        (150, 200),
    )
    print("TEST 3")
    print("Tripwire crossed:", crossed)
    print()


if __name__ == "__main__":
    test_low_risk_daytime_vehicle()
    test_high_risk_unknown_person()
    test_tripwire()
