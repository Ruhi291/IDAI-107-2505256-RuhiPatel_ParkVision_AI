"""
ParkVision AI - insight_logic.py
Takes slot-level predictions and produces occupancy stats,
congestion level, and a plain-language recommendation.
"""


def get_insights(total_slots, occupied_slots):
    if total_slots == 0:
        return {
            "total": 0,
            "occupied": 0,
            "available": 0,
            "occupancy_pct": 0,
            "congestion": "Unknown",
            "recommendation": "No parking slots detected in this image."
        }

    available_slots = total_slots - occupied_slots
    occupancy_pct = (occupied_slots / total_slots) * 100

    if occupancy_pct < 40:
        congestion = "Low"
    elif occupancy_pct <= 75:
        congestion = "Moderate"
    else:
        congestion = "High"

    if occupancy_pct >= 95:
        recommendation = "Parking lot is nearly full. Consider looking for another lot."
    elif occupancy_pct >= 75:
        recommendation = "Parking is filling up fast. Limited spots remaining."
    elif occupancy_pct >= 40:
        recommendation = "Moderate availability. You should be able to find a spot."
    else:
        recommendation = "Plenty of spots available. Go ahead and park."

    return {
        "total": total_slots,
        "occupied": occupied_slots,
        "available": available_slots,
        "occupancy_pct": round(occupancy_pct, 1),
        "congestion": congestion,
        "recommendation": recommendation
    }


if __name__ == "__main__":
    print(get_insights(total_slots=50, occupied_slots=42))
    print(get_insights(total_slots=50, occupied_slots=15))
    print(get_insights(total_slots=50, occupied_slots=48))
