def marathon_pace(distance_km, total_time_min):
    pace = total_time_min / distance_km
    print(f"Average pace: {pace:.2f} min/km")

# Example: Marathon is 42.195 km, time is 240 minutes
marathon_pace(42.195, 240)