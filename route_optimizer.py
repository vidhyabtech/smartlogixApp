import ast
import json
import math
import os
import numpy as np
import pandas as pd
import psycopg2
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import pydeck as pdk
import streamlit as st


def get_db_connection():
    """Establish connection to the SmartLogix PostgreSQL database."""
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "smartlogix_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
    )


def extract_route_data_from_db(route_id, num_deliveries=5):
    """Fetches route waypoints or dynamically generates custom stops if route ID is custom."""
    conn = None
    try:
        conn = get_db_connection()
        query = """
            SELECT g.route_id, g.vehicle_id, g.vehicle_type, g.route_date, g.waypoints,
                   g.planned_distance_km, g.stops_planned,
                   f.capacity_kg, f.max_range_km, f.hub_code
            FROM gps_routes g
            LEFT JOIN fleet_vehicles f ON g.vehicle_id = f.vehicle_id
            WHERE g.route_id = %s
        """
        df = pd.read_sql(query, conn, params=(route_id,))
        conn.close()

        # Dynamic custom route generator based on user-selected number of deliveries
        if df.empty:
            base_lat, base_lon = 12.9716, 77.5946
            locations = [(base_lat, base_lon)]  # Index 0: Hub

            import random
            random.seed(42)
            for _ in range(num_deliveries):
                lat = base_lat + random.uniform(-0.06, 0.06)
                lon = base_lon + random.uniform(-0.06, 0.06)
                locations.append((lat, lon))

            locations.append((base_lat, base_lon))  # Return depot at the end

            optimized_order = list(range(len(locations)))

            return {
                "route_id": route_id,
                "vehicle_id": "VEH-CUSTOM-01",
                "vehicle_type": "Truck",
                "route_date": "2026-05-15",
                "total_stops": num_deliveries,
                "origin_hub": "HUB-CUS (Custom Hub)",
                "locations": locations,
                "optimized_order": optimized_order,
                "capacity_kg": 800.0,
                "max_range_km": 400.0,
                "planned_distance_km": float(num_deliveries * 4.2),
            }

        route_row = df.iloc[0]
        waypoints_raw = route_row["waypoints"]

        waypoints_list = []
        if isinstance(waypoints_raw, str):
            try:
                waypoints_list = json.loads(waypoints_raw)
            except Exception:
                try:
                    waypoints_list = ast.literal_eval(waypoints_raw)
                except Exception as err:
                    print(f"Error parsing waypoint string: {err}")
        elif isinstance(waypoints_raw, (list, tuple)):
            waypoints_list = waypoints_raw

        locations = []
        for wp in waypoints_list:
            if isinstance(wp, dict):
                lat = wp.get("lat")
                lon = wp.get("lon")
                locations.append((float(lat or 12.97), float(lon or 77.59)))
            elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
                locations.append((float(wp[0] or 12.97), float(wp[1] or 77.59)))

        if not locations:
            locations = [(12.9716, 77.5946), (12.9352, 77.6245)]

        total_stops = len(locations)
        vehicle_id = route_row.get("vehicle_id")
        safe_veh_id = str(vehicle_id) if pd.notna(vehicle_id) and str(vehicle_id).strip() != "" else "VEH-0594"
        optimized_order = list(range(len(locations))) + [0]

        hub_code = route_row.get("hub_code")
        safe_hub = str(hub_code) if pd.notna(hub_code) and str(hub_code).strip() != "" else "HUB-BEN"

        hub_city_map = {
            "HUB-LUC": "Lucknow", "HUB-DEL": "Delhi", "HUB-CHA": "Chandigarh",
            "HUB-BHU": "Bhubaneswar", "HUB-GUR": "Gurugram", "HUB-HYD": "Hyderabad",
            "HUB-MUM": "Mumbai", "HUB-KOC": "Kochi", "HUB-GUW": "Guwahati",
            "HUB-PUN": "Pune", "HUB-JAI": "Jaipur", "HUB-CHE": "Chennai",
            "HUB-AHM": "Ahmedabad", "HUB-MAD": "Madurai", "HUB-NAG": "Nagpur",
            "HUB-COI": "Coimbatore", "HUB-MYS": "Mysore", "HUB-VIS": "Visakhapatnam",
            "HUB-KOL": "Kolkata", "HUB-BEN": "Bengaluru"
        }
        hub_city = hub_city_map.get(safe_hub, "Bengaluru")

        return {
            "route_id": route_row["route_id"],
            "vehicle_id": safe_veh_id,
            "vehicle_type": str(route_row.get("vehicle_type", "Truck")),
            "route_date": str(route_row.get("route_date", "2024-05-15")),
            "total_stops": total_stops,
            "origin_hub": f"{safe_hub} ({hub_city})",
            "locations": locations,
            "optimized_order": optimized_order,
            "capacity_kg": float(route_row.get("capacity_kg", 500.0) or 500.0),
            "max_range_km": float(route_row.get("max_range_km", 150.0) or 150.0),
            "planned_distance_km": float(route_row.get("planned_distance_km", 0.0) or 0.0),
        }

    except Exception as e:
        if conn:
            conn.close()
        print(f"Error fetching route {route_id}: {e}")
        return {
            "route_id": route_id,
            "vehicle_id": "VEH-CUSTOM-01",
            "vehicle_type": "Truck",
            "route_date": "2026-05-15",
            "total_stops": num_deliveries,
            "origin_hub": "HUB-CUS (Custom Hub)",
            "locations": [(12.9716, 77.5946), (12.9352, 77.6245), (12.9716, 77.5946)],
            "optimized_order": [0, 1, 2],
            "capacity_kg": 800.0,
            "max_range_km": 400.0,
            "planned_distance_km": 15.0,
        }


def optimize_route(route_id="RTE-00002", city="Bengaluru", num_deliveries=5):
    """Main optimization entry point accepting and utilizing num_deliveries."""
    route_info = extract_route_data_from_db(route_id, num_deliveries=num_deliveries)
    locations = route_info["locations"]
    origin_hub = route_info["origin_hub"]
    avg_speed, drain_rate = get_environmental_factors_from_db(city)

    dist_matrix = create_distance_matrix(locations)
    int_dist_matrix = (dist_matrix * 1000).astype(int).tolist()

    data = {"distance_matrix": int_dist_matrix, "num_vehicles": 1, "depot": 0}

    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return route_info

    index = routing.Start(0)
    optimized_order = []
    total_distance_m = 0

    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        optimized_order.append(node)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        total_distance_m += routing.GetArcCostForVehicle(previous_index, index, 0)

    optimized_order.append(manager.IndexToNode(index))

    total_dist_km = round(total_distance_m / 1000.0, 2)
    est_time_mins = round((total_dist_km / max(avg_speed, 1.0)) * 60)
    battery_usage = min(100.0, round(total_dist_km * drain_rate, 1))

    return {
        "route_id": route_info["route_id"],
        "vehicle_id": route_info.get("vehicle_id"),
        "vehicle_type": route_info["vehicle_type"],
        "route_date": route_info["route_date"],
        "origin_hub": origin_hub,
        "total_stops": num_deliveries,
        "optimized_order": optimized_order,
        "total_distance_km": total_dist_km,
        "estimated_time_mins": est_time_mins,
        "battery_usage_pct": battery_usage,
        "avg_speed_kmh": round(avg_speed, 1),
        "locations": locations,
        "capacity_kg": route_info["capacity_kg"],
        "max_range_km": route_info["max_range_km"],
        "planned_distance_km": route_info["planned_distance_km"],
    }


def get_environmental_factors_from_db(city="Bengaluru"):
    """Queries live traffic_data and weather_data tables for speed and wind metrics."""
    effective_speed_kmh = 30.0
    wind_speed_kmh = 10.0

    try:
        conn = get_db_connection()
        traffic_query = (
            "SELECT AVG(avg_speed_kmph) as avg_speed FROM traffic_data WHERE LOWER(city)"
            " = LOWER(%s)"
        )
        traffic_df = pd.read_sql(traffic_query, conn, params=(city,))
        if not traffic_df.empty and pd.notna(traffic_df.iloc[0]["avg_speed"]):
            effective_speed_kmh = float(traffic_df.iloc[0]["avg_speed"])

        weather_query = (
            "SELECT AVG(wind_speed_kmph) as wind_speed FROM weather_data WHERE"
            " LOWER(city) = LOWER(%s)"
        )
        weather_df = pd.read_sql(weather_query, conn, params=(city,))
        if not weather_df.empty and pd.notna(weather_df.iloc[0]["wind_speed"]):
            wind_speed_kmh = float(weather_df.iloc[0]["wind_speed"])

        conn.close()
    except Exception:
        pass

    battery_drain_rate = 2.5 + (wind_speed_kmh * 0.08)
    return effective_speed_kmh, battery_drain_rate


def create_distance_matrix(locations):
    num_points = len(locations)
    dist_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            dx = (locations[i][0] - locations[j][0]) * 111
            dy = (locations[i][1] - locations[j][1]) * 111
            dist_matrix[i][j] = math.sqrt(dx**2 + dy**2)
    return dist_matrix