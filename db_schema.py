# db_schema.py

CUSTOMER_SCHEMA = """
Table 1: products (product_id, product_name, category, price_amount, stock_qty, avg_rating, specs)
Table 2: orders (order_id, customer_id, product_id, quantity, order_date, order_status, promised_eta_hours)
Table 3: customer_reviews (review_id, order_id, product_id, rating, review_title, review_text)
"""

FULL_ADMIN_SCHEMA = """
Table 1: customers (customer_id, customer_name, email, phone, city, state, pincode, is_prime_member)
Table 2: products (product_id, product_name, category, price_amount, stock_qty, avg_rating, specs)
Table 3: orders (order_id, customer_id, product_id, quantity, order_date, origin_city, destination_city, order_status, delivery_priority, promised_eta_hours, actual_delivery_hours)
Table 4: delivery_logs (log_id, order_id, event_type, event_timestamp, location_city, remarks, exception_code)
Table 5: customer_reviews (review_id, order_id, product_id, rating, review_title, review_text)
Table 6: fleet_vehicles (vehicle_id, vehicle_type, model_name, capacity_kg, max_range_km, fleet_status, driver_id, last_service_date)
Table 7: maintenance_history (work_order_id, vehicle_id, service_date, service_type, cost_inr, downtime_hours, failure_reported)
Table 8: drone_telemetry (flight_id, drone_id, flight_duration_min, battery_health_pct, motor_temp_c, maintenance_required)
Table 9: gps_routes (route_id, vehicle_id, planned_distance_km, actual_distance_km, stops_planned, stops_completed)
Table 10: traffic_data (traffic_id, record_date, hour_of_day, city, congestion_index, road_closure)
Table 11: weather_data (weather_id, record_date, city, precipitation_mm, wind_speed_kmph, weather_condition)
"""


def get_schema_for_user(login_type, emp_role=None):
    """Returns database schema based on user type without intermediate RBAC restrictions."""
    if login_type == "Customer":
        return CUSTOMER_SCHEMA

    # Standard Employee / Operations View gets full unified schema access
    return FULL_ADMIN_SCHEMA