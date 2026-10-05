from dotenv import load_dotenv
load_dotenv(override=True)
import math
import os
import random
import time
import uuid
from datetime import datetime
import folium
import joblib
import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from db_manager import fetch_all_orders_df, get_connection, insert_new_customer_order,fetch_products_df
from streamlit_folium import st_folium
from sqlalchemy import create_engine
import psycopg2
import json
from route_optimizer import (
    extract_route_data_from_db,
    get_db_connection,
    optimize_route,
)
import pydeck as pdk
from chatbot import customer_ai_chatbot_response
from rag_assistant import smartlogix_ai_chatbot_response
from ultralytics import YOLO
import PIL.Image
from sqlalchemy import text

# ==========================================
# PAGE CONFIGURATION & SESSION STATE
# ==========================================
st.set_page_config(
    page_title="SmartLogix AI Platform", page_icon="🚚", layout="wide"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "emp_role" not in st.session_state:
    st.session_state.emp_role = "Manager / Admin"
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "delivery_location" not in st.session_state:
    st.session_state.delivery_location = "Porur, Chennai - 600116"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "📊 Dashboard"

def logout():
    st.session_state.clear()
    st.rerun()


USER_CREDENTIALS = {
    "emp@smartlogix.com": {
        "pass": "admin123",
        "role": "Company Employee (Operations)",
    },
    "user@gmail.com": {"pass": "user123", "role": "Customer Account"},
}

# ==========================================
# ML ARTIFACT LOADING & INFERENCE FUNCTIONS
# (Must be defined BEFORE any page logic/calls)
# ==========================================

DB_URI = os.getenv(
    "DATABASE_URL", "postgresql://postgres:password@localhost:5432/smartlogix_db"
)
engine = create_engine(DB_URI)


# ==========================================
# ETA MODEL ARTIFACT LOADING & INFERENCE
# ==========================================

@st.cache_resource
def load_eta_models():
    """Loads trained two-stage ETA ML artifacts directly from the 'eta/' folder."""
    try:
        model_short = joblib.load("eta/model_short_haul.pkl")
        reg_long = joblib.load("eta/model_long_haul_reg.pkl")
        clf_long = joblib.load("eta/model_long_haul_clf.pkl")
        features_short = joblib.load("eta/features_short.pkl")
        features_long = joblib.load("eta/features_long.pkl")
        return model_short, reg_long, clf_long, features_short, features_long, True
    except Exception as e:
        # Fallback if files aren't found in local path
        return None, None, None, [], [], False

# Unpack global ETA models
model_short, reg_long, clf_long, features_short, features_long, eta_models_loaded = load_eta_models()


def compute_live_model_eta(order_row):
    """
    Computes ETA exclusively through your trained ML models:
    - Automatically separates short-haul (<= 50 km) vs long-haul (> 50 km).
    - Applies two-stage classification + regression for long haul.
    """
    if not eta_models_loaded:
        # Fallback if models are not loaded yet
        return float(order_row.get("predicted_eta_hours", 2.3))

    try:
        # Extract features from order row
        distance = float(order_row.get("distance_km", 10.0))
        weight = float(order_row.get("package_weight_kg", 1.0))
        qty = float(order_row.get("quantity", 1))
        speed = float(order_row.get("avg_speed_kmph", 40.0))
        
        # Build single-row dataframe for inference
        df_inf = pd.DataFrame([{
            "distance_km": distance,
            "package_weight_kg": weight,
            "quantity": qty,
            "transport_mode": order_row.get("transport_mode", "Truck"),
            "delivery_priority": order_row.get("delivery_priority", "Standard"),
            "is_fragile": int(order_row.get("is_fragile", 0)),
            "is_hazmat": int(order_row.get("is_hazmat", 0)),
            "cold_chain_required": int(order_row.get("cold_chain_required", 0)),
            "capacity_kg": float(order_row.get("capacity_kg", 500.0)),
            "max_range_km": float(order_row.get("max_range_km", 300.0)),
            "avg_speed_kmph": speed,
            "odometer_km": float(order_row.get("odometer_km", 1000.0)),
            "battery_capacity_wh": float(order_row.get("battery_capacity_wh", 5000.0)),
            "weather_condition_at_dest": order_row.get("weather_condition_at_dest", "Clear"),
            "order_date": order_row.get("order_date", pd.Timestamp.now())
        }])

        # Temporal features
        df_inf["order_date"] = pd.to_datetime(df_inf["order_date"], errors="coerce")
        df_inf["day_of_week"] = df_inf["order_date"].dt.dayofweek.fillna(0)
        df_inf["is_weekend"] = (df_inf["day_of_week"] >= 5).astype(int)
        df_inf["month"] = df_inf["order_date"].dt.month.fillna(1)

        # Physics & interaction features
        df_inf["weight_per_km"] = df_inf["package_weight_kg"] / (df_inf["distance_km"] + 0.001)
        df_inf["expected_base_time_hrs"] = df_inf["distance_km"] / (df_inf["avg_speed_kmph"] + 1.0)

        if distance <= 50:
            # --- SHORT HAUL INFERENCE ---
            X = pd.get_dummies(
                df_inf[[
                    "distance_km", "package_weight_kg", "quantity", "transport_mode",
                    "delivery_priority", "is_fragile", "is_hazmat", "cold_chain_required",
                    "capacity_kg", "max_range_km", "avg_speed_kmph", "odometer_km",
                    "battery_capacity_wh", "weather_condition_at_dest", "day_of_week",
                    "is_weekend", "month", "weight_per_km", "expected_base_time_hrs"
                ]],
                columns=["transport_mode", "delivery_priority", "weather_condition_at_dest"],
                drop_first=False
            ).astype(float)
            
            X = X.reindex(columns=features_short, fill_value=0.0)
            pred_log = model_short.predict(X)[0]
            return float(np.expm1(pred_log))
        else:
            # --- LONG HAUL TWO-STAGE INFERENCE ---
            X_base = pd.get_dummies(
                df_inf[[
                    "distance_km", "package_weight_kg", "quantity", "transport_mode",
                    "delivery_priority", "is_fragile", "is_hazmat", "cold_chain_required",
                    "capacity_kg", "max_range_km", "avg_speed_kmph", "odometer_km",
                    "battery_capacity_wh", "weather_condition_at_dest", "day_of_week",
                    "is_weekend", "month", "weight_per_km", "expected_base_time_hrs"
                ]],
                columns=["transport_mode", "delivery_priority", "weather_condition_at_dest"],
                drop_first=False
            ).astype(float)
            
            base_cols = [c for c in features_long if c != "prob_severe_delay"]
            X_base = X_base.reindex(columns=base_cols, fill_value=0.0)

            # Stage 1: Disruption probability
            prob_severe = float(clf_long.predict_proba(X_base)[:, 1][0])

            # Stage 2: Feature augmentation
            X_aug = X_base.copy()
            X_aug["prob_severe_delay"] = prob_severe
            X_aug = X_aug.reindex(columns=features_long, fill_value=0.0)

            pred_log = reg_long.predict(X_aug)[0]
            return float(np.expm1(pred_log))

    except Exception as ex:
        # Fallback to database value if any inference error occurs
        return float(order_row.get("predicted_eta_hours", 2.3))


@st.cache_data(ttl=30)
def load_live_orders_from_db():
  """Fetches live orders joined with fleet vehicle data from PostgreSQL."""
  query = """
        SELECT 
            O.*, 
            F.capacity_kg, 
            F.max_range_km, 
            F.avg_speed_kmph, 
            F.odometer_km, 
            F.battery_capacity_wh
        FROM orders O
        LEFT JOIN fleet_vehicles F ON O.assigned_vehicle_id = F.vehicle_id
    """
  try:
    df = pd.read_sql(query, con=engine)
  except Exception:
    # Fallback mock record if database connection is inactive during testing
    df = pd.DataFrame([
        {
            "order_item_id": "ORD-100462",
            "order_id": "ORD-100462",
            "transport_mode": "Bike",
            "predicted_eta_hours": 2.3,
            "distance_km": 8.6,
            "assigned_vehicle_id": "VEH-0108",
            "destination_city": "Chandigarh",
            "order_value_inr": 97308,
            "order_date": "2026-07-29 23:12:00",
            "order_status": "in_transit",
            "customer_name": "Fatima Rao",
            "package_weight_kg": 0.83,
            "product_name": "Zenith Smartwatch Plus",
        }
    ])

  df["formatted_date"] = pd.to_datetime(
      df["order_date"], errors="coerce"
  ).dt.strftime("%d Jul, %I:%M %p")
  df["display_status"] = df["order_status"].str.title()
  if "is_delayed" in df.columns:
    df.loc[df["is_delayed"] == True, "display_status"] = "Delayed"
  return df

# Place this helper function near the top of your main dashboard logic
def format_hrs_mins(hours_val):
    total_mins = int(round(hours_val * 60))
    h = total_mins // 60
    m = total_mins % 60
    if h > 0 and m > 0:
        return f"{h}h {m}m"
    elif h > 0:
        return f"{h}h"
    else:
        return f"{m}m"

# Load Artifacts Safely
@st.cache_resource
def load_logistics_artifacts():
  try:
    model = joblib.load("logistics_mode/logistics_mode_classifier.pkl")
    encoder = joblib.load("logistics_mode/logistics_mode_encoder.pkl")
    model_cols = joblib.load("logistics_mode/logistics_mode_features.pkl")
    return model, encoder, model_cols
  except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    return None, None, None


model, encoder, model_cols = load_logistics_artifacts()

#route optimization

# Helper function to populate the route dropdown selector
def fetch_available_routes():
  """Fetches all available route IDs from the database for the dropdown."""
  try:
    conn = get_db_connection()
    query = "SELECT route_id FROM gps_routes ORDER BY route_id"
    df = pd.read_sql(query, conn)
    conn.close()
    return df
  except Exception as e:
    # Fallback dataframe if database connection fails temporarily
    print(f"Database error fetching route list: {e}")
    return pd.DataFrame(
        {
            "route_id": [
                "RTE-00001",
                "RTE-00002",
                "RTE-00003",
                "RTE-00045",
                "RTE-000147",
            ]
        }
    )
# --- HELPER FUNCTION TO FETCH DRONE DATA USING YOUR EXISTING DB PATTERN ---
def fetch_drone_fleet_data():
    """Fetches drone fleet vehicles and telemetry from the PostgreSQL database."""
    try:
        conn = get_db_connection()
        
        # Query specifically for drones from fleet_vehicles
        fleet_query = "SELECT * FROM fleet_vehicles WHERE LOWER(vehicle_type) = 'drone' ORDER BY vehicle_id;"
        drones_fleet = pd.read_sql(fleet_query, conn)
        
        # Query drone telemetry
        telemetry_query = "SELECT * FROM drone_telemetry;"
        telemetry_df = pd.read_sql(telemetry_query, conn)
        
        conn.close()
        return drones_fleet, telemetry_df
        
    except Exception as e:
        print(f"Database error fetching drone data: {e}")
        st.error(f"Database connection error: {e}")
        return pd.DataFrame(), pd.DataFrame()

def render_products_page():
    st.title("Products (Operations View)")

    # --- Custom CSS for uniform image sizing ---
    st.markdown(
        """
        
    """,
        unsafe_allow_html=True,
    )

    # Search bar at the top
    search_query = st.text_input("Search products...", placeholder="Search by name or keyword...", label_visibility="collapsed")

    conn = get_connection()
    if not conn:
        st.error("❌ Database connection failed.")
        return

    try:
        with conn.cursor() as cursor:
            # 1. Fetch categories for dropdown
            cursor.execute("""
                SELECT DISTINCT INITCAP(TRIM(category)) 
                FROM products 
                WHERE category IS NOT NULL AND category != '' 
                ORDER BY 1;
            """)
            db_categories = [row[0] for row in cursor.fetchall()]
            categories = ["All"] + db_categories

            selected_category = st.selectbox("Select Category", categories)
            st.markdown("---")

            # 2. Build base WHERE clause and parameters for filtering
            where_clause = "WHERE 1=1"
            params = []

            if selected_category != "All":
                where_clause += " AND INITCAP(TRIM(category)) = %s"
                params.append(selected_category)

            if search_query:
                where_clause += " AND (product_name ILIKE %s OR category ILIKE %s)"
                params.extend([f"%{search_query}%", f"%{search_query}%"])

            # 3. Get total count of matching rows for pagination
            cursor.execute(f"SELECT COUNT(*) FROM products {where_clause};", tuple(params))
            total_count = cursor.fetchone()[0]

            if total_count == 0:
                st.info("No products found matching your filters.")
                conn.close()
                return

            # 4. Pagination Setup (16 items per page)
            items_per_page = 16
            total_pages = max(1, (total_count + items_per_page - 1) // items_per_page)

            if "prod_page" not in st.session_state:
                st.session_state.prod_page = 1

            if st.session_state.prod_page > total_pages:
                st.session_state.prod_page = total_pages

            offset = (st.session_state.prod_page - 1) * items_per_page

            # 5. Fetch paginated rows in exact database physical (insertion) order
            query = f"""
                SELECT product_id, product_name, category, price_amount, price_currency, stock_qty, avg_rating, image_url 
                FROM products {where_clause}
                ORDER BY ctid ASC
                LIMIT %s OFFSET %s;
            """
            cursor.execute(query, tuple(params) + (items_per_page, offset))
            rows = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description]
            products = [dict(zip(colnames, row)) for row in rows]

        conn.close()

        # Product Grid (4 columns)
        cols = st.columns(4)
        for idx, prod in enumerate(products):
            with cols[idx % 4]:
                with st.container(border=True):
                    img_url = prod.get('image_url')
                    
                    # Custom CSS to enforce uniform image sizing and cropping inside containers
                    st.markdown("""
                        <style>
                        /* Target images specifically inside containers */
                        div[data-testid="stVerticalBlock"] img {
                            object-fit: cover !important;
                            width: 100% !important;
                            height: 180px !important; /* Adjust this height as needed */
                            border-radius: 4px;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    if img_url and str(img_url).startswith("http"):
                        try:
                            st.image(img_url, use_container_width=True)
                        except Exception:
                            st.markdown("📦 *Image unavailable*")
                    else:
                        st.markdown("📦 *No Image Provided*")
                    
                    st.markdown(f"**{prod.get('product_name')}**")
                    
                    # Currency & Price formatting
                    raw_currency = prod.get('price_currency')
                    currency = "₹" if not raw_currency or str(raw_currency).strip().upper() in ["INR", ""] else str(raw_currency).strip().upper()
                    
                    price_val = prod.get('price_amount')
                    formatted_price = f"{price_val:,.2f}" if isinstance(price_val, (int, float)) else str(price_val)
                    
                    st.markdown(f"### {currency} {formatted_price}")
                    st.markdown(f"⭐ {prod.get('avg_rating', '4.5')} / 5")
                    st.markdown(f"📦 **Stock:** {prod.get('stock_qty', 'N/A')}")

        st.markdown("---")

        # --- Pagination Controls at the Bottom ---
        col_p1, col_p2, col_p3 = st.columns([2, 3, 2])
        with col_p2:
            page_options = list(range(1, total_pages + 1))
            selected_page = st.selectbox(
                f"Page {st.session_state.prod_page} of {total_pages} (Total Products: {total_count})",
                options=page_options,
                index=st.session_state.prod_page - 1,
                key="pagination_selectbox"
            )
            
            if selected_page != st.session_state.prod_page:
                st.session_state.prod_page = selected_page
                st.rerun()

    except Exception as e:
        st.error(f"Error loading catalog from database: {e}")
        if conn:
            conn.close()

# ==========================================
# 1. AUTHENTICATION PAGE
# ==========================================
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<h2>SmartLogix.in</h2>", unsafe_allow_html=True)
        st.markdown(
            "<div>Sign-in to access Intelligent Logistics Platform</div>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.subheader("Sign in")
            email_input = st.text_input(
                "Enter email", placeholder="e.g. emp@smartlogix.com"
            )
            password_input = st.text_input("Password", type="password")

            if st.button("Continue"):
                email_clean = email_input.strip().lower()
                if (
                    email_clean in USER_CREDENTIALS
                    and USER_CREDENTIALS[email_clean]["pass"] == password_input
                ):
                    st.session_state.logged_in = True
                    st.session_state.user_email = email_clean
                    st.session_state.user_role = USER_CREDENTIALS[email_clean][
                        "role"
                    ]
                    st.rerun()
                else:
                    st.error("Invalid Email or Password!")

            st.caption(
                "**Demo Credentials:**\n"
                "- Employee: `emp@smartlogix.com` | Pass: `admin123`\n"
                "- Customer: `user@gmail.com` | Pass: `user123`"
            )

    st.stop()

# ==========================================
# 2. LOGGED-IN DASHBOARDS & NAVIGATION
# ==========================================
user_role = st.session_state.get("user_role", "Customer Account")

if user_role == "Company Employee (Operations)":
    with st.sidebar:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] { background-color: #ffffff; }
            .nav-header {
                font-size: 11px; font-weight: 700; color: #8C98A6;
                letter-spacing: 1.2px; margin-top: 20px; margin-bottom: 6px;
                padding-left: 10px; text-transform: uppercase;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        def nav_item(label, page_key):
            is_active = st.session_state.selected_page == page_key
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{page_key}", type=btn_type):
                st.session_state.selected_page = page_key
                st.rerun()

        st.markdown(
            '<div class="nav-header">OPERATIONS</div>', unsafe_allow_html=True
        )
        nav_item("📊 Dashboard", "📊 Dashboard")
        nav_item("📦 Orders", "📦 Orders")
        nav_item("📍 Delivery Tracking", "📍 Delivery Tracking")
        nav_item("🧩 Choose Delivery Mode", "🧩 Choose Delivery Mode")
        nav_item("❇️ Route Optimization", "❇️ Route Optimization")
        nav_item("🚚 Fleet Management", "🚚 Fleet Management")
        nav_item("🛸 Drone Management", "🛸 Drone Management")
        nav_item("🤖 AI Assistant", "🤖 AI Assistant")
        nav_item("🛍️ Products", "🛍️ Products")
        
        st.divider()
        if st.button("🚪 Sign Out", key="sidebar_signout"):
            logout()

    selected_page = st.session_state.selected_page


    if selected_page == "📊 Dashboard":
        st.title("🏢 SmartLogix AI — Operations Dashboard")
        try:
            conn = get_db_connection()
        except Exception as e:
            st.error(f"Database connection failed: {e}")


        # --- 1. Top 5 KPI Metrics (Using exact DDL column names) ---
        try:
            total_orders_df = pd.read_sql("SELECT COUNT(*) as total FROM orders", conn)
            delivered_df = pd.read_sql(
                "SELECT COUNT(*) as delivered FROM orders WHERE order_status ="
                " 'Delivered'",
                conn,
            )
            in_transit_df = pd.read_sql(
                "SELECT COUNT(*) as in_transit FROM orders WHERE order_status = 'In"
                " Transit'",
                conn,
            )
            delayed_df = pd.read_sql(
                "SELECT COUNT(*) as delayed FROM orders WHERE is_delayed = TRUE", conn
            )
            revenue_df = pd.read_sql(
                "SELECT SUM(order_value_inr) as total_rev FROM orders", conn
            )

            total_orders = int(total_orders_df.iloc[0]["total"] or 520)
            delivered_count = int(delivered_df.iloc[0]["delivered"] or 316)
            in_transit_count = int(in_transit_df.iloc[0]["in_transit"] or 136)
            delayed_count = int(delayed_df.iloc[0]["delayed"] or 39)
            total_revenue = float(revenue_df.iloc[0]["total_rev"] or 71000000)
            rev_cr = total_revenue / 10000000  # Convert to Crores
        except Exception:
            total_orders, delivered_count, in_transit_count, delayed_count, rev_cr = (
                520,
                316,
                136,
                39,
                7.10,
            )

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                label="Total Orders",
                value=f"{total_orders:,}",
                delta="18.2% from last month",
            )
        with col2:
            st.metric(
                label="Delivered",
                value=f"{delivered_count:,}",
                delta="12.5% from last month",
            )
        with col3:
            st.metric(
                label="In Transit",
                value=f"{in_transit_count:,}",
                delta="8.4% from last month",
            )
        with col4:
            st.metric(
                label="Delayed",
                value=f"{delayed_count:,}",
                delta="-4.3% from last month",
                delta_color="inverse",
            )
        with col5:
            st.metric(
                label="Total Revenue",
                value=f"₹{rev_cr:.2f} Cr",
                delta="22.6% from last month",
            )

        st.markdown("---")

        # --- 2. Middle Section: Line Chart & Donut Chart ---
        mid_left, mid_right = st.columns([1.3, 1])

        with mid_left:
            st.subheader("Delivery Overview")
            st.caption("Orders by terminal status using `order_date` and `order_status`")

            try:
                trend_query = """
                        SELECT order_date, order_status AS status, COUNT(*) as orders 
                        FROM orders 
                        WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
                        GROUP BY order_date, order_status 
                        ORDER BY order_date ASC;
                    """
                trend_data = pd.read_sql(trend_query, conn)
                if trend_data.empty:
                    raise ValueError("No rows")
            except Exception:
                trend_data = pd.DataFrame({
                "order_date": ["2026-07-19", "2026-07-21", "2026-07-23"],
                "status": ["Delivered", "Delivered", "In Transit"],
                "orders": [12, 8, 4],
            })

            line_chart = (
                alt.Chart(trend_data)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("order_date:N", title=""),
                    y=alt.Y("orders:Q", title=""),
                    color=alt.Color(
                        "status:N",
                        legend=alt.Legend(
                            orient="top", title=None, direction="horizontal"
                        ),
                    ),
                    tooltip=["order_date", "status", "orders"],
                )
                .properties(height=260)
            )
            st.altair_chart(line_chart, use_container_width=True)

        with mid_right:
            st.subheader("Deliveries by Mode")
            st.caption("Grouped by `transport_mode` in `orders` table")

            try:
                mode_query = """
                        SELECT transport_mode AS mode, COUNT(*) as count 
                        FROM orders 
                        GROUP BY transport_mode;
                    """
                mode_data = pd.read_sql(mode_query, conn)
                if mode_data.empty:
                    raise ValueError("No data")
            except Exception:
                mode_data = pd.DataFrame({
                "mode": ["Truck", "Bike", "Drone", "Van", "Air Cargo", "Ship"],
                "count": [38, 18, 16, 15, 11, 3],
            })

            fig = px.pie(
                mode_data,
                values="count",
                names="mode",
                hole=0.65,
                color_discrete_sequence=[
                    "#9b59b6",
                    "#2ecc71",
                    "#3498db",
                    "#e67e22",
                    "#1abc9c",
                    "#e74c3c",
                ],
            )
            fig.update_traces(textinfo="percent", textposition="inside")
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.0),
                margin=dict(t=10, b=10, l=10, r=10),
                height=260,
                annotations=[
                    dict(
                        text=(
                            f"<b>{total_orders}</b><br><span"
                            " style='font-size:10px'>orders</span>"
                        ),
                        x=0.38,
                        y=0.5,
                        showarrow=False,
                        font_size=16,
                    )
                ],
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- 3. Bottom Section: Recent Orders Table & Drone Watchlist ---
        bot_left, bot_right = st.columns([1.3, 1])

        with bot_left:
            st.subheader("Recent Orders")
            st.caption(
                "Joined from `orders` and `customers` using `customer_id` and correct"
                " DDL columns"
            )

            try:
                orders_df = pd.read_sql(
                """
                        SELECT o.order_id AS "ORDER", 
                            c.customer_name AS "CUSTOMER", 
                            o.transport_mode AS "MODE", 
                            o.origin_city || ' → ' || o.destination_city AS "ROUTE", 
                            o.promised_eta_hours || 'h' AS "ETA", 
                            o.order_status AS "STATUS"
                        FROM orders o
                        LEFT JOIN customers c ON o.customer_id = c.customer_id
                        ORDER BY o.order_date DESC LIMIT 5
                    """,
                conn,
            )
                if orders_df.empty:
                    raise ValueError("Empty")
            except Exception:
                orders_df = pd.DataFrame({
                "ORDER": ["ORD-100844", "ORD-100383"],
                "CUSTOMER": ["Arjun Reddy", "Joseph Chowdhury"],
                "MODE": ["Truck", "Bike"],
                "ROUTE": ["Visakhapatnam → Coimbatore", "Mysuru → Bhopal"],
                "ETA": ["23.7h", "2.2h"],
                "STATUS": ["Delivered", "Delivered"],
            })

            st.dataframe(orders_df, use_container_width=True)

        with bot_right:
            st.subheader("Drone Health Watchlist")
            st.caption(
                "maintenance_required drone list"
            )

            try:
                watchlist_df = pd.read_sql(
                    """
                    SELECT f.vehicle_id AS id, 
                           f.hub_code AS hub, 
                           t.battery_health_pct AS health, 
                           t.motor_temp_c AS temp, 
                           f.last_service_date 
                    FROM fleet_vehicles f
                    JOIN drone_telemetry t ON f.vehicle_id = t.drone_id
                    WHERE LOWER(f.vehicle_type) = 'drone' AND t.maintenance_required = 1
                    LIMIT 5
                    """,
                    conn,
                )
                if watchlist_df.empty:
                    raise ValueError("No drones")

                for _, d in watchlist_df.iterrows():
                    st.markdown(
                        f"**{d['id']}**   {d['hub']}",
                        unsafe_allow_html=True,
                    )
                    st.progress(int(d["health"]))
                    st.markdown(
                        f"<p style='font-size: 11px; color: #666; margin-top:"
                        f" -10px;'>Battery Health = {d['health']}% · Last Service:"
                        f" {d['last_service_date']}</p>",
                        unsafe_allow_html=True,
            )
            except Exception:
            # Define fallback list securely here so it never throws a NameError
                drones = [
                    {"id": "DRN-006", "hub": "Mysuru", "health": 63},
                    {"id": "DRN-008", "hub": "Kolkata", "health": 55},
                    {"id": "DRN-010", "hub": "Mumbai", "health": 67},
                ]
                for d in drones:
                    st.markdown(
                    f"**{d['id']}** &nbsp; <span style='color: gray; font-size:"
                    f" 13px;'>{d['hub']}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(d["health"])
                st.markdown(
                    f"<p style='font-size: 11px; color: #666; margin-top:"
                    f" -10px;'>Battery Health = {d['health']}% · Last Service:"
                    " 2026-03-04</p>",
                    unsafe_allow_html=True,
                )

            conn.close()

    elif selected_page == "📦 Orders":
        st.title("My Orders")
        st.caption("Live orders stream connected directly to PostgreSQL `smartlogix` database.")

        @st.cache_data(ttl=30)
        def load_order_data_from_db():
            query = """
                SELECT 
                    O.*, 
                    C.customer_name, 
                    P.product_name,
                    F.avg_speed_kmph
                FROM orders O
                LEFT JOIN customers C ON O.customer_id = C.customer_id
                LEFT JOIN products P ON O.product_id = P.product_id
                LEFT JOIN fleet_vehicles F ON O.assigned_vehicle_id = F.vehicle_id
            """
            try:
                df = pd.read_sql(query, con=engine)
            except Exception:
                # Fallback mock dataframe if database connection fails during local testing
                df = pd.DataFrame([{
                    "order_id": "ORD-100462",
                    "customer_name": "Fatima Rao",
                    "product_name": "Zenith Smartwatch Plus",
                    "quantity": 1,
                    "origin_city": "Surat",
                    "destination_city": "Chandigarh",
                    "transport_mode": "Bike",
                    "order_status": "in_transit",
                    "is_delayed": False,
                    "order_date": "2026-07-29 23:12:00",
                    "distance_km": 8.6,
                    "package_weight_kg": 0.83,
                    "order_value_inr": 97308.0
                }])

            # Handle clean identifier columns for filtering
            df["clean_order_id"] = df["order_id"].astype(str).str.strip().str.upper()
            
            # Format date strings safely
            df["formatted_date"] = pd.to_datetime(df["order_date"], errors="coerce").dt.strftime("%d Jul, %I:%M %p")
            
            # Display mappings
            df["items_display"] = df["quantity"].astype(str) + " x " + df["product_name"].fillna("Standard Item")
            df["route_display"] = df["origin_city"].fillna("Hub") + " → " + df["destination_city"].fillna("Destination")

            mode_icons = {
                "Truck": "🚚 Truck",
                "Van": "🚐 Van",
                "Bike": "🏍️ Bike",
                "Drone": "🛸 Drone",
                "Air Cargo": "✈️ Air Cargo",
                "Ship": "🚢 Ship"
            }
            df["mode_display"] = df["transport_mode"].map(mode_icons).fillna(df["transport_mode"])

            df["display_status"] = df["order_status"].str.title()
            if "is_delayed" in df.columns:
                df.loc[df["is_delayed"] == True, "display_status"] = "Delayed"

            return df

        orders_df = load_order_data_from_db()

        # Initialize session state toggle for the wizard
        if "show_new_shipment_wizard" not in st.session_state:
            st.session_state.show_new_shipment_wizard = False

        col_search, col_btn = st.columns([4, 1])
        with col_search:
            search_query = st.text_input(
                "Search",
                placeholder="Filter by order / customer / city...",
                label_visibility="collapsed"
            )
        with col_btn:
            if st.button("+ New Shipment", use_container_width=True):
                st.session_state.show_new_shipment_wizard = True
                st.toast("Opening new shipment wizard...")
                st.rerun()

        # Render the form wizard when active
        if st.session_state.show_new_shipment_wizard:
            with st.form("new_shipment_form", border=True):
                st.subheader("📦 New Shipment Wizard")
                st.caption("Enter the shipment details below to write directly to PostgreSQL.")

                city_list = ["Ahmedabad", "Bengaluru", "Bhubaneswar", "Chandigarh", "Chennai", "Coimbatore", "Delhi", "Guwahati", "Gurugram", "Hyderabad", "Jaipur", "Kochi", "Kolkata", "Lucknow", "Madurai", "Mumbai", "Mysuru", "Nagpur", "Pune", "Visakhapatnam"]

                col1, col2 = st.columns(2)
                with col1:
                    customer_name = st.text_input("Customer Name", value="", placeholder="Enter customer name...")
                    
                    origin_city = st.selectbox(
                        "Origin City", 
                        options=city_list, 
                        index=None, 
                        placeholder="Select origin city..."
                    )
                    
                    destination_city = st.selectbox(
                        "Destination City", 
                        options=city_list, 
                        index=None, 
                        placeholder="Select destination city..."
                    )
                    
                    transport_mode = st.selectbox(
                        "Transport Mode", 
                        options=["Truck", "Van", "Bike", "Drone", "Air Cargo", "Ship"],
                        index=None,
                        placeholder="Select transport mode..."
                    )

                with col2:
                    product_name = st.text_input("Product Name", value="", placeholder="Enter product name...")
                    quantity = st.number_input("Quantity", min_value=1, value=None, placeholder="Enter quantity...")
                    package_weight_kg = st.number_input("Package Weight (kg)", min_value=0.1, value=None, placeholder="Enter weight...")
                    distance_km = st.number_input("Distance (km)", min_value=0.1, value=None, placeholder="Enter distance...")
                    order_value_inr = st.number_input("Order Value (₹)", min_value=100.0, value=None, placeholder="Enter order value...")

                submitted = st.form_submit_button(
                    "💾 Save & Insert into Database", type="primary"
                )

                if submitted:
                    if not customer_name or not product_name or not origin_city or not destination_city or not transport_mode or not quantity or not package_weight_kg or not distance_km or not order_value_inr:
                        st.error("Please fill in all required fields before saving.")
                    else:
                        try:
                            from sqlalchemy import text
                            import random

                            new_order_id = f"ORD-{random.randint(100000, 999999)}"
                            dummy_cust_id = f"CUST-{random.randint(1000, 9999)}"
                            dummy_prod_id = f"PROD-{random.randint(1000, 9999)}"

                            with engine.begin() as conn:
                                # 1. Ensure a dummy customer record exists so foreign keys/lookups don't break
                                conn.execute(
                                    text("""
                                        INSERT INTO customers (customer_id, customer_name, city, signup_date)
                                        VALUES (:cid, :cname, :ccity, CURRENT_DATE)
                                        ON CONFLICT (customer_id) DO NOTHING;
                                    """),
                                    {"cid": dummy_cust_id, "cname": customer_name, "ccity": origin_city}
                                )

                                # 2. Ensure a dummy product record exists
                                conn.execute(
                                    text("""
                                        INSERT INTO products (product_id, product_name, weight_kg, price_amount)
                                        VALUES (:pid, :pname, :pwt, :pval)
                                        ON CONFLICT (product_id) DO NOTHING;
                                    """),
                                    {"pid": dummy_prod_id, "pname": product_name, "pwt": package_weight_kg, "pval": order_value_inr}
                                )

                                # 3. Insert into orders table matching your exact columns
                                insert_sql = text("""
                                    INSERT INTO orders (
                                        order_id, customer_id, product_id, origin_city, destination_city, 
                                        transport_mode, package_weight_kg, order_value_inr, quantity, 
                                        distance_km, order_status, order_date
                                    )
                                    VALUES (
                                        :order_id, :cust_id, :prod_id, :origin, :dest, 
                                        :mode, :weight, :val, :qty, 
                                        :dist, 'processing', CURRENT_DATE
                                    )
                                """)
                                conn.execute(
                                    insert_sql,
                                    {
                                        "order_id": new_order_id,
                                        "cust_id": dummy_cust_id,
                                        "prod_id": dummy_prod_id,
                                        "origin": origin_city,
                                        "dest": destination_city,
                                        "mode": transport_mode,
                                        "weight": package_weight_kg,
                                        "val": order_value_inr,
                                        "qty": quantity,
                                        "dist": distance_km
                                    }
                                )

                            st.success(f"New shipment `{new_order_id}` successfully added to PostgreSQL!")
                            st.session_state.show_new_shipment_wizard = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Database insertion failed: {e}")

                if st.form_submit_button("Cancel"):
                    st.session_state.show_new_shipment_wizard = False
                    st.rerun()
        
        tab_all, tab_transit, tab_delivered, tab_delayed, tab_processing, tab_recent = st.tabs(
            ["All", "In Transit", "Delivered", "Delayed", "Processing","Recent"]
        )

        def filter_and_display(df, status_filter=None):
            filtered = df.copy()

            if status_filter:
                filtered = filtered[filtered["display_status"].str.lower() == status_filter.lower()]

            if search_query:
                q = search_query.lower()
                mask = (
                    filtered["order_id"].str.lower().str.contains(q, na=False) |
                    filtered["customer_name"].str.lower().str.contains(q, na=False) |
                    filtered["destination_city"].str.lower().str.contains(q, na=False)
                )
                filtered = filtered[mask]

            if filtered.empty:
                st.info("No matching orders found.")
                return

            display_table = pd.DataFrame({
                "ORDER ID": filtered["order_id"],
                "DATE": filtered["formatted_date"],
                "CUSTOMER": filtered["customer_name"],
                "ITEMS": filtered["items_display"],
                "MODE": filtered["mode_display"],
                "ROUTE": filtered["route_display"],
                "DISTANCE": filtered["distance_km"].round(1).astype(str) + " km",
                "WEIGHT": filtered["package_weight_kg"].round(2).astype(str) + " kg",
                "STATUS": filtered["display_status"],
                "AMOUNT": filtered["order_value_inr"].apply(lambda x: f"₹{x:,.2f}" if pd.notnull(x) else "₹0.00")
            })

            st.dataframe(display_table, use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(display_table)} matching orders (pulled live from PostgreSQL)")

        with tab_all:
            filter_and_display(orders_df, None)
        with tab_transit:
            filter_and_display(orders_df, "In Transit")
        with tab_delivered:
            filter_and_display(orders_df, "Delivered")
        with tab_delayed:
            filter_and_display(orders_df, "Delayed")
        with tab_processing:
            filter_and_display(orders_df, "Processing")
        with tab_recent:
            st.subheader("📥 Newly Placed Orders (Pending Processing)")
            st.caption("Review and process incoming customer orders before moving them to active transit.")
            
            # Clear cache button to force-refresh if data is lagging
            if st.button("🔄 Refresh Orders Stream"):
                st.cache_data.clear()
                st.rerun()

            # Filter for orders that are processing or have empty/null status
            recent_orders = orders_df[
                orders_df["order_status"].fillna("processing").str.lower().isin(["processing", "pending", "in_transit", ""])
            ]
            
            # Alternatively, sort by date/order_item_id to show the absolute newest first
            if "order_item_id" in orders_df.columns:
                orders_df = orders_df.sort_values(by="order_item_id", ascending=False)
                recent_orders = orders_df # Allow selecting any recent order from the top

            if recent_orders.empty:
                st.info("No recent or pending orders to process.")
            else:
                selected_order = st.selectbox("Select Order ID to Process", recent_orders["order_id"].unique())

                # Get specific order row
                order_row = orders_df[orders_df["order_id"] == selected_order].iloc[0]
                product_id = order_row.get("product_id")
                customer_id = order_row.get("customer_id")

                # Fetch customer and product names safely from database for display
                try:
                    with engine.connect() as conn:
                        cust_res = conn.execute(text("SELECT customer_name FROM customers WHERE customer_id = :cid"), {"cid": customer_id}).fetchone()
                        cust_name = cust_res[0] if cust_res and cust_res[0] else customer_id

                        prod_res = conn.execute(text("SELECT product_name, weight_kg, is_fragile, is_hazmat, requires_cold_chain FROM products WHERE product_id = :pid"), {"pid": product_id}).fetchone()
                        if prod_res:
                            item_name = prod_res[0]
                            default_weight = float(prod_res[1]) if prod_res[1] is not None else 1.5
                            def_fragile = bool(prod_res[2])
                            def_hazmat = bool(prod_res[3])
                            def_cold_chain = bool(prod_res[4])
                        else:
                            item_name = product_id
                            default_weight, def_fragile, def_hazmat, def_cold_chain = 1.5, False, False, False
                except Exception:
                    cust_name, item_name = customer_id, product_id
                    default_weight, def_fragile, def_hazmat, def_cold_chain = 1.5, False, False, False

                with st.form(key="process_recent_order_form"):
                    st.markdown(f"**Processing Order:** `{order_row['order_id']}` | **Customer:** {cust_name} | **Item:** {item_name}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        city_list = ["Ahmedabad", "Bengaluru", "Bhubaneswar", "Chandigarh", "Chennai", "Coimbatore", "Delhi", "Guwahati", "Gurugram", "Hyderabad", "Jaipur", "Kochi", "Kolkata", "Lucknow", "Madurai", "Mumbai", "Mysuru", "Nagpur", "Pune", "Visakhapatnam"]
                        
                        # Set default index safely based on what was inserted
                        orig_val = order_row.get("origin_city")
                        orig_idx = city_list.index(orig_val) if orig_val in city_list else 0
                        origin_city = st.selectbox("Origin City", city_list, index=orig_idx)

                        dest_val = order_row.get("destination_city")
                        dest_idx = city_list.index(dest_val) if dest_val in city_list else 1
                        destination_city = st.selectbox("Destination City", city_list, index=dest_idx)

                        distance = st.number_input("Delivery Distance (km)", min_value=0.1, value=float(order_row.get("distance_km", 25.0) or 25.0))

                    with col2:
                        weight = st.number_input("Package Weight (kg)", min_value=0.1, value=float(order_row.get("package_weight_kg", default_weight) or default_weight))
                        quantity = st.number_input("Package Quantity", min_value=1, value=int(order_row.get("quantity", 1) or 1))
                        priority = st.selectbox("Delivery Priority", ["STANDARD", "EXPRESS", "ECONOMY", "SAME_DAY"])

                    with col3:
                        weather = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Rain", "Sunny", "Fog", "Heavy Rain", "Thunderstorm"])
                        next_status = st.selectbox("Update Status To", ["Processing", "In Transit"])
                        
                    st.markdown("**Special Handling Requirements**")
                    h_col1, h_col2, h_col3 = st.columns(3)
                    with h_col1:
                        is_fragile = st.checkbox("Fragile", value=def_fragile)
                    with h_col2:
                        is_hazmat = st.checkbox("Hazmat", value=def_hazmat)
                    with h_col3:
                        cold_chain_required = st.checkbox("Cold Chain Required", value=def_cold_chain)

                    submit_btn = st.form_submit_button("🚀 Run ML Prediction & Update Order Status", type="primary", use_container_width=True)
                    
                    if submit_btn:
                        try:
                            input_df = pd.DataFrame([{
                                "origin_city": origin_city,
                                "destination_city": destination_city,
                                "distance_km": distance,
                                "package_weight_kg": weight,
                                "delivery_priority": priority,
                                "weather_condition": weather,
                                "is_fragile": int(is_fragile),
                                "is_hazmat": int(is_hazmat),
                                "cold_chain_required": int(cold_chain_required),
                            }])
                            X_encoded = pd.get_dummies(input_df).reindex(columns=model_cols, fill_value=0)
                            pred_encoded = model.predict(X_encoded)[0]
                            predicted_mode = encoder.inverse_transform([pred_encoded])[0]
                        except Exception as ex:
                            predicted_mode = "Truck"
                            st.warning(f"ML Mode prediction fallback triggered: {ex}")

                        mock_order_row = {
                            "distance_km": distance,
                            "package_weight_kg": weight,
                            "quantity": quantity,
                            "transport_mode": predicted_mode,
                            "delivery_priority": priority,
                            "is_fragile": int(is_fragile),
                            "is_hazmat": int(is_hazmat),
                            "cold_chain_required": int(cold_chain_required),
                            "capacity_kg": 500.0,
                            "max_range_km": 300.0,
                            "avg_speed_kmph": 45.0,
                            "odometer_km": 1000.0,
                            "battery_capacity_wh": 5000.0,
                            "weather_condition_at_dest": weather,
                            "order_date": pd.Timestamp.now(),
                        }
                        
                        try:
                            pred_eta_hours = compute_live_model_eta(mock_order_row)
                        except Exception:
                            pred_eta_hours = distance / 45.0

                        update_query = text("""
                            UPDATE orders 
                            SET order_status = :status, 
                                transport_mode = :mode, 
                                package_weight_kg = :weight, 
                                distance_km = :dist,
                                promised_eta_hours = :eta
                            WHERE order_id = :order_id
                        """)
                        try:
                            with engine.begin() as conn:
                                conn.execute(
                                    update_query,
                                    {
                                        "status": next_status,
                                        "mode": predicted_mode,
                                        "weight": weight,
                                        "dist": distance,
                                        "eta": pred_eta_hours,
                                    "order_id": selected_order
                                    }
                                )
                            st.cache_data.clear() # Clear cache so changes reflect instantly
                            st.success(f"Order `{selected_order}` successfully updated! Assigned Mode: **{predicted_mode}** | Computed ETA: **{round(pred_eta_hours * 60, 1)} mins**")
                            st.rerun()
                        except Exception as db_ex:
                            st.error(f"Database update failed: {db_ex}")
        #with st.expander("How this works — pipeline & AWS flow: Order list & filtering"):
            #st.write(
                #"Orders are ingested via partition-aware S3 streams, transformed through AWS Glue jobs, "
                #"persisted in PostgreSQL (`smartlogix`), and queried with low-latency indexing semantics for dashboard rendering.")

    elif selected_page == "📍 Delivery Tracking":
        st.title("📍 Live Delivery Tracking & ETA Estimation")
        st.caption("Inspect live order telemetry, customer details, product catalogs, and chronological delivery timelines.")

        # Fetch live order data from PostgreSQL
        df_db = load_live_orders_from_db()

        if df_db.empty:
            st.warning("No orders found in the `smartlogix` database.")
        else:
            # Dynamically determine the correct ID column from the orders dataset
            id_col = "order_id" if "order_id" in df_db.columns else "order_item_id"
            
            # Extract unique order IDs directly from the dataset
            order_ids = df_db[id_col].dropna().astype(str).unique().tolist()

            col_sel1, col_sel2 = st.columns([2, 3])
            with col_sel1:
                tracked_order_id = st.selectbox(
                    "Select Order to Track", 
                    options=order_ids
                )
            with col_sel2:
                st.markdown(f"**Active Stream:** `slx-live-positions` (Dataset records: {len(df_db)})")

            st.divider()

            # Filter dataframe row for the selected order ID from the dataset
            order_row = df_db[df_db[id_col].astype(str) == tracked_order_id].iloc[0]

            # ⚡ DYNAMICALLY COMPUTE ETA USING YOUR TRAINED ML MODELS
            model_eta_val = compute_live_model_eta(order_row)

            col_left, col_right = st.columns([1.1, 1.4], gap="medium")

            with col_left:
                with st.container(border=True):
                    # 1. Dynamically sync status with distance remaining
                    dist_val = float(order_row.get("distance_km", 0.0))
                    current_status = "Delivered" if dist_val <= 0.1 else "In Transit"
                    sub_c1, sub_c2 = st.columns([2, 1])
                    with sub_c1:
                        st.subheader("Order Details")
                    with sub_c2:
                        badge_color = "#16a34a" if current_status == "Delivered" else "#2563eb"
                        status_val = order_row.get('display_status', order_row.get('order_status', 'In Transit'))
                        st.markdown(f"{status_val}", unsafe_allow_html=True)

                    st.markdown("---")
                    
                    details_data = {
                        "Order ID": order_row.get(id_col),
                        "Delivery Mode": f"{order_row.get('transport_mode', 'Ground')} ({order_row.get('assigned_vehicle_id', 'VEH-0108')})",
                        "Estimated Model ETA": f"{model_eta_val:.2f} hours",  # ⚡ Updated to use model prediction
                        "Distance Remaining": f"{order_row.get('distance_km', 8.6)} km",
                        "Vehicle ID": order_row.get("assigned_vehicle_id", "VEH-0108"),
                        "Customer": order_row.get("customer_id", "Valued Customer"),
                        "Destination City": order_row.get("destination_city", "Destination Hub"),
                        "Package": f"{order_row.get('package_weight_kg', 0.83)} kg · {order_row.get('product_name', 'Standard Cargo')}",
                        "Order Value": f"₹{order_row.get('order_value_inr', 0):,.2f}"
                    }

                    for k, v in details_data.items():
                        row_c1, row_c2 = st.columns([1.2, 1.8])
                        row_c1.markdown(f"{k}", unsafe_allow_html=True)
                        row_c2.markdown(f"{v}", unsafe_allow_html=True)

                    st.markdown("---")
                    
                    if "live_updates_active" not in st.session_state:
                        st.session_state.live_updates_active = True

                    if st.session_state.live_updates_active:
                        if st.button("⏸️ Pause Live Updates", use_container_width=True, type="secondary"):
                            st.session_state.live_updates_active = False
                            st.rerun()
                        st.caption("Streaming — one simulated poll every 3 s (IoT Core → Kinesis → PostgreSQL)")
                    else:
                        if st.button("▶️ Resume Live Updates", use_container_width=True, type="primary"):
                            st.session_state.live_updates_active = True
                            st.rerun()
                        st.caption("Updates paused by operator.")

            with col_right:
                with st.container(border=True):
                    st.markdown(f"##### 🗺️ Telemetry Map — {order_row.get('assigned_vehicle_id', 'VEH-0108')}")
                    st.caption(f"Battery: 44% · Speed: {order_row.get('avg_speed_kmph', 49)} km/h · {order_row.get('distance_km', 8.6)} km away")

                    # Quick lookup for major origin hubs/cities
                    origin_coords_map = {
                        "chennai": [13.0827, 80.2707], "mumbai": [19.0760, 72.8777],
                        "delhi": [28.6139, 77.2090], "bengaluru": [12.9716, 77.5946],
                        "kolkata": [22.5726, 88.3639], "hyderabad": [17.3850, 78.4867],
                        "pune": [18.5204, 73.8567], "chandigarh": [30.7333, 76.7794],
                        "jaipur": [26.9124, 75.7873], "guwahati": [26.1445, 91.7362],
                        "lucknow": [26.8467, 80.9462], "bhubaneswar": [20.2961, 85.8245],
                        "nagpur": [21.1458, 79.0882], "coimbatore": [11.0168, 76.9558],
                        "madurai": [9.9252, 78.1198], "visakhapatnam": [17.6868, 83.2185],
                        "kochi": [9.9312, 76.2673]
                    }

                    orig_city_key = str(order_row.get("origin_city", "Chennai")).strip().lower()
                    origin_coords = origin_coords_map.get(orig_city_key, [13.0827, 80.2707])

                    dest_lat = float(order_row.get("destination_lat", 28.4595))
                    dest_lon = float(order_row.get("destination_lon", 77.0266))
                    dest_coords = [dest_lat, dest_lon]

                    m = folium.Map(location=dest_coords, zoom_start=5, tiles="OpenStreetMap")

                    origin_hub = order_row.get("origin_hub", "HUB")
                    origin_city = order_row.get("origin_city", "Chennai")
                    folium.Marker(
                        origin_coords,
                        popup=f"Origin: {origin_hub} ({origin_city})",
                        icon=folium.Icon(color="green", icon="play")
                    ).add_to(m)

                    dest_city = order_row.get("destination_city", "Destination")
                    dest_state = order_row.get("destination_state", "")
                    folium.Marker(
                        dest_coords,
                        popup=f"Destination: {dest_city}, {dest_state} (Pin: {order_row.get('destination_pinkcode', '')})",
                        icon=folium.Icon(color="red", icon="flag")
                    ).add_to(m)

                    folium.PolyLine(
                        [origin_coords, dest_coords],
                        color="#2563eb", weight=4, opacity=0.8, dash_array="5"
                    ).add_to(m)

                    st_folium(m, height=350, use_container_width=True)

            # ==========================================
            # ETA RE-ESTIMATION & ALERTS SECTION
            # ==========================================
            st.markdown("### ⚡ ETA Re-estimation & Intelligence")

            col_eta1, col_eta2 = st.columns([2.2, 1], gap="medium")

            with col_eta1:
                with st.container(border=True):
                    st.markdown("##### ETA Re-estimation (live)")
                    st.divider()

                    # ⚡ Use dynamic ML model ETA here
                    db_eta = model_eta_val
                    promised_eta = db_eta + 0.2
                    
                    def format_hrs_mins(hours_val):
                        total_mins = int(round(hours_val * 60))
                        h = total_mins // 60
                        m = total_mins % 60
                        if h > 0 and m > 0:
                            return f"{h}h {m}m"
                        elif h > 0:
                            return f"{h}h"
                        else:
                            return f"{m}m"

                    delay_hrs = float(order_row.get("delay_hours", 0.0))
                    drift_mins = int(delay_hrs * 60)
                    if drift_mins == 0:
                        drift_mins = -5 

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Promised ETA", format_hrs_mins(promised_eta))
                    c2.metric("Model ETA (now)", format_hrs_mins(db_eta))
                    c3.metric("Drift", f"{drift_mins} min", delta=f"{drift_mins} min", delta_color="inverse")
                    c4.metric("Weather", "Clear")

            with col_eta2:
                with st.container(border=True):
                    alert_header_c1, alert_header_c2 = st.columns([2, 1])
                    with alert_header_c1:
                        st.markdown("##### Delivery Alerts")
                    with alert_header_c2:
                        st.markdown("View All", unsafe_allow_html=True)

                    v_id = order_row.get('assigned_vehicle_id', 'VEH-0108')
                    dest = order_row.get('destination_city', 'Destination Hub')
                    dist = float(order_row.get('distance_km', 5.0))
                    delayed = bool(order_row.get('is_delayed', False))
                    delay_h = float(order_row.get('delay_hours', 0.0))

                    dynamic_alerts = []
                    if delayed:
                        dynamic_alerts.append(("⚠️", f"SLA Risk: Delayed by {delay_h:.1f}h towards {dest}", "Active Delay Flag"))
                    else:
                        dynamic_alerts.append(("✅", f"Shipment on schedule for {dest}", "Normal Status"))

                    if dist < 15:
                        dynamic_alerts.append(("🛵", f"Vehicle {v_id} is approaching {dest} ({dist:.1f} km left)", "Proximity Alert"))
                    else:
                        dynamic_alerts.append(("📍", f"Vehicle {v_id} en route to {dest} ({dist:.1f} km away)", "Telemetry Update"))

                    dynamic_alerts.append(("📦", f"Handled via {order_row.get('transport_mode', 'Ground')} transport", "Hub Scan"))
                    dynamic_alerts.append(("📋", f"Order status logged as: {order_row.get('order_status', 'In Transit')}", "Database Sync"))

                    for icon, alert_text, time_ago in dynamic_alerts:
                        st.markdown(f"{icon} **{alert_text}**")
                        st.markdown(f"{time_ago}", unsafe_allow_html=True)
                        st.divider()

            with col_right:
                with st.container(border=True):
                    st.markdown("##### Delivery Timeline")
                    
                    order_date = order_row.get('formatted_date', order_row.get('created_at', 'Recent'))
                    status_val = order_row.get('order_status', 'In Transit')
                    mode_val = order_row.get('transport_mode', 'Ground')

                    timeline_events = [
                        ("🟢", "Order Placed", f"{order_date} · Logged in database"),
                        ("🟢", "Payment Confirmed", "Verified via secure payment gateway"),
                        ("🟢", "Picked From Hub", f"Scanned and loaded onto {mode_val} ({v_id})"),
                        ("🟢", "In Transit", f"Active GPS telemetry · Status: {status_val}"),
                        ("🔵", "Destination Approach", f"Final routing toward {dest} ({dist:.1f} km remaining)")
                    ]
                    
                    for icon, title, desc in timeline_events:
                        st.markdown(f"{icon} **{title}**")
                        st.markdown(f"{desc}", unsafe_allow_html=True)
                        st.markdown("")

    
    elif selected_page == "🧩 Choose Delivery Mode":
        st.markdown("### 📦 Live Shipment Dispatch & Mode Simulator (ML Driven)")
        st.caption(
            "Enter shipment parameters to calculate transport mode and ETA via loaded"
            " .pkl models."
        )

        # Complete list of unique cities
        city_list = [
            "Ahmedabad",
            "Bengaluru",
            "Bhubaneswar",
            "Chandigarh",
            "Chennai",
            "Coimbatore",
            "Delhi",
            "Guwahati",
            "Gurugram",
            "Hyderabad",
            "Jaipur",
            "Kochi",
            "Kolkata",
            "Lucknow",
            "Madurai",
            "Mumbai",
            "Mysuru",
            "Nagpur",
            "Pune",
            "Visakhapatnam",
        ]

        with st.form(key="main_dispatch_form"):
            f_col1, f_col2, f_col3 = st.columns(3)

            with f_col1:
                origin_city = st.selectbox("Origin City", city_list, index=3)
                destination_city = st.selectbox("Destination City", city_list, index=0)
                distance = st.number_input(
                    "Delivery Distance (km)", min_value=0.1, value=25.0
                )

            with f_col2:
                weight = st.number_input("Shipment Weight (kg)", min_value=0.1, value=5.0)
                quantity = st.number_input("Package Quantity", min_value=1, value=2)
                
            with f_col3:
                # Corrected delivery priority options matching dataset value counts[cite: 6]
                priority = st.selectbox(
                    "Delivery Priority", ["STANDARD", "EXPRESS", "ECONOMY", "SAME_DAY"]
                )
                # Corrected weather options matching destination weather value counts[cite: 7]
                weather = st.selectbox(
                    "Weather Condition",
                    [
                        "Clear",
                        "Cloudy",
                        "Rain",
                        "Sunny",
                        "Fog",
                        "Heavy Rain",
                        "Thunderstorm",
                    ],
                )

            st.markdown("**Special Handling Requirements**")
            is_fragile = st.checkbox("Fragile", value=False)
            is_hazmat = st.checkbox("Hazmat", value=False)
            cold_chain_required = st.checkbox("Cold Chain Required", value=False)

            submit_dispatch = st.form_submit_button(
                "🚀 Run ML Logistics & ETA Prediction",
                use_container_width=True,
            )

        st.divider()

        if submit_dispatch:
            # 1. Build DataFrame matching your training features[cite: 5] using actual distance input
            input_df = pd.DataFrame([{
                "origin_city": origin_city,
                "destination_city": destination_city,
                "distance_km": distance,
                "package_weight_kg": weight,
                "delivery_priority": priority,
                "weather_condition": weather,
                "is_fragile": int(is_fragile),
                "is_hazmat": int(is_hazmat),
                "cold_chain_required": int(cold_chain_required),
            }])

            # 2. Predict Transport Mode
            try:
                X_encoded = pd.get_dummies(input_df).reindex(columns=model_cols, fill_value=0)
                pred_encoded = model.predict(X_encoded)[0]
                rec_mode = encoder.inverse_transform([pred_encoded])[0]
            except Exception as ex:
                rec_mode = "TRUCK"

            # 3. Compute ETA using real distance
            mock_order_row = {
                "distance_km": distance,
                "package_weight_kg": weight,
                "quantity": quantity,
                "transport_mode": rec_mode,
                "delivery_priority": priority,
                "is_fragile": int(is_fragile),
                "is_hazmat": int(is_hazmat),
                "cold_chain_required": int(cold_chain_required),
                "capacity_kg": 500.0,
                "max_range_km": 300.0,
                "avg_speed_kmph": 45.0,
                "odometer_km": 1000.0,
                "battery_capacity_wh": 5000.0,
                "weather_condition_at_dest": weather,
                "order_date": pd.Timestamp.now(),
            }

            pred_eta_hours = compute_live_model_eta(mock_order_row)
            pred_eta_mins = round(pred_eta_hours * 60, 1)

            # 4. Display Results
            st.subheader("🤖 Real-Time ML Model Outputs")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.success(f"**Recommended Transport Mode:** `{rec_mode}`")
            with res_col2:
                st.info(
                f"**Predicted Delivery ETA:** `{pred_eta_mins} Minutes` ("
                f"{round(pred_eta_hours, 2)} Hours)"
            )

            ml_results_df = pd.DataFrame([
                {
                    "Transport Mode": rec_mode,
                    "Predicted ETA (Mins)": pred_eta_mins,
                    "Origin": origin_city,
                    "Destination": destination_city,
                    "Distance (km)": distance,
                    "Priority": priority,
                    "Weather": weather,
                    "Weight (kg)": weight,
                    "Quantity": quantity,
                }
            ])
            st.dataframe(ml_results_df, use_container_width=True)

    elif selected_page == "❇️ Route Optimization":
        st.title("🗺️ OR-Tools Fleet Route Optimizer & Control Tower")
        st.caption(
            "Connected to PostgreSQL database (`gps_routes`, `fleet_vehicles`,"
            " `traffic_data`, `weather_data`) for live sequencing."
        )

        routes_df = fetch_available_routes()

        # 3-Column Layout: [Settings | Map | Route Stop List Card] matching the reference UI
        col_settings, col_map, col_panel = st.columns([1, 1.4, 1.1])

        # 1. Left Column: Optimization Settings
        with col_settings:
            st.subheader("Optimization Settings")

            input_mode = st.radio(
                "Route Selection Mode", ["Select Existing Route", "Enter Custom Route ID"]
            )

            selected_route_id = "RTE-00001"  
            num_deliveries = 5

            if input_mode == "Select Existing Route":
                if not routes_df.empty:
                    selected_route_id = st.selectbox(
                    "Target Route ID", routes_df["route_id"].tolist()
                )
                else:
                    selected_route_id = st.text_input(
                    "Target Route ID", value="RTE-00001"
                )
            else:
                selected_route_id = st.text_input(
                "Custom Route ID", value="RTE-CUSTOM-01"
            )
            num_deliveries = st.number_input(
                "Max Number of Deliveries (Stops)",
                min_value=2,
                max_value=30,
                value=5,
                step=1,
            )

            optimization_mode = st.selectbox(
                "Mode", ["Truck", "Van", "Bike", "Air-cargo", "Ship", "Drone Delivery"]
            )
            max_range = st.selectbox("Max Range", ["35 km", "100 km", "400 km"])
            payload_capacity = st.selectbox(
                "Payload Capacity", ["5 kg", "100 kg", "800 kg"]
            )

            st.markdown("**Optimize for:**")
            opt_objective = st.radio(
                "Objective",
                ["Shortest Distance", "Min Delivery Time", "Min Energy Usage"],
                label_visibility="collapsed",
            )

            run_opt = st.button(
                "Optimize Route", type="primary", use_container_width=True
            )

            target_rte = selected_route_id
            target_deliveries = num_deliveries
            result = optimize_route(target_rte, num_deliveries=target_deliveries)

        locations = result.get("locations", [])
        optimized_order = result.get("optimized_order", [])

        # 2. Middle Column: Geospatial Route Map
        with col_map:
            st.subheader("Geospatial Route Map")
            if locations and optimized_order:
        
                ordered_coords = [
                    [locations[i][1], locations[i][0]] for i in optimized_order
                ]
                path_data = [{"path": ordered_coords, "name": "Optimized Route Path"}]

                point_data = []
                for seq_idx, loc_idx in enumerate(optimized_order):
                    lat, lon = locations[loc_idx]
                    if seq_idx == 0:
                        label = "Start Point (Warehouse)"
                    elif seq_idx == len(optimized_order) - 1:
                        label = "Return to Base"
                    else:
                        label = f"Stop {seq_idx}"

                    point_data.append(
                        {"coordinates": [lon, lat], "name": label, "order": seq_idx}
                    )

                avg_lat = sum([loc[0] for loc in locations]) / len(locations)
                avg_lon = sum([loc[1] for loc in locations]) / len(locations)

                view_state = pdk.ViewState(
                    latitude=avg_lat,
                    longitude=avg_lon,
                    zoom=12,
                    pitch=20,
                    bearing=0,
                    controller=True,
                )

                path_layer = pdk.Layer(
                    "PathLayer",
                    path_data,
                    get_path="path",
                    get_color=[31, 119, 180],
                    width_scale=20,
                    width_min_pixels=5,
                )

                scatter_layer = pdk.Layer(
                    "ScatterplotLayer",
                    point_data,
                    get_position="coordinates",
                    get_color=[255, 75, 75],
                    get_radius=180,
                    pickable=True,
                )

                r = pdk.Deck(
                    layers=[path_layer, scatter_layer],
                    initial_view_state=view_state,
                    tooltip={"text": "{name}"},
                    map_style="mapbox://styles/mapbox/light-v9",
                )
                st.pydeck_chart(r, use_container_width=True, height=520)
            else:
                st.warning("No coordinate waypoints available.")

            # 3. Right Column: Optimized Route Card & Stop List (Matching reference layout)
            with col_panel:
                st.subheader("Optimized Route")

                if len(optimized_order) > 1:
                    card_data = []
                    cum_dist = 0.0

                    for i in range(len(optimized_order) - 1):
                        frm_idx = optimized_order[i]
                        to_idx = optimized_order[i + 1]

                        if frm_idx == to_idx:
                            continue
                        if frm_idx >= len(locations) or to_idx >= len(locations):
                            continue

                        p1 = locations[frm_idx]
                        p2 = locations[to_idx]
                        leg_dist = math.sqrt(
                            ((p1[0] - p2[0]) * 111) ** 2 + ((p1[1] - p2[1]) * 111) ** 2
                        )
                        cum_dist += leg_dist

                        if i == 0:
                            from_label = "Start Point"
                        else:
                            from_label = f"Stop {i}"

                        if i == len(optimized_order) - 2:
                            to_label = "Return to Base"
                        else:
                            to_label = f"Stop {i+1}"

                        card_data.append({
                            "Seq": i + 1,
                            "Stop Name": to_label,
                            "Distance": f"{round(cum_dist, 1)} km",
                        })

                    # Display clean UI list card for the route stops
                    import pandas as pd

                    st.dataframe(
                        pd.DataFrame(card_data), use_container_width=True, hide_index=True
                    )

                    st.markdown("---")
                    actual_dist = round(cum_dist, 1)
                    st.markdown(f"**Total Distance:** {actual_dist} km")
                    st.markdown(
                        f"**Estimated Time:** {round(actual_dist * 1.6)} mins"
                    )  # Estimated dynamic calculation
                    st.markdown(f"**Battery Usage:** {min(int(actual_dist * 2.5), 100)}%")
                else:
                    st.info("Run optimization to generate route sequence.")


    elif selected_page == "🚚 Fleet Management":
        st.title("🚗 Fleet Management & Asset Control")
        st.caption("Connected to PostgreSQL database (`fleet_vehicles`, `maintenance_history`) with ML Predictive Maintenance.")

        # --- 1. Load Trained Machine Learning Model (Optional fallback reference) ---
        model_path = "maintenance_prediction/fleet_maintenance_model.pkl"
        maintenance_model = None
        if os.path.exists(model_path):
            try:
                maintenance_model = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Could not load local ML model artifact: {e}")

        # --- 2. Fetch Live Data Directly from PostgreSQL ---
        @st.cache_data(ttl=30)
        def load_fleet_data():
            conn = get_db_connection()
            query = """
                SELECT 
                    vehicle_id, 
                    vehicle_type as type, 
                    model_name as model, 
                    hub_code as hub, 
                    capacity_kg, 
                    max_range_km, 
                    avg_speed_kmph, 
                    odometer_km, 
                    fleet_status as status, 
                    last_service_date as last_service,
                    predicted_maintenance
                FROM fleet_vehicles
            """
            df = pd.read_sql(query, conn)
            conn.close()
            return df

        try:
            fleet_df = load_fleet_data()
        except Exception as e:
            st.error(f"❌ Database Connection Error: Unable to fetch records from `fleet_vehicles`. Details: {e}")
            st.stop()

        if fleet_df.empty:
            st.warning("⚠️ The `fleet_vehicles` table is currently empty.")
            st.stop()

        # --- 3. Compute Dynamic KPI Metrics from Database ---
        total_assets = len(fleet_df)
        in_service_df = fleet_df[fleet_df['status'].str.lower() == 'in service']
        maint_df = fleet_df[fleet_df['status'].str.lower() == 'maintenance']
        ml_risk_df = fleet_df[fleet_df['predicted_maintenance'] == 1]

        in_service_count = len(in_service_df)
        maint_count = len(maint_df)
        ml_risk_count = len(ml_risk_df)
        
        utilization_pct = round((in_service_count / total_assets) * 100, 1) if total_assets > 0 else 0
        ml_risk_pct = round((ml_risk_count / total_assets) * 100, 1) if total_assets > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Assets", value=f"{total_assets:,}", delta="across all hubs")
        with col2:
            st.metric(label="In Service", value=f"{in_service_count:,}", delta=f"{utilization_pct}% utilization", delta_color="normal")
        with col3:
            st.metric(label="In Maintenance", value=f"{maint_count:,}", delta="downtime tracked", delta_color="inverse")
        with col4:
            st.metric(label="ML Flagged At-Risk", value=f"{ml_risk_count:,}", delta=f"{ml_risk_pct}% of fleet", delta_color="inverse")

        st.markdown("---")

        # --- 4. Middle Section: Dynamic Availability & Status Split ---
        col_left, col_right = st.columns([1.5, 1])

        with col_left:
            st.subheader("Availability by Vehicle Type")
            type_counts = fleet_df['type'].value_counts()
            max_type_count = type_counts.max() if not type_counts.empty else 100
            
            for v_type, count in type_counts.items():
                st.text(f"{v_type}")
                st.progress(min(count / max_type_count, 1.0))
                st.caption(f"active capacity pool: {count} units")

        with col_right:
            st.subheader("Fleet Status Split")
            status_counts = fleet_df['status'].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            
            import altair as alt
            chart = alt.Chart(status_counts).mark_arc(innerRadius=60, outerRadius=90).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Status", type="nominal",
                                scale=alt.Scale(domain=["In Service", "Idle", "Maintenance", "Out of Service"],
                                                range=["#2ca02c", "#aec7e8", "orange", "#d62728"])),
                tooltip=["Status", "Count"]
            ).properties(height=220)
            st.altair_chart(chart, use_container_width=True)

        st.markdown("---")

        # --- 5. Bottom Section: Fleet Register & Database Filters ---
        st.subheader("Fleet Register")
        st.caption("Live feed from PostgreSQL database with pre-calculated ML Predictive Maintenance flags.")

        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        with f_col1:
            search_query = st.text_input("Search", placeholder="Search vehicle ID / hub...", label_visibility="collapsed")
        with f_col2:
            available_types = ["All Types"] + list(fleet_df['type'].dropna().unique())
            type_filter = st.selectbox("Filter Type", available_types, label_visibility="collapsed")
        with f_col3:
            available_statuses = ["All Statuses"] + list(fleet_df['status'].dropna().unique())
            status_filter = st.selectbox("Filter Status", available_statuses, label_visibility="collapsed")

        # --- Filter dataframe ---
        filtered_df = fleet_df.copy()
        if search_query:
            query_lower = search_query.lower()
            filtered_df = filtered_df[filtered_df.apply(lambda row: row.astype(str).str.lower().str.contains(query_lower).any(), axis=1)]
        if type_filter != "All Types":
            filtered_df = filtered_df[filtered_df["type"].str.lower() == type_filter.lower()]
        if status_filter != "All Statuses":
            filtered_df = filtered_df[filtered_df["status"].str.lower() == status_filter.lower()]

        # --- Pagination Setup ---
        total_records = len(filtered_df)
        page_size = 10
        total_pages = max(1, (total_records + page_size - 1) // page_size)

        p_col1, p_col2, p_col3 = st.columns([2, 2, 3])
        with p_col1:
            st.write(f"**Total Matching Assets:** {total_records}")
        with p_col2:
            current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, label_visibility="collapsed")
        with p_col3:
            st.markdown(f" Page {current_page} of {total_pages}", unsafe_allow_html=True)
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = filtered_df.iloc[start_idx:end_idx]

        st.markdown("---")

        # Table Header Layout
        h1, h2, h3, h4, h5, h6, h7, h8 = st.columns([1.1, 1, 1.2, 1, 1, 1, 1.4, 1])
        h1.markdown("**Vehicle ID**")
        h2.markdown("**Type**")
        h3.markdown("**Model**")
        h4.markdown("**Hub**")
        h5.markdown("**Odometer**")
        h6.markdown("**Status**")
        h7.markdown("**ML Risk Status**")
        h8.markdown("**Action**")
        st.markdown(" ", unsafe_allow_html=True)

        # Render Paginated Rows
        for idx, row in page_df.iterrows():
            veh_id = row.get("vehicle_id")
            c_veh, c_type, c_model, c_hub, c_odo, c_stat, c_ml, c_act = st.columns([1.1, 1, 1.2, 1, 1, 1, 1.4, 1])
            
            c_veh.markdown(f"**{veh_id}**")
            c_type.markdown(f"{row.get('type')}")
            c_model.markdown(f"{row.get('model')}")
            c_hub.markdown(f"{row.get('hub')}")
            odo_val = float(row.get("odometer_km", 0))
            c_odo.markdown(f"{odo_val:,.0f} km")
            
            status_txt = row.get("status", "In Service")
            if status_txt.lower() == "in service":
                c_stat.markdown(":green[In Service]")
            elif status_txt.lower() == "maintenance":
                c_stat.markdown(":orange[Maintenance]")
            else:
                c_stat.markdown(f"{status_txt}")
            
            pred_val = int(row.get("predicted_maintenance", 0))
            
            with c_ml:
                if pred_val == 1:
                    st.markdown("⚠️ :red[**Maintenance Req**]")
                else:
                    st.markdown("✅ :green[**Healthy**]")
                    
            with c_act:
                if st.button("Inspect", key=f"insp_{veh_id}_{idx}"):
                    st.session_state["active_inspect"] = veh_id if st.session_state.get("active_inspect") != veh_id else None

            st.markdown(" ", unsafe_allow_html=True)

            # --- Inline Inspector Drawer ---
            if st.session_state.get("active_inspect") == veh_id:
                with st.container():
                    st.markdown(f"""
        🛠️ AI Diagnostic & Maintenance Inspector — Asset: {veh_id}
                    """, unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            label="Model Maintenance Status", 
                            value="Required" if pred_val == 1 else "Healthy", 
                            delta="PostgreSQL Sync", 
                            delta_color="inverse" if pred_val == 1 else "normal"
                        )
                        st.write(f"**Current Odometer:** {odo_val:,.0f} km")
                    with col_b:
                        st.write(f"**Operational Status:** {status_txt}")
                        st.write("**AI Recommendation:** " + ("Schedule immediate service workshop check." if pred_val == 1 else "Asset operating within safe tolerances."))
                    
                    # Fetch historical work orders from DB for this specific vehicle
                    try:
                        conn = get_db_connection()
                        history_df = pd.read_sql("SELECT work_order_id, service_date, service_type, parts_replaced, cost_inr, downtime_hours FROM maintenance_history WHERE vehicle_id = %s", conn, params=(veh_id,))
                        conn.close()
                        
                        if not history_df.empty:
                            history_df = history_df.rename(columns={
                                "work_order_id": "Work Order ID",
                                "service_date": "Service Date",
                                "service_type": "Service Type",
                                "parts_replaced": "Parts Replaced",
                                "cost_inr": "Cost (INR)",
                                "downtime_hours": "Downtime (hrs)"
                            })
                            st.markdown("**Historical Maintenance Logs:**")
                            st.dataframe(history_df, use_container_width=True)
                        else:
                            st.info(f"No prior work orders logged in database for {veh_id}.")
                    except Exception as db_err:
                        st.warning(f"Could not retrieve historical work orders: {db_err}")
                        
                    if st.button("Close Inspector", key=f"close_{veh_id}_{idx}"):
                        st.session_state["active_inspect"] = None
                            
                    st.markdown("---")

    elif selected_page == "🛸 Drone Management":
        
        st.title("🔍 AI Drone Health Inspection")
        st.write("Upload a drone image to analyze its condition using your trained YOLOv11 classification model.")

        # Path to your trained model weights
        model_path = "runs/classify/drone_classifier/weights/best.pt"

        if not os.path.exists(model_path):
            st.warning("⚠️ Trained model weights (`best.pt`) not found! Please run your training script first.")
        else:
            # Load the model with caching so it doesn't reload on every click
            @st.cache_resource
            def load_model():
                return YOLO(model_path)

            model = load_model()

            # File uploader for drone images
            uploaded_file = st.file_uploader("Choose a drone image...", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                # Display uploaded image
                image = PIL.Image.open(uploaded_file)
                st.image(image, caption="Uploaded Drone Image for Inspection", width=400)

                if st.button("Run AI Damage Detection", type="primary"):
                    with st.spinner("Analyzing drone structure..."):
                        # Perform inference
                        results = model(image)
                        
                        # Extract prediction details
                        top1_index = results[0].probs.top1
                        top1_conf = results[0].probs.top1conf.item()
                        class_names = results[0].names
                        predicted_class = class_names[top1_index]

                        st.markdown("### 📊 Inspection Results")
                        
                        # Display success for healthy, warning/error for damages
                        if predicted_class == "healthy":
                            st.success(f"**Condition:** {predicted_class.upper()} 🟢")
                        else:
                            st.error(f"**Condition / Damage Detected:** {predicted_class.upper()} 🔴")
                        
                        st.metric(label="Confidence Score", value=f"{top1_conf:.2%}")

                        # Expandable breakdown of all classes
                        with st.expander("View detailed probabilities for all categories"):
                            probs = results[0].probs.data.tolist()
                            for idx, prob in enumerate(probs):
                                st.write(f"- **{class_names[idx]}**: {prob:.2%}")

# --- SECTION 2: FLEET REGISTRY & TELEMETRY CONTROL CENTER ---
        st.subheader("🚁 Drone Fleet & Telemetry Analytics")
        
        # Load data using your database connection function
        drones_fleet, telemetry_df = fetch_drone_fleet_data()

        if drones_fleet.empty:
            st.warning("No drone records found in the database. Please verify your `fleet_vehicles` table.")
        else:
            # --- TOP METRICS ROW ---
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_drones = len(drones_fleet)
            active_drones = len(drones_fleet[drones_fleet['fleet_status'].str.lower() == 'in service'])
            maint_drones = len(drones_fleet[drones_fleet['fleet_status'].str.lower() == 'maintenance'])
            avg_battery_health = telemetry_df['battery_health_pct'].mean() if not telemetry_df.empty else 0
            total_flight_hrs = telemetry_df['cumulative_flight_hours'].max() if not telemetry_df.empty else 0
            
            with col1:
                st.metric("Total Drones", total_drones)
            with col2:
                active_pct = round((active_drones / total_drones) * 100, 1) if total_drones > 0 else 0
                st.metric("In Service", active_drones, delta=f"{active_pct}%")
            with col3:
                st.metric("In Maintenance", maint_drones, delta_color="inverse")
            with col4:
                st.metric("Avg Fleet Battery Health", f"{round(avg_battery_health, 1)}%")
            with col5:
                st.metric("Max Cumulative Hours", f"{round(total_flight_hrs, 1)} hrs")

            st.markdown("---")

            # --- TABS FOR ORGANIZED VIEWS ---
            tab1, tab2, tab3 = st.tabs(["📊 Fleet Registry & Specs", "🌡️ Diagnostics & Telemetry", "⚠️ Maintenance & Error Alerts"])

            with tab1:
                st.subheader("Active Drone Fleet Inventory")
                status_filter = st.selectbox("Filter by Fleet Status", ["All"] + list(drones_fleet['fleet_status'].dropna().unique()))
                
                filtered_fleet = drones_fleet if status_filter == "All" else drones_fleet[drones_fleet['fleet_status'] == status_filter]
                    
                st.dataframe(filtered_fleet[[
                    'vehicle_id', 'model_name', 'hub_code', 'capacity_kg', 
                    'max_range_km', 'avg_speed_kmph', 'odometer_km', 
                    'fleet_status', 'battery_capacity_wh', 'last_service_date'
                ]], use_container_width=True)

            with tab2:
                st.subheader("Real-Time Flight Telemetry & Stress Analytics")
                st.caption("Left: Motor heat vs. vibration stress correlation | Right: Battery health degradation across charge cycles")
                if not telemetry_df.empty:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("##### Motor Temperature (°C) vs. Vibration (RMS)")
                        st.scatter_chart(telemetry_df, x='motor_temp_c', y='vibration_rms', color='gps_signal_quality')
                    with col_b:
                        st.markdown("##### Battery Degradation: Cycles vs. Health %")
                        st.line_chart(telemetry_df.sort_values('battery_cycles').set_index('battery_cycles')['battery_health_pct'])
                else:
                    st.info("No telemetry records available.")

            with tab3:
                st.subheader("Drone Maintenance Log & Error Inspector")
                if not telemetry_df.empty:
                    alert_df = telemetry_df[(telemetry_df['maintenance_required'] == 1) | (telemetry_df['error_codes'] != 'NONE')]
                    st.warning(f"⚠️ Total flagged telemetry logs requiring engineering review: **{len(alert_df)}**")
                    st.dataframe(alert_df[[
                        'flight_id', 'drone_id', 'flight_timestamp', 'battery_health_pct', 
                        'motor_temp_c', 'vibration_rms', 'error_codes', 'route_deviation_m'
                    ]], use_container_width=True)
                else:
                    st.info("No alert records available.")
                
    elif selected_page == "🤖 AI Assistant":

        st.title("🤖 Smartlogix AI Assistant")
        st.caption("Natural Language Database Interface powered by Google Gemini & LangChain SQL RAG.")

        # --- 1. Initialize Session State for Chat History ---
        if "ai_chat_messages" not in st.session_state:
            st.session_state["ai_chat_messages"] = [
                {
                    "role": "assistant",
                    "content": "👋 Hello! I am your Smartlogix operational assistant. Ask me anything about your active database (e.g., *'Show me all pending orders'*, *'How many vehicles are currently in maintenance?'*, or *'List top 5 heavy cargo drones'*)."
                }
            ]

        # --- 2. Top Quick Prompt Suggestions ---
        st.markdown("**💡 Quick Suggested Prompts:**")
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        
        # Use session state hooks or temporary query variables for quick chips
        suggested_query = None
        with q_col1:
            if st.button("🚚 Active Fleet Status", use_container_width=True):
                suggested_query = "How many vehicles are currently in service versus maintenance?"
        with q_col2:
            if st.button("📦 Pending Orders", use_container_width=True):
                suggested_query = "Show me all orders that are currently pending shipment."
        with q_col3:
            if st.button("🌦️ Recent Weather Alerts", use_container_width=True):
                suggested_query = "Show recent weather data entries affecting hubs."
        with q_col4:
            if st.button("⭐ Customer Reviews", use_container_width=True):
                suggested_query = "What are the latest low-rating customer reviews?"

        st.markdown("---")

        # --- 3. Render Chat History Container ---
        chat_container = st.container()
        with chat_container:
            for message in st.session_state["ai_chat_messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # --- 4. Chat Input & Response Execution Logic ---
        # Combine user typing or quick-chip click into active prompt input
        user_input = st.chat_input("Ask a question about Smartlogix data...")
        
        target_prompt = suggested_query if suggested_query else user_input

        if target_prompt:
            # Append user message to history
            st.session_state["ai_chat_messages"].append({"role": "user", "content": target_prompt})
            with st.chat_message("user"):
                st.markdown(target_prompt)

            # Generate Assistant Response using your backend RAG function
            with st.chat_message("assistant"):
                with st.spinner("Analyzing schema & querying PostgreSQL database..."):
                    # Retrieve login context if defined elsewhere in your session, with defaults fallback
                    current_login_type = st.session_state.get("login_type", "Employee")
                    current_emp_role = st.session_state.get("emp_role", "Admin")
                    
                    # Call your core RAG backend function
                    response_payload = smartlogix_ai_chatbot_response(
                        prompt_query=target_prompt, 
                        login_type=current_login_type, 
                        emp_role=current_emp_role
                    )
                    
                    # Check if response contains developer SQL audit block to render clean layout expansion
                    if "🔍 View Generated SQL Query" in response_payload:
                        parts = response_payload.split("🔍 View Generated SQL Query")
                        main_answer = parts[0].replace("📊 **Database Response:**", "").strip()
                        sql_query_text = parts[1].replace("(Developer Audit)", "").strip()
                        
                        # st.markdown(f"📊 **Database Response:**\n\n{main_answer}")
                        st.markdown(f"{main_answer}")
                        
                        # with st.expander("🔍 View Generated SQL Query (Developer Audit)"):
                        #     st.code(sql_query_text, language="sql")
                            
                        final_storage_string = f"📊 **Database Response:**\n{main_answer}\n\n🔍 View Generated SQL Query (Developer Audit)\n```sql\n{sql_query_text}\n```"
                    else:
                        st.markdown(response_payload)
                        final_storage_string = response_payload

            # Append assistant response to history
            st.session_state["ai_chat_messages"].append({"role": "assistant", "content": final_storage_string})
            
            # Rerun to cleanly clear quick chips context state if clicked
            if suggested_query:
                st.rerun()

        # --- 5. Bottom Action Controls ---
        col_c1, col_c2 = st.columns([6, 1])
        with col_c2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state["ai_chat_messages"] = [
                    {
                        "role": "assistant",
                        "content": "👋 Chat history cleared. How else can I help you navigate your logistics database?"
                    }
                ]
                st.rerun()       

    elif selected_page == "🛍️ Products":    
        render_products_page()
   
# ------------------------------------------
# INTERFACE B: CUSTOMER VIEW (Complete & Polished)
# ------------------------------------------

if user_role == "Customer Account":
    st.title("📦 SmartLogix AI — Customer Portal")

    # Initialize Session States
    if 'cart' not in st.session_state:
        st.session_state.cart = set()
    if 'wishlist' not in st.session_state:
        st.session_state.wishlist = set()
    if 'fk_page' not in st.session_state:
        st.session_state.fk_page = 1
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "all"
    if 'delivery_location' not in st.session_state:
        st.session_state.delivery_location = "600116, Chennai"
    if "cart_qty" not in st.session_state:
        st.session_state.cart_qty = {}
    if "checkout_success" not in st.session_state:
        st.session_state.checkout_success = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Top Header Bar: Location Selector + Search + Account + Cart + AI Bot
    sc_loc, sc1, sc2, sc3, sc4 = st.columns([1.5, 3, 1.2, 1, 1.2]) 
    
    with sc_loc:
        loc_display = st.session_state.delivery_location[:18]
        with st.popover(f"📍 {loc_display}...", use_container_width=True):
            st.markdown("**Select Delivery Location**")
            new_pincode = st.text_input("Enter Pincode / Area:", value=st.session_state.delivery_location, key="popover_pincode_input")
            if st.button("Apply Location", key="apply_loc_btn"):
                st.session_state.delivery_location = new_pincode
                st.toast(f"Location updated to: {new_pincode}")
                st.rerun()

    with sc1:
        search_query = st.text_input("🔍 Search Products", placeholder="Search products, brands and more...", key="catalog_search", label_visibility="collapsed")

    with sc2:
        with st.popover("👤 My Account", use_container_width=True):
            if st.button("👤 My Profile", key="acc_profile", use_container_width=True):
                st.session_state.view_mode = "profile"
                st.rerun()
            
            wish_count = f" ({len(st.session_state.wishlist)})" if len(st.session_state.wishlist) > 0 else ""
            if st.button(f"❤️ Wishlist{wish_count}", key="acc_wishlist", use_container_width=True):
                st.session_state.view_mode = "wishlist"
                st.rerun()

            if st.button("📦 My Orders", key="acc_orders", use_container_width=True):
                st.session_state.view_mode = "orders"
                st.rerun()

            st.divider()
            if st.button("🚪 Logout", key="acc_logout", use_container_width=True):
                logout()

    with sc3:
        cart_active = st.session_state.view_mode == "cart"
        if st.button(f"🛒 Cart ({len(st.session_state.cart)})", key="nav_cart_btn", use_container_width=True, type="primary" if cart_active else "secondary"):
            st.session_state.view_mode = "all" if cart_active else "cart"
            st.rerun()

    with sc4:
        with st.popover("💬 Chatbot", use_container_width=True):
            st.markdown("### SmartLogix Assistant")
            current_role = st.session_state.get("user_role", "Customer")
            st.caption(f"Context Mode: **{current_role}**")

            chat_container = st.container(height=320)
            with chat_container:
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f'{msg["content"]}', unsafe_allow_html=True)
                    else:
                        st.markdown(f'{msg["content"]}', unsafe_allow_html=True)

            with st.form(key="popover_ai_form", clear_on_submit=True):
                user_prompt = st.text_input("Message", placeholder="Type a message...", label_visibility="collapsed")
                submitted = st.form_submit_button("Send Message", use_container_width=True)
                
                if submitted and user_prompt.strip():
                    st.session_state.messages.append({"role": "user", "content": user_prompt})
                    bot_reply = customer_ai_chatbot_response(
                        user_prompt=user_prompt,
                        vectorstore=st.session_state.get("vectorstore")
                    )
                    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                    st.rerun()



    # Fetch the products catalog dataframe once (or cache/load it as needed)
    products_df = fetch_products_df()

    def get_product_info(item_id):
        """Helper to pull name and price from products dataframe based on product_id."""
        if not products_df.empty:
            match = products_df[products_df['product_id'] == item_id]
            if not match.empty:
                return {
                    'name': match.iloc[0]['product_name'],
                    'price': match.iloc[0]['price_amount'],
                    'category': match.iloc[0].get('category', 'General')
                }
        # Fallback if product_id isn't found
        return {'name': f'Product {item_id}', 'price': 0.00, 'category': 'General'}

    # VIEW: CART
    if st.session_state.view_mode == "cart":
        hdr_col1, hdr_col2 = st.columns([4, 1])
        with hdr_col1:
            st.subheader("🛒 Shopping Cart")
        with hdr_col2:
            if st.button("✖ Back to Catalog", key="back_from_cart", use_container_width=True):
                st.session_state.view_mode = "all"
                st.rerun()
        
        if not st.session_state.get('cart'):
            st.info("Your shopping cart is currently empty.")
        else:
            st.success(f"You have {len(st.session_state.cart)} item(s) in your cart.")
            st.divider()
            
            for item_id in list(st.session_state.cart):
                prod = get_product_info(item_id)
                
                col_img, col_info, col_action = st.columns([1, 3, 1])
                with col_img:
                    st.markdown("📦 **Item**")
                with col_info:
                    st.markdown(f"**Product ID:** `{item_id}`")
                    st.markdown(f"**{prod['name']}**")
                    st.markdown(f"**Price:** ₹{prod['price']:,.2f}")
                with col_action:
                    if st.button("Remove", key=f"remove_cart_{item_id}"):
                        st.session_state.cart.remove(item_id)
                        st.rerun()
                st.divider()
                
            if st.button("Proceed to Checkout", type="primary", use_container_width=True):
                st.session_state.view_mode = "checkout"
                st.rerun()

    # VIEW: WISHLIST
    elif st.session_state.view_mode == "wishlist":
        hdr_col1, hdr_col2 = st.columns([4, 1])
        with hdr_col1:
            st.subheader("❤️ My Wishlist")
        with hdr_col2:
            if st.button("✖ Back to Catalog", key="back_from_wishlist", use_container_width=True):
                st.session_state.view_mode = "all"
                st.rerun()
                
        if not st.session_state.get('wishlist'):
            st.info("Your wishlist is currently empty. Browse the catalog and save items!")
        else:
            st.success(f"You have {len(st.session_state.wishlist)} item(s) saved in your wishlist.")
            st.divider()
            
            for item_id in list(st.session_state.wishlist):
                prod = get_product_info(item_id)
                
                col_img, col_info, col_action = st.columns([1, 3, 1])
                with col_img:
                    st.markdown("❤️ **Saved**")
                with col_info:
                    st.markdown(f"**Product ID:** `{item_id}`")
                    st.markdown(f"**{prod['name']}**")
                    st.markdown(f"**Price:** ₹{prod['price']:,.2f}")
                with col_action:
                    if st.button("Move to Cart", key=f"wish_to_cart_{item_id}", type="primary"):
                        if 'cart' not in st.session_state:
                            st.session_state.cart = set()
                        st.session_state.cart.add(item_id)
                        st.session_state.wishlist.remove(item_id)
                        st.toast("Moved item from Wishlist to Cart!")
                        st.rerun()
                    if st.button("Remove", key=f"remove_wish_{item_id}"):
                        st.session_state.wishlist.remove(item_id)
                        st.rerun()
                st.divider()

    elif st.session_state.view_mode == "checkout":
        st.subheader("💳 Checkout Form")
        
        if st.button("✖ Back to Cart"):
            st.session_state.view_mode = "cart"
            st.rerun()
            
        st.divider()
        
        with st.form("basic_checkout"):
            name = st.text_input("Full Name", value="Customer")
            address = st.text_area("Delivery Address", value="Porur, Chennai - 600116")
            payment = st.selectbox("Payment Method", ["UPI", "Credit Card", "Cash on Delivery"])
            
            submitted = st.form_submit_button("Place Order", type="primary", use_container_width=True)
            if submitted:
                from db_manager import insert_new_customer_order, fetch_products_df
                
                # Fetch products to compute total value of items in cart
                products_df = fetch_products_df()
                total_value = 0.0
                first_product_id = "PRD-001"
                
                cart_items = list(st.session_state.get('cart', []))
                if cart_items:
                    first_product_id = cart_items[0]
                    if not products_df.empty:
                        matched_prods = products_df[products_df['product_id'].isin(cart_items)]
                        if not matched_prods.empty:
                            total_value = float(matched_prods['price_amount'].sum())
                
                # Generate unique order ID
                new_order_id = f"ORD-{random.randint(100000, 999999)}"
                
                # Build payload matching your database function parameters
                order_payload = {
                    "order_id": new_order_id,
                    "customer_id": "CUST-001",
                    "product_id": first_product_id,
                    "quantity": len(cart_items) if cart_items else 1,
                    "destination_city": address,
                    "destination_pincode": 600116,
                    "destination_lat": 13.0382,
                    "destination_lon": 80.1565,
                    "payment_mode": payment,
                    "order_value_inr": total_value if total_value > 0 else 1999.00
                }
                
                # Insert into PostgreSQL database
                success = insert_new_customer_order(order_payload)
                
                if success:
                    st.success(f"Order Placed Successfully! 🎉 (Order ID: {new_order_id})")
                    # Clear the cart after successful order placement
                    st.session_state.cart = set()
                else:
                    st.error("Failed to save order to database. Please check your connection.")

    # VIEW: MY ORDERS (Side-by-Side Timeline and Compact Corner Map)
    elif st.session_state.view_mode == "orders":
        hdr_col1, hdr_col2 = st.columns([4, 1])
        with hdr_col1:
            st.subheader("📦 My Orders")
        with hdr_col2:
            if st.button("✖ Back to Catalog", key="back_from_orders", use_container_width=True):
                st.session_state.view_mode = "all"
                st.rerun()

        st.markdown("Here is your most recently placed order, live tracking, and destination map.")
        st.divider()

        from db_manager import fetch_customer_orders
        
        customer_id = "CUST-001"
        orders_df = fetch_customer_orders(customer_id)

        if orders_df.empty:
            st.info("No recent orders found in the database. Place an order from the checkout page to see it here!")
        else:
            latest_order = orders_df.iloc[0]
            
            # Display Last Placed Order Summary Card
            with st.container():
                st.markdown("### 🔔 Last Placed Order Summary")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric(label="Order ID", value=str(latest_order.get('order_id', 'N/A')))
                with col_b:
                    st.metric(label="Total Value", value=f"₹{float(latest_order.get('order_value_inr', 0.0)):,.2f}")
                with col_c:
                    st.metric(label="Payment Mode", value=str(latest_order.get('payment_mode', 'UPI')))

                st.markdown(f"**📦 Product ID:** `{latest_order.get('product_id', 'N/A')}`")
                st.markdown(f"**📍 Delivery Destination:** {latest_order.get('destination_city', 'Porur, Chennai - 600116')}")
                
                st.divider()

                # 🔀 SIDE-BY-SIDE LAYOUT: Timeline on Left, Compact Corner Map on Right
                left_col, right_col = st.columns([1.5, 1], gap="medium")

                with left_col:
                    st.markdown("#### ⏳ Shipment Timeline")
                    current_status = str(latest_order.get('order_status', 'Placed'))
                    
                    st.markdown(f"**Current Status:** `{current_status}`")
                    
                    # Dynamically adjust progress bar based on status
                    progress_val = 25 if current_status == "Placed" else (75 if current_status == "Processing" else 100)
                    st.progress(progress_val, text=f"Order status: {current_status}")
                    
                    st.markdown(
                        f"""
                        * {'✅' if progress_val >= 25 else '⏳'} **Order Placed** (Successfully Verified)
                        * {'✅' if progress_val >= 75 else '⏳'} **Packed & Dispatched** (Porur Hub)
                        * {'🚚' if progress_val >= 100 else '⏳'} **Out for Delivery / Processing**
                        """
                    )

                with right_col:
                    st.markdown("#### 📍 Location Map")
                    lat = float(latest_order.get('destination_lat', 13.0382))
                    lon = float(latest_order.get('destination_lon', 80.1565))
                    
                    import folium
                    from streamlit_folium import st_folium

                    # Create a compact map for the corner
                    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")
                    folium.Marker(
                        [lat, lon],
                        popup=f"Order: {latest_order.get('order_id')}",
                        icon=folium.Icon(color="red", icon="info-sign")
                    ).add_to(m)

                    # Render smaller map to fit the side column cleanly
                    st_folium(m, width=320, height=220, returned_objects=[])
                
                st.divider()
                
            st.markdown("### 📜 Full Order History")
            st.dataframe(
                orders_df[['order_id', 'product_id', 'quantity', 'order_value_inr', 'payment_mode', 'order_status', 'destination_city']], 
                use_container_width=True
            )

    # VIEW: OTHER ACCOUNT VIEWS (Profile, Coupons, Gift Cards, Notifications)
    elif st.session_state.view_mode in ["profile", "coupons", "gift_cards", "notifications"]:
        st.divider()
        hdr_col1, hdr_col2 = st.columns([4, 1])
        with hdr_col1:
            # VIEW: MY PROFILE
            if st.session_state.view_mode == "profile":
                hdr_col1, hdr_col2 = st.columns([4, 1])
                with hdr_col1:
                    st.subheader("👤 My Profile")
                                      
                # Layout using Tabs or Columns for a clean dashboard look
                prof_tab1, prof_tab2, prof_tab3 = st.tabs(["Personal Details", "Saved Addresses", "Account Security"])

                with prof_tab1:
                    st.markdown("### Personal Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_input("Full Name", value="user")
                        st.text_input("Email Address", value="user@gmail.com")
                    with col2:
                        st.text_input("Phone Number", value="+91 98765 43210")
                        st.text_input("Customer ID", value="CUST-001", disabled=True)
                    
                    if st.button("Update Profile", type="primary"):
                        st.success("Profile details updated successfully!")

                with prof_tab2:
                    st.markdown("### Saved Delivery Addresses")
                    
                    # Address Card 1
                    st.markdown("""Default Shipping Address Porur, Chennai - 600116  Tamil Nadu, India

                    """, 
                    unsafe_allow_html=True
                )
                
                if st.button("➕ Add New Address"):
                    st.toast("Address management modal opened.")

                with prof_tab3:
                    st.markdown("### Account Security & Preferences")
                    st.selectbox("Preferred Communication Channel", ["Email", "SMS", "WhatsApp Notifications"])
                    st.toggle("Two-Factor Authentication (2FA)", value=True)
                    st.toggle("Receive Order Tracking Alerts", value=True)
                    
                    if st.button("Save Preferences"):
                        st.success("Preferences saved!")
            elif st.session_state.view_mode == "coupons":
                st.subheader("🎟️ My Coupons")
                st.success("Available Coupon: **SMARTLOGIX10** (10% OFF on all electronics)")
            elif st.session_state.view_mode == "gift_cards":
                st.subheader("💳 Gift Cards")
                st.info("Gift Card Balance: ₹0.00")
            elif st.session_state.view_mode == "notifications":
                st.subheader("🔔 Notifications")
                st.success("Your recent order has been dispatched.")
                
        with hdr_col2:
            if st.button("✖ Back to Catalog", key="back_from_account", use_container_width=True):
                st.session_state.view_mode = "all"
                st.rerun()

#VIEW 3: PRODUCT CATALOG / CART / WISHLIST (WITH SEARCH FILTERING)
    else:
# --- 1. SAFE CATALOG FETCHING & SEARCH FILTERING ---
        filtered_df = fetch_products_df()
        if filtered_df is None:
            filtered_df = pd.DataFrame()

    # Apply search filtering logic dynamically
        if search_query and not filtered_df.empty:
            query = search_query.strip().lower()
            mask = filtered_df['product_name'].astype(str).str.lower().str.contains(query, na=False)
            if 'category' in filtered_df.columns:
                mask = mask | filtered_df['category'].astype(str).str.lower().str.contains(query, na=False)
            filtered_df = filtered_df[mask]

        current_page = st.session_state.get("fk_page", 1)

        try:
            cols_per_row = 4
            items_per_page = 12
            total_items = len(filtered_df)
            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

            if current_page > total_pages or current_page < 1:
                current_page = 1
                st.session_state.fk_page = 1

            start_idx = (current_page - 1) * items_per_page
            page_items = filtered_df.iloc[start_idx : start_idx + items_per_page].to_dict('records')

            if len(page_items) == 0:
                st.info("No products match your search query or filters.")
                if st.button("Clear Search & View All"):
                    st.rerun()
            else:
                for row_idx in range(0, len(page_items), cols_per_row):
                    row_items = page_items[row_idx : row_idx + cols_per_row]
                    cols = st.columns(cols_per_row)

                    for col, item in zip(cols, row_items):
                        p_id = item["product_id"]
                        avg_rating = item.get("avg_rating", 4.2)
                        rating_val = 4.2 if pd.isna(avg_rating) else round(float(avg_rating), 1)

                        raw_price = item.get("price_amount", 0)
                        try:
                            display_price = float(raw_price) if pd.notna(raw_price) and str(raw_price).strip() != "" else 0.0
                        except (ValueError, TypeError):
                            display_price = 0.0

                        raw_url = item.get("image_url")
                        image_url = (
                            str(raw_url).strip()
                            if pd.notna(raw_url) and str(raw_url).strip() != ""
                            else "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=500"
                        )

                        with col:
                            st.markdown(
                                    f"""
                                    <div style="
                                        position: relative; 
                                        height: 170px; 
                                        width: 100%; 
                                        background: #f1f5f9; 
                                        border-radius: 10px; 
                                        display: flex; 
                                        align-items: center; 
                                        justify-content: center; 
                                        overflow: hidden;
                                        border: 1px solid #e2e8f0;
                                        margin-bottom: 8px;
                                    ">
                                        <img src="{image_url}" style="max-height: 150px; max-width: 90%; object-fit: contain;">
                                        <span style="
                                            position: absolute; 
                                            bottom: 8px; 
                                            left: 8px; 
                                            background: #15803d; 
                                            color: white; 
                                            padding: 2px 7px; 
                                            border-radius: 6px; 
                                            font-size: 11px; 
                                            font-weight: 700;
                                        ">★ {rating_val}</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )                                                 
                            st.markdown(f"{item['product_name']}", unsafe_allow_html=True)
                            st.markdown(f"""₹{display_price:,.0f}""", unsafe_allow_html=True)

                            in_cart = p_id in st.session_state.setdefault("cart", set())
                            in_wish = p_id in st.session_state.setdefault("wishlist", set())

                            b1, b2, b3 = st.columns([1, 1, 1.8])
                            with b1:
                                if st.button("✓" if in_cart else "🛒", key=f"cart_{p_id}_{current_page}", use_container_width=True):
                                    if in_cart:
                                        st.session_state.cart.remove(p_id)
                                    else:
                                        st.session_state.cart.add(p_id)
                                    st.rerun()

                            with b2:
                                if st.button("❤️" if in_wish else "🤍", key=f"wish_{p_id}_{current_page}", use_container_width=True):
                                    if in_wish:
                                        st.session_state.wishlist.remove(p_id)
                                    else:
                                        st.session_state.wishlist.add(p_id)
                                    st.rerun()

                            with b3:
                                if st.button("Buy", key=f"buy_{p_id}_{current_page}", type="primary", use_container_width=True):
                                    st.session_state.cart = {p_id}
                                    st.session_state.setdefault("cart_qty", {})[p_id] = 1
                                    st.session_state.view_mode = "cart"
                                    st.rerun()

                # Pagination Controls
                if total_pages > 1:
                    st.markdown(" ", unsafe_allow_html=True)
                    nav_cols = st.columns([1] + [0.6] * min(total_pages, 5) + [1])

                    with nav_cols[0]:
                        if current_page > 1:
                            if st.button("⬅ PREV", key="page_btn_prev", use_container_width=True):
                                st.session_state.fk_page -= 1
                                st.rerun()

                    start_p = max(1, current_page - 2)
                    end_p = min(total_pages, start_p + 4)
                    page_numbers = list(range(start_p, end_p + 1))

                    for idx, p_num in enumerate(page_numbers):
                        with nav_cols[idx + 1]:
                            btn_type = "primary" if p_num == current_page else "secondary"
                            if st.button(str(p_num), type=btn_type, key=f"page_btn_{p_num}", use_container_width=True):
                                st.session_state.fk_page = p_num
                                st.rerun()

                    with nav_cols[-1]:
                        if current_page < total_pages:
                            if st.button("NEXT ➡", key="page_btn_next", use_container_width=True):
                                st.session_state.fk_page += 1
                                st.rerun()

        except Exception as e:
            st.error(f"Error loading catalog data: {e}")