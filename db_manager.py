import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st

# PostgreSQL Credentials
DB_CONFIG = {
    "host": "localhost",
    "dbname": "smartlogix_db",
    "user": "postgres",
    "password": "password",  # Update with your actual password
    "port": "5432",
}


def get_connection():
    """Establishes database connection with safe error handling."""
    try:
        return psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as e:
        st.error(f"⚠️ Database Connection Failed: {e}")
        return None


def insert_new_customer_order(order_data):
    """Inserts a newly placed order from Customer View into PostgreSQL orders table."""
    conn = get_connection()
    if not conn:
        return False

    cursor = conn.cursor()
    query = """
        INSERT INTO orders (
            order_id, customer_id, product_id, quantity, order_date, 
            destination_city, destination_pincode, destination_lat, destination_lon,
            payment_mode, order_value_inr, transport_mode, order_status
        ) VALUES (%s, %s, %s, %s, CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    try:
        cursor.execute(
            query,
            (
                order_data["order_id"],
                order_data.get("customer_id", "CUST-GUEST"),
                order_data.get("product_id", "PRD-001"),
                order_data.get("quantity", 1),
                order_data.get("destination_city", "Porur, Chennai"),
                order_data.get("destination_pincode", 600116),
                order_data.get("destination_lat", 13.0382),
                order_data.get("destination_lon", 80.1565),
                order_data.get("payment_mode", "UPI"),
                order_data.get("order_value_inr", 1999.00),
                "Pending",  # Transport mode initially pending ML classification
                "Placed",   # Initial status
            ),
        )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Failed to insert order: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


@st.cache_data(ttl=5)
def fetch_all_orders_df():
    """Fetches all orders from the 'orders' table."""
    conn = get_connection()
    if not conn:
        return pd.DataFrame()

    try:
        query = "SELECT * FROM orders ORDER BY order_item_id DESC;"
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching orders: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def fetch_customers_df():
    """Fetches all customers from the 'customers' table."""
    conn = get_connection()
    if not conn:
        return pd.DataFrame()
    try:
        query = "SELECT * FROM customers;"
        return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Error fetching customers: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def fetch_products_df():
    """Fetches all products from the 'products' table (matching DDL schema)."""
    conn = get_connection()
    if not conn:
        return pd.DataFrame()
    try:
        # Corrected from 'cleaned_product_catalog' to match DDL table 'products'
        query = "SELECT * FROM products;"
        return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Error fetching products: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def fetch_delivery_logs_df():
    """Fetches all delivery logs from the 'delivery_logs' table."""
    conn = get_connection()
    if not conn:
        return pd.DataFrame()
    try:
        query = "SELECT * FROM delivery_logs;"
        return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Error fetching delivery logs: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def update_order_transport_mode(order_id, predicted_mode):
    """Updates order transport mode after ML classification."""
    conn = get_connection()
    if not conn:
        return False

    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE orders SET transport_mode = %s WHERE order_id = %s;",
            (predicted_mode, order_id),
        )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Failed to update transport mode: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def fetch_customer_orders(customer_id="CUST-001"):
    """Fetches orders for a specific customer from the 'orders' table."""
    conn = get_connection()
    if not conn:
        return pd.DataFrame()

    try:
        query = "SELECT * FROM orders WHERE customer_id = %s ORDER BY order_item_id DESC;"
        df = pd.read_sql_query(query, conn, params=(customer_id,))
        return df
    except Exception as e:
        st.error(f"Error fetching customer orders: {e}")
        return pd.DataFrame()
    finally:
        conn.close()