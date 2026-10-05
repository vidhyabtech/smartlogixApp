import os
import json
from db_manager import get_connection
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv("1.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY missing! Check your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# TOOL DEFINITIONS FOR GEMINI FUNCTION CALLING
# ==========================================

def search_products(keyword: str = None, max_price: float = None, category: str = None) -> str:
    """
    Search the products catalog based on keywords, optional maximum price limit, or category.
    Use this tool when users ask for recommendations, specific items, or items under a certain budget.
    """
    conn = get_connection()
    if not conn:
        return json.dumps({"error": "Database connection failed."})

    try:
        with conn.cursor() as cursor:
            query = """
                SELECT product_id, product_name, category, price_amount, price_currency, stock_qty, avg_rating, specs 
                FROM products 
                WHERE 1=1
            """
            params = []

            if keyword:
                query += " AND (product_name ILIKE %s OR category ILIKE %s OR sub_category ILIKE %s OR tags ILIKE %s)"
                kw_wildcard = f"%{keyword}%"
                params.extend([kw_wildcard, kw_wildcard, kw_wildcard, kw_wildcard])

            if max_price is not None:
                query += " AND price_amount <= %s"
                params.append(max_price)

            if category:
                query += " AND category ILIKE %s"
                params.append(f"%{category}%")

            query += " ORDER BY avg_rating DESC LIMIT 5;"
            
            cursor.execute(query, tuple(params))
            results = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description]
            
            products_list = [dict(zip(colnames, row)) for row in results]
            conn.close()
            return json.dumps(products_list, default=str)

    except Exception as e:
        if conn:
            conn.close()
        return json.dumps({"error": str(e)})


def compare_products(product_keyword_1: str, product_keyword_2: str) -> str:
    """
    Compare two products side-by-side using keywords matching their product names.
    Use this tool when users ask to compare two specific products or models.
    """
    conn = get_connection()
    if not conn:
        return json.dumps({"error": "Database connection failed."})

    try:
        with conn.cursor() as cursor:
            query = """
                SELECT product_id, product_name, category, price_amount, price_currency, stock_qty, avg_rating, weight_kg, specs 
                FROM products 
                WHERE product_name ILIKE %s OR product_name ILIKE %s
                LIMIT 2;
            """
            cursor.execute(query, (f"%{product_keyword_1}%", f"%{product_keyword_2}%"))
            results = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description]
            
            comparison_list = [dict(zip(colnames, row)) for row in results]
            conn.close()
            return json.dumps(comparison_list, default=str)

    except Exception as e:
        if conn:
            conn.close()
        return json.dumps({"error": str(e)})


def summarize_product_reviews(product_keyword: str) -> str:
    """
    Fetch and summarize customer reviews, ratings, and sentiment for a given product keyword.
    Use this tool when users ask what people are saying about an item, customer feedback, or reviews.
    """
    conn = get_connection()
    if not conn:
        return json.dumps({"error": "Database connection failed."})

    try:
        with conn.cursor() as cursor:
            query = """
                SELECT r.rating, r.review_title, r.review_text, r.sentiment_label, p.product_name
                FROM customer_reviews r
                JOIN products p ON r.product_id = p.product_id
                WHERE p.product_name ILIKE %s
                LIMIT 5;
            """
            cursor.execute(query, (f"%{product_keyword}%",))
            results = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description]
            
            reviews_list = [dict(zip(colnames, row)) for row in results]
            conn.close()
            return json.dumps(reviews_list, default=str)

    except Exception as e:
        if conn:
            conn.close()
        return json.dumps({"error": str(e)})


def track_order(order_id: str) -> str:
    """
    Track the real-time status, shipping progress, transport mode, and delivery details of an order.
    Use this tool when a user asks about order tracking, delivery status, or delivery database logs.
    """
    conn = get_connection()
    if not conn:
        return json.dumps({"error": "Database connection failed."})

    try:
        with conn.cursor() as cursor:
            query = """
                SELECT o.order_id, o.order_status, o.transport_mode, o.destination_city, 
                       o.assigned_vehicle_id, o.promised_eta_hours, o.order_value_inr,
                       d.event_type, d.location_city, d.event_timestamp
                FROM orders o
                LEFT JOIN delivery_logs d ON o.order_id = d.order_id
                WHERE o.order_id ILIKE %s
                ORDER BY d.event_seq DESC
                LIMIT 3;
            """
            cursor.execute(query, (f"%{order_id}%",))
            results = cursor.fetchall()
            
            if not results:
                conn.close()
                return json.dumps({"error": f"Order ID {order_id} not found."})

            colnames = [desc[0] for desc in cursor.description]
            logs_list = [dict(zip(colnames, row)) for row in results]
            conn.close()
            return json.dumps(logs_list, default=str)

    except Exception as e:
        if conn:
            conn.close()
        return json.dumps({"error": str(e)})


# Map all tools
available_tools = {
    "search_products": search_products,
    "compare_products": compare_products,
    "summarize_product_reviews": summarize_product_reviews,
    "track_order": track_order
}


# ==========================================
# MAIN CHATBOT FUNCTION WITH GEMINI TOOLS
# ==========================================

def customer_ai_chatbot_response(user_prompt, vectorstore=None, **kwargs):
    """
    Intelligent LLM assistant supporting product search, comparisons, 
    review summaries, order tracking, and RAG knowledge retrieval.
    """
    system_instruction = (
        "You are SmartLogix AI, an e-commerce shopping assistant. "
        "CRITICAL: Whenever a user asks for recommendations, products, items, shoes, mobiles, or budgets, "
        "you MUST call the 'search_products' tool. Never give a generic refusal or fallback response "
        "when a product search tool is available."
    )

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[search_products, compare_products, summarize_product_reviews, track_order],
                temperature=0.3,
            ),
        )

        if response.function_calls:
            tool_outputs = []
            for function_call in response.function_calls:
                name = function_call.name
                args = function_call.args

                if name in available_tools:
                    result_json_str = available_tools[name](**args)
                    tool_outputs.append(
                        types.Part.from_function_response(
                            name=name,
                            response={"result": json.loads(result_json_str)}
                        )
                    )

            final_response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[
                    types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]),
                    types.Content(role="model", parts=response.candidates[0].content.parts),
                    types.Content(role="user", parts=tool_outputs)
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3
                )
            )
            return final_response.text

        if response.text:
            return response.text

    except Exception as e:
        print(f"❌ Gemini Chatbot Error: {e}")

    # Fallback to vector store RAG for FAQs or general policies
    if vectorstore:
        try:
            docs = vectorstore.similarity_search(user_prompt, k=2)
            if docs:
                return f"Here is what I found in our knowledge base:\n\n" + "\n".join([f"> {d.page_content}" for d in docs])
        except Exception:
            pass

    return f"I'm here to help! Ask me to recommend products[cite: 4], compare items[cite: 4], read customer reviews[cite: 4], or track order status[cite: 4]."