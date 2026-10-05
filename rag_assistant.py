import os
from sqlalchemy import create_engine
import streamlit as st
from dotenv import load_dotenv
import re

# Ensure environment variables are loaded
load_dotenv(override=True)

# LangChain imports for SQL RAG
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

# Import your dynamic schema selector from db_schema.py
from db_schema import get_schema_for_user

# Pulls DATABASE_URL from your .env file automatically
DB_URI = os.getenv(
    "DATABASE_URL", "postgresql://postgres:password@localhost:5432/smartlogix_db"
)

@st.cache_resource
def get_sql_database_connection(login_type="Employee", emp_role=None):
    """Connects LangChain SQLDatabase wrapper to PostgreSQL based on user role."""
    try:
        # Safely check if 'Customer' is part of the role string from Streamlit session
        if login_type and "Customer" in login_type:
            allowed_tables = ["products", "orders", "customer_reviews"]
        else:
            # Full admin/employee access to all 11 tables
            allowed_tables = [
                "customers", "products", "orders", "delivery_logs", 
                "customer_reviews", "fleet_vehicles", "maintenance_history", 
                "drone_telemetry", "gps_routes", "traffic_data", "weather_data"
            ]

        db = SQLDatabase.from_uri(
            DB_URI, 
            include_tables=allowed_tables,
            sample_rows_in_table_info=2
        )
        return db
    except Exception as e:
        print(f"Error connecting to database for RAG: {e}")
        return None

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def smartlogix_ai_chatbot_response(prompt_query: str, login_type="Employee", emp_role=None) -> str:
    """
    Core RAG function using Gemini API, local .env configurations, and dynamic role checking.
    Converts raw SQL outputs into clean, natural human-readable answers.
    """
    # 1. Handle casual chat/greetings so they don't hit the database
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good evening"]
    if prompt_query.strip().lower() in greetings:
        return f"👋 Hello! I am ready to assist you under the **{login_type}** profile. Ask me any question about your database (e.g., *Show me status of order ORD-004097* or *Tell me about customer Abhis*)."

    db = get_sql_database_connection(login_type, emp_role)
    
    if db is None:
        return "⚠️ Database connection is unavailable. Please check your PostgreSQL configuration."

    if not os.getenv("GOOGLE_API_KEY"):
        return "⚠️ `GOOGLE_API_KEY` was not found in your `.env` file. Please check your configuration."

    try:
        # Initialize Gemini model (gemini-1.5-flash)
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

        # Create the Text-to-SQL generation chain
        write_query = create_sql_query_chain(llm, db)
        execute_query = QuerySQLDataBaseTool(db=db)

        # Step 1: Generate SQL from user prompt using Gemini
        generated_sql = write_query.invoke({"question": prompt_query})
        
        # Robustly extract the SQL query starting from 'SELECT'
        match = re.search(r"(SELECT\s.+)", str(generated_sql), re.IGNORECASE | re.DOTALL)
        if match:
            cleaned_sql = match.group(1).strip()
        else:
            cleaned_sql = str(generated_sql).replace("SQLQuery:", "").strip()
            
        # Strip trailing markdown code blocks if present
        if cleaned_sql.endswith("```"):
            cleaned_sql = cleaned_sql.rsplit("```", 1)[0].strip()
        if cleaned_sql.startswith("```sql"):
            cleaned_sql = cleaned_sql[6:].strip()

        # Step 2: Execute SQL safely against PostgreSQL
        query_result = execute_query.invoke(cleaned_sql)

        # Step 3: Use Gemini to translate raw database outputs into clean natural language sentences
        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL execution result, write a polite, professional, and clear natural language answer for an enterprise dashboard user. 
            Do not mention technical terms like tuples or raw data arrays unless asked. If the result is empty, clearly state that no matching records were found.

            User Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            
            Answer:"""
        )
        
        answer_chain = answer_prompt | llm | StrOutputParser()
        final_natural_answer = answer_chain.invoke({
            "question": prompt_query,
            "query": cleaned_sql,
            "result": query_result
        })

        # Step 4: Return clean response with SQL hidden behind the collapsible developer audit view
        return f"""{final_natural_answer.strip()}

🔍 View Generated SQL Query (Developer Audit)
```sql
{cleaned_sql}
```"""

    except Exception as e:
        return f"An error occurred while running the Gemini RAG pipeline: {str(e)}"