import snowflake.connector
import os
import json
from datetime import datetime
from typing import List, Dict, Any

class SnowflakeManager:
    def __init__(self):
        """Initializes connection to Snowflake."""
        self.user = os.environ.get("SNOWFLAKE_USER")
        self.password = os.environ.get("SNOWFLAKE_PASSWORD")
        self.account = os.environ.get("SNOWFLAKE_ACCOUNT")
        self.warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE")
        self.database = os.environ.get("SNOWFLAKE_DATABASE")
        self.schema = os.environ.get("SNOWFLAKE_SCHEMA")
        
        self.conn: Any = None
        self.is_connected = False
        
        self.connect()
        
    def connect(self):
        """Attempts to connect to Snowflake and creates the table if it doesn't exist."""
        if not all([self.user, self.password, self.account, self.database, self.schema]):
            print("Warning: Missing Snowflake credentials in environment variables. Chat history will not be saved.")
            return
            
        try:
            self.conn = snowflake.connector.connect(
                user=self.user,
                password=self.password,
                account=self.account,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema
            )
            self.is_connected = True
            print("Successfully connected to Snowflake.")
            self._create_table_if_not_exists()
        except Exception as e:
            print(f"Error connecting to Snowflake: {e}")
            
    def _create_table_if_not_exists(self):
        """Creates the chat_history table if required."""
        if not self.conn:
            return
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.schema}.chat_history (
            id STRING DEFAULT UUID_STRING(),
            session_id STRING,
            document_name STRING,
            query STRING,
            answer STRING,
            retrieved_chunks VARIANT,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query)
        except Exception as e:
            print(f"Error creating table: {e}")

    def save_chat(self, session_id: str, document_name: str, query: str, answer: str, retrieved_chunks: List[Dict[str, Any]]):
        """Saves a conversation turn to Snowflake."""
        if not self.is_connected or not self.conn:
            return
        
        # Convert chunks list to JSON string for VARIANT column
        chunks_json = json.dumps(retrieved_chunks)
        # Handle potential quotes in text
        safe_query = query.replace("'", "''")
        safe_answer = answer.replace("'", "''")
        safe_doc = document_name.replace("'", "''")
        
        query_sql = f"""
        INSERT INTO {self.database}.{self.schema}.chat_history 
        (session_id, document_name, query, answer, retrieved_chunks)
        SELECT 
            '{session_id}', 
            '{safe_doc}', 
            '{safe_query}', 
            '{safe_answer}', 
            PARSE_JSON('{chunks_json}')
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query_sql)
        except Exception as e:
            print(f"Error saving chat: {e}")

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves chat history for a given session."""
        if not self.is_connected or not self.conn:
            return []
            
        query = f"""
        SELECT query, answer, retrieved_chunks, created_at 
        FROM {self.database}.{self.schema}.chat_history
        WHERE session_id = '{session_id}'
        ORDER BY created_at ASC
        """
        
        history = []
        try:
            with self.conn.cursor(snowflake.connector.DictCursor) as cur:
                for row in cur.execute(query):
                    # For DictCursor, keys are in uppercase
                    history.append({
                        "query": row["QUERY"],
                        "answer": row["ANSWER"],
                        "retrieved_chunks": json.loads(row["RETRIEVED_CHUNKS"]) if isinstance(row["RETRIEVED_CHUNKS"], str) else row["RETRIEVED_CHUNKS"],
                        "timestamp": str(row["CREATED_AT"])
                    })
        except Exception as e:
            print(f"Error retrieving history: {e}")
            
        return history
