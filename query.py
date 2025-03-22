import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# Connection string
conn_string = "postgresql://postgres:TvROSyEFxjKowwovGUSBuLHumfmhzuck@yamanote.proxy.rlwy.net:42901/railway"

class PostgresQueryExecutor:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
    
    def execute_query(self, query, query_type="select", params=None):
        """
        Execute a SQL query against PostgreSQL.
        
        Parameters:
        - query: SQL query string
        - query_type: One of "select", "insert", "update", "delete", "create", "drop", or "other"
        - params: Optional parameters for parameterized queries
        
        Returns:
        - For "select": DataFrame with results
        - For other query types: Number of affected rows or success message
        """
        conn = None
        cursor = None
        
        try:
            # Connect to the database
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Execute the query
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Process the result based on query type
            if query_type.lower() == "select":
                # For SELECT queries, return a DataFrame
                columns = [desc[0] for desc in cursor.description]
                result = cursor.fetchall()
                
                # Convert to DataFrame
                df = pd.DataFrame(result, columns=columns)
                return df
            else:
                # For non-SELECT queries, commit changes and return affected rows
                conn.commit()
                if cursor.rowcount >= 0:
                    return f"Query executed successfully. Rows affected: {cursor.rowcount}"
                return "Query executed successfully."
                
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error executing query: {str(e)}")
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def select(self, query, params=None):
        """Execute a SELECT query and return results as DataFrame"""
        return self.execute_query(query, "select", params)
    
    def insert(self, query, params=None):
        """Execute an INSERT query"""
        return self.execute_query(query, "insert", params)
    
    def update(self, query, params=None):
        """Execute an UPDATE query"""
        return self.execute_query(query, "update", params)
    
    def delete(self, query, params=None):
        """Execute a DELETE query"""
        return self.execute_query(query, "delete", params)
    
    def execute_script(self, script):
        """Execute a multi-statement SQL script"""
        conn = None
        
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Execute the entire script
            cursor.execute(script)
            
            # Commit changes
            conn.commit()
            return "SQL script executed successfully"
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error executing script: {str(e)}")
            
        finally:
            if conn:
                conn.close()
    
    def list_tables(self):
        """List all tables in the database"""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        return self.select(query)
    
    def describe_table(self, table_name):
        """Get column information for a specific table"""
        query = """
        SELECT column_name, data_type, character_maximum_length, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """
        return self.select(query, (table_name,))

# Example usage
if __name__ == "__main__":
    # Create a query executor
    db = PostgresQueryExecutor(conn_string)
    
    # result_df = db.select("SELECT * FROM results LIMIT 10")
    # print("SELECT results:")
    # print(result_df)
    
    # Example INSERT query
    # insert_result = db.insert("INSERT INTO my_table (column1, column2) VALUES (%s, %s)", ("value1", "value2"))
    # print(insert_result)
    
    # Example DELETE all records
    delete_result = db.delete("DELETE FROM results")
    print(delete_result)
    
    # List all tables
    # tables = db.list_tables()
    # print("Tables in the database:")
    # print(tables)
    
    # Describe a table
    # table_info = db.describe_table("results")
    # print("Table structure:")
    # print(table_info)
    
    # Execute multi-statement script
    # script_result = db.execute_script("""
    #     BEGIN;
    #     CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY, name TEXT);
    #     INSERT INTO test_table (name) VALUES ('test1'), ('test2');
    #     COMMIT;
    # """)
    # print(script_result)