import os
import openai
from sqlalchemy import (
    create_engine,

)
from pyvis.network import Network
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatResponse
from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.retrievers import SQLRetriever
from typing import List
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
    CustomQueryComponent,
)
from llama_index.core import VectorStoreIndex

os.environ["OPENAI_API_KEY"]=
openai.api_key = os.environ["OPENAI_API_KEY"]

llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")

# 1. Database Connection Setup
# ----------------------------
engine = create_engine("sqlite:///school_management2.db")  
sql_database = SQLDatabase(engine)


# 2. Table Schema Configuration
# -----------------------------
# Define table schema objects with optional descriptions

table_node_mapping = SQLTableNodeMapping(sql_database)


table_schema_objs = [
    SQLTableSchema(table_name="Counsellors"),
    SQLTableSchema(table_name="Tickets"),
    SQLTableSchema(table_name="Enquiries"),
    SQLTableSchema(table_name="SchoolTours"),
    SQLTableSchema(table_name="Feedback"),
    SQLTableSchema(table_name="Payments"),
    SQLTableSchema(table_name="Queries"),
    SQLTableSchema(table_name="Communication"),
]

# Build searchable index of table schemas
obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)


query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=[schema.table_name for schema in table_schema_objs],  
    llm=llm
)


# Configure SQL retriever
sql_retriever = SQLRetriever(sql_database)


# 4. Core Processing Functions
# ----------------------------
def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
    """
    Retrieve a combined schema context string for the specified SQL tables.

    Args:
        table_schema_objs (List[SQLTableSchema]): A list of SQLTableSchema objects,
            each representing a database table and optional descriptive context.

    Returns:
        str: A single string containing schema information for all tables, 
        including optional descriptions, separated by double newlines.

    Reason:
        This function prepares the schema context required by the LLM to 
        understand the structure and purpose of the database tables. 
        This context improves the accuracy of text-to-SQL query generation 
        by providing both structural and descriptive information.
    """
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)


def parse_response_to_sql(response: ChatResponse) -> str:
    """
    Extract and clean the SQL query from the LLM's response.

    Args:
        response (ChatResponse): The response object from the LLM, 
            containing the generated SQL query.

    Returns:
        str: A cleaned SQL query string extracted from the response.
            If no valid SQL query is found, returns an empty string.

    Reason:
        LLM-generated responses may contain explanations, formatting, 
        or additional details. This function isolates the actual SQL 
        query string from such responses for execution in the database.
    """
    response = response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    sql_query = response.strip().strip("```").strip()
    print(f"Generated SQL Query: {sql_query}")  
    return sql_query

# 5. Pipeline Components Setup
# ----------------------------
# Create function components for pipeline
table_parser_component = FnComponent(fn=get_table_context_str)
sql_parser_component = FnComponent(fn=parse_response_to_sql)

# Custom Prompt Templates
text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
    dialect=engine.dialect.name
)
response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
response_synthesis_prompt = PromptTemplate(
    response_synthesis_prompt_str,
)


# 6. Query Pipeline Assembly
# --------------------------
qp = QP(
    modules={
        "input": InputComponent(),
        "table_retriever": obj_retriever,
        "table_output_parser": table_parser_component,
        "text2sql_prompt": text2sql_prompt,
        "text2sql_llm": llm,
        "sql_output_parser": sql_parser_component,
        "sql_retriever": sql_retriever,
        "response_synthesis_prompt": response_synthesis_prompt,
        "response_synthesis_llm": llm,
    },
    verbose=True,
)

qp.add_chain(["input", "table_retriever", "table_output_parser"])
qp.add_link("input", "text2sql_prompt", dest_key="query_str")
qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
qp.add_chain(
    ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
)
qp.add_link(
    "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
)
qp.add_link(
    "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
)
qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

# 7. Visualization Generation
# ---------------------------
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(qp.dag)


with open("text2sql_dag.html", "w", encoding="utf-8") as file:
    file.write(net.html)

from IPython.display import display, HTML

with open("text2sql_dag.html", "r") as file:
    html_content = file.read()

display(HTML(html_content))

# 8. Execution Example
# --------------------
response = qp.run(
    query="Which month experiences the maximum number of school tours??")
print(str(response))


sql_query =  parse_response_to_sql(response)  
if sql_query:
    print(f"Generated SQL Query: {sql_query}")
else:
    print("No SQL Query found in the response.")
