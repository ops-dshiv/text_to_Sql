# Text-to-SQL Query Engine with LlamaIndex and OpenAI

This repository demonstrates the implementation of a Text-to-SQL query engine leveraging LlamaIndex, OpenAI's GPT-3.5, and SQLAlchemy. The pipeline takes natural language queries and converts them into executable SQL queries for a given database schema.

## Features
- Automated SQL query generation from natural language.
- Retrieval-augmented responses from database tables.
- Modular and extensible query pipeline.
- Interactive pipeline visualization using PyVis.

---

## Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Required libraries (install using `requirements.txt`).
- OpenAI API key (set it as an environment variable).

### Setting Up

1. Clone this repository:
    ```bash
    git clone <repo_url>
    cd <repo_directory>
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the OpenAI API key:
    ```bash
    export OPENAI_API_KEY=<your_openai_api_key>
    ```

4. Create the SQLite database `school_management2.db` and define the following tables:
    - Counsellors
    - Tickets
    - Enquiries
    - SchoolTours
    - Feedback
    - Payments
    - Queries
    - Communication

---

## Step-by-Step Flow

### 1. **Database Connection Setup**
The engine connects to a SQLite database.

```python
engine = create_engine("sqlite:///school_management2.db")
sql_database = SQLDatabase(engine)
```

### 2. **Table Schema Configuration**
Define table schemas for use in the query pipeline.

```python
table_schema_objs = [
    SQLTableSchema(table_name="Counsellors"),
    SQLTableSchema(table_name="Tickets"),
    SQLTableSchema(table_name="Enquiries"),
    ...
]

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
```

### 3. **Query Engine Initialization**

Initialize the LlamaIndex-based Text-to-SQL Query Engine:

```python
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=[schema.table_name for schema in table_schema_objs],
    llm=llm,
)
```

### 4. **Function Definitions**

- **`get_table_context_str`:** Prepares table schema context for the LLM.
- **`parse_response_to_sql`:** Extracts and cleans the SQL query from the LLM response.

### 5. **Query Pipeline Configuration**

Create a modular query pipeline with individual components:

```python
qp = QP(
    modules={
        "input": InputComponent(),
        "table_retriever": obj_retriever,
        "table_output_parser": table_parser_component,
        "text2sql_prompt": text2sql_prompt,
        "text2sql_llm": llm,
        ...
    },
    verbose=True,
)
```

Define chains and links to execute the flow.

### 6. **Pipeline Visualization**

Generate and display the pipeline DAG using PyVis:

```python
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(qp.dag)

with open("text2sql_dag.html", "w") as file:
    file.write(net.html)
```

### 7. **Execution Example**

Run the pipeline with a sample query:

```python
response = qp.run(
    query="Which month experiences the maximum number of school tours?"
)

sql_query = parse_response_to_sql(response)
if sql_query:
    print(f"Generated SQL Query: {sql_query}")
else:
    print("No SQL Query found in the response.")
```

---

## File Structure

- `main.py`: Main script containing the implementation.
- `requirements.txt`: Dependency list.
- `school_management2.db`: Example SQLite database.
- `text2sql_dag.html`: Pipeline visualization.

---

## Running the Project

1. Start by setting up the environment and database.
2. Run the script:
    ```bash
    python main.py
    ```
3. Open `text2sql_dag.html` to visualize the pipeline.

---

## Customization

- Add descriptions to `SQLTableSchema` objects for enhanced context.
- Modify prompt templates for specific SQL dialects or tasks.
- Extend the pipeline by adding new components to handle custom logic.

---

## Contributions

Contributions are welcome! Feel free to submit pull requests or open issues for feature requests and bug fixes.

---

## License

This project is licensed under the MIT License.
