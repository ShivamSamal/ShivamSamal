import os
import pymysql
from urllib.parse import quote_plus
from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.chains.sql_database.prompt import SQL_PROMPTS
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
import re
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
os.environ["GROQ_API_KEY"] = "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"
db_user = "root"
db_password = "Shivam@2005"
db_host = "localhost"
db_name = "fashion_db"
db_port=3306
db_password = quote_plus(db_password)
# prompt = """
# You are an expert SQL query generator. Your task is to carefully interpret natural language questions, extract the intent, and convert them into accurate SQL queries based on the provided database schema.

# ### Database Schema:
# 1. **Designer Table**: Contains details about designers.
#    - `Designer.name`: The designer's name.
#    - `Designer.country`: The designer's country.
#    - `Designer.created_at`: The date when the designer created the cloth.
#    - `Designer.id`: Each designer has a unique identifier .
   
# 2. **Dress Table**: Contains information about dresses.
#    - `Dress.name`: The brand name of the dress.
#    - `Dress.style`: The style of the dress.
#    - `Dress.designer_id`: Foreign key linking to `Designer.id`.
#    - `Dress.price`: The price of the dress.
   
# 3. **Glasses Table**: Contains details about glasses.
#    - `Glasses.name`: The brand name of the glasses.
#    - `Glasses.type`: The type of glasses.
#    - `Glasses.designer_id`: Foreign key linking to `Designer.id`.
#    - `Glasses.price`: The price of the glasses.

# ### Key Relationships:
# - Both `Dress.designer_id` and `Glasses.designer_id` reference `Designer.id`.
# - Use JOIN statements to combine data from multiple tables when necessary.

# ### General Instructions:
# - **Interpretation:** Fully understand the user's intent before generating the SQL query.
# - **Accuracy:** Ensure the query matches the schema, including table names, field names, and relationships.
# - **Optimization:** Write queries that are efficient and free of syntax errors.
# - **Conditions:** If the query includes filters such as specific dates, countries, or price ranges, include the appropriate `WHERE` clauses.
# - **Sorting and Limiting:** If the question requires sorting or limiting results, use `ORDER BY` and `LIMIT` appropriately.
# - **Flexibility:** Adapt the SQL query based on the specific details provided in the question. Even if the question varies in structure or phrasing, ensure the SQL output directly addresses it.

# ### Response Format:
# - Only return the SQL query as your output. Do not include additional commentary or explanation.
# - Ensure the final query is syntactically correct and ready to execute.

# ### Example:
# **Question:** "List the top 5 most expensive dresses along with their designer names."
  
# **SQL Output:**
# ```sql
# SELECT Dress.name, Designer.name, Dress.price
# FROM Dress
# JOIN Designer ON Dress.designer_id = Designer.designer_id
# ORDER BY Dress.price DESC
# LIMIT 5;


#     """
llm=ChatGroq(temperature=0,model_name="mixtral-8x7b-32768")

#  # db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=1,include_tables=['customers','orders'],custom_table_info={'customers':"customer"})
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# connection = pymysql.connect(
#         host=db_host,
#         user=db_user,
#         database=db_name,
#         password="Shivam@2005",
#         charset='utf8mb4',
#         cursorclass=pymysql.cursors.DictCursor
#     )
# context = db.get_context()

# generate_query= create_sql_query_chain(llm,db)
# # prompt_with_context = generate_query.get_prompts()[0].partial(table_info=context["table_info"])
# # prompt = ChatPromptTemplate([
# #     ("system", "You are a SQL expert. Respond only with the corrected SQL query—no explanations or extra text."),
# #     ("user", "Here is an SQL query with an error: {query}. Fix the error and respond only with the corrected SQL query.")
# # ])
# memory = ConversationBufferMemory(memory_key="history", return_messages=True)





# # Define the prompt for generating SQL queries based on conversation history.
# Answer_prompt = ChatPromptTemplate.from_messages([
#     ('system', 'You are an SQL expert. Generate SQL queries from natural language questions for a fashion database.'),
#     ('system', 'Keep conversation context in mind. If the question contains pronouns like "that" or "these", assume '),
#     ('user', 'Conversation history: {history}\nQuestion: {question}\nGenerate an SQL query according to conversation history and question:')
# ])

# # Define a prompt for converting SQL query results into a natural language answer.
# Answer_nl = ChatPromptTemplate.from_messages([
#     ('system', 'You are best at natural language.'),
#     ('user', 'Answer: {answer}\nQuestion: {question}\nGenerate a natural language answer. Just give me a direct answer only, no extra text.')
# ])

# while True:
#     # Get the user question from input
#     question = input("Question -> ")

#     # (Optional) If you want to use a separate chain to generate an initial query,
#     # you can do so here. For now, we are using the conversation chain with memory.
#     # e.g., initial_query = generate_query.invoke({"question": question})

#     # Add the new question to conversation memory.
#     memory.chat_memory.add_user_message(question)

#     # Create the chain for SQL query generation using the conversation memory.
#     # Note: We use only Answer_prompt here (no concatenation with any other prompt).
#     chain = LLMChain(llm=llm, prompt=Answer_prompt+prompt, memory=memory)

#     # Generate the SQL query from the conversation context.
#     sql_query = chain.run({"question": question})
#     print("Generated SQL Query:")
#     print(sql_query)

#     # Add the generated SQL query as an assistant message to the memory.
#     memory.chat_memory.add_ai_message(sql_query)

#     # If the query is wrapped in a markdown code block, extract the SQL.
#     result_query = sql_query
#     regex = r"```sql([^`]*)```"
#     match = re.search(regex, result_query, re.DOTALL)
#     if match:
#         result_query = match.group(1).strip()

#     # Execute the SQL query using your database connection.
#     with connection.cursor() as cursor:
#         try:
#             # Remove newline characters from the query if necessary.
#             cursor.execute(result_query.replace('\n', ' '))
#             answer_data = cursor.fetchall()
#         except Exception as e:
#             print("SQL execution error:", e)
#             continue  # Skip to the next iteration if SQL execution fails.

#     # Now create a new chain to generate a natural language answer from the SQL result.
#     new_chain = LLMChain(llm=llm, prompt=Answer_nl)
#     # Convert the answer data to a string (or format it as needed).
#     answer_str = str(answer_data)

#     inputs = {
#         'question': question,
#         'answer': answer_str
#     }

#     # Generate the natural language answer.
#     nl_answer = new_chain.run(inputs)
#     print("Final Answer:")
#     print(nl_answer)
template="Give me error free code"

generate_query = create_sql_query_chain(llm, db)

query = generate_query.invoke({"question": "what is price of  dress?"})

execute_query = QuerySQLDataBaseTool(db=db)
answer = execute_query.invoke(query)
# print(answer)
answer_prompt = PromptTemplate.from_template(
     """Given the following user question, corresponding SQL query, and SQL result, Give the user-friendly answer that means just give my question's answer in a human friendlu .

 Question: {question}
 SQL Query: {query}
 SQL Result: {answer}
 Answer: """
 )

rephrase_answer = answer_prompt | llm | StrOutputParser()
input_k={
    'question':"How many dresses are there and which are they?",
    'query':query,
    'answer':answer
}


answer=rephrase_answer.invoke(input_k)
print(answer)
examples= [
     {
         "input": " How many dress are there?",
         "query": "Select count(*) from dress;"
     },
     {
         "input": "What is most expensive dress cost",
         "query": "SELECT MAX(price) FROM dress;"
     },
    
 ]
example_prompt = ChatPromptTemplate.from_messages(
     [
         ("human", "{input}\nSQLQuery:"),
         ("ai", "{query}"),
     ]
 )
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    # input_variables=["input","top_k"],
    input_variables=["input"],
)

vectorstore=Chroma()
vectorstore.delete_collection()

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    vectorstore,
    k=2,
    input_keys=["input"],
)
example_selector.select_examples({"input": "how many glasses are there?"})
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    input_variables=["input","top_k"],
)
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specificed.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

generate_query = create_sql_query_chain(llm, db,final_prompt)
chain = (
RunnablePassthrough.assign(query=generate_query).assign(
    result=itemgetter("query") | execute_query
)
| rephrase_answer
)
answer=chain.invoke({"question": "which one is expensive dress?"})
print(answer)
