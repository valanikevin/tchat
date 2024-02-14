from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv


load_dotenv()

chat = ChatOpenAI()


prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
)

while True:
    content = input(">> ")
    result = chain({"content": content})
    print(result["text"])
