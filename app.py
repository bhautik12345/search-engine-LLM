import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import streamlit as st

#####code of search-engine

load_dotenv()

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

search = DuckDuckGoSearchRun(name='Search')

st.title("🔎 LangChain - Chat with search")
"""
we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""
st.sidebar.title('Settings')
groq_api_key = st.sidebar.text_input('put your Groq API key',type='password')

if 'messages' not in st.session_state:
    st.session_state['messages']=[
        {'role':'assistant','content':'Hi,I am chatbot who can search the web How can i help you today?'}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder='what is Generative AI?'):
    st.session_state.messages.append({'role':'user','content':prompt})
    st.chat_message('user').write(prompt)

    llm = ChatGroq(model='Llama3-8b-8192',api_key=groq_api_key,streaming=True)
    tools=[search,wiki,arxiv]

    search_agent = initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True) # Chat_zero_shot..menas chat history yad rakhine output appse

    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)
