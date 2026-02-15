import json
import os
from pydoc_data.topics import topics
from typing import Annotated, Literal, TypedDict
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

system_prompt = SystemMessage(content="Bạn là chuyên gia tìm kiếm thông tin. Hãy dùng công cụ tavily để trả lời.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0,
    max_retries=2,
)

tavily_client = TavilySearch(max_results=3,topic="general")

@tool
def web_search_tool(query:str):
    """
        Công cụ tìm kiếm thông tin trên internet.
        Sử dụng công cụ này khi cần tìm thông tin thời gian thực, tin tức, hoặc sự kiện.
        """
    try:
        result = tavily_client.invoke(query)
        if isinstance(result, (list,dict)):
            return json.dumps(result, ensure_ascii=False)

        return str(result)
    except Exception as e:
        return f"Lỗi khi tìm kiếm: {str(e)}"

# Gắn công cụ vào não của Gemini
tools = [web_search_tool]
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState):
    try:
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    except:
        raise


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    try:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"
    except:
        raise


workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)
workflow.add_edge("tools", "agent")

app = workflow.compile()


# if __name__ == "__main__":
#     # Câu hỏi test
#     user_query = "Tìm kiếm thông tin về iPhone 16 mới nhất, nó đã ra mắt chưa?"
#
#     # System Prompt giúp Gemini định hình vai trò
#     system_prompt = SystemMessage(content="Bạn là chuyên gia tìm kiếm thông tin. Hãy dùng công cụ tavily để trả lời.")
#
#     inputs = {"messages": [system_prompt, HumanMessage(content=user_query)]}
#
#     # In ra luồng chạy
#     for output in app.stream(inputs):
#         for key, value in output.items():
#             print(f"Node '{key}': Đã chạy xong.")
#             # Uncomment dòng dưới nếu muốn xem nội dung chi tiết từng bước
#             # print(value)
#
#     print("\n--- KẾT QUẢ CUỐI CÙNG ---")
#     # Lấy tin nhắn cuối cùng từ AI
#     final_response = output['agent']['messages'][0].content
#     print(final_response)