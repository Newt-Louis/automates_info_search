import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from agent import app as agent_app, system_prompt
from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Request model
class AgentRequest(BaseModel):
    """Request model for agent invocation."""
    prompt: str

# Lỗi ở đây, hiện tại response với AgenResponse không khớp cấu trúc với nhau
# Do giới hạn gửi request 1 ngày của GEMINI nên cần debug kĩ hơn
# Response model
class AgentResponse(BaseModel):
    """Response model for agent invocation."""
    response: str

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/agent")
async def invoke_agent(request: AgentRequest):
    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        inputs = {"messages": [system_prompt,HumanMessage(content=request.prompt)]}
        # inputs = [("system",system_prompt),("human",HumanMessage(content=request.prompt))]
        output_state = agent_app.invoke(inputs)
        result = output_state["messages"][-1].content
        return result

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error invoking agent: {str(e)}")
