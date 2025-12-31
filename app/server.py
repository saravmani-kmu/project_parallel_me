import sys
import os

# Add the project root to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from app.models import InputRequest, OutputResponse
from app.graph import graph
from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI(title="LangGraph Chat API")

@app.post("/chat", response_model=OutputResponse)
async def chat(request: InputRequest):
    try:
        inputs = {
            "messages": [HumanMessage(content=request.message)],
            "model_provider": request.model_provider
        }
        
        result = await graph.ainvoke(inputs)
        
        # Extract the last message content
        last_message = result["messages"][-1].content
        return OutputResponse(response=last_message)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
