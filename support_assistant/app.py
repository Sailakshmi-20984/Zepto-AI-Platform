from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pydantic import BaseModel
from graph import app_graph, QueryResponse

app = FastAPI(title="Zepto Support Assistant API")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

class QueryRequest(BaseModel):
    query: str

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        initial_state = {"query": request.query, "intent": "", "sources": [], "response": None}
        final_state = app_graph.invoke(initial_state)
        return final_state["response"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)