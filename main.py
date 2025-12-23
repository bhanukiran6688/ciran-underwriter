from typing import Any, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings
from graph.workflow import WorkflowState, build_workflow
from schemas import UnderwritingRequest, UnderwritingResponse


@asynccontextmanager
async def lifespan(_: FastAPI):
    llm = ChatGoogleGenerativeAI(model=settings.GOOGLE_MODEL_NAME)
    app.state.workflow = build_workflow(llm=llm)
    try:
        yield
    finally:
        # Place for cleanup if needed
        pass


app = FastAPI(title="CIRAN Underwriter", version="1.0", lifespan=lifespan)


@app.post("/underwrite", response_model=UnderwritingResponse)
def underwrite(request: UnderwritingRequest, http_req: Request):
    """Run the underwriting pipeline end-to-end."""
    try:
        initial_state = WorkflowState(
            request=request.model_dump(),
            hazard_scores=None,
            loss_estimates=None,
            recommendation=None
        )
        workflow = http_req.app.state.workflow  # properly typed via Request.app
        result_state: Dict[str, Any] = workflow.invoke(initial_state)

        response = UnderwritingResponse.model_validate(result_state)
        return JSONResponse(content=response.model_dump(mode="json"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Underwriting failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        app="main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False
    )
