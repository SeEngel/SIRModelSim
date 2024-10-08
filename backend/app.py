import numpy as np
from scipy.integrate import odeint
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import uvicorn
from pydantic import Field
import uuid
import asyncio

app = FastAPI()
tasks: Dict[str, Dict] = {}
COMPLETED_TASKS = 0
REQUESTED_TASKS = 0

class Parameters(BaseModel):
    S0: float = Field(..., example=0.99, description="Initial susceptible population")
    I0: float = Field(..., example=0.01, description="Initial infected population")
    R0: float = Field(..., example=0.0, description="Initial recovered population")
    muh: float = Field(..., example=0.01, description="Birth and death rate")
    eta: float = Field(..., example=0.1, description="Infection coefficient")
    gamma: float = Field(..., example=0.05, description="Recovery rate")
    N: int = Field(..., example=100, description="Total population")
    k: float = Field(..., example=0.1, description="Interaction rate")
    t_end: float = Field(..., example=160.0, description="End time for the simulation")
    t_steps: float = Field(..., example=300.0, description="Steps the time points at which to store the computed solution")

def simulate_sir(params: Parameters):

    def sir_model(y, t, muh, eta, gamma, N, k):
        S, I, R = y
        dSdt = muh * (N - S) - eta * k * I * S
        dIdt = eta * k * I * S - (gamma + muh) * I
        dRdt = gamma * I - muh * R
        return [dSdt, dIdt, dRdt]

    y0 = [params.S0, params.I0, params.R0]
    t = np.linspace(0, params.t_end, int(params.t_steps))
    solution = odeint(sir_model, y0, t, args=(params.muh, params.eta, params.gamma, params.N, params.k))
    S, I, R = solution.T
    return {"t": t.tolist(), "S": S.tolist(), "I": I.tolist(), "R": R.tolist()}

@app.post("/simulate", summary="Simulate SIR Model", description="Simulates the behavior of an SIR model given the parameters.\n\nReturns:\n- task_id: Hash value to retrieve the result")
async def simulate(params: Parameters):
    global REQUESTED_TASKS  # Add this line
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": None}
    REQUESTED_TASKS += 1
    
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, simulate_sir, params)
    
    async def task_done_callback(fut):
        result = await fut
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result
        
    asyncio.ensure_future(task_done_callback(future))
    return {"task_id": task_id}

@app.get("/result/{task_id}", summary="Get Simulation Result", description="Retrieve the result of the simulation using the task_id")
async def get_result(task_id: str):
    global COMPLETED_TASKS  # Add this line
    if task_id not in tasks:
        return {"status": "error", "message": "Task ID does not exist or has already been retrieved"}
    
    task = tasks[task_id]
    if task["status"] == "processing":
        return {"status": "processing", "message": "Still calculating solution"}
    else:
        result = task["result"]
        del tasks[task_id]
        COMPLETED_TASKS += 1
        return {"status": "completed", "result": result}

@app.get("/health", summary="Health Check", description="Check the health status of the application")
async def health_check():
    global REQUESTED_TASKS  # Add this line
    global COMPLETED_TASKS
    return {"status": "healthy", 
            "completed_tasks": COMPLETED_TASKS,
            "requested_tasks": REQUESTED_TASKS}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)