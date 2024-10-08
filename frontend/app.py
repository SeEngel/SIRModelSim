# streamlit_app.py
import os
import streamlit as st
import requests
import time
import subprocess
import pandas as pd

# Start FastAPI service if not running
def start_fastapi_service():
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            raise requests.ConnectionError
    except requests.ConnectionError:
        subprocess.Popen(["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"])

start_fastapi_service()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["SIR System Simulation", "Noise on parameters"])

if page == "SIR System Simulation":
    # Streamlit app layout
    st.title("SIR System Simulation")

    # Input fields for SIR model parameters
    S0 = st.number_input("Initial susceptible population (S0)", value=0.99)
    I0 = st.number_input("Initial infected population (I0)", value=0.01)
    R0 = st.number_input("Initial recovered population (R0)", value=0.0)
    muh = st.number_input("Birth and death rate (muh)", value=0.01)
    eta = st.number_input("Infection coefficient (eta)", value=0.1)
    gamma = st.number_input("Recovery rate (gamma)", value=0.05)
    N = st.number_input("Total population (N)", value=100)
    k = st.number_input("Interaction rate (k)", value=0.1)

    # Input fields for time parameters
    t_end = st.number_input("End time for the simulation (t_end)", value=160.0)
    t_steps = st.number_input("Steps the time points at which to store the computed solution (t_steps)", value=300.0)

    # Button to start simulation
    if st.button("Simulate"):
        # Send parameters to backend
        params = {
            "S0": S0,
            "I0": I0,
            "R0": R0,
            "muh": muh,
            "eta": eta,
            "gamma": gamma,
            "N": N,
            "k": k,
            "t_end": t_end,
            "t_steps": t_steps
        }
        response = requests.post("http://localhost:8000/simulate", json=params)
        task_id = response.json()["task_id"]
        
        # Poll for results
        status = "processing"
        while status == "processing":
            result_response = requests.get(f"http://localhost:8000/result/{task_id}")
            result = result_response.json()
            status = result["status"]
            if status == "processing":
                st.write("Processing...")
                time.sleep(1)
        
        # Display results
        if status == "completed":
            result = result["result"]
            st.write("Simulation Completed!")
            # Create a DataFrame for better plotting
            data = pd.DataFrame({
            "t": result["t"],
            "S": result["S"],
            "I": result["I"],
            "R": result["R"]
            })
            # Plot S, I, R against time
            st.line_chart(data.set_index("t"))
            # Add legend
            st.write("""
            **Legend:**
            - **S**: Susceptible population
            - **I**: Infected population
            - **R**: Recovered population
            """)
        else:
            st.write(result["message"])

elif page == "Noise on parameters":
    # Load the experiment page
    st.title("Noise on Parameters Page")
    path_experiments = os.path.join(os.path.dirname(__file__), "pages/noise_on_parameter.py")
    exec(open(path_experiments).read())