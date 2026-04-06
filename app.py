!pip install streamlit scipy numpy matplotlib
!npm install -g localtunnel

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

st.title("Cost vs Risk Optimization Dashboard")

D = st.number_input("Total Demand", value=1000)

cost = np.array([
    st.number_input("Cost A", value=10),
    st.number_input("Cost B", value=12),
    st.number_input("Cost C", value=11)
])

risk = np.array([
    st.number_input("Risk A", value=0.2),
    st.number_input("Risk B", value=0.5),
    st.number_input("Risk C", value=0.3)
])

capacity = np.array([
    st.number_input("Capacity A", value=500),
    st.number_input("Capacity B", value=700),
    st.number_input("Capacity C", value=400)
])

lam = st.slider("Cost vs Risk Preference (λ)", 0.0, 1.0, 0.5)

cost_list = []
risk_list = []

for w in np.linspace(0,1,20):

    obj = w*cost + (1-w)*risk

    A_ub = [
        [-1,-1,-1],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ]

    b_ub = [
        -D,
        capacity[0],
        capacity[1],
        capacity[2]
    ]

    bounds = [(0,None)]*3

    res = linprog(c=obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    if res.success:
        x = res.x
        cost_val = np.dot(cost,x)
        risk_val = np.dot(risk,x)/D

        cost_list.append(cost_val)
        risk_list.append(risk_val)

obj = lam*cost + (1-lam)*risk

res = linprog(c=obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

if res.success:
    x = res.x
    C_star = np.dot(cost,x)
    R_star = np.dot(risk,x)/D

fig, ax = plt.subplots()

ax.plot(cost_list, risk_list, marker='o', label="Pareto Frontier")
ax.scatter(C_star, R_star, s=100, label="Selected")

C_vals = np.linspace(min(cost_list), max(cost_list), 100)
k = lam*(C_star/max(cost_list)) + (1-lam)*R_star

R_vals = (k - lam*(C_vals/max(cost_list))) / (1-lam + 1e-6)

ax.plot(C_vals, R_vals, linestyle='--', label="Indifference Line")

ax.set_xlabel("Cost")
ax.set_ylabel("Risk")
ax.legend()

st.pyplot(fig)

st.write("Allocation:", x)
st.write("Cost:", C_star)
st.write("Risk:", R_star)
