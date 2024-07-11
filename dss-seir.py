import streamlit as st
import numpy as np
from scipy.integrate import odeint, simpson
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Define the SEIR model differential equations
def seir_model(y, t, r, K, beta, gamma_E, gamma_I, mu, zeta, rho, dE, cE, dI, cI, dR, cR):
    S, E, I, R = y
    dSdt = r * (K - (S + E + I + R)) - beta * S * I - mu * S
    dEdt = beta * S * I - gamma_E * E - mu * E - rho * dE * cE * E
    dIdt = gamma_E * E - gamma_I * I - mu * I - rho * dI * cI * I
    dRdt = gamma_I * I - (mu + zeta) * R - rho * dR * cR * R
    return [dSdt, dEdt, dIdt, dRdt]

# Function to compute the income
def compute_income(S, t, rho, K, m=0.05, cost=1, start_day=200):
    start_idx = np.searchsorted(t, start_day)  # Find the index corresponding to the start day
    integral_S = simpson(S[start_idx:], x=t[start_idx:])  # Integrate from start_day to T
    income = (m * integral_S) - (rho * cost * K)
    return income
    
# Function to compute the SEIR model for a given rho and return income
def income_for_rho(rho, r, K, beta, gamma_E, gamma_I, mu, zeta, dE, cE, dI, cI, dR, cR, y0, t):
    ret = odeint(seir_model, y0, t, args=(r, K, beta, gamma_E, gamma_I, mu, zeta, rho, dE, cE, dI, cI, dR, cR))
    S, _, _, _ = ret.T
    return -compute_income(S, t, rho, K) 
    
# Streamlit app
def main():
    # st.title("SEIR Plant Disease")
    
    r = 0.08
    beta = 0.0005
    gamma_E = 0.05 
    gamma_I = 0.01
    mu = 0.01
    zeta = 0.02
    K = 1000
    
    # Input fields for initial conditions
    st.sidebar.header("Initial Conditions")
    S0 = st.sidebar.number_input("Initial Susceptible Population ($S_0$)", min_value=0, value=990, max_value=K)
    E0 = st.sidebar.number_input("Initial Exposed Population ($E_0$)", min_value=0, value=10, max_value=K)
    I0 = st.sidebar.number_input("Initial Infected Population ($I_0$)", min_value=0, value=0, max_value=K)
    R0 = st.sidebar.number_input("Initial Recovered Population ($R_0$)", min_value=0, value=0, max_value=K)
    
    # Control parameters 
    st.sidebar.header("Control Parameter")
    rho = st.sidebar.slider('Roguing rate (ρ)', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    
    # Input fields for strategic setup
    st.sidebar.header("Strategic Setup")
    dE = st.sidebar.slider('Detection probability of E', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    cE = st.sidebar.slider('Compliance to rogue E', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    dI = st.sidebar.slider('Detection probability of I', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    cI = st.sidebar.slider('Compliance to rogue I', min_value=0.0, max_value=1.0, value=0.9, step=0.1)
    dR = st.sidebar.slider('Detection probability of R', min_value=0.0, max_value=1.0, value=0.9, step=0.1)
    cR = st.sidebar.slider('Compliance to rogue R', min_value=0.0, max_value=1.0, value=0.9, step=0.1)
    
    # Time points (in days)
    T = 365
    t = np.linspace(0, T, T)

    # Initial conditions vector
    y0 = [S0, E0, I0, R0]

    # Integrate the SEIR equations over the time grid, t
    ret = odeint(seir_model, y0, t, args=(r, K, beta, gamma_E, gamma_I, mu, zeta, rho, dE, cE, dI, cI, dR, cR))
    S, E, I, R = ret.T

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(t, S, 'b', label='Susceptible')
    ax.plot(t, E, 'y', label='Exposed')
    ax.plot(t, I, 'r', label='Infected')
    ax.plot(t, R, 'g', label='Removed')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('Observable: Disease progression')
    ax.legend()
    ax.grid()

    st.pyplot(fig)
    col1, col2, _, col3, col4 = st.columns([10, 12, 3, 7, 12])
    with col1:
        st.markdown("### <u>Observable:</u> ", unsafe_allow_html=True)
    with col2:
        # Compute the income
        income = compute_income(S, t, rho, K)
        # Display the income result
        st.markdown(f"#### Income= ${income:.2f}")
    with col3:
        st.markdown("### <u>Output:</u> ", unsafe_allow_html=True)
    with col4:
        # Optimize rho to maximize income
        result = minimize_scalar(income_for_rho, bounds=(0, 1), method='bounded', args=(r, K, beta, gamma_E, gamma_I, mu, zeta, dE, cE, dI, cI, dR, cR, y0, t))
        optimal_rho = result.x
        # Display the optimal rho
        st.markdown(f"#### Optimal roguing rate ρ = {optimal_rho:.2f}")
        st.markdown(f"Meaning survey and rogue each {1/optimal_rho:.2f} days")
    
    # Add LaTeX description of the SEIR model
    st.markdown(r"""
    ## SEIR Model Equations

    The SEIR model is governed by the following set of differential equations:

    $$
    \begin{align*}
    \frac{dS}{dt} &= r\big(K -(S+E+I+R) \big) -\beta S I - \mu S\\
    \frac{dE}{dt} &= \beta S I - \gamma_E E - \mu E - d_E c_E \rho E \\
    \frac{dI}{dt} &= \gamma_E E - \gamma_I I - \mu I - d_I c_I \rho I\\
    \frac{dR}{dt} &= \gamma_I I - (\mu + \zeta) R - d_R c_R \rho R
    \end{align*}
    $$

    where:
    - $S(t)$ is the number of susceptible individuals at time $t$
    - $E(t)$ is the number of exposed (but not yet infectious) individuals at time $t$
    - $I(t)$ is the number of infectious individuals at time $t$
    - $R(t)$ is the number of removed individuals at time $t$
    - $K$ is the number of plants in the field (field density)
    - $r$ is the rate at which clean seed are replanted
    - $\beta$ is the infection rate
    - $\mu$ is the plant natural mortality
    - $\gamma_E$ is the disease progression rate from latent to infectious
    - $\gamma_I$ is the disease progression rate from infectious to post-infectious 
    - $d_E$, $d_I$ and $d_R$ are the probability that a grower detects the disease at stages $E$, $I$ and $R$ respectively
    - $c_E$, $c_I$ and $c_R$ are the levels of compliance of a grower to rogue detected plants at stages $E$, $I$ and $R$ respectively
    - $\rho$ is the roguing rate
    
    ### Income Calculation

    The income is calculated based on the following formula:

    $$
    \text{Income} = \left( m \times \int_{t_{start}}^T S(t) \, dt \right) - (\rho \times \text{cost} \times K)
    $$

    where:
    - $ m $ is a plant to income conversion parameter.
    - $t_{\text{fruit}}$ is the emergence date of the fruit.
    - $ \text{cost} $ is the cost of roguing per unit of plant.

    This formula accounts for the revenue generated from the susceptible population and the cost associated with roguing efforts.
    
    ### Parameter values
   
    Plant and intrinsic disease parameters are:
    - $r = 0.08$
    - $K = 1000$
    - $\beta = 5 \times 10^{-3}$
    - $\gamma_E = 0.05$
    - $\gamma_I = 0.01$
    - $\mu = 0.01$
    - $\zeta = 0.02$
    
    Economic parameters are:
    - $m = 0.05 $
    - $cost = $ \$2 
    """)

if __name__ == "__main__":
    main()
