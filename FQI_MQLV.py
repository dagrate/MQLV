import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import bspline
import bspline.splinelab as splinelab
import sys
sys.path.append("..")


isbs = 0          # 1 BS model / 0 mean reverting model
isvanilla = 0    # 1 vanilla option / 0 digital option
iscall = 1       # 1 call payoff / 0 put payoff
# Parameters for MC simulation of stock prices
S0 = 1          # initial stock price
K = 1           # option strike
r = 0.00        # risk-free rate
M = 0.5         # maturity
T = 5           # number of time steps
N_MC = 50000    # number of paths
isexp = 7       # exp choice when N_MC = 50,000 -> 4 5 6 7
nb_plots = 10   # number of paths to plot
risk_lambda = 0.00001  # risk aversion parameter => 0 pure hedge

print("bs_model =>", isbs)
if isbs == 1:
    mu = r  # 0.05    # drift
    sigma = 0.15      # volatility
else:
    a = 0.3          # speed reversion
    b = 1             # long term mean
    sigma = 0.15      # volatility

Klist = np.arange(98, 104, 2) / 100  # [0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03]
resultsRL = []
resultsBS = []
for K in Klist:
    delta_t = M / T                # time interval
    gamma = np.exp(- r * delta_t)  # discount factor

    # ### Monte Carlo Paths Generation
    # S <- stock price = time serie values
    # X <- state variable
    # RN <- random numbers
    if False:
        np.random.seed(42)
        S = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
        S.loc[:, 0] = S0
        RN = pd.DataFrame(np.random.randn(N_MC, T), index=range(1, N_MC + 1),
                          columns=range(1, T + 1))
    
        if isbs == 0:
            X = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
            X.loc[:, 0] = S0
    
        for t in range(1, T + 1):
            if isbs == 1:
                S.loc[:, t] = S.loc[:, t - 1] * np.exp((mu - 0.5 * sigma**2) * (
                    delta_t) + sigma * np.sqrt(delta_t) * RN.loc[:, t])
            else:
                S.loc[:, t] = S.loc[:, t - 1] + a * (b - S.loc[:, t - 1]) * delta_t + \
                    sigma * np.sqrt(delta_t) * RN.loc[:, t]
                X.loc[:, t] = X.loc[:, t - 1] + sigma * np.sqrt(delta_t) * RN.loc[:, t]
    
        if isbs == 1:
            X = - (mu - 0.5 * sigma**2) * np.arange(T + 1) * delta_t + np.log(S)
        stop
    else:
        if N_MC == 20000:
            S = np.genfromtxt('./data/paths1.csv', 
                              delimiter=',')
            X = np.genfromtxt('./data/statevariable1.csv', 
                              delimiter=',')
            S = pd.DataFrame(S, index=range(1, N_MC + 1))
            X = pd.DataFrame(X, index=range(1, N_MC + 1))
        elif N_MC == 30000:
            S = np.genfromtxt('./data/paths2.csv', 
                              delimiter=',')
            X = np.genfromtxt('./data/statevariable2.csv', 
                              delimiter=',')
            S = pd.DataFrame(S, index=range(1, N_MC + 1))
            X = pd.DataFrame(X, index=range(1, N_MC + 1))
        elif N_MC == 40000:
            S = np.genfromtxt('./data/paths3.csv', 
                              delimiter=',')
            X = np.genfromtxt('./data/statevariable3.csv', 
                              delimiter=',')
            S = pd.DataFrame(S, index=range(1, N_MC + 1))
            X = pd.DataFrame(X, index=range(1, N_MC + 1))
        elif N_MC == 50000:
            S = np.genfromtxt('./data/paths'+str(isexp)+'.csv', 
                              delimiter=',')
            X = np.genfromtxt('./data/statevariable'+str(isexp)+'.csv', 
                              delimiter=',')
            S = pd.DataFrame(S, index=range(1, N_MC + 1))
            X = pd.DataFrame(X, index=range(1, N_MC + 1))

    # we write the Monte Carlo paths
    if False:
        np.savetxt("/data/paths7.csv", 
                   S, delimiter=",")
        np.savetxt("/data/statevariable7.csv", 
                   X, delimiter=",")

    delta_S = S.loc[:, 1:T].values - np.exp(r * delta_t) * S.loc[:, 0:T - 1]
    delta_S_hat = delta_S.apply(lambda x: x - np.mean(x), axis=0)

    #stop

    # plot paths
    step_size = N_MC // nb_plots
    idx_plot = np.arange(step_size, N_MC, step_size)
    plt.plot(S.T.iloc[:, idx_plot])
    plt.xlabel('Time Steps')
    plt.title('Stock Price Sample Paths')
    plt.show()

    plt.plot(X.T.iloc[:, idx_plot])
    plt.xlabel('Time Steps')
    plt.ylabel('State Variable')
    plt.show()

    # STOP 

    def terminal_payoff(ST, K):
        # ST   final stock price
        # K    strike
        if isvanilla == 1:
            if iscall == 1:
                payoff = max(ST - K, 0)
            else:
                payoff = max(K - ST, 0)
        else:
            if iscall == 1:
                payoff = np.int32(ST >= K)
            else:
                payoff = np.int32(ST <= K)
        return payoff


    # ### Spline basis functions definition
    X_min = np.min(np.min(X))
    X_max = np.max(np.max(X))
    print('X.shape = ', X.shape)
    print('X_min, X_max = ', X_min, X_max)

    p = 4              # 3 <- cubic, 4 <- B-spline
    ncolloc = 12
    tau = np.linspace(X_min, X_max, ncolloc)
    k = splinelab.aptknt(tau, p)
    basis = bspline.Bspline(k, p)
    f = plt.figure()
    print('Number of points k = ', len(k))
    basis.plot()

    # ### Make data matrices with feature values
    # "Features" here are the values of basis functions at data points
    # The outputs are 3D arrays of dimensions num_tSteps x num_MC x num_basis
    num_t_steps = T + 1
    num_basis = ncolloc
    data_mat_t = np.zeros((num_t_steps, N_MC, num_basis))
    print('num_basis = ', num_basis)
    print('dim data_mat_t = ', data_mat_t.shape)

    # fill it, expand function in finite dimensional space
    # in neural network the basis is the neural network itself
    for i in np.arange(num_t_steps):
        x = X.values[:, i]
        data_mat_t[i, :, :] = np.array([basis(el) for el in x])
    # save these data matrices for future re-use
    # np.save('data_mat_m=r_A_%d' % N_MC, data_mat_t)
    print(data_mat_t.shape)  # shape num_steps x N_MC x num_basis
    print(len(k))


    # ## Dynamic Programming solution for QLBS
    # ## Part 1: Implement functions to compute optimal hedges


    def function_A_vec(t, delta_S_hat, data_mat, reg_param):
        """
        function_A_vec - compute the matrix A_{nm}
        Eq. (52) in QLBS Q-Learner in the Black-Scholes-Merton article

        Arguments:
        t <- time index, a scalar, an index into time axis of data_mat
        delta_S_hat <- pandas.DataFrame of dimension N_MC x T
        data_mat <- pandas.DataFrame of dimension T x N_MC x num_basis
        reg_param <- a scalar, regularization parameter

        Return:
        np.array <- matrix A_{nm} of dimension num_basis x num_basis
        """
        X_mat = data_mat[t, :, :]
        num_basis_funcs = X_mat.shape[1]
        this_dS = delta_S_hat.iloc[:, t]
        hat_dS2 = (this_dS ** 2).values.reshape(-1, 1)
        A_mat = np.dot(X_mat.T, X_mat * hat_dS2) + reg_param * np.eye(
            num_basis_funcs)
        return A_mat


    def function_B_vec(t, Pi_hat, delta_S_hat=delta_S_hat, S=S,
                       data_mat=data_mat_t, gamma=gamma, risk_lambda=risk_lambda):
        """
        function_B_vec - compute vector B_{n}
        Eq. (52) QLBS Q-Learner in the Black-Scholes-Merton article

        Arguments:
        t <- time index, a scalar, an index into time axis of delta_S_hat
        Pi_hat <- pandas.DataFrame of dimension N_MC x T of portfolio values
        delta_S_hat <- pandas.DataFrame of dimension N_MC x T
        S <- pandas.DataFrame of simulated stock prices
        data_mat <- pandas.DataFrame of dimension T x N_MC x num_basis
        gamma <- one time-step discount factor $exp(-r \delta t)$
        risk_lambda <- risk aversion coefficient, a small positive number

        Return:
        B_vec <- np.array() of dimension num_basis x 1
        """
        coef = 0.  # 0 <- pure risk hedge || 1.0/(2 * gamma * risk_lambda)
        # tmp = Pi_hat.iloc[:, t + 1] * delta_S_hat.loc[:, t] + coef * (
        #     np.exp((mu - r) * delta_t)) * S.loc[:, t]
        tmp = Pi_hat.iloc[:, t + 1] * delta_S_hat.loc[:, t] + coef * (
            np.exp(r * delta_t)) * S.loc[:, t]
        X_mat = data_mat[t, :, :]
        B_vec = np.dot(X_mat.T, tmp)
        return B_vec


    # ## Compute optimal hedge and portfolio value
    Pi = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    Pi.iloc[:, T] = S.loc[:, T].apply(lambda x: terminal_payoff(x, K))

    Pi_hat = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    Pi_hat.iloc[:, -1] = Pi.iloc[:, -1] - np.mean(Pi.iloc[:, -1])

    # optimal hedge
    a = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    a.iloc[:, -1] = 0

    reg_param = 1e-3
    for t in range(T - 1, -1, -1):
        A_mat = function_A_vec(t, delta_S_hat, data_mat_t, reg_param)
        B_vec = function_B_vec(t, Pi_hat, delta_S_hat, S, data_mat_t)
        phi = np.dot(np.linalg.inv(A_mat), B_vec)

        a.loc[:, t] = np.dot(data_mat_t[t, :, :], phi)
        Pi.loc[:, t] = gamma * (Pi.loc[:, t + 1] - a.loc[:, t] * delta_S.loc[:, t])
        Pi_hat.loc[:, t] = Pi.loc[:, t] - np.mean(Pi.loc[:, t])

    a = a.astype('float')
    Pi = Pi.astype('float')
    Pi_hat = Pi_hat.astype('float')

    # plot paths
    plt.plot(a.T.iloc[:, idx_plot])
    plt.xlabel('Time Steps')
    plt.title('Optimal Hedge')
    plt.show()

    plt.plot(Pi.T.iloc[:, idx_plot])
    plt.xlabel('Time Steps')
    plt.title('Portfolio Value')
    plt.show()


    # ## Part 2: Compute the optimal Q-function with the DP approach


    def function_C_vec(t, data_mat, reg_param):
        """
        function_C_vec - calculate C_{nm} matrix
        Eq. (56) in QLBS Q-Learner in the Black-Scholes-Merton article

        Arguments:
        t <- time index, a scalar, an index into time axis of data_mat
        data_mat <- pandas.DataFrame of basis functions T x N_MC x num_basis
        reg_param <- regularization parameter, a scalar

        Return:
        C_mat <- np.array of dimension num_basis x num_basis
        """
        X_mat = data_mat[t, :, :]
        num_basis_funcs = X_mat.shape[1]
        C_mat = np.dot(X_mat.T, X_mat) + reg_param * np.eye(num_basis_funcs)
        return C_mat


    def function_D_vec(t, Q, R, data_mat, gamma=gamma):
        """
        function_D_vec - calculate D_{nm} vector
        Eq. (56) in QLBS Q-Learner in the Black-Scholes-Merton article

        Arguments:
        t <- time index, a scalar, an index into time axis of data_mat
        Q <- pandas.DataFrame of Q-function values of dimension N_MC x T
        R <- pandas.DataFrame of rewards of dimension N_MC x T
        data_mat <- pandas.DataFrame of basis functions T x N_MC x num_basis
        gamma <- one time-step discount factor $exp(-r \delta t)$

        Return:
        D_vec - np.array of dimension num_basis x 1
        """
        X_mat = data_mat[t, :, :]
        tmp = R.loc[:, t] + gamma * Q.loc[:, t + 1]
        D_vec = np.dot(X_mat.T, tmp)
        return D_vec


    # Compare the QLBS price to European put price given by Black-Sholes formula.
    def bs_put(t, S0=S0, K=K, r=r, sigma=sigma, T=M):
        d1 = (np.log(S0 / K) + (r +
                                0.5 * sigma**2) * (T - t)) / sigma / np.sqrt(T - t)
        d2 = (np.log(S0 / K) + (r -
                                0.5 * sigma**2) * (T - t)) / sigma / np.sqrt(T - t)
        price = K * np.exp(-r * (T - t)) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        return price


    def bs_call(t, S0=S0, K=K, r=r, sigma=sigma, T=M):
        d1 = (np.log(S0 / K) + (r +
                                0.5 * sigma**2) * (T - t)) / sigma / np.sqrt(T - t)
        d2 = (np.log(S0 / K) + (r -
                                0.5 * sigma**2) * (T - t)) / sigma / np.sqrt(T - t)
        price = S0 * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
        return price


    # ## Hedging and Pricing with Reinforcement Learning
    # Batch-mode off-policy model-free Q-Learning by Fitted Q-Iteration.
    # ## Part 3: Make off-policy data
    # - on-policy data - contains an optimal action and the corresponding reward
    # - off-policy data - contains random action and the corresponding reward
    eta = 0.5
    reg_param = 1e-3
    np.random.seed(42)

    a_op = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    a_op.iloc[:, -1] = 0

    Pi_op = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    Pi_op.iloc[:, -1] = S.iloc[:, -1].apply(lambda x: terminal_payoff(x, K))

    Pi_op_hat = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    Pi_op_hat.iloc[:, -1] = Pi_op.iloc[:, -1] - np.mean(Pi_op.iloc[:, -1])

    R_op = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    R_op.iloc[:, -1] = - risk_lambda * np.var(Pi_op.iloc[:, -1])

    # Backward loop
    for t in range(T - 1, -1, -1):
        # 1. Compute the optimal policy, and write the result to a_op
        a_op.iloc[:, t] = a.loc[:, t]

        # 2. Now disturb these values by a random noise
        # uniform r.v. in the interval [1−η,1+η]
        noise = np.random.uniform(1 - eta, 1 + eta, N_MC)
        a_op.iloc[:, t] = a_op.iloc[:, t] * noise

        # 3. Compute portfolio values corresponding to observed actions
        Pi_op.iloc[:, t] = gamma * (Pi_op.iloc[:, t + 1] - a_op.iloc[:, t] *
                                    delta_S.iloc[:, t])
        Pi_op_hat.iloc[:, t] = Pi_op.iloc[:, t] - np.mean(Pi_op.iloc[:, t])

        # 4. Compute rewards corrresponding to observed actions
        R_op.iloc[:, t] = gamma * a_op.loc[:, t] * delta_S.iloc[:, t] - (
            risk_lambda * np.var(Pi_op.iloc[:, t]))

    print('done with backward loop!')

    # ## Override on-policy data with off-policy data
    a = a_op.copy()      # distrubed actions
    Pi = Pi_op.copy()    # disturbed portfolio values
    Pi_hat = Pi_op_hat.copy()
    R = R_op.copy()

    num_MC = a.shape[0]
    num_TS = a.shape[1]
    a_1_1 = a.values.reshape((1, num_MC, num_TS))
    a_1_2 = 0.5 * a_1_1**2
    ones_3d = np.ones((1, num_MC, num_TS))

    A_stack = np.vstack((ones_3d, a_1_1, a_1_2))
    print(A_stack.shape)

    data_mat_swap_idx = np.swapaxes(data_mat_t, 0, 2)
    print(data_mat_swap_idx.shape)

    # expand dimensions of matrices to multiply element-wise
    A_2 = np.expand_dims(A_stack, axis=1)
    data_mat_swap_idx = np.expand_dims(data_mat_swap_idx, axis=0)

    Psi_mat = np.multiply(A_2, data_mat_swap_idx)

    # now concatenate columns along the first dimension
    # Psi_mat = Psi_mat.reshape(-1, a.shape[0], a.shape[1], order='F')
    Psi_mat = Psi_mat.reshape(-1, N_MC, T + 1, order='F')
    print(Psi_mat.shape)

    Psi_1_aux = np.expand_dims(Psi_mat, axis=1)
    Psi_2_aux = np.expand_dims(Psi_mat, axis=0)
    print(Psi_1_aux.shape, Psi_2_aux.shape)

    S_t_mat = np.sum(np.multiply(Psi_1_aux, Psi_2_aux), axis=2)
    print(S_t_mat.shape)

    def function_S_vec(t, S_t_mat, reg_param):
        """
        function_S_vec - calculate S_{nm} matrix
        Eq. (75) in QLBS Q-Learner in the Black-Scholes-Merton article
        num_Qbasis = 3 x num_basis (1, a_t, 0.5 a_t^2)

        Arguments:
        t <- time index, a scalar, an index into time axis of S_t_mat
        S_t_mat <- pandas.DataFrame of dimension num_Qbasis x num_Qbasis x T
        reg_param <- regularization parameter, a scalar

        Return:
        S_mat_reg <- num_Qbasis x num_Qbasis
        """
        X_mat = S_t_mat[:, :, t]
        num_basis_funcs = X_mat.shape[1]
        S_mat_reg = X_mat + reg_param * np.eye(num_basis_funcs)
        return S_mat_reg

    def dropout(arr, num):
        """
        DROPOUT: fix randomly a number of elements of an array to 0
        to compensate the overestimation of Q-values.
        arr -> array to modify
        num -> number of elements to fix equal to 0
        """
        temp = np.asarray(arr)   # Cast to numpy array
        shape = temp.shape       # Store original shape
        temp = temp.flatten()    # Flatten to 1D
        inds = np.random.choice(temp.size, size=num,
                                replace=False)  # Get random indices
        temp[inds] = np.zeros(shape=num)  # Fill with something
        temp = temp.reshape(shape)  # Restore original shape
        return temp

    def function_M_vec(t, Q_star, R, Psi_mat_t, gamma=gamma):
        """
        function_M_vec - calculate M_{nm} vector
        Eq. (75) in QLBS Q-Learner in the Black-Scholes-Merton article
        num_Qbasis = 3 x num_basis (1, a_t, 0.5 a_t^2)

        Arguments:
        t <- time index, a scalar, an index into time axis of S_t_mat
        Q_star <- pandas.DataFrame of Q-function values of dimension N_MC x T
        R <- pandas.DataFrame of rewards of dimension N_MC x T
        Psi_mat_t <- pandas.DataFrame of dimension num_Qbasis x N_MC
        gamma <- one time-step discount factor $exp(-r \delta t)$

        Return:
        M_t - np.array of dimension num_Qbasis x 1
        """
        dropout_int = 1
        tmp = R.loc[:, t] + gamma * Q_star.loc[:, t + 1]

        if dropout_int == 1:
            # Q-Learning over-estimates the Q-values.
            # We dropout some Q-values to compensate the over-estimation.
            tmp = dropout(tmp, np.int32(0.00001 * len(tmp.ravel())))

        elif dropout_int == 2:
            # we implement double Q-learning updates for faster convergence.
            print("to be implemented")
            # p = np.random.random()
            # if (p < .5):
            # tmp = R.loc[:, t] + gamma * Q2_star.loc[:, t + 1]
            # Q1[prev_s][prev_a] = Q1[prev_s][prev_a] + alpha * (
            # r + GAMMA * Q2[s][a] - Q1[prev_s][prev_a])
            # else:
            # tmp = R.loc[:,t] + gamma * Q1_star.loc[:,t+1]
            # Q2[prev_s][prev_a] = Q2[prev_s][prev_a] + alpha * (
            # r + GAMMA * Q1[s][a] - Q2[prev_s][prev_a])

        M_t = np.dot(Psi_mat_t, tmp)
        return M_t

    # In[] Fitted Q Iteration (FQI)
    # implied Q-function by input data (using the first form in Eq.(68))
    Q_RL = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    Q_RL.iloc[:, -1] = - Pi.iloc[:, -1] - risk_lambda * np.var(Pi.iloc[:, -1])

    a_opt = np.zeros((N_MC, T + 1))
    a_star = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    a_star.iloc[:, -1] = 0

    Q_star = pd.DataFrame([], index=range(1, N_MC + 1), columns=range(T + 1))
    Q_star.iloc[:, -1] = - Pi.iloc[:, -1] - risk_lambda * np.var(Pi.iloc[:, -1])

    max_Q_star = np.zeros((N_MC, T + 1))
    max_Q_star[:, -1] = Q_RL.iloc[:, -1].values
    Q_star_bis = np.zeros((N_MC, T + 1))
    Q_star_bis[:, -1] = Q_RL.iloc[:, -1].values
    Q_star_1 = np.zeros((N_MC, T + 1))
    Q_star_1[:, -1] = Q_RL.iloc[:, -1].values
    Q_star_2 = np.zeros((N_MC, T + 1))
    Q_star_2[:, -1] = Q_RL.iloc[:, -1].values
    alpha = 0.8

    num_basis = data_mat_t.shape[2]

    reg_param = 1e-3
    hyper_param = 1e-1

    def maxQA(data_mat_t_t, phi, U_mat):
        U_W_0 = U_mat[0, :]
        U_W_1 = U_mat[1, :]
        U_W_2 = U_mat[2, :]

        next_a = np.dot(data_mat_t_t, phi)
        maxq = U_W_0 + next_a * U_W_1 + 0.5 * (next_a**2) * U_W_2
        return next_a, maxq

    # Backward loop
    for t in range(T - 1, -1, -1):
        S_mat_reg = function_S_vec(t, S_t_mat, reg_param)
        M_t = function_M_vec(t, Q_star, R, Psi_mat[:, :, t], gamma)
        W_t = np.dot(np.linalg.inv(S_mat_reg), M_t)
        W_mat = W_t.reshape((3, num_basis), order='F')

        Phi_mat = data_mat_t[t, :, :].T

        U_mat = np.dot(W_mat, Phi_mat)

        # compute vectors U_W^0,U_W^1,U_W^2 as rows of matrix U_mat
        U_W_0 = U_mat[0, :]
        U_W_1 = U_mat[1, :]
        U_W_2 = U_mat[2, :]

        # we use hedges computed as in DP approach to avoid the errors of
        # function approximation to back-propagate.
        A_mat = function_A_vec(t, delta_S_hat, data_mat_t, reg_param)
        B_vec = function_B_vec(t, Pi_hat, delta_S_hat, S, data_mat_t)
        phi = np.dot(np.linalg.inv(A_mat), B_vec)

        a_opt[:, t] = np.dot(data_mat_t[t, :, :], phi)
        a_star.loc[:, t] = a_opt[:, t]

        max_Q_star[:, t] = U_W_0 + a_opt[:, t] * U_W_1 + (
            0.5 * (a_opt[:, t]**2) * U_W_2)

        Q_star.loc[:, t] = max_Q_star[:, t]

        Q_star_bis[:, t] = R.loc[:, t] + gamma * Q_star_bis[:, t + 1]

        p = np.random.random()
        next_a, maxq = maxQA(data_mat_t[t, :, :], phi, U_mat)
        if (p >= .0):
            if t == T - 1:
                Q_star_1[:, t] = R.loc[:, t] + gamma * Q_star_1[:, t + 1]
            else:
                nxt_a = np.argmin(np.asarray(a)[:, t + 1])
                Q_star_1[:, t] = R.loc[:, t] + gamma * Q_star_1[nxt_a, t + 1]
        else:
            Q_star_2[:, t] = Q_star_2[:, t] + alpha * (
                R.loc[:, t] + gamma * Q_star_1[:, t + 1] - Q_star_2[:, t])

        Psi_t = Psi_mat[:, :, t].T
        Q_RL.loc[:, t] = np.dot(Psi_t, W_t)

        # trim outliers for Q_RL
        up_percentile_Q_RL = 95
        low_percentile_Q_RL = 5
        low_perc_Q_RL, up_perc_Q_RL = np.percentile(Q_RL.loc[:, t], (
            [low_percentile_Q_RL, up_percentile_Q_RL]))

        # trim outliers in values of max_Q_star:
        flag_lower = Q_RL.loc[:, t].values < low_perc_Q_RL
        flag_upper = Q_RL.loc[:, t].values > up_perc_Q_RL
        Q_RL.loc[flag_lower, t] = low_perc_Q_RL
        Q_RL.loc[flag_upper, t] = up_perc_Q_RL


    # plot simulations
    f, axarr = plt.subplots(3, 1)
    f.subplots_adjust(hspace=.5)
    f.set_figheight(8.0)
    f.set_figwidth(8.0)

    step_size = N_MC // nb_plots
    idx_plot = np.arange(step_size, N_MC, step_size)
    axarr[0].plot(a_star.T.iloc[:, idx_plot])
    axarr[0].set_xlabel('Time Steps')
    axarr[0].set_title(r'Optimal action $a_t^{\star}$')

    axarr[1].plot(Q_RL.T.iloc[:, idx_plot])
    axarr[1].set_xlabel('Time Steps')
    axarr[1].set_title(r'Q-function $Q_t^{\star} (X_t, a_t)$')

    axarr[2].plot(Q_star.T.iloc[:, idx_plot])
    axarr[2].set_xlabel('Time Steps')
    axarr[2].set_title(r'Optimal Q-function $Q_t^{\star} (X_t, a_t^{\star})$')

    plt.savefig('QLBS_FQI_off_policy_summary_ATM_eta_%d.png' % (100 * eta),
                dpi=600)
    plt.show()


    num_path = 120
    # a from the DP method and a_star from the RL method are now identical
    plt.plot(a.T.iloc[:, num_path], label="DP Action")
    plt.plot(a_star.T.iloc[:, num_path], label="RL Action")
    plt.legend()
    plt.xlabel('Time Steps')
    plt.title('Optimal Action Comparison Between DP and RL')
    plt.show()

    # ## Summary of the RL-based pricing with QLBS
    def dig_put_BS(t, S0=S0, K=K, r=r, sigma=sigma, T=M):
        epsilon = 1 / 10000
        return -((bs_put(0, S0, K, r, sigma, M) -
              bs_put(0, S0, K + epsilon, r, sigma, M)) / epsilon)

    def dig_call_BS(t, S0=S0, K=K, r=r, sigma=sigma, T=M):
        epsilon = 1 / 10000
        return (bs_call(0, S0, K, r, sigma, M) -
                bs_call(0, S0, K + epsilon, r, sigma, M)) / epsilon

    dig_Qstar = np.mean(-Q_star.iloc[:, 0])

    print('---------------------------------')
    print('        RL Option Pricing         ')
    print('---------------------------------\n')
    print('%-25s' % ('Initial Stock Price:'), S0)
    print('%-25s' % ('Drift of Stock:'), r)
    print('%-25s' % ('Volatility of Stock:'), sigma)
    print('%-25s' % ('Risk-free Rate:'), r)
    print('%-25s' % ('Risk aversion parameter :'), risk_lambda)
    print('%-25s' % ('Strike:'), K)
    print('%-25s' % ('Maturity:'), M)
    print('%-25s' % ('Time Steps Number:'), T)
    print('%-25s' % ('Monte Carlo Paths:'), N_MC)
    print('%-26s %.4f' % ('\nRL Price (%):', dig_Qstar * 100))
    print('%-26s %.4f' % ('\nBS Call Price (%):', bs_call(0) * 100))
    print('%-25s %.4f' % ('BS Put Price (%):', bs_put(0) * 100))
    print('%-26s %.4f' % ('\nBS Dig. Call Price (%):', dig_call_BS(0) * 100))
    print('%-25s %.4f' % ('BS Dig. Put Price (%):', dig_put_BS(0) * 100))
    # print('%-25s %.4f' % ("Q_star_bis:", np.mean(-Q_star_bis[:, 0])))
    # print('%-25s %.4f' % (
    # "Double QL:", np.mean(-Q_star_1[:, 0] - Q_star_2[:, 0])))
    print('\n')

    resultsRL.append(dig_Qstar * 100)
    resultsBS.append(dig_call_BS(0) * 100)

# In[]: compare the different approach
# # plot one path
# plt.plot(C_QLBS.T.iloc[:,[200]])
# plt.xlabel('Time Steps')
# plt.title('QLBS RL Option Price')
# plt.show()

# part5 = str(C_QLBS.iloc[0, 0])
# print("Q_star")
# print(Q_star.head())
# print("Q_RL")
# print(Q_RL.head())
# print("Q_star_bis")
# print(pd.DataFrame(Q_star_bis).head())
# print("Q_DQN")
# print(pd.DataFrame(Q_star_1).head())
# print(pd.DataFrame(Q_star_2).head())
