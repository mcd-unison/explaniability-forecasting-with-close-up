import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from math import comb
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

class elements:

    def __init__(self, model, X_train_sample, component_to_close_up=None, initial_bias=None, split_func=None, joint_func=None):

        self.model = model  
        self.sample = X_train_sample
        self.close_up_component = component_to_close_up
        self.initial_bias = initial_bias
        self.decomposition =  split_func
        self.join = joint_func

    # Transform list of matrix into tensor
    def list_input_to_tensor_input(self,X_mask_list):
        X_mask = []
        for j in range(0,len(X_mask_list[0])):
            X_mask.append(np.array([X_mask_list[i][j] for i in range(0,len(X_mask_list))]))
        return X_mask

    # Estimate $f_\theta|_{comp}(z)$
    def f_composition(self,X_split,z): 
        X_mask_list = []
        numeric_index_list = [num for num in list(X_split.keys()) if isinstance(num, int)]
        for U in  self.sample: 
            U_split = self.decomposition(U)
            for i in numeric_index_list:
                if z[i] == 1:
                    U_split.update({i: X_split[i]})
            U_mask = self.join(U_split)
            X_mask_list.append(U_mask)
        X_mask = self.list_input_to_tensor_input(X_mask_list)
        y_mask = self.model.predict(X_mask)
        return  np.mean(y_mask, axis=0)

    # Generate ${0,1}^n$
    def binary_index(self,n):
        binary_index_list = [[]]
        for i in range(0, n):
            list_update = []
            for l in binary_index_list:
                list_update = list_update + [l + [0],l + [1]]  
            binary_index_list = list_update
        return binary_index_list
    
    # Estimate weight $\pi_n(z)$
    def pi_n(self,z):
        n = len(z)
        size = z.count(1)
        return (n - 1) / (comb(n, size) * size * (n-size))

    # Short notation for $F(z):=f_\theta|X_{comp}(z)$
    def F(self,X_split,z):
        F = self.f_composition(X_split,z)
        return  F.reshape(F.shape[0],)

    def explainability_model_exact(self,X_observation):
        X_split = self.decomposition(X_observation)
        # Cardinality of the interpretable components
        numeric_index_list = [num for num in list(X_split.keys()) if isinstance(num, int)]
        n = len(numeric_index_list) 
        # Sample size
        bin_index = self.binary_index(n)
        # Initial bias
        zero = bin_index[0]
        phi_0 = self.F(X_split,zero)
        # Number of future data
        tau = phi_0.shape[0]
        # Creation of Shapley values list
        explaniability_matrix = []
        for i in numeric_index_list:
            Shap_i = np.zeros(tau,) 
            bin_index_minus_i = [list for list in bin_index if list[i] == 0]
            for z in bin_index_minus_i:
                z_ones = [x for x in z if x == 1]
                weight_i = len(z_ones) * math.factorial(n-len(z_ones)-1)/ math.factorial(n) 
                z_i = z[:i] + [1] + z[i+1:]
                Shap_i = Shap_i + weight_i * (self.F(X_split,z_i) - self.F(X_split,z))
            explaniability_matrix.append(Shap_i)
        return phi_0, np.array(explaniability_matrix)

    #Explainability initial model 
    def explainability_model(self,X_observation):
        X_split = self.decomposition(X_observation)
        # Cardinality of the interpretable components
        numeric_index_list = [num for num in list(X_split.keys()) if isinstance(num, int)]
        n = len(numeric_index_list)
        # Sample size
        bin_index = self.binary_index(n)
        N = len(bin_index)
        # Estimation of y_target, X_domain and sample weight 
        zero = bin_index[0]
        one = bin_index[N-1]
        # Sample index
        sample_index = bin_index[1:N-1]
        #
        X = []  
        for z in sample_index:
            X.append(np.array(z)-np.ones(n)) 
        X_domain = np.array(X)
        #
        y = []  
        for z in sample_index:
            y.append(np.array(self.F(X_split,z)-self.F(X_split,one)))  
        y_target = np.array(y)
        # 
        Pi = []  
        for z in sample_index:
            Pi.append(self.pi_n(z))  
        Pi_weight = np.array(Pi)
        
        phi_0 = self.F(X_split,zero)
        f_x = self.F(X_split,one)
        expicability_model = LinearRegression().fit(X_domain, y_target, sample_weight=Pi_weight)
        Phi_fitted = expicability_model.predict(np.identity(n)).T
        error = Phi_fitted.dot(np.ones(n)) - f_x + phi_0
        explaniability_matrix = Phi_fitted.T - error/n
        
        return phi_0, explaniability_matrix


    #Explainability initial model 
    def close_up_explainability_model(self,X_observation,exact=None):
        # Calculate auxiliar explaniability model
        if exact is None:
            close_up_auxiliar_model = self.explainability_model(X_observation)
        else:
            close_up_auxiliar_model = self.explainability_model_exact(X_observation)
        # Cardinality of the interpretable components set
        X_split = self.decomposition(X_observation)
        numeric_index_list = [num for num in list(X_split.keys()) if isinstance(num, int)]
        n_close_up = len(numeric_index_list)
        n_time = close_up_auxiliar_model[0].shape[0]
        one = list(np.ones(2**n_close_up-1))
        # Calculate close up explaniability model
        phi_0 = self.initial_bias
        sum_all = close_up_auxiliar_model[1].T.dot(np.ones(n_close_up))
        error = (self.close_up_component - sum_all) / n_close_up 
        explainability_matrix = []
        for k in range(n_close_up):
             phi_close_up = close_up_auxiliar_model[1][k] + error 
             explainability_matrix.append(phi_close_up)
        return phi_0, np.array(explainability_matrix), error


def local_plot(explainability_model,component_names,base_vector=None):
    fig = go.Figure()
    data = explainability_model[1].T
    tau = data.shape[0]
    df_explain_matrix = pd.DataFrame(data=data, columns=component_names)
    x_names = component_names
    if base_vector is None:
        start = explainability_model[0]
    else:
        start = explainability_model[0] + base_vector
    fig = go.Figure()
    fig.update_layout(
        title='Local explainability',
        showlegend=False,
        autosize=False,  
        width=1000,      
        height=600,      
        yaxis=dict(title='Explainability values'),
        xaxis=dict(title='Interpretable components'),
        font=dict(size=11,color='gray'),
        plot_bgcolor='white',
        font_family='Helvetica'
    )
    fig.add_annotation(
        xref='paper',
        yref='paper',
        x=0,  
        y=1.08,  
        text='positive values  ↑',
        showarrow=False,
        font=dict(size=10, color='#afab50'),
        bgcolor='white'
    )
    fig.add_annotation(
        xref='paper',
        yref='paper',
        x=0,  
        y=1.05,  
        text='negative values  ↓',
        showarrow=False,
        font=dict(size=10, color='#5054af'),
        bgcolor='white'
    )
    for i in range(0,tau):
        fig.add_trace(go.Waterfall(
            orientation = 'v',
            name= 'h ' + str(i),
            x = x_names,
            base = start[i],
            y = list(df_explain_matrix.loc[i].values),
            connector={'visible': False},
            decreasing = {'marker':{'color':'#5054af'}},
            increasing = {'marker':{'color':'#afab50'}},
        ))
    return fig
