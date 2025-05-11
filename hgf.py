import arviz as az
import jax
import jax.numpy as jnp

import pymc as pm
from pyhgf import load_data
from pyhgf.distribution import HGFDistribution
from pyhgf.model import HGF
from pyhgf.response import first_level_binary_surprise
import pandas as pd
import numpy as np 
import pyhgf






from pyhgf.response import binary_softmax, first_level_binary_surprise

def response_function(hgf, response_function_inputs, response_function_parameters=None):
    """A simple response function returning the binary surprise."""

  
    beliefs = hgf.node_trajectories[0]["expected_mean"]

    # get the decision from the inputs to the response function
    return jnp.sum(jnp.where(response_function_inputs, -jnp.log(beliefs), -jnp.log(1.0 - beliefs)))

# Initialize an empty list to collect results
results_list = []

# Function to compute response function (given in original code)

for subj in subjects:
    
    # Read data for current subject
    df = pd.read_csv('/Users/kristina/Documents/comp_neuroscience/asd_hgf/{0}.csv'.format(subj))
    u = np.array(df['choice'], dtype=np.float32)
    responses = np.array(df['outcome'], dtype=np.float32)
    subject_id = subj  # Extract subject ID from file name
    
    # Initialize HGF with dummy tonic volatility, calculate actual volatility later
    agent = HGF(
        n_levels=2,
        model_type="binary",
        initial_mean={"1": 0., "2": 0.5},
        initial_precision={"1": 0.5, "2": 1.0},
        tonic_volatility={"2": -4.0},  # Temporary value, will adjust based on Bayesian model below
    ).input_data(input_data=u)
    
    # Define HGF distribution and response function
    hgf_logp_op = HGFDistribution(
        n_levels=2,
        model_type="binary",
        input_data=jnp.array(u[jnp.newaxis, :], dtype=jnp.float32),
        response_function=response_function,
        response_function_inputs=responses[jnp.newaxis, :]
    )
    
    # Use PyMC3 to sample the tonic volatility for Level 2
    with pm.Model() as sigmoid_hgf:
        tonic_volatility_2 = pm.Normal("tonic_volatility_2", -4.0, 4.0)  # Prior over tonic volatility
        pm.Potential("hgf_loglike", hgf_logp_op(tonic_volatility_2=tonic_volatility_2))

    with sigmoid_hgf:
        #tonic_volatility_2 = pm.Normal("tonic_volatility_2", -4.0, 4.0)
        start = {'tonic_volatility_2': -2.0}  # Specify a more reasonable start
        sigmoid_hgf_idata = pm.sample(chains=4, cores=1, init='adapt_diag', start=start)

    # Extract the summary of tonic volatility
    v = az.summary(sigmoid_hgf_idata, var_names=["tonic_volatility_2"])
    tonic_volatility_value = v['mean']['tonic_volatility_2']
    print(tonic_volatility_value)
    # Re-run HGF with the computed tonic volatility value
    two_levels_hgf = HGF(
        n_levels=2,
        model_type="binary",
        initial_mean={"1": 0., "2": 0.1},
        initial_precision={"1": 0.5, "2": 2.0},
        tonic_volatility={"2": tonic_volatility_value},  # Use sampled volatility
    ).input_data(input_data=u)
    hgf_df = two_levels_hgf.to_pandas()
    
    two_levels_hgf.plot_trajectories()
    trajectories = two_levels_hgf.node_trajectories
    two_levels_hgf.plot_trajectories(show_total_surprise=True);
   

    # Add these to your result DataFrame
    result = hgf_df[['x_1_precision', 'total_surprise', 
                     'x_0_surprise', 'x_1_surprise', 'x_0_expected_mean', 'x_1_expected_mean']].copy()
    
 
  
    result['subject_id'] = subject_id
    result['trial_number'] = np.arange(1, len(result) + 1)
    result['round'] = df['run']
    result['response_time'] = df['rt']
    result['response_time']=df['response_time']
    
    # Append the result to the results_list
    results_list.append(result)

# Combine all subjects' results into a single DataFrame
final_df = pd.concat(results_list, ignore_index=True)

# Display or save the final dataframe
print(final_df.head())
final_df.to_csv('/Users/kristina/Documents/comp_neuroscience/asd_hgf_new.csv', index=False)
