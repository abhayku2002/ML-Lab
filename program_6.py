import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the structure of the network
model = BayesianNetwork([('Rain', 'WetGrass'), ('Sprinkler', 'WetGrass'), ('Rain', 'Sprinkler')])

# Step 2: Define the CPDs
cpd_rain = TabularCPD(variable='Rain', variable_card=2, values=[[0.7], [0.3]])
cpd_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                           values=[[0.5, 0.9], [0.5, 0.1]],
                           evidence=['Rain'], evidence_card=[2])
cpd_wet_grass = TabularCPD(variable='WetGrass', variable_card=2,
                           values=[[0.99, 0.9, 0.8, 0.0], 
                                   [0.01, 0.1, 0.2, 1.0]],
                           evidence=['Sprinkler', 'Rain'], evidence_card=[2, 2])

# Add the CPDs to the model
model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wet_grass)

# Step 3: Check if the model is valid
assert model.check_model()

# Step 4: Perform inference
infer = VariableElimination(model)

# Query the probability of Wet Grass being True given that Rain is True
prob_wet_grass_given_rain = infer.query(variables=['WetGrass'], evidence={'Rain': 1})
print("P(WetGrass | Rain):")
print(prob_wet_grass_given_rain)

# Query the probability of Rain being True given that Wet Grass is True
prob_rain_given_wet_grass = infer.query(variables=['Rain'], evidence={'WetGrass': 1})
print("P(Rain | WetGrass):")
print(prob_rain_given_wet_grass)
