Usage



Energy system operation



First of all, prepare your OpenAI key.

Note, the LLM optimization process costs dozens of dollars if you run hundreds of iterations for the problem whose output dimension is high (>10).

LLM solver is a little bit complex across totally different tasks, which need customization by human.

Take LLM_optimizer_DR.py as an example:

1. Define decision variables from line 18.

2. Define the objective and constraints from line 32

3. Define your prompt in line 79 and 106

4. Define the initialization values for decision variables in line 119 and 164

5. Define the setup parameters from 137 and 301

6. Use the defined constraint in line 244

7. If you want to change constraints, refer to line 312

   



LLM code generator can be achieved by online version, such as ChatGPT-4o webpage:

1. Describe your problem in human language, including decision role(s), decision variable(s), objective(s), and constraint(s). **Then ask the LLM to formulate the system model in mathematical language.**
2. Ask LLM to generate Gurobi codes based on the formulated model using a target coding language, such as Python.
3. If you meet bugs, input the bug to the LLM and obtain new codes.
4. Repeat 3, then you can get a well-run code.