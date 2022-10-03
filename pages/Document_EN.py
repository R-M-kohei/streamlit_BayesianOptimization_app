import streamlit as st

st.title("How to use")
st.subheader("Introduction")
st.write("Bayesian optimisation is one of the black box function optimisation "
         "methods often used in the field of machine learning for hyper-parameter tuning. "
         "Bayesian optimisation is also of interest in materials development, "
         "where it can be used for fast optimisation of composition designs and process conditions. "
         "This application is a no-code, Bayesian optimisation tool that suggests the next candidate "
         "experiment. It can be used even when multiple requirements need to be met,"
         " such as density and tensile strength or permeability. Bayesian optimisation "
         "➡ actual experiment ➡ add to dataset ➡ Bayesian optimisation... "
         "This cycle can be repeated for efficient experimentation."
         )

st.subheader("1. Dataset")
st.write("Upload a dataset (experiment results or calculation results) "
         "that has been obtained in advance. Upload a csv file containing "
         "explanatory and objective variables.")
st.write("Examples）")
st.write("Explanatory variables：experimental conditions (mix ratio of material A, mix ratio of material B, "
         "temperature conditions, pressure ... etc.)")
st.write("Objective variables：Properties()")

st.subheader("2. Setting of X")
st.write("Set the explanatory variables. Select the column name of the explanatory variable."
         "After selection, set the upper and lower limits for each explanatory variable.")

st.subheader("3. Setting of Y")
st.write("Setting of Y Sets the objective variable. Select the column name of the objective variable. "
         "After selection, choose from the following options for each objective variable to suit "
         "your requirements.")
st.write("Maximization : Search for the maximum value of the objective variable.")
st.write("Minimization : searches for the minimum value of the objective variable.")
st.write("Range : searches for conditions such that the objective variable falls within a certain range. "
         "Specify the range when using this setting. ")

st.subheader("4. Acquisition function")
st.write("Sets the acquisition function. When there is only one objective variable, "
         "Probability of improvement (PI) and Expected improvement (EI) can be used. "
         "PI is more likely to be trapped in a minimum value than EI, but currently "
         "only PI is available because it is easier to handle when there are multiple "
         "objective functions.")

st.subheader("5. Get the next condition")
st.write("The next experimental condition is obtained by Bayesian optimisation: "
         "click on Perform and after a few moments it will be output as 'Next condition")


