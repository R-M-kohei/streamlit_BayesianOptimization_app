import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  WhiteKernel, RBF, ConstantKernel
import numpy as np
import streamlit as st

st.title("Bayesian Optimization")
st.subheader("[How to use](https://r-m-kohei-streamlit-bayesianoptimization-a-streamlit-app-lvwbbc.streamlitapp.com/Document)")

st.write("1. Dataset")
uploaded_data_csv = st.file_uploader("Upload dataset",
                                     type="csv",
                                     accept_multiple_files=False)

if uploaded_data_csv:
    df = pd.read_csv(uploaded_data_csv)
    df
    if len(df)<2:
        st.header("Not enough data. At least 2 instances are required.")
    else:
        df_columns = df.columns
        col1, col2, col3 = st.columns(3)

        # range of X
        col1.write("2. Setting of X")
        X_columns = col1.multiselect("X (Select one or more)", df_columns)
        if X_columns:
            X_min_list = []
            X_max_list = []
            for i, X_column in enumerate(X_columns):
                X_min_list += [col1.number_input("Min of {}".format(X_column))]
                X_max_list += [col1.number_input("Max of {}".format(X_column), value=1)]
                if X_min_list[i]>=X_max_list[i]:
                    col1.write("'Min of {}' must be lower than 'Max of {}'.".format(X_column, X_column))


        col2.write("3. Setting of Y")
        # maximization, minimization or specific range
        Y_columns = col2.multiselect("Y (Select one or more)", set(df_columns)^set(X_columns))
        if Y_columns:
            Y_settings = []
            global Y_min_list
            Y_min_list= []
            Y_max_list = []
            for i, Y_column in enumerate(Y_columns):
                Y_settings += [col2.radio("Setting of {}".format(Y_column),
                                          ("Maximization", "Minimization", "Range"))]
                if Y_settings[i] =="Range":
                    Y_min_list += [col2.number_input("Min of {}".format(Y_column))]
                    Y_max_list += [col2.number_input("Max of {}".format(Y_column), value=1)]
                    if Y_min_list[i]>=Y_max_list[i]:
                        col2.write("'Min of {}' must be lower than 'Max of {}.'".format(Y_column, Y_column))
                else:
                    Y_min_list += [0]
                    Y_max_list += [0]

        # acquisition function
        col3.write("4. Acquisition function")
        if len(Y_columns) == 1:
            if Y_settings[0] == "Range":
                col3.write("*Use probability of improvement")
                acquisition_function = "Probability of improvement"
            else:
                acquisition_function = col3.radio("",
                                                  ("Probability of improvement", "Expected improvement"))
        elif len(Y_columns) > 1:
            col3.write("*Use probability of improvement")
            acquisition_function = "Probability of improvement"

        if (len(Y_columns)>0) & (len(X_columns)>0):
            st.write("5.Get the next condition")
            perform = st.button("Perform")
            num_X_for_predictions = int(1e5)
            if perform:
                st.text("Modeling of Gaussian Processing...")
                # settings
                fold_number = 10
                relaxation_value = 0.01

                Y = df[Y_columns]
                X = df[X_columns]
                X_for_predictions_dict = {}
                for j, X_column in enumerate(X_columns):
                    X_for_predictions_dict[X_column] = np.random.uniform(X_min_list[j],
                                                                         X_max_list[j],
                                                                         num_X_for_predictions)
                X_for_predictions_df = pd.DataFrame(X_for_predictions_dict)
                X_autoscaled = (X - X.mean()) / X.std()
                X_for_predictions_autoscaled = (X_for_predictions_df - X.mean()) / X.std()
                Y_autoscaled = (Y - Y.mean()) - Y.std()
                Y_mean = Y.mean()
                Y_std = Y.std()

                estimated_y_for_prediction = np.zeros([X_for_predictions_df.shape[0], len(Y_columns)])
                std_of_estimated_y_for_prediction = np.zeros([X_for_predictions_df.shape[0], len(Y_columns)])
                for y_number in range(len(Y_columns)):
                    st.text("Objective function : {}".format(Y_columns[y_number]))
                    model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(), alpha=0)
                    model.fit(X_autoscaled, Y_autoscaled.iloc[:, y_number])
                    estimated_y_for_prediction_tmp, std_of_estimated_y_for_prediction_tmp = model.predict(
                        X_for_predictions_autoscaled, return_std=True)
                    estimated_y_for_prediction[:, y_number] = estimated_y_for_prediction_tmp
                    std_of_estimated_y_for_prediction[:, y_number] = std_of_estimated_y_for_prediction_tmp

                estimated_y_for_prediction = pd.DataFrame(estimated_y_for_prediction)
                estimated_y_for_prediction.columns = Y.columns
                estimated_y_for_prediction = estimated_y_for_prediction * Y_std + Y_mean
                std_of_estimated_y_for_prediction = pd.DataFrame(std_of_estimated_y_for_prediction)
                std_of_estimated_y_for_prediction.columns = Y.columns
                std_of_estimated_y_for_prediction = std_of_estimated_y_for_prediction * Y.std()

                # PI
                if (len(Y_columns) > 1) or (acquisition_function=="Probability of improvement"):
                    probabilities = np.zeros(estimated_y_for_prediction.shape)
                    for y_number in range(len(Y_columns)):
                        if Y_settings[y_number] == "Maximization":
                            probabilities[:, y_number] = 1 - norm.cdf(
                                max(Y.iloc[:, y_number]) + Y_std.iloc[y_number] * relaxation_value,
                                loc=estimated_y_for_prediction.iloc[:, y_number],
                                scale=std_of_estimated_y_for_prediction.iloc[:, y_number])
                        elif Y_settings[y_number] == "Minimization":
                            probabilities[:, y_number] = norm.cdf(
                                min(Y.iloc[:, y_number]) - Y_std.iloc[y_number] * relaxation_value,
                                loc=estimated_y_for_prediction.iloc[:, y_number],
                                scale=std_of_estimated_y_for_prediction.iloc[:, y_number])

                        elif Y_settings[y_number] == "Range":
                            probabilities[:, y_number] = norm.cdf(Y_max_list[y_number],
                                                                  loc=estimated_y_for_prediction.iloc[:, y_number],
                                                                  scale=std_of_estimated_y_for_prediction.iloc[:,
                                                                        y_number]) - norm.cdf(
                                Y_min_list[y_number],
                                loc=estimated_y_for_prediction.iloc[:, y_number],
                                scale=std_of_estimated_y_for_prediction.iloc[:, y_number])

                        probabilities[std_of_estimated_y_for_prediction.iloc[:, y_number] <= 0, y_number] = 0

                    probabilities = pd.DataFrame(probabilities)
                    probabilities.columns = map(lambda  x: "PI of " + x, Y_columns)
                    probabilities.index = X_for_predictions_df.index
                    sum_of_log_probabilities = (np.log(probabilities)).sum(axis=1)
                    sum_of_log_probabilities = pd.DataFrame(sum_of_log_probabilities)
                    sum_of_log_probabilities[sum_of_log_probabilities == -np.inf] = -10 ** 100
                    sum_of_log_probabilities.columns = ['sum_of_log_probabilities']
                    sum_of_log_probabilities.index = X_for_predictions_df.index

                    st.write('Max of sum of log(probability) : {0}'.format(sum_of_log_probabilities.iloc[:, 0].max()))
                    probabilities
                    probabilities.loc[[sum_of_log_probabilities.iloc[:, 0].idxmax()], :]
                    st.write("Next condition")
                    X_for_predictions_df.loc[[sum_of_log_probabilities.iloc[:, 0].idxmax()], :]

                # EI
                elif acquisition_function == "Expected improvement":
                    if Y_settings[0] == "Maximization":
                        Y_autoscaled_max = Y_autoscaled[Y_columns[0]].max()
                        acquisition_function_values = (estimated_y_for_prediction_tmp - Y_autoscaled_max - relaxation_value) * \
                                                      norm.cdf((estimated_y_for_prediction_tmp -
                                                          Y_autoscaled_max - relaxation_value) /
                                                               std_of_estimated_y_for_prediction_tmp) + \
                                                      std_of_estimated_y_for_prediction_tmp * \
                                                      norm.pdf((estimated_y_for_prediction_tmp -
                                                          Y_autoscaled_max - relaxation_value) /
                                                               std_of_estimated_y_for_prediction_tmp)
                        st.write('Max of EI : {0}'.format(max(acquisition_function_values)))
                        X_for_predictions_df.loc[[np.argmax(acquisition_function_values)], :]
                    elif Y_settings[0] == "Minimization":
                        Y_autoscaled_max = -Y_autoscaled[Y_columns[0]].min()
                        acquisition_function_values = (-estimated_y_for_prediction_tmp - Y_autoscaled_max - relaxation_value) * \
                                                      norm.cdf((-estimated_y_for_prediction_tmp -
                                                          Y_autoscaled_max - relaxation_value) /
                                                               std_of_estimated_y_for_prediction_tmp) + \
                                                      std_of_estimated_y_for_prediction_tmp * \
                                                      norm.pdf((-estimated_y_for_prediction_tmp -
                                                          Y_autoscaled_max - relaxation_value) /
                                                               std_of_estimated_y_for_prediction_tmp)
                        st.write('Max of EI : {0}'.format(max(acquisition_function_values)))
                        X_for_predictions_df.loc[[np.argmax(acquisition_function_values)], :]
