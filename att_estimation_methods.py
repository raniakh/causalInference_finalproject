from sklearn.linear_model import LogisticRegression
import numpy as np


def estimate_propensity(X, treatment):
    # Learn propensity score
    lr_learner = LogisticRegression(C=10, solver='lbfgs', max_iter=10000)
    lr_learner.fit(X, treatment)
    propensity_score = lr_learner.predict_proba(X)[:, 1]
    return propensity_score


def IPW_ATT_calculation(propensity_score, treatment, outcome):
    inverse_propensity_factor = propensity_score / (1 - propensity_score)
    treatment_element = (treatment * outcome).sum() / treatment.sum()
    control = 1 - treatment
    control_element = (control * outcome * inverse_propensity_factor).sum() / (
                control * inverse_propensity_factor).sum()

    return treatment_element - control_element


def t_learner_att_prediction(covariates, treatment, outcome, control_model, treated_model):
    control_covariates = covariates[treatment == 0]
    treated_covariates = covariates[treatment == 1]
    control_outcome = outcome[treatment == 0]
    treated_outcome = outcome[treatment == 1]

    control_model.fit(control_covariates, control_outcome)
    treated_model.fit(treated_covariates, treated_outcome)

    # For ATT estimation we need only the treated group
    control_prediction = control_model.predict(treated_covariates)
    treatment_prediction = treated_model.predict(treated_covariates)

    pure_prediction_att = (treatment_prediction - control_prediction).sum() / treated_covariates.shape[0]
    prediction_observation_att = (outcome[treatment == 1] - control_prediction).sum() / treated_covariates.shape[0]

    return pure_prediction_att, prediction_observation_att


def s_learner_att_prediction(covariates, treatment, outcome, model, two_d_plus_one=False):
    treated = covariates[treatment == 1]

    if two_d_plus_one:
        interaction_with_treatment = np.apply_along_axis(lambda vec1, vec2: vec1 * vec2,
                                                         0, covariates, treatment)
        X = np.column_stack([covariates, interaction_with_treatment, treatment])
        counterfactual_treated = np.column_stack([treated, np.zeros((treated.shape[0], treated.shape[1] + 1))])
    else:
        X = np.column_stack([covariates, treatment])
        counterfactual_treated = np.column_stack([treated, np.zeros(treated.shape[0])])

    model.fit(X, outcome)
    actual_treated = X[treatment == 1]

    counterfactual_pred = model.predict(counterfactual_treated)
    actual_pred = model.predict(actual_treated)

    return np.mean(actual_pred - counterfactual_pred)


def doubly_robust_att(covariates, treatment, outcome, control_model, treated_model):

    control_covariates = covariates[treatment == 0]
    treated_covariates = covariates[treatment == 1]
    control_outcome = outcome[treatment == 0]
    treated_outcome = outcome[treatment == 1]
    control_model.fit(control_covariates, control_outcome)
    treated_model.fit(treated_covariates, treated_outcome)

    # prediction on treated population
    control_prediction = control_model.predict(treated_covariates)
    treatment_prediction = treated_model.predict(treated_covariates)

    propensity = estimate_propensity(covariates, treatment)
    g1 = treatment_prediction + (1/propensity[treatment == 1])*(treated_outcome - treatment_prediction)
    g0 = control_prediction

    return np.mean(g1-g0)
