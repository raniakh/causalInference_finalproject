import pandas as pd
from att_estimation_methods import *
import xgboost as xgb


def confidence_interval_IPW(data, bootstrap_samples):
    """
    return ATT estimation and 95% confidence interval using IPW method and bootstrap.
    """
    outcome = data["outcome"]
    treatment = data["T"]
    covariates = data.loc[:, ~data.columns.isin(['T', 'outcome'])]

    # estimation ATT from original sample
    propensity = estimate_propensity(covariates, treatment)
    att = IPW_ATT_calculation(propensity, treatment, outcome)

    delta = []
    for i in range(bootstrap_samples):
        sample_indices = np.random.choice(data.index, size=len(data))
        bootstrap_sample = data.loc[sample_indices, :].copy()
        bootstrap_treatment = bootstrap_sample["T"]
        bootstrap_outcome = bootstrap_sample["outcome"]
        bootstrap_covariates = bootstrap_sample.loc[:, ~bootstrap_sample.columns.isin(['T', 'outcome'])]
        propensity = estimate_propensity(bootstrap_covariates, bootstrap_treatment)
        bootstrap_att = IPW_ATT_calculation(propensity, bootstrap_treatment, bootstrap_outcome)
        delta.append(bootstrap_att - att)
    lower, higher = np.quantile(delta, [0.025, 0.975], interpolation='lower')
    return att, [att - higher, att - lower]


def confidence_interval_s_learner(data, bootstrap_samples):
    """
    return ATT estimation and 95% confidence interval using s-learner method and bootstrap.
    """
    outcome = data["outcome"]
    treatment = data["T"]
    covariates = data.loc[:, ~data.columns.isin(['T', 'outcome'])]

    # estimation ATT from original sample
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=1,
                             max_depth=5, alpha=10, n_estimators=10)
    att = s_learner_att_prediction(covariates, treatment, outcome, model, two_d_plus_one=True)

    delta = []
    for i in range(bootstrap_samples):
        sample_indices = np.random.choice(data.index, size=len(data))
        bootstrap_sample = data.loc[sample_indices, :].copy()
        bootstrap_treatment = bootstrap_sample["T"]
        bootstrap_outcome = bootstrap_sample["outcome"]
        bootstrap_covariates = bootstrap_sample.loc[:, ~bootstrap_sample.columns.isin(['T', 'outcome'])]
        model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=1,
                                 max_depth=5, alpha=10, n_estimators=10)
        bootstrap_att = s_learner_att_prediction(bootstrap_covariates, bootstrap_treatment, bootstrap_outcome, model,
                                                 two_d_plus_one=True)
        delta.append(bootstrap_att - att)
    lower, higher = np.quantile(delta, [0.025, 0.975], interpolation='lower')
    return att, [att - higher, att - lower]


def confidence_interval_t_learner(data, bootstrap_samples):
    """
    return ATT estimation and 95% confidence interval using t-learner method and bootstrap.
    """
    outcome = data["outcome"]
    treatment = data["T"]
    covariates = data.loc[:, ~data.columns.isin(['T', 'outcome'])]

    # estimation ATT from original sample
    treated_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=1,
                                     max_depth=5, alpha=10, n_estimators=10)
    control_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=1,
                                     max_depth=5, alpha=10, n_estimators=10)
    att = t_learner_att_prediction(covariates, treatment, outcome, control_model, treated_model)[1]

    delta = []
    for i in range(bootstrap_samples):
        sample_indices = np.random.choice(data.index, size=len(data))
        bootstrap_sample = data.loc[sample_indices, :].copy()
        bootstrap_treatment = bootstrap_sample["T"]
        bootstrap_outcome = bootstrap_sample["outcome"]
        bootstrap_covariates = bootstrap_sample.loc[:, ~bootstrap_sample.columns.isin(['T', 'outcome'])]
        treated_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=1,
                                         max_depth=5, alpha=10, n_estimators=10)
        control_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=1,
                                         max_depth=5, alpha=10, n_estimators=10)
        bootstrap_att = t_learner_att_prediction(bootstrap_covariates, bootstrap_treatment, bootstrap_outcome,
                                                 control_model, treated_model)[1]
        delta.append(bootstrap_att - att)
    lower, higher = np.quantile(delta, [0.025, 0.975], interpolation='lower')
    return att, [att - higher, att - lower]


def confidence_interval_doubly_robust_estimator(data, bootstrap_samples):
    """
    return ATT estimation 95% confidence interval using the doubly robust estimator method and bootstrap.
    """
    outcome = data["outcome"]
    treatment = data["T"]
    covariates = data.loc[:, ~data.columns.isin(['T', 'outcome'])]

    # estimation ATT from original sample
    treated_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=1,
                                     max_depth=5, alpha=10, n_estimators=10)
    control_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=1,
                                     max_depth=5, alpha=10, n_estimators=10)
    att = doubly_robust_att(covariates, treatment, outcome, control_model, treated_model)
    delta = []
    for i in range(bootstrap_samples):
        sample_indices = np.random.choice(data.index, size=len(data))
        bootstrap_sample = data.loc[sample_indices, :].copy()
        bootstrap_treatment = bootstrap_sample["T"]
        bootstrap_outcome = bootstrap_sample["outcome"]
        bootstrap_covariates = bootstrap_sample.loc[:, ~bootstrap_sample.columns.isin(['T', 'outcome'])]
        treated_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=1,
                                         max_depth=5, alpha=10, n_estimators=10)
        control_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=1,
                                         max_depth=5, alpha=10, n_estimators=10)
        bootstrap_att = doubly_robust_att(bootstrap_covariates, bootstrap_treatment, bootstrap_outcome,
                                          control_model, treated_model)
        delta.append(bootstrap_att - att)
    lower, higher = np.quantile(delta, [0.025, 0.975], interpolation='lower')
    return att, [att - higher, att - lower]


def main():
    paths = ['canada_treatment_1', 'canada_treatment_2', 'australia_treatment_2', 'ireland_treatment_2']
    # paths = ['canada12_treatment_1','canada12_treatment_2', 'canada_treatment_4']
    for path in paths:
        data = pd.read_csv("data_by_country/{}.csv".format(path))

        att_ipw, confidence_interval_ipw = confidence_interval_IPW(data, 500)
        att_t, confidence_interval_t = confidence_interval_t_learner(data, 500)
        att_s, confidence_interval_s = confidence_interval_s_learner(data, 500)
        att_dr, confidence_interval_dr = confidence_interval_doubly_robust_estimator(data, 500)

        label = path.split("_")
        prefix = "{}, {} {}".format(label[0].capitalize(), label[1], label[2])
        print("{} ATT estimation by IPW: {}, with confidence interval {}".format(prefix, att_ipw,
                                                                                 confidence_interval_ipw))
        print("{} ATT estimation by t-learner: {}, with confidence interval {}".format(prefix, att_t,
                                                                                       confidence_interval_t))
        print("{} ATT estimation by s-learner: {}, with confidence interval {}".format(prefix, att_s,
                                                                                       confidence_interval_s))
        print("{} ATT estimation by doubly estimator: {}, with confidence interval {}".format(prefix, att_dr,
                                                                                              confidence_interval_dr))
        print()


if __name__ == "__main__":
    main()
