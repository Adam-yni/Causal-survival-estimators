import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.integrate import trapezoid

class Kaplan_Meier_Estimator:
    def __init__(self, times, events, treatment):
        self.times = times
        self.events = events
        self.treatment = treatment

    def _fit_model(self, covariates, target, model_type, n_estimators=50, max_depth=5):
        """
        Train a model (logistic regression, random forest, or XGBoost) and return predicted probabilities.
        """
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=123)
        elif model_type == "logistic_regression":
            model = LogisticRegression(max_iter=5000, random_state=123)
        elif model_type == "xgboost":
            model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, use_label_encoder=False, eval_metric="logloss", random_state=123)
        else:
            raise ValueError("Invalid model_type. Choose 'random_forest', 'logistic_regression', or 'xgboost'.")

        model.fit(covariates, target)
        probabilities = model.predict_proba(covariates)[:, 1]  # Probabilities for the positive class
        
        return np.clip(probabilities, 1e-6, 1 - 1e-6)  # Avoid probabilities of exactly 0 or 1

    def _calculate_survival_probs(self, Dk, Nk):
        """
        Calculate survival probabilities using Kaplan-Meier formula.
        """
        survival_prob = 1
        survival_probs = []

        for d, n in zip(Dk, Nk):
            if n > 0:
                survival_prob *= (1 - d / n)
            survival_probs.append(survival_prob)

        return survival_probs

    def Estimation(self, a):
        """
        Classic Kaplan-Meier estimation.
        """
        unique_times = np.sort(np.unique(self.times))
        Dk_a = self.Dk(a, unique_times)
        Nk_a = self.Nk(a, unique_times)

        survival_probs = self._calculate_survival_probs(Dk_a, Nk_a)
        return unique_times, survival_probs

    def IPTW_Estimation(self, a, covariates, model_type="xgboost", n_estimators=500, max_depth=1):
        """
        Inverse Probability of Treatment Weighting (IPTW) estimation.
        """
        propensity_scores = self._fit_model(covariates, self.treatment, model_type, n_estimators, max_depth)
        unique_times = np.sort(np.unique(self.times))

        Dk_a = self.IPTW_Dk(a, unique_times, propensity_scores)
        Nk_a = self.IPTW_Nk(a, unique_times, propensity_scores)

        survival_probs = self._calculate_survival_probs(Dk_a, Nk_a)
        return unique_times, survival_probs

    def IPTW_IPCW_Estimation(self, a, covariates, model_type="xgboost", n_estimators=50, max_depth=1):
        """
        IPTW combined with IPCW estimation.
        """
        propensity_scores = self._fit_model(covariates, self.treatment, model_type, n_estimators, max_depth)
        censoring_indicator = 1 - self.events
        censoring_probs = self._fit_model(covariates, censoring_indicator, model_type, n_estimators, max_depth)

        unique_times = np.sort(np.unique(self.times))

        Dk_a = self.IPTW_IPCW_Dk(a, unique_times, propensity_scores, censoring_probs)
        Nk_a = self.IPTW_IPCW_Nk(a, unique_times, propensity_scores, censoring_probs)

        survival_probs = self._calculate_survival_probs(Dk_a, Nk_a)
        return unique_times, survival_probs

    def IPCW_Estimation(self, a, covariates, model_type="xgboost", n_estimators=500, max_depth=10):
        """
        Inverse Probability of Censoring Weighting (IPCW) estimation.
        """
        censoring_indicator = 1 - self.events
        censoring_probs = self._fit_model(covariates, censoring_indicator, model_type, n_estimators, max_depth)

        unique_times = np.sort(np.unique(self.times))

        Dk_a = self.IPCW_Dk(a, unique_times, censoring_probs)
        Nk_a = self.IPCW_Nk(a, unique_times, censoring_probs)

        survival_probs = self._calculate_survival_probs(Dk_a, Nk_a)
        return unique_times, survival_probs

    def Dk(self, a, unique_times):
        return np.array([np.sum((self.times == t) & (self.treatment == a) & (self.events == 1)) for t in unique_times])

    def Nk(self, a, unique_times):
        return np.array([np.sum((self.times >= t) & (self.treatment == a)) for t in unique_times])

    def IPTW_Dk(self, a, unique_times, propensity_scores):
        return np.array([
            np.sum(((a / propensity_scores) + ((1 - a) / (1 - propensity_scores))) *
                   ((self.times == t) & (self.treatment == a) & (self.events == 1)))
            for t in unique_times
        ])

    def IPTW_Nk(self, a, unique_times, propensity_scores):
        return np.array([
            np.sum(((a / propensity_scores) + ((1 - a) / (1 - propensity_scores))) *
                   ((self.times >= t) & (self.treatment == a)))
            for t in unique_times
        ])

    def IPTW_IPCW_Dk(self, a, unique_times, propensity_scores, censoring_probs):
        return np.array([
            np.sum(((a / propensity_scores) + ((1 - a) / (1 - propensity_scores))) *
                   ((1 - self.events) / ( censoring_probs) + (self.events) / (1-censoring_probs)) *
                   ((self.times == t) & (self.treatment == a) & (self.events == 1)))
            for t in unique_times
        ])

    def IPTW_IPCW_Nk(self, a, unique_times, propensity_scores, censoring_probs):
        return np.array([
            np.sum(((a / propensity_scores) + ((1 - a) / (1 - propensity_scores))) *
                   ((1 - self.events) / ( censoring_probs) + (self.events) / (1-censoring_probs))*
                   ((self.times >= t) & (self.treatment == a)))
            for t in unique_times
        ])

    def IPCW_Dk(self, a, unique_times, censoring_probs):
        return np.array([
            np.sum((1 / censoring_probs) *
                   ((self.times == t) & (self.treatment == a) & (self.events == 1)))
            for t in unique_times
        ])

    def IPCW_Nk(self, a, unique_times, censoring_probs):
        return np.array([
            np.sum((1 / censoring_probs) *
                   ((self.times >= t) & (self.treatment == a)))
            for t in unique_times
        ])
    
    def calculate_causal_effect(self, covariates=None, model_type="xgboost", n_estimators=50, max_depth=1, method="IPTW_IPCW"):
        """
        Calculate the causal effect of treatment on outcome using different methods.
        """
        if method == "Estimation":
            unique_times_treated, survival_probs_treated = self.Estimation(a=1)
            unique_times_control, survival_probs_control = self.Estimation(a=0)
        elif method == "IPCW":
            unique_times_treated, survival_probs_treated = self.IPCW_Estimation(
                a=1, covariates=covariates, model_type=model_type, n_estimators=n_estimators, max_depth=max_depth
            )
            unique_times_control, survival_probs_control = self.IPCW_Estimation(
                a=0, covariates=covariates, model_type=model_type, n_estimators=n_estimators, max_depth=max_depth
            )
        elif method == "IPTW":
            unique_times_treated, survival_probs_treated = self.IPTW_Estimation(
                a=1, covariates=covariates, model_type=model_type, n_estimators=n_estimators, max_depth=max_depth
            )
            unique_times_control, survival_probs_control = self.IPTW_Estimation(
                a=0, covariates=covariates, model_type=model_type, n_estimators=n_estimators, max_depth=max_depth
            )
        elif method == "IPTW_IPCW":
            unique_times_treated, survival_probs_treated = self.IPTW_IPCW_Estimation(
                a=1, covariates=covariates, model_type=model_type, n_estimators=n_estimators, max_depth=max_depth
            )
            unique_times_control, survival_probs_control = self.IPTW_IPCW_Estimation(
                a=0, covariates=covariates, model_type=model_type, n_estimators=n_estimators, max_depth=max_depth
            )
        else:
            raise ValueError("Invalid method. Choose from 'Estimation', 'IPCW', 'IPTW', or 'IPTW_IPCW'.")

        auc_treated = trapezoid(survival_probs_treated, unique_times_treated)
        auc_control = trapezoid(survival_probs_control, unique_times_control)

        Causal_effect = auc_treated / auc_control  # Causal effect of treatment on outcome

        return Causal_effect