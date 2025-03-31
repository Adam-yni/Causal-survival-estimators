import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

class Kaplan_Meier_Estimator:
    def __init__(self, times, events, treatment):
        self.times = times
        self.events = events
        self.treatment = treatment

    def Dk(self, a, unique_times):
        return np.array([np.sum((self.times == t) & (self.treatment == a) & (self.events == 1) ) for t in unique_times])
    def Nk(self, a, unique_times):
        return np.array([np.sum((self.times >= t) & (self.treatment == a) ) for t in unique_times])
    
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
    
    def IPCW_Dk(self, a, unique_times, censoring_probs):
            return np.array([
                np.sum(
                       (1 / censoring_probs) * 
                       ((self.times == t) & (self.treatment == a) & (self.events == 1)))
                for t in unique_times
            ])
        
    def IPCW_Nk(self, a, unique_times, censoring_probs):
            return np.array([
                np.sum( 
                       (1 / censoring_probs) * 
                       ((self.times >= t) & (self.treatment == a)))
                for t in unique_times
            ])
    
    def IPTW_IPCW_Dk(self, a, unique_times, propensity_scores, censoring_probs):
            return np.array([
                np.sum(((a / propensity_scores) + ((1 - a) / (1 - propensity_scores))) * 
                       (1 / censoring_probs) * 
                       ((self.times == t) & (self.treatment == a) & (self.events == 1)))
                for t in unique_times
            ])
        
    def IPTW_IPCW_Nk(self, a, unique_times, propensity_scores, censoring_probs):
            return np.array([
                np.sum(((a / propensity_scores) + ((1 - a) / (1 - propensity_scores))) * 
                       (1 / censoring_probs) * 
                       ((self.times >= t) & (self.treatment == a)))
                for t in unique_times
            ])
    
    def Estimation(self,a):
        unique_times = np.sort(np.unique(self.times))
        Dk_a = self.Dk(a, unique_times)
        Nk_a = self.Nk(a, unique_times)

        survival_prob = 1
        survival_probs = []

        for d, n in zip(Dk_a, Nk_a):
            if n > 0:
                survival_prob *= (1- d/n)
            survival_probs.append(survival_prob)
        return unique_times, survival_probs
    

    def IPTW_Estimation(self, a, covariates: np.ndarray, n_estimators, max_depth):
        
        # Fit logistic regression model for propensity scores
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=123)
        
        model.fit(covariates,self.treatment.reshape(-1, 1))
        propensity_scores = model.predict_proba(covariates)[:, 1]
        #In order not to violate positivity assumption all the probabilities need to verify 0<probability<1
        #Otherwise causal estimation will fail
        unique_times = np.sort(np.unique(self.times))
        
        Dk_a = self.IPTW_Dk(a, unique_times, propensity_scores)
        Nk_a = self.IPTW_Nk(a, unique_times, propensity_scores)

        survival_prob = 1
        survival_probs = []

        for d, n in zip(Dk_a, Nk_a):
            if n > 0:
                survival_prob *= (1 - d / n)
            survival_probs.append(survival_prob)
        
        return unique_times, survival_probs
    
    def IPTW_IPCW_Estimation(self, a, covariates: np.ndarray, n_estimators, max_depth):
        
        # Fit logistic regression model for propensity scores
        model_ps = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=123)
        model_ps.fit(covariates, self.treatment.reshape(-1, 1))
        propensity_scores = model_ps.predict_proba(covariates)[:, 1]

        # Print F1 score for propensity score model
        propensity_preds = model_ps.predict(covariates)
        f1_propensity = f1_score(self.treatment, propensity_preds)
        print(f"F1 Score for Propensity Score Model: {f1_propensity}")

        # Fit logistic regression model for censoring probabilities
        censoring_indicator = 1 - self.events
        model_censoring = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=123)
        model_censoring.fit(covariates, censoring_indicator)
        censoring_probs = model_censoring.predict_proba(covariates)[:, 1]
        #In order not to violate positivity assumption all the probabilities need to verify 0<probability<1
        #Otherwise causal estimation will fail

        # Print F1 score for censoring probability model
        censoring_preds = model_censoring.predict(covariates)
        f1_censoring = f1_score(censoring_indicator, censoring_preds)
        print(f"F1 Score for Censoring Probability Model: {f1_censoring}")

        unique_times = np.sort(np.unique(self.times))
        
        Dk_a = self.IPTW_IPCW_Dk(a, unique_times, propensity_scores, censoring_probs)
        Nk_a = self.IPTW_IPCW_Nk(a, unique_times, propensity_scores, censoring_probs)

        survival_prob = 1
        survival_probs = []

        for d, n in zip(Dk_a, Nk_a):
            if n > 0:
                survival_prob *= (1 - d / n)
            survival_probs.append(survival_prob)
        
        return unique_times, survival_probs
    

    def IPCW_Estimation(self, a, covariates: np.ndarray, n_estimators=50, max_depth=5):
        unique_times = np.sort(np.unique(self.times))
        # Fit RandomForest model for censoring probabilities
        
        censoring_indicator = 1 - self.events
        
        model_censoring = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=123)
        model_censoring.fit(covariates, censoring_indicator)
        censoring_probs = model_censoring.predict_proba(covariates)[:, 1]
        #In order not to violate positivity assumption all the probabilities need to verify 0<probability<1
        #Otherwise causal estimation will fail
        # Print F1 score for censoring probability model
        censoring_preds = model_censoring.predict(covariates)
        f1_censoring = f1_score(censoring_indicator, censoring_preds)
        print(f"F1 Score for Censoring Probability Model: {f1_censoring}")

        Dk_a = self.IPCW_Dk(a, unique_times, censoring_probs)
        Nk_a = self.IPCW_Nk(a, unique_times, censoring_probs)

        survival_prob = 1
        survival_probs = []

        for d, n in zip(Dk_a, Nk_a):
            if n > 0:
                survival_prob *= (1 - d / n)
            survival_probs.append(survival_prob)
        
        return unique_times, survival_probs
       
    def Plot(self, times_treated, survival_probs_treated, times_control, survival_probs_control):
        plt.step(times_treated, survival_probs_treated, where="post", label="Kaplan Meier Estimate (Treated)")
        plt.step(times_control, survival_probs_control, where="post", label="Kaplan Meier Estimate (Control)")

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.title("Kaplan-Meier Survival Curve by Treatlent Group")
        plt.grid = True
        plt.show()


    
    
