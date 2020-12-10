import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
import statsmodels 
from tqdm import tnrange, tqdm_notebook
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import time
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.special



def fleiss_kappa(table, alpha=0.05):
    ''' Fleiss' kappa for interrater reliability. Two-sided hypothesis test H_0: kappa=0.

    Parameters
    ----------
    table: array_like, 2-D
        assumes subjects in rows and categories in columns

    Returns
    -------
        Fleiss's kappa statistic for interrater agreement, confidence intervals, p-value
        
    TODO
    ----
    What happens if kappa < 0 with p-value? 
    
    
    '''

    table = np.asarray(table) 
    n_sub, n_cat = table.shape
    n_rater = table.sum(1)[0]
    n_total = table.sum()
    
    # Proportion of all assignments in j-th category
    # p_j = 1/(N*n) * sum_i_N ( n_ij ) 

    p_j = (1./(n_rater*n_sub)) * table.sum(axis=0)

    # extent of agreement for the i-th subject 
    # P_i = 1/(n*(n-1)) * sum_j_k ( n_ij**2) - n 

    P_i = []
    for i in range(0,n_sub):
        _temp = (1./(n_rater*(n_rater-1))) * ( (table[i,:]**2).sum() - n_rater ) 
        P_i.append(_temp) 
    P_i = np.asarray(P_i)
    
    P_mean = 1/n_sub*P_i.sum()
    
    P_exp = (p_j**2).sum()
    
    std_err = np.sqrt( (P_mean*(1-P_exp))/(n_sub*(1-P_exp)**2) )
    kappa = (P_mean-P_exp)/(1-P_exp)
    statistic = kappa/std_err
    p_value = 1 - stats.norm.cdf(statistic)
    
    return kappa, (kappa - std_err*stats.norm.ppf(1-alpha/2), kappa + std_err*stats.norm.ppf(1-alpha/2)), p_value
    
    
def cohen_kappa(table, alpha=0.05):
    ''' Cohen's kappa for interrater reliability
    
    Parameters
    ----------
    table: array_like, 2-D
        square array with results of two raters, one rater in rows, second rater in columns
    alpha: int
        
    Returns
    -------
        Cohen's kappa statistic for interrater agreement and confidence intervals
    
    '''
        

class MantelHan():
    ''' Mantel-Haenszel Test to assess association between disease and exposure after controlling for one or more confounding variables. Either a list containing several 2x2 contingency tables, or a 2x2xk ndarray in which each slice along the third axis is a 2x2 contingency table.
    
    TODO: check if pvalue is correct!
    
    '''
    
    def __init__(self,tables,alpha=0.05 ):
        
        # 1.Form k strata based on levels of the confounding variable and construct a 2 × 2 table
        # within each stratum
        tables = np.asarray( tables )
        self.ntables = tables.shape[2]
        
        # 2.Compute the total number of observed units in the (1,1) cells
        obs11 = 0
        for table in tables:
            obs11 = obs11 + table[0,0]
        
        # 3.Compute the total number of expected units in the (1,1) cells
        exp11_tot = 0
        for table in tables:
            exp11 = (table.sum(axis=1)[0] * table.sum(axis=0)[0])/table.sum()
            exp11_tot = exp11_tot + exp11
        
        # 4.Compute the variance of O under H 0
        var_tot = 0
        for table in tables:
            var = (table.sum(axis=1)[0] * table.sum(axis=0)[0] * table.sum(axis=1)[1] * table.sum(axis=0)[1] ) / ( np.square(table.sum())*(table.sum()-1) )
            var_tot = var_tot + var
        
        if var_tot < 5:
            warnings.warn('Only use this test if variance >=5',UserWarning )
            
        # 5.Calculate the test statistic
        
        self.test_statistic = (np.abs(obs11-exp11_tot)-0.5)**2/var_tot
        
        # 6. two-sided test with significance level alpha, reject H0 if test_statistic > X(1,1-alpha)
        self.hypothesis = self.test_statistic>stats.chi2.ppf(1-alpha,df=1)
        
        # 7. The exact p-value for this test is given by
        
        self.pvalue = stats.chi2.sf(self.test_statistic,df=1)
        
        # assess strength of association
        # this calculation is extremely messy and I hope noone ever has to check this again
        
        num1, sumr, num2, sums, num3 = 0,0,0,0,0
        num11,den11 = 0,0
        
        for table in tables:
            num = (table[0,0]*table[1,1]/table.sum())
            den = (table[0,1]*table[1,0]/table.sum())
            num11 = num11 + num
            den11 = den11 + den
            
            p = (table[0,0]+table[1,1])/table.sum()
            q = (table[0,1]+table[1,0])/table.sum()
            r = table[0,0]*table[1,1]/table.sum()
            s = table[0,1]*table[1,0]/table.sum()
            
            num1 = num1 + p*r
            sumr = sumr + r
            num2 = num2 + (p*s+q*r)
            sums = sums + s
            num3 = num3 + q*s
        
        self.var_log_odds = num1/(2*sumr**2) + num2/(2*sumr*sums) + num3/(2*sums**2)
        self.odds_mh = num11/den11
        self.ci_low = np.exp(np.log(self.odds_mh) - stats.norm.ppf(1-alpha/2)*np.sqrt(self.var_log_odds))
        self.ci_up = np.exp(np.log(self.odds_mh) + stats.norm.ppf(1-alpha/2)*np.sqrt(self.var_log_odds))
        
        # Odds of having disease when exposed is odds_mh times larger than the odds of having disease among persons who were not exposed with 95% CI
        
        # Test for Homogeneity of Odds Ratios
        
        
        def test_equal_odds():
            ''' Test that all odds ratios are identical. Uses Woolf Method weighted average of the squared deviations of log odds_mh_i from average log odds_mh with i indexing strata of the ptential effect modifier. If the underlying OR is different across k strat, there is said to be an interaction or effect modificatio nbetween exposure E and variable C. C is referred to as an effect modifier. Only compute common odds ratio if strength of association is the same within strata!
            '''
            # 1. Calculate the test statistic
            self.test_statistic_hom = 0
            aver_odds = 0 
            
            for table in tables:
                aver_odds = aver_odds + np.log((table[0,0]*table[1,1])/(table[0,1]*table[1,0]))
            log_aver_odds = aver_odds/self.ntables
            
            for table in tables:
                weight = 1/(1/table[0,0]+1/table[1,1]+1/table[1,0]+1/table[0,1])
                odds = (table[0,0]*table[1,1])/(table[0,1]*table[1,0])
                self.test_statistic_hom = self.test_statistic_hom + weight*(np.log(odds)-log_aver_odds)**2
            
            # 2. test if h0 rejected
            
            self.hypothesis_hom = self.test_statistic_hom>stats.chi2.ppf(1-alpha,df=self.ntables-1)
            
            # 3. pvalue
            
            self.pvalue_hom = stats.chi2.sf(self.test_statistic_hom,df=self.ntables-1)
            
            return self.hypothesis_hom
            
        # Test for equal strength of association in strata
        if test_equal_odds():
            warnings.warn('Strength of association is not equal. Do not calculate common odds ratio', UserWarning)
        
        
            





def odds_ratio(table,design,alpha=0.05):
    ''' Assuming a 2x2 contigency table. Design can be prospective or retrospective. Prospective study disease-odds-ratio, proportion with disease among exposed and unexposed. Retrospective study exposure-odds-ratio, proportion with exposure among diseases and nondiseased.
    
    Parameters
    ----------
    
    table: pd.DataFrame, array-like
    ------
    
    design: either prosp or retro
    ------
    
    '''
    
    if design=='prosp':
        p1 = (table[0,0]/(table[0,0]+table[0,1]))
        p2 = (table[1,0]/(table[1,0]+table[1,1]))
        if ((table.sum(axis=1)[0]*p1*(1-p1)>=5) & (table.sum(axis=1)[1]*p2*(1-p2)>=5)):
            odds = (table[0,0]*table[1,1])/(table[0,1]*table[1,0])
            c1 = np.log(odds) - stats.norm.sf(1-alpha/2)*np.sqrt(1/table[0,0]+1/table[1,1]+1/table[1,0]+1/table[0,1])
            c2 = np.log(odds) + stats.norm.sf(1-alpha/2)*np.sqrt(1/table[0,0]+1/table[1,1]+1/table[1,0]+1/table[0,1])

        else:
            odds = (table[0,0]*table[1,1])/(table[0,1]*table[1,0])
            warnings.warn('Normal assumption not satisfied. Cannot compute confidence intervals',UserWarning)
            return odds
        
    else:
        p1 = table[0,0]/(table[0,0]+table[1,0])
        p2 = table[0,1]/(table[0,1]+table[1,1])
        if ((table.sum(axis=0)[0]*p1*(1-p1)>=5) & (table.sum(axis=0)[1]*p2*(1-p2)>=5)):
            odds = (table[0,0]*table[1,1])/(table[0,1]*table[1,0])
            c1 = np.log(odds) - stats.norm.sf(1-alpha/2)*np.sqrt(1/table[0,0]+1/table[1,1]+1/table[1,0]+1/table[0,1])
            c2 = np.log(odds) + stats.norm.sf(1-alpha/2)*np.sqrt(1/table[0,0]+1/table[1,1]+1/table[1,0]+1/table[0,1])
        else:
            odds = (table[0,0]*table[1,1])/(table[0,1]*table[1,0])
            warnings.warn('Normal assumption not satisfied. Cannot compute confidence intervals',UserWarning)
            return odds
            
    colum = ['Odds Ratio','Conf_low','Conf_up']
    data = [odds,np.exp(c1),np.exp(c2)]
    df = pd.DataFrame(columns= colum)
    df.loc[0] = data
        
    return df
        

class anova():
    ''' Takes a DataFrame and calculates means. 
    
    Parameters
    ----------
    
    dv: string, datacolumn
    
    between: string, column with inbetween factors
    
    data: DataFrame containing datacolumn
    
    var,freq: string
    should be keywords, only if N and variance known
    
    DataFrame has the following form
    
        x    y
    0    1    0.267845
    1    1    -0.935268
    2    2    0.008010
    3    2    0.313719
    4    3    1.49725
    
    or 
        mean    index     var freq
    0    23.343
    1    34.343
    
    use DataFrame.reset_index()
    
    # what happens if data is not orderd and starts with a 2?
    
    '''

    def __init__(self, data, between, dv, var=None,freq=None):
        table = data
        self.factors = data[between].unique()
        values = data[dv].values
        
        

        
        # depends on form of data
        def helper_simple_table():
            
            self.means = data[dv]
            self.grand_mean = self.means.mean()
            self.group_effects = self.means - self.grand_mean
            self.var = data[var]
            
            freq_n = data[freq]
                
            self.mse, num, den = 0,0,0
                
            for index,varr in enumerate(self.var):
                    
                num = num + (freq_n[index]-1)*varr 
                den = den + (freq_n[index]-1)
                
            self.mse=num/den

        def helper_complex_table(): 
        
            self.means = data.groupby(by=between).mean()[dv].values
            self.grand_mean = self.means.mean()
            self.group_effects = self.means - self.grand_mean

            self.var = data.groupby(by=between).var()[dv].values
            
            self.mse, num, den = 0,0,0
            
            for index, varr in enumerate(self.var):
            
                num = num + (data.loc[data[between] == self.factors[index] ,between].value_counts().values[0]-1)*varr
                den = (data.loc[data[between] == self.factors[index] ,between].value_counts().values[0]-1)
                    
            self.mse = num/den                 
        
        if (var==None) & (freq==None): helper_complex_table()
        else: helper_simple_table()
                
        

        
    
        
    
    def get_table(self):
        ''' Reorder data and return dataframe with factors and mean. For visualization purposes.
        doesnt work for simple table
        '''
    
        columns = ['Mean']
       
        return pd.DataFrame(data=self.means,index=self.factors,columns=columns)
                
        
    
        


class LinearRegression():

    def __init__(self,exog=None,endog=None,*args,**kwargs):
        self.exog = exog.astype('float64')
        self.endog = endog
        
    def variance_inflation_factor(self):
    
        design_matrix = statsmodels.tools.tools.add_constant(self.exog)
        
        
        vif = np.zeros_like(design_matrix.columns,dtype='float64')
        
        for index,predictor in enumerate(design_matrix.columns):
            vif[index] = statsmodels.stats.outliers_influence.variance_inflation_factor(design_matrix.values,
            index)
                
        return pd.Series(vif,index=design_matrix.columns)
        
        






def root_mean_error(test, predict):

    return np.sqrt(((test-predict)**2).sum() * 1/test.shape[0])
    
    

class ModelSelection():
    ''' make sure it already contains dummy variables. constant is added'''
    def __init__(self,exog=None,endog=None,*args,**kwargs):
    
        self.exog = sm.tools.tools.add_constant(exog)
        self.endog = endog
        self.predictors = exog.columns
        
    def BackwardElimination(self,crit_val=2,model='OLS'):
        if model == 'OLS':
            compare = True
            final_model = self.exog
            while(compare):
                tvalue = sm.OLS(self.endog,final_model).fit().tvalues
                size_new = tvalue.size
                
                if (tvalue[np.abs(tvalue)>crit_val].size) != (size_new):
                    final_model = final_model[tvalue[np.abs(tvalue)>crit_val].index]
                    
                else:
                    compare = False 
                    
            return final_model
        else:
            compare = True
            final_model = self.exog
            while(compare):
                tvalue = sm.Logit(self.endog,final_model).fit().tvalues
                size_new = tvalue.size
                
                if (tvalue[np.abs(tvalue)>crit_val].size) != (size_new):
                    final_model = final_model[tvalue[np.abs(tvalue)>crit_val].index]
                    
                else:
                    compare = False         
            return final_model
        
    def Bivariate_treshold(self,pcrit=0.1):
       
        indices = []
        for index, name in enumerate(self.predictors):
            pval = sm.OLS(self.endog,sm.add_constant(self.exog[name])).fit().pvalues.values[0]
           
            if pval < pcrit: indices.append(name)
            
        return sm.add_constant(self.exog[indices])
        
        
        
                

class McNemar():
    ''' Perform McNemar test. Data can also be 2x2 table! Data should contain a variable called caco which indicates disease status and an independent variable. Returns matched contigency table. Data should also contain a matched set indicator. Caco, exp, ind contain the names of the columns in the dataframe. this only work if there is one case and multiple controls per indicator. Exposure indicator such that 0 is unexposed 1 exposed. Caco 0 control 1 case.
    
    Parameters
    ----------
    
    data array_like, pandas.DataFrame
    ----
    caco string
    ----
    exp string
    ---
    ind string
    ---
    
    Return pandas.DataFrame
    ------
    TODO: What happens if divide by zero when giving a table with 0 values?
    
    '''

    def __init__(self,data,caco='',exp='',ind='',alpha=0.05):
    
        if caco != "":
    
            indicators = data[ind].unique()
            
            ca_exp_co_exp = 0
            ca_not_co_exp = 0
            ca_exp_co_not = 0
            ca_not_co_not = 0
            
            for element in indicators:
                cases_exp_df = data[ data[ind] == element][[caco,exp]]
                
                # check if cases were exposed
                was_exp = cases_exp_df[ cases_exp_df[caco]==1 ][exp].values[0]
                controls_exp_arr = cases_exp_df[cases_exp_df[caco]==0][exp].values
                
                for co_exp in controls_exp_arr:
                    if (was_exp==1) & (co_exp==1): ca_exp_co_exp = ca_exp_co_exp + 1
                    elif (was_exp==1) & (co_exp==0): ca_exp_co_not = ca_exp_co_not + 1
                    elif (was_exp==0) & (co_exp==1): ca_not_co_exp = ca_not_co_exp + 1
                    else: ca_not_co_not = ca_not_co_not + 1
                    
                  
            self.calculated_values_table = np.array([[ca_exp_co_exp,ca_exp_co_not,ca_exp_co_exp+ca_exp_co_not],
                                  [ca_not_co_exp,ca_not_co_not,ca_not_co_exp+ca_not_co_not],
                                  [ca_exp_co_exp+ca_not_co_exp,ca_exp_co_not+ca_not_co_not,
                                  ca_exp_co_exp+ca_exp_co_not+ca_not_co_exp+ca_not_co_not]
                                ])
            self.values_for_calculation = self.calculated_values_table[0:2,0:2]
            self.odds_ratio = self.values_for_calculation[0,1]/self.values_for_calculation[1,0]
            
            # conduct hypothesis test H_0: p = 1/2
            
            n_discordant = self.values_for_calculation[0,1] + self.values_for_calculation[1,0]
            type_a_discordant = self.values_for_calculation[0,1]
            type_b_discordant = self.values_for_calculation[1,0]
            
            # normal approximation to binomial,
            self.h0 = False
            if n_discordant >= 20:
                self.method = 'Chi-squared'
                self.df = 1
                self.test_statistic = (np.abs(type_a_discordant-type_b_discordant)-1)**2/(type_a_discordant+type_b_discordant)
                if self.test_statistic > stats.chi2.ppf(1-alpha,1): 
                    self.h0 = True
                    self.pval = 1-stats.chi2.cdf(self.test_statistic,df=1)
                else: self.h0 = False
            # exact method
            else:
                self.method = 'Exact'
                self.df = None
                self.test_statistic = None
                if type_a_discordant < n_discordant/2:
                    self.pval = 0
                    for index in np.arange(0,type_a_discordant+1):
                        self.pval = self.pval + scipy.special.binom(n_discordant,index)*(1/2)**n_discordant
                    self.pval = 2*self.pval
                    
                elif type_a_discordant > n_discordant/2:
                    self.pval = 0
                    for index in np.arange(type_a_discordant,n_discordant+1):
                        self.pval = self.pval + scipy.special.binom(n_discordant,index)*(1/2)**n_discordant
                    self.pval = 2*self.pval 
                    
                else: self.pval = 1        
        else:
            # here is code when data contains 2x2 table
            self.odds_ratio = data[0,1]/data[1,0]
            ca_exp_co_exp = data[0,0]
            ca_not_co_exp = data[1,0]
            ca_exp_co_not = data[0,1]
            ca_not_co_not = data[1,1]
            
            self.calculated_values_table = np.array([[ca_exp_co_exp,ca_exp_co_not,ca_exp_co_exp+ca_exp_co_not],
                                  [ca_not_co_exp,ca_not_co_not,ca_not_co_exp+ca_not_co_not],
                                  [ca_exp_co_exp+ca_not_co_exp,ca_exp_co_not+ca_not_co_not,
                                  ca_exp_co_exp+ca_exp_co_not+ca_not_co_exp+ca_not_co_not]
                                ])
            self.values_for_calculation = self.calculated_values_table[0:2,0:2]
            
            # conduct hypothesis test H_0: p = 1/2
            
            n_discordant = self.values_for_calculation[0,1] + self.values_for_calculation[1,0]
            type_a_discordant = self.values_for_calculation[0,1]
            type_b_discordant = self.values_for_calculation[1,0]
            
            # normal approximation to binomial,
            self.h0 = False
            if n_discordant >= 20:
                self.method = 'Chi-squared'
                self.df = 1
                self.test_statistic = (np.abs(type_a_discordant-type_b_discordant)-1)**2/(type_a_discordant+type_b_discordant)
                if self.test_statistic > stats.chi2.ppf(1-alpha,1): 
                    self.h0 = True
                    self.pval = 1-stats.chi2.cdf(self.test_statistic,df=1)
                else: self.h0 = False
            # exact method
            else:
                self.method = 'Exact'
                self.df = None
                self.test_statistic = None
                if type_a_discordant < n_discordant/2:
                    self.pval = 0
                    for index in np.arange(0,type_a_discordant+1):
                        self.pval = self.pval + scipy.special.binom(n_discordant,index)*(1/2)**n_discordant
                    self.pval = 2*self.pval
                    
                elif type_a_discordant > n_discordant/2:
                    self.pval = 0
                    for index in np.arange(type_a_discordant,n_discordant+1):
                        self.pval = self.pval + scipy.special.binom(n_discordant,index)*(1/2)**n_discordant
                    self.pval = 2*self.pval
                else:
                    self.pval = 1
    
                
        
        
                
    def get_table_caco(self):
        index = [['Cases'],['Exposed','Unexposed','Total']]
        colum = [['Controls'],['Exposed','Unexposed','Total']]
        data = self.calculated_values_table
        return pd.DataFrame(data=data,index=pd.MultiIndex.from_product(index),
                    columns=pd.MultiIndex.from_product(colum))
    
    def get_table_pre_post(self):
        """ Returns a labeled Table for comparison of a pre and post treatment plan """

        index = [['Post'],['Yes','No','Total']]
        colum = [['Pre'],['Yes','No','Total']]
        data = self.calculated_values_table
        return pd.DataFrame(data=data,index=pd.MultiIndex.from_product(index),
                    columns=pd.MultiIndex.from_product(colum))
                    
    def summary(self):
        print(    ' McNemar\'s Test \n',
               '---------------------- \n',
               'Method    ', self.method , '\n',
               'Statistic ' ,self.test_statistic ,'\n',
               'DF        ', self.df ,'\n',
               'pvalue    ', self.pval, '\n',
               'Odds ratio', self.odds_ratio
               )
        

















    
