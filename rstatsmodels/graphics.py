import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from itertools import combinations
from statsmodels.stats import weightstats
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: bartlett or levene test?

def ttest_ind( data , alpha = 0.05 ):

	''' 
Return results of independent t-test as pandas.DataFrame. Tests for unequal variance.

Parameter:
----------

data: DataFrame which columns are all compared with each other


	'''


	ncol = data.shape[1]
	nrows = data.shape[0]
	names = data.columns.values
	comb = list( combinations( range(0,ncol), 2 ) )
	summary = pd.DataFrame( {'Comparison':np.zeros(len(comb)),'Diff_Means':np.zeros(len(comb)),'t_stat':np.zeros(len(comb)),'t_crit':np.zeros(len(comb)),'tconfint_low':np.zeros(len(comb)),'tconfint_up':np.zeros(len(comb)),
'pval':np.zeros(len(comb)),'equal_var':np.zeros(len(comb)),'H_0':np.zeros(len(comb)),'Dof':np.zeros(len(comb)) } )

	for index, sample_comb in enumerate(comb):

		# fill in equal_var
		var1, var2 = data.iloc[:,sample_comb[0]].dropna(), data.iloc[:,sample_comb[1]].dropna()
		pval_levene = stats.levene(var1,var2,center='mean')[1] # assumes that values come from an underlying normal distribution and not skewed
		if pval_levene < alpha : summary.loc[index,'equal_var'] = False
		else: summary.loc[index,'equal_var'] = True

		# conduct t-test
		if summary.loc[index,'equal_var']: usevar='pooled'
		else: usevar='unequal'
		summary.loc[index,'t_stat'] = weightstats.ttest_ind(data.iloc[:,sample_comb[0]].dropna(),data.iloc[:,sample_comb[1]].dropna(),usevar=usevar)[0] # two-sided test
		summary.loc[index,'pval'] = weightstats.ttest_ind(data.iloc[:,sample_comb[0]].dropna(),data.iloc[:,sample_comb[1]].dropna(),usevar=usevar)[1]
		summary.loc[index,'Dof'] = weightstats.ttest_ind(data.iloc[:,sample_comb[0]].dropna(),data.iloc[:,sample_comb[1]].dropna(),usevar=usevar)[2]
		
		

		# variables to compare
		summary.loc[index,'Comparison'] = str(names[sample_comb[0]]) + '-' + str(names[sample_comb[1]])

		# means

		mean_1, mean_2 = data.iloc[:,sample_comb[0]].mean(),data.iloc[:,sample_comb[1]].mean()
		summary.loc[index,'Diff_Means'] = mean_1 - mean_2 

		# critical t-value

		if summary.loc[index,'t_stat'] < 0: summary.loc[index,'t_crit'] = stats.t.ppf(alpha/2, summary.loc[index,'Dof']-2)
		else: summary.loc[index,'t_crit'] = stats.t.ppf(1-alpha/2, summary.loc[index,'Dof']-2)
		

		# confidence interval

		cm = sms.CompareMeans(sms.DescrStatsW(data.iloc[:,sample_comb[0]].dropna()),sms.DescrStatsW( data.iloc[:,sample_comb[1]].dropna() ) )
		summary.loc[index,'tconfint_low'] = cm.tconfint_diff(alpha=alpha,usevar=usevar)[0]
		summary.loc[index,'tconfint_up'] = cm.tconfint_diff(alpha=alpha,usevar=usevar)[1]

		# H_0
		if summary.loc[index,'pval'] > alpha: summary.loc[index,'H_0'] = True
		else: summary.loc[index,'H_0'] = False

	return summary
	

def exp_dis_table(table):
	''' 2x2 Table for exposure and disease in Case-Control study. Exposure status is organized in rows
		Disease status is organized in columns 
	
	Parameters
	----------
	table: array_like, 2-D
		Assumes that exposure in rows, disease in columns
	
	Returns
	-------
		Multiindex DataFrame to visualize 2x2 Table
		
	TODO
	----
		Might be better to change the outcome to markdown
		
	'''
	
	index = [['Exposure'],['Yes','No']]
	columns = [['Disease'],['Yes','No']]
	data = np.asarray( table ) 
	
	df = pd.DataFrame(data,index=pd.MultiIndex.from_product(index),columns=pd.MultiIndex.from_product(columns))
    				 
	return df
	
	
def anova_2x2table(table, datacol):
	''' 2x2 Anova Table. One column contains individual data for different levels. Other columns contain different levels of ANOVA.
	Data should be in the following form:
	
	--------------------------
	--FactorA--FactorB--Data--
	--   1   --  1   -- 0.21--
	--   1   --  1   -- 0.22--
	--   1	 --  2   -- xx  --
	--   2   --  1   -- xx  --
	--------------------------
	
	Parameters
	-----------
	table: array_like, DataFrame
		Different levels of each factor organized in columns. 
	
	datacol: array_like, 1-D
		Means of different levels. Datacolumn from table.
		
	Returns
	-------
		Multiindex DataFrame to visualize Anova
		
	TODO
	----
	maybe add column and row means, and overall mean to output
		
	'''
	
	# reorder
	combinations = pd.Series( zip(table.iloc[:,0],table.iloc[:,1]) ).unique()
	vec_mean = []
	for element in combinations:
		new = ( table[[table.iloc[:,0].name,table.iloc[:,1].name]] == element )
		mean = datacol[new.all(axis=1)].mean()
		vec_mean.append(mean)
	factorA = table.iloc[:,0].unique()
	factorB = table.iloc[:,1].unique()
	arrays = [list(factorA),list(factorB)]
	index = pd.MultiIndex.from_product(arrays)
	df_means = pd.DataFrame({'Means':vec_mean},index=index) # this is a interim dataframe which is grouby by Factor A and Factor B and Means
	
	# put data into a human-readable format
	anova_levels = table.drop(columns=datacol.name)
	index = [[anova_levels.columns[0]], anova_levels.iloc[:,0].unique() ]
	columns = [[anova_levels.columns[1]], anova_levels.iloc[:,1].unique() ]
	n_rows = anova_levels.iloc[:,0].unique().size 
	n_cols = anova_levels.iloc[:,1].unique().size
	data = np.asarray(vec_mean).reshape(n_rows, n_cols, order='C')
	
	
	df = pd.DataFrame(data,index=pd.MultiIndex.from_product(index),columns=pd.MultiIndex.from_product(columns))
	# add column and row mean
	#df[ (table.iloc[:,1].name,'Row_mean') ] = df.mean(axis=1).values
	#df[ (table.iloc[:,0].name,'Col_mean') ] = df.mean(axis=0).values
	
	
	return df
	
	
def main_effects( table ):
		''' Table is a 2way Anova table. Calculate the main effects of Factor A and Factor B and interaction effect. Factor A is organized in rows, Factor B is organized in Columns. The main effects of a factor are the differences between factor level means and overall mean. Interaction effect is the differnece between observed cell mean and expected under additivity. Table should be in the form:
			factor b
			1	2
factora	1	0.359490	-0.676666
		2	0.357976	-0.457606
		3	0.498626	-0.397681
		
		Parameters
		----------
		table: array_like, DataFrame
		
		Returns
		-------
		2 arrays with main effects, 1 array with interaction effects
		
		'''
		df = table
		row_means = df.mean(axis=1).values
		col_means = df.mean(axis=0).values	
		grand_mean = df.mean().mean()
		main_effects_A = row_means - grand_mean
		main_effects_B = col_means - grand_mean
		interaction_effect = np.zeros_like(df.values)
		for i,row in enumerate(main_effects_A):
			for j,col in enumerate(main_effects_B):
				interaction_effect[i,j] = df.values[i,j] - (grand_mean + row + col)
				
				
		
		return main_effects_A, main_effects_B, interaction_effect
		
def mean_plot( table ):
			''' Display the mean plot for 2x2 Anova table. Means plots are a way of displaying the means by levels of the two factos. Parallel lines indicate that effects of two factors are additive meaning that cell mean are equal to grand mean plus main effects of each factor. Factor A is on x-axis.
			
			Parameters
			----------
			table: array_like, DataFrame
			
			Returns
			-------
			axis for means plot
			
			'''
			df=table
			fig, ax = plt.subplots(figsize=(18,10))
			fig.suptitle('Mean plot')
			ax.set_xlabel(df.index[0][0],labelpad=15)
			ax.set_ylabel('Cell Means',labelpad=15)
			factorA_levels = np.asarray( list( zip(*df.index) )[1] )
			factorB_levels = np.asarray( list( zip(*df.columns) ) )
			n_lines = df.values.shape[1]
			data = df.values
			plt.xticks(factorA_levels)
			
			for index, level in enumerate(np.asarray( list( zip(*df.columns) )[1] )):
				y = data[:,index]
				sns.lineplot(factorA_levels,y,ax=ax,marker='o',label=str(level))
				
			plt.legend(title=df.columns[0][0],loc='best',labels=np.asarray( list( zip(*df.columns) )[1] ) )
				
				
			
			return ax
			
			
def anova2x2_constraints( table ):

	main_effects_A, main_effects_B, interaction_effect = main_effects( table )
	
	factorA = main_effects_A.mean() == 0
	factorB = main_effects_B.mean() == 0
	inter_row = np.all( interaction_effect.mean(axis=0) == 0)
	inter_col = np.all( interaction_effect.mean(axis=1) == 0)
	
	if ( factorA & factorB & inter_row & inter_col ):
		print('Constraints for factor effects model are fulfilled')
	else:
		print( main_effects_A.mean() ,'\n', main_effects_B.mean(), '\n',
			 interaction_effect.mean(axis=0), '\n', interaction_effect.mean(axis=1) )

	
				
				
				
			
		



		

		




