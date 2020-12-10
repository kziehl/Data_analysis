import pandas as pd
import numpy as np
import itertools











def Frequency_table( data ):

	''' Returns a Dataframe containing the category, absolute and relative frequency and the cumulative frequencies.

Parameter
---------

data: pandas.Series object with categorical values

	'''
	
	data_len = data.shape[0]
	data_pairs = list( data.value_counts().items() )
	freq_table = pd.DataFrame()
	freq_table['category'] = np.zeros( len( data_pairs ) )
	freq_table['Frequency'] = np.zeros( len( data_pairs ) )
	freq_table['Percent'] = np.zeros( len( data_pairs ) )

	for index, item in enumerate( data_pairs ):
		freq_table.loc[index,'category'] = item[0]
		freq_table.loc[index,'Frequency'] = item[1]
		freq_table.loc[index,'Percent'] = (item[1]/data_len)*100

	freq_table['Cumulative_Frequency'] = list(itertools.accumulate(freq_table['Frequency'],lambda x,y : x+y))
	freq_table['Cumulative_Percent'] = (freq_table['Cumulative_Frequency']/data_len)*100

	return freq_table

	
	
