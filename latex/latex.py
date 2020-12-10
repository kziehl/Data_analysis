import pandas as pd
import numpy as np




def table( data, label='', caption='',index=True):
	''' Returns Latex code for a table 
Parameters:
------------

data: pandas.DataFrame that is structured like a table
label: Contains the label of the table. Later used with \\ref{} 
caption: Contains caption for the table

	'''

	envb, enve = '\\begin{table}\n' , '\end{table}'
	tab = data.to_latex(index=index).replace('\n\\toprule','').replace('\\bottomrule\n','').replace('\midrule','\hline')
	label = '\label{' +label + '}\n'
	caption = '\caption{' + caption + '}\n'

	return envb + '\centering\n' + caption + '\\begin{adjustbox}{width=\\textwidth}\n' +  tab + '\end{adjustbox}\n' + label + enve
	
