# Jump-Start Example: Python analysis of MSPA Software Survey

# Update 2017-09-21 by Tom Miller and Kelsey O'Neill
# Update 2018-06-30 by Tom Miller v005 transformation code added
# Update 2018-07-01 by John Kiley - Additional exhibits introduced

# tested under Python 3.6.1 :: Anaconda custom (x86_64)
# on Windows 10.0 and Mac OS Sierra 10.12.2 

# shows how to read in data from a comma-delimited text file
# manipuate data, create new count variables, define categorical variables,


# work with dictionaries and lambda mapping functions for recoding data 

# visualizations in this program are routed to external pdf files
# so they may be included in printed or electronic reports

# prepare for Python version 3x features and functions
# these two lines of code are needed for Python 2.7 only
# commented out for Python 3.x versions
# from __future__ import division, print_function
# from future_builtins import ascii, filter, hex, map, oct, zip

# external libraries for visualizations and data manipulation
# ensure that these packages have been installed prior to calls
import pandas as pd  # data frame operations  
import numpy as np  # arrays and math functions
import matplotlib.pyplot as plt  # static plotting
import seaborn as sns  # pretty plotting, including heat map

# correlation heat map setup for seaborn
def corr_chart(df_corr):
    corr=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, mask=top, cmap='coolwarm', 
        center = 0, square=True, 
        linewidths=.5, cbar_kws={'shrink':.5}, 
        annot = True, annot_kws={'size': 9}, fmt = '.3f')           
    plt.xticks(rotation=45) # rotate variable labels on columns (x axis)
    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)
    plt.title('Correlation Heat Map')   
    plt.savefig('plot-corr-map.pdf', 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)      

np.set_printoptions(precision=3)


# read in comma-delimited text file, creating a pandas DataFrame object
# note that IPAddress is formatted as an actual IP address
# but is actually a random-hash of the original IP address
valid_survey_input = pd.read_csv('mspa-survey-data.csv')

# use the RespondentID as label for the rows... the index of DataFrame
valid_survey_input.set_index('RespondentID', drop = True, inplace = True)

# examine the structure of the DataFrame object
print('\nContents of initial survey data ---------------')

# could use len() or first index of shape() to get number of rows/observations
print('\nNumber of Respondents =', len(valid_survey_input)) 

# show the column/variable names of the DataFrame
# note that RespondentID is no longer present
print(valid_survey_input.columns)

# abbreviated printing of the first five rows of the data frame
print(pd.DataFrame.head(valid_survey_input)) 

# shorten the variable/column names for software preference variables
survey_df = valid_survey_input.rename(index=str, columns={
    'Personal_JavaScalaSpark': 'My_Java',
    'Personal_JavaScriptHTMLCSS': 'My_JS',
    'Personal_Python': 'My_Python',
    'Personal_R': 'My_R',
    'Personal_SAS': 'My_SAS',
    'Professional_JavaScalaSpark': 'Prof_Java',
    'Professional_JavaScriptHTMLCSS': 'Prof_JS',
    'Professional_Python': 'Prof_Python',
    'Professional_R': 'Prof_R',
    'Professional_SAS': 'Prof_SAS',
    'Industry_JavaScalaSpark': 'Ind_Java',
    'Industry_JavaScriptHTMLCSS': 'Ind_JS',
    'Industry_Python': 'Ind_Python',
    'Industry_R': 'Ind_R',
    'Industry_SAS': 'Ind_SAS'})
    

# define subset DataFrame for analysis of software preferences 
software_df = survey_df.loc[:, 'My_Java':'Ind_SAS']
                     
# single scatter plot example
fig, axis = plt.subplots()
axis.set_xlabel('Personal Preference for R')
axis.set_ylabel('Personal Preference for Python')
plt.title('R and Python Preferences')
scatter_plot = axis.scatter(survey_df['My_R'], 
    survey_df['My_Python'],
    facecolors = 'none', 
    edgecolors = 'blue') 
plt.savefig('plot-scatter-r-python.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  

survey_df_labels = [
    'Personal Preference for Java/Scala/Spark',
    'Personal Preference for Java/Script/HTML/CSS',
    'Personal Preference for Python',
    'Personal Preference for R',
    'Personal Preference for SAS',
    'Professional Java/Scala/Spark',
    'Professional JavaScript/HTML/CSS',
    'Professional Python',
    'Professional R',
    'Professional SAS',
    'Industry Java/Scala/Spark',
    'Industry Java/Script/HTML/CSS',
    'Industry Python',
    'Industry R',
    'Industry SAS'        
]    

# create a set of scatter plots for personal preferences
for i in range(5):
    for j in range(5):
        if i != j:
            file_title = survey_df.columns[i] + '_and_' + survey_df.columns[j]
            plot_title = survey_df.columns[i] + ' and ' + survey_df.columns[j]
            fig, axis = plt.subplots()
            axis.set_xlabel(survey_df_labels[i])
            axis.set_ylabel(survey_df_labels[j])
            plt.title(plot_title)
            scatter_plot = axis.scatter(survey_df[survey_df.columns[i]], 
            survey_df[survey_df.columns[j]],
            facecolors = 'none', 
            edgecolors = 'blue') 
            plt.savefig(file_title + '.pdf', 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=True, pad_inches=0.25, frameon=None)  


# examine intercorrelations among software preference variables
# with correlation matrix/heat map
corr_chart(df_corr = software_df) 

# descriptive statistics for software preference variables
print('\nDescriptive statistics for survey data ---------------')
print(software_df.describe())

# descriptive statistics for one variable
print('\nDescriptive statistics for courses completed ---------------')
print(survey_df['Courses_Completed'].describe())

# ----------------------------------------------------------
# transformation code added with version v005
# ----------------------------------------------------------
# transformations a la Scikit Learn
# documentation at http://scikit-learn.org/stable/auto_examples/
#                  preprocessing/plot_all_scaling.html#sphx-glr-auto-
#                  examples-preprocessing-plot-all-scaling-py
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer


# transformations a la Scikit Learn
# select variable to examine, eliminating missing data codes
col_names = list(survey_df)
col_names = col_names[:20]
col_names
survey_df.dtypes

for i in range(len(col_names)): 
    cols = col_names[i]

    X = survey_df[[cols]].dropna()

    # Seaborn provides a convenient way to dshow the effects of transformations
    # on the distribution of values being transformed
    # Documentation at https://seaborn.pydata.org/generated/seaborn.distplot.html

    #this is the list of transformations to perform on the data
    distributions = [
            ('Unscaled data', X, 'Unscaled'),
            ('Data after standard scaling',
             StandardScaler().fit_transform(X), 'standard'),
             ('Data after min-max scaling',
              MinMaxScaler().fit_transform(X), 'min_max'),
              ('Data after max-abs scaling',
               MaxAbsScaler().fit_transform(X),'max_abs'),
               ('Data after robust scaling',
                RobustScaler(quantile_range=(25, 75)).fit_transform(X),'robust'),
                ('Data after quantile transformation (uniform pdf)',
                 QuantileTransformer(output_distribution='uniform')
                 .fit_transform(X),'quantile'),
                ('Data after quantile transformation (gaussian pdf)',
                 QuantileTransformer(output_distribution='normal')
                 .fit_transform(X),'gaussian'),
                ('Data after sample-wise L2 normalizing',
                 Normalizer().fit_transform(X),'normalizing'),
                ('Data after Log + 1', np.log(X+1), 'Log + 1'),
                ]

    #define function to produce the graphic
    def create_plts(x):
        temp = distributions[x][2]
        temp, ax = plt.subplots()
        sns.distplot(distributions[x][1]).set_title(distributions[x][0])
        temp.savefig(cols + distributions[x][0] + '.pdf', 
                     bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                     orientation='portrait', papertype=None, format=None, 
                     transparent=True, pad_inches=0.25, frameon=None)  
        ax.set(xlabel=cols, ylabel='Frequency')
    
    #loop through and print all transformations in distributions
    for q in range(len(distributions)):
        create_plts(q)

#pivot tables
#set pivot
pivot = 'Graduate_Date'

#create pivot table of my preferences for populating heatmaps
table_my = pd.pivot_table(survey_df, index=[pivot], values=[col_names[0],col_names[1],col_names[2],col_names[3],col_names[4]],
                           aggfunc=[np.mean], dropna=True, margins=True)

#create heat maps
plt.gcf().clear()
my_heat = sns.heatmap(table_my, annot=True, fmt='.2g', cmap='RdYlGn')
figure = my_heat.get_figure()    
figure.savefig('my_grad_heatmap'+ '.pdf', bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b',
               orientation='portrait', papertype=None, format=None, 
               transparent=True, pad_inches=0.25, frameon=None)
plt.show()

#create pivot table of pro preferences for populating heatmaps
table_pro = pd.pivot_table(survey_df, index=[pivot], values=[col_names[5], col_names[6],col_names[7],col_names[8],col_names[9]], 
                           aggfunc=[np.mean], dropna=True, margins=True)

#create heat maps
plt.gcf().clear()
pro_heat = sns.heatmap(table_pro, annot=True, fmt='.2g', cmap='RdYlGn')
figure = pro_heat.get_figure()    
figure.savefig('pro_grad_heatmap'+ '.pdf', bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b',
               orientation='portrait', papertype=None, format=None, 
               transparent=True, pad_inches=0.25, frameon=None)
plt.show()

#create pivot table of ind preferences for populating heatmaps
table_ind = pd.pivot_table(survey_df, index=[pivot], values=[col_names[10], col_names[11],col_names[12],col_names[13],col_names[14]], 
                           aggfunc=[np.mean], dropna=True, margins=True)

#create heat maps
plt.gcf().clear()
ind_heat = sns.heatmap(table_ind, annot=True, fmt='.2g', cmap='RdYlGn')
figure = ind_heat.get_figure()    
figure.savefig('ind_grad_heatmap'+ '.pdf', bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b',
               orientation='portrait', papertype=None, format=None, 
               transparent=True, pad_inches=0.25, frameon=None)
plt.show()

#create pivot table of class preferences for populating heatmaps
table_class = pd.pivot_table(survey_df, index=[pivot], values=[col_names[15], col_names[16],col_names[17],col_names[18]], 
                           aggfunc=[np.mean], dropna=True, margins=True)

#create heat maps
plt.gcf().clear()
class_heat = sns.heatmap(table_class, annot=True, fmt='.2g', cmap='RdYlGn')
figure = class_heat.get_figure() 
figure.savefig('class_grad_heatmap'+ '.pdf', bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b',
               orientation='portrait', papertype=None, format=None, 
               transparent=True, pad_inches=0.25, frameon=None)
plt.show()

#pivot tables
#set pivot
pivot = 'Courses_Completed'

#create pivot table of my preferences for populating heatmaps
table_my = pd.pivot_table(survey_df, index=[pivot], values=[col_names[0],col_names[1],col_names[2],col_names[3],col_names[4]],
                           aggfunc=[np.mean], dropna=True, margins=True)

#create heat maps
plt.gcf().clear()
my_heat = sns.heatmap(table_my, annot=True, fmt='.2g', cmap='RdYlGn')
figure = my_heat.get_figure()    
figure.savefig('my_course_heatmap'+ '.pdf', bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b',
               orientation='portrait', papertype=None, format=None, 
               transparent=True, pad_inches=0.25, frameon=None)
plt.show()

#create pivot table of pro preferences for populating heatmaps
table_pro = pd.pivot_table(survey_df, index=[pivot], values=[col_names[5], col_names[6],col_names[7],col_names[8],col_names[9]], 
                           aggfunc=[np.mean], dropna=True, margins=True)

#create heat maps
plt.gcf().clear()
pro_heat = sns.heatmap(table_pro, annot=True, fmt='.2g', cmap='RdYlGn')
figure = pro_heat.get_figure()    
figure.savefig('pro_course_heatmap'+ '.pdf', bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b',
               orientation='portrait', papertype=None, format=None, 
               transparent=True, pad_inches=0.25, frameon=None)
plt.show()

#create pivot table of ind preferences for populating heatmaps
table_ind = pd.pivot_table(survey_df, index=[pivot], values=[col_names[10], col_names[11],col_names[12],col_names[13],col_names[14]], 
                           aggfunc=[np.mean], dropna=True, margins=True)

#create heat maps
plt.gcf().clear()
ind_heat = sns.heatmap(table_ind, annot=True, fmt='.2g', cmap='RdYlGn')
figure = ind_heat.get_figure()    
figure.savefig('ind_course_heatmap'+ '.pdf', bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b',
               orientation='portrait', papertype=None, format=None, 
               transparent=True, pad_inches=0.25, frameon=None)
plt.show()

#create pivot table of class preferences for populating heatmaps
table_class = pd.pivot_table(survey_df, index=[pivot], values=[col_names[15], col_names[16],col_names[17],col_names[18]], 
                           aggfunc=[np.mean], dropna=True, margins=True)

#create heat maps
plt.gcf().clear()
class_heat = sns.heatmap(table_class, annot=True, fmt='.2g', cmap='RdYlGn')
figure = class_heat.get_figure() 
figure.savefig('class_course_heatmap'+ '.pdf', bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b',
               orientation='portrait', papertype=None, format=None, 
               transparent=True, pad_inches=0.25, frameon=None)
plt.show()

#number of records
survey_df.shape[0]
