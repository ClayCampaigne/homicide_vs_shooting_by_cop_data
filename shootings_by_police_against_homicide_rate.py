#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True)


# In[ ]:


# Compare killings by police vs intentional homicide rate, across countries


# In[ ]:


homicide_path = 'homicide_data_by_country.csv'
shooting_by_police_path = 'police_killings_by_country.csv'
# https://worldpopulationreview.com/country-rankings/police-killings-by-country
# https://www.indexmundi.com/facts/indicators/VC.IHR.PSRC.P5/rankings


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
h = pd.read_csv(homicide_path)
s = pd.read_csv(shooting_by_police_path)
s.columns = [elem.replace('country', 'Country') for elem in s.columns]
data_by_country = pd.merge(h,s, on='Country', how='inner')

data_by_country.loc[:,'ratePer10M'] = data_by_country.loc[:,'ratePer10M'] + .001

data_by_country['log_homicide_rate'] = np.log(data_by_country.loc[:,'homicide rate'])
data_by_country['log_ratePer10M'] = np.log(data_by_country.loc[:,'ratePer10M'])

fig, ax = plt.subplots(figsize=(20,12))
ax.scatter(x=np.log(data_by_country.loc[:,'homicide rate']),
           y=np.log(data_by_country.loc[:,'ratePer10M']),
           label = data_by_country['Country'])

for i, txt in enumerate(data_by_country['Country']):
    plt.annotate(txt, (np.log(data_by_country.loc[i,'homicide rate']), np.log(data_by_country.loc[i, 'ratePer10M'])))

sns.regplot(x="log_homicide_rate", y="log_ratePer10M", data=data_by_country, ax=ax)
    
fig.savefig('pofat_vs_hom_by_country.png')


# In[ ]:


fig, ax = plt.subplots(figsize=(20,12))
ax.scatter(x=data_by_country['homicide rate'],
           y=data_by_country['ratePer10M'],
           label = data_by_country['Country'])


for i, txt in enumerate(data_by_country['Country']):
    plt.annotate(txt, ((data_by_country.loc[i,'homicide rate']), (data_by_country.loc[i, 'ratePer10M'])))

sns.regplot(x="homicide rate", y="ratePer10M", data=data_by_country, ax=ax)    

fig.savefig('pofat_vs_hom_by_country_nolog.png')


# In[ ]:


data_by_country[['homicide rate', 'ratePer10M']].corr()


# In[ ]:


np.log(data_by_country[['homicide rate', 'ratePer10M']]).corr()


# In[ ]:





# # Same comparison across states

# In[ ]:



us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
    'United States': 'ALL'
}


# In[ ]:


police_fatality_wapo_p = 'fatal-police-shootings-data-wapo.csv'
homicide_by_state_p = 'homicide_by_state_wikipedia.csv'
# https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_intentional_homicide_rate
# https://data.world/awram/us-police-involved-fatalities


# In[ ]:


# state abbreviation dictionary: https://gist.github.com/rogerallen/1583593

police_fatality = pd.read_csv(police_fatality_wapo_p, parse_dates = ['date'])
police_fatality['year'] = police_fatality['date'].dt.year
gb = police_fatality.groupby(['year', 'state']).count()
gb = gb.iloc[:,0]

po_fat = gb.unstack(['year'])
po_fat.columns = [str(elem) for elem in po_fat.columns]
po_fat = po_fat.reset_index()
po_fat['comp_years_pofat'] = po_fat[['2015', '2016', '2017', '2018', '2019']].sum(axis=1)

h_by_s = pd.read_csv(homicide_by_state_p)
h_by_s['state'] = h_by_s['state'].str.strip()
h_by_s['Murder Count'] = h_by_s['Murder Count'].str.replace(',','').astype(int)
h_by_s['relpop'] = h_by_s['Murder Count']/h_by_s['2019']
h_by_s['state'] = h_by_s['state'].apply(lambda entry: us_state_abbrev[entry])
h_by_s['comp_years_homicide'] = h_by_s[['2019', '2018', '2017', '2016', '2015']].sum(axis=1)

data_by_state = pd.merge(h_by_s[['state', 'comp_years_homicide', 'relpop']],
                po_fat[['state', 'comp_years_pofat']],
                on = 'state',
                how = 'inner')
data_by_state['pofat_rate'] = data_by_state['comp_years_pofat']/data_by_state['relpop']

data_by_state['log_comp_years_homicide'] = np.log(data_by_state['comp_years_homicide'])
data_by_state['log_pofat_rate'] = np.log(data_by_state['pofat_rate'])

fig, ax = plt.subplots(figsize=(20,12))
ax.scatter(x=np.log(data_by_state['comp_years_homicide']),
           y=np.log(data_by_state['pofat_rate']))

for i, txt in enumerate(data_by_state['state']):
    plt.annotate(txt, (np.log(data_by_state.loc[i,'comp_years_homicide']), np.log(data_by_state.loc[i, 'pofat_rate'])))

sns.regplot(x="log_comp_years_homicide", y="log_pofat_rate", data=data_by_state, ax=ax) 

fig.savefig('pofat_vs_hom_by_US_state.png')


# In[ ]:


data_by_state[['comp_years_homicide', 'pofat_rate']].corr()


# In[ ]:


np.log(data_by_state[['pofat_rate','comp_years_homicide']]).corr()


# In[ ]:



fig, ax = plt.subplots(figsize=(20,12))
ax.scatter(x=(data_by_state['comp_years_homicide']),
           y=(data_by_state['pofat_rate']))

for i, txt in enumerate(data_by_state['state']):
    plt.annotate(txt, ((data_by_state.loc[i,'comp_years_homicide']), (data_by_state.loc[i, 'pofat_rate'])))
sns.regplot(x="comp_years_homicide", y="pofat_rate", data=data_by_state, ax=ax) 

fig.savefig('pofat_vs_hom_by_US_staten_nolog.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




