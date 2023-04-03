#!/usr/bin/env python
# coding: utf-8

# # Coffee Data Analysis and Visualization

# ## By Dyl Benson

# ## Setup

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#ML
from prophet import Prophet
sns.set(font_scale=1.5)


# ## Read in and prepare data

# In[2]:


pptg = pd.read_csv('./data/prices-paid-to-growers.csv')
rp = pd.read_csv('./data/retail-prices.csv')
cc = pd.read_csv('./data/Coffee-characteristics.csv')
cc = cc.drop(columns=['Farm.Name', 'Lot.Number', 'Certification.Address', 'Certification.Contact', 'Altitude', 'Region', 'Species', 'Mill', 'ICO.Number'])
dcc = pd.read_csv('./data/Domestic_Coffee_Consumption.csv')
dc = pd.read_csv('./data/domestic-consumption.csv')
tp = pd.read_csv('./data/total-production.csv')


# In[3]:


#define directory paths

df_paths=[
    "./data/domestic-consumption.csv",
    "./data/exports-calendar-year.csv",
    "./data/exports-crop-year.csv",
    "./data/gross-opening-stocks.csv",
    "./data/total-production.csv"
]

dfs=[pd.read_csv(df_path) for df_path in df_paths]

#define function making mean value of every column and attaching it to country
def get_means(df):
    df=df.copy()
    countries=df[df.columns[0]]
    mean=df.mean(axis=1)
    df=pd.concat([countries,mean],axis=1)
    df.columns=['country',countries.name]
    return df


# In[4]:


#define function that creates data frames
def make_df(dfs):
    
    # Process all DataFrames
    processed_dfs = []
    for df in dfs:
        processed_dfs.append(get_means(df))
        
    # Merge DataFrames
    df = processed_dfs[0]
    
    for i in range(1, len(processed_dfs)):
        df = df.merge(processed_dfs[i], on='country')
    
    return df

data=make_df(dfs)


# In[5]:


##rename columns and output to same csv (already done, doesn't need rerunning)
#df = df.rename(columns={'1990/91': '1990', '1991/92': '1991', '1992/93': '1992', '1993/94': '1993', '1994/95': '1994', '1995/96': '1995', '1996/97': '1996', '1997/98': '1997', '1998/99': '1998', '1999/00': '1999', '2000/01': '2000', '2001/02': '2001', '2002/03': '2002', '2003/04': '2003', '2004/05': '2004', '2005/06': '2005', '2006/07': '2006', '2007/08': '2007', '2008/09': '2008', '2009/10': '2009', '2010/11': '2010', '2011/12': '2011', '2012/13': '2012', '2013/14': '2013', '2014/15': '2014', '2015/16': '2015', '2016/17': '2016', '2017/18': '2017', '2018/19': '2018', '2019/20': '2019'})
#df.to_csv('Domestic_Coffee_Consumption.csv')

#Ensure no null values exist in our data
data.isna().sum() #returns False for all
data = data.dropna()

#Ensure no duplicate rows exist in our data
data.loc[data.duplicated()] #Nothing returned
data = data.drop_duplicates()

#reset data frame and index, sorting by domestic consumption
data = data.sort_values(by='domestic_consumption', ascending=False)
data = data.reset_index(drop=True)
data.head()


# ## Analysis and Visualization

# In[6]:


sns.set(rc={"figure.figsize":(40, 10)})
consume_barplot = sns.barplot(x=data['country'], y = data['domestic_consumption'])
consume_barplot.set_ylabel('Coffee Consumed (kg)')
consume_barplot.set_xlabel('Country')
consume_barplot.set_title('Domestic Coffee Consumption in Exporting Countries', fontdict={'size': 30, 'weight': 'bold'})
consume_barplot.set_xticklabels(consume_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.ylim(0,16000)
plt.show()


# In[7]:


#Create bar graph of top ten Countries by coffee consumption
top_ten_consume = data.head(10)
sns.set(rc={"figure.figsize":(40, 10)})
consume_topten_barplot = sns.barplot(x=top_ten_consume['country'], y = top_ten_consume['domestic_consumption'])
consume_topten_barplot.set_ylabel('Coffee Consumed (kg)')
consume_topten_barplot.set_xlabel('Country')
consume_topten_barplot.set_title('Domestic Coffee Consumption in Exporting Countries (Top Ten)', fontdict={'size': 30, 'weight': 'bold'})
plt.ylim(0,16000)
plt.show()


# In[8]:


#Create bar graph of top ten Countries by coffee exports
data = data.sort_values(by='exports', ascending=False)
export_barplot = sns.barplot(x=data['country'], y = data['exports'])
export_barplot.set_ylabel('Coffee Exported (kg)')
export_barplot.set_xlabel('Country')
export_barplot.set_title('Coffee Exported by Country', fontdict={'size': 30, 'weight': 'bold'})
export_barplot.set_xticklabels(consume_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.ylim(0,27000)
plt.show()


# In[9]:


#Create bar graph of top ten Countries by coffee exports
data = data.sort_values(by='exports', ascending=False)
top_ten_export = data.head(10)
export_topten_barplot = sns.barplot(x=top_ten_export['country'], y = top_ten_export['exports'])
export_topten_barplot.set_ylabel('Coffee Exported (kg)')
export_topten_barplot.set_xlabel('Country')
export_topten_barplot.set_title('Coffee Exported by Country (Top Ten)', fontdict={'size': 30, 'weight': 'bold'})
plt.ylim(0,30000)
plt.show()


# In[10]:


#Drop the outlier in the data (Brazil)
no_brazil = data.drop(data.query("country=='Brazil'").index)
no_brazil = no_brazil.sort_values(by='domestic_consumption', ascending=False)


# In[11]:


#Create same bar graphs, but excluding Brazil
top_ten_consume = no_brazil.head(10)
sns.set(rc={"figure.figsize":(40, 10)})
consume_barplot = sns.barplot(x=top_ten_consume['country'], y = top_ten_consume['domestic_consumption'], dodge=False)
consume_barplot.set_ylabel('Coffee Consumed (kg)')
consume_barplot.set_xlabel('Year')
consume_barplot.set_title('Domestic Coffee Consumption in Exporting Countries (Excluding Brazil)', fontdict={'size': 30, 'weight': 'bold'})
plt.ylim(0,3000)
plt.show()


# In[12]:


#Create bar graph of top ten Countries by coffee exports
data = data.sort_values(by='exports', ascending=False)
top_ten_export = data.head(10)
export_barplot = sns.barplot(x=top_ten_export['country'], y = top_ten_export['exports'])
export_barplot.set_ylabel('Coffee Exported (kg)')
export_barplot.set_xlabel('Year')
export_barplot.set_title('Coffee Exported by Country (Top Ten)', fontdict={'size': 30, 'weight': 'bold'})
plt.ylim(0,14000)
plt.show()


# ## Examine Correlation in the data

# In[13]:


#Create heatmap of correlated data
stats = data[['exports', 'domestic_consumption', 'exports_crop_year', 'gross_opening_stocks', 'total_production']]
sns.set_theme(style="white")
corr = stats.corr(method = 'pearson',  # The method of correlation
                  min_periods = 1 )
corr.style.background_gradient(cmap='coolwarm')


# In[14]:


#Scatter plot comparing domestic consumption x exports (excluding Brazil)
sns.set(rc={"figure.figsize":(20, 5)})
scatter = sns.scatterplot(data=no_brazil, x='domestic_consumption', y='exports', legend='auto', s=50)
scatter.set_title("Comparing Domestic Coffee Consumption with Exports Among Exporting Countries", fontdict={'size': 20, 'weight': 'bold'})
scatter.set_xlabel('Domestic Consumption (kg)', fontdict={'size': 15})
scatter.set_ylabel('Exports (kg)', fontdict={'size': 15})
plt.ylim(-1000, 14000)
plt.xlim(-100, 3000)
plt.show()


# In[15]:


#Create a grid of pairplots between domestic consumption, exports, and production
pairplot = sns.pairplot(data, vars=['domestic_consumption', 'exports', 'total_production'])
plt.show()


# In[16]:


#Create new dataframe of coffee types, clean data a bit (dcc = Domestic_Coffee_Consumption.csv)
types = dcc['Coffee type']
types = types.replace({'Robusta/Arabica':'Both'})
types = types.replace({'Arabica/Robusta':'Both'})
pie = types.value_counts()


# In[17]:


# Defining colors for the pie chart
colors = ['sienna', 'peru', 'cornsilk']
  
# Define the ratio of gap of each fragment in a tuple
x = (0.05, 0.05, 0.05)

#Create pie chart of coffee types
plt.ylabel(None)
pie.plot(kind='pie', title="Coffee Types Consumed in Exporting Countries", autopct='%1.0f%%', colors=colors, explode=x)
plt.ylabel(None)
plt.show()


# ## Examine Domestic Consumption Over Time

# In[18]:


# (dc = domestic-consumption.csv)
#sort by consumption in 2018
dc = dc.sort_values(by='2018', ascending=False)
dc = dc.reset_index(drop=True)
top_ten = dc.head(10)

#Transpose the data frame
pivot = top_ten.transpose()

#rename columns to row 1
pivot.columns = pivot.iloc[0]

#drop first two rows
pivot = pivot.iloc[3:]

#rename index
pivot.index.names = ['Year']

pivot2 = pivot.copy()

#Drop the outlier in the data (Brazil)
pivot2.drop('Brazil', axis=1, inplace=True) 

top_ten_consume_overtime = pivot2.head(10)

sns.set(rc={"figure.figsize":(30, 10)}) 

consume_plot = sns.lineplot(data=top_ten_consume_overtime, dashes=False)
consume_plot.set_title("Coffee Consumption Across Exporting Countries Over Time", fontdict={'size': 30, 'weight': 'bold'})
consume_plot.set_xlabel('Year', fontdict={'size': 15})
consume_plot.set_ylabel('Coffee Consumed (kg)', fontdict={'size': 15})
consume_plot.legend()
plt.ylim(0, 2250)
plt.show()


# ## Examine Percent change in consumption

# In[19]:


dc['total_increase'] = dc['1990']/dc['2018']*100
inf = dc['total_increase'] == np.inf
dc.loc[inf, 'total_increase'] = 0
dc["total_increase"].fillna(0, inplace = True)
#dc['total_increase'].round(decimals = 2)
dc = np.round(dc, decimals = 2)
dc = dc.sort_values(by='total_increase', ascending=False)
increase_consume_barplot = sns.barplot(x=dc['domestic_consumption'], y = dc['total_increase'])
increase_consume_barplot.set_ylabel('% Increase')
increase_consume_barplot.set_xlabel('Country')
increase_consume_barplot.set_title('Increase in Coffee Consumption among Exporting Countries between 1990 and 2018', fontdict={'size': 30, 'weight': 'bold'})
increase_consume_barplot.set_xticklabels(increase_consume_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.ylim(0,900)
plt.show()


# ## Analyze Brazil

# In[20]:


#Reset pivot and create Brazil coffee consumption dataframe
pivot.drop(pivot.columns.difference(['Brazil']), 1, inplace=True)
Brazil_consume = pivot.copy()
Brazil_consume.rename(columns={'Brazil': 'consumption'}, inplace=True)

#Create another Brazil dataframe from the production data (tp = total-production.csv)
pivot2 = tp.transpose()
pivot2.columns = pivot2.iloc[0]
pivot2 = pivot2.drop('total_production')
pivot2.index.names = ['Year']
pivot2.drop(pivot2.columns.difference(['Brazil']), 1, inplace=True)
Brazil_prod = pivot2.copy()
Brazil_prod.rename(columns={'Brazil': 'production'}, inplace=True)
Brazil_prod.head()

#Combine the two
Brazil = pd.concat([Brazil_prod, Brazil_consume], axis=1)
Brazil.head(10)


# In[21]:


#Create line graph of Brazilian coffee consumption and production over time.
sns.set(rc={"figure.figsize":(40, 10)})
brazil_consumption = sns.lineplot(data=Brazil, dashes=False)
brazil_consumption.set_title('Coffee Production x Consumption in Brazil Over Time', fontdict={'size': 30, 'weight': 'bold'})
brazil_consumption.set_xlabel('Year', fontdict={'size': 15})
brazil_consumption.set_ylabel('Coffee Consumed (billion kg)', fontdict={'size': 15})
plt.ylim(0)
plt.show()


# ## Examine Retail Prices vs Pay to Growers

# In[22]:


#sort by retail price in 2018 (rp = retail-prices.csv)
rp = rp.sort_values(by='2018', ascending=False)
rp = rp.reset_index(drop=True)
rp.head()


# In[23]:


#Clean data and plot retail prices on line graph
pivot = rp.transpose()
pivot.columns = pivot.iloc[0]
pivot = pivot.iloc[1:]
pivot.index.names = ['Year']
pivot['Years'] = pivot.index
pivot['average'] = rp.mean()
retail_prices = sns.lineplot(data=pivot)
retail_prices.set_title('Retail Price of Coffee Across Importing Countries Over Time', fontdict={'size': 30, 'weight': 'bold'})
retail_prices.set_xlabel('Year', fontdict={'size': 15})
retail_prices.set_ylabel('Price of coffee per gram', fontdict={'size': 15})
retail_prices.legend()
plt.ylim(0, 50)
plt.show()


# In[24]:


retail = rp.copy()
retail['total_increase'] = retail['1990']/retail['2018']*100
inf = retail['total_increase'] == np.inf
retail.loc[inf, 'total_increase'] = 0
retail["total_increase"].fillna(0, inplace = True)
retail = np.round(retail, decimals = 2)
retail = retail.sort_values(by='total_increase', ascending=False)
increase_rp_barplot = sns.barplot(x=retail['retail_prices'], y = retail['total_increase'])
increase_rp_barplot.set_ylabel('% Increase')
increase_rp_barplot.set_xlabel('Country')
increase_rp_barplot.set_title('Change in Retail Price of Coffee among Importing Countries between 1990 and 2018', fontdict={'size': 30, 'weight': 'bold'})
increase_rp_barplot.set_xticklabels(increase_rp_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.ylim(0,200)
plt.show()


# In[25]:


pivot2 = pptg.transpose()
pivot2.columns = pivot2.iloc[0]
pivot2 = pivot2.iloc[1:]
pivot2.index.names = ['Year']
pivot2['Years'] = pivot2.index
pivot2['average'] = pptg.mean()
grower_pay = sns.lineplot(data=pivot2)
grower_pay.set_title('Coffee Grower Pay Across Exporting Countries Over Time', fontdict={'size': 30, 'weight': 'bold'})
grower_pay.set_xlabel('Year', fontdict={'size': 15})
grower_pay.set_ylabel('Pay per gram of Coffee', fontdict={'size': 15})
grower_pay.legend()
plt.ylim(0, 6)
plt.show()


# In[26]:


pay_to_grow = pptg.copy()
pay_to_grow['total_increase'] = pay_to_grow['1990']/pay_to_grow['2018']*100
inf = pay_to_grow['total_increase'] == np.inf
pay_to_grow.loc[inf, 'total_increase'] = 0
pay_to_grow["total_increase"].fillna(0, inplace = True)
pay_to_grow = np.round(pay_to_grow, decimals = 2)
pay_to_grow = pay_to_grow.sort_values(by='total_increase', ascending=False)
increase_pptg_barplot = sns.barplot(x=pay_to_grow['prices_paid_to_growers'], y = pay_to_grow['total_increase'])
increase_pptg_barplot.set_ylabel('% Increase')
increase_pptg_barplot.set_xlabel('Country')
increase_pptg_barplot.set_title('Change in Pay To Coffee Growers among Exporting Countries between 1990 and 2018', fontdict={'size': 30, 'weight': 'bold'})
increase_pptg_barplot.set_xticklabels(increase_pptg_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.ylim(0,200)
plt.show()


# In[27]:


#Combine average price to average pay over time
pivot.drop(pivot.columns.difference(['average']), 1, inplace=True)
avg_price = pivot.copy()
avg_price.rename(columns={'average': 'avg_price'}, inplace=True)

pivot2.drop(pivot2.columns.difference(['average']), 1, inplace=True)
avg_pay = pivot2.copy()
avg_pay.rename(columns={'average': 'avg_pay'}, inplace=True)

#Combine the two
compare_price_pay = pd.concat([avg_price, avg_pay], axis=1)
compare_price_pay = pd.DataFrame(compare_price_pay)
compare_price_pay.tail()


# In[28]:


#Plot the two
sns.set(font_scale=1.5)
price_pay_plot = sns.lineplot(data=compare_price_pay, dashes=False)
price_pay_plot.set_title('Average Price of Coffee x Average Pay to Grower Over Time', fontdict={'size': 30, 'weight': 'bold'})
price_pay_plot.set_xlabel('Year', fontdict={'size': 20})
price_pay_plot.set_ylabel('Price', fontdict={'size': 20})
plt.legend(loc='upper left', labels=['Price', 'Pay'])
plt.ylim(0,20)
plt.show()


# # Coffee Characteristics

# In[29]:


# clean data (cc = Coffee-characteristics.csv)
cc = cc[pd.to_numeric(cc['ID'], errors='coerce').notnull()]
cc.head()


# In[30]:


countries = cc.copy()
countries.drop(countries.columns.difference(['Country.of.Origin']), 1, inplace=True)
country_counts = countries.value_counts()
country_counts = pd.DataFrame(country_counts)
country_counts = country_counts.reset_index()
country_counts.columns=['Origin', 'Count']

#the top 5
country_counts2 = country_counts[:15].copy()

#others
new_row = pd.DataFrame(data = {
    'Origin' : ['Other'],
    'Count' : [country_counts['Count'][15:].sum()]
})

#combining top 5 with others
pie = pd.concat([country_counts2, new_row])

sns.set(rc={"figure.figsize":(40, 12)})

#define colors
colors = ['sienna', 'saddlebrown', 'chocolate', 'sandybrown', 'peru', 'peachpuff', 'linen', 'seashell']

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return my_format

pie_chart = pie.plot(kind = 'pie', y = 'Count', labels = pie['Origin'], colors=colors, autopct=autopct_format(pie))
pie_chart.set_title('Coffee Country of Origin', fontdict={'size': 30, 'weight': 'bold'})
plt.legend([],[], frameon=False)
plt.ylabel(None)
plt.show()


# ## Predict average production over next few years

# ## Pay

# In[31]:


predict_pay = avg_pay.rename(columns={'avg_pay': 'ds'})
predict_pay['y'] = predict_pay.index
predict_pay.tail()


# In[32]:


split_date = '2018'
pay_train = predict_pay.loc[predict_pay['y'] <= split_date].copy()
pay_test = predict_pay.loc[predict_pay['y'] > split_date].copy()

# Plot train and test so you can see where we have split
pay_test = pay_test.rename(columns={'ds': 'TEST SET'}) 
pay_train = pay_train.rename(columns={'ds': 'TRAINING SET'})
pay_set = pay_train.merge(pay_test, how = 'outer')
pay_set.index = pay_set['y']
#pay_set.plot(figsize=(10, 5), title='Avg Pay over time', style='.', ms=1)
pay_set_plot = sns.scatterplot(data=pay_set, s=150)
pay_set_plot.set_title('Coffee Grower Pay Across Exporting Countries Over Time', fontdict={'size': 30, 'weight': 'bold'})
plt.ylim(0, 5)
pay_set_plot.set_xlabel('Year', fontdict={'size': 15})
pay_set_plot.set_ylabel('Pay per gram of Coffee', fontdict={'size': 15})
pay_set_plot.legend()
#plt.xlim(1990, 2018)
plt.show()


# In[33]:


# Format data for prophet model using ds and y
pay_train_prophet = pay_train.reset_index() \
    .rename(columns={'y':'ds',
                     'TRAINING SET':'y'})

pay_train_prophet.head()


# In[34]:


get_ipython().run_cell_magic('time', '', 'model = Prophet()\nmodel.fit(pay_train_prophet)\n')


# In[35]:


#Predict the future
future = model.make_future_dataframe(periods=20, freq='y', include_history=False)
forecast = model.predict(future)


# In[36]:


fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(forecast['ds'], forecast['yhat'], color='r')
fig = model.plot(forecast, ax=ax)
#ax.set_ylim(0, 70000)
plot = plt.suptitle('Predicting Future Pay to Coffee Growers')
ax.set_xlabel("Year")
ax.set_ylabel("Pay per gram of Coffee")


# ## Price

# In[37]:


predict_price = avg_price.rename(columns={'avg_price': 'ds'})
predict_price['y'] = predict_price.index
predict_price.tail()


# In[38]:


split_date = '2018'
price_train = predict_price.loc[predict_price['y'] <= split_date].copy()
price_test = predict_price.loc[predict_price['y'] > split_date].copy()

# Plot train and test so you can see where we have split
price_test = price_test.rename(columns={'ds': 'TEST SET'}) 
price_train = price_train.rename(columns={'ds': 'TRAINING SET'})
price_set = price_train.merge(price_test, how = 'outer')
price_set.index = price_set['y']
price_set_plot = sns.scatterplot(data=price_set, s=150)
price_set_plot.set_title('Coffee Grower price Across Exporting Countries Over Time', fontdict={'size': 30, 'weight': 'bold'})
plt.ylim(0, 20)
price_set_plot.set_xlabel('Year', fontdict={'size': 15})
price_set_plot.set_ylabel('price per gram of Coffee', fontdict={'size': 15})
price_set_plot.legend()
#plt.xlim(1990, 2018)
plt.show()


# In[39]:


# Format data for prophet model using ds and y
price_train_prophet = price_train.reset_index() \
    .rename(columns={'y':'ds',
                     'TRAINING SET':'y'})

price_train_prophet.head()


# In[40]:


get_ipython().run_cell_magic('time', '', 'model2 = Prophet()\nmodel2.fit(price_train_prophet)\n')


# In[41]:


#Predict the future
future2 = model2.make_future_dataframe(periods=20, freq='y', include_history=False)
forecast2 = model2.predict(future2)


# In[42]:


fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(forecast2['ds'], forecast2['yhat'], color='r')
fig = model2.plot(forecast2, ax=ax)
#ax.set_ylim(0, 70000)
plot = plt.suptitle('Predicting Future Price of Coffee')
ax.set_xlabel("Year")
ax.set_ylabel("Price per gram of Coffee")
plt.show()


# In[ ]:




