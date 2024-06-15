import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import scipy.stats as stats

warnings.filterwarnings('ignore')
path=r'C:\Users\SOHAIB\Downloads\Bengaluru_House_Data (2).csv'

df=pd.read_csv(path)
print(df.head().to_string())

print(df.isnull().sum())
print(df.info())


df.balcony.fillna(df['balcony'].mean(),inplace=True)

print(df.select_dtypes(include=['float']).isnull().sum())
print(sns.pairplot(df))
plt.show()

for i in df.columns:
    print(df[i].value_counts())
    print('next')
num_var=['bath','balcony','price']
sns.heatmap(df[num_var].corr(),cmap='coolwarm',annot=True)
plt.show()

print('Data null in Percentage')
print(df.isnull().sum()/df.shape[0]*100)
print(sns.heatmap(df.isnull()))
plt.show()

df.drop(columns='society',inplace=True)
print('Data null in Percentage')
print(df.isnull().sum()/df.shape[0]*100)

df.dropna(inplace=True)
print(df.isnull().sum())
print(df.shape)

#feature Enginerring

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

#converting the total_sqft in numaric

print(df.total_sqft.value_counts())

total_sqft_int = []
for str_val in df['total_sqft']:
    try:
        total_sqft_int.append(float(str_val))  # if '123.4' like this value in str then conver in float
    except:
        try:
            temp = []
            temp = str_val.split('-')
            total_sqft_int.append(
                (float(temp[0]) + float(temp[-1])) / 2)  # '123 - 534' this str value split and take mean
        except:
            total_sqft_int.append(np.nan)  # if value not contain in above format then consider as nan


df['total_sqft']=total_sqft_int
print(df['total_sqft'].isnull().sum())
df.dropna(inplace=True)
print(df.shape)
df.reset_index(inplace=True)
print(df['size'].value_counts())
list_BHK=[]
for i in df['size']:
    i_spit=i.split()
    list_BHK.append(int(i_spit[0]))
df['BHK']=list_BHK
print(df.shape)
print(df.tail())
df.reset_index(inplace=True)
print(df.info())


#now finding the outliers
print('Now finding the outliers')
def diagnotic_plots(df,var):

    plt.figure(figsize=(16,9))

    #histogram
    plt.subplot(1,3,1)
    print(sns.distplot(df[var],rug=True,bins=30))
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1,3,2)
    stats.probplot(df[var],dist='norm',plot=plt)
    plt.title('Probability Plot')
    plt.ylabel('Variable quantilies')

    #box plot

    plt.subplot(1,3,3)
    sns.boxplot(df[var])
    plt.title('Box plot')

    plt.show()
num_var=['bath','balcony','total_sqft','price']
for var in num_var:
    print('_____{}_______'.format(var))
    diagnotic_plots(df,var)

print(df[df['total_sqft']/df['BHK'] <350].head())
df=df[~(df['total_sqft']/df['BHK'] <350)]
print(df.shape)


df['Price_per_sqft']=df['price']*100000/df['total_sqft']
print(df.head())
print(df.describe())

#now remvoing oultier

def remove_pps_outlies(df):
    df_out=pd.DataFrame()
    for key,val in df.groupby('location'):
        m=np.mean(val['Price_per_sqft'])
        st=np.std(val['Price_per_sqft'])
        reduced_df=val[(val.Price_per_sqft > (m-st)) & (val.Price_per_sqft <=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df=remove_pps_outlies(df)
print(df.shape)


def plot_scatter_plot(df,location):
    BHK2=df[(df['location']==location) & (df['BHK']==2)]
    BHK3=df[(df['location']==location) & (df['BHK']==3)]
    plt.figure(figsize=(16,9))
    plt.scatter(BHK2.total_sqft,BHK2.price,color='blue',label='2 BHK',s=50)
    plt.scatter(BHK3.total_sqft,BHK3.price,color='red',label='3 BHK',s=50,marker='*')
    plt.xlabel('Total sqft')
    plt.ylabel('Price')
    plt.legend()
    plt.title('location')
    plt.show()

plot_scatter_plot(df,'Rajaji Nagar')


def remove_the_outliers(df):
    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):
        bhk_stats = {}

        # Compute statistics for each BHK within the location
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['Price_per_sqft']),
                'std': np.std(bhk_df['Price_per_sqft']),
                'count': bhk_df.shape[0]
            }

        # Identify outliers within the location
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices,
                                            bhk_df[bhk_df['Price_per_sqft'] < stats['mean']].index.values)

    return df.drop(exclude_indices, axis='index')


# Apply the function to the dataframe
df=remove_the_outliers(df)
print(df.shape)
print(plot_scatter_plot(df,'Rajaji Nagar'))

print(df.bath.unique())
df=df[df.bath<df.BHK+2]
print(df.shape)
df_2=df

for i,var in enumerate(num_var):
    plt.subplot(3,2,i+1)
    sns.boxplot(df[var])
plt.show()
print(df.columns)
df.drop(columns=['area_type','availability','level_0','index','location','size'],axis=1,inplace=True)
print(df.head())
print(df.shape)
df.to_csv('Clean_data.csv',index=False)
#df_3=pd.get_dummies(df_2,drop_first=True,columns=['area_type','availability','location'])
#Categorical Variable Encoding
#print(df_3.shape)
print(df_2.columns)
df_2.drop()
