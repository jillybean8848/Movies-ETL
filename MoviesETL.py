#!/usr/bin/env python
# coding: utf-8

# In[117]:


import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import time
from config import db_password


# In[6]:


file_dir="C:/Users/jilly/OneDrive/Documents/Class Activites/Mod 8 and 9/"


# In[7]:


f'{file_dir}wikipedia-movies.json'


# In[8]:


with open(f'{file_dir}/wikipedia-movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)


# In[9]:


len(wiki_movies_raw)


# In[10]:


kaggle_metadata = pd.read_csv(f'{file_dir}movies_metadata.csv', low_memory=False)
ratings = pd.read_csv(f'{file_dir}ratings.csv')


# In[11]:


# First 5 records
wiki_movies_raw[:5]


# In[12]:


ratings.head()


# In[13]:


kaggle_metadata.head()


# In[14]:


wiki_movies_df = pd.DataFrame(wiki_movies_raw)
wiki_movies_df.head()


# In[15]:


len(wiki_movies_df.columns.tolist())


# In[16]:


wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie]
len(wiki_movies)


# In[17]:


wiki_movies_df1 = pd.DataFrame(wiki_movies)
wiki_movies_df1.head()


# In[18]:


wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]

wiki_movies_df2 = pd.DataFrame(wiki_movies)
wiki_movies_df2.head()


# In[19]:


def clean_movie(movie):
    movie = dict(movie) 
    #create a non-destructive copy
    return movie


# In[20]:


wiki_movies_df[wiki_movies_df['Arabic'].notnull()]


# In[21]:


wiki_movies_df[wiki_movies_df['Arabic'].notnull()]['url']


# In[22]:


sorted(wiki_movies_df.columns.tolist())


# In[23]:


#  Make an empty dict to hold all of the alternative titles.
def clean_movie(movie):
   movie = dict(movie) 
   
   #create a non-destructive copy
   alt_titles = {}
   
   # Loop through a list of all alternative title keys
   # combine alternate titles into one list

   for key in ['Also known as','Arabic','Cantonese','Chinese','French',
               'Hangul','Hebrew','Hepburn','Japanese','Literally',
               'Mandarin','McCune–Reischauer','Original title','Polish',
               'Revised Romanization','Romanized','Russian',
               'Simplified','Traditional','Yiddish']:
       
       # Check if the current key exists in the movie object.
       if key in movie:
           
           # If so, remove the key-value pair and add to the alternative titles dictionary.
           alt_titles[key] = movie[key]
           movie.pop(key)
           
   # After looping through every key, add the alternative titles dict to the movie object.
   if len(alt_titles) > 0:
       movie['alt_titles'] = alt_titles
     
   # merge column names
   def change_column_name(old_name, new_name):
       if old_name in movie:
           movie[new_name] = movie.pop(old_name)
   change_column_name('Adaptation by', 'Writer(s)')
   change_column_name('Country of origin', 'Country')
   change_column_name('Directed by', 'Director')
   change_column_name('Distributed by', 'Distributor')
   change_column_name('Edited by', 'Editor(s)')
   change_column_name('Length', 'Running time')
   change_column_name('Original release', 'Release date')
   change_column_name('Music by', 'Composer(s)')
   change_column_name('Produced by', 'Producer(s)')
   change_column_name('Producer', 'Producer(s)')
   change_column_name('Productioncompanies ', 'Production company(s)')
   change_column_name('Productioncompany ', 'Production company(s)')
   change_column_name('Released', 'Release Date')
   change_column_name('Release Date', 'Release date')
   change_column_name('Screen story by', 'Writer(s)')
   change_column_name('Screenplay by', 'Writer(s)')
   change_column_name('Story by', 'Writer(s)')
   change_column_name('Theme music composer', 'Composer(s)')
   change_column_name('Written by', 'Writer(s)')
   
   return movie


# In[24]:


# make a list of cleaned movies with a list comprehension
clean_movies = [clean_movie(movie) for movie in wiki_movies]

# Create a DataFrame from clean_movies and print the columns as a list
wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())


# In[25]:


# Extract movie IDs and add them to a new column in the DataFrame
wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))


# In[26]:


# Drop any duplicates of movie IDs
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
wiki_movies_df.head()


# In[27]:


# get the count of null values for each column
[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]


# In[28]:


# make a list of columns that have less than 90% null values
wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]

# select the columns from the wiki movies DataFrame
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

len(wiki_movies_df)
wiki_movies_df.head(4)


# In[29]:


# make a data series that drops missing values 
box_office = wiki_movies_df['Box office'].dropna()
len(box_office)


# In[30]:


# create a is_not_a_string function
def is_not_a_string(x):
    return type(x) != str
box_office[box_office.map(is_not_a_string)]


# In[31]:


# create lambda function to identify values that are not a string
box_office[box_office.map(lambda x: type(x) != str)]


# In[32]:


box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)
box_office


# In[33]:


# create a variable (form_one) that contains the form $123.4 million/billion
form_one = r'\$\d+\.?\d*\s*[mb]illion'


# In[34]:


# count how many box office values match form_one
box_office.str.contains(form_one, flags=re.IGNORECASE).sum()


# In[35]:


# create a variable (form_two) that contains the form $123,456,789
form_two = r'\$\d{1,3}(?:,\d{3})+'

# count how many box office values match form_two
box_office.str.contains(form_two, flags=re.IGNORECASE).sum()


# In[36]:


# create a list of values that match the variables
matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)

#return box office values that aren't described by either form_one or form_two
box_office[~matches_form_one & ~matches_form_two]


# In[38]:


def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan
    
    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):
        
        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)
        
        # convert to float and multiply by a million
        value = float(s) * 10**6
        
        # return value
        return value
    
    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):
        
        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)
        
        # convert to float and multiply by a billion
        value = float(s) * 10**9
        
        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):
        
        # remove dollar sign and commas
        s = re.sub('\$|,','', s)
        
        # convert to float
        value = float(s)
        
        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan


# In[39]:


# From the wiki movies df, extract the form_one and form_two matches from the box_office column and parse the values into numeric values
wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

sorted(wiki_movies_df.columns.tolist())


# In[40]:


wiki_movies_df['box_office']


# In[ ]:


# Drop the orgininal Box office column
wiki_movies_df.drop('Box office', axis=1, inplace=True)


# In[43]:


# Create a budget variable
budget = wiki_movies_df['Budget'].dropna()

# Convert any lists to strings:
budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)

# remove any values between a dollar sign and a hyphen (for budgets given in ranges):
budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[44]:


# create a list of values that match the variables for budget
matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)

# return budget values that aren't described by either form_one or form_two
budget[~matches_form_one & ~matches_form_two]


# In[45]:


# Remove the citation references

budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]


# In[46]:


# From the wiki movies df, extract the form_one and form_two matches from the budget column and parse the values into numeric values
wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

sorted(wiki_movies_df.columns.tolist())


# In[47]:


# From the wiki movies df, extract the form_one and form_two matches from the budget column and parse the values into numeric values
wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

sorted(wiki_movies_df.columns.tolist())


# In[48]:


# Drop the orgininal budget column
wiki_movies_df.drop('Budget', axis=1, inplace=True)
sorted(wiki_movies_df.columns.tolist())


# In[49]:


#Make a variable that holds the non-null values of Release date in the DataFrame, converting lists to strings:

release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[50]:


# Create a list of formats that match the variables for release date
date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'


# In[51]:


# Extract the dates
release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)


# In[56]:


wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)


# In[57]:


#Make a variable that holds the non-null values of running time in the DataFrame, converting lists to strings:

running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
running_time


# In[59]:


#How many running times match the running_time format, i.e. 100 minutes.
running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE, na=False).sum()


# In[60]:


running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE, na=False) != True]


# In[61]:


running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE, na=False).sum()


# In[62]:


running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE, na=False) != True]


# In[63]:


# capture some of the remaining entries by relaxing the condition
running_time[running_time.str.contains(r'\d*\s*m', flags=re.IGNORECASE) != True]


# In[66]:


#Extract the digits for running time
running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')


# In[67]:


#Convert the running time data to numeric values
running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)


# In[68]:


#Save the running time data to the dataframe
wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)


# In[69]:


#Drop Running time from the dataset with the following code:
wiki_movies_df.drop('Running time', axis=1, inplace=True)


# In[70]:


sorted(wiki_movies_df.columns.tolist())


# In[71]:


#Check that all of the columns came in as the correct data types.
kaggle_metadata.dtypes


# In[72]:


#Check that all the values are either True or False for the "adult" and "video" columns.
kaggle_metadata['adult'].value_counts()


# In[73]:


# Remove bad data
kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]


# In[74]:


#Keep rows where adult is 'False' and drop the adult column.
kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')


# In[75]:


# look at the values of the video column:
kaggle_metadata['video'].value_counts()


# In[76]:


#Convert the data tpe for the video column
kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'


# In[77]:


#Convert numeric columns to the proper datatype
kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')


# In[78]:


#Convert release date to datetime
kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])


# In[79]:


#Set the null_counts option to True
ratings.info(null_counts=True)


# In[80]:


#Convert the rating data to a datetime datatype
pd.to_datetime(ratings['timestamp'], unit='s')


# In[81]:


#Ratings histogram
pd.options.display.float_format = '{:20,.2f}'.format
ratings['rating'].plot(kind='hist')

ratings['rating'].describe()


# In[82]:


#Identify repeating columns
movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])
movies_df


# In[83]:


# Inspect the titles from the wiki and kaggle data
movies_df[['title_wiki','title_kaggle']]


# In[84]:


# Inspect the titles from the wiki and kaggle data that don't match
movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']]


# In[85]:


#Show any rows where title_kaggle is null
movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]


# In[86]:


#Fill in missing runtime values with zero and make the scatter plot:

movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')


# In[87]:


#Fill enpty budget values with zero and make a scatter plot:
movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')


# In[88]:


#Fill in missing box office values with zero and make a scatter plot:
movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')


# In[89]:


#Filter the scatter plot for everything less than $1 billion in box_office.
movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')


# In[90]:


#Release date scatter plot
movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')


# In[91]:


# look for any movie whose release date according to Wikipedia is after 1996, but 
# whose release date according to Kaggle is before 1965:
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]

# get index for movies that fit the criteria
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index

# drop rows that fit the criteria
movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)


# In[92]:


# check if any null values exist in the release date data
movies_df[movies_df['release_date_wiki'].isnull()]


# In[94]:


#Convert Language to tuples
movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)


# In[95]:


#Collect the value counts of each row in the original_language column
movies_df['original_language'].value_counts(dropna=False)


# In[96]:


#Review production company data
movies_df[['Production company(s)','production_companies']]


# In[97]:


# Competing data:
# Wiki                     Movielens                Resolution
#--------------------------------------------------------------------------
# title_wiki               title_kaggle             Drop Wikipedia.
# running_time             runtime                  Keep kaggle; fill in zeros with Wikipedia data.
# budget_wiki              budget_kaggle            Keep Kaggle; fill in zeros with Wikipedia data.
# box_office               revenue                  Keep Kaggle; fill in zeros with Wikipedia data.
# release_date_wiki        release_date_kaggle      Drop Wikipedia.
# Language                 original_language        Drop Wikipedia.
# Production company(s)    production_companies     Drop Wikipedia.


# In[98]:


# drop the title_wiki, release_date_wiki, Language, and Production company(s) columns.
movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)


# In[99]:


# make a function that fills in missing data for a column pair and then drops the redundant column.

def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)
    
# run the function for the three column pairs
fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df


# In[101]:


# check for columns wiht only one value

for col in movies_df.columns:
    # convert lists to tuples
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)
        
movies_df['video'].value_counts(dropna=False)


# In[102]:


sorted(wiki_movies_df.columns.tolist())


# In[103]:


#reorder the dataframe columns to make the dataset easier to read
movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]

# rename dataframe columns 
movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)

movies_df


# In[104]:


#Used groupby on the "movieId" and "rating" columns and coutn.
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)
rating_counts


# In[105]:


# pivot data so that movieId is the index, the columns will be all the rating values, and the rows will be the counts for each rating value.
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)                 .pivot(index='movieId',columns='rating', values='count')


# In[106]:


#Rename columns
rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]


# In[107]:


#Merge rating_counts into movies_df
movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

#Fill missing values 
movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)


# In[109]:


#Create database engine to communicate with SQL server
"postgres://[user]:[password]@[location]:[port]/[database]"

#Connection string for local server
db_string = f"postgres://postgres:@127.0.0.1:5432/movie_data"


# In[112]:


# create the database engine
engine = create_engine


# In[115]:


# save the movies_df DataFrame to a SQL table
movies_df.to_sql(name="movies", con=engine)


# In[114]:


#Import the CSV using the chunksize= parameter in read_csv()

#Print number of imported rows

# create a variable for the number of rows imported
rows_imported = 0

# get the start_time from time.time()
start_time = time.time()

for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):

    # print out the range of rows that are being imported
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')

    data.to_sql(name='ratings', con=engine, if_exists='append')

    # increment the number of rows imported by the size of 'data'
    rows_imported += len(data)

    # add elapsed time to final print out
    # print that the rows have finished importing
    print(f'Done. {time.time() - start_time} total seconds elapsed')

