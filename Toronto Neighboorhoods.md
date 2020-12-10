# Segmenting and Clustering Neighborhoods in Toronto

## Introduction

In this assignment, I will be required to explore, segment, and cluster the neighborhoods in the city of Toronto. However, unlike New York, the neighborhood data is not readily available on the internet. What is interesting about the field of data science is that each project can be challenging in its unique way, so I need to learn to be agile and refine the skill to learn new libraries and tools quickly depending on the project.



```python
#conda update -n base -c defaults conda
```


```python
#Downoad the packages and dependencies
import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from bs4 import BeautifulSoup 

import json # library to handle JSON files

!pip install  geopy  
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

!pip install folium==0.5
import folium # map rendering library

print('Libraries imported.')


```

    Requirement already satisfied: geopy in c:\programdata\anaconda3\lib\site-packages (2.0.0)
    Requirement already satisfied: geographiclib<2,>=1.49 in c:\programdata\anaconda3\lib\site-packages (from geopy) (1.50)
    Requirement already satisfied: folium==0.5 in c:\programdata\anaconda3\lib\site-packages (0.5.0)
    Requirement already satisfied: jinja2 in c:\programdata\anaconda3\lib\site-packages (from folium==0.5) (2.11.1)
    Requirement already satisfied: six in c:\programdata\anaconda3\lib\site-packages (from folium==0.5) (1.14.0)
    Requirement already satisfied: branca in c:\programdata\anaconda3\lib\site-packages (from folium==0.5) (0.4.1)
    Requirement already satisfied: requests in c:\programdata\anaconda3\lib\site-packages (from folium==0.5) (2.22.0)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\programdata\anaconda3\lib\site-packages (from jinja2->folium==0.5) (1.1.1)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\programdata\anaconda3\lib\site-packages (from requests->folium==0.5) (1.25.8)
    Requirement already satisfied: certifi>=2017.4.17 in c:\programdata\anaconda3\lib\site-packages (from requests->folium==0.5) (2019.11.28)
    Requirement already satisfied: idna<2.9,>=2.5 in c:\programdata\anaconda3\lib\site-packages (from requests->folium==0.5) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in c:\programdata\anaconda3\lib\site-packages (from requests->folium==0.5) (3.0.4)
    Libraries imported.
    

## 1. Import and Explore the Dataset


```python
source = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M").text
soup = BeautifulSoup(source, 'lxml')

table = soup.find("table")
table_rows = table.tbody.find_all("tr")

res = []
for tr in table_rows:
    td = tr.find_all("td")
    row = [tr.text for tr in td]
    
    # Only process the cells that have an assigned borough. Ignore cells with a borough that is Not assigned.
    if row != [] and row[1] != "Not assigned\n":
        # If a cell has a borough but a "Not assigned" neighborhood, then the neighborhood will be the same as the borough.
        if "Not assigned" in row[2]: 
            row[2] = row[1]
        res.append(row)

# Dataframe with 3 columns
df = pd.DataFrame(res, columns = ["PostalCode", "Borough", "Neighborhood"])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A\n</td>
      <td>North York\n</td>
      <td>Parkwoods\n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A\n</td>
      <td>North York\n</td>
      <td>Victoria Village\n</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A\n</td>
      <td>Downtown Toronto\n</td>
      <td>Regent Park, Harbourfront\n</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A\n</td>
      <td>North York\n</td>
      <td>Lawrence Manor, Lawrence Heights\n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M7A\n</td>
      <td>Downtown Toronto\n</td>
      <td>Queen's Park, Ontario Provincial Government\n</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Remove "\n" at the end of each string in the Neighborhood column
df["Neighborhood"] = df["Neighborhood"].str.replace("\n","")
df["PostalCode"] = df["PostalCode"].str.replace("\n","")
df["Borough"] = df["Borough"].str.replace("\n","")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Group by Neighborhood
df = df.groupby(["PostalCode", "Borough"])["Neighborhood"].apply(", ".join).reset_index()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.shape)
```

    (103, 3)
    

## 2. Latitude and Longitude


```python
df_geo_coor = pd.read_csv("https://cocl.us/Geospatial_data")
df_geo_coor.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Merge the geodf with the initial df
df_toronto = pd.merge(df, df_geo_coor, how='left', left_on = 'PostalCode', right_on = 'Postal Code')
# remove the "Postal Code" column
df_toronto.drop("Postal Code", axis=1, inplace=True)
df_toronto.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Clustering Toronto Neighboorhoods

### 3a. Lat and Long for Toronto


```python
address = "Toronto, ON"

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The coordinates for Toronto are {}, {}.'.format(latitude, longitude))
```

    The coordinates for Toronto are 43.6534817, -79.3839347.
    

### 3b. Map of Toronto with the Neighborhood data overlaid


```python
# Toronto map with lat and long values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)
map_toronto

```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMycsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzNDgxNywtNzkuMzgzOTM0N10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfNTZiODAxODhiNWMwNDRmNGI1MzllZmRmMTQ3ZGQzNWQgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgIAo8L3NjcmlwdD4= onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
#add map markers
for lat, lng, borough, neighborhood in zip(
        df_toronto['Latitude'], 
        df_toronto['Longitude'], 
        df_toronto['Borough'], 
        df_toronto['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  

map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMycsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzNDgxNywtNzkuMzgzOTM0N10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfNTZiODAxODhiNWMwNDRmNGI1MzllZmRmMTQ3ZGQzNWQgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI5NjI2ZTQzZGQyMTQ2YjU4OGFhZmZmNjQwNThlZGUwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODA2Njg2Mjk5OTk5OTk2LC03OS4xOTQzNTM0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lNmIyZTZlZjQxNjc0ZDc0YWYzMDhmZDJkZDUxZGQ3ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iOTRjODk4ZTM0ODI0MGRlYjIzZjU5MDhjYmUwYTYyOSA9ICQoJzxkaXYgaWQ9Imh0bWxfYjk0Yzg5OGUzNDgyNDBkZWIyM2Y1OTA4Y2JlMGE2MjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1hbHZlcm4sIFJvdWdlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTZiMmU2ZWY0MTY3NGQ3NGFmMzA4ZmQyZGQ1MWRkN2Uuc2V0Q29udGVudChodG1sX2I5NGM4OThlMzQ4MjQwZGViMjNmNTkwOGNiZTBhNjI5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI5NjI2ZTQzZGQyMTQ2YjU4OGFhZmZmNjQwNThlZGUwLmJpbmRQb3B1cChwb3B1cF9lNmIyZTZlZjQxNjc0ZDc0YWYzMDhmZDJkZDUxZGQ3ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNDI4ZDI2OWRhZjA0ZGFjYWNjNmNlMjI0NGM0ZWE2MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4NDUzNTEsLTc5LjE2MDQ5NzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRmMzRmZWVkNWFlODQ2MzBiMDlhY2I1YzFhNDZkYmNhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I3NmFkZjI1YzM3ZDQ5ZjlhMGJkNGQ5MjVkZTdiNjc2ID0gJCgnPGRpdiBpZD0iaHRtbF9iNzZhZGYyNWMzN2Q0OWY5YTBiZDRkOTI1ZGU3YjY3NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um91Z2UgSGlsbCwgUG9ydCBVbmlvbiwgSGlnaGxhbmQgQ3JlZWssIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80ZjM0ZmVlZDVhZTg0NjMwYjA5YWNiNWMxYTQ2ZGJjYS5zZXRDb250ZW50KGh0bWxfYjc2YWRmMjVjMzdkNDlmOWEwYmQ0ZDkyNWRlN2I2NzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDQyOGQyNjlkYWYwNGRhY2FjYzZjZTIyNDRjNGVhNjAuYmluZFBvcHVwKHBvcHVwXzRmMzRmZWVkNWFlODQ2MzBiMDlhY2I1YzFhNDZkYmNhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFhODE2NTE5OWJlODQyNzJiNDM2NTQ0Y2E2MGEwMTNkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzYzNTcyNiwtNzkuMTg4NzExNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNjNjYjFhNDM1NGI0NzYyYmU4YWU0OTVhN2QyNTU1MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81NzI5MWE3NmFkZjg0YTRiODNiM2M2ZmUzNTZjZDM1MSA9ICQoJzxkaXYgaWQ9Imh0bWxfNTcyOTFhNzZhZGY4NGE0YjgzYjNjNmZlMzU2Y2QzNTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkd1aWxkd29vZCwgTW9ybmluZ3NpZGUsIFdlc3QgSGlsbCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M2M2NiMWE0MzU0YjQ3NjJiZThhZTQ5NWE3ZDI1NTUwLnNldENvbnRlbnQoaHRtbF81NzI5MWE3NmFkZjg0YTRiODNiM2M2ZmUzNTZjZDM1MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xYTgxNjUxOTliZTg0MjcyYjQzNjU0NGNhNjBhMDEzZC5iaW5kUG9wdXAocG9wdXBfYzYzY2IxYTQzNTRiNDc2MmJlOGFlNDk1YTdkMjU1NTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTUxYmQ5NjFiNzY4NDZiYThlZGNlNDdkMTE3NGM1N2MgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzA5OTIxLC03OS4yMTY5MTc0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNmZlOGE1MjUzYWU0NGE4OTJhZjdhOWUzMGI3YjY0ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zOTFjZjZlYWM5YjM0NmEyODcwYTNhMDZmY2EzMzViZSA9ICQoJzxkaXYgaWQ9Imh0bWxfMzkxY2Y2ZWFjOWIzNDZhMjg3MGEzYTA2ZmNhMzM1YmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldvYnVybiwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q2ZmU4YTUyNTNhZTQ0YTg5MmFmN2E5ZTMwYjdiNjRmLnNldENvbnRlbnQoaHRtbF8zOTFjZjZlYWM5YjM0NmEyODcwYTNhMDZmY2EzMzViZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NTFiZDk2MWI3Njg0NmJhOGVkY2U0N2QxMTc0YzU3Yy5iaW5kUG9wdXAocG9wdXBfZDZmZThhNTI1M2FlNDRhODkyYWY3YTllMzBiN2I2NGYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjZjYTU4OWRiMWM2NDY0Yjg4N2JkY2Y5YzNjYzhlOTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzMxMzYsLTc5LjIzOTQ3NjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZhNGIwMmNlZTQ0ZjQwNzU4MjA4OTNjYTg5NDhkODUyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzczZjg5N2M3NWJlOTQyOTg4ODk4MmFlZGU3MDhmN2FkID0gJCgnPGRpdiBpZD0iaHRtbF83M2Y4OTdjNzViZTk0Mjk4ODg5ODJhZWRlNzA4ZjdhZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VkYXJicmFlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmE0YjAyY2VlNDRmNDA3NTgyMDg5M2NhODk0OGQ4NTIuc2V0Q29udGVudChodG1sXzczZjg5N2M3NWJlOTQyOTg4ODk4MmFlZGU3MDhmN2FkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I2Y2E1ODlkYjFjNjQ2NGI4ODdiZGNmOWMzY2M4ZTk2LmJpbmRQb3B1cChwb3B1cF9mYTRiMDJjZWU0NGY0MDc1ODIwODkzY2E4OTQ4ZDg1Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xYTZmM2RlYmQ1Nzk0NmUwODk5MTgzM2YyOTg3NmExYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc0NDczNDIsLTc5LjIzOTQ3NjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhjZWQ4MWQ4Yzk0ZjQzYzY5YWRkODBlNzA5OWYyZDAyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdlYjZjZjYzZGMyMzQ1NGNhZDdkMzY2MWJkNjhlM2UzID0gJCgnPGRpdiBpZD0iaHRtbF83ZWI2Y2Y2M2RjMjM0NTRjYWQ3ZDM2NjFiZDY4ZTNlMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U2NhcmJvcm91Z2ggVmlsbGFnZSwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhjZWQ4MWQ4Yzk0ZjQzYzY5YWRkODBlNzA5OWYyZDAyLnNldENvbnRlbnQoaHRtbF83ZWI2Y2Y2M2RjMjM0NTRjYWQ3ZDM2NjFiZDY4ZTNlMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xYTZmM2RlYmQ1Nzk0NmUwODk5MTgzM2YyOTg3NmExYS5iaW5kUG9wdXAocG9wdXBfOGNlZDgxZDhjOTRmNDNjNjlhZGQ4MGU3MDk5ZjJkMDIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTJlNmJkM2U2MWM4NGVkYTk5YjVlZTdhNDIwNDY4MzYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Mjc5MjkyLC03OS4yNjIwMjk0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMmNjNTllYWUzZDY0YmExODJlODllZDNkNDEyNTBiZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wZTA1OWJmMTBhYzY0MDdkODhkMWEwY2M1MDNiM2YzNyA9ICQoJzxkaXYgaWQ9Imh0bWxfMGUwNTliZjEwYWM2NDA3ZDg4ZDFhMGNjNTAzYjNmMzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktlbm5lZHkgUGFyaywgSW9udmlldywgRWFzdCBCaXJjaG1vdW50IFBhcmssIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jMmNjNTllYWUzZDY0YmExODJlODllZDNkNDEyNTBiZi5zZXRDb250ZW50KGh0bWxfMGUwNTliZjEwYWM2NDA3ZDg4ZDFhMGNjNTAzYjNmMzcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTJlNmJkM2U2MWM4NGVkYTk5YjVlZTdhNDIwNDY4MzYuYmluZFBvcHVwKHBvcHVwX2MyY2M1OWVhZTNkNjRiYTE4MmU4OWVkM2Q0MTI1MGJmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYyN2U5ZDU1ZmIwODQ2OTg4OTdkZmJkNmYyYzllZGNkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzExMTExNzAwMDAwMDA0LC03OS4yODQ1NzcyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M3ZTc2N2FhYmEzNzQwNGI5MmVhYWNiNDkzYzViZDlmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NlYWVhZGJjMTdhYjQwYTY4MTQyNzAxYWE4MGI5OTI4ID0gJCgnPGRpdiBpZD0iaHRtbF9jZWFlYWRiYzE3YWI0MGE2ODE0MjcwMWFhODBiOTkyOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R29sZGVuIE1pbGUsIENsYWlybGVhLCBPYWtyaWRnZSwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M3ZTc2N2FhYmEzNzQwNGI5MmVhYWNiNDkzYzViZDlmLnNldENvbnRlbnQoaHRtbF9jZWFlYWRiYzE3YWI0MGE2ODE0MjcwMWFhODBiOTkyOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MjdlOWQ1NWZiMDg0Njk4ODk3ZGZiZDZmMmM5ZWRjZC5iaW5kUG9wdXAocG9wdXBfYzdlNzY3YWFiYTM3NDA0YjkyZWFhY2I0OTNjNWJkOWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzRjMzZlNmJmNjBlNDI2Y2E1ZmZlYzJjODU3NzRmMjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTYzMTYsLTc5LjIzOTQ3NjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YyZjFiMGMwYTQ5YjQwY2E5ZjAzZDMwNjU4MTg2ZTcyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRlZmMwYjRmNTBmZDQ5Y2RhMTc1MjQwZjU4MzUwYzQ4ID0gJCgnPGRpdiBpZD0iaHRtbF80ZWZjMGI0ZjUwZmQ0OWNkYTE3NTI0MGY1ODM1MGM0OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2xpZmZzaWRlLCBDbGlmZmNyZXN0LCBTY2FyYm9yb3VnaCBWaWxsYWdlIFdlc3QsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMmYxYjBjMGE0OWI0MGNhOWYwM2QzMDY1ODE4NmU3Mi5zZXRDb250ZW50KGh0bWxfNGVmYzBiNGY1MGZkNDljZGExNzUyNDBmNTgzNTBjNDgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzRjMzZlNmJmNjBlNDI2Y2E1ZmZlYzJjODU3NzRmMjguYmluZFBvcHVwKHBvcHVwX2YyZjFiMGMwYTQ5YjQwY2E5ZjAzZDMwNjU4MTg2ZTcyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFlMzYwMjkyYmFkMTRhNWQ5MzAzZWQzZGE5OTA4Zjg3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkyNjU3MDAwMDAwMDA0LC03OS4yNjQ4NDgxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg5NjA1ZDI4YzlhODQwOTJhYjg4YmE3MDhhNjk2ZDk4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VkZWE3OThmZjMzNDRiNWRiZWIzMzYxMDI4NTdjYzdkID0gJCgnPGRpdiBpZD0iaHRtbF9lZGVhNzk4ZmYzMzQ0YjVkYmViMzM2MTAyODU3Y2M3ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmlyY2ggQ2xpZmYsIENsaWZmc2lkZSBXZXN0LCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODk2MDVkMjhjOWE4NDA5MmFiODhiYTcwOGE2OTZkOTguc2V0Q29udGVudChodG1sX2VkZWE3OThmZjMzNDRiNWRiZWIzMzYxMDI4NTdjYzdkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFlMzYwMjkyYmFkMTRhNWQ5MzAzZWQzZGE5OTA4Zjg3LmJpbmRQb3B1cChwb3B1cF84OTYwNWQyOGM5YTg0MDkyYWI4OGJhNzA4YTY5NmQ5OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80NWI4ZThlN2Q2YmE0ZGE0OGM0ODVhYzEzYzY0MWY3ZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1NzQwOTYsLTc5LjI3MzMwNDAwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ5MjNkYzVjNjcxYTQzYzZiNzhmODJiYjc1OWYzMTY1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBjNWMxMmFjMWRjNzRkODk4MWFkMGFlODFlOWU2MDNmID0gJCgnPGRpdiBpZD0iaHRtbF8wYzVjMTJhYzFkYzc0ZDg5ODFhZDBhZTgxZTllNjAzZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9yc2V0IFBhcmssIFdleGZvcmQgSGVpZ2h0cywgU2NhcmJvcm91Z2ggVG93biBDZW50cmUsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80OTIzZGM1YzY3MWE0M2M2Yjc4ZjgyYmI3NTlmMzE2NS5zZXRDb250ZW50KGh0bWxfMGM1YzEyYWMxZGM3NGQ4OTgxYWQwYWU4MWU5ZTYwM2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDViOGU4ZTdkNmJhNGRhNDhjNDg1YWMxM2M2NDFmN2YuYmluZFBvcHVwKHBvcHVwXzQ5MjNkYzVjNjcxYTQzYzZiNzhmODJiYjc1OWYzMTY1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM5OGM1NDQwZjQwYjQ4YzE4NTk2N2QwNWM4ODEzMjBjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzUwMDcxNTAwMDAwMDA0LC03OS4yOTU4NDkxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M2ZWQ2ZThkODg2NDRiMTk4ZTQ2NjExNWM1ODYyNzQ5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzgxN2ViN2Q4YTIzNDQ1YzI4ZWZjYmRkNTEzY2E2MzE4ID0gJCgnPGRpdiBpZD0iaHRtbF84MTdlYjdkOGEyMzQ0NWMyOGVmY2JkZDUxM2NhNjMxOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2V4Zm9yZCwgTWFyeXZhbGUsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jNmVkNmU4ZDg4NjQ0YjE5OGU0NjYxMTVjNTg2Mjc0OS5zZXRDb250ZW50KGh0bWxfODE3ZWI3ZDhhMjM0NDVjMjhlZmNiZGQ1MTNjYTYzMTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzk4YzU0NDBmNDBiNDhjMTg1OTY3ZDA1Yzg4MTMyMGMuYmluZFBvcHVwKHBvcHVwX2M2ZWQ2ZThkODg2NDRiMTk4ZTQ2NjExNWM1ODYyNzQ5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhlODUxNGZlOTZkZjQyZWI4NjJkNTYwODc2MDIyZTA2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzk0MjAwMywtNzkuMjYyMDI5NDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzc0MzZlZWIzYWEzNGE5ZWJjMmE0NWFlNGVmNTc2MzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTM0MjRlY2FlYjhjNDRlZDg4ZmY2MTE0ZjZlM2I2ODggPSAkKCc8ZGl2IGlkPSJodG1sXzEzNDI0ZWNhZWI4YzQ0ZWQ4OGZmNjExNGY2ZTNiNjg4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BZ2luY291cnQsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jNzQzNmVlYjNhYTM0YTllYmMyYTQ1YWU0ZWY1NzYzMS5zZXRDb250ZW50KGh0bWxfMTM0MjRlY2FlYjhjNDRlZDg4ZmY2MTE0ZjZlM2I2ODgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGU4NTE0ZmU5NmRmNDJlYjg2MmQ1NjA4NzYwMjJlMDYuYmluZFBvcHVwKHBvcHVwX2M3NDM2ZWViM2FhMzRhOWViYzJhNDVhZTRlZjU3NjMxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY2Njg0ZTdkMDYyNjQ5MTY4OGM4YmViMTE0MjM0NTBlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzgxNjM3NSwtNzkuMzA0MzAyMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNTIxMzQ1M2E4Zjg0ZTljOTJlNmVmZmI5NDI0ZWM4ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81ZDE3MTIzMjFlNTc0OGE1YjhjYjgxOTNlZTUwNDU0NSA9ICQoJzxkaXYgaWQ9Imh0bWxfNWQxNzEyMzIxZTU3NDhhNWI4Y2I4MTkzZWU1MDQ1NDUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNsYXJrcyBDb3JuZXJzLCBUYW0gTyYjMzk7U2hhbnRlciwgU3VsbGl2YW4sIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zNTIxMzQ1M2E4Zjg0ZTljOTJlNmVmZmI5NDI0ZWM4Zi5zZXRDb250ZW50KGh0bWxfNWQxNzEyMzIxZTU3NDhhNWI4Y2I4MTkzZWU1MDQ1NDUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjY2ODRlN2QwNjI2NDkxNjg4YzhiZWIxMTQyMzQ1MGUuYmluZFBvcHVwKHBvcHVwXzM1MjEzNDUzYThmODRlOWM5MmU2ZWZmYjk0MjRlYzhmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJjNWQyYjRlZjllMzRhM2VhYjU5YzUwYWU5NjQ1OTlkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODE1MjUyMiwtNzkuMjg0NTc3Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MzhmMWE4M2ZjNTI0NTYwYjcxY2Q2NGJjZmEyYWI2ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iMmRiYThhZTA3YTY0N2Q1YTEzNjIwN2UxNWVjNzk5MyA9ICQoJzxkaXYgaWQ9Imh0bWxfYjJkYmE4YWUwN2E2NDdkNWExMzYyMDdlMTVlYzc5OTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1pbGxpa2VuLCBBZ2luY291cnQgTm9ydGgsIFN0ZWVsZXMgRWFzdCwgTCYjMzk7QW1vcmVhdXggRWFzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUzOGYxYTgzZmM1MjQ1NjBiNzFjZDY0YmNmYTJhYjZmLnNldENvbnRlbnQoaHRtbF9iMmRiYThhZTA3YTY0N2Q1YTEzNjIwN2UxNWVjNzk5Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYzVkMmI0ZWY5ZTM0YTNlYWI1OWM1MGFlOTY0NTk5ZC5iaW5kUG9wdXAocG9wdXBfNTM4ZjFhODNmYzUyNDU2MGI3MWNkNjRiY2ZhMmFiNmYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTk5OTljMWQ4ZTRlNGFhM2E0YTYwM2UyMWZmZDBlODAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43OTk1MjUyMDAwMDAwMDUsLTc5LjMxODM4ODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMThlOTI5YzdmMmE2NDdhOWFkNzM3OTg1YTQxMWVkZDggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzRkYzFkOGE4MDVlNGNmODk5ZTQzNzFiZTU4NWQzMTUgPSAkKCc8ZGl2IGlkPSJodG1sXzM0ZGMxZDhhODA1ZTRjZjg5OWU0MzcxYmU1ODVkMzE1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdGVlbGVzIFdlc3QsIEwmIzM5O0Ftb3JlYXV4IFdlc3QsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xOGU5MjljN2YyYTY0N2E5YWQ3Mzc5ODVhNDExZWRkOC5zZXRDb250ZW50KGh0bWxfMzRkYzFkOGE4MDVlNGNmODk5ZTQzNzFiZTU4NWQzMTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTk5OTljMWQ4ZTRlNGFhM2E0YTYwM2UyMWZmZDBlODAuYmluZFBvcHVwKHBvcHVwXzE4ZTkyOWM3ZjJhNjQ3YTlhZDczNzk4NWE0MTFlZGQ4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzcyZjFlODg3OGJjMDQxMGQ5NDFlZTViOThhM2QxODRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODM2MTI0NzAwMDAwMDA2LC03OS4yMDU2MzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MGZhZTNmNTMxNjM0YTg5OTk5OTMxMjM0NjIwOWNiYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iMTViNGZlNjVjNGE0YjZjODRkNTM3ZTk1ZjAxMzNkNSA9ICQoJzxkaXYgaWQ9Imh0bWxfYjE1YjRmZTY1YzRhNGI2Yzg0ZDUzN2U5NWYwMTMzZDUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVwcGVyIFJvdWdlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTBmYWUzZjUzMTYzNGE4OTk5OTkzMTIzNDYyMDljYmMuc2V0Q29udGVudChodG1sX2IxNWI0ZmU2NWM0YTRiNmM4NGQ1MzdlOTVmMDEzM2Q1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzcyZjFlODg3OGJjMDQxMGQ5NDFlZTViOThhM2QxODRiLmJpbmRQb3B1cChwb3B1cF81MGZhZTNmNTMxNjM0YTg5OTk5OTMxMjM0NjIwOWNiYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80MzJhNDc4YzNiYjI0MDlhYTVhOGQ1YjZjY2UxZmVkMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjgwMzc2MjIsLTc5LjM2MzQ1MTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzU3MTNjYWE4MTM2NGEzYjllN2U4ZWYwMTA1ZmM3NGQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWVkNDc3MTZhMTM4NDFhYjk2MzE2ZWIxNWUxODk4NjggPSAkKCc8ZGl2IGlkPSJodG1sXzFlZDQ3NzE2YTEzODQxYWI5NjMxNmViMTVlMTg5ODY4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IaWxsY3Jlc3QgVmlsbGFnZSwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzU3MTNjYWE4MTM2NGEzYjllN2U4ZWYwMTA1ZmM3NGQuc2V0Q29udGVudChodG1sXzFlZDQ3NzE2YTEzODQxYWI5NjMxNmViMTVlMTg5ODY4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQzMmE0NzhjM2JiMjQwOWFhNWE4ZDViNmNjZTFmZWQxLmJpbmRQb3B1cChwb3B1cF83NTcxM2NhYTgxMzY0YTNiOWU3ZThlZjAxMDVmYzc0ZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNmNkYzVhZTRkZjg0YmM2YWE0MzA2ZmE3OTFkZTg0YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc3ODUxNzUsLTc5LjM0NjU1NTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTFjY2YyNTQ2NGYwNDcwM2ExMTE0MWQ2MjM1NDY0YjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2FiY2Y1ZGUxZmYxNDU3MTk4MDE0NzE5MjVhNDMwOWIgPSAkKCc8ZGl2IGlkPSJodG1sXzdhYmNmNWRlMWZmMTQ1NzE5ODAxNDcxOTI1YTQzMDliIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GYWlydmlldywgSGVucnkgRmFybSwgT3Jpb2xlLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MWNjZjI1NDY0ZjA0NzAzYTExMTQxZDYyMzU0NjRiMS5zZXRDb250ZW50KGh0bWxfN2FiY2Y1ZGUxZmYxNDU3MTk4MDE0NzE5MjVhNDMwOWIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjZjZGM1YWU0ZGY4NGJjNmFhNDMwNmZhNzkxZGU4NGEuYmluZFBvcHVwKHBvcHVwXzkxY2NmMjU0NjRmMDQ3MDNhMTExNDFkNjIzNTQ2NGIxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NkMzA0NjIyNWFkMjRjMWZhZWY0M2UwNjQyY2Y3NTVmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzg2OTQ3MywtNzkuMzg1OTc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRiNGQ0MWNhY2M4MzRmNDNiOWIyNWIzYjMzZjA4YWNjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFjNTE3MzZlNjkyZDQxZWFhYWE4ODdiNjZiMmNjODY5ID0gJCgnPGRpdiBpZD0iaHRtbF8xYzUxNzM2ZTY5MmQ0MWVhYWFhODg3YjY2YjJjYzg2OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF5dmlldyBWaWxsYWdlLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80YjRkNDFjYWNjODM0ZjQzYjliMjViM2IzM2YwOGFjYy5zZXRDb250ZW50KGh0bWxfMWM1MTczNmU2OTJkNDFlYWFhYTg4N2I2NmIyY2M4NjkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2QzMDQ2MjI1YWQyNGMxZmFlZjQzZTA2NDJjZjc1NWYuYmluZFBvcHVwKHBvcHVwXzRiNGQ0MWNhY2M4MzRmNDNiOWIyNWIzYjMzZjA4YWNjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNlMWVlOTRjZWQ4NDQwMGM4YjU1NmZhZjAxZTA2ZmU2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU3NDkwMiwtNzkuMzc0NzE0MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmQwYTRhZTI1ZmIyNGU4N2E5YWQ2MjQyMThlZjBjMWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjU4MTBkNjUzN2UxNDk5OGFlNWY4NmM2ZmM3MjFhMTQgPSAkKCc8ZGl2IGlkPSJodG1sXzY1ODEwZDY1MzdlMTQ5OThhZTVmODZjNmZjNzIxYTE0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Zb3JrIE1pbGxzLCBTaWx2ZXIgSGlsbHMsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZkMGE0YWUyNWZiMjRlODdhOWFkNjI0MjE4ZWYwYzFiLnNldENvbnRlbnQoaHRtbF82NTgxMGQ2NTM3ZTE0OTk4YWU1Zjg2YzZmYzcyMWExNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zZTFlZTk0Y2VkODQ0MDBjOGI1NTZmYWYwMWUwNmZlNi5iaW5kUG9wdXAocG9wdXBfNmQwYTRhZTI1ZmIyNGU4N2E5YWQ2MjQyMThlZjBjMWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmNmMWU4MDAwMWQzNGU4MWFiZWE2MmEyMzMyZjNjNDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODkwNTMsLTc5LjQwODQ5Mjc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUzMDcyYWYwNTM4YzRiNzFiYmQ0MTE4NjA0MGFmNmRlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E4MmUzYWM0M2JmNjQ0OTg5MGEzMTE0MTgwZWMwZmMyID0gJCgnPGRpdiBpZD0iaHRtbF9hODJlM2FjNDNiZjY0NDk4OTBhMzExNDE4MGVjMGZjMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2lsbG93ZGFsZSwgTmV3dG9uYnJvb2ssIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUzMDcyYWYwNTM4YzRiNzFiYmQ0MTE4NjA0MGFmNmRlLnNldENvbnRlbnQoaHRtbF9hODJlM2FjNDNiZjY0NDk4OTBhMzExNDE4MGVjMGZjMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iY2YxZTgwMDAxZDM0ZTgxYWJlYTYyYTIzMzJmM2M0MC5iaW5kUG9wdXAocG9wdXBfNTMwNzJhZjA1MzhjNGI3MWJiZDQxMTg2MDQwYWY2ZGUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2FkYmMwMzM0YzE2NGQ4NTkyZThlNmM3OGFjMDU3OGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzAxMTk5LC03OS40MDg0OTI3OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85YmJkODZkNjE2NTQ0NDQ5YmE5ZDYxYTU0YzRlMjA3MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81NmFjNmY3YzEyZDQ0ZGUzYTIwYjdhNThhNWVkYzRkYSA9ICQoJzxkaXYgaWQ9Imh0bWxfNTZhYzZmN2MxMmQ0NGRlM2EyMGI3YTU4YTVlZGM0ZGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldpbGxvd2RhbGUsIFdpbGxvd2RhbGUgRWFzdCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWJiZDg2ZDYxNjU0NDQ0OWJhOWQ2MWE1NGM0ZTIwNzIuc2V0Q29udGVudChodG1sXzU2YWM2ZjdjMTJkNDRkZTNhMjBiN2E1OGE1ZWRjNGRhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdhZGJjMDMzNGMxNjRkODU5MmU4ZTZjNzhhYzA1NzhmLmJpbmRQb3B1cChwb3B1cF85YmJkODZkNjE2NTQ0NDQ5YmE5ZDYxYTU0YzRlMjA3Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yNjNlZTJjOTg2ZTU0ZTIwYTM4NjU0NmEwYTRlOTYwOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1Mjc1ODI5OTk5OTk5NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hN2JjOWRiNTdiOTA0OTdlYWNkN2RmMWYyMDU1ZWU3NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80Y2E0MGFkMTRkYzY0ODdlOTNhNjc5ODQ1OTE5MTllNiA9ICQoJzxkaXYgaWQ9Imh0bWxfNGNhNDBhZDE0ZGM2NDg3ZTkzYTY3OTg0NTkxOTE5ZTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPllvcmsgTWlsbHMgV2VzdCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTdiYzlkYjU3YjkwNDk3ZWFjZDdkZjFmMjA1NWVlNzcuc2V0Q29udGVudChodG1sXzRjYTQwYWQxNGRjNjQ4N2U5M2E2Nzk4NDU5MTkxOWU2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI2M2VlMmM5ODZlNTRlMjBhMzg2NTQ2YTBhNGU5NjA5LmJpbmRQb3B1cChwb3B1cF9hN2JjOWRiNTdiOTA0OTdlYWNkN2RmMWYyMDU1ZWU3Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82Y2E3YjZjYjJjYzM0NjIxYTYwNjZkYmFmMmM1NjAzNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4MjczNjQsLTc5LjQ0MjI1OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjMyNGI0ZTU3ZTM3NDU2MTg1NGUwYjM4NWM1N2JlYTQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDBhZTYxZDI2ZTZlNGUxYWI3NDY1Y2UyYzM0NDU1NzUgPSAkKCc8ZGl2IGlkPSJodG1sXzAwYWU2MWQyNmU2ZTRlMWFiNzQ2NWNlMmMzNDQ1NTc1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XaWxsb3dkYWxlLCBXaWxsb3dkYWxlIFdlc3QsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2YzMjRiNGU1N2UzNzQ1NjE4NTRlMGIzODVjNTdiZWE0LnNldENvbnRlbnQoaHRtbF8wMGFlNjFkMjZlNmU0ZTFhYjc0NjVjZTJjMzQ0NTU3NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82Y2E3YjZjYjJjYzM0NjIxYTYwNjZkYmFmMmM1NjAzNC5iaW5kUG9wdXAocG9wdXBfZjMyNGI0ZTU3ZTM3NDU2MTg1NGUwYjM4NWM1N2JlYTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGIwZTViOGRiZDRhNDQ2Yjg2N2ZjYWViNDQ4OGYzZjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTMyNTg2LC03OS4zMjk2NTY1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I0MTVjYzBjMGQ0ZDQ1MjA4MmY5YzI1ZjE0ZDQyNmYwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYyODA5ZWMyNmU0NjQzMzRhNWQyNmJjYzg5ODc3NDg2ID0gJCgnPGRpdiBpZD0iaHRtbF82MjgwOWVjMjZlNDY0MzM0YTVkMjZiY2M4OTg3NzQ4NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya3dvb2RzLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNDE1Y2MwYzBkNGQ0NTIwODJmOWMyNWYxNGQ0MjZmMC5zZXRDb250ZW50KGh0bWxfNjI4MDllYzI2ZTQ2NDMzNGE1ZDI2YmNjODk4Nzc0ODYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGIwZTViOGRiZDRhNDQ2Yjg2N2ZjYWViNDQ4OGYzZjQuYmluZFBvcHVwKHBvcHVwX2I0MTVjYzBjMGQ0ZDQ1MjA4MmY5YzI1ZjE0ZDQyNmYwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYyMTdiOTIwMzJhYzQ2N2NiOGMyMTgxNzc2MTQ0YzczID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzQ1OTA1Nzk5OTk5OTk2LC03OS4zNTIxODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzM2NWVjMTI3MTBhNDAxYWJiYWJmNzJmMDM3MzQ1ZDQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjBiMjZlYzMwNDhjNDAyN2E0Y2Y1ODBkZDk5NTA2NjUgPSAkKCc8ZGl2IGlkPSJodG1sXzIwYjI2ZWMzMDQ4YzQwMjdhNGNmNTgwZGQ5OTUwNjY1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb24gTWlsbHMsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MzNjVlYzEyNzEwYTQwMWFiYmFiZjcyZjAzNzM0NWQ0LnNldENvbnRlbnQoaHRtbF8yMGIyNmVjMzA0OGM0MDI3YTRjZjU4MGRkOTk1MDY2NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MjE3YjkyMDMyYWM0NjdjYjhjMjE4MTc3NjE0NGM3My5iaW5kUG9wdXAocG9wdXBfYzM2NWVjMTI3MTBhNDAxYWJiYWJmNzJmMDM3MzQ1ZDQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjAxZGZhMzk1ZTJjNDZjNGIxNzM0MmRkNTE2YTUwMjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MjU4OTk3MDAwMDAwMSwtNzkuMzQwOTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U1YmMxNDFkN2Y4ZTQxZDc4NWQ1OWNjMjcwMGQyNzYzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk1ODQ5YWY2ZDA4NTQ1MTU4Y2U0NDU4Y2YzZmQzMGQyID0gJCgnPGRpdiBpZD0iaHRtbF85NTg0OWFmNmQwODU0NTE1OGNlNDQ1OGNmM2ZkMzBkMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9uIE1pbGxzLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lNWJjMTQxZDdmOGU0MWQ3ODVkNTljYzI3MDBkMjc2My5zZXRDb250ZW50KGh0bWxfOTU4NDlhZjZkMDg1NDUxNThjZTQ0NThjZjNmZDMwZDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjAxZGZhMzk1ZTJjNDZjNGIxNzM0MmRkNTE2YTUwMjQuYmluZFBvcHVwKHBvcHVwX2U1YmMxNDFkN2Y4ZTQxZDc4NWQ1OWNjMjcwMGQyNzYzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA1MjZjNWMxZmU4MTRmZTE5MjM2YTdlYjU4ZDE2MWNjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU0MzI4MywtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNTY1YmUzZGY5NWI0OWQ5OTA2NjY4NjE5MTM2NDc5MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yYzYxYWFjMmQxMjQ0MTJkYWU4MGVjMjBlNGY0YjdlYSA9ICQoJzxkaXYgaWQ9Imh0bWxfMmM2MWFhYzJkMTI0NDEyZGFlODBlYzIwZTRmNGI3ZWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJhdGh1cnN0IE1hbm9yLCBXaWxzb24gSGVpZ2h0cywgRG93bnN2aWV3IE5vcnRoLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mNTY1YmUzZGY5NWI0OWQ5OTA2NjY4NjE5MTM2NDc5MS5zZXRDb250ZW50KGh0bWxfMmM2MWFhYzJkMTI0NDEyZGFlODBlYzIwZTRmNGI3ZWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDUyNmM1YzFmZTgxNGZlMTkyMzZhN2ViNThkMTYxY2MuYmluZFBvcHVwKHBvcHVwX2Y1NjViZTNkZjk1YjQ5ZDk5MDY2Njg2MTkxMzY0NzkxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzgyNzYwOThkOTRmZTRjMmQ4MjU1NWJkYzQwZDdmNzZmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzY3OTgwMywtNzkuNDg3MjYxOTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTg3M2FiYzg2NjdkNDcxOGE2MTFkNDc2YTA4YzdhMTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGM5ZjgwOGU5MmIyNGViNWIxNzVlMjViZmIwZDZjYTUgPSAkKCc8ZGl2IGlkPSJodG1sX2RjOWY4MDhlOTJiMjRlYjViMTc1ZTI1YmZiMGQ2Y2E1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aHdvb2QgUGFyaywgWW9yayBVbml2ZXJzaXR5LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ODczYWJjODY2N2Q0NzE4YTYxMWQ0NzZhMDhjN2ExMC5zZXRDb250ZW50KGh0bWxfZGM5ZjgwOGU5MmIyNGViNWIxNzVlMjViZmIwZDZjYTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODI3NjA5OGQ5NGZlNGMyZDgyNTU1YmRjNDBkN2Y3NmYuYmluZFBvcHVwKHBvcHVwXzk4NzNhYmM4NjY3ZDQ3MThhNjExZDQ3NmEwOGM3YTEwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZiMTAzN2RlMmE0MjRkNjA4MzYwNWYxMjUxY2FkYTJmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzM3NDczMjAwMDAwMDA0LC03OS40NjQ3NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zZDVmNjE0OTg3ZDc0NTBmYTM5MzVkOGNlNTk0NWEzZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZjBjMmQ1NjkxZDM0MjhjYmY1MTc4ZmM2ZWEzMTgxOCA9ICQoJzxkaXYgaWQ9Imh0bWxfYmYwYzJkNTY5MWQzNDI4Y2JmNTE3OGZjNmVhMzE4MTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd25zdmlldywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2Q1ZjYxNDk4N2Q3NDUwZmEzOTM1ZDhjZTU5NDVhM2Yuc2V0Q29udGVudChodG1sX2JmMGMyZDU2OTFkMzQyOGNiZjUxNzhmYzZlYTMxODE4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZiMTAzN2RlMmE0MjRkNjA4MzYwNWYxMjUxY2FkYTJmLmJpbmRQb3B1cChwb3B1cF8zZDVmNjE0OTg3ZDc0NTBmYTM5MzVkOGNlNTk0NWEzZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80ZjI0NjMxZjVlY2U0YzFjOGQ3MzM5NjM2ODg1ZDAxZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTAxNDYsLTc5LjUwNjk0MzZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOThlMWRhOTM4MGY5NDQzNGJkNGZhNzRjYWFhOTE5MDkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmE2NGFhZmFmNjcyNDZmMDhiNTEyOGY5ZDNmODllZjUgPSAkKCc8ZGl2IGlkPSJodG1sXzZhNjRhYWZhZjY3MjQ2ZjA4YjUxMjhmOWQzZjg5ZWY1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk4ZTFkYTkzODBmOTQ0MzRiZDRmYTc0Y2FhYTkxOTA5LnNldENvbnRlbnQoaHRtbF82YTY0YWFmYWY2NzI0NmYwOGI1MTI4ZjlkM2Y4OWVmNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80ZjI0NjMxZjVlY2U0YzFjOGQ3MzM5NjM2ODg1ZDAxZS5iaW5kUG9wdXAocG9wdXBfOThlMWRhOTM4MGY5NDQzNGJkNGZhNzRjYWFhOTE5MDkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWU5YWZlZmQ0NzA0NDU5ZDgzOWEwOGZiYzFkYmE1ZDYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Mjg0OTY0LC03OS40OTU2OTc0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZTRjZDUyNzU1Yzk0NGMyYTU1ZTJlNzQwODc3MmMxMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kNzhjODRiZmVkMmI0MGNlYjBhZTU2N2I5ZGNlY2E2OSA9ICQoJzxkaXYgaWQ9Imh0bWxfZDc4Yzg0YmZlZDJiNDBjZWIwYWU1NjdiOWRjZWNhNjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd25zdmlldywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGU0Y2Q1Mjc1NWM5NDRjMmE1NWUyZTc0MDg3NzJjMTAuc2V0Q29udGVudChodG1sX2Q3OGM4NGJmZWQyYjQwY2ViMGFlNTY3YjlkY2VjYTY5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VlOWFmZWZkNDcwNDQ1OWQ4MzlhMDhmYmMxZGJhNWQ2LmJpbmRQb3B1cChwb3B1cF9kZTRjZDUyNzU1Yzk0NGMyYTU1ZTJlNzQwODc3MmMxMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNDJiOWMwNzk4OTk0Nzc3ODBjMDc1MWY0OGUxNDllZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc2MTYzMTMsLTc5LjUyMDk5OTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EyMWY0NTg5Y2ExNTQwZDFhNzZiZWU4M2ZmZmY2YjQ5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk3MzcyODBhZTc4ZDQzOGY4MDFjZWYzOTcwMzM3Mjg3ID0gJCgnPGRpdiBpZD0iaHRtbF85NzM3MjgwYWU3OGQ0MzhmODAxY2VmMzk3MDMzNzI4NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnN2aWV3LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMjFmNDU4OWNhMTU0MGQxYTc2YmVlODNmZmZmNmI0OS5zZXRDb250ZW50KGh0bWxfOTczNzI4MGFlNzhkNDM4ZjgwMWNlZjM5NzAzMzcyODcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjQyYjljMDc5ODk5NDc3NzgwYzA3NTFmNDhlMTQ5ZWUuYmluZFBvcHVwKHBvcHVwX2EyMWY0NTg5Y2ExNTQwZDFhNzZiZWU4M2ZmZmY2YjQ5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U3NjdhODgwZTAzODRiMTViMWE1NzNmYTI5MDYzMjQ5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI1ODgyMjk5OTk5OTk1LC03OS4zMTU1NzE1OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84NDc1YzYxNTU4ZjA0YTI5YjQ3YzA3YjViNGZlY2U1ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MzAwZGEwYzQxMGQ0YzExODJhNTU2MTQxNzQwZDA0OCA9ICQoJzxkaXYgaWQ9Imh0bWxfNzMwMGRhMGM0MTBkNGMxMTgyYTU1NjE0MTc0MGQwNDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlZpY3RvcmlhIFZpbGxhZ2UsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg0NzVjNjE1NThmMDRhMjliNDdjMDdiNWI0ZmVjZTVmLnNldENvbnRlbnQoaHRtbF83MzAwZGEwYzQxMGQ0YzExODJhNTU2MTQxNzQwZDA0OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lNzY3YTg4MGUwMzg0YjE1YjFhNTczZmEyOTA2MzI0OS5iaW5kUG9wdXAocG9wdXBfODQ3NWM2MTU1OGYwNGEyOWI0N2MwN2I1YjRmZWNlNWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmZiN2Y1NjA3YzMzNDhmZTk2Y2NmZjc2OGE4YjgwOWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDYzOTcyLC03OS4zMDk5MzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTAyN2NiYmU3YWNkNGU2MDg2MmRmNjRjN2UzNjdlN2QgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTYxMGQ2MWM3ZmJjNDFmMGJjNzRlYzg4NzYxNzhiOTIgPSAkKCc8ZGl2IGlkPSJodG1sX2E2MTBkNjFjN2ZiYzQxZjBiYzc0ZWM4ODc2MTc4YjkyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJrdmlldyBIaWxsLCBXb29kYmluZSBHYXJkZW5zLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzkwMjdjYmJlN2FjZDRlNjA4NjJkZjY0YzdlMzY3ZTdkLnNldENvbnRlbnQoaHRtbF9hNjEwZDYxYzdmYmM0MWYwYmM3NGVjODg3NjE3OGI5Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mZmI3ZjU2MDdjMzM0OGZlOTZjY2ZmNzY4YThiODA5Yy5iaW5kUG9wdXAocG9wdXBfOTAyN2NiYmU3YWNkNGU2MDg2MmRmNjRjN2UzNjdlN2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTQzY2Q4ZTMxZTQ1NDZjYWJhMTczODkxYjRlOWRmOGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTUzNDM5MDAwMDAwMDUsLTc5LjMxODM4ODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDQ5NjUzMTM1YzZjNGEzYjkxNmZmM2I0MDUzZjM1NTkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjYxZTZhOTA5NjE2NDQ3NDk3ZDA2MzFkMTVkNjcwYmEgPSAkKCc8ZGl2IGlkPSJodG1sXzY2MWU2YTkwOTYxNjQ0NzQ5N2QwNjMxZDE1ZDY3MGJhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Xb29kYmluZSBIZWlnaHRzLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzA0OTY1MzEzNWM2YzRhM2I5MTZmZjNiNDA1M2YzNTU5LnNldENvbnRlbnQoaHRtbF82NjFlNmE5MDk2MTY0NDc0OTdkMDYzMWQxNWQ2NzBiYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNDNjZDhlMzFlNDU0NmNhYmExNzM4OTFiNGU5ZGY4Yi5iaW5kUG9wdXAocG9wdXBfMDQ5NjUzMTM1YzZjNGEzYjkxNmZmM2I0MDUzZjM1NTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzI4Zjk4ZTg2ZWExNGQ2NTkzOGM4OTlkN2Y0NTI0YWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzYzNTczOTk5OTk5OSwtNzkuMjkzMDMxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yYjNjMTQzNTAxM2U0ZDE1YmY3ODhkMTZhNGU1NTA3MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ZjVhNWEyYmU4ZDc0ZTk5ODgwMTMwMzI2ZDlkNTUxYSA9ICQoJzxkaXYgaWQ9Imh0bWxfNmY1YTVhMmJlOGQ3NGU5OTg4MDEzMDMyNmQ5ZDU1MWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBCZWFjaGVzLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJiM2MxNDM1MDEzZTRkMTViZjc4OGQxNmE0ZTU1MDcyLnNldENvbnRlbnQoaHRtbF82ZjVhNWEyYmU4ZDc0ZTk5ODgwMTMwMzI2ZDlkNTUxYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83MjhmOThlODZlYTE0ZDY1OTM4Yzg5OWQ3ZjQ1MjRhZS5iaW5kUG9wdXAocG9wdXBfMmIzYzE0MzUwMTNlNGQxNWJmNzg4ZDE2YTRlNTUwNzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODJiYTcxZmY1OTVkNGZkOWJlMzEzMjlhYmJkZTJkZjUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDkwNjA0LC03OS4zNjM0NTE3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU2NjBhZTdlMGFjZDRkOTc4OWM4N2NkNWFhZDcxYWEyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhkMTY1MGQ5NjdjMzQ3OGNiODRhNGQ3MTY2ODI5NDlhID0gJCgnPGRpdiBpZD0iaHRtbF84ZDE2NTBkOTY3YzM0NzhjYjg0YTRkNzE2NjgyOTQ5YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGVhc2lkZSwgRWFzdCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NjYwYWU3ZTBhY2Q0ZDk3ODljODdjZDVhYWQ3MWFhMi5zZXRDb250ZW50KGh0bWxfOGQxNjUwZDk2N2MzNDc4Y2I4NGE0ZDcxNjY4Mjk0OWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODJiYTcxZmY1OTVkNGZkOWJlMzEzMjlhYmJkZTJkZjUuYmluZFBvcHVwKHBvcHVwXzU2NjBhZTdlMGFjZDRkOTc4OWM4N2NkNWFhZDcxYWEyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJhODY3ZDQxM2VjNDRmNWJhYmVhN2VhZjlmZTJjMDNhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA1MzY4OSwtNzkuMzQ5MzcxOTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTkyYjgyNWQyZjRkNDcyNDlhYTk3NmJjMzJkYjBhN2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDJlZjkwODU5ZTYwNGIxZDgwODU1OTY5ZTI5MGNkM2YgPSAkKCc8ZGl2IGlkPSJodG1sXzQyZWY5MDg1OWU2MDRiMWQ4MDg1NTk2OWUyOTBjZDNmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaG9ybmNsaWZmZSBQYXJrLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U5MmI4MjVkMmY0ZDQ3MjQ5YWE5NzZiYzMyZGIwYTdiLnNldENvbnRlbnQoaHRtbF80MmVmOTA4NTllNjA0YjFkODA4NTU5NjllMjkwY2QzZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYTg2N2Q0MTNlYzQ0ZjViYWJlYTdlYWY5ZmUyYzAzYS5iaW5kUG9wdXAocG9wdXBfZTkyYjgyNWQyZjRkNDcyNDlhYTk3NmJjMzJkYjBhN2IpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGJmZDBhNWYzNTViNDg1N2FkMzM2NWQzYTUwZDQ1Y2IgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODUzNDcsLTc5LjMzODEwNjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmRiNjM5NmIxOWI1NDM0NTg3ZmU3MTc2MjNkMjk2ODMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWUzOTliMTNjMjM3NDg0MmI5ZWY3N2MyZmMzZTUyMzEgPSAkKCc8ZGl2IGlkPSJodG1sX2VlMzk5YjEzYzIzNzQ4NDJiOWVmNzdjMmZjM2U1MjMxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FYXN0IFRvcm9udG8sIEJyb2FkdmlldyBOb3J0aCAoT2xkIEVhc3QgWW9yayksIEVhc3QgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMmRiNjM5NmIxOWI1NDM0NTg3ZmU3MTc2MjNkMjk2ODMuc2V0Q29udGVudChodG1sX2VlMzk5YjEzYzIzNzQ4NDJiOWVmNzdjMmZjM2U1MjMxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhiZmQwYTVmMzU1YjQ4NTdhZDMzNjVkM2E1MGQ0NWNiLmJpbmRQb3B1cChwb3B1cF8yZGI2Mzk2YjE5YjU0MzQ1ODdmZTcxNzYyM2QyOTY4Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80YTM1NmU1ZTA3ZTA0ZTM5YjkyZTk0NWFjY2M0MDA5MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU1NzEsLTc5LjM1MjE4OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MzhmMjFmZjBjMDQ0YmFkYTk1ZDg3OTRmYjc1MjliNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jZWNmM2JlYzY2Yjk0M2M5YTc4OTVlZTk0NTc2MDIxYiA9ICQoJzxkaXYgaWQ9Imh0bWxfY2VjZjNiZWM2NmI5NDNjOWE3ODk1ZWU5NDU3NjAyMWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBEYW5mb3J0aCBXZXN0LCBSaXZlcmRhbGUsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTM4ZjIxZmYwYzA0NGJhZGE5NWQ4Nzk0ZmI3NTI5YjQuc2V0Q29udGVudChodG1sX2NlY2YzYmVjNjZiOTQzYzlhNzg5NWVlOTQ1NzYwMjFiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRhMzU2ZTVlMDdlMDRlMzliOTJlOTQ1YWNjYzQwMDkyLmJpbmRQb3B1cChwb3B1cF81MzhmMjFmZjBjMDQ0YmFkYTk1ZDg3OTRmYjc1MjliNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMGRjZWIzNDRmM2Y0ZmQ1OWM3NTczZWVmNDdhNDk4MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2ODk5ODUsLTc5LjMxNTU3MTU5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ExN2RiMmQ5OWM0MDQ0MjViNmI5Y2I3ZDU5OWEwMDU5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc2Y2NjNjRkNmFhODQ1MzdiNGEyNTkxZmE3NGY4OGU3ID0gJCgnPGRpdiBpZD0iaHRtbF83NmNjYzY0ZDZhYTg0NTM3YjRhMjU5MWZhNzRmODhlNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SW5kaWEgQmF6YWFyLCBUaGUgQmVhY2hlcyBXZXN0LCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ExN2RiMmQ5OWM0MDQ0MjViNmI5Y2I3ZDU5OWEwMDU5LnNldENvbnRlbnQoaHRtbF83NmNjYzY0ZDZhYTg0NTM3YjRhMjU5MWZhNzRmODhlNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMGRjZWIzNDRmM2Y0ZmQ1OWM3NTczZWVmNDdhNDk4Mi5iaW5kUG9wdXAocG9wdXBfYTE3ZGIyZDk5YzQwNDQyNWI2YjljYjdkNTk5YTAwNTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTA3OWEyNjgyYWExNDRlNTg0ZWM3NzA4YmZiMTE2MzQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTk1MjU1LC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjYyN2QzOTg4ZmNkNGY2NjljNjIxNWJkMzcyNzY2MWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGM2YmEyMGVmNzJkNDZhNzkxNjhjNTJiZWM2ODM5YzkgPSAkKCc8ZGl2IGlkPSJodG1sXzBjNmJhMjBlZjcyZDQ2YTc5MTY4YzUyYmVjNjgzOWM5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdHVkaW8gRGlzdHJpY3QsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjYyN2QzOTg4ZmNkNGY2NjljNjIxNWJkMzcyNzY2MWMuc2V0Q29udGVudChodG1sXzBjNmJhMjBlZjcyZDQ2YTc5MTY4YzUyYmVjNjgzOWM5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzEwNzlhMjY4MmFhMTQ0ZTU4NGVjNzcwOGJmYjExNjM0LmJpbmRQb3B1cChwb3B1cF9mNjI3ZDM5ODhmY2Q0ZjY2OWM2MjE1YmQzNzI3NjYxYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80MDEwYmQ1ZTVmMWE0MTcxYmQ3ODZiZjVhZGJhMGVmNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyODAyMDUsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjhhMDIzMDNhMThhNDMxMzk1Njg0ZDQ3Yzk2M2MxOTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGJkZTdiYzFmNjY5NDM1MmE3ZjQxNDYyYjk5YWM5MDEgPSAkKCc8ZGl2IGlkPSJodG1sX2RiZGU3YmMxZjY2OTQzNTJhN2Y0MTQ2MmI5OWFjOTAxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MYXdyZW5jZSBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y4YTAyMzAzYTE4YTQzMTM5NTY4NGQ0N2M5NjNjMTk3LnNldENvbnRlbnQoaHRtbF9kYmRlN2JjMWY2Njk0MzUyYTdmNDE0NjJiOTlhYzkwMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MDEwYmQ1ZTVmMWE0MTcxYmQ3ODZiZjVhZGJhMGVmNS5iaW5kUG9wdXAocG9wdXBfZjhhMDIzMDNhMThhNDMxMzk1Njg0ZDQ3Yzk2M2MxOTcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWRlZDAyZDA3YmI2NGE0MzlmNmQzNzZjZmNmOTdmNzcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTI3NTExLC03OS4zOTAxOTc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzczMThlNDEyNGJiYjQxYmI4OGE1MTBkOTk3YTk2NmYwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA3NzVmYTRhMDkyODQ3NmRiNDE5MDZiYWEwMmI5Nzg2ID0gJCgnPGRpdiBpZD0iaHRtbF8wNzc1ZmE0YTA5Mjg0NzZkYjQxOTA2YmFhMDJiOTc4NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSBOb3J0aCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83MzE4ZTQxMjRiYmI0MWJiODhhNTEwZDk5N2E5NjZmMC5zZXRDb250ZW50KGh0bWxfMDc3NWZhNGEwOTI4NDc2ZGI0MTkwNmJhYTAyYjk3ODYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWRlZDAyZDA3YmI2NGE0MzlmNmQzNzZjZmNmOTdmNzcuYmluZFBvcHVwKHBvcHVwXzczMThlNDEyNGJiYjQxYmI4OGE1MTBkOTk3YTk2NmYwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRjYjNlZDVkNDhkNTRlNjdiYjdkOGRlYmRiNjg0OWIyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE1MzgzNCwtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWJmNDQyNzdjNGE2NDA2MjgxOTI3MjhmMDY5MDJkYWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTAyODllYjA3OTFiNDhmNDkxYjU0OGI1ZTUyNDVjNzAgPSAkKCc8ZGl2IGlkPSJodG1sXzkwMjg5ZWIwNzkxYjQ4ZjQ5MWI1NDhiNWU1MjQ1YzcwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBUb3JvbnRvIFdlc3QsICBMYXdyZW5jZSBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzliZjQ0Mjc3YzRhNjQwNjI4MTkyNzI4ZjA2OTAyZGFjLnNldENvbnRlbnQoaHRtbF85MDI4OWViMDc5MWI0OGY0OTFiNTQ4YjVlNTI0NWM3MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80Y2IzZWQ1ZDQ4ZDU0ZTY3YmI3ZDhkZWJkYjY4NDliMi5iaW5kUG9wdXAocG9wdXBfOWJmNDQyNzdjNGE2NDA2MjgxOTI3MjhmMDY5MDJkYWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWY5MzhkOTRhZmM4NDdhMzg3NzgyNWEzNzA3MmZhOGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDQzMjQ0LC03OS4zODg3OTAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzFmNzQ0NzExODU3YzQzMDRiZWZhMzViMzQ0MmEwNzljID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E0NTkzNTdjN2ZjMTQ0NzA4MmZlYWE4MjlmY2U0OTM0ID0gJCgnPGRpdiBpZD0iaHRtbF9hNDU5MzU3YzdmYzE0NDcwODJmZWFhODI5ZmNlNDkzNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xZjc0NDcxMTg1N2M0MzA0YmVmYTM1YjM0NDJhMDc5Yy5zZXRDb250ZW50KGh0bWxfYTQ1OTM1N2M3ZmMxNDQ3MDgyZmVhYTgyOWZjZTQ5MzQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWY5MzhkOTRhZmM4NDdhMzg3NzgyNWEzNzA3MmZhOGEuYmluZFBvcHVwKHBvcHVwXzFmNzQ0NzExODU3YzQzMDRiZWZhMzViMzQ0MmEwNzljKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I5NDcyZGEwYWZmMTQyNGRiMzgxYjM2MmI0ZjU4ZGI5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg5NTc0MywtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGRkYjQzNzExNGNjNDVjZjhhMGVjYTFhYjFmZmYxYmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTljOTQwYTc4N2RiNGI0N2E0YTJhNmE3MTQ5NTJiMjYgPSAkKCc8ZGl2IGlkPSJodG1sX2E5Yzk0MGE3ODdkYjRiNDdhNGEyYTZhNzE0OTUyYjI2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb29yZSBQYXJrLCBTdW1tZXJoaWxsIEVhc3QsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGRkYjQzNzExNGNjNDVjZjhhMGVjYTFhYjFmZmYxYmIuc2V0Q29udGVudChodG1sX2E5Yzk0MGE3ODdkYjRiNDdhNGEyYTZhNzE0OTUyYjI2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I5NDcyZGEwYWZmMTQyNGRiMzgxYjM2MmI0ZjU4ZGI5LmJpbmRQb3B1cChwb3B1cF80ZGRiNDM3MTE0Y2M0NWNmOGEwZWNhMWFiMWZmZjFiYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYjgzMDk4N2Y4ZDM0YjViYTVkOGFhZWIyMzVjZmM1NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4NjQxMjI5OTk5OTk5LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzFlMDNlODQ0NzNkMzQ2MmE5NjE0ZDYzZmQxNGMwYTViID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2MwNzg2MTQwZmEyMjRkNzQ5ZGUzYzlhMDViYmE2YTRhID0gJCgnPGRpdiBpZD0iaHRtbF9jMDc4NjE0MGZhMjI0ZDc0OWRlM2M5YTA1YmJhNmE0YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3VtbWVyaGlsbCBXZXN0LCBSYXRobmVsbHksIFNvdXRoIEhpbGwsIEZvcmVzdCBIaWxsIFNFLCBEZWVyIFBhcmssIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWUwM2U4NDQ3M2QzNDYyYTk2MTRkNjNmZDE0YzBhNWIuc2V0Q29udGVudChodG1sX2MwNzg2MTQwZmEyMjRkNzQ5ZGUzYzlhMDViYmE2YTRhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBiODMwOTg3ZjhkMzRiNWJhNWQ4YWFlYjIzNWNmYzU3LmJpbmRQb3B1cChwb3B1cF8xZTAzZTg0NDczZDM0NjJhOTYxNGQ2M2ZkMTRjMGE1Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82OGE1ZTE3ZTM0MGM0YWQ0YWRiMzdhNzg0NWEyYjQxNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU2MjYsLTc5LjM3NzUyOTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzczZjI0OTRkMmE1ODRjNDlhN2UzMWM2MzE1NGRhMWE3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBjNzFjYTM3NmFhYTRiM2RhNzdjMDVmOGE4ZGY2MjhmID0gJCgnPGRpdiBpZD0iaHRtbF8wYzcxY2EzNzZhYWE0YjNkYTc3YzA1ZjhhOGRmNjI4ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWRhbGUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzczZjI0OTRkMmE1ODRjNDlhN2UzMWM2MzE1NGRhMWE3LnNldENvbnRlbnQoaHRtbF8wYzcxY2EzNzZhYWE0YjNkYTc3YzA1ZjhhOGRmNjI4Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82OGE1ZTE3ZTM0MGM0YWQ0YWRiMzdhNzg0NWEyYjQxNy5iaW5kUG9wdXAocG9wdXBfNzNmMjQ5NGQyYTU4NGM0OWE3ZTMxYzYzMTU0ZGExYTcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjU5MDU1NzU4NDk0NDRiNGJiMjY1MjUyODE2OGRmZjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc5NjcsLTc5LjM2NzY3NTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmQ4YTU5MTkwYWM2NGU4OTgwOWVmZjZiMmQ4MTc3MzUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmVkZWU5ZmUzNDk5NDNhYThkMzNmMTJmNDk4OTg3YzkgPSAkKCc8ZGl2IGlkPSJodG1sX2JlZGVlOWZlMzQ5OTQzYWE4ZDMzZjEyZjQ5ODk4N2M5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdC4gSmFtZXMgVG93biwgQ2FiYmFnZXRvd24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZkOGE1OTE5MGFjNjRlODk4MDllZmY2YjJkODE3NzM1LnNldENvbnRlbnQoaHRtbF9iZWRlZTlmZTM0OTk0M2FhOGQzM2YxMmY0OTg5ODdjOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82NTkwNTU3NTg0OTQ0NGI0YmIyNjUyNTI4MTY4ZGZmMi5iaW5kUG9wdXAocG9wdXBfNmQ4YTU5MTkwYWM2NGU4OTgwOWVmZjZiMmQ4MTc3MzUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzU3ZDY3YTIxMTBlNDg3Y2EzYWJkOTg0ZjJmM2JiOTMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNDUxMmY3N2E2MGQ0NGM1YTM0MzdhOGI2OWY4MDNkZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kYTMzOGM0MzJiOTA0OTUzYTRhZmJmMjEzYzM4NDg1NiA9ICQoJzxkaXYgaWQ9Imh0bWxfZGEzMzhjNDMyYjkwNDk1M2E0YWZiZjIxM2MzODQ4NTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNodXJjaCBhbmQgV2VsbGVzbGV5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNDUxMmY3N2E2MGQ0NGM1YTM0MzdhOGI2OWY4MDNkZC5zZXRDb250ZW50KGh0bWxfZGEzMzhjNDMyYjkwNDk1M2E0YWZiZjIxM2MzODQ4NTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzU3ZDY3YTIxMTBlNDg3Y2EzYWJkOTg0ZjJmM2JiOTMuYmluZFBvcHVwKHBvcHVwXzI0NTEyZjc3YTYwZDQ0YzVhMzQzN2E4YjY5ZjgwM2RkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVmYjk4MWQ2OTg4YjQ4M2ZiYTZjZWU1YmEyNGQ2ZWI4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU0MjU5OSwtNzkuMzYwNjM1OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lY2IwNmM3NzRmMDY0MmY3YTNiM2UwMTgxNmQzMGM2ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81MDQ3MjE5ZDY4ZmI0OWU0YjAxZjI5MjFiNGYzOWM0NSA9ICQoJzxkaXYgaWQ9Imh0bWxfNTA0NzIxOWQ2OGZiNDllNGIwMWYyOTIxYjRmMzljNDUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJlZ2VudCBQYXJrLCBIYXJib3VyZnJvbnQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VjYjA2Yzc3NGYwNjQyZjdhM2IzZTAxODE2ZDMwYzZlLnNldENvbnRlbnQoaHRtbF81MDQ3MjE5ZDY4ZmI0OWU0YjAxZjI5MjFiNGYzOWM0NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81ZmI5ODFkNjk4OGI0ODNmYmE2Y2VlNWJhMjRkNmViOC5iaW5kUG9wdXAocG9wdXBfZWNiMDZjNzc0ZjA2NDJmN2EzYjNlMDE4MTZkMzBjNmUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGE1NDM3YzFmMDI1NDU1ZGEzZmYyZGYxZmM0NDNjNzIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTcxNjE4LC03OS4zNzg5MzcwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iNzc5ZWRjMTEzYTQ0MTlkYTBhMTgwMWZlZmMwNDY2OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hZDU3NzIwZDBkN2Q0M2VkYWE5NWFlM2E5Mjc1NzFkOSA9ICQoJzxkaXYgaWQ9Imh0bWxfYWQ1NzcyMGQwZDdkNDNlZGFhOTVhZTNhOTI3NTcxZDkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdhcmRlbiBEaXN0cmljdCwgUnllcnNvbiwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjc3OWVkYzExM2E0NDE5ZGEwYTE4MDFmZWZjMDQ2Njguc2V0Q29udGVudChodG1sX2FkNTc3MjBkMGQ3ZDQzZWRhYTk1YWUzYTkyNzU3MWQ5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBhNTQzN2MxZjAyNTQ1NWRhM2ZmMmRmMWZjNDQzYzcyLmJpbmRQb3B1cChwb3B1cF9iNzc5ZWRjMTEzYTQ0MTlkYTBhMTgwMWZlZmMwNDY2OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xODYxZDYzNjQ4ZTA0Y2EzYjk5ODMwZTgyYzcwZjI0YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTQ5MzksLTc5LjM3NTQxNzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODM4M2RkYmQwNjU1NDc2NGI4MmI2MDQ5OWYzM2U0ZmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjY0YzM1Y2NlNWU5NDA1NDljMmZlZjVhOTRhMDNjNWEgPSAkKCc8ZGl2IGlkPSJodG1sX2I2NGMzNWNjZTVlOTQwNTQ5YzJmZWY1YTk0YTAzYzVhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdC4gSmFtZXMgVG93biwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODM4M2RkYmQwNjU1NDc2NGI4MmI2MDQ5OWYzM2U0ZmMuc2V0Q29udGVudChodG1sX2I2NGMzNWNjZTVlOTQwNTQ5YzJmZWY1YTk0YTAzYzVhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE4NjFkNjM2NDhlMDRjYTNiOTk4MzBlODJjNzBmMjRhLmJpbmRQb3B1cChwb3B1cF84MzgzZGRiZDA2NTU0NzY0YjgyYjYwNDk5ZjMzZTRmYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NTI4ZmY1NjkxZjQ0OTliYWE3YmYwOTc3ZWE1YzhiNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NDc3MDc5OTk5OTk5NiwtNzkuMzczMzA2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82OTZmYjczN2UyN2E0NGIzOGMwYWIxNzFiZDhkNzM4NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mOGU1ZmI1NzM0OTQ0YjBkODdjZWYwMjAxZGY2MGY5OCA9ICQoJzxkaXYgaWQ9Imh0bWxfZjhlNWZiNTczNDk0NGIwZDg3Y2VmMDIwMWRmNjBmOTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJlcmN6eSBQYXJrLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82OTZmYjczN2UyN2E0NGIzOGMwYWIxNzFiZDhkNzM4NS5zZXRDb250ZW50KGh0bWxfZjhlNWZiNTczNDk0NGIwZDg3Y2VmMDIwMWRmNjBmOTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTUyOGZmNTY5MWY0NDk5YmFhN2JmMDk3N2VhNWM4YjYuYmluZFBvcHVwKHBvcHVwXzY5NmZiNzM3ZTI3YTQ0YjM4YzBhYjE3MWJkOGQ3Mzg1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRhNGY4ZDFiMzNjZTQzNWFhNGRjZTQwYTU5ZDNhZDg2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3OTUyNCwtNzkuMzg3MzgyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82NjhkZTJkMDQwZGE0MTZiODc1YjNlMDMzZWU3ZTk4OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yZTdlODcxODM0OTI0YWI5YTAwZGZkMTE2ODk2YmQ2MCA9ICQoJzxkaXYgaWQ9Imh0bWxfMmU3ZTg3MTgzNDkyNGFiOWEwMGRmZDExNjg5NmJkNjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgQmF5IFN0cmVldCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjY4ZGUyZDA0MGRhNDE2Yjg3NWIzZTAzM2VlN2U5ODkuc2V0Q29udGVudChodG1sXzJlN2U4NzE4MzQ5MjRhYjlhMDBkZmQxMTY4OTZiZDYwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRhNGY4ZDFiMzNjZTQzNWFhNGRjZTQwYTU5ZDNhZDg2LmJpbmRQb3B1cChwb3B1cF82NjhkZTJkMDQwZGE0MTZiODc1YjNlMDMzZWU3ZTk4OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZGUwYjU4YTQzZTU0N2MwOWRhMWIwZGNkOWQ1YjZlMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDU3MTIwMDAwMDAxLC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg1ZTJmODc2OTQzNjQ2YmI4NmNlMTc4NWIwNjlkNTE4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzllNWJiODE1OGE2MzQzYTliZTljZGRjYjM2ZjAxY2QzID0gJCgnPGRpdiBpZD0iaHRtbF85ZTViYjgxNThhNjM0M2E5YmU5Y2RkY2IzNmYwMWNkMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmljaG1vbmQsIEFkZWxhaWRlLCBLaW5nLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84NWUyZjg3Njk0MzY0NmJiODZjZTE3ODViMDY5ZDUxOC5zZXRDb250ZW50KGh0bWxfOWU1YmI4MTU4YTYzNDNhOWJlOWNkZGNiMzZmMDFjZDMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWRlMGI1OGE0M2U1NDdjMDlkYTFiMGRjZDlkNWI2ZTMuYmluZFBvcHVwKHBvcHVwXzg1ZTJmODc2OTQzNjQ2YmI4NmNlMTc4NWIwNjlkNTE4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRjM2NjZmZhYTM4MDQ5MzM5MTU5ZDE5NGVjY2ZhZjRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywtNzkuMzgxNzUyMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTBmY2NhM2RmZDljNDliY2I2ZjE2YjJhZjBhYWFlYjcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDRiM2ZlODIxYWIxNDBkZWEwZDM3NGQ4YjYwMzBlMzkgPSAkKCc8ZGl2IGlkPSJodG1sXzA0YjNmZTgyMWFiMTQwZGVhMGQzNzRkOGI2MDMwZTM5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQgRWFzdCwgVW5pb24gU3RhdGlvbiwgVG9yb250byBJc2xhbmRzLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MGZjY2EzZGZkOWM0OWJjYjZmMTZiMmFmMGFhYWViNy5zZXRDb250ZW50KGh0bWxfMDRiM2ZlODIxYWIxNDBkZWEwZDM3NGQ4YjYwMzBlMzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGMzY2NmZmFhMzgwNDkzMzkxNTlkMTk0ZWNjZmFmNGIuYmluZFBvcHVwKHBvcHVwXzUwZmNjYTNkZmQ5YzQ5YmNiNmYxNmIyYWYwYWFhZWI3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhjZjc4NTg0NTFjOTQxMDE5MjljMzIxZDViMWE5ZTk2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ3MTc2OCwtNzkuMzgxNTc2NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTkzOWY1Nzk5MWVlNDQwNWJmZDBkMTUxZmE5ZTI0MzggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDY3YTNhYWFkNTcxNDU4MDlkZmJlZDVmMTM2MWI2Y2YgPSAkKCc8ZGl2IGlkPSJodG1sX2Q2N2EzYWFhZDU3MTQ1ODA5ZGZiZWQ1ZjEzNjFiNmNmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ub3JvbnRvIERvbWluaW9uIENlbnRyZSwgRGVzaWduIEV4Y2hhbmdlLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85OTM5ZjU3OTkxZWU0NDA1YmZkMGQxNTFmYTllMjQzOC5zZXRDb250ZW50KGh0bWxfZDY3YTNhYWFkNTcxNDU4MDlkZmJlZDVmMTM2MWI2Y2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGNmNzg1ODQ1MWM5NDEwMTkyOWMzMjFkNWIxYTllOTYuYmluZFBvcHVwKHBvcHVwXzk5MzlmNTc5OTFlZTQ0MDViZmQwZDE1MWZhOWUyNDM4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NjMGFiMDRhMmJkYTQzMDlhMjVlMjJjMGRmYmNmZTFkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4MTk4NSwtNzkuMzc5ODE2OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDBmNTIzMDgxNzcxNGM5NmE2MDMwZDQ5N2FhM2IzNmEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzM0ZGNkOTk4YTFmNDg4YmI4MmY5YzVlYzQzZDgzZWIgPSAkKCc8ZGl2IGlkPSJodG1sXzczNGRjZDk5OGExZjQ4OGJiODJmOWM1ZWM0M2Q4M2ViIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Db21tZXJjZSBDb3VydCwgVmljdG9yaWEgSG90ZWwsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQwZjUyMzA4MTc3MTRjOTZhNjAzMGQ0OTdhYTNiMzZhLnNldENvbnRlbnQoaHRtbF83MzRkY2Q5OThhMWY0ODhiYjgyZjljNWVjNDNkODNlYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jYzBhYjA0YTJiZGE0MzA5YTI1ZTIyYzBkZmJjZmUxZC5iaW5kUG9wdXAocG9wdXBfNDBmNTIzMDgxNzcxNGM5NmE2MDMwZDQ5N2FhM2IzNmEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmI0MTUzZDcxNjdjNDZkMTg5YzQyNjgzOTRkYmE0MzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MzMyODI1LC03OS40MTk3NDk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhlYjk2MGEwZTI2NjRiNTQ5YThlNzU4ZDU3Y2ZkYTg5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q2ZDc1MWRjODQ4MzRkYTk4MGQ1YjM4YzViMjM1MmY4ID0gJCgnPGRpdiBpZD0iaHRtbF9kNmQ3NTFkYzg0ODM0ZGE5ODBkNWIzOGM1YjIzNTJmOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVkZm9yZCBQYXJrLCBMYXdyZW5jZSBNYW5vciBFYXN0LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84ZWI5NjBhMGUyNjY0YjU0OWE4ZTc1OGQ1N2NmZGE4OS5zZXRDb250ZW50KGh0bWxfZDZkNzUxZGM4NDgzNGRhOTgwZDViMzhjNWIyMzUyZjgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmI0MTUzZDcxNjdjNDZkMTg5YzQyNjgzOTRkYmE0MzAuYmluZFBvcHVwKHBvcHVwXzhlYjk2MGEwZTI2NjRiNTQ5YThlNzU4ZDU3Y2ZkYTg5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2RhYjJkOWYxZDIxYTRiNjJhODI1MGYxYjM5MjI0NGY0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzExNjk0OCwtNzkuNDE2OTM1NTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDVhOTNjNDJjMTdjNGZmNWJmMjVjZWZkMDFiYzRhYzcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGM4YmIzMzhhMTZjNGMwMGI2ZTQ0OWRkMWZjZWFlOWIgPSAkKCc8ZGl2IGlkPSJodG1sXzRjOGJiMzM4YTE2YzRjMDBiNmU0NDlkZDFmY2VhZTliIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlbGF3biwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kNWE5M2M0MmMxN2M0ZmY1YmYyNWNlZmQwMWJjNGFjNy5zZXRDb250ZW50KGh0bWxfNGM4YmIzMzhhMTZjNGMwMGI2ZTQ0OWRkMWZjZWFlOWIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGFiMmQ5ZjFkMjFhNGI2MmE4MjUwZjFiMzkyMjQ0ZjQuYmluZFBvcHVwKHBvcHVwX2Q1YTkzYzQyYzE3YzRmZjViZjI1Y2VmZDAxYmM0YWM3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZmYzIwNmNmMzBmNTQwNDk5NmM0NWUzNTVmZTYyNmVkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjk2OTQ3NiwtNzkuNDExMzA3MjAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWQ1MDAwNjk1OGQ0NGJjYmI0NWE0ZWQ3YTc4ZjBhNzYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmI1ZjhlZjAxZWFiNGYzYjlkYmFmNzdhYWNjMTRlNzggPSAkKCc8ZGl2IGlkPSJodG1sX2JiNWY4ZWYwMWVhYjRmM2I5ZGJhZjc3YWFjYzE0ZTc4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Gb3Jlc3QgSGlsbCBOb3J0aCAmYW1wOyBXZXN0LCBGb3Jlc3QgSGlsbCBSb2FkIFBhcmssIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWQ1MDAwNjk1OGQ0NGJjYmI0NWE0ZWQ3YTc4ZjBhNzYuc2V0Q29udGVudChodG1sX2JiNWY4ZWYwMWVhYjRmM2I5ZGJhZjc3YWFjYzE0ZTc4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZmYzIwNmNmMzBmNTQwNDk5NmM0NWUzNTVmZTYyNmVkLmJpbmRQb3B1cChwb3B1cF9hZDUwMDA2OTU4ZDQ0YmNiYjQ1YTRlZDdhNzhmMGE3Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82NzQ4Mjc3YzViNzk0Y2I1YWMzZTZhMTcxYWZhMDNkMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MjcwOTcsLTc5LjQwNTY3ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzgxZjA4ODk2YzJmYzRjMDA4YjI2NjQyYjdmMzIwMzc1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RhMjk4ZTQ0ZDViYTRkZWJhNzlhMDhlYTI0ZWY4YTUzID0gJCgnPGRpdiBpZD0iaHRtbF9kYTI5OGU0NGQ1YmE0ZGViYTc5YTA4ZWEyNGVmOGE1MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEFubmV4LCBOb3J0aCBNaWR0b3duLCBZb3JrdmlsbGUsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODFmMDg4OTZjMmZjNGMwMDhiMjY2NDJiN2YzMjAzNzUuc2V0Q29udGVudChodG1sX2RhMjk4ZTQ0ZDViYTRkZWJhNzlhMDhlYTI0ZWY4YTUzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY3NDgyNzdjNWI3OTRjYjVhYzNlNmExNzFhZmEwM2QyLmJpbmRQb3B1cChwb3B1cF84MWYwODg5NmMyZmM0YzAwOGIyNjY0MmI3ZjMyMDM3NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yYWIyYzA1MWIyN2M0ZDNjOGM1MDU0YzNkOGQyMDI2MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjY5NTYsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWUwNWQzZjkzMTJlNDhkOWEzZDc2YWU2Y2I3ZGFjZjIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzQ1ZGY5N2JiMGYwNGVmZGIwZTNkMTBmNDM2NmRlNDYgPSAkKCc8ZGl2IGlkPSJodG1sXzc0NWRmOTdiYjBmMDRlZmRiMGUzZDEwZjQzNjZkZTQ2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIFRvcm9udG8sIEhhcmJvcmQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzllMDVkM2Y5MzEyZTQ4ZDlhM2Q3NmFlNmNiN2RhY2YyLnNldENvbnRlbnQoaHRtbF83NDVkZjk3YmIwZjA0ZWZkYjBlM2QxMGY0MzY2ZGU0Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYWIyYzA1MWIyN2M0ZDNjOGM1MDU0YzNkOGQyMDI2MS5iaW5kUG9wdXAocG9wdXBfOWUwNWQzZjkzMTJlNDhkOWEzZDc2YWU2Y2I3ZGFjZjIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDdmZTdkNTE0NGZhNDgzZThmNWQ2ZWFkMDM0OWUxN2IgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTMyMDU3LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EwMDk4NTgzZTQ1MDRiMjQ5N2FhZDZlMWJkMTk3OWNlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzExZTIyNWU4Yzk0YTQ1NDhhMDkwMWFlYjU3YzMzNmYyID0gJCgnPGRpdiBpZD0iaHRtbF8xMWUyMjVlOGM5NGE0NTQ4YTA5MDFhZWI1N2MzMzZmMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2Vuc2luZ3RvbiBNYXJrZXQsIENoaW5hdG93biwgR3JhbmdlIFBhcmssIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2EwMDk4NTgzZTQ1MDRiMjQ5N2FhZDZlMWJkMTk3OWNlLnNldENvbnRlbnQoaHRtbF8xMWUyMjVlOGM5NGE0NTQ4YTA5MDFhZWI1N2MzMzZmMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80N2ZlN2Q1MTQ0ZmE0ODNlOGY1ZDZlYWQwMzQ5ZTE3Yi5iaW5kUG9wdXAocG9wdXBfYTAwOTg1ODNlNDUwNGIyNDk3YWFkNmUxYmQxOTc5Y2UpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzZkMWJmNjc4MDliNDRjMzhkYzg2OWMxYmU3NzI2ZTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ4M2JkMTQ5MmQxMzQxNjA4YzE1NzM4MGM5MTdiNGM4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg5MWVmOTNiY2Q1ZjRjOWZhMDI2OGUwMjEzMDA2NDUyID0gJCgnPGRpdiBpZD0iaHRtbF84OTFlZjkzYmNkNWY0YzlmYTAyNjhlMDIxMzAwNjQ1MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q04gVG93ZXIsIEtpbmcgYW5kIFNwYWRpbmEsIFJhaWx3YXkgTGFuZHMsIEhhcmJvdXJmcm9udCBXZXN0LCBCYXRodXJzdCBRdWF5LCBTb3V0aCBOaWFnYXJhLCBJc2xhbmQgYWlycG9ydCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDgzYmQxNDkyZDEzNDE2MDhjMTU3MzgwYzkxN2I0Yzguc2V0Q29udGVudChodG1sXzg5MWVmOTNiY2Q1ZjRjOWZhMDI2OGUwMjEzMDA2NDUyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M2ZDFiZjY3ODA5YjQ0YzM4ZGM4NjljMWJlNzcyNmUyLmJpbmRQb3B1cChwb3B1cF80ODNiZDE0OTJkMTM0MTYwOGMxNTczODBjOTE3YjRjOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81NGRlOGFhMTMzMmY0MjExYTNiODMyN2NhOWVhNTc2NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NjQzNTIsLTc5LjM3NDg0NTk5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E4MjRkOWEwODkyOTQ3ZjI4M2U1YTEzNTZiMmI1NmE2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA5NmM1YjUzNGEzNzRiNjVhM2Y0NGU3MGFhODQzYzMwID0gJCgnPGRpdiBpZD0iaHRtbF8wOTZjNWI1MzRhMzc0YjY1YTNmNDRlNzBhYTg0M2MzMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3RuIEEgUE8gQm94ZXMsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E4MjRkOWEwODkyOTQ3ZjI4M2U1YTEzNTZiMmI1NmE2LnNldENvbnRlbnQoaHRtbF8wOTZjNWI1MzRhMzc0YjY1YTNmNDRlNzBhYTg0M2MzMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NGRlOGFhMTMzMmY0MjExYTNiODMyN2NhOWVhNTc2Ni5iaW5kUG9wdXAocG9wdXBfYTgyNGQ5YTA4OTI5NDdmMjgzZTVhMTM1NmIyYjU2YTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjBhYjNiZDJmNTYxNDg4OWE5MDE2YjkwNmU5NjQ1MTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg0MjkyLC03OS4zODIyODAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZjNGM0ZjQ0Mzg5NDRjZDU4MzJmODE5NDlmZDYwZGYzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzM3YTFmZjNlODYxZTRhOTRiMThjMWVhMmI2YmM0ZmRkID0gJCgnPGRpdiBpZD0iaHRtbF8zN2ExZmYzZTg2MWU0YTk0YjE4YzFlYTJiNmJjNGZkZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rmlyc3QgQ2FuYWRpYW4gUGxhY2UsIFVuZGVyZ3JvdW5kIGNpdHksIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZjNGM0ZjQ0Mzg5NDRjZDU4MzJmODE5NDlmZDYwZGYzLnNldENvbnRlbnQoaHRtbF8zN2ExZmYzZTg2MWU0YTk0YjE4YzFlYTJiNmJjNGZkZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MGFiM2JkMmY1NjE0ODg5YTkwMTZiOTA2ZTk2NDUxNy5iaW5kUG9wdXAocG9wdXBfZmM0YzRmNDQzODk0NGNkNTgzMmY4MTk0OWZkNjBkZjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWEyZGYyN2M2NjIwNDhiMjljYzM4YzI4ZWMyYWE2MDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTg1MTc5OTk5OTk5OTYsLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZmM2RkYmViZmJhNDRlYjViMzdjNjI0NzUyMjNlNGRhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYzZWJhZTZiNjQyOTRkNjA5NzU1Y2Q3ZDM0OTU2M2ZhID0gJCgnPGRpdiBpZD0iaHRtbF82M2ViYWU2YjY0Mjk0ZDYwOTc1NWNkN2QzNDk1NjNmYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGF3cmVuY2UgTWFub3IsIExhd3JlbmNlIEhlaWdodHMsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZmM2RkYmViZmJhNDRlYjViMzdjNjI0NzUyMjNlNGRhLnNldENvbnRlbnQoaHRtbF82M2ViYWU2YjY0Mjk0ZDYwOTc1NWNkN2QzNDk1NjNmYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hYTJkZjI3YzY2MjA0OGIyOWNjMzhjMjhlYzJhYTYwMC5iaW5kUG9wdXAocG9wdXBfNmYzZGRiZWJmYmE0NGViNWIzN2M2MjQ3NTIyM2U0ZGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDg5MmJiMDNjNDYyNGJlY2JiMTMzYTgwMGE2OTNiYzYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDk1NzcsLTc5LjQ0NTA3MjU5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y4YTI5ZGMwZmViNjQxYTk4NWQ2YmJjMDhjNDE5ZDNmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc5MmYwNjA2YzgxNTQ1OWJhNmY2YjFjMTA2YzFmMmY3ID0gJCgnPGRpdiBpZD0iaHRtbF83OTJmMDYwNmM4MTU0NTliYTZmNmIxYzEwNmMxZjJmNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R2xlbmNhaXJuLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mOGEyOWRjMGZlYjY0MWE5ODVkNmJiYzA4YzQxOWQzZi5zZXRDb250ZW50KGh0bWxfNzkyZjA2MDZjODE1NDU5YmE2ZjZiMWMxMDZjMWYyZjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDg5MmJiMDNjNDYyNGJlY2JiMTMzYTgwMGE2OTNiYzYuYmluZFBvcHVwKHBvcHVwX2Y4YTI5ZGMwZmViNjQxYTk4NWQ2YmJjMDhjNDE5ZDNmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NhODBmOGY3NmZiODQxNDM5Mjk4NmMyNzdiNWI5NDBhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkzNzgxMywtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfN2JkMmZiNWYyNzA2NDczM2IxNmExZDAzYTA3MDAwMTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDNiMWMwNTUyOGRjNDc1ZTgxM2IxYzU2MWUyMjY2NTUgPSAkKCc8ZGl2IGlkPSJodG1sXzAzYjFjMDU1MjhkYzQ3NWU4MTNiMWM1NjFlMjI2NjU1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IdW1ld29vZC1DZWRhcnZhbGUsIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdiZDJmYjVmMjcwNjQ3MzNiMTZhMWQwM2EwNzAwMDE1LnNldENvbnRlbnQoaHRtbF8wM2IxYzA1NTI4ZGM0NzVlODEzYjFjNTYxZTIyNjY1NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jYTgwZjhmNzZmYjg0MTQzOTI5ODZjMjc3YjViOTQwYS5iaW5kUG9wdXAocG9wdXBfN2JkMmZiNWYyNzA2NDczM2IxNmExZDAzYTA3MDAwMTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzA2MjNkNWNhMDEzNDZhNGE5NDFjNmEwOTA4NDk1OTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODkwMjU2LC03OS40NTM1MTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmNjZWRkNDQzZGFlNDMyZDg2YTdhNTM1NTY5MjZjMWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzg4NjYxNTQ5OGE3NDBiMjg5MTYxN2RmZDRlMDIwOTYgPSAkKCc8ZGl2IGlkPSJodG1sXzc4ODY2MTU0OThhNzQwYjI4OTE2MTdkZmQ0ZTAyMDk2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYWxlZG9uaWEtRmFpcmJhbmtzLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82Y2NlZGQ0NDNkYWU0MzJkODZhN2E1MzU1NjkyNmMxYy5zZXRDb250ZW50KGh0bWxfNzg4NjYxNTQ5OGE3NDBiMjg5MTYxN2RmZDRlMDIwOTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzA2MjNkNWNhMDEzNDZhNGE5NDFjNmEwOTA4NDk1OTAuYmluZFBvcHVwKHBvcHVwXzZjY2VkZDQ0M2RhZTQzMmQ4NmE3YTUzNTU2OTI2YzFjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJiODQ0YzhkOTQ3YzQ3NDlhM2E5MWU4NGFkNDk2ODk3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5NTQyLC03OS40MjI1NjM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U5YTk0YzI4MmZkNjQ4ZGQ5NGM0MTUwY2ZhMjkxZjA4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YxZDU0OWFhNjhiNDRjZGQ4NDcwYmNjZjM5NTJiOWRjID0gJCgnPGRpdiBpZD0iaHRtbF9mMWQ1NDlhYTY4YjQ0Y2RkODQ3MGJjY2YzOTUyYjlkYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hyaXN0aWUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U5YTk0YzI4MmZkNjQ4ZGQ5NGM0MTUwY2ZhMjkxZjA4LnNldENvbnRlbnQoaHRtbF9mMWQ1NDlhYTY4YjQ0Y2RkODQ3MGJjY2YzOTUyYjlkYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYjg0NGM4ZDk0N2M0NzQ5YTNhOTFlODRhZDQ5Njg5Ny5iaW5kUG9wdXAocG9wdXBfZTlhOTRjMjgyZmQ2NDhkZDk0YzQxNTBjZmEyOTFmMDgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTJlZjM0ZTQyNWNlNDMwNmJjMTU5NTYwM2IwM2VlYWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjkwMDUxMDAwMDAwMSwtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNTNhZDhkNDU3YTU0NjQ0OGJmMGFkYmJmNTFlYTNjMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZGUyYmRlZDNhMzQ0YTEzYjExNmVlYWJmMmI4ZTcxZSA9ICQoJzxkaXYgaWQ9Imh0bWxfOWRlMmJkZWQzYTM0NGExM2IxMTZlZWFiZjJiOGU3MWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkR1ZmZlcmluLCBEb3ZlcmNvdXJ0IFZpbGxhZ2UsIFdlc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjUzYWQ4ZDQ1N2E1NDY0NDhiZjBhZGJiZjUxZWEzYzEuc2V0Q29udGVudChodG1sXzlkZTJiZGVkM2EzNDRhMTNiMTE2ZWVhYmYyYjhlNzFlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzkyZWYzNGU0MjVjZTQzMDZiYzE1OTU2MDNiMDNlZWFlLmJpbmRQb3B1cChwb3B1cF9mNTNhZDhkNDU3YTU0NjQ0OGJmMGFkYmJmNTFlYTNjMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMjUwYjljODIwMmU0NzIzODg5MDliZDI1YmM4NjQ5MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMjRhZTlmNjQ0ZjY0ZmQwYjI4ZjFhNDc3OGE3YmFkNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jYjQyY2ZkYmVjNDQ0NTYxYjI4MGJkOGI5OWUwNjA4ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfY2I0MmNmZGJlYzQ0NDU2MWIyODBiZDhiOTllMDYwOGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxpdHRsZSBQb3J0dWdhbCwgVHJpbml0eSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMjRhZTlmNjQ0ZjY0ZmQwYjI4ZjFhNDc3OGE3YmFkNi5zZXRDb250ZW50KGh0bWxfY2I0MmNmZGJlYzQ0NDU2MWIyODBiZDhiOTllMDYwOGUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTI1MGI5YzgyMDJlNDcyMzg4OTA5YmQyNWJjODY0OTIuYmluZFBvcHVwKHBvcHVwX2IyNGFlOWY2NDRmNjRmZDBiMjhmMWE0Nzc4YTdiYWQ2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y2YWRhMmFiZTI3NzRkZWU4MTc4ZDliZTgzZWQ3MmNhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2ODQ3MiwtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWI1MDA2ODA5MTQ5NGVmMjk4Mzk4NzdkYTgwMmQwYmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDFiMDk5YTBmZmUxNDNlMmI3MjNkMzhiMjRiOTQ4Y2MgPSAkKCc8ZGl2IGlkPSJodG1sXzAxYjA5OWEwZmZlMTQzZTJiNzIzZDM4YjI0Yjk0OGNjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ccm9ja3RvbiwgUGFya2RhbGUgVmlsbGFnZSwgRXhoaWJpdGlvbiBQbGFjZSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xYjUwMDY4MDkxNDk0ZWYyOTgzOTg3N2RhODAyZDBiZi5zZXRDb250ZW50KGh0bWxfMDFiMDk5YTBmZmUxNDNlMmI3MjNkMzhiMjRiOTQ4Y2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjZhZGEyYWJlMjc3NGRlZTgxNzhkOWJlODNlZDcyY2EuYmluZFBvcHVwKHBvcHVwXzFiNTAwNjgwOTE0OTRlZjI5ODM5ODc3ZGE4MDJkMGJmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VhMTRmMWJhYTc5NDRlZWViODNkZDQyZDAyNmY1YTAxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEzNzU2MjAwMDAwMDA2LC03OS40OTAwNzM4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNiNGYyZTQ5N2U5YTQ2MjFiYTljMmM3YWE2OTg4MWFhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZhZTA1NTdiNzAwZTQ5MjRiNTAzOTUyMGJjZWMzMjRjID0gJCgnPGRpdiBpZD0iaHRtbF9mYWUwNTU3YjcwMGU0OTI0YjUwMzk1MjBiY2VjMzI0YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGggUGFyaywgTWFwbGUgTGVhZiBQYXJrLCBVcHdvb2QgUGFyaywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2I0ZjJlNDk3ZTlhNDYyMWJhOWMyYzdhYTY5ODgxYWEuc2V0Q29udGVudChodG1sX2ZhZTA1NTdiNzAwZTQ5MjRiNTAzOTUyMGJjZWMzMjRjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VhMTRmMWJhYTc5NDRlZWViODNkZDQyZDAyNmY1YTAxLmJpbmRQb3B1cChwb3B1cF8zYjRmMmU0OTdlOWE0NjIxYmE5YzJjN2FhNjk4ODFhYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MWYwMDY3MWYxMjA0YWUxYWE5Nzg3YThkMWI1NTRjOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5MTExNTgsLTc5LjQ3NjAxMzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ2ZTgzZjcwM2YxZTRkN2I4OTFmYmUzYWNjM2ZlNTExID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhjYjU3OWMyMTdmNzQzZjJhYWE2Y2Y5Y2UyMzE3ODRkID0gJCgnPGRpdiBpZD0iaHRtbF84Y2I1NzljMjE3Zjc0M2YyYWFhNmNmOWNlMjMxNzg0ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGVsIFJheSwgTW91bnQgRGVubmlzLCBLZWVsc2RhbGUgYW5kIFNpbHZlcnRob3JuLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80NmU4M2Y3MDNmMWU0ZDdiODkxZmJlM2FjYzNmZTUxMS5zZXRDb250ZW50KGh0bWxfOGNiNTc5YzIxN2Y3NDNmMmFhYTZjZjljZTIzMTc4NGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjFmMDA2NzFmMTIwNGFlMWFhOTc4N2E4ZDFiNTU0YzkuYmluZFBvcHVwKHBvcHVwXzQ2ZTgzZjcwM2YxZTRkN2I4OTFmYmUzYWNjM2ZlNTExKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E4ZmQ5ZWQ2NDc0NDRlMjliNTk0MzlkNThkODc0M2MxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjczMTg1Mjk5OTk5OTksLTc5LjQ4NzI2MTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ExYmEwOGYyYjY5YTQ2ZmFiYWNkY2I0NjAxOTEwMzVjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlmMDc2YzFkNjRkNDRkODA4YWRmYzY1ZjYyNTljOGI1ID0gJCgnPGRpdiBpZD0iaHRtbF85ZjA3NmMxZDY0ZDQ0ZDgwOGFkZmM2NWY2MjU5YzhiNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBUaGUgSnVuY3Rpb24gTm9ydGgsIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ExYmEwOGYyYjY5YTQ2ZmFiYWNkY2I0NjAxOTEwMzVjLnNldENvbnRlbnQoaHRtbF85ZjA3NmMxZDY0ZDQ0ZDgwOGFkZmM2NWY2MjU5YzhiNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hOGZkOWVkNjQ3NDQ0ZTI5YjU5NDM5ZDU4ZDg3NDNjMS5iaW5kUG9wdXAocG9wdXBfYTFiYTA4ZjJiNjlhNDZmYWJhY2RjYjQ2MDE5MTAzNWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODQyMjIyZjA5YTlkNDRjNDljNzY5OTBkMWIwYWJiZTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjE2MDgzLC03OS40NjQ3NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZTZkZjQzYWMxMTA0ODA0YWZjYWUwM2ZmYThiZjg2ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lMGU4ODVmZTNmZTc0OWZjOWNhYTZlNjk4NjEzNmZlYyA9ICQoJzxkaXYgaWQ9Imh0bWxfZTBlODg1ZmUzZmU3NDlmYzljYWE2ZTY5ODYxMzZmZWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhpZ2ggUGFyaywgVGhlIEp1bmN0aW9uIFNvdXRoLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzllNmRmNDNhYzExMDQ4MDRhZmNhZTAzZmZhOGJmODZlLnNldENvbnRlbnQoaHRtbF9lMGU4ODVmZTNmZTc0OWZjOWNhYTZlNjk4NjEzNmZlYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NDIyMjJmMDlhOWQ0NGM0OWM3Njk5MGQxYjBhYmJlOS5iaW5kUG9wdXAocG9wdXBfOWU2ZGY0M2FjMTEwNDgwNGFmY2FlMDNmZmE4YmY4NmUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODcyNjEzMTAxMzBjNGM3Yjk5NmZjZWE0NTc2OTRjNGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg5NTk3LC03OS40NTYzMjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWJlMDQyYzU2MWI0NDU4Yzk4MTRjN2RkOTVmMzE0ZjAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmVmZWJkZWM1NWY4NDkxODk5NTQ5NDAwOTVjMGEzZjUgPSAkKCc8ZGl2IGlkPSJodG1sX2ZlZmViZGVjNTVmODQ5MTg5OTU0OTQwMDk1YzBhM2Y1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJrZGFsZSwgUm9uY2VzdmFsbGVzLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzliZTA0MmM1NjFiNDQ1OGM5ODE0YzdkZDk1ZjMxNGYwLnNldENvbnRlbnQoaHRtbF9mZWZlYmRlYzU1Zjg0OTE4OTk1NDk0MDA5NWMwYTNmNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NzI2MTMxMDEzMGM0YzdiOTk2ZmNlYTQ1NzY5NGM0ZS5iaW5kUG9wdXAocG9wdXBfOWJlMDQyYzU2MWI0NDU4Yzk4MTRjN2RkOTVmMzE0ZjApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDVmMTZmNDc3MjhmNGI0Mzk2MmFlNTkzNDY1NDM4N2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE1NzA2LC03OS40ODQ0NDk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzgxYzAzODA1NGNkOTRhMWNhZWQxYTdhODk5ZThmMGI3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhmNzBhOTE4NjQ4ODQ0MmViYzgxYmIzMjMyMDc4NWYyID0gJCgnPGRpdiBpZD0iaHRtbF84ZjcwYTkxODY0ODg0NDJlYmM4MWJiMzIzMjA3ODVmMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBTd2Fuc2VhLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzgxYzAzODA1NGNkOTRhMWNhZWQxYTdhODk5ZThmMGI3LnNldENvbnRlbnQoaHRtbF84ZjcwYTkxODY0ODg0NDJlYmM4MWJiMzIzMjA3ODVmMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wNWYxNmY0NzcyOGY0YjQzOTYyYWU1OTM0NjU0Mzg3ZS5iaW5kUG9wdXAocG9wdXBfODFjMDM4MDU0Y2Q5NGExY2FlZDFhN2E4OTllOGYwYjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmFjZDAyMDhlOTM2NDhmZjgzYmYyZDE4NGI2YTg1ZjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjIzMDE1LC03OS4zODk0OTM4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IyYWY5NThiZmVkNDQzZTc5YjMxZDFhNDBmYmQ1ZDBkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y5YjYyYjM4YTBlOTQ1N2ZhZGU2ZjhiNmQxY2M4ZGE3ID0gJCgnPGRpdiBpZD0iaHRtbF9mOWI2MmIzOGEwZTk0NTdmYWRlNmY4YjZkMWNjOGRhNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UXVlZW4mIzM5O3MgUGFyaywgT250YXJpbyBQcm92aW5jaWFsIEdvdmVybm1lbnQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IyYWY5NThiZmVkNDQzZTc5YjMxZDFhNDBmYmQ1ZDBkLnNldENvbnRlbnQoaHRtbF9mOWI2MmIzOGEwZTk0NTdmYWRlNmY4YjZkMWNjOGRhNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mYWNkMDIwOGU5MzY0OGZmODNiZjJkMTg0YjZhODVmOC5iaW5kUG9wdXAocG9wdXBfYjJhZjk1OGJmZWQ0NDNlNzliMzFkMWE0MGZiZDVkMGQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGQ2OTcwZDZiZmE2NGJkZGI5OTgyMzQ3MjE4YWE5NWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY5NjU2LC03OS42MTU4MTg5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85MTYzNzlkOThkOGY0MTIwOTVkOGIwMGI0OWRjZmJmMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNDUwZjAzZTQ5ODI0MThiODc5OGJmZGZhZjBmYTQ4ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfZTQ1MGYwM2U0OTgyNDE4Yjg3OThiZmRmYWYwZmE0OGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhbmFkYSBQb3N0IEdhdGV3YXkgUHJvY2Vzc2luZyBDZW50cmUsIE1pc3Npc3NhdWdhPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MTYzNzlkOThkOGY0MTIwOTVkOGIwMGI0OWRjZmJmMi5zZXRDb250ZW50KGh0bWxfZTQ1MGYwM2U0OTgyNDE4Yjg3OThiZmRmYWYwZmE0OGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGQ2OTcwZDZiZmE2NGJkZGI5OTgyMzQ3MjE4YWE5NWQuYmluZFBvcHVwKHBvcHVwXzkxNjM3OWQ5OGQ4ZjQxMjA5NWQ4YjAwYjQ5ZGNmYmYyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA4NDExOGRkOWQ1MTRlMzk5NmYzMTIyZDViZjQ5M2I2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNzQzOSwtNzkuMzIxNTU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ1NTk4MGVkYmZhMjRmNmVhN2I1YmVjNDU5MTgzNWRjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FjYzNkMDVjZDg3MDQzMmJhZjE5YjIwMThmZWVlOTI0ID0gJCgnPGRpdiBpZD0iaHRtbF9hY2MzZDA1Y2Q4NzA0MzJiYWYxOWIyMDE4ZmVlZTkyNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QnVzaW5lc3MgcmVwbHkgbWFpbCBQcm9jZXNzaW5nIENlbnRyZSwgU291dGggQ2VudHJhbCBMZXR0ZXIgUHJvY2Vzc2luZyBQbGFudCBUb3JvbnRvLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ1NTk4MGVkYmZhMjRmNmVhN2I1YmVjNDU5MTgzNWRjLnNldENvbnRlbnQoaHRtbF9hY2MzZDA1Y2Q4NzA0MzJiYWYxOWIyMDE4ZmVlZTkyNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wODQxMThkZDlkNTE0ZTM5OTZmMzEyMmQ1YmY0OTNiNi5iaW5kUG9wdXAocG9wdXBfNDU1OTgwZWRiZmEyNGY2ZWE3YjViZWM0NTkxODM1ZGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjgwZTViYTAzYTE0NDZlYThhZGVkZjZkZmE4YzI0YzYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MDU2NDY2LC03OS41MDEzMjA3MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MjY3NjU3NDg4Y2Y0MDAyYmMxNWU4YjM5YWEyNjBiYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83ZGNkMDNjNDI1ZGQ0YTVkYmE3YjY0N2I5MTk2Yjk2NSA9ICQoJzxkaXYgaWQ9Imh0bWxfN2RjZDAzYzQyNWRkNGE1ZGJhN2I2NDdiOTE5NmI5NjUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5ldyBUb3JvbnRvLCBNaW1pY28gU291dGgsIEh1bWJlciBCYXkgU2hvcmVzLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUyNjc2NTc0ODhjZjQwMDJiYzE1ZThiMzlhYTI2MGJjLnNldENvbnRlbnQoaHRtbF83ZGNkMDNjNDI1ZGQ0YTVkYmE3YjY0N2I5MTk2Yjk2NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mODBlNWJhMDNhMTQ0NmVhOGFkZWRmNmRmYThjMjRjNi5iaW5kUG9wdXAocG9wdXBfNTI2NzY1NzQ4OGNmNDAwMmJjMTVlOGIzOWFhMjYwYmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTU1NGM5NGRjODBmNDcwZGFmY2E5NzY1NzExM2FiMWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MDI0MTM3MDAwMDAwMSwtNzkuNTQzNDg0MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWY0ODE2ZjcxODg2NDYyOGJlODk4YjQ3NjA5OWNjMDQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDgxYjFjOWYxZWI4NGZkOTlhYjcyZDM2NzViNDEzZDEgPSAkKCc8ZGl2IGlkPSJodG1sXzQ4MWIxYzlmMWViODRmZDk5YWI3MmQzNjc1YjQxM2QxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BbGRlcndvb2QsIExvbmcgQnJhbmNoLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FmNDgxNmY3MTg4NjQ2MjhiZTg5OGI0NzYwOTljYzA0LnNldENvbnRlbnQoaHRtbF80ODFiMWM5ZjFlYjg0ZmQ5OWFiNzJkMzY3NWI0MTNkMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NTU0Yzk0ZGM4MGY0NzBkYWZjYTk3NjU3MTEzYWIxYy5iaW5kUG9wdXAocG9wdXBfYWY0ODE2ZjcxODg2NDYyOGJlODk4YjQ3NjA5OWNjMDQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGRhMmIwN2MyZGVhNDkwMmE1YmUxMWZkODRmYWJiYWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTM2NTM2MDAwMDAwMDUsLTc5LjUwNjk0MzZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGZjY2EzYTBmNzYwNDI5NmJjODczMTgwMDUyOGZiN2EgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTJmOTVjMjc2NDU3NGUxNWIzY2U5NzkxOGM5NTMwMWUgPSAkKCc8ZGl2IGlkPSJodG1sXzEyZjk1YzI3NjQ1NzRlMTViM2NlOTc5MThjOTUzMDFlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgS2luZ3N3YXksIE1vbnRnb21lcnkgUm9hZCwgT2xkIE1pbGwgTm9ydGgsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGZjY2EzYTBmNzYwNDI5NmJjODczMTgwMDUyOGZiN2Euc2V0Q29udGVudChodG1sXzEyZjk1YzI3NjQ1NzRlMTViM2NlOTc5MThjOTUzMDFlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBkYTJiMDdjMmRlYTQ5MDJhNWJlMTFmZDg0ZmFiYmFlLmJpbmRQb3B1cChwb3B1cF80ZmNjYTNhMGY3NjA0Mjk2YmM4NzMxODAwNTI4ZmI3YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNmFkNTI5ZWYyNmY0ODFkOGUzZDc4YzFhYzZhN2Y0YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjI1NzksLTc5LjQ5ODUwOTA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNmNWY3MGJmMTE5NzRhZDk4NDg1NDQyNWE4MWMxY2JmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNiOGJiMGVjNzAzNDQyYWE4MDQ0MWJjZTkwYjU4MWMzID0gJCgnPGRpdiBpZD0iaHRtbF8zYjhiYjBlYzcwMzQ0MmFhODA0NDFiY2U5MGI1ODFjMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+T2xkIE1pbGwgU291dGgsIEtpbmcmIzM5O3MgTWlsbCBQYXJrLCBTdW5ueWxlYSwgSHVtYmVyIEJheSwgTWltaWNvIE5FLCBUaGUgUXVlZW5zd2F5IEVhc3QsIFJveWFsIFlvcmsgU291dGggRWFzdCwgS2luZ3N3YXkgUGFyayBTb3V0aCBFYXN0LCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNmNWY3MGJmMTE5NzRhZDk4NDg1NDQyNWE4MWMxY2JmLnNldENvbnRlbnQoaHRtbF8zYjhiYjBlYzcwMzQ0MmFhODA0NDFiY2U5MGI1ODFjMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iNmFkNTI5ZWYyNmY0ODFkOGUzZDc4YzFhYzZhN2Y0YS5iaW5kUG9wdXAocG9wdXBfM2Y1ZjcwYmYxMTk3NGFkOTg0ODU0NDI1YTgxYzFjYmYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzA2OWNiZmU1YzkwNDdlN2I0ZDkzYjU1ZTExZmU3NGQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg4NDA4LC03OS41MjA5OTk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iM2I3ZTZkYzE4MjQ0MGE4OTcwMGM5NGM2ZWQ1NWFkNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wM2NjZWZmZmNlMDk0M2UyYjg4YTkwMGY2Y2NjMThlMSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDNjY2VmZmZjZTA5NDNlMmI4OGE5MDBmNmNjYzE4ZTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1pbWljbyBOVywgVGhlIFF1ZWVuc3dheSBXZXN0LCBTb3V0aCBvZiBCbG9vciwgS2luZ3N3YXkgUGFyayBTb3V0aCBXZXN0LCBSb3lhbCBZb3JrIFNvdXRoIFdlc3QsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjNiN2U2ZGMxODI0NDBhODk3MDBjOTRjNmVkNTVhZDcuc2V0Q29udGVudChodG1sXzAzY2NlZmZmY2UwOTQzZTJiODhhOTAwZjZjY2MxOGUxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2MwNjljYmZlNWM5MDQ3ZTdiNGQ5M2I1NWUxMWZlNzRkLmJpbmRQb3B1cChwb3B1cF9iM2I3ZTZkYzE4MjQ0MGE4OTcwMGM5NGM2ZWQ1NWFkNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMDAyYzc0ZjQxMWU0MjUyODY3YjljYTA4YzIzZWFlOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzg1NTYsLTc5LjUzMjI0MjQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVhMWI3OGVkZWQ5OTRlM2U4ZDBkMmUwNzA3MWYwNjEwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY3ZWZiY2IzMjdjNzRkY2M4OTE1YjYyYWI4YWY0ODkyID0gJCgnPGRpdiBpZD0iaHRtbF82N2VmYmNiMzI3Yzc0ZGNjODkxNWI2MmFiOGFmNDg5MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SXNsaW5ndG9uIEF2ZW51ZSwgSHVtYmVyIFZhbGxleSBWaWxsYWdlLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVhMWI3OGVkZWQ5OTRlM2U4ZDBkMmUwNzA3MWYwNjEwLnNldENvbnRlbnQoaHRtbF82N2VmYmNiMzI3Yzc0ZGNjODkxNWI2MmFiOGFmNDg5Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kMDAyYzc0ZjQxMWU0MjUyODY3YjljYTA4YzIzZWFlOC5iaW5kUG9wdXAocG9wdXBfNWExYjc4ZWRlZDk5NGUzZThkMGQyZTA3MDcxZjA2MTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmMwNDM2NWIzY2I5NDAzNTgyOGExY2U4ODQ5N2U3YjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA5NDMyLC03OS41NTQ3MjQ0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iYmVhZTQ1YzE3Njc0YTg5YmMxNWFhOWEwYmRiYjlhMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hYmIzN2NkMWMyODM0OTNlOGE3NGE1M2ZhZDI0ZTIxNSA9ICQoJzxkaXYgaWQ9Imh0bWxfYWJiMzdjZDFjMjgzNDkzZThhNzRhNTNmYWQyNGUyMTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3QgRGVhbmUgUGFyaywgUHJpbmNlc3MgR2FyZGVucywgTWFydGluIEdyb3ZlLCBJc2xpbmd0b24sIENsb3ZlcmRhbGUsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmJlYWU0NWMxNzY3NGE4OWJjMTVhYTlhMGJkYmI5YTEuc2V0Q29udGVudChodG1sX2FiYjM3Y2QxYzI4MzQ5M2U4YTc0YTUzZmFkMjRlMjE1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZjMDQzNjViM2NiOTQwMzU4MjhhMWNlODg0OTdlN2IyLmJpbmRQb3B1cChwb3B1cF9iYmVhZTQ1YzE3Njc0YTg5YmMxNWFhOWEwYmRiYjlhMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jOTY5NTFhNjE5NDc0ZDFmYTNjZGZlYWY4NDk2ZjI0OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MzUxNTIsLTc5LjU3NzIwMDc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzdkMjBiY2M3MmFiYjQ2ZGQ4Yjc4OTAxODFkMmE1ZmUxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2IxM2UyMTA2MTc5ZjRmMGY5NDU2ZmQxMjBhYTIxMTc4ID0gJCgnPGRpdiBpZD0iaHRtbF9iMTNlMjEwNjE3OWY0ZjBmOTQ1NmZkMTIwYWEyMTE3OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RXJpbmdhdGUsIEJsb29yZGFsZSBHYXJkZW5zLCBPbGQgQnVybmhhbXRob3JwZSwgTWFya2xhbmQgV29vZCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZDIwYmNjNzJhYmI0NmRkOGI3ODkwMTgxZDJhNWZlMS5zZXRDb250ZW50KGh0bWxfYjEzZTIxMDYxNzlmNGYwZjk0NTZmZDEyMGFhMjExNzgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzk2OTUxYTYxOTQ3NGQxZmEzY2RmZWFmODQ5NmYyNDguYmluZFBvcHVwKHBvcHVwXzdkMjBiY2M3MmFiYjQ2ZGQ4Yjc4OTAxODFkMmE1ZmUxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzMyOWI5NWRjNmE2NzRkNjZiMWMzZTQ0YzNlNzY1NzM4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU2MzAzMywtNzkuNTY1OTYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWVlOTkwMTdhZTc4NGZhY2EwZTg4NzQyMWIyZTcxZWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTQ1OTgxNWJkOTJjNDIwOGIzOWQ0MDMyNjFhZTY0NGIgPSAkKCc8ZGl2IGlkPSJodG1sX2U0NTk4MTViZDkyYzQyMDhiMzlkNDAzMjYxYWU2NDRiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IdW1iZXIgU3VtbWl0LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lZWU5OTAxN2FlNzg0ZmFjYTBlODg3NDIxYjJlNzFlZi5zZXRDb250ZW50KGh0bWxfZTQ1OTgxNWJkOTJjNDIwOGIzOWQ0MDMyNjFhZTY0NGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzI5Yjk1ZGM2YTY3NGQ2NmIxYzNlNDRjM2U3NjU3MzguYmluZFBvcHVwKHBvcHVwX2VlZTk5MDE3YWU3ODRmYWNhMGU4ODc0MjFiMmU3MWVmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MzNTI2MTM0MTg1MzQ4ZGNhYWM3ZDc4NjYyMTY4MGRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI0NzY1OSwtNzkuNTMyMjQyNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjA0NzY1ZmM4MDkwNGU5ZDliNzhlZWNiMDRlNjQxOWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjcyYzU0YWZjYmMyNDNhYmJhY2FkZGU4ZjhkOWM5NDAgPSAkKCc8ZGl2IGlkPSJodG1sXzI3MmM1NGFmY2JjMjQzYWJiYWNhZGRlOGY4ZDljOTQwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IdW1iZXJsZWEsIEVtZXJ5LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yMDQ3NjVmYzgwOTA0ZTlkOWI3OGVlY2IwNGU2NDE5ZS5zZXRDb250ZW50KGh0bWxfMjcyYzU0YWZjYmMyNDNhYmJhY2FkZGU4ZjhkOWM5NDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzM1MjYxMzQxODUzNDhkY2FhYzdkNzg2NjIxNjgwZGIuYmluZFBvcHVwKHBvcHVwXzIwNDc2NWZjODA5MDRlOWQ5Yjc4ZWVjYjA0ZTY0MTllKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBhNmExZDQwYWNkMTQ1MDg5YWFkNjIyZGEwZDdkNTAyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2ODc2LC03OS41MTgxODg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNzQxNzliZmMyNzE0NjE1YjUxOTU4OWE2YTA1NWUyNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MTE4Y2I0YTNhMjk0MGIzOGVmNmI1OTUwYmQ1YjQ2ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfODExOGNiNGEzYTI5NDBiMzhlZjZiNTk1MGJkNWI0NmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3RvbiwgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjc0MTc5YmZjMjcxNDYxNWI1MTk1ODlhNmEwNTVlMjYuc2V0Q29udGVudChodG1sXzgxMThjYjRhM2EyOTQwYjM4ZWY2YjU5NTBiZDViNDZlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBhNmExZDQwYWNkMTQ1MDg5YWFkNjIyZGEwZDdkNTAyLmJpbmRQb3B1cChwb3B1cF9mNzQxNzliZmMyNzE0NjE1YjUxOTU4OWE2YTA1NWUyNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNTdlNTY2NWFhMGQ0YjBiYmRhOTViOWViYjhmNTMwNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5NjMxOSwtNzkuNTMyMjQyNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZjAzYmQwZjI0MDU3NGNlMThkMTI0ZjhkM2Y2MzYzYTMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWRlNzYwZjlmMDIxNGVkN2EwMDBjODhlMWY2ZWUwNzQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDc1ZWQxOWFhN2RhNGY0MWJhYmIyN2I0MWI0NjZhMDUgPSAkKCc8ZGl2IGlkPSJodG1sXzA3NWVkMTlhYTdkYTRmNDFiYWJiMjdiNDFiNDY2YTA1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0bW91bnQsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWRlNzYwZjlmMDIxNGVkN2EwMDBjODhlMWY2ZWUwNzQuc2V0Q29udGVudChodG1sXzA3NWVkMTlhYTdkYTRmNDFiYWJiMjdiNDFiNDY2YTA1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y1N2U1NjY1YWEwZDRiMGJiZGE5NWI5ZWJiOGY1MzA1LmJpbmRQb3B1cChwb3B1cF9hZGU3NjBmOWYwMjE0ZWQ3YTAwMGM4OGUxZjZlZTA3NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mOTQ1MTQ4NjA4NDU0ZDU1OTdmYTQ1OGZhNWI1M2JjYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4ODkwNTQsLTc5LjU1NDcyNDQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2YwM2JkMGYyNDA1NzRjZTE4ZDEyNGY4ZDNmNjM2M2EzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E2N2RkOGYyYTYyYzQ4MjdiMDNjMjNkY2FkNGRiMDhmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U1YjI5OGM1YTI4YjQ0MDQ4ZDg4NzVlZjEzMjA5NjhkID0gJCgnPGRpdiBpZD0iaHRtbF9lNWIyOThjNWEyOGI0NDA0OGQ4ODc1ZWYxMzIwOTY4ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2luZ3N2aWV3IFZpbGxhZ2UsIFN0LiBQaGlsbGlwcywgTWFydGluIEdyb3ZlIEdhcmRlbnMsIFJpY2h2aWV3IEdhcmRlbnMsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTY3ZGQ4ZjJhNjJjNDgyN2IwM2MyM2RjYWQ0ZGIwOGYuc2V0Q29udGVudChodG1sX2U1YjI5OGM1YTI4YjQ0MDQ4ZDg4NzVlZjEzMjA5NjhkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y5NDUxNDg2MDg0NTRkNTU5N2ZhNDU4ZmE1YjUzYmNiLmJpbmRQb3B1cChwb3B1cF9hNjdkZDhmMmE2MmM0ODI3YjAzYzIzZGNhZDRkYjA4Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kZWI1YmNjZDBiNjI0YjcwYmNmNjIyOWNjOGU0ZDdmOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTQxNjM5OTk5OTk5NiwtNzkuNTg4NDM2OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85Y2I2OTkxODIxMzM0NWNiOTgyMTA2N2UwYTY3MmY0YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yMDk3NGUyNzY2MDk0NzA0OTkyZmI1NGU4ZmU2NWRhOSA9ICQoJzxkaXYgaWQ9Imh0bWxfMjA5NzRlMjc2NjA5NDcwNDk5MmZiNTRlOGZlNjVkYTkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNvdXRoIFN0ZWVsZXMsIFNpbHZlcnN0b25lLCBIdW1iZXJnYXRlLCBKYW1lc3Rvd24sIE1vdW50IE9saXZlLCBCZWF1bW9uZCBIZWlnaHRzLCBUaGlzdGxldG93biwgQWxiaW9uIEdhcmRlbnMsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWNiNjk5MTgyMTMzNDVjYjk4MjEwNjdlMGE2NzJmNGIuc2V0Q29udGVudChodG1sXzIwOTc0ZTI3NjYwOTQ3MDQ5OTJmYjU0ZThmZTY1ZGE5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RlYjViY2NkMGI2MjRiNzBiY2Y2MjI5Y2M4ZTRkN2Y5LmJpbmRQb3B1cChwb3B1cF85Y2I2OTkxODIxMzM0NWNiOTgyMTA2N2UwYTY3MmY0Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZjMwYWE2OGY2Njk0NGVjYmRhY2FhMTBkYmUyZTVhNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNjc0ODI5OTk5OTk5NCwtNzkuNTk0MDU0NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9mMDNiZDBmMjQwNTc0Y2UxOGQxMjRmOGQzZjYzNjNhMyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82YWVmNTU2YzcyMmQ0ZTdjOGNlM2U2ODI1ZGRjNTQzOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YTdiMDJjNmQ4OTE0NTA3YmI0Nzk1MDM0NjAxNGYwNyA9ICQoJzxkaXYgaWQ9Imh0bWxfN2E3YjAyYzZkODkxNDUwN2JiNDc5NTAzNDYwMTRmMDciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRod2VzdCwgV2VzdCBIdW1iZXIgLSBDbGFpcnZpbGxlLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZhZWY1NTZjNzIyZDRlN2M4Y2UzZTY4MjVkZGM1NDM5LnNldENvbnRlbnQoaHRtbF83YTdiMDJjNmQ4OTE0NTA3YmI0Nzk1MDM0NjAxNGYwNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lZjMwYWE2OGY2Njk0NGVjYmRhY2FhMTBkYmUyZTVhNS5iaW5kUG9wdXAocG9wdXBfNmFlZjU1NmM3MjJkNGU3YzhjZTNlNjgyNWRkYzU0MzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCjwvc2NyaXB0Pg== onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### 3c. Foursquare Credentials


```python
#Removed the inputs for privacy
CLIENT_ID = ''
CLIENT_SECRET = ''
VERSION = ''
```

### 3d. Explore the first neighborhood in the dataset


```python
neighborhood_name = df_toronto.loc[0, 'Neighborhood']
print(f"The Neighborhood's name is '{neighborhood_name}'.")

```

    The Neighborhood's name is 'Malvern, Rouge'.
    


```python
#get the neighborhood's lat/long values
neighborhood_latitude = df_toronto.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = df_toronto.loc[0, 'Longitude'] # neighborhood longitude value

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))
```

    Latitude and longitude values of Malvern, Rouge are 43.806686299999996, -79.19435340000001.
    

### 3e. The top 100 venues that are in The Beaches within a radius of 500 meters.


```python
LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION,
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)

# get the result to a json file
results = requests.get(url).json()

```


```python
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
```


```python
#Clean the json file and push to a pd df
venues = results['response']['groups'][0]['items']
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues
```

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: FutureWarning: pandas.io.json.json_normalize is deprecated, use pandas.json_normalize instead
      This is separate from the ipykernel package so we can avoid doing imports until
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>categories</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wendy’s</td>
      <td>Fast Food Restaurant</td>
      <td>43.807448</td>
      <td>-79.199056</td>
    </tr>
  </tbody>
</table>
</div>



### 3f. Explore Neighborhoods


```python

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    venues_list=[]
    
    for name, lat, lng in zip(names, latitudes, longitudes):
        # print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

```


```python
#New df w/ venue info
toronto_venues = getNearbyVenues(names=df_toronto['Neighborhood'],
                                   latitudes=df_toronto['Latitude'],
                                   longitudes=df_toronto['Longitude']
                                  )

toronto_venues.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
      <td>Wendy’s</td>
      <td>43.807448</td>
      <td>-79.199056</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>43.784535</td>
      <td>-79.160497</td>
      <td>Royal Canadian Legion</td>
      <td>43.782533</td>
      <td>-79.163085</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>43.784535</td>
      <td>-79.160497</td>
      <td>SEBS Engineering Inc. (Sustainable Energy and ...</td>
      <td>43.782371</td>
      <td>-79.156820</td>
      <td>Construction &amp; Landscaping</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
      <td>RBC Royal Bank</td>
      <td>43.766790</td>
      <td>-79.191151</td>
      <td>Bank</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
      <td>G &amp; G Electronics</td>
      <td>43.765309</td>
      <td>-79.191537</td>
      <td>Electronics Store</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Count the venues in neighborhoods
toronto_venues.groupby('Neighborhood').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Agincourt</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Alderwood, Long Branch</th>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Bathurst Manor, Wilson Heights, Downsview North</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>Bayview Village</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Bedford Park, Lawrence Manor East</th>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>Berczy Park</th>
      <td>55</td>
      <td>55</td>
      <td>55</td>
      <td>55</td>
      <td>55</td>
      <td>55</td>
    </tr>
    <tr>
      <th>Birch Cliff, Cliffside West</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Brockton, Parkdale Village, Exhibition Place</th>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>Business reply mail Processing Centre, South Central Letter Processing Plant Toronto</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>CN Tower, King and Spadina, Railway Lands, Harbourfront West, Bathurst Quay, South Niagara, Island airport</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Caledonia-Fairbanks</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Canada Post Gateway Processing Centre</th>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th>Cedarbrae</th>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Central Bay Street</th>
      <td>68</td>
      <td>68</td>
      <td>68</td>
      <td>68</td>
      <td>68</td>
      <td>68</td>
    </tr>
    <tr>
      <th>Christie</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Church and Wellesley</th>
      <td>75</td>
      <td>75</td>
      <td>75</td>
      <td>75</td>
      <td>75</td>
      <td>75</td>
    </tr>
    <tr>
      <th>Clarks Corners, Tam O'Shanter, Sullivan</th>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Cliffside, Cliffcrest, Scarborough Village West</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Commerce Court, Victoria Hotel</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Davisville</th>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>Davisville North</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Del Ray, Mount Dennis, Keelsdale and Silverthorn</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Don Mills</th>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>Dorset Park, Wexford Heights, Scarborough Town Centre</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Downsview</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Dufferin, Dovercourt Village</th>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th>East Toronto, Broadview North (Old East York)</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Eringate, Bloordale Gardens, Old Burnhamthorpe, Markland Wood</th>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Fairview, Henry Farm, Oriole</th>
      <td>70</td>
      <td>70</td>
      <td>70</td>
      <td>70</td>
      <td>70</td>
      <td>70</td>
    </tr>
    <tr>
      <th>First Canadian Place, Underground city</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Forest Hill North &amp; West, Forest Hill Road Park</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Garden District, Ryerson</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Glencairn</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Golden Mile, Clairlea, Oakridge</th>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>Guildwood, Morningside, West Hill</th>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Harbourfront East, Union Station, Toronto Islands</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>High Park, The Junction South</th>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
    </tr>
    <tr>
      <th>Hillcrest Village</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Humber Summit</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Humberlea, Emery</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Humewood-Cedarvale</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>India Bazaar, The Beaches West</th>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
    </tr>
    <tr>
      <th>Kennedy Park, Ionview, East Birchmount Park</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Kensington Market, Chinatown, Grange Park</th>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
    </tr>
    <tr>
      <th>Kingsview Village, St. Phillips, Martin Grove Gardens, Richview Gardens</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Lawrence Manor, Lawrence Heights</th>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th>Lawrence Park</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Leaside</th>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>Little Portugal, Trinity</th>
      <td>45</td>
      <td>45</td>
      <td>45</td>
      <td>45</td>
      <td>45</td>
      <td>45</td>
    </tr>
    <tr>
      <th>Malvern, Rouge</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Milliken, Agincourt North, Steeles East, L'Amoreaux East</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Mimico NW, The Queensway West, South of Bloor, Kingsway Park South West, Royal York South West</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Moore Park, Summerhill East</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>New Toronto, Mimico South, Humber Bay Shores</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>North Park, Maple Leaf Park, Upwood Park</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>North Toronto West,  Lawrence Park</th>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>Northwest, West Humber - Clairville</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Northwood Park, York University</th>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Old Mill South, King's Mill Park, Sunnylea, Humber Bay, Mimico NE, The Queensway East, Royal York South East, Kingsway Park South East</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Parkdale, Roncesvalles</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Parkview Hill, Woodbine Gardens</th>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>Parkwoods</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Queen's Park, Ontario Provincial Government</th>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>Regent Park, Harbourfront</th>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
    </tr>
    <tr>
      <th>Richmond, Adelaide, King</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Rosedale</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Roselawn</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Rouge Hill, Port Union, Highland Creek</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Runnymede, Swansea</th>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>Runnymede, The Junction North</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Scarborough Village</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>South Steeles, Silverstone, Humbergate, Jamestown, Mount Olive, Beaumond Heights, Thistletown, Albion Gardens</th>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>St. James Town</th>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
    </tr>
    <tr>
      <th>St. James Town, Cabbagetown</th>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Steeles West, L'Amoreaux West</th>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Stn A PO Boxes</th>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
    </tr>
    <tr>
      <th>Studio District</th>
      <td>37</td>
      <td>37</td>
      <td>37</td>
      <td>37</td>
      <td>37</td>
      <td>37</td>
    </tr>
    <tr>
      <th>Summerhill West, Rathnelly, South Hill, Forest Hill SE, Deer Park</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>The Annex, North Midtown, Yorkville</th>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
    </tr>
    <tr>
      <th>The Beaches</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>The Danforth West, Riverdale</th>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
    </tr>
    <tr>
      <th>The Kingsway, Montgomery Road, Old Mill North</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Thorncliffe Park</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Toronto Dominion Centre, Design Exchange</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>University of Toronto, Harbord</th>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>Victoria Village</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>West Deane Park, Princess Gardens, Martin Grove, Islington, Cloverdale</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Westmount</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Weston</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Wexford, Maryvale</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Willowdale, Willowdale East</th>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>Willowdale, Willowdale West</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Woburn</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Woodbine Heights</th>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>York Mills West</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>York Mills, Silver Hills</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### Determine Unique venue categories


```python
print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))
```

    There are 273 uniques categories.
    

### 3f. Neighborhood Analysis


```python
# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yoga Studio</th>
      <th>Accessories Store</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>Aquarium</th>
      <th>Art Gallery</th>
      <th>Art Museum</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Garage</th>
      <th>Auto Workshop</th>
      <th>BBQ Joint</th>
      <th>Baby Store</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Baseball Field</th>
      <th>Baseball Stadium</th>
      <th>Basketball Stadium</th>
      <th>Beach</th>
      <th>Bed &amp; Breakfast</th>
      <th>Beer Bar</th>
      <th>Beer Store</th>
      <th>Belgian Restaurant</th>
      <th>Bike Shop</th>
      <th>Bistro</th>
      <th>Boat or Ferry</th>
      <th>Bookstore</th>
      <th>Boutique</th>
      <th>Brazilian Restaurant</th>
      <th>Breakfast Spot</th>
      <th>Brewery</th>
      <th>Bridal Shop</th>
      <th>Bubble Tea Shop</th>
      <th>Building</th>
      <th>Burger Joint</th>
      <th>Burrito Place</th>
      <th>Bus Line</th>
      <th>Bus Station</th>
      <th>Business Service</th>
      <th>Butcher</th>
      <th>Café</th>
      <th>Cajun / Creole Restaurant</th>
      <th>Camera Store</th>
      <th>Candy Store</th>
      <th>Caribbean Restaurant</th>
      <th>Cheese Shop</th>
      <th>Chinese Restaurant</th>
      <th>Chocolate Shop</th>
      <th>Church</th>
      <th>Climbing Gym</th>
      <th>Clothing Store</th>
      <th>Cocktail Bar</th>
      <th>Coffee Shop</th>
      <th>College Arts Building</th>
      <th>College Auditorium</th>
      <th>College Cafeteria</th>
      <th>College Gym</th>
      <th>College Rec Center</th>
      <th>College Stadium</th>
      <th>Colombian Restaurant</th>
      <th>Comfort Food Restaurant</th>
      <th>Comic Shop</th>
      <th>Concert Hall</th>
      <th>Construction &amp; Landscaping</th>
      <th>Convenience Store</th>
      <th>Cosmetics Shop</th>
      <th>Coworking Space</th>
      <th>Creperie</th>
      <th>Cuban Restaurant</th>
      <th>Cupcake Shop</th>
      <th>Curling Ice</th>
      <th>Dance Studio</th>
      <th>Deli / Bodega</th>
      <th>Department Store</th>
      <th>Dessert Shop</th>
      <th>Dim Sum Restaurant</th>
      <th>Diner</th>
      <th>Discount Store</th>
      <th>Distribution Center</th>
      <th>Dog Run</th>
      <th>Doner Restaurant</th>
      <th>Donut Shop</th>
      <th>Drugstore</th>
      <th>Dumpling Restaurant</th>
      <th>Eastern European Restaurant</th>
      <th>Electronics Store</th>
      <th>Escape Room</th>
      <th>Ethiopian Restaurant</th>
      <th>Event Space</th>
      <th>Falafel Restaurant</th>
      <th>Farmers Market</th>
      <th>Fast Food Restaurant</th>
      <th>Field</th>
      <th>Filipino Restaurant</th>
      <th>Fish &amp; Chips Shop</th>
      <th>Fish Market</th>
      <th>Flea Market</th>
      <th>Food &amp; Drink Shop</th>
      <th>Food Court</th>
      <th>Food Truck</th>
      <th>Fountain</th>
      <th>French Restaurant</th>
      <th>Fried Chicken Joint</th>
      <th>Frozen Yogurt Shop</th>
      <th>Fruit &amp; Vegetable Store</th>
      <th>Furniture / Home Store</th>
      <th>Gaming Cafe</th>
      <th>Garden</th>
      <th>Garden Center</th>
      <th>Gas Station</th>
      <th>Gastropub</th>
      <th>Gay Bar</th>
      <th>General Entertainment</th>
      <th>General Travel</th>
      <th>German Restaurant</th>
      <th>Gift Shop</th>
      <th>Gluten-free Restaurant</th>
      <th>Golf Course</th>
      <th>Gourmet Shop</th>
      <th>Greek Restaurant</th>
      <th>Grocery Store</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Hakka Restaurant</th>
      <th>Harbor / Marina</th>
      <th>Hardware Store</th>
      <th>Health &amp; Beauty Service</th>
      <th>Health Food Store</th>
      <th>Historic Site</th>
      <th>History Museum</th>
      <th>Hobby Shop</th>
      <th>Hockey Arena</th>
      <th>Home Service</th>
      <th>Hookah Bar</th>
      <th>Hospital</th>
      <th>Hostel</th>
      <th>Hotel</th>
      <th>Hotel Bar</th>
      <th>IT Services</th>
      <th>Ice Cream Shop</th>
      <th>Indian Restaurant</th>
      <th>Indie Movie Theater</th>
      <th>Indoor Play Area</th>
      <th>Intersection</th>
      <th>Irish Pub</th>
      <th>Italian Restaurant</th>
      <th>Japanese Restaurant</th>
      <th>Jazz Club</th>
      <th>Jewelry Store</th>
      <th>Juice Bar</th>
      <th>Kids Store</th>
      <th>Kitchen Supply Store</th>
      <th>Korean BBQ Restaurant</th>
      <th>Korean Restaurant</th>
      <th>Lake</th>
      <th>Latin American Restaurant</th>
      <th>Light Rail Station</th>
      <th>Lingerie Store</th>
      <th>Liquor Store</th>
      <th>Locksmith</th>
      <th>Lounge</th>
      <th>Luggage Store</th>
      <th>Malay Restaurant</th>
      <th>Market</th>
      <th>Martial Arts School</th>
      <th>Massage Studio</th>
      <th>Medical Center</th>
      <th>Mediterranean Restaurant</th>
      <th>Men's Store</th>
      <th>Metro Station</th>
      <th>Mexican Restaurant</th>
      <th>Middle Eastern Restaurant</th>
      <th>Miscellaneous Shop</th>
      <th>Mobile Phone Shop</th>
      <th>Modern European Restaurant</th>
      <th>Molecular Gastronomy Restaurant</th>
      <th>Monument / Landmark</th>
      <th>Moroccan Restaurant</th>
      <th>Motel</th>
      <th>Movie Theater</th>
      <th>Museum</th>
      <th>Music Venue</th>
      <th>Neighborhood</th>
      <th>New American Restaurant</th>
      <th>Nightclub</th>
      <th>Noodle House</th>
      <th>Office</th>
      <th>Opera House</th>
      <th>Optical Shop</th>
      <th>Organic Grocery</th>
      <th>Other Great Outdoors</th>
      <th>Park</th>
      <th>Performing Arts Venue</th>
      <th>Pet Store</th>
      <th>Pharmacy</th>
      <th>Pizza Place</th>
      <th>Plane</th>
      <th>Playground</th>
      <th>Plaza</th>
      <th>Poke Place</th>
      <th>Pool</th>
      <th>Portuguese Restaurant</th>
      <th>Poutine Place</th>
      <th>Print Shop</th>
      <th>Pub</th>
      <th>Ramen Restaurant</th>
      <th>Record Shop</th>
      <th>Recording Studio</th>
      <th>Rental Car Location</th>
      <th>Restaurant</th>
      <th>River</th>
      <th>Roof Deck</th>
      <th>Sake Bar</th>
      <th>Salad Place</th>
      <th>Salon / Barbershop</th>
      <th>Sandwich Place</th>
      <th>Scenic Lookout</th>
      <th>Sculpture Garden</th>
      <th>Seafood Restaurant</th>
      <th>Shoe Store</th>
      <th>Shopping Mall</th>
      <th>Shopping Plaza</th>
      <th>Skate Park</th>
      <th>Skating Rink</th>
      <th>Smoke Shop</th>
      <th>Smoothie Shop</th>
      <th>Snack Place</th>
      <th>Soccer Field</th>
      <th>Social Club</th>
      <th>Soup Place</th>
      <th>Southern / Soul Food Restaurant</th>
      <th>Spa</th>
      <th>Speakeasy</th>
      <th>Sporting Goods Shop</th>
      <th>Sports Bar</th>
      <th>Stadium</th>
      <th>Stationery Store</th>
      <th>Steakhouse</th>
      <th>Strip Club</th>
      <th>Supermarket</th>
      <th>Supplement Shop</th>
      <th>Sushi Restaurant</th>
      <th>Swim School</th>
      <th>Tailor Shop</th>
      <th>Taiwanese Restaurant</th>
      <th>Tanning Salon</th>
      <th>Tea Room</th>
      <th>Tennis Court</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Theme Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Turkish Restaurant</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Warehouse Store</th>
      <th>Wine Bar</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Malvern, Rouge</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Group by the neighbrhoods and take the mean of the category frequencies


```python
toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Yoga Studio</th>
      <th>Accessories Store</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>Aquarium</th>
      <th>Art Gallery</th>
      <th>Art Museum</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Garage</th>
      <th>Auto Workshop</th>
      <th>BBQ Joint</th>
      <th>Baby Store</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Baseball Field</th>
      <th>Baseball Stadium</th>
      <th>Basketball Stadium</th>
      <th>Beach</th>
      <th>Bed &amp; Breakfast</th>
      <th>Beer Bar</th>
      <th>Beer Store</th>
      <th>Belgian Restaurant</th>
      <th>Bike Shop</th>
      <th>Bistro</th>
      <th>Boat or Ferry</th>
      <th>Bookstore</th>
      <th>Boutique</th>
      <th>Brazilian Restaurant</th>
      <th>Breakfast Spot</th>
      <th>Brewery</th>
      <th>Bridal Shop</th>
      <th>Bubble Tea Shop</th>
      <th>Building</th>
      <th>Burger Joint</th>
      <th>Burrito Place</th>
      <th>Bus Line</th>
      <th>Bus Station</th>
      <th>Business Service</th>
      <th>Butcher</th>
      <th>Café</th>
      <th>Cajun / Creole Restaurant</th>
      <th>Camera Store</th>
      <th>Candy Store</th>
      <th>Caribbean Restaurant</th>
      <th>Cheese Shop</th>
      <th>Chinese Restaurant</th>
      <th>Chocolate Shop</th>
      <th>Church</th>
      <th>Climbing Gym</th>
      <th>Clothing Store</th>
      <th>Cocktail Bar</th>
      <th>Coffee Shop</th>
      <th>College Arts Building</th>
      <th>College Auditorium</th>
      <th>College Cafeteria</th>
      <th>College Gym</th>
      <th>College Rec Center</th>
      <th>College Stadium</th>
      <th>Colombian Restaurant</th>
      <th>Comfort Food Restaurant</th>
      <th>Comic Shop</th>
      <th>Concert Hall</th>
      <th>Construction &amp; Landscaping</th>
      <th>Convenience Store</th>
      <th>Cosmetics Shop</th>
      <th>Coworking Space</th>
      <th>Creperie</th>
      <th>Cuban Restaurant</th>
      <th>Cupcake Shop</th>
      <th>Curling Ice</th>
      <th>Dance Studio</th>
      <th>Deli / Bodega</th>
      <th>Department Store</th>
      <th>Dessert Shop</th>
      <th>Dim Sum Restaurant</th>
      <th>Diner</th>
      <th>Discount Store</th>
      <th>Distribution Center</th>
      <th>Dog Run</th>
      <th>Doner Restaurant</th>
      <th>Donut Shop</th>
      <th>Drugstore</th>
      <th>Dumpling Restaurant</th>
      <th>Eastern European Restaurant</th>
      <th>Electronics Store</th>
      <th>Escape Room</th>
      <th>Ethiopian Restaurant</th>
      <th>Event Space</th>
      <th>Falafel Restaurant</th>
      <th>Farmers Market</th>
      <th>Fast Food Restaurant</th>
      <th>Field</th>
      <th>Filipino Restaurant</th>
      <th>Fish &amp; Chips Shop</th>
      <th>Fish Market</th>
      <th>Flea Market</th>
      <th>Food &amp; Drink Shop</th>
      <th>Food Court</th>
      <th>Food Truck</th>
      <th>Fountain</th>
      <th>French Restaurant</th>
      <th>Fried Chicken Joint</th>
      <th>Frozen Yogurt Shop</th>
      <th>Fruit &amp; Vegetable Store</th>
      <th>Furniture / Home Store</th>
      <th>Gaming Cafe</th>
      <th>Garden</th>
      <th>Garden Center</th>
      <th>Gas Station</th>
      <th>Gastropub</th>
      <th>Gay Bar</th>
      <th>General Entertainment</th>
      <th>General Travel</th>
      <th>German Restaurant</th>
      <th>Gift Shop</th>
      <th>Gluten-free Restaurant</th>
      <th>Golf Course</th>
      <th>Gourmet Shop</th>
      <th>Greek Restaurant</th>
      <th>Grocery Store</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Hakka Restaurant</th>
      <th>Harbor / Marina</th>
      <th>Hardware Store</th>
      <th>Health &amp; Beauty Service</th>
      <th>Health Food Store</th>
      <th>Historic Site</th>
      <th>History Museum</th>
      <th>Hobby Shop</th>
      <th>Hockey Arena</th>
      <th>Home Service</th>
      <th>Hookah Bar</th>
      <th>Hospital</th>
      <th>Hostel</th>
      <th>Hotel</th>
      <th>Hotel Bar</th>
      <th>IT Services</th>
      <th>Ice Cream Shop</th>
      <th>Indian Restaurant</th>
      <th>Indie Movie Theater</th>
      <th>Indoor Play Area</th>
      <th>Intersection</th>
      <th>Irish Pub</th>
      <th>Italian Restaurant</th>
      <th>Japanese Restaurant</th>
      <th>Jazz Club</th>
      <th>Jewelry Store</th>
      <th>Juice Bar</th>
      <th>Kids Store</th>
      <th>Kitchen Supply Store</th>
      <th>Korean BBQ Restaurant</th>
      <th>Korean Restaurant</th>
      <th>Lake</th>
      <th>Latin American Restaurant</th>
      <th>Light Rail Station</th>
      <th>Lingerie Store</th>
      <th>Liquor Store</th>
      <th>Locksmith</th>
      <th>Lounge</th>
      <th>Luggage Store</th>
      <th>Malay Restaurant</th>
      <th>Market</th>
      <th>Martial Arts School</th>
      <th>Massage Studio</th>
      <th>Medical Center</th>
      <th>Mediterranean Restaurant</th>
      <th>Men's Store</th>
      <th>Metro Station</th>
      <th>Mexican Restaurant</th>
      <th>Middle Eastern Restaurant</th>
      <th>Miscellaneous Shop</th>
      <th>Mobile Phone Shop</th>
      <th>Modern European Restaurant</th>
      <th>Molecular Gastronomy Restaurant</th>
      <th>Monument / Landmark</th>
      <th>Moroccan Restaurant</th>
      <th>Motel</th>
      <th>Movie Theater</th>
      <th>Museum</th>
      <th>Music Venue</th>
      <th>New American Restaurant</th>
      <th>Nightclub</th>
      <th>Noodle House</th>
      <th>Office</th>
      <th>Opera House</th>
      <th>Optical Shop</th>
      <th>Organic Grocery</th>
      <th>Other Great Outdoors</th>
      <th>Park</th>
      <th>Performing Arts Venue</th>
      <th>Pet Store</th>
      <th>Pharmacy</th>
      <th>Pizza Place</th>
      <th>Plane</th>
      <th>Playground</th>
      <th>Plaza</th>
      <th>Poke Place</th>
      <th>Pool</th>
      <th>Portuguese Restaurant</th>
      <th>Poutine Place</th>
      <th>Print Shop</th>
      <th>Pub</th>
      <th>Ramen Restaurant</th>
      <th>Record Shop</th>
      <th>Recording Studio</th>
      <th>Rental Car Location</th>
      <th>Restaurant</th>
      <th>River</th>
      <th>Roof Deck</th>
      <th>Sake Bar</th>
      <th>Salad Place</th>
      <th>Salon / Barbershop</th>
      <th>Sandwich Place</th>
      <th>Scenic Lookout</th>
      <th>Sculpture Garden</th>
      <th>Seafood Restaurant</th>
      <th>Shoe Store</th>
      <th>Shopping Mall</th>
      <th>Shopping Plaza</th>
      <th>Skate Park</th>
      <th>Skating Rink</th>
      <th>Smoke Shop</th>
      <th>Smoothie Shop</th>
      <th>Snack Place</th>
      <th>Soccer Field</th>
      <th>Social Club</th>
      <th>Soup Place</th>
      <th>Southern / Soul Food Restaurant</th>
      <th>Spa</th>
      <th>Speakeasy</th>
      <th>Sporting Goods Shop</th>
      <th>Sports Bar</th>
      <th>Stadium</th>
      <th>Stationery Store</th>
      <th>Steakhouse</th>
      <th>Strip Club</th>
      <th>Supermarket</th>
      <th>Supplement Shop</th>
      <th>Sushi Restaurant</th>
      <th>Swim School</th>
      <th>Tailor Shop</th>
      <th>Taiwanese Restaurant</th>
      <th>Tanning Salon</th>
      <th>Tea Room</th>
      <th>Tennis Court</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Theme Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Turkish Restaurant</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Warehouse Store</th>
      <th>Wine Bar</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Agincourt</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alderwood, Long Branch</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.142857</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.142857</td>
      <td>0.285714</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.142857</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.142857</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bathurst Manor, Wilson Heights, Downsview North</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.095238</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.095238</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bayview Village</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bedford Park, Lawrence Manor East</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.090909</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.090909</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.090909</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#10 most common venues
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Agincourt</td>
      <td>Lounge</td>
      <td>Skating Rink</td>
      <td>Latin American Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Clothing Store</td>
      <td>Drugstore</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alderwood, Long Branch</td>
      <td>Pizza Place</td>
      <td>Gym</td>
      <td>Pharmacy</td>
      <td>Coffee Shop</td>
      <td>Sandwich Place</td>
      <td>Pub</td>
      <td>Women's Store</td>
      <td>Dog Run</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bathurst Manor, Wilson Heights, Downsview North</td>
      <td>Coffee Shop</td>
      <td>Bank</td>
      <td>Mobile Phone Shop</td>
      <td>Bridal Shop</td>
      <td>Sandwich Place</td>
      <td>Diner</td>
      <td>Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Supermarket</td>
      <td>Middle Eastern Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bayview Village</td>
      <td>Café</td>
      <td>Japanese Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Bank</td>
      <td>Women's Store</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bedford Park, Lawrence Manor East</td>
      <td>Coffee Shop</td>
      <td>Sandwich Place</td>
      <td>Italian Restaurant</td>
      <td>Greek Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Pharmacy</td>
      <td>Pizza Place</td>
      <td>Pub</td>
      <td>Café</td>
      <td>Butcher</td>
    </tr>
  </tbody>
</table>
</div>



### 3g. KMeans Clustering


```python
# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
```




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])




```python
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = df_toronto

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
      <td>4.0</td>
      <td>Fast Food Restaurant</td>
      <td>Drugstore</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Dumpling Restaurant</td>
      <td>Wings Joint</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>43.784535</td>
      <td>-79.160497</td>
      <td>1.0</td>
      <td>Construction &amp; Landscaping</td>
      <td>Bar</td>
      <td>Women's Store</td>
      <td>Dumpling Restaurant</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Drugstore</td>
      <td>Eastern European Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
      <td>1.0</td>
      <td>Restaurant</td>
      <td>Medical Center</td>
      <td>Intersection</td>
      <td>Mexican Restaurant</td>
      <td>Bank</td>
      <td>Rental Car Location</td>
      <td>Breakfast Spot</td>
      <td>Electronics Store</td>
      <td>Drugstore</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Mexican Restaurant</td>
      <td>Korean BBQ Restaurant</td>
      <td>Women's Store</td>
      <td>Drugstore</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
      <td>1.0</td>
      <td>Hakka Restaurant</td>
      <td>Athletics &amp; Sports</td>
      <td>Bakery</td>
      <td>Gas Station</td>
      <td>Caribbean Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Bank</td>
      <td>Fried Chicken Joint</td>
      <td>Dog Run</td>
      <td>Distribution Center</td>
    </tr>
  </tbody>
</table>
</div>



#### Data visulation of the Clusters


```python
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], 
                                  toronto_merged['Longitude'],
                                  toronto_merged['Neighborhood'],
                                  toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow,
        fill=True,
        fill_color=rainbow,
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZScsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzNDgxNywtNzkuMzgzOTM0N10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfMzExMzUyNGRlM2Q2NGViZGI5ZDk2M2JiNGExYjU0NjAgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNmNDcyNzhiNzE5ZDQ1NzM4N2Q2MTA5YTBmZmEyYTdiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODA2Njg2Mjk5OTk5OTk2LC03OS4xOTQzNTM0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2VmNDZkMjA2YjI4YjQxODBiZmM3MmU5YTdkMDZhNDM3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzAwYzBlMjNlMTEwNjRjZTlhYjI4NzkwYjQxY2ExYTZlID0gJCgnPGRpdiBpZD0iaHRtbF8wMGMwZTIzZTExMDY0Y2U5YWIyODc5MGI0MWNhMWE2ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWFsdmVybiwgUm91Z2UgQ2x1c3RlciA0LjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VmNDZkMjA2YjI4YjQxODBiZmM3MmU5YTdkMDZhNDM3LnNldENvbnRlbnQoaHRtbF8wMGMwZTIzZTExMDY0Y2U5YWIyODc5MGI0MWNhMWE2ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zZjQ3Mjc4YjcxOWQ0NTczODdkNjEwOWEwZmZhMmE3Yi5iaW5kUG9wdXAocG9wdXBfZWY0NmQyMDZiMjhiNDE4MGJmYzcyZTlhN2QwNmE0MzcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzBjM2M3ZDZkMDI5NDZjNjg2ZmQ1MmUyZjVhZWJjOTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODQ1MzUxLC03OS4xNjA0OTcwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzEzZGQwMTlmNGY3ZTQ3ZWQ5ZGJhNTJjYmMyYjYyMDQyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E0NDM3YWMxY2JmMzRmOTU4ODA1ZDE5NzRkNWIyYTYxID0gJCgnPGRpdiBpZD0iaHRtbF9hNDQzN2FjMWNiZjM0Zjk1ODgwNWQxOTc0ZDViMmE2MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um91Z2UgSGlsbCwgUG9ydCBVbmlvbiwgSGlnaGxhbmQgQ3JlZWsgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzEzZGQwMTlmNGY3ZTQ3ZWQ5ZGJhNTJjYmMyYjYyMDQyLnNldENvbnRlbnQoaHRtbF9hNDQzN2FjMWNiZjM0Zjk1ODgwNWQxOTc0ZDViMmE2MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83MGMzYzdkNmQwMjk0NmM2ODZmZDUyZTJmNWFlYmM5MC5iaW5kUG9wdXAocG9wdXBfMTNkZDAxOWY0ZjdlNDdlZDlkYmE1MmNiYzJiNjIwNDIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDg4NmNmY2NiOTkxNGUwOTk1Yzk4ZDAxM2E5ZjRhZjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NjM1NzI2LC03OS4xODg3MTE1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDY4ODNlYWM1NzY5NDI1MDlmYTNkNmQzMTM2MjgxMDQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWY5NmNhMTdhNTBkNDg5ZWJmNTJjNmRiNTJlOGExNGUgPSAkKCc8ZGl2IGlkPSJodG1sX2VmOTZjYTE3YTUwZDQ4OWViZjUyYzZkYjUyZThhMTRlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HdWlsZHdvb2QsIE1vcm5pbmdzaWRlLCBXZXN0IEhpbGwgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ2ODgzZWFjNTc2OTQyNTA5ZmEzZDZkMzEzNjI4MTA0LnNldENvbnRlbnQoaHRtbF9lZjk2Y2ExN2E1MGQ0ODllYmY1MmM2ZGI1MmU4YTE0ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kODg2Y2ZjY2I5OTE0ZTA5OTVjOThkMDEzYTlmNGFmMS5iaW5kUG9wdXAocG9wdXBfNDY4ODNlYWM1NzY5NDI1MDlmYTNkNmQzMTM2MjgxMDQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTFkZGUzNjhhMGQ5NDFjNTk2YWJlMjY1NjdhNDEzNDggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzA5OTIxLC03OS4yMTY5MTc0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzIxNWIzNTdkZDBiOTQzZDNhMGQzNTkwY2U0YWM2ZThiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YzZGM1OGFmZWI1MTQ4NjVhODA2Nzc4YzJhMTFkNDJiID0gJCgnPGRpdiBpZD0iaHRtbF9mM2RjNThhZmViNTE0ODY1YTgwNjc3OGMyYTExZDQyYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V29idXJuIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yMTViMzU3ZGQwYjk0M2QzYTBkMzU5MGNlNGFjNmU4Yi5zZXRDb250ZW50KGh0bWxfZjNkYzU4YWZlYjUxNDg2NWE4MDY3NzhjMmExMWQ0MmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTFkZGUzNjhhMGQ5NDFjNTk2YWJlMjY1NjdhNDEzNDguYmluZFBvcHVwKHBvcHVwXzIxNWIzNTdkZDBiOTQzZDNhMGQzNTkwY2U0YWM2ZThiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ1MWE1YzMxMzZhNDQzYzA4ZmQ3MzA0MDI1OTVkNGNiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzczMTM2LC03OS4yMzk0NzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JkYWQwM2I1ZjBmODRkZTk4NDhlZGQ2ZjAwN2Y2MDFlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VkMjM2ZGQ3MDA3OTQ2MTlhZTMzMzBiZWY3Nzk0ZDVmID0gJCgnPGRpdiBpZD0iaHRtbF9lZDIzNmRkNzAwNzk0NjE5YWUzMzMwYmVmNzc5NGQ1ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VkYXJicmFlIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZGFkMDNiNWYwZjg0ZGU5ODQ4ZWRkNmYwMDdmNjAxZS5zZXRDb250ZW50KGh0bWxfZWQyMzZkZDcwMDc5NDYxOWFlMzMzMGJlZjc3OTRkNWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDUxYTVjMzEzNmE0NDNjMDhmZDczMDQwMjU5NWQ0Y2IuYmluZFBvcHVwKHBvcHVwX2JkYWQwM2I1ZjBmODRkZTk4NDhlZGQ2ZjAwN2Y2MDFlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzEzYzJhY2ZlZTIyNTRmMDM4MWU4N2FlNmQ2YTAxYTY2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzQ0NzM0MiwtNzkuMjM5NDc2MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xN2M3MmYwMWExZTQ0YmZjODRlOTdhNzMyNDJjYzhlMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jNTFlMmQ0MzJjZDY0Njc1OTQ4Yzc1MzU3MDdkMjRiOSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzUxZTJkNDMyY2Q2NDY3NTk0OGM3NTM1NzA3ZDI0YjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNjYXJib3JvdWdoIFZpbGxhZ2UgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE3YzcyZjAxYTFlNDRiZmM4NGU5N2E3MzI0MmNjOGUyLnNldENvbnRlbnQoaHRtbF9jNTFlMmQ0MzJjZDY0Njc1OTQ4Yzc1MzU3MDdkMjRiOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xM2MyYWNmZWUyMjU0ZjAzODFlODdhZTZkNmEwMWE2Ni5iaW5kUG9wdXAocG9wdXBfMTdjNzJmMDFhMWU0NGJmYzg0ZTk3YTczMjQyY2M4ZTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGNjYWY3MTU5ZGM4NGJkYTg4MjlkYzZhNjIxOGY4N2YgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Mjc5MjkyLC03OS4yNjIwMjk0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNiZmI4ZGU5ZGE5MjRkNDQ4M2YwMzRkNDBhNjlkOTJlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E1MDBiMGE1NDRlMjQyY2E5Mzc5MWEyNTE5YjdiZWExID0gJCgnPGRpdiBpZD0iaHRtbF9hNTAwYjBhNTQ0ZTI0MmNhOTM3OTFhMjUxOWI3YmVhMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2VubmVkeSBQYXJrLCBJb252aWV3LCBFYXN0IEJpcmNobW91bnQgUGFyayBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2JmYjhkZTlkYTkyNGQ0NDgzZjAzNGQ0MGE2OWQ5MmUuc2V0Q29udGVudChodG1sX2E1MDBiMGE1NDRlMjQyY2E5Mzc5MWEyNTE5YjdiZWExKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhjY2FmNzE1OWRjODRiZGE4ODI5ZGM2YTYyMThmODdmLmJpbmRQb3B1cChwb3B1cF8zYmZiOGRlOWRhOTI0ZDQ0ODNmMDM0ZDQwYTY5ZDkyZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80OTRiNmZlMDllNDI0YmRlYjIzNDUxYTUzY2U3NmUwNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMTExMTcwMDAwMDAwNCwtNzkuMjg0NTc3Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcyNzgwM2VjZjgzMjQ0OGRiNTI3MmEzYWQ4MWVhZjFmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI3MzQwMWNhNjg2YjQzNWViZTQ2M2NlYzg2OTE0NDQzID0gJCgnPGRpdiBpZD0iaHRtbF8yNzM0MDFjYTY4NmI0MzVlYmU0NjNjZWM4NjkxNDQ0MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R29sZGVuIE1pbGUsIENsYWlybGVhLCBPYWtyaWRnZSBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzI3ODAzZWNmODMyNDQ4ZGI1MjcyYTNhZDgxZWFmMWYuc2V0Q29udGVudChodG1sXzI3MzQwMWNhNjg2YjQzNWViZTQ2M2NlYzg2OTE0NDQzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ5NGI2ZmUwOWU0MjRiZGViMjM0NTFhNTNjZTc2ZTA0LmJpbmRQb3B1cChwb3B1cF83Mjc4MDNlY2Y4MzI0NDhkYjUyNzJhM2FkODFlYWYxZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jYTRhMzU3MzllOTg0Njk2YjFiMjZiYTAyOGZmMzUyNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxNjMxNiwtNzkuMjM5NDc2MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNjUyOTMyNmI4YTI0MzIxODQxY2U2NzcxOGUzOGQ5NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wYTA2NWMwMTg1ZjM0MDlkODgwZWY5YTRhYTJlZDQzNiA9ICQoJzxkaXYgaWQ9Imh0bWxfMGEwNjVjMDE4NWYzNDA5ZDg4MGVmOWE0YWEyZWQ0MzYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNsaWZmc2lkZSwgQ2xpZmZjcmVzdCwgU2NhcmJvcm91Z2ggVmlsbGFnZSBXZXN0IENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kNjUyOTMyNmI4YTI0MzIxODQxY2U2NzcxOGUzOGQ5Ny5zZXRDb250ZW50KGh0bWxfMGEwNjVjMDE4NWYzNDA5ZDg4MGVmOWE0YWEyZWQ0MzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2E0YTM1NzM5ZTk4NDY5NmIxYjI2YmEwMjhmZjM1MjYuYmluZFBvcHVwKHBvcHVwX2Q2NTI5MzI2YjhhMjQzMjE4NDFjZTY3NzE4ZTM4ZDk3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQwZGViYjhiYmRlNjRmOWU5MjI2ODM0OWQ4ZjgyNmI0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkyNjU3MDAwMDAwMDA0LC03OS4yNjQ4NDgxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzE5Yzk5ZDViNWI1NGU5OGEzMmRmMzE0YWNjMDNlOTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWZmYTAyZTM2MTFlNDdjZWExNzlmNDhlYmQyYWE4ZDggPSAkKCc8ZGl2IGlkPSJodG1sXzVmZmEwMmUzNjExZTQ3Y2VhMTc5ZjQ4ZWJkMmFhOGQ4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CaXJjaCBDbGlmZiwgQ2xpZmZzaWRlIFdlc3QgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MxOWM5OWQ1YjViNTRlOThhMzJkZjMxNGFjYzAzZTk2LnNldENvbnRlbnQoaHRtbF81ZmZhMDJlMzYxMWU0N2NlYTE3OWY0OGViZDJhYThkOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MGRlYmI4YmJkZTY0ZjllOTIyNjgzNDlkOGY4MjZiNC5iaW5kUG9wdXAocG9wdXBfYzE5Yzk5ZDViNWI1NGU5OGEzMmRmMzE0YWNjMDNlOTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2EwYTgzM2ZlMDNmNGMyZmFlMjQ3ZDAwNjM0NzI3YWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTc0MDk2LC03OS4yNzMzMDQwMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI1MjI3NGM4ZjM1MzQ3ODdhYmNlNzM2NGQ4Mzc0YmJiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlhMTNmMzlmNzgwZjRkMTg5N2I5MGVjZTUyYzNmNjEwID0gJCgnPGRpdiBpZD0iaHRtbF85YTEzZjM5Zjc4MGY0ZDE4OTdiOTBlY2U1MmMzZjYxMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9yc2V0IFBhcmssIFdleGZvcmQgSGVpZ2h0cywgU2NhcmJvcm91Z2ggVG93biBDZW50cmUgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI1MjI3NGM4ZjM1MzQ3ODdhYmNlNzM2NGQ4Mzc0YmJiLnNldENvbnRlbnQoaHRtbF85YTEzZjM5Zjc4MGY0ZDE4OTdiOTBlY2U1MmMzZjYxMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83YTBhODMzZmUwM2Y0YzJmYWUyNDdkMDA2MzQ3MjdhZi5iaW5kUG9wdXAocG9wdXBfMjUyMjc0YzhmMzUzNDc4N2FiY2U3MzY0ZDgzNzRiYmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzg2ZDQ1MzMwZDg5NDE5M2E5Y2ZkOGRkZDZiZGIyNmIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTAwNzE1MDAwMDAwMDQsLTc5LjI5NTg0OTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZDA5ZTM0OTU2NjQ0YmI2YjAxZTdjMzliNzgwMzJhOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wYzg1MGVhZmEyM2Q0OWIzOTMxNGE3NDhhZjViOTBkNiA9ICQoJzxkaXYgaWQ9Imh0bWxfMGM4NTBlYWZhMjNkNDliMzkzMTRhNzQ4YWY1YjkwZDYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldleGZvcmQsIE1hcnl2YWxlIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZDA5ZTM0OTU2NjQ0YmI2YjAxZTdjMzliNzgwMzJhOS5zZXRDb250ZW50KGh0bWxfMGM4NTBlYWZhMjNkNDliMzkzMTRhNzQ4YWY1YjkwZDYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzg2ZDQ1MzMwZDg5NDE5M2E5Y2ZkOGRkZDZiZGIyNmIuYmluZFBvcHVwKHBvcHVwX2JkMDllMzQ5NTY2NDRiYjZiMDFlN2MzOWI3ODAzMmE5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2RmYzNlMWU5NWFhYTRlYzM4NGMxNzMyY2JmMWZhNGRjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzk0MjAwMywtNzkuMjYyMDI5NDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zOTkyMzk3NjVkMDQ0YWQ4Yjc0YjBkNGMwNzkzZGI2YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ZDFlYzM0MzI0Mjg0NGM3YmQzNGNhOWQ2OTY1ZWE5MiA9ICQoJzxkaXYgaWQ9Imh0bWxfNmQxZWMzNDMyNDI4NDRjN2JkMzRjYTlkNjk2NWVhOTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFnaW5jb3VydCBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzk5MjM5NzY1ZDA0NGFkOGI3NGIwZDRjMDc5M2RiNmMuc2V0Q29udGVudChodG1sXzZkMWVjMzQzMjQyODQ0YzdiZDM0Y2E5ZDY5NjVlYTkyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RmYzNlMWU5NWFhYTRlYzM4NGMxNzMyY2JmMWZhNGRjLmJpbmRQb3B1cChwb3B1cF8zOTkyMzk3NjVkMDQ0YWQ4Yjc0YjBkNGMwNzkzZGI2Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MmY2ZDYzMmQxYmI0MzdjOGVmNDk1N2NjZTlhNTMzYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4MTYzNzUsLTc5LjMwNDMwMjFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMDU1YmU0MDY4NjY0MzliODY0ZjRjNjgwNTZiNzdiMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85Yzc2MTFiMDkwZDc0MzA5YTM0MTJlODA4MDcwNmJmYiA9ICQoJzxkaXYgaWQ9Imh0bWxfOWM3NjExYjA5MGQ3NDMwOWEzNDEyZTgwODA3MDZiZmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNsYXJrcyBDb3JuZXJzLCBUYW0gTyYjMzk7U2hhbnRlciwgU3VsbGl2YW4gQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MwNTViZTQwNjg2NjQzOWI4NjRmNGM2ODA1NmI3N2IxLnNldENvbnRlbnQoaHRtbF85Yzc2MTFiMDkwZDc0MzA5YTM0MTJlODA4MDcwNmJmYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MmY2ZDYzMmQxYmI0MzdjOGVmNDk1N2NjZTlhNTMzYS5iaW5kUG9wdXAocG9wdXBfYzA1NWJlNDA2ODY2NDM5Yjg2NGY0YzY4MDU2Yjc3YjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWU0OWQzZGFiMDlkNDRmYzhlY2MxYTA1YWJmNjlhMmYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My44MTUyNTIyLC03OS4yODQ1NzcyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTZhNmExNmIxOTBhNDFiMWIzMjU3OWYzNzdmMWQyY2YgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzI5MDIyNjNlZjk3NDlkY2E0NWEzZTlhNTNiNTlhY2IgPSAkKCc8ZGl2IGlkPSJodG1sXzcyOTAyMjYzZWY5NzQ5ZGNhNDVhM2U5YTUzYjU5YWNiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NaWxsaWtlbiwgQWdpbmNvdXJ0IE5vcnRoLCBTdGVlbGVzIEVhc3QsIEwmIzM5O0Ftb3JlYXV4IEVhc3QgQ2x1c3RlciAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE2YTZhMTZiMTkwYTQxYjFiMzI1NzlmMzc3ZjFkMmNmLnNldENvbnRlbnQoaHRtbF83MjkwMjI2M2VmOTc0OWRjYTQ1YTNlOWE1M2I1OWFjYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xZTQ5ZDNkYWIwOWQ0NGZjOGVjYzFhMDVhYmY2OWEyZi5iaW5kUG9wdXAocG9wdXBfMTZhNmExNmIxOTBhNDFiMWIzMjU3OWYzNzdmMWQyY2YpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWZjY2U1ZGE3YWVlNDY4NjliNTY4ZjNjN2MyNjM4NTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43OTk1MjUyMDAwMDAwMDUsLTc5LjMxODM4ODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84ZDQ2MjQzNGMyNGU0ODhiYjE1OTliMTQ4ZjRiYWI4MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mODM4NzFhODY2YzY0NmIyODYwNTU2MTg2Y2FkYzc3MiA9ICQoJzxkaXYgaWQ9Imh0bWxfZjgzODcxYTg2NmM2NDZiMjg2MDU1NjE4NmNhZGM3NzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0ZWVsZXMgV2VzdCwgTCYjMzk7QW1vcmVhdXggV2VzdCBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGQ0NjI0MzRjMjRlNDg4YmIxNTk5YjE0OGY0YmFiODMuc2V0Q29udGVudChodG1sX2Y4Mzg3MWE4NjZjNjQ2YjI4NjA1NTYxODZjYWRjNzcyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFmY2NlNWRhN2FlZTQ2ODY5YjU2OGYzYzdjMjYzODUyLmJpbmRQb3B1cChwb3B1cF84ZDQ2MjQzNGMyNGU0ODhiYjE1OTliMTQ4ZjRiYWI4Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80OTdkYTQwNzJjYjg0YTYxYmM0ODE2Y2Y3NjdiNTA0YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjgzNjEyNDcwMDAwMDAwNiwtNzkuMjA1NjM2MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lOTcyMmI4MmQ2Nzk0ZDgzYTk0MjY3NmNiOGI5NDYzOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jMjcyMTNjZmM4OGI0MTc4OGU4NGRjMWRjNTQ5MzMwYSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzI3MjEzY2ZjODhiNDE3ODhlODRkYzFkYzU0OTMzMGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVwcGVyIFJvdWdlIENsdXN0ZXIgbmFuPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lOTcyMmI4MmQ2Nzk0ZDgzYTk0MjY3NmNiOGI5NDYzOC5zZXRDb250ZW50KGh0bWxfYzI3MjEzY2ZjODhiNDE3ODhlODRkYzFkYzU0OTMzMGEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDk3ZGE0MDcyY2I4NGE2MWJjNDgxNmNmNzY3YjUwNGMuYmluZFBvcHVwKHBvcHVwX2U5NzIyYjgyZDY3OTRkODNhOTQyNjc2Y2I4Yjk0NjM4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FjMjg4MGQ5NGM0ODRjZTNhNGYwYjhhYzU1MjgzYjFjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODAzNzYyMiwtNzkuMzYzNDUxN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRjYzlhMTQwMDkzMzQwZGU5MzBlNjg1OWE4YWFmNzE1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJkMGNkYTlkNGFlMjRjYzViMmFiMTAwMmVkNjJiZTU1ID0gJCgnPGRpdiBpZD0iaHRtbF8yZDBjZGE5ZDRhZTI0Y2M1YjJhYjEwMDJlZDYyYmU1NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGlsbGNyZXN0IFZpbGxhZ2UgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRjYzlhMTQwMDkzMzQwZGU5MzBlNjg1OWE4YWFmNzE1LnNldENvbnRlbnQoaHRtbF8yZDBjZGE5ZDRhZTI0Y2M1YjJhYjEwMDJlZDYyYmU1NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hYzI4ODBkOTRjNDg0Y2UzYTRmMGI4YWM1NTI4M2IxYy5iaW5kUG9wdXAocG9wdXBfNGNjOWExNDAwOTMzNDBkZTkzMGU2ODU5YThhYWY3MTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjBjYmEyNjU5YTkzNGQ3NTk0NGYxMDBjM2Q5YWNkOTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Nzg1MTc1LC03OS4zNDY1NTU3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYmI4OWY2Y2EwOWM3NDhlZThiY2MyYmJkNTMxNDZlZjIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDk1MjljNGJjZWJjNDJjYWI5YWQ2MGIwMTQ0YjI1ODQgPSAkKCc8ZGl2IGlkPSJodG1sXzQ5NTI5YzRiY2ViYzQyY2FiOWFkNjBiMDE0NGIyNTg0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GYWlydmlldywgSGVucnkgRmFybSwgT3Jpb2xlIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iYjg5ZjZjYTA5Yzc0OGVlOGJjYzJiYmQ1MzE0NmVmMi5zZXRDb250ZW50KGh0bWxfNDk1MjljNGJjZWJjNDJjYWI5YWQ2MGIwMTQ0YjI1ODQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjBjYmEyNjU5YTkzNGQ3NTk0NGYxMDBjM2Q5YWNkOTAuYmluZFBvcHVwKHBvcHVwX2JiODlmNmNhMDljNzQ4ZWU4YmNjMmJiZDUzMTQ2ZWYyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzEwODVmMmRkOGI4OTRiMjg5NTUwOTA4ODNhYzg5YjM4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzg2OTQ3MywtNzkuMzg1OTc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDlmNDk2NzMxNTc0NDY0OTk1MmYyZjE0MzgyM2NhZmQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjU3YWE2YzI1MWIxNGVlYjk2NzgzMjEyOTY3ZGVkMjYgPSAkKCc8ZGl2IGlkPSJodG1sXzI1N2FhNmMyNTFiMTRlZWI5Njc4MzIxMjk2N2RlZDI2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CYXl2aWV3IFZpbGxhZ2UgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzA5ZjQ5NjczMTU3NDQ2NDk5NTJmMmYxNDM4MjNjYWZkLnNldENvbnRlbnQoaHRtbF8yNTdhYTZjMjUxYjE0ZWViOTY3ODMyMTI5NjdkZWQyNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMDg1ZjJkZDhiODk0YjI4OTU1MDkwODgzYWM4OWIzOC5iaW5kUG9wdXAocG9wdXBfMDlmNDk2NzMxNTc0NDY0OTk1MmYyZjE0MzgyM2NhZmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTVjMGNhMGNmNzU0NDZlZmEwMmYxNTZkNDVhN2ZhNGMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTc0OTAyLC03OS4zNzQ3MTQwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U0ZDU0YmUzMWJmZDQxMTZhNjk3YjIzNmNhMGJlYTkyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RiYTZkM2I4MzcxZDQ1OGQ5NzY1MzUyYjlmYWQwZDhmID0gJCgnPGRpdiBpZD0iaHRtbF9kYmE2ZDNiODM3MWQ0NThkOTc2NTM1MmI5ZmFkMGQ4ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+WW9yayBNaWxscywgU2lsdmVyIEhpbGxzIENsdXN0ZXIgMi4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lNGQ1NGJlMzFiZmQ0MTE2YTY5N2IyMzZjYTBiZWE5Mi5zZXRDb250ZW50KGh0bWxfZGJhNmQzYjgzNzFkNDU4ZDk3NjUzNTJiOWZhZDBkOGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTVjMGNhMGNmNzU0NDZlZmEwMmYxNTZkNDVhN2ZhNGMuYmluZFBvcHVwKHBvcHVwX2U0ZDU0YmUzMWJmZDQxMTZhNjk3YjIzNmNhMGJlYTkyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI4Yjc5MTRjOGRjOTQxOTA4MDU5ZDk1MWYwZDI3NmI3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzg5MDUzLC03OS40MDg0OTI3OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y1YTAyMmY1ZGMxNTQ3YWQ4ZWUzMGM1ZmIzYTJiZDY3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI3ZDgxMzYwYTE4MTRiMmFhOGEyOTRkYzFjZTA0ZWY1ID0gJCgnPGRpdiBpZD0iaHRtbF8yN2Q4MTM2MGExODE0YjJhYThhMjk0ZGMxY2UwNGVmNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2lsbG93ZGFsZSwgTmV3dG9uYnJvb2sgQ2x1c3RlciBuYW48L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y1YTAyMmY1ZGMxNTQ3YWQ4ZWUzMGM1ZmIzYTJiZDY3LnNldENvbnRlbnQoaHRtbF8yN2Q4MTM2MGExODE0YjJhYThhMjk0ZGMxY2UwNGVmNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yOGI3OTE0YzhkYzk0MTkwODA1OWQ5NTFmMGQyNzZiNy5iaW5kUG9wdXAocG9wdXBfZjVhMDIyZjVkYzE1NDdhZDhlZTMwYzVmYjNhMmJkNjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmQ2Y2MzYjlhMDljNDcxNzhkNDMwMzE2NjQ0M2E2ZDYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzAxMTk5LC03OS40MDg0OTI3OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q2Y2Q0ZGFmNWZkNzRlZjM5MDA2NTBlZTY3Njk5NTY3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY3ZTAyZTljODE4YTQ0ODFiZWE4ZDgyNDY3ZmRlZjFjID0gJCgnPGRpdiBpZD0iaHRtbF82N2UwMmU5YzgxOGE0NDgxYmVhOGQ4MjQ2N2ZkZWYxYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2lsbG93ZGFsZSwgV2lsbG93ZGFsZSBFYXN0IENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kNmNkNGRhZjVmZDc0ZWYzOTAwNjUwZWU2NzY5OTU2Ny5zZXRDb250ZW50KGh0bWxfNjdlMDJlOWM4MThhNDQ4MWJlYThkODI0NjdmZGVmMWMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmQ2Y2MzYjlhMDljNDcxNzhkNDMwMzE2NjQ0M2E2ZDYuYmluZFBvcHVwKHBvcHVwX2Q2Y2Q0ZGFmNWZkNzRlZjM5MDA2NTBlZTY3Njk5NTY3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VjNTZiMjI1NTNmYjQ1MDViYjYyOTEwNDdhZDM1MjYxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzUyNzU4Mjk5OTk5OTk2LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTY5OTRmM2FjZGRmNDdkMmE1MWU0ZGE3MDFjNDY2M2QgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTAzYzU5ZTQ3MGQzNDY2NjlkY2ZjZTIwODJkZTQyZmIgPSAkKCc8ZGl2IGlkPSJodG1sX2EwM2M1OWU0NzBkMzQ2NjY5ZGNmY2UyMDgyZGU0MmZiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Zb3JrIE1pbGxzIFdlc3QgQ2x1c3RlciAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU2OTk0ZjNhY2RkZjQ3ZDJhNTFlNGRhNzAxYzQ2NjNkLnNldENvbnRlbnQoaHRtbF9hMDNjNTllNDcwZDM0NjY2OWRjZmNlMjA4MmRlNDJmYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYzU2YjIyNTUzZmI0NTA1YmI2MjkxMDQ3YWQzNTI2MS5iaW5kUG9wdXAocG9wdXBfNTY5OTRmM2FjZGRmNDdkMmE1MWU0ZGE3MDFjNDY2M2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGVkNzAzYzM2NzcwNDBlN2IzYTZkNzMxYzJkOWQ1ZjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODI3MzY0LC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTUxOGMyNDhkYjFiNDdkNDhhYTY4NWNkNDAzOWUzMTIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzlmNWY3ZmVmNzgxNGE2M2JmODc3OGU4NzQ5OTcyMWYgPSAkKCc8ZGl2IGlkPSJodG1sX2M5ZjVmN2ZlZjc4MTRhNjNiZjg3NzhlODc0OTk3MjFmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XaWxsb3dkYWxlLCBXaWxsb3dkYWxlIFdlc3QgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U1MThjMjQ4ZGIxYjQ3ZDQ4YWE2ODVjZDQwMzllMzEyLnNldENvbnRlbnQoaHRtbF9jOWY1ZjdmZWY3ODE0YTYzYmY4Nzc4ZTg3NDk5NzIxZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZWQ3MDNjMzY3NzA0MGU3YjNhNmQ3MzFjMmQ5ZDVmMS5iaW5kUG9wdXAocG9wdXBfZTUxOGMyNDhkYjFiNDdkNDhhYTY4NWNkNDAzOWUzMTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTg3Nzk3MmY1N2Y2NGNkNTkxMzNkYmQyOTI2YTE5ZDkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTMyNTg2LC03OS4zMjk2NTY1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODM0N2Y5Yzg0YjA0NDg0MGE4YjM0YjM3OTA0MDE3YTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmRmYTI5MTI2NGE2NGI4OTkzNDI5NjUzN2IyOGZjM2EgPSAkKCc8ZGl2IGlkPSJodG1sXzJkZmEyOTEyNjRhNjRiODk5MzQyOTY1MzdiMjhmYzNhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJrd29vZHMgQ2x1c3RlciAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzgzNDdmOWM4NGIwNDQ4NDBhOGIzNGIzNzkwNDAxN2E1LnNldENvbnRlbnQoaHRtbF8yZGZhMjkxMjY0YTY0Yjg5OTM0Mjk2NTM3YjI4ZmMzYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lODc3OTcyZjU3ZjY0Y2Q1OTEzM2RiZDI5MjZhMTlkOS5iaW5kUG9wdXAocG9wdXBfODM0N2Y5Yzg0YjA0NDg0MGE4YjM0YjM3OTA0MDE3YTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWNjMDNhYjViYjUyNGQ1OGFjOGNlMjg5MjkwYWViZTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NDU5MDU3OTk5OTk5OTYsLTc5LjM1MjE4OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkzZjJkMTY5MThjNjRiMDg5MjcwZGEyZjEwOTEyMDg5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc0ZTFjYmU4ZjZmMTRmZjBhOWJhMmUxNzUxOWIyMWU5ID0gJCgnPGRpdiBpZD0iaHRtbF83NGUxY2JlOGY2ZjE0ZmYwYTliYTJlMTc1MTliMjFlOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9uIE1pbGxzIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85M2YyZDE2OTE4YzY0YjA4OTI3MGRhMmYxMDkxMjA4OS5zZXRDb250ZW50KGh0bWxfNzRlMWNiZThmNmYxNGZmMGE5YmEyZTE3NTE5YjIxZTkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWNjMDNhYjViYjUyNGQ1OGFjOGNlMjg5MjkwYWViZTguYmluZFBvcHVwKHBvcHVwXzkzZjJkMTY5MThjNjRiMDg5MjcwZGEyZjEwOTEyMDg5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FlN2Q0ZDU1MGI5MTQ4NmFhNjRjY2YzMTc0NzY4NWViID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI1ODk5NzAwMDAwMDEsLTc5LjM0MDkyM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JlNWFiOTIzZmU1MDRjYWNhM2Q1ZDY1NTVkYzVhMzc5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RmOTM0Zjg1NzI1MDQ4OWI4Zjc5ZmJmOGYyNDI2ZmU1ID0gJCgnPGRpdiBpZD0iaHRtbF9kZjkzNGY4NTcyNTA0ODliOGY3OWZiZjhmMjQyNmZlNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9uIE1pbGxzIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZTVhYjkyM2ZlNTA0Y2FjYTNkNWQ2NTU1ZGM1YTM3OS5zZXRDb250ZW50KGh0bWxfZGY5MzRmODU3MjUwNDg5YjhmNzlmYmY4ZjI0MjZmZTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWU3ZDRkNTUwYjkxNDg2YWE2NGNjZjMxNzQ3Njg1ZWIuYmluZFBvcHVwKHBvcHVwX2JlNWFiOTIzZmU1MDRjYWNhM2Q1ZDY1NTVkYzVhMzc5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzlhMWI5YjgyY2FhMDQxN2M5OWQzYTZkNTlmYmQ4MmI4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU0MzI4MywtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcwYWY3ZjU3YjAzMDRkN2Y5YTY0Njg5MjA5YTEzYmI0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RjYjIwZTJkNTBjNDQ4N2ViYTVmNTVmNmE1NjY5NDBlID0gJCgnPGRpdiBpZD0iaHRtbF9kY2IyMGUyZDUwYzQ0ODdlYmE1ZjU1ZjZhNTY2OTQwZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF0aHVyc3QgTWFub3IsIFdpbHNvbiBIZWlnaHRzLCBEb3duc3ZpZXcgTm9ydGggQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzcwYWY3ZjU3YjAzMDRkN2Y5YTY0Njg5MjA5YTEzYmI0LnNldENvbnRlbnQoaHRtbF9kY2IyMGUyZDUwYzQ0ODdlYmE1ZjU1ZjZhNTY2OTQwZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85YTFiOWI4MmNhYTA0MTdjOTlkM2E2ZDU5ZmJkODJiOC5iaW5kUG9wdXAocG9wdXBfNzBhZjdmNTdiMDMwNGQ3ZjlhNjQ2ODkyMDlhMTNiYjQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmU2NzM3ODdkMDYyNDcwMGFkYWY2Njg1Zjc4MGI0OTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Njc5ODAzLC03OS40ODcyNjE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMxZDBjZTEwNTI2ZjRjNGRiZTJiYzA0Y2U1NjRkM2FlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFkYjdhZDc2MmVlMzQ3YTc5M2E3ODA5Y2E4ODU3MjJlID0gJCgnPGRpdiBpZD0iaHRtbF8xZGI3YWQ3NjJlZTM0N2E3OTNhNzgwOWNhODg1NzIyZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGh3b29kIFBhcmssIFlvcmsgVW5pdmVyc2l0eSBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzFkMGNlMTA1MjZmNGM0ZGJlMmJjMDRjZTU2NGQzYWUuc2V0Q29udGVudChodG1sXzFkYjdhZDc2MmVlMzQ3YTc5M2E3ODA5Y2E4ODU3MjJlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZlNjczNzg3ZDA2MjQ3MDBhZGFmNjY4NWY3ODBiNDk1LmJpbmRQb3B1cChwb3B1cF8zMWQwY2UxMDUyNmY0YzRkYmUyYmMwNGNlNTY0ZDNhZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84OTNmZjg5YjgyZDk0ZDMwODU1Y2Q1MWE3MGE5NTIzZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczNzQ3MzIwMDAwMDAwNCwtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84ZTE2Zjc4Y2I3ZGY0OTM3YmUxOGYxMDQ2ZmFlYmE5NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZTQ1NmJkODcyODY0YzQwODMyOTRjMjkwMjhlYmUyNCA9ICQoJzxkaXYgaWQ9Imh0bWxfYmU0NTZiZDg3Mjg2NGM0MDgzMjk0YzI5MDI4ZWJlMjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd25zdmlldyBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGUxNmY3OGNiN2RmNDkzN2JlMThmMTA0NmZhZWJhOTUuc2V0Q29udGVudChodG1sX2JlNDU2YmQ4NzI4NjRjNDA4MzI5NGMyOTAyOGViZTI0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg5M2ZmODliODJkOTRkMzA4NTVjZDUxYTcwYTk1MjNlLmJpbmRQb3B1cChwb3B1cF84ZTE2Zjc4Y2I3ZGY0OTM3YmUxOGYxMDQ2ZmFlYmE5NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZTQ4Nzg4ZWExZDM0OGRjYmViNzQwM2I2ODE1ZDhjYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTAxNDYsLTc5LjUwNjk0MzZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kYTI3NzU5ZWU4NDM0N2Y0YWViZGE5MzFjN2VmOTIyYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jOTQ2YWZiYTc5NzM0NjFkOWVhNmE2NDIzOThkYjdkZCA9ICQoJzxkaXYgaWQ9Imh0bWxfYzk0NmFmYmE3OTczNDYxZDllYTZhNjQyMzk4ZGI3ZGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd25zdmlldyBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGEyNzc1OWVlODQzNDdmNGFlYmRhOTMxYzdlZjkyMmMuc2V0Q29udGVudChodG1sX2M5NDZhZmJhNzk3MzQ2MWQ5ZWE2YTY0MjM5OGRiN2RkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VlNDg3ODhlYTFkMzQ4ZGNiZWI3NDAzYjY4MTVkOGNjLmJpbmRQb3B1cChwb3B1cF9kYTI3NzU5ZWU4NDM0N2Y0YWViZGE5MzFjN2VmOTIyYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81ODkxOGRhMTI2MjU0ZTQ1YWZmNzczOTllNDU1Y2Y5OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyODQ5NjQsLTc5LjQ5NTY5NzQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTdiM2UxYjk3NjExNDgyMzk3NGY1MmMxYzAwNDM3NGIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODZmOTJjZTk2ZWFjNDlmZWIyODU0YzZmYmYwOGZjMTMgPSAkKCc8ZGl2IGlkPSJodG1sXzg2ZjkyY2U5NmVhYzQ5ZmViMjg1NGM2ZmJmMDhmYzEzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U3YjNlMWI5NzYxMTQ4MjM5NzRmNTJjMWMwMDQzNzRiLnNldENvbnRlbnQoaHRtbF84NmY5MmNlOTZlYWM0OWZlYjI4NTRjNmZiZjA4ZmMxMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81ODkxOGRhMTI2MjU0ZTQ1YWZmNzczOTllNDU1Y2Y5OS5iaW5kUG9wdXAocG9wdXBfZTdiM2UxYjk3NjExNDgyMzk3NGY1MmMxYzAwNDM3NGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWM0ZjkxMzEwOTcyNDc3YmE1NGFiNGQ3YWJkZGFkZWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NjE2MzEzLC03OS41MjA5OTk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FjZDY4NmJkMjcxOTRlMDVhNGU1ODlkN2U3ZDY0NTc0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzIxODhiMTdmYzExZDQwZWU5OTNiYjUzMjU4NmQ5ODg3ID0gJCgnPGRpdiBpZD0iaHRtbF8yMTg4YjE3ZmMxMWQ0MGVlOTkzYmI1MzI1ODZkOTg4NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnN2aWV3IENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hY2Q2ODZiZDI3MTk0ZTA1YTRlNTg5ZDdlN2Q2NDU3NC5zZXRDb250ZW50KGh0bWxfMjE4OGIxN2ZjMTFkNDBlZTk5M2JiNTMyNTg2ZDk4ODcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWM0ZjkxMzEwOTcyNDc3YmE1NGFiNGQ3YWJkZGFkZWUuYmluZFBvcHVwKHBvcHVwX2FjZDY4NmJkMjcxOTRlMDVhNGU1ODlkN2U3ZDY0NTc0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZkZjZiMDY0YjQyMTQ1ZDlhODBmNDRiZWUxMzg0YzQ4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI1ODgyMjk5OTk5OTk1LC03OS4zMTU1NzE1OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzEyNDJmNTI0MWJkYTRkMzc4MzVhODllZThiODY0NmNiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RmZjZkYzRiMWM0ZDQwYzY4ZWQ5NDdmOTU5ODNiOTIzID0gJCgnPGRpdiBpZD0iaHRtbF9kZmY2ZGM0YjFjNGQ0MGM2OGVkOTQ3Zjk1OTgzYjkyMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VmljdG9yaWEgVmlsbGFnZSBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTI0MmY1MjQxYmRhNGQzNzgzNWE4OWVlOGI4NjQ2Y2Iuc2V0Q29udGVudChodG1sX2RmZjZkYzRiMWM0ZDQwYzY4ZWQ5NDdmOTU5ODNiOTIzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZkZjZiMDY0YjQyMTQ1ZDlhODBmNDRiZWUxMzg0YzQ4LmJpbmRQb3B1cChwb3B1cF8xMjQyZjUyNDFiZGE0ZDM3ODM1YTg5ZWU4Yjg2NDZjYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kOWZhODI5NTRjOTc0ZmE4YmI1NTYxZGY3Yjc0YzcxMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNjM5NzIsLTc5LjMwOTkzN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY5NWZjMTk1OTIwNjQwMzY5OWZlNjY5ODE2ZDE5NGI0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2MyMGZiNTNjY2M0MDQ1MjNhYmFjMmVkOTk2MGRiZmJkID0gJCgnPGRpdiBpZD0iaHRtbF9jMjBmYjUzY2NjNDA0NTIzYWJhYzJlZDk5NjBkYmZiZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya3ZpZXcgSGlsbCwgV29vZGJpbmUgR2FyZGVucyBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjk1ZmMxOTU5MjA2NDAzNjk5ZmU2Njk4MTZkMTk0YjQuc2V0Q29udGVudChodG1sX2MyMGZiNTNjY2M0MDQ1MjNhYmFjMmVkOTk2MGRiZmJkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q5ZmE4Mjk1NGM5NzRmYThiYjU1NjFkZjdiNzRjNzEzLmJpbmRQb3B1cChwb3B1cF82OTVmYzE5NTkyMDY0MDM2OTlmZTY2OTgxNmQxOTRiNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83ZTEwYTBmYThkNTg0MDNmOWMzODE1ZmIyYjAyMDI4MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5NTM0MzkwMDAwMDAwNSwtNzkuMzE4Mzg4N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY1YTRlOThkY2JiYzQ3ZGZiNjJhOWViNGIyZDdmNDc3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY0ZWNjNzcxZTJiZTQ2MWU4ZDA3NGI2MmJlYWRhMjQ3ID0gJCgnPGRpdiBpZD0iaHRtbF82NGVjYzc3MWUyYmU0NjFlOGQwNzRiNjJiZWFkYTI0NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V29vZGJpbmUgSGVpZ2h0cyBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjVhNGU5OGRjYmJjNDdkZmI2MmE5ZWI0YjJkN2Y0Nzcuc2V0Q29udGVudChodG1sXzY0ZWNjNzcxZTJiZTQ2MWU4ZDA3NGI2MmJlYWRhMjQ3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdlMTBhMGZhOGQ1ODQwM2Y5YzM4MTVmYjJiMDIwMjgwLmJpbmRQb3B1cChwb3B1cF82NWE0ZTk4ZGNiYmM0N2RmYjYyYTllYjRiMmQ3ZjQ3Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81Y2JmYWMyZDFlNzQ0MjAwOGQxODdlZmY0YTBlMzdhMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3NjM1NzM5OTk5OTk5LC03OS4yOTMwMzEyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzVhNTYwMWQxYTA4NDZkOTg0NDQyYWZhMmNkZTNlZDYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjVmYzVlOWQzN2NjNDBmOWI0MDY4OTUyMzk5MzA4NzIgPSAkKCc8ZGl2IGlkPSJodG1sXzI1ZmM1ZTlkMzdjYzQwZjliNDA2ODk1MjM5OTMwODcyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgQmVhY2hlcyBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzVhNTYwMWQxYTA4NDZkOTg0NDQyYWZhMmNkZTNlZDYuc2V0Q29udGVudChodG1sXzI1ZmM1ZTlkMzdjYzQwZjliNDA2ODk1MjM5OTMwODcyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzVjYmZhYzJkMWU3NDQyMDA4ZDE4N2VmZjRhMGUzN2ExLmJpbmRQb3B1cChwb3B1cF83NWE1NjAxZDFhMDg0NmQ5ODQ0NDJhZmEyY2RlM2VkNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wZmUyNzdjNGY2YmM0NGRlOTYwMWM3MjcyZmMxZjE2MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwOTA2MDQsLTc5LjM2MzQ1MTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84YzllMWVmYjA5Yjg0NGFjYmEyMzJjNzczYzY5M2E3ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80OGFhMzgzNDNiNTg0NDlkOWNlZDcwMTRkOTJmNWFmYyA9ICQoJzxkaXYgaWQ9Imh0bWxfNDhhYTM4MzQzYjU4NDQ5ZDljZWQ3MDE0ZDkyZjVhZmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxlYXNpZGUgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhjOWUxZWZiMDliODQ0YWNiYTIzMmM3NzNjNjkzYTdkLnNldENvbnRlbnQoaHRtbF80OGFhMzgzNDNiNTg0NDlkOWNlZDcwMTRkOTJmNWFmYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZmUyNzdjNGY2YmM0NGRlOTYwMWM3MjcyZmMxZjE2MS5iaW5kUG9wdXAocG9wdXBfOGM5ZTFlZmIwOWI4NDRhY2JhMjMyYzc3M2M2OTNhN2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzRiNjE5ODZhYTg0NDk0Njg0ODIzYmJkNzJiZTNjOTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDUzNjg5LC03OS4zNDkzNzE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JmNjlmYTkyZGM5MjQyZTI4MTk4YTY3YTU3ZDljODBkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y3NTZkNWRiYWI1MDRkYTFhZDk1MTNlOTZiNGZmMTI1ID0gJCgnPGRpdiBpZD0iaHRtbF9mNzU2ZDVkYmFiNTA0ZGExYWQ5NTEzZTk2YjRmZjEyNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhvcm5jbGlmZmUgUGFyayBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmY2OWZhOTJkYzkyNDJlMjgxOThhNjdhNTdkOWM4MGQuc2V0Q29udGVudChodG1sX2Y3NTZkNWRiYWI1MDRkYTFhZDk1MTNlOTZiNGZmMTI1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM0YjYxOTg2YWE4NDQ5NDY4NDgyM2JiZDcyYmUzYzk5LmJpbmRQb3B1cChwb3B1cF9iZjY5ZmE5MmRjOTI0MmUyODE5OGE2N2E1N2Q5YzgwZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83ZjM3M2E2YzQ4YjI0MTI1ODdlYzI4ODFjMDgzNWQzNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4NTM0NywtNzkuMzM4MTA2NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2MzYTkyOTQ1NzllODQzYzhhZjliMjhiZjlhMWIxM2FkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdjNGIwNDgxZWU5YzQ5Njk4YjRmZWU0MGQxMWZiMTRhID0gJCgnPGRpdiBpZD0iaHRtbF83YzRiMDQ4MWVlOWM0OTY5OGI0ZmVlNDBkMTFmYjE0YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvLCBCcm9hZHZpZXcgTm9ydGggKE9sZCBFYXN0IFlvcmspIENsdXN0ZXIgMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jM2E5Mjk0NTc5ZTg0M2M4YWY5YjI4YmY5YTFiMTNhZC5zZXRDb250ZW50KGh0bWxfN2M0YjA0ODFlZTljNDk2OThiNGZlZTQwZDExZmIxNGEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2YzNzNhNmM0OGIyNDEyNTg3ZWMyODgxYzA4MzVkMzcuYmluZFBvcHVwKHBvcHVwX2MzYTkyOTQ1NzllODQzYzhhZjliMjhiZjlhMWIxM2FkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY3Y2NjM2FjYTgxMjQwNmM5NTBkODk5MWY3NmZiZTVkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTU3MSwtNzkuMzUyMTg4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGIwYjZiMmVmOGRiNDE1NGFhOTQyZmU0OGViMzAzMTQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjQ1ZDY0NGNjZGIwNGI5NGIzNDU3OTdkNWZkMzM2ZDggPSAkKCc8ZGl2IGlkPSJodG1sXzI0NWQ2NDRjY2RiMDRiOTRiMzQ1Nzk3ZDVmZDMzNmQ4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgRGFuZm9ydGggV2VzdCwgUml2ZXJkYWxlIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80YjBiNmIyZWY4ZGI0MTU0YWE5NDJmZTQ4ZWIzMDMxNC5zZXRDb250ZW50KGh0bWxfMjQ1ZDY0NGNjZGIwNGI5NGIzNDU3OTdkNWZkMzM2ZDgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjdjY2MzYWNhODEyNDA2Yzk1MGQ4OTkxZjc2ZmJlNWQuYmluZFBvcHVwKHBvcHVwXzRiMGI2YjJlZjhkYjQxNTRhYTk0MmZlNDhlYjMwMzE0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI1YTNmMmVlZjM1YTRmODE4NGViMWE4MDAzMjk3NGI4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY4OTk4NSwtNzkuMzE1NTcxNTk5OTk5OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ODc2MGFkNWNiMWE0YmY2YWQ1NjZjNTE5YzFlM2RlNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hYzVhM2U2YjVjODc0NGQzOGUyMWJiY2NiMGE2YmY4YSA9ICQoJzxkaXYgaWQ9Imh0bWxfYWM1YTNlNmI1Yzg3NDRkMzhlMjFiYmNjYjBhNmJmOGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkluZGlhIEJhemFhciwgVGhlIEJlYWNoZXMgV2VzdCBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTg3NjBhZDVjYjFhNGJmNmFkNTY2YzUxOWMxZTNkZTcuc2V0Q29udGVudChodG1sX2FjNWEzZTZiNWM4NzQ0ZDM4ZTIxYmJjY2IwYTZiZjhhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI1YTNmMmVlZjM1YTRmODE4NGViMWE4MDAzMjk3NGI4LmJpbmRQb3B1cChwb3B1cF81ODc2MGFkNWNiMWE0YmY2YWQ1NjZjNTE5YzFlM2RlNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lMGE1MDQwYWQ3MGQ0Nzc1ODU2MTVjNjcyYzMwYWQyYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1OTUyNTUsLTc5LjM0MDkyM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQyYTM4ZGE0MjA4YjQ5ODY4NzE0MjA2YzVmZjI4OWYwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzExYmVkNDE3NmRmYzRmMjJhMDFiNTViYzdmMDQ2YWMzID0gJCgnPGRpdiBpZD0iaHRtbF8xMWJlZDQxNzZkZmM0ZjIyYTAxYjU1YmM3ZjA0NmFjMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3R1ZGlvIERpc3RyaWN0IENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80MmEzOGRhNDIwOGI0OTg2ODcxNDIwNmM1ZmYyODlmMC5zZXRDb250ZW50KGh0bWxfMTFiZWQ0MTc2ZGZjNGYyMmEwMWI1NWJjN2YwNDZhYzMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTBhNTA0MGFkNzBkNDc3NTg1NjE1YzY3MmMzMGFkMmEuYmluZFBvcHVwKHBvcHVwXzQyYTM4ZGE0MjA4YjQ5ODY4NzE0MjA2YzVmZjI4OWYwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2EwYzRmM2Q3MDA3NTQwNTI5MzY5NWM1MGI5MTZiMGEwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI4MDIwNSwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RmNDg3ZGUxYmZmYTRmMjdiMTM2ZTU3MzI2M2Q4M2NkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg3YmViZTliNWViMjRiNDZhOTZiMjdjMDI0ZjYzMGM3ID0gJCgnPGRpdiBpZD0iaHRtbF84N2JlYmU5YjVlYjI0YjQ2YTk2YjI3YzAyNGY2MzBjNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGF3cmVuY2UgUGFyayBDbHVzdGVyIDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGY0ODdkZTFiZmZhNGYyN2IxMzZlNTczMjYzZDgzY2Quc2V0Q29udGVudChodG1sXzg3YmViZTliNWViMjRiNDZhOTZiMjdjMDI0ZjYzMGM3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2EwYzRmM2Q3MDA3NTQwNTI5MzY5NWM1MGI5MTZiMGEwLmJpbmRQb3B1cChwb3B1cF9kZjQ4N2RlMWJmZmE0ZjI3YjEzNmU1NzMyNjNkODNjZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lMjc1YjYwODA5ZjQ0MmM0YWMxYTE5YzczODlkYzBjNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMjc1MTEsLTc5LjM5MDE5NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ZjhmZGQ5YmQzMzE0NjhkYmY3MGJiYjcwMmZkOWU3NCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hM2RkZTMyMmZlZjY0NTEyYWE1NzZjOGQ2M2NiMDU3ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfYTNkZGUzMjJmZWY2NDUxMmFhNTc2YzhkNjNjYjA1N2UiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUgTm9ydGggQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRmOGZkZDliZDMzMTQ2OGRiZjcwYmJiNzAyZmQ5ZTc0LnNldENvbnRlbnQoaHRtbF9hM2RkZTMyMmZlZjY0NTEyYWE1NzZjOGQ2M2NiMDU3ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lMjc1YjYwODA5ZjQ0MmM0YWMxYTE5YzczODlkYzBjNC5iaW5kUG9wdXAocG9wdXBfNGY4ZmRkOWJkMzMxNDY4ZGJmNzBiYmI3MDJmZDllNzQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjkxZjM4MjkyNTI4NDY5Y2FjY2NlZGU2MjI5YjNhZmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTUzODM0LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkxZDJmOTk5OTViYzQ2ZTA5NTJkZDU0MmI2ZWExZTczID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQxNjYwNzIxYWQwZDQ0NGY5N2M2NzY0NWQyYjllNDhiID0gJCgnPGRpdiBpZD0iaHRtbF80MTY2MDcyMWFkMGQ0NDRmOTdjNjc2NDVkMmI5ZTQ4YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGggVG9yb250byBXZXN0LCAgTGF3cmVuY2UgUGFyayBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTFkMmY5OTk5NWJjNDZlMDk1MmRkNTQyYjZlYTFlNzMuc2V0Q29udGVudChodG1sXzQxNjYwNzIxYWQwZDQ0NGY5N2M2NzY0NWQyYjllNDhiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY5MWYzODI5MjUyODQ2OWNhY2NjZWRlNjIyOWIzYWZjLmJpbmRQb3B1cChwb3B1cF85MWQyZjk5OTk1YmM0NmUwOTUyZGQ1NDJiNmVhMWU3Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82M2E4NzU4OTIxN2I0MzAyOWY1ZWM4ZjFhNTAzOGY3MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNDMyNDQsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lZDgwYzU3ZjAwNTc0MDhlYTM1OTA5MzMzOWZkMGY0OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kOTY5MTI5YzQ3MmY0YzdlODI0YjMyYjU1NmY0YTljZCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDk2OTEyOWM0NzJmNGM3ZTgyNGIzMmI1NTZmNGE5Y2QiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VkODBjNTdmMDA1NzQwOGVhMzU5MDkzMzM5ZmQwZjQ4LnNldENvbnRlbnQoaHRtbF9kOTY5MTI5YzQ3MmY0YzdlODI0YjMyYjU1NmY0YTljZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82M2E4NzU4OTIxN2I0MzAyOWY1ZWM4ZjFhNTAzOGY3MC5iaW5kUG9wdXAocG9wdXBfZWQ4MGM1N2YwMDU3NDA4ZWEzNTkwOTMzMzlmZDBmNDgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTg0M2RhMWZlNzc3NDM1Yzg5YWNhYzA4ZjZlYzg5NzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODk1NzQzLC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYwMjFmZTMxNWRiYzQ5YTg5NTJiMjFlNzYzOWU3MzRhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVlNjc4M2M3OWJkOTQzNjNiNDM2MDQzMTJjZmFjZTUzID0gJCgnPGRpdiBpZD0iaHRtbF81ZTY3ODNjNzliZDk0MzYzYjQzNjA0MzEyY2ZhY2U1MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TW9vcmUgUGFyaywgU3VtbWVyaGlsbCBFYXN0IENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82MDIxZmUzMTVkYmM0OWE4OTUyYjIxZTc2MzllNzM0YS5zZXRDb250ZW50KGh0bWxfNWU2NzgzYzc5YmQ5NDM2M2I0MzYwNDMxMmNmYWNlNTMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTg0M2RhMWZlNzc3NDM1Yzg5YWNhYzA4ZjZlYzg5NzAuYmluZFBvcHVwKHBvcHVwXzYwMjFmZTMxNWRiYzQ5YTg5NTJiMjFlNzYzOWU3MzRhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFlOTcyZmM5ZGJkNjQ1Njk4YzBkMzgwN2YzYTU1NzBiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg2NDEyMjk5OTk5OTksLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84Mjk1Y2U5ZjVlZTY0ZWRiOWY1OTRhOTQ5ODdiMzVlMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNTVmOThlZTc3MWQ0NjUyOTQ4ODRlNzYxODM0OTAxYSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTU1Zjk4ZWU3NzFkNDY1Mjk0ODg0ZTc2MTgzNDkwMWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN1bW1lcmhpbGwgV2VzdCwgUmF0aG5lbGx5LCBTb3V0aCBIaWxsLCBGb3Jlc3QgSGlsbCBTRSwgRGVlciBQYXJrIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84Mjk1Y2U5ZjVlZTY0ZWRiOWY1OTRhOTQ5ODdiMzVlMS5zZXRDb250ZW50KGh0bWxfZTU1Zjk4ZWU3NzFkNDY1Mjk0ODg0ZTc2MTgzNDkwMWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWU5NzJmYzlkYmQ2NDU2OThjMGQzODA3ZjNhNTU3MGIuYmluZFBvcHVwKHBvcHVwXzgyOTVjZTlmNWVlNjRlZGI5ZjU5NGE5NDk4N2IzNWUxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc3OTRiMGYwZjVmNDQ1OTNhODU0OTNlMDQ1MTAzZWYyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTYyNiwtNzkuMzc3NTI5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zYmFiMDM3YjY1ODM0MDEwOThlNDA0MTIwNjhjMDQ3MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YmM5NmMzNGM1NTQ0Nzg0YjhmY2RmNmUyYjE1YmU5MyA9ICQoJzxkaXYgaWQ9Imh0bWxfN2JjOTZjMzRjNTU0NDc4NGI4ZmNkZjZlMmIxNWJlOTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VkYWxlIENsdXN0ZXIgMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zYmFiMDM3YjY1ODM0MDEwOThlNDA0MTIwNjhjMDQ3MC5zZXRDb250ZW50KGh0bWxfN2JjOTZjMzRjNTU0NDc4NGI4ZmNkZjZlMmIxNWJlOTMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzc5NGIwZjBmNWY0NDU5M2E4NTQ5M2UwNDUxMDNlZjIuYmluZFBvcHVwKHBvcHVwXzNiYWIwMzdiNjU4MzQwMTA5OGU0MDQxMjA2OGMwNDcwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzcyODQxNjViMTU2OTQ4Y2U4NzBiYTU3ZWU4NDRkNDk4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY3OTY3LC03OS4zNjc2NzUzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGZiZTIxYWFlYmQ0NDBiZDkwNDZkZjE2YWRlZjU3OTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOGU0ZjA4YTQ1ZmUwNGFlM2I0OGI1MzdhMzljYTcyNDkgPSAkKCc8ZGl2IGlkPSJodG1sXzhlNGYwOGE0NWZlMDRhZTNiNDhiNTM3YTM5Y2E3MjQ5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdC4gSmFtZXMgVG93biwgQ2FiYmFnZXRvd24gQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRmYmUyMWFhZWJkNDQwYmQ5MDQ2ZGYxNmFkZWY1Nzk4LnNldENvbnRlbnQoaHRtbF84ZTRmMDhhNDVmZTA0YWUzYjQ4YjUzN2EzOWNhNzI0OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83Mjg0MTY1YjE1Njk0OGNlODcwYmE1N2VlODQ0ZDQ5OC5iaW5kUG9wdXAocG9wdXBfNGZiZTIxYWFlYmQ0NDBiZDkwNDZkZjE2YWRlZjU3OTgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDJmZGZlNWU5YjQ5NGYwYjk3NGNjYjc4MmIwYjY4OTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg4N2M3ZTZlOTlhNjQ5MTQ4ZjQ1NTc3ZTZjNDBmNTdmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzM0ZDUwMWVlMDgwMDRjYjRhMjMzYTIyYjJmM2E1N2YxID0gJCgnPGRpdiBpZD0iaHRtbF8zNGQ1MDFlZTA4MDA0Y2I0YTIzM2EyMmIyZjNhNTdmMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2h1cmNoIGFuZCBXZWxsZXNsZXkgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg4N2M3ZTZlOTlhNjQ5MTQ4ZjQ1NTc3ZTZjNDBmNTdmLnNldENvbnRlbnQoaHRtbF8zNGQ1MDFlZTA4MDA0Y2I0YTIzM2EyMmIyZjNhNTdmMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wMmZkZmU1ZTliNDk0ZjBiOTc0Y2NiNzgyYjBiNjg5OC5iaW5kUG9wdXAocG9wdXBfODg3YzdlNmU5OWE2NDkxNDhmNDU1NzdlNmM0MGY1N2YpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzQ4ZGEzYTlkYmI2NDkyOGJjZDQ2MmQyMzIwMmMzODQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTQyNTk5LC03OS4zNjA2MzU5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2I4ZmM1NmRmMTY5NGJmYTk0YzU3YTZkNDRlYjY4YWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjFlOWRmOTI4MDBiNDEwNGIzZjI0NWY0ZGY2ZWJhZTYgPSAkKCc8ZGl2IGlkPSJodG1sXzIxZTlkZjkyODAwYjQxMDRiM2YyNDVmNGRmNmViYWU2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SZWdlbnQgUGFyaywgSGFyYm91cmZyb250IENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jYjhmYzU2ZGYxNjk0YmZhOTRjNTdhNmQ0NGViNjhhZi5zZXRDb250ZW50KGh0bWxfMjFlOWRmOTI4MDBiNDEwNGIzZjI0NWY0ZGY2ZWJhZTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzQ4ZGEzYTlkYmI2NDkyOGJjZDQ2MmQyMzIwMmMzODQuYmluZFBvcHVwKHBvcHVwX2NiOGZjNTZkZjE2OTRiZmE5NGM1N2E2ZDQ0ZWI2OGFmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I5ZTBlMTc4MGJmMDRhYjFiN2RkYjE4MzVkOTI0MmM4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZmZjZjE0YWZjNGE0N2Q4YWZkNjYwYTMyZDIzNWE5YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kMDAyM2RiNGYwOGY0ZTUwYWM2ZmYwMjI5NzhjZThkOCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDAwMjNkYjRmMDhmNGU1MGFjNmZmMDIyOTc4Y2U4ZDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdhcmRlbiBEaXN0cmljdCwgUnllcnNvbiBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmZmY2YxNGFmYzRhNDdkOGFmZDY2MGEzMmQyMzVhOWMuc2V0Q29udGVudChodG1sX2QwMDIzZGI0ZjA4ZjRlNTBhYzZmZjAyMjk3OGNlOGQ4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I5ZTBlMTc4MGJmMDRhYjFiN2RkYjE4MzVkOTI0MmM4LmJpbmRQb3B1cChwb3B1cF9iZmZjZjE0YWZjNGE0N2Q4YWZkNjYwYTMyZDIzNWE5Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYTcwNjVlYmU3MDY0NzM2OTQwZGQ4OGZkNGZhMDk1MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTQ5MzksLTc5LjM3NTQxNzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hOWNlMGEwMDA3YjA0YjI0YjdlYmJlMWIxMDQ2NTNhMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80YmQ1YzdiMmQyMmU0NDVjYmRmMTRiODdmZDFlYThmYSA9ICQoJzxkaXYgaWQ9Imh0bWxfNGJkNWM3YjJkMjJlNDQ1Y2JkZjE0Yjg3ZmQxZWE4ZmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hOWNlMGEwMDA3YjA0YjI0YjdlYmJlMWIxMDQ2NTNhMy5zZXRDb250ZW50KGh0bWxfNGJkNWM3YjJkMjJlNDQ1Y2JkZjE0Yjg3ZmQxZWE4ZmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGE3MDY1ZWJlNzA2NDczNjk0MGRkODhmZDRmYTA5NTAuYmluZFBvcHVwKHBvcHVwX2E5Y2UwYTAwMDdiMDRiMjRiN2ViYmUxYjEwNDY1M2EzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBlNDEyYzYzOGFhZDRmNWY5YjMyNjAyZGViNzM1N2JkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ0NzcwNzk5OTk5OTk2LC03OS4zNzMzMDY0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTE3MTQzZGQzZWY0NGNiOTk1MjU4Yjg5ZDViMzA3NzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2Y0YmE1ZTRiMzYzNDQ2ZTg0OGQ3YmMzM2U1ODg5OWIgPSAkKCc8ZGl2IGlkPSJodG1sX2NmNGJhNWU0YjM2MzQ0NmU4NDhkN2JjMzNlNTg4OTliIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZXJjenkgUGFyayBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTE3MTQzZGQzZWY0NGNiOTk1MjU4Yjg5ZDViMzA3NzEuc2V0Q29udGVudChodG1sX2NmNGJhNWU0YjM2MzQ0NmU4NDhkN2JjMzNlNTg4OTliKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBlNDEyYzYzOGFhZDRmNWY5YjMyNjAyZGViNzM1N2JkLmJpbmRQb3B1cChwb3B1cF85MTcxNDNkZDNlZjQ0Y2I5OTUyNThiODlkNWIzMDc3MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZDBkNjcwMDBmYmE0M2FjOTQ3YTNjMzZiOGU0NTYwZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1Nzk1MjQsLTc5LjM4NzM4MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNzMxNGY4MDJmYmE0ZDhiYTZlM2Y2NGIwZTVkODc0OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wNDIxMWE2NjM1YTQ0ZWQ1YTZlZjE1OWY4OThkNzMwNSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDQyMTFhNjYzNWE0NGVkNWE2ZWYxNTlmODk4ZDczMDUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgQmF5IFN0cmVldCBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjczMTRmODAyZmJhNGQ4YmE2ZTNmNjRiMGU1ZDg3NDkuc2V0Q29udGVudChodG1sXzA0MjExYTY2MzVhNDRlZDVhNmVmMTU5Zjg5OGQ3MzA1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzlkMGQ2NzAwMGZiYTQzYWM5NDdhM2MzNmI4ZTQ1NjBmLmJpbmRQb3B1cChwb3B1cF9mNzMxNGY4MDJmYmE0ZDhiYTZlM2Y2NGIwZTVkODc0OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84NzI3YTc0ZDUyNTM0MzM1OGEyZTJkYTkwOTE3YzBlMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDU3MTIwMDAwMDAxLC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDY1MTk2OTBmODZkNGJkMWJhOWQ5NTM2NDA5ZTVkNjMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTQyZDk2N2MwNDBhNGRlNTg5YTVlZWQ3ZGIyZDg2NGUgPSAkKCc8ZGl2IGlkPSJodG1sXzE0MmQ5NjdjMDQwYTRkZTU4OWE1ZWVkN2RiMmQ4NjRlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SaWNobW9uZCwgQWRlbGFpZGUsIEtpbmcgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ2NTE5NjkwZjg2ZDRiZDFiYTlkOTUzNjQwOWU1ZDYzLnNldENvbnRlbnQoaHRtbF8xNDJkOTY3YzA0MGE0ZGU1ODlhNWVlZDdkYjJkODY0ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NzI3YTc0ZDUyNTM0MzM1OGEyZTJkYTkwOTE3YzBlMi5iaW5kUG9wdXAocG9wdXBfNDY1MTk2OTBmODZkNGJkMWJhOWQ5NTM2NDA5ZTVkNjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGJlMzA0Y2JiZTViNGRmMmE4OGVmMDRhODJjMzNmMGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDA4MTU3LC03OS4zODE3NTIyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYyZjgxMTM3YTZhMTQ5MDg5YTU0YmQwMWNlMTNlZGM5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYxZjk3ZTE1NTZjZTQ5ZmI5ZmYzNGJmYTYzMGFiY2M5ID0gJCgnPGRpdiBpZD0iaHRtbF82MWY5N2UxNTU2Y2U0OWZiOWZmMzRiZmE2MzBhYmNjOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250IEVhc3QsIFVuaW9uIFN0YXRpb24sIFRvcm9udG8gSXNsYW5kcyBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjJmODExMzdhNmExNDkwODlhNTRiZDAxY2UxM2VkYzkuc2V0Q29udGVudChodG1sXzYxZjk3ZTE1NTZjZTQ5ZmI5ZmYzNGJmYTYzMGFiY2M5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBiZTMwNGNiYmU1YjRkZjJhODhlZjA0YTgyYzMzZjBiLmJpbmRQb3B1cChwb3B1cF82MmY4MTEzN2E2YTE0OTA4OWE1NGJkMDFjZTEzZWRjOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84OGVmNmZlNzk2ZjM0MGQ5OWViOWQ1ODgxYzUyODdjNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTAxZGZlODEzMDA5NDk4YWFkOWIzNDk4MTMxZmJkMTQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTMzZGZlMzBhNjVmNDc0NWIzNmVmYWFlY2IzNzhmYTIgPSAkKCc8ZGl2IGlkPSJodG1sXzkzM2RmZTMwYTY1ZjQ3NDViMzZlZmFhZWNiMzc4ZmEyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ub3JvbnRvIERvbWluaW9uIENlbnRyZSwgRGVzaWduIEV4Y2hhbmdlIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lMDFkZmU4MTMwMDk0OThhYWQ5YjM0OTgxMzFmYmQxNC5zZXRDb250ZW50KGh0bWxfOTMzZGZlMzBhNjVmNDc0NWIzNmVmYWFlY2IzNzhmYTIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODhlZjZmZTc5NmYzNDBkOTllYjlkNTg4MWM1Mjg3YzYuYmluZFBvcHVwKHBvcHVwX2UwMWRmZTgxMzAwOTQ5OGFhZDliMzQ5ODEzMWZiZDE0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYzZjQzMGRjZjdkMTRhZTU5NjYyNzM4Nzc1MjU0OTk0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4MTk4NSwtNzkuMzc5ODE2OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNGY4NTE1ODJjYzg0NmM0ODIzMTYwNjEwOTRkNDhkZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84NDkxNGU4NjAzZDM0ZTU0YjkyZjgyZTJiOTZlNTQ2MyA9ICQoJzxkaXYgaWQ9Imh0bWxfODQ5MTRlODYwM2QzNGU1NGI5MmY4MmUyYjk2ZTU0NjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNvbW1lcmNlIENvdXJ0LCBWaWN0b3JpYSBIb3RlbCBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzRmODUxNTgyY2M4NDZjNDgyMzE2MDYxMDk0ZDQ4ZGQuc2V0Q29udGVudChodG1sXzg0OTE0ZTg2MDNkMzRlNTRiOTJmODJlMmI5NmU1NDYzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYzZjQzMGRjZjdkMTRhZTU5NjYyNzM4Nzc1MjU0OTk0LmJpbmRQb3B1cChwb3B1cF8zNGY4NTE1ODJjYzg0NmM0ODIzMTYwNjEwOTRkNDhkZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jYzZlMTg4NTM5ZmI0ZjkzOGQ5Mzg2ZWJmMzg1MjFkNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczMzI4MjUsLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84ZDIzMzA2YTdjOWY0ZDE0YTQ5YTlhYWIzMjNlZTc3MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NDZiZDRlYTMxNzY0Mzk4ODAxYTZmYWM2Y2I4MTAyNiA9ICQoJzxkaXYgaWQ9Imh0bWxfNzQ2YmQ0ZWEzMTc2NDM5ODgwMWE2ZmFjNmNiODEwMjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJlZGZvcmQgUGFyaywgTGF3cmVuY2UgTWFub3IgRWFzdCBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGQyMzMwNmE3YzlmNGQxNGE0OWE5YWFiMzIzZWU3NzMuc2V0Q29udGVudChodG1sXzc0NmJkNGVhMzE3NjQzOTg4MDFhNmZhYzZjYjgxMDI2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NjNmUxODg1MzlmYjRmOTM4ZDkzODZlYmYzODUyMWQ3LmJpbmRQb3B1cChwb3B1cF84ZDIzMzA2YTdjOWY0ZDE0YTQ5YTlhYWIzMjNlZTc3Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZDhmNjRjYTM2MGE0NTM0ODU2MWEzM2IyOThkNTBmMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMTY5NDgsLTc5LjQxNjkzNTU5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjM5M2NjY2ZiNTJlNDQwMWE4YjlkMWIwOTk1MjFhZjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODdjMThhYjdiODk4NDBmMmJlN2M0Nzk1ZTU1Y2QzZDkgPSAkKCc8ZGl2IGlkPSJodG1sXzg3YzE4YWI3Yjg5ODQwZjJiZTdjNDc5NWU1NWNkM2Q5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlbGF3biBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjM5M2NjY2ZiNTJlNDQwMWE4YjlkMWIwOTk1MjFhZjYuc2V0Q29udGVudChodG1sXzg3YzE4YWI3Yjg5ODQwZjJiZTdjNDc5NWU1NWNkM2Q5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzlkOGY2NGNhMzYwYTQ1MzQ4NTYxYTMzYjI5OGQ1MGYzLmJpbmRQb3B1cChwb3B1cF8yMzkzY2NjZmI1MmU0NDAxYThiOWQxYjA5OTUyMWFmNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNGQwMmZhNGE0NWQ0MjMxYmI0NGE1YTZjMzE5ZGZmMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5Njk0NzYsLTc5LjQxMTMwNzIwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDkwYzUxNjdhZjU3NDExNjg3MTE0MTFlYzA0MGE4NzkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2UyOTEzYjA0OGIxNGVlNmIyZTU0MDI3MzBhMDQyZjMgPSAkKCc8ZGl2IGlkPSJodG1sXzNlMjkxM2IwNDhiMTRlZTZiMmU1NDAyNzMwYTA0MmYzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Gb3Jlc3QgSGlsbCBOb3J0aCAmYW1wOyBXZXN0LCBGb3Jlc3QgSGlsbCBSb2FkIFBhcmsgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ5MGM1MTY3YWY1NzQxMTY4NzExNDExZWMwNDBhODc5LnNldENvbnRlbnQoaHRtbF8zZTI5MTNiMDQ4YjE0ZWU2YjJlNTQwMjczMGEwNDJmMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lNGQwMmZhNGE0NWQ0MjMxYmI0NGE1YTZjMzE5ZGZmMS5iaW5kUG9wdXAocG9wdXBfNDkwYzUxNjdhZjU3NDExNjg3MTE0MTFlYzA0MGE4NzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTJiYWQxM2E4YjAwNGIwY2I2MzhmOGEzZjVkNjhkMTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzI3MDk3LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZkM2I0YzQzYzc2MDQ2MDE5ODJmMWJhNTY3ZTlhNWI5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U4ZjZiNzI5NWVlOTRhMDc5MjgxOTU3Njg4YjQ0ZTljID0gJCgnPGRpdiBpZD0iaHRtbF9lOGY2YjcyOTVlZTk0YTA3OTI4MTk1NzY4OGI0NGU5YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEFubmV4LCBOb3J0aCBNaWR0b3duLCBZb3JrdmlsbGUgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZkM2I0YzQzYzc2MDQ2MDE5ODJmMWJhNTY3ZTlhNWI5LnNldENvbnRlbnQoaHRtbF9lOGY2YjcyOTVlZTk0YTA3OTI4MTk1NzY4OGI0NGU5Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85MmJhZDEzYThiMDA0YjBjYjYzOGY4YTNmNWQ2OGQxMi5iaW5kUG9wdXAocG9wdXBfNmQzYjRjNDNjNzYwNDYwMTk4MmYxYmE1NjdlOWE1YjkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDg4ZTExZmM1ODM2NGY2N2FhZWZlMjA2ZWY3OTUxYzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI2OTU2LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTkyYzk4YjExOWM2NGI3MjljNjM1ZmViMWNhMDhmMmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmI2NWNiOWY5ZjZkNDVjOTgzYmYwNGQ3NTFmZTI4YzggPSAkKCc8ZGl2IGlkPSJodG1sX2JiNjVjYjlmOWY2ZDQ1Yzk4M2JmMDRkNzUxZmUyOGM4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIFRvcm9udG8sIEhhcmJvcmQgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk5MmM5OGIxMTljNjRiNzI5YzYzNWZlYjFjYTA4ZjJiLnNldENvbnRlbnQoaHRtbF9iYjY1Y2I5ZjlmNmQ0NWM5ODNiZjA0ZDc1MWZlMjhjOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kODhlMTFmYzU4MzY0ZjY3YWFlZmUyMDZlZjc5NTFjMy5iaW5kUG9wdXAocG9wdXBfOTkyYzk4YjExOWM2NGI3MjljNjM1ZmViMWNhMDhmMmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODc2ZTY1ZDRiMTViNDMzNjlmOTUxMzI2OWVjMzNiY2IgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTMyMDU3LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGM2ZGMyNmVjMTg0NGMyY2E1MmMyN2MyYzJhNzVhZDYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOGJlY2UyYjA2YjMwNGZjOGFmMDRjZThhNjMzZDg0MTEgPSAkKCc8ZGl2IGlkPSJodG1sXzhiZWNlMmIwNmIzMDRmYzhhZjA0Y2U4YTYzM2Q4NDExIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LZW5zaW5ndG9uIE1hcmtldCwgQ2hpbmF0b3duLCBHcmFuZ2UgUGFyayBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGM2ZGMyNmVjMTg0NGMyY2E1MmMyN2MyYzJhNzVhZDYuc2V0Q29udGVudChodG1sXzhiZWNlMmIwNmIzMDRmYzhhZjA0Y2U4YTYzM2Q4NDExKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg3NmU2NWQ0YjE1YjQzMzY5Zjk1MTMyNjllYzMzYmNiLmJpbmRQb3B1cChwb3B1cF80YzZkYzI2ZWMxODQ0YzJjYTUyYzI3YzJjMmE3NWFkNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZmExYzJlM2I4ZTY0OTEyOGY3MzQ1NjQ1YTg4MGM5MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kYTQyYjQ3NTI3Y2E0YmEyODVlNmQ5ODdiNmJmMmMyNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NWYwNDUzZjk0Yjk0ZGNkYWM0OTRjMjI4ZDUzNTRlZiA9ICQoJzxkaXYgaWQ9Imh0bWxfNjVmMDQ1M2Y5NGI5NGRjZGFjNDk0YzIyOGQ1MzU0ZWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNOIFRvd2VyLCBLaW5nIGFuZCBTcGFkaW5hLCBSYWlsd2F5IExhbmRzLCBIYXJib3VyZnJvbnQgV2VzdCwgQmF0aHVyc3QgUXVheSwgU291dGggTmlhZ2FyYSwgSXNsYW5kIGFpcnBvcnQgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RhNDJiNDc1MjdjYTRiYTI4NWU2ZDk4N2I2YmYyYzI1LnNldENvbnRlbnQoaHRtbF82NWYwNDUzZjk0Yjk0ZGNkYWM0OTRjMjI4ZDUzNTRlZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lZmExYzJlM2I4ZTY0OTEyOGY3MzQ1NjQ1YTg4MGM5MC5iaW5kUG9wdXAocG9wdXBfZGE0MmI0NzUyN2NhNGJhMjg1ZTZkOTg3YjZiZjJjMjUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmRmZDY1NTE1ODhkNGYzNzhmMjdiNDk2MDM4ZTRhNTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDY0MzUyLC03OS4zNzQ4NDU5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA3NTE3YTFjNDYzNzQ4Y2FiNzU3ZGYwOWVmMzc2YzQ2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzkzNzAxZjYxMDRhNTRiNWZiM2Q3ODAyMjA4ZGFkNmJmID0gJCgnPGRpdiBpZD0iaHRtbF85MzcwMWY2MTA0YTU0YjVmYjNkNzgwMjIwOGRhZDZiZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3RuIEEgUE8gQm94ZXMgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzA3NTE3YTFjNDYzNzQ4Y2FiNzU3ZGYwOWVmMzc2YzQ2LnNldENvbnRlbnQoaHRtbF85MzcwMWY2MTA0YTU0YjVmYjNkNzgwMjIwOGRhZDZiZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mZGZkNjU1MTU4OGQ0ZjM3OGYyN2I0OTYwMzhlNGE1NS5iaW5kUG9wdXAocG9wdXBfMDc1MTdhMWM0NjM3NDhjYWI3NTdkZjA5ZWYzNzZjNDYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzIxYjgwYTA4ZWMwNGQzZjg5Y2E5ZTk4MGRjMWViMTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg0MjkyLC03OS4zODIyODAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGJlMzYzNGI0NGVjNDJiOGExMjQ1MmZhNWE0NzdkMzcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGE0MDM3NTJmYmRiNDQ4MmFmMjY0ODIxOTA3OTI4MzggPSAkKCc8ZGl2IGlkPSJodG1sXzRhNDAzNzUyZmJkYjQ0ODJhZjI2NDgyMTkwNzkyODM4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GaXJzdCBDYW5hZGlhbiBQbGFjZSwgVW5kZXJncm91bmQgY2l0eSBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGJlMzYzNGI0NGVjNDJiOGExMjQ1MmZhNWE0NzdkMzcuc2V0Q29udGVudChodG1sXzRhNDAzNzUyZmJkYjQ0ODJhZjI2NDgyMTkwNzkyODM4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzcyMWI4MGEwOGVjMDRkM2Y4OWNhOWU5ODBkYzFlYjExLmJpbmRQb3B1cChwb3B1cF9kYmUzNjM0YjQ0ZWM0MmI4YTEyNDUyZmE1YTQ3N2QzNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNjcxZDRlOTgxOGU0Zjc3YWFmNzgyODA5NzFmZWIzNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxODUxNzk5OTk5OTk5NiwtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85MTllZjk3ZDJiMjM0ZjJkYTBiZjJjMWQ3M2VjNGEzMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83ZjcyZDY5NTgyNDA0NDljYWJmNTliZDc3MGUyZTY4NiA9ICQoJzxkaXYgaWQ9Imh0bWxfN2Y3MmQ2OTU4MjQwNDQ5Y2FiZjU5YmQ3NzBlMmU2ODYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxhd3JlbmNlIE1hbm9yLCBMYXdyZW5jZSBIZWlnaHRzIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MTllZjk3ZDJiMjM0ZjJkYTBiZjJjMWQ3M2VjNGEzMi5zZXRDb250ZW50KGh0bWxfN2Y3MmQ2OTU4MjQwNDQ5Y2FiZjU5YmQ3NzBlMmU2ODYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjY3MWQ0ZTk4MThlNGY3N2FhZjc4MjgwOTcxZmViMzUuYmluZFBvcHVwKHBvcHVwXzkxOWVmOTdkMmIyMzRmMmRhMGJmMmMxZDczZWM0YTMyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2RiM2RjOTZlNWYwMjQyODRhZDNiYjVkN2VjM2Y1ZjMyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA5NTc3LC03OS40NDUwNzI1OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcwYzVlMzk4NmQxNjRhYjU5ZWE4NjJlMzI4NmZmMmRhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMzMDA1ZjllNmQ3NjQ5ZjJiMzM1MGM1ZmQxZWU2YzNjID0gJCgnPGRpdiBpZD0iaHRtbF8zMzAwNWY5ZTZkNzY0OWYyYjMzNTBjNWZkMWVlNmMzYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R2xlbmNhaXJuIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83MGM1ZTM5ODZkMTY0YWI1OWVhODYyZTMyODZmZjJkYS5zZXRDb250ZW50KGh0bWxfMzMwMDVmOWU2ZDc2NDlmMmIzMzUwYzVmZDFlZTZjM2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGIzZGM5NmU1ZjAyNDI4NGFkM2JiNWQ3ZWMzZjVmMzIuYmluZFBvcHVwKHBvcHVwXzcwYzVlMzk4NmQxNjRhYjU5ZWE4NjJlMzI4NmZmMmRhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhmYWIzNDc1NmIyNjQ5NjZiMjdkN2E5MjNlYTY1OGIzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkzNzgxMywtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMzJkZjIyZTM5YzA0NDI1ODI3MmNiZmI5ZTc1ZDE5YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81OGE2NjU2YzE1Mzc0OWQyYWZkZTE5ZDQ1MjE0NTRjOCA9ICQoJzxkaXYgaWQ9Imh0bWxfNThhNjY1NmMxNTM3NDlkMmFmZGUxOWQ0NTIxNDU0YzgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWV3b29kLUNlZGFydmFsZSBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTMyZGYyMmUzOWMwNDQyNTgyNzJjYmZiOWU3NWQxOWEuc2V0Q29udGVudChodG1sXzU4YTY2NTZjMTUzNzQ5ZDJhZmRlMTlkNDUyMTQ1NGM4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhmYWIzNDc1NmIyNjQ5NjZiMjdkN2E5MjNlYTY1OGIzLmJpbmRQb3B1cChwb3B1cF9lMzJkZjIyZTM5YzA0NDI1ODI3MmNiZmI5ZTc1ZDE5YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zODMzNWE2MDZhZjc0YzgwYWMxNjZmMDAwZWNiMWU4NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4OTAyNTYsLTc5LjQ1MzUxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RiMTQ4YzFiYzhmNDRhZDFiNzI5OTJkNTk2MjA2OGVlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZjZDg1NGEzNjY2ZjQwNjQ4MGEzNTYwZGU2NGY0NjU2ID0gJCgnPGRpdiBpZD0iaHRtbF9mY2Q4NTRhMzY2NmY0MDY0ODBhMzU2MGRlNjRmNDY1NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2FsZWRvbmlhLUZhaXJiYW5rcyBDbHVzdGVyIDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGIxNDhjMWJjOGY0NGFkMWI3Mjk5MmQ1OTYyMDY4ZWUuc2V0Q29udGVudChodG1sX2ZjZDg1NGEzNjY2ZjQwNjQ4MGEzNTYwZGU2NGY0NjU2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM4MzM1YTYwNmFmNzRjODBhYzE2NmYwMDBlY2IxZTg0LmJpbmRQb3B1cChwb3B1cF9kYjE0OGMxYmM4ZjQ0YWQxYjcyOTkyZDU5NjIwNjhlZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83ZTc5MjRiMDFjN2Q0MTkyYTE3ODllNWI4ZTA3N2YyYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTU0MiwtNzkuNDIyNTYzN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2VkMjM5MmJmZWE0MTQ3OGU4NWZhMDFhYTU5MzNhYzcyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzkwMDdjNzNmNzQ4YjQyYzNhMzJmNTRhODE3YWFmMTk1ID0gJCgnPGRpdiBpZD0iaHRtbF85MDA3YzczZjc0OGI0MmMzYTMyZjU0YTgxN2FhZjE5NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hyaXN0aWUgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VkMjM5MmJmZWE0MTQ3OGU4NWZhMDFhYTU5MzNhYzcyLnNldENvbnRlbnQoaHRtbF85MDA3YzczZjc0OGI0MmMzYTMyZjU0YTgxN2FhZjE5NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83ZTc5MjRiMDFjN2Q0MTkyYTE3ODllNWI4ZTA3N2YyYi5iaW5kUG9wdXAocG9wdXBfZWQyMzkyYmZlYTQxNDc4ZTg1ZmEwMWFhNTkzM2FjNzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWI4ZjQyMDRlYzc4NDJmZWE5ZGIxODI2ZjIzOWI1ZjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjkwMDUxMDAwMDAwMSwtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVjMjQ0MWM5ZTA2MDQyYjBiYzczYzkxM2FmY2YzOWJlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZkMzg2OWU2Y2JlMTQxZDU5MWUxZDE3MzIwYzA3YmIyID0gJCgnPGRpdiBpZD0iaHRtbF82ZDM4NjllNmNiZTE0MWQ1OTFlMWQxNzMyMGMwN2JiMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RHVmZmVyaW4sIERvdmVyY291cnQgVmlsbGFnZSBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNWMyNDQxYzllMDYwNDJiMGJjNzNjOTEzYWZjZjM5YmUuc2V0Q29udGVudChodG1sXzZkMzg2OWU2Y2JlMTQxZDU5MWUxZDE3MzIwYzA3YmIyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFiOGY0MjA0ZWM3ODQyZmVhOWRiMTgyNmYyMzliNWY2LmJpbmRQb3B1cChwb3B1cF81YzI0NDFjOWUwNjA0MmIwYmM3M2M5MTNhZmNmMzliZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82N2NmMzIxZGQ3MjQ0YzhmYTVlN2M3MGQ4OGUxZTE3YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U3OWEwZWYwZjg1MzRjM2FhNmVkMGVhOTM1OTYzODM0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M0NzUyZGNjMmQ0YjRiNGY4NzU3NzI3ZGE0OGIzNDllID0gJCgnPGRpdiBpZD0iaHRtbF9jNDc1MmRjYzJkNGI0YjRmODc1NzcyN2RhNDhiMzQ5ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGl0dGxlIFBvcnR1Z2FsLCBUcmluaXR5IENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lNzlhMGVmMGY4NTM0YzNhYTZlZDBlYTkzNTk2MzgzNC5zZXRDb250ZW50KGh0bWxfYzQ3NTJkY2MyZDRiNGI0Zjg3NTc3MjdkYTQ4YjM0OWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjdjZjMyMWRkNzI0NGM4ZmE1ZTdjNzBkODhlMWUxN2MuYmluZFBvcHVwKHBvcHVwX2U3OWEwZWYwZjg1MzRjM2FhNmVkMGVhOTM1OTYzODM0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA0NGJmZmU4NWRjZjRjNzE5NzBlYWY4ZmY3ZGM0ZWQ4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2ODQ3MiwtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jN2I3ZjgyZGY1NGU0YWJkODhkMjEzMzJhYzg5NmJmNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lMzRjODdjMmVhNjY0MjhiYjBkMzY5MmYxNjUxYzZiNiA9ICQoJzxkaXYgaWQ9Imh0bWxfZTM0Yzg3YzJlYTY2NDI4YmIwZDM2OTJmMTY1MWM2YjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyb2NrdG9uLCBQYXJrZGFsZSBWaWxsYWdlLCBFeGhpYml0aW9uIFBsYWNlIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jN2I3ZjgyZGY1NGU0YWJkODhkMjEzMzJhYzg5NmJmNy5zZXRDb250ZW50KGh0bWxfZTM0Yzg3YzJlYTY2NDI4YmIwZDM2OTJmMTY1MWM2YjYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDQ0YmZmZTg1ZGNmNGM3MTk3MGVhZjhmZjdkYzRlZDguYmluZFBvcHVwKHBvcHVwX2M3YjdmODJkZjU0ZTRhYmQ4OGQyMTMzMmFjODk2YmY3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE5N2NkY2E1OWY3MjQwOGNhZGQ5Y2IwNjc0YzVkNTY0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEzNzU2MjAwMDAwMDA2LC03OS40OTAwNzM4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTVlZjFlZjdjODk5NGVmNGI2ZGJhMzMzNjYzYzExYmEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWM1M2Y4OTlkNjEwNGM1Mzk5MjM3Zjk0MjdiNzZiOGQgPSAkKCc8ZGl2IGlkPSJodG1sX2FjNTNmODk5ZDYxMDRjNTM5OTIzN2Y5NDI3Yjc2YjhkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBQYXJrLCBNYXBsZSBMZWFmIFBhcmssIFVwd29vZCBQYXJrIENsdXN0ZXIgMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NWVmMWVmN2M4OTk0ZWY0YjZkYmEzMzM2NjNjMTFiYS5zZXRDb250ZW50KGh0bWxfYWM1M2Y4OTlkNjEwNGM1Mzk5MjM3Zjk0MjdiNzZiOGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTk3Y2RjYTU5ZjcyNDA4Y2FkZDljYjA2NzRjNWQ1NjQuYmluZFBvcHVwKHBvcHVwXzU1ZWYxZWY3Yzg5OTRlZjRiNmRiYTMzMzY2M2MxMWJhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBmMmVjNDIzMGI5NzRiOTRiOWRjZjc1MzllNDNjYWM1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkxMTE1OCwtNzkuNDc2MDEzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYjdjYzUzYzhlMzM0OTAyOGYwYTk4NzY1YjViYTU1MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jNjc3NjRjOTJkZWE0NjE5YjRmNmYzY2QzN2I2NGU2OCA9ICQoJzxkaXYgaWQ9Imh0bWxfYzY3NzY0YzkyZGVhNDYxOWI0ZjZmM2NkMzdiNjRlNjgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRlbCBSYXksIE1vdW50IERlbm5pcywgS2VlbHNkYWxlIGFuZCBTaWx2ZXJ0aG9ybiBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWI3Y2M1M2M4ZTMzNDkwMjhmMGE5ODc2NWI1YmE1NTEuc2V0Q29udGVudChodG1sX2M2Nzc2NGM5MmRlYTQ2MTliNGY2ZjNjZDM3YjY0ZTY4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBmMmVjNDIzMGI5NzRiOTRiOWRjZjc1MzllNDNjYWM1LmJpbmRQb3B1cChwb3B1cF9hYjdjYzUzYzhlMzM0OTAyOGYwYTk4NzY1YjViYTU1MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80MWEzYmI1NzA0OWI0NWFjODBkN2U0NTEyYjRlYzkxYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MzE4NTI5OTk5OTk5LC03OS40ODcyNjE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJhZDJlYTI0MmJlNjRiZTNhNzU2ODBhZjY2ZjdmNjNjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E0MTI1OWExZWY1ZTQ2ODBhZGE5NTk3ODVlZWI1ZmEyID0gJCgnPGRpdiBpZD0iaHRtbF9hNDEyNTlhMWVmNWU0NjgwYWRhOTU5Nzg1ZWViNWZhMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBUaGUgSnVuY3Rpb24gTm9ydGggQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJhZDJlYTI0MmJlNjRiZTNhNzU2ODBhZjY2ZjdmNjNjLnNldENvbnRlbnQoaHRtbF9hNDEyNTlhMWVmNWU0NjgwYWRhOTU5Nzg1ZWViNWZhMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MWEzYmI1NzA0OWI0NWFjODBkN2U0NTEyYjRlYzkxYS5iaW5kUG9wdXAocG9wdXBfMmFkMmVhMjQyYmU2NGJlM2E3NTY4MGFmNjZmN2Y2M2MpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTlkZTIxNzY5ZTQ2NDRmOGE0NmY3ODcyZmIxZjUzMzQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjE2MDgzLC03OS40NjQ3NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I1ZTMzNWNhYTM5YjQ3NjQ5YjljMjNlNjgwNzZmMDNiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFmNGQ5NmE0MDcwMjRjYTJhNTE5NzZlNzFjYmM4YzgxID0gJCgnPGRpdiBpZD0iaHRtbF8xZjRkOTZhNDA3MDI0Y2EyYTUxOTc2ZTcxY2JjOGM4MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGlnaCBQYXJrLCBUaGUgSnVuY3Rpb24gU291dGggQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I1ZTMzNWNhYTM5YjQ3NjQ5YjljMjNlNjgwNzZmMDNiLnNldENvbnRlbnQoaHRtbF8xZjRkOTZhNDA3MDI0Y2EyYTUxOTc2ZTcxY2JjOGM4MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lOWRlMjE3NjllNDY0NGY4YTQ2Zjc4NzJmYjFmNTMzNC5iaW5kUG9wdXAocG9wdXBfYjVlMzM1Y2FhMzliNDc2NDliOWMyM2U2ODA3NmYwM2IpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2U4YmE4Yzg4MWQzNDllODgwM2NjODRiMDE3ZDlkZmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg5NTk3LC03OS40NTYzMjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMjVhNDZiYjJmZjU0ZjczYTY1YzZhYTY1ZWQ3ZTNiNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hODc5OWU1MDA3ZGE0MDljYTI1OGQyY2E5ZTliNjcwMSA9ICQoJzxkaXYgaWQ9Imh0bWxfYTg3OTllNTAwN2RhNDA5Y2EyNThkMmNhOWU5YjY3MDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhcmtkYWxlLCBSb25jZXN2YWxsZXMgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzEyNWE0NmJiMmZmNTRmNzNhNjVjNmFhNjVlZDdlM2I2LnNldENvbnRlbnQoaHRtbF9hODc5OWU1MDA3ZGE0MDljYTI1OGQyY2E5ZTliNjcwMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83ZThiYThjODgxZDM0OWU4ODAzY2M4NGIwMTdkOWRmZC5iaW5kUG9wdXAocG9wdXBfMTI1YTQ2YmIyZmY1NGY3M2E2NWM2YWE2NWVkN2UzYjYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGJjNmU1ZmJiY2I2NGVhNjg2YjU3NGE4OGQ4YjViNGQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE1NzA2LC03OS40ODQ0NDk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGQ0ZTEyNGFmOTM5NDk5NWJiYzM2YzgyZWFiOGI4ZWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2U1ZjFhNjU2NDM0NDEwYWE2YzI5Yzk3ZGFlMjBhZWIgPSAkKCc8ZGl2IGlkPSJodG1sXzNlNWYxYTY1NjQzNDQxMGFhNmMyOWM5N2RhZTIwYWViIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SdW5ueW1lZGUsIFN3YW5zZWEgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhkNGUxMjRhZjkzOTQ5OTViYmMzNmM4MmVhYjhiOGVlLnNldENvbnRlbnQoaHRtbF8zZTVmMWE2NTY0MzQ0MTBhYTZjMjljOTdkYWUyMGFlYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84YmM2ZTVmYmJjYjY0ZWE2ODZiNTc0YTg4ZDhiNWI0ZC5iaW5kUG9wdXAocG9wdXBfOGQ0ZTEyNGFmOTM5NDk5NWJiYzM2YzgyZWFiOGI4ZWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2JjYWJmY2U2ODRhNDVmMGEzOTEyNzViM2QzYzMyMzUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjIzMDE1LC03OS4zODk0OTM4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTIyNGI3ZWNlYTNmNDJlZThiYjcyZjM4ZDZlZGI1Y2MgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTU3ZTc5NTUxNTgxNGY3Zjg5NTdjODc3MmIyNTU1MmEgPSAkKCc8ZGl2IGlkPSJodG1sXzk1N2U3OTU1MTU4MTRmN2Y4OTU3Yzg3NzJiMjU1NTJhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5RdWVlbiYjMzk7cyBQYXJrLCBPbnRhcmlvIFByb3ZpbmNpYWwgR292ZXJubWVudCBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTIyNGI3ZWNlYTNmNDJlZThiYjcyZjM4ZDZlZGI1Y2Muc2V0Q29udGVudChodG1sXzk1N2U3OTU1MTU4MTRmN2Y4OTU3Yzg3NzJiMjU1NTJhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdiY2FiZmNlNjg0YTQ1ZjBhMzkxMjc1YjNkM2MzMjM1LmJpbmRQb3B1cChwb3B1cF85MjI0YjdlY2VhM2Y0MmVlOGJiNzJmMzhkNmVkYjVjYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mZGZhMTM4YmQyZTI0N2UxYjI5NzU0NTJhZjI5YjIxMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjk2NTYsLTc5LjYxNTgxODk5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzM0MmZjMDEwYWFjNGI1NTg3ZTlkZDg4NWM3MjEzOTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjFkZmU4ZjAwYWYyNDM1NThjYjk5YzMyY2E4MWQyNzIgPSAkKCc8ZGl2IGlkPSJodG1sXzYxZGZlOGYwMGFmMjQzNTU4Y2I5OWMzMmNhODFkMjcyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYW5hZGEgUG9zdCBHYXRld2F5IFByb2Nlc3NpbmcgQ2VudHJlIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jMzQyZmMwMTBhYWM0YjU1ODdlOWRkODg1YzcyMTM5NS5zZXRDb250ZW50KGh0bWxfNjFkZmU4ZjAwYWYyNDM1NThjYjk5YzMyY2E4MWQyNzIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmRmYTEzOGJkMmUyNDdlMWIyOTc1NDUyYWYyOWIyMTMuYmluZFBvcHVwKHBvcHVwX2MzNDJmYzAxMGFhYzRiNTU4N2U5ZGQ4ODVjNzIxMzk1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFjMDdkZmUyZWRkZDRlOTBhZjAxZTRjMmMwODU1MWY1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNzQzOSwtNzkuMzIxNTU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzIxYmQxMzkxNDM0NGI3MTg0YzNlNGVmYzgyYjIxNWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDZjNTgwMmU4ZTY2NGIzYWE1NjYwMzE0ZWIwOTNjNzUgPSAkKCc8ZGl2IGlkPSJodG1sX2Q2YzU4MDJlOGU2NjRiM2FhNTY2MDMxNGViMDkzYzc1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CdXNpbmVzcyByZXBseSBtYWlsIFByb2Nlc3NpbmcgQ2VudHJlLCBTb3V0aCBDZW50cmFsIExldHRlciBQcm9jZXNzaW5nIFBsYW50IFRvcm9udG8gQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzcyMWJkMTM5MTQzNDRiNzE4NGMzZTRlZmM4MmIyMTVjLnNldENvbnRlbnQoaHRtbF9kNmM1ODAyZThlNjY0YjNhYTU2NjAzMTRlYjA5M2M3NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xYzA3ZGZlMmVkZGQ0ZTkwYWYwMWU0YzJjMDg1NTFmNS5iaW5kUG9wdXAocG9wdXBfNzIxYmQxMzkxNDM0NGI3MTg0YzNlNGVmYzgyYjIxNWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjZjMTM0NzMzZmM0NGRlMWExMGNmODdlYWM2MTRhOTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MDU2NDY2LC03OS41MDEzMjA3MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMzMTU5NDhmN2U4MTRhODZhYzJmNGI4Y2I5MjA0YzY4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzEyYzc3M2UyYzEyMzQ4OWFhZGNkZWQ1MDMxOGQ3NTIxID0gJCgnPGRpdiBpZD0iaHRtbF8xMmM3NzNlMmMxMjM0ODlhYWRjZGVkNTAzMThkNzUyMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TmV3IFRvcm9udG8sIE1pbWljbyBTb3V0aCwgSHVtYmVyIEJheSBTaG9yZXMgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMzMTU5NDhmN2U4MTRhODZhYzJmNGI4Y2I5MjA0YzY4LnNldENvbnRlbnQoaHRtbF8xMmM3NzNlMmMxMjM0ODlhYWRjZGVkNTAzMThkNzUyMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mNmMxMzQ3MzNmYzQ0ZGUxYTEwY2Y4N2VhYzYxNGE5NS5iaW5kUG9wdXAocG9wdXBfMzMxNTk0OGY3ZTgxNGE4NmFjMmY0YjhjYjkyMDRjNjgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTMyYjhkMDBlMGE3NDZiMzk4ZGJiYTYzYzliZTNlZDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MDI0MTM3MDAwMDAwMSwtNzkuNTQzNDg0MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81NTViMjExZGViNjE0YjQzODM1NTgxNjhhODY2OTE2ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kMzRjNzAxYzMzNDU0MjZhOGE1NWE2NGU3NGFjZmZhMSA9ICQoJzxkaXYgaWQ9Imh0bWxfZDM0YzcwMWMzMzQ1NDI2YThhNTVhNjRlNzRhY2ZmYTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFsZGVyd29vZCwgTG9uZyBCcmFuY2ggQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU1NWIyMTFkZWI2MTRiNDM4MzU1ODE2OGE4NjY5MTZkLnNldENvbnRlbnQoaHRtbF9kMzRjNzAxYzMzNDU0MjZhOGE1NWE2NGU3NGFjZmZhMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMzJiOGQwMGUwYTc0NmIzOThkYmJhNjNjOWJlM2VkMC5iaW5kUG9wdXAocG9wdXBfNTU1YjIxMWRlYjYxNGI0MzgzNTU4MTY4YTg2NjkxNmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjRlOGM1OWE4Y2ZkNDg3MWIyZTczMzI3YTg3NTQzMzggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTM2NTM2MDAwMDAwMDUsLTc5LjUwNjk0MzZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85OTk1N2NlMjFmMDI0OWM4OTY0NzliMGU1YjMyOTliMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yNjE0YTA4OTIwZDI0MTk0OTE0N2JkZjg2ZWUxOTgxZSA9ICQoJzxkaXYgaWQ9Imh0bWxfMjYxNGEwODkyMGQyNDE5NDkxNDdiZGY4NmVlMTk4MWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBLaW5nc3dheSwgTW9udGdvbWVyeSBSb2FkLCBPbGQgTWlsbCBOb3J0aCBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTk5NTdjZTIxZjAyNDljODk2NDc5YjBlNWIzMjk5YjAuc2V0Q29udGVudChodG1sXzI2MTRhMDg5MjBkMjQxOTQ5MTQ3YmRmODZlZTE5ODFlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I0ZThjNTlhOGNmZDQ4NzFiMmU3MzMyN2E4NzU0MzM4LmJpbmRQb3B1cChwb3B1cF85OTk1N2NlMjFmMDI0OWM4OTY0NzliMGU1YjMyOTliMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84NDU3NTA3ZTI1ZTE0YmU4OGJiNTQyMjg4ZTBmMWIzOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjI1NzksLTc5LjQ5ODUwOTA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDM5NTcyNTA5YTRmNGE3ZGI5MzE1OWQ1ZDk3ZjY0OTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTczNDZmZmU3MTUxNGNjZGFlZmU3NWJhN2U1Njc3MDIgPSAkKCc8ZGl2IGlkPSJodG1sXzk3MzQ2ZmZlNzE1MTRjY2RhZWZlNzViYTdlNTY3NzAyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5PbGQgTWlsbCBTb3V0aCwgS2luZyYjMzk7cyBNaWxsIFBhcmssIFN1bm55bGVhLCBIdW1iZXIgQmF5LCBNaW1pY28gTkUsIFRoZSBRdWVlbnN3YXkgRWFzdCwgUm95YWwgWW9yayBTb3V0aCBFYXN0LCBLaW5nc3dheSBQYXJrIFNvdXRoIEVhc3QgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQzOTU3MjUwOWE0ZjRhN2RiOTMxNTlkNWQ5N2Y2NDk2LnNldENvbnRlbnQoaHRtbF85NzM0NmZmZTcxNTE0Y2NkYWVmZTc1YmE3ZTU2NzcwMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NDU3NTA3ZTI1ZTE0YmU4OGJiNTQyMjg4ZTBmMWIzOS5iaW5kUG9wdXAocG9wdXBfNDM5NTcyNTA5YTRmNGE3ZGI5MzE1OWQ1ZDk3ZjY0OTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDZhMDlmMGI0MDVjNGYwZmJkNGJhYzlmMjZlN2U2ODcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg4NDA4LC03OS41MjA5OTk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ3OTg4YTgxMDAxYTQ0NjM5ZDFhNzZiZTQxOWViNDRkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFiODZjYWFkNjM3NTQ5ZjRiYjRlOWJlOTU1OTg5OTY5ID0gJCgnPGRpdiBpZD0iaHRtbF8xYjg2Y2FhZDYzNzU0OWY0YmI0ZTliZTk1NTk4OTk2OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWltaWNvIE5XLCBUaGUgUXVlZW5zd2F5IFdlc3QsIFNvdXRoIG9mIEJsb29yLCBLaW5nc3dheSBQYXJrIFNvdXRoIFdlc3QsIFJveWFsIFlvcmsgU291dGggV2VzdCBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDc5ODhhODEwMDFhNDQ2MzlkMWE3NmJlNDE5ZWI0NGQuc2V0Q29udGVudChodG1sXzFiODZjYWFkNjM3NTQ5ZjRiYjRlOWJlOTU1OTg5OTY5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q2YTA5ZjBiNDA1YzRmMGZiZDRiYWM5ZjI2ZTdlNjg3LmJpbmRQb3B1cChwb3B1cF80Nzk4OGE4MTAwMWE0NDYzOWQxYTc2YmU0MTllYjQ0ZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MjFkNDA5ZTYzMjI0MDNiYWQzNDVkMmFiMTc4Y2NhOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzg1NTYsLTc5LjUzMjI0MjQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzY2MjIxZjQwMDE4NDY2MGJiZWRiMjc1M2FlYWJhMTIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWJiMTY0NDIzNjMzNGZlM2IyMTFmMDQ4NmZlOTExNTUgPSAkKCc8ZGl2IGlkPSJodG1sXzViYjE2NDQyMzYzMzRmZTNiMjExZjA0ODZmZTkxMTU1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Jc2xpbmd0b24gQXZlbnVlLCBIdW1iZXIgVmFsbGV5IFZpbGxhZ2UgQ2x1c3RlciBuYW48L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc2NjIyMWY0MDAxODQ2NjBiYmVkYjI3NTNhZWFiYTEyLnNldENvbnRlbnQoaHRtbF81YmIxNjQ0MjM2MzM0ZmUzYjIxMWYwNDg2ZmU5MTE1NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81MjFkNDA5ZTYzMjI0MDNiYWQzNDVkMmFiMTc4Y2NhOC5iaW5kUG9wdXAocG9wdXBfNzY2MjIxZjQwMDE4NDY2MGJiZWRiMjc1M2FlYWJhMTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjMzNzVhMDE0MjYwNDg4OGJhYjhkYzI5YTBmNGE1NzkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA5NDMyLC03OS41NTQ3MjQ0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZjM2JlOTkxZjEzNjQ1OGRhYmI3OGY1OGI5YTlhMGVjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBlNGJjN2FiOGE1YzRlNzc4NjMxYTQ0YWFlMDg3MWM2ID0gJCgnPGRpdiBpZD0iaHRtbF8wZTRiYzdhYjhhNWM0ZTc3ODYzMWE0NGFhZTA4NzFjNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdCBEZWFuZSBQYXJrLCBQcmluY2VzcyBHYXJkZW5zLCBNYXJ0aW4gR3JvdmUsIElzbGluZ3RvbiwgQ2xvdmVyZGFsZSBDbHVzdGVyIDMuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmMzYmU5OTFmMTM2NDU4ZGFiYjc4ZjU4YjlhOWEwZWMuc2V0Q29udGVudChodG1sXzBlNGJjN2FiOGE1YzRlNzc4NjMxYTQ0YWFlMDg3MWM2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYzMzc1YTAxNDI2MDQ4ODhiYWI4ZGMyOWEwZjRhNTc5LmJpbmRQb3B1cChwb3B1cF9mYzNiZTk5MWYxMzY0NThkYWJiNzhmNThiOWE5YTBlYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNzhjMmFmMWU5YmY0NWNhOTk0YWI0NDUyMjE3NTdiNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MzUxNTIsLTc5LjU3NzIwMDc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjE1NzBjMTE0YWE3NDM3MmEyNjM2MzE3MzBiYzZiZWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzRkMmRmOGFlZWRhNDdjMDkzYTk5MTU5MzE1OTc3MTcgPSAkKCc8ZGl2IGlkPSJodG1sX2M0ZDJkZjhhZWVkYTQ3YzA5M2E5OTE1OTMxNTk3NzE3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FcmluZ2F0ZSwgQmxvb3JkYWxlIEdhcmRlbnMsIE9sZCBCdXJuaGFtdGhvcnBlLCBNYXJrbGFuZCBXb29kIENsdXN0ZXIgMS4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yMTU3MGMxMTRhYTc0MzcyYTI2MzYzMTczMGJjNmJlYy5zZXRDb250ZW50KGh0bWxfYzRkMmRmOGFlZWRhNDdjMDkzYTk5MTU5MzE1OTc3MTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzc4YzJhZjFlOWJmNDVjYTk5NGFiNDQ1MjIxNzU3YjQuYmluZFBvcHVwKHBvcHVwXzIxNTcwYzExNGFhNzQzNzJhMjYzNjMxNzMwYmM2YmVjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY1YTdhOTdhMDg5NDQyMzI4NjZhZjhlNmI1M2MwNjA4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU2MzAzMywtNzkuNTY1OTYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kODhlOTg1OTllMDI0NjNlYWVmMTMzOTVkZGZlYjcwMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jOWQwOGZjYWJkNzU0YzdhOGNkY2ZmYzhhYjRiNjkzMyA9ICQoJzxkaXYgaWQ9Imh0bWxfYzlkMDhmY2FiZDc1NGM3YThjZGNmZmM4YWI0YjY5MzMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWJlciBTdW1taXQgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q4OGU5ODU5OWUwMjQ2M2VhZWYxMzM5NWRkZmViNzAxLnNldENvbnRlbnQoaHRtbF9jOWQwOGZjYWJkNzU0YzdhOGNkY2ZmYzhhYjRiNjkzMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82NWE3YTk3YTA4OTQ0MjMyODY2YWY4ZTZiNTNjMDYwOC5iaW5kUG9wdXAocG9wdXBfZDg4ZTk4NTk5ZTAyNDYzZWFlZjEzMzk1ZGRmZWI3MDEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDI3OTkxZjc1ODJiNGUwYThiNGJlMTMyMDZiNTQzNWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MjQ3NjU5LC03OS41MzIyNDI0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZjMmIzMDcxMzMxNjQyNGZhYjM4MTE0ZjFhNDI0OGM2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2MzNTliYmQ4ZjFhZjQ3MTNiYjIxNzYyNTBkNTgyYTkyID0gJCgnPGRpdiBpZD0iaHRtbF9jMzU5YmJkOGYxYWY0NzEzYmIyMTc2MjUwZDU4MmE5MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SHVtYmVybGVhLCBFbWVyeSBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmMyYjMwNzEzMzE2NDI0ZmFiMzgxMTRmMWE0MjQ4YzYuc2V0Q29udGVudChodG1sX2MzNTliYmQ4ZjFhZjQ3MTNiYjIxNzYyNTBkNTgyYTkyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQyNzk5MWY3NTgyYjRlMGE4YjRiZTEzMjA2YjU0MzViLmJpbmRQb3B1cChwb3B1cF82YzJiMzA3MTMzMTY0MjRmYWIzODExNGYxYTQyNDhjNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82N2M3Yjc3YTAyOWY0MzBmYjAzMmFiY2YxMmM2ZmExMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNjg3NiwtNzkuNTE4MTg4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNmE2YzM4ZWYxMGY0MGE1ODUwNTg2YWNhMjA0NjYxNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80YmU2ODk3NzA1MmM0NjU5ODc5ZmY0Yzk4MDBhZTllYiA9ICQoJzxkaXYgaWQ9Imh0bWxfNGJlNjg5NzcwNTJjNDY1OTg3OWZmNGM5ODAwYWU5ZWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3RvbiBDbHVzdGVyIDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzZhNmMzOGVmMTBmNDBhNTg1MDU4NmFjYTIwNDY2MTcuc2V0Q29udGVudChodG1sXzRiZTY4OTc3MDUyYzQ2NTk4NzlmZjRjOTgwMGFlOWViKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY3YzdiNzdhMDI5ZjQzMGZiMDMyYWJjZjEyYzZmYTEzLmJpbmRQb3B1cChwb3B1cF9jNmE2YzM4ZWYxMGY0MGE1ODUwNTg2YWNhMjA0NjYxNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82NDgyMTY1ODZjYWE0YWY1OTQyZDFlZDAzN2FiY2NjZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5NjMxOSwtNzkuNTMyMjQyNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zZjhiZjljYTY5ODk0MDk2YjQwZmY3M2Y2MWMwYWFlZSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hN2UwODQzY2IwMGM0ZjdjODBkY2Q5ZTJmYWIzOGJlNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yYWRjMDk5NmFhMmU0YTRmODU3ODZlMWE1YmE2NmYxMCA9ICQoJzxkaXYgaWQ9Imh0bWxfMmFkYzA5OTZhYTJlNGE0Zjg1Nzg2ZTFhNWJhNjZmMTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3Rtb3VudCBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTdlMDg0M2NiMDBjNGY3YzgwZGNkOWUyZmFiMzhiZTUuc2V0Q29udGVudChodG1sXzJhZGMwOTk2YWEyZTRhNGY4NTc4NmUxYTViYTY2ZjEwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY0ODIxNjU4NmNhYTRhZjU5NDJkMWVkMDM3YWJjY2NmLmJpbmRQb3B1cChwb3B1cF9hN2UwODQzY2IwMGM0ZjdjODBkY2Q5ZTJmYWIzOGJlNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iN2MyMWMxNjFhM2E0MTQyOTUyYjZhNjRhMGE2MzQwYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4ODkwNTQsLTc5LjU1NDcyNDQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiBbCiAgICAiIzgwMDBmZiIsCiAgICAiIzAwYjVlYiIsCiAgICAiIzgwZmZiNCIsCiAgICAiI2ZmYjM2MCIsCiAgICAiI2ZmMDAwMCIKICBdLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfM2Y4YmY5Y2E2OTg5NDA5NmI0MGZmNzNmNjFjMGFhZWUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDA3MjlmYTAxZDdjNDQzZTg5NTI3MGRhYTVhZmYyZDQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTEwYTQ4NThmMjYyNDQzYjllMTViMmJhNjIwMmViZWUgPSAkKCc8ZGl2IGlkPSJodG1sX2UxMGE0ODU4ZjI2MjQ0M2I5ZTE1YjJiYTYyMDJlYmVlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nc3ZpZXcgVmlsbGFnZSwgU3QuIFBoaWxsaXBzLCBNYXJ0aW4gR3JvdmUgR2FyZGVucywgUmljaHZpZXcgR2FyZGVucyBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDA3MjlmYTAxZDdjNDQzZTg5NTI3MGRhYTVhZmYyZDQuc2V0Q29udGVudChodG1sX2UxMGE0ODU4ZjI2MjQ0M2I5ZTE1YjJiYTYyMDJlYmVlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I3YzIxYzE2MWEzYTQxNDI5NTJiNmE2NGEwYTYzNDBhLmJpbmRQb3B1cChwb3B1cF9kMDcyOWZhMDFkN2M0NDNlODk1MjcwZGFhNWFmZjJkNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80MDdlNzE2ZWRjNTY0Mzk2YmVlYjAxMGNkZjE4NThiMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTQxNjM5OTk5OTk5NiwtNzkuNTg4NDM2OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzExNTZiNzNkZTZlNzQ5NWY5M2M2NTQxOTEwMzQ3YTU3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI4YTRiY2U3ZTgxNjQzNTNhZjZmMjJlZjY1OGFkODcwID0gJCgnPGRpdiBpZD0iaHRtbF8yOGE0YmNlN2U4MTY0MzUzYWY2ZjIyZWY2NThhZDg3MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U291dGggU3RlZWxlcywgU2lsdmVyc3RvbmUsIEh1bWJlcmdhdGUsIEphbWVzdG93biwgTW91bnQgT2xpdmUsIEJlYXVtb25kIEhlaWdodHMsIFRoaXN0bGV0b3duLCBBbGJpb24gR2FyZGVucyBDbHVzdGVyIDEuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTE1NmI3M2RlNmU3NDk1ZjkzYzY1NDE5MTAzNDdhNTcuc2V0Q29udGVudChodG1sXzI4YTRiY2U3ZTgxNjQzNTNhZjZmMjJlZjY1OGFkODcwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQwN2U3MTZlZGM1NjQzOTZiZWViMDEwY2RmMTg1OGIwLmJpbmRQb3B1cChwb3B1cF8xMTU2YjczZGU2ZTc0OTVmOTNjNjU0MTkxMDM0N2E1Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MzRiZGE1NmM4NzE0NDI3YjNlZTlmMzRkZGE0YjNmZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNjc0ODI5OTk5OTk5NCwtNzkuNTk0MDU0NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6IFsKICAgICIjODAwMGZmIiwKICAgICIjMDBiNWViIiwKICAgICIjODBmZmI0IiwKICAgICIjZmZiMzYwIiwKICAgICIjZmYwMDAwIgogIF0sCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogWwogICAgIiM4MDAwZmYiLAogICAgIiMwMGI1ZWIiLAogICAgIiM4MGZmYjQiLAogICAgIiNmZmIzNjAiLAogICAgIiNmZjAwMDAiCiAgXSwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzNmOGJmOWNhNjk4OTQwOTZiNDBmZjczZjYxYzBhYWVlKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM0NWIwZmEwMGY0MzQ0MGVhNGY5ZDVlMzA1NGY3ZjM2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA1MGMxYzhhZjcxZTQwODFiZjVlYzM4ZDAyMDM5NGZkID0gJCgnPGRpdiBpZD0iaHRtbF8wNTBjMWM4YWY3MWU0MDgxYmY1ZWMzOGQwMjAzOTRmZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGh3ZXN0LCBXZXN0IEh1bWJlciAtIENsYWlydmlsbGUgQ2x1c3RlciAxLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM0NWIwZmEwMGY0MzQ0MGVhNGY5ZDVlMzA1NGY3ZjM2LnNldENvbnRlbnQoaHRtbF8wNTBjMWM4YWY3MWU0MDgxYmY1ZWMzOGQwMjAzOTRmZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MzRiZGE1NmM4NzE0NDI3YjNlZTlmMzRkZGE0YjNmZS5iaW5kUG9wdXAocG9wdXBfMzQ1YjBmYTAwZjQzNDQwZWE0ZjlkNWUzMDU0ZjdmMzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCjwvc2NyaXB0Pg== onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### 3g. Cluster Analysis

##### Cluster 1


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>Scarborough</td>
      <td>0.0</td>
      <td>Park</td>
      <td>Playground</td>
      <td>Bakery</td>
      <td>Intersection</td>
      <td>Escape Room</td>
      <td>Electronics Store</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
      <td>Drugstore</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>23</th>
      <td>North York</td>
      <td>0.0</td>
      <td>Park</td>
      <td>Convenience Store</td>
      <td>Women's Store</td>
      <td>Drugstore</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>25</th>
      <td>North York</td>
      <td>0.0</td>
      <td>Food &amp; Drink Shop</td>
      <td>Park</td>
      <td>Drugstore</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Women's Store</td>
    </tr>
    <tr>
      <th>40</th>
      <td>East York</td>
      <td>0.0</td>
      <td>Intersection</td>
      <td>Park</td>
      <td>Convenience Store</td>
      <td>Drugstore</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Central Toronto</td>
      <td>0.0</td>
      <td>Park</td>
      <td>Swim School</td>
      <td>Bus Line</td>
      <td>Donut Shop</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Drugstore</td>
      <td>Dim Sum Restaurant</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Downtown Toronto</td>
      <td>0.0</td>
      <td>Park</td>
      <td>Playground</td>
      <td>Trail</td>
      <td>Escape Room</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
      <td>Drugstore</td>
      <td>Department Store</td>
    </tr>
    <tr>
      <th>74</th>
      <td>York</td>
      <td>0.0</td>
      <td>Park</td>
      <td>Women's Store</td>
      <td>Pool</td>
      <td>Donut Shop</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
    </tr>
    <tr>
      <th>79</th>
      <td>North York</td>
      <td>0.0</td>
      <td>Park</td>
      <td>Construction &amp; Landscaping</td>
      <td>Bakery</td>
      <td>Women's Store</td>
      <td>Drugstore</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>98</th>
      <td>York</td>
      <td>0.0</td>
      <td>Park</td>
      <td>Women's Store</td>
      <td>Drugstore</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Dumpling Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



##### Cluster 2


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Construction &amp; Landscaping</td>
      <td>Bar</td>
      <td>Women's Store</td>
      <td>Dumpling Restaurant</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Drugstore</td>
      <td>Eastern European Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Restaurant</td>
      <td>Medical Center</td>
      <td>Intersection</td>
      <td>Mexican Restaurant</td>
      <td>Bank</td>
      <td>Rental Car Location</td>
      <td>Breakfast Spot</td>
      <td>Electronics Store</td>
      <td>Drugstore</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Mexican Restaurant</td>
      <td>Korean BBQ Restaurant</td>
      <td>Women's Store</td>
      <td>Drugstore</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Hakka Restaurant</td>
      <td>Athletics &amp; Sports</td>
      <td>Bakery</td>
      <td>Gas Station</td>
      <td>Caribbean Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Bank</td>
      <td>Fried Chicken Joint</td>
      <td>Dog Run</td>
      <td>Distribution Center</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Smoke Shop</td>
      <td>Playground</td>
      <td>Jewelry Store</td>
      <td>Women's Store</td>
      <td>Donut Shop</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Department Store</td>
      <td>Coffee Shop</td>
      <td>Hobby Shop</td>
      <td>Train Station</td>
      <td>Drugstore</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Bus Line</td>
      <td>Bakery</td>
      <td>Soccer Field</td>
      <td>Ice Cream Shop</td>
      <td>Metro Station</td>
      <td>Bus Station</td>
      <td>Intersection</td>
      <td>Park</td>
      <td>Drugstore</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Motel</td>
      <td>American Restaurant</td>
      <td>Women's Store</td>
      <td>Dessert Shop</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>College Stadium</td>
      <td>Skating Rink</td>
      <td>General Entertainment</td>
      <td>Café</td>
      <td>Doner Restaurant</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Indian Restaurant</td>
      <td>Pet Store</td>
      <td>Vietnamese Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Doner Restaurant</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Auto Garage</td>
      <td>Middle Eastern Restaurant</td>
      <td>Sandwich Place</td>
      <td>Bakery</td>
      <td>Smoke Shop</td>
      <td>Accessories Store</td>
      <td>Concert Hall</td>
      <td>Construction &amp; Landscaping</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Lounge</td>
      <td>Skating Rink</td>
      <td>Latin American Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Clothing Store</td>
      <td>Drugstore</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Pizza Place</td>
      <td>Intersection</td>
      <td>Fried Chicken Joint</td>
      <td>Bank</td>
      <td>Pharmacy</td>
      <td>Fast Food Restaurant</td>
      <td>Noodle House</td>
      <td>Italian Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Gas Station</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Scarborough</td>
      <td>1.0</td>
      <td>Fast Food Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Pizza Place</td>
      <td>Indian Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Sandwich Place</td>
      <td>Coffee Shop</td>
      <td>Bank</td>
      <td>Pharmacy</td>
      <td>Grocery Store</td>
    </tr>
    <tr>
      <th>17</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Athletics &amp; Sports</td>
      <td>Mediterranean Restaurant</td>
      <td>Dog Run</td>
      <td>Golf Course</td>
      <td>Pool</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>18</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Juice Bar</td>
      <td>Women's Store</td>
      <td>Shoe Store</td>
      <td>Cosmetics Shop</td>
      <td>Bank</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>19</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Café</td>
      <td>Japanese Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Bank</td>
      <td>Women's Store</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>22</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Ramen Restaurant</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Sushi Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Pizza Place</td>
      <td>Sandwich Place</td>
      <td>Plaza</td>
      <td>Shopping Mall</td>
      <td>Movie Theater</td>
    </tr>
    <tr>
      <th>24</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Pharmacy</td>
      <td>Butcher</td>
      <td>Pizza Place</td>
      <td>Grocery Store</td>
      <td>Coffee Shop</td>
      <td>Doner Restaurant</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
    </tr>
    <tr>
      <th>26</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Gym</td>
      <td>Beer Store</td>
      <td>Coffee Shop</td>
      <td>Japanese Restaurant</td>
      <td>Art Gallery</td>
      <td>Sporting Goods Shop</td>
      <td>Bike Shop</td>
      <td>Café</td>
      <td>Supermarket</td>
      <td>Discount Store</td>
    </tr>
    <tr>
      <th>27</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Gym</td>
      <td>Beer Store</td>
      <td>Coffee Shop</td>
      <td>Japanese Restaurant</td>
      <td>Art Gallery</td>
      <td>Sporting Goods Shop</td>
      <td>Bike Shop</td>
      <td>Café</td>
      <td>Supermarket</td>
      <td>Discount Store</td>
    </tr>
    <tr>
      <th>28</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Bank</td>
      <td>Mobile Phone Shop</td>
      <td>Bridal Shop</td>
      <td>Sandwich Place</td>
      <td>Diner</td>
      <td>Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Supermarket</td>
      <td>Middle Eastern Restaurant</td>
    </tr>
    <tr>
      <th>29</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Furniture / Home Store</td>
      <td>Metro Station</td>
      <td>Miscellaneous Shop</td>
      <td>Caribbean Restaurant</td>
      <td>Massage Studio</td>
      <td>Coffee Shop</td>
      <td>Bar</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
    </tr>
    <tr>
      <th>30</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Grocery Store</td>
      <td>Park</td>
      <td>Shopping Mall</td>
      <td>Business Service</td>
      <td>Airport</td>
      <td>Liquor Store</td>
      <td>Gym / Fitness Center</td>
      <td>Baseball Field</td>
      <td>Home Service</td>
      <td>Food Truck</td>
    </tr>
    <tr>
      <th>31</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Grocery Store</td>
      <td>Park</td>
      <td>Shopping Mall</td>
      <td>Business Service</td>
      <td>Airport</td>
      <td>Liquor Store</td>
      <td>Gym / Fitness Center</td>
      <td>Baseball Field</td>
      <td>Home Service</td>
      <td>Food Truck</td>
    </tr>
    <tr>
      <th>32</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Grocery Store</td>
      <td>Park</td>
      <td>Shopping Mall</td>
      <td>Business Service</td>
      <td>Airport</td>
      <td>Liquor Store</td>
      <td>Gym / Fitness Center</td>
      <td>Baseball Field</td>
      <td>Home Service</td>
      <td>Food Truck</td>
    </tr>
    <tr>
      <th>33</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Grocery Store</td>
      <td>Park</td>
      <td>Shopping Mall</td>
      <td>Business Service</td>
      <td>Airport</td>
      <td>Liquor Store</td>
      <td>Gym / Fitness Center</td>
      <td>Baseball Field</td>
      <td>Home Service</td>
      <td>Food Truck</td>
    </tr>
    <tr>
      <th>34</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Portuguese Restaurant</td>
      <td>Pizza Place</td>
      <td>French Restaurant</td>
      <td>Coffee Shop</td>
      <td>Hockey Arena</td>
      <td>Intersection</td>
      <td>Doner Restaurant</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
    </tr>
    <tr>
      <th>35</th>
      <td>East York</td>
      <td>1.0</td>
      <td>Pizza Place</td>
      <td>Pet Store</td>
      <td>Athletics &amp; Sports</td>
      <td>Gastropub</td>
      <td>Intersection</td>
      <td>Pharmacy</td>
      <td>Bus Line</td>
      <td>Breakfast Spot</td>
      <td>Bank</td>
      <td>Gym / Fitness Center</td>
    </tr>
    <tr>
      <th>36</th>
      <td>East York</td>
      <td>1.0</td>
      <td>Skating Rink</td>
      <td>Park</td>
      <td>Athletics &amp; Sports</td>
      <td>Beer Store</td>
      <td>Dance Studio</td>
      <td>Curling Ice</td>
      <td>Intersection</td>
      <td>Doner Restaurant</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
    </tr>
    <tr>
      <th>37</th>
      <td>East Toronto</td>
      <td>1.0</td>
      <td>Trail</td>
      <td>Health Food Store</td>
      <td>Pub</td>
      <td>Donut Shop</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
    </tr>
    <tr>
      <th>38</th>
      <td>East York</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Sporting Goods Shop</td>
      <td>Bank</td>
      <td>Burger Joint</td>
      <td>Furniture / Home Store</td>
      <td>Shopping Mall</td>
      <td>Grocery Store</td>
      <td>Electronics Store</td>
      <td>Sports Bar</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>39</th>
      <td>East York</td>
      <td>1.0</td>
      <td>Sandwich Place</td>
      <td>Indian Restaurant</td>
      <td>Yoga Studio</td>
      <td>Middle Eastern Restaurant</td>
      <td>Burger Joint</td>
      <td>Bus Line</td>
      <td>Restaurant</td>
      <td>Pizza Place</td>
      <td>Pharmacy</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>41</th>
      <td>East Toronto</td>
      <td>1.0</td>
      <td>Greek Restaurant</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Bookstore</td>
      <td>Furniture / Home Store</td>
      <td>Ice Cream Shop</td>
      <td>Restaurant</td>
      <td>Brewery</td>
      <td>Bubble Tea Shop</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>42</th>
      <td>East Toronto</td>
      <td>1.0</td>
      <td>Park</td>
      <td>Pub</td>
      <td>Brewery</td>
      <td>Sandwich Place</td>
      <td>Burrito Place</td>
      <td>Fast Food Restaurant</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Italian Restaurant</td>
      <td>Restaurant</td>
      <td>Steakhouse</td>
    </tr>
    <tr>
      <th>43</th>
      <td>East Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Brewery</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Gastropub</td>
      <td>American Restaurant</td>
      <td>Bookstore</td>
      <td>Ice Cream Shop</td>
      <td>Middle Eastern Restaurant</td>
      <td>Yoga Studio</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Central Toronto</td>
      <td>1.0</td>
      <td>Gym / Fitness Center</td>
      <td>Hotel</td>
      <td>Dance Studio</td>
      <td>Department Store</td>
      <td>Sandwich Place</td>
      <td>Dog Run</td>
      <td>Breakfast Spot</td>
      <td>Food &amp; Drink Shop</td>
      <td>Park</td>
      <td>German Restaurant</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Central Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Clothing Store</td>
      <td>Yoga Studio</td>
      <td>Restaurant</td>
      <td>Bagel Shop</td>
      <td>Café</td>
      <td>Diner</td>
      <td>Ice Cream Shop</td>
      <td>Furniture / Home Store</td>
      <td>Sporting Goods Shop</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Central Toronto</td>
      <td>1.0</td>
      <td>Dessert Shop</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Gym</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Sushi Restaurant</td>
      <td>Coffee Shop</td>
      <td>Pharmacy</td>
      <td>Brewery</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Central Toronto</td>
      <td>1.0</td>
      <td>Trail</td>
      <td>Playground</td>
      <td>Drugstore</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Women's Store</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Central Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Liquor Store</td>
      <td>Supermarket</td>
      <td>Vietnamese Restaurant</td>
      <td>Fried Chicken Joint</td>
      <td>Sushi Restaurant</td>
      <td>Light Rail Station</td>
      <td>Pizza Place</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Pizza Place</td>
      <td>Café</td>
      <td>Market</td>
      <td>Italian Restaurant</td>
      <td>Pub</td>
      <td>Park</td>
      <td>Bakery</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Japanese Restaurant</td>
      <td>Gay Bar</td>
      <td>Sushi Restaurant</td>
      <td>Restaurant</td>
      <td>Yoga Studio</td>
      <td>Bubble Tea Shop</td>
      <td>Pub</td>
      <td>Café</td>
      <td>Men's Store</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Pub</td>
      <td>Bakery</td>
      <td>Breakfast Spot</td>
      <td>Café</td>
      <td>Theater</td>
      <td>Hotel</td>
      <td>Chocolate Shop</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Bubble Tea Shop</td>
      <td>Japanese Restaurant</td>
      <td>Cosmetics Shop</td>
      <td>Theater</td>
      <td>Diner</td>
      <td>Fast Food Restaurant</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Cocktail Bar</td>
      <td>Restaurant</td>
      <td>Beer Bar</td>
      <td>Gastropub</td>
      <td>American Restaurant</td>
      <td>Cosmetics Shop</td>
      <td>Hotel</td>
      <td>Seafood Restaurant</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Beer Bar</td>
      <td>Cheese Shop</td>
      <td>Cocktail Bar</td>
      <td>Seafood Restaurant</td>
      <td>Bakery</td>
      <td>Farmers Market</td>
      <td>Sporting Goods Shop</td>
      <td>Bistro</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Italian Restaurant</td>
      <td>Sandwich Place</td>
      <td>Department Store</td>
      <td>Thai Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Burger Joint</td>
      <td>Bubble Tea Shop</td>
      <td>Salad Place</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Hotel</td>
      <td>Gym</td>
      <td>Bar</td>
      <td>Clothing Store</td>
      <td>Thai Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Salad Place</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Aquarium</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Fried Chicken Joint</td>
      <td>Scenic Lookout</td>
      <td>Brewery</td>
      <td>Restaurant</td>
      <td>Pizza Place</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Seafood Restaurant</td>
      <td>Salad Place</td>
      <td>Japanese Restaurant</td>
      <td>American Restaurant</td>
      <td>Concert Hall</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>American Restaurant</td>
      <td>Gym</td>
      <td>Seafood Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Japanese Restaurant</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>62</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Sandwich Place</td>
      <td>Italian Restaurant</td>
      <td>Greek Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Pharmacy</td>
      <td>Pizza Place</td>
      <td>Pub</td>
      <td>Café</td>
      <td>Butcher</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Central Toronto</td>
      <td>1.0</td>
      <td>Garden</td>
      <td>Music Venue</td>
      <td>Doner Restaurant</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Donut Shop</td>
      <td>Department Store</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Central Toronto</td>
      <td>1.0</td>
      <td>Park</td>
      <td>Jewelry Store</td>
      <td>Sushi Restaurant</td>
      <td>Trail</td>
      <td>Electronics Store</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
      <td>Drugstore</td>
      <td>Donut Shop</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Central Toronto</td>
      <td>1.0</td>
      <td>Sandwich Place</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Indian Restaurant</td>
      <td>Donut Shop</td>
      <td>Burger Joint</td>
      <td>Middle Eastern Restaurant</td>
      <td>BBQ Joint</td>
      <td>Pub</td>
      <td>Liquor Store</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Café</td>
      <td>Bookstore</td>
      <td>Bakery</td>
      <td>Bar</td>
      <td>Japanese Restaurant</td>
      <td>Sandwich Place</td>
      <td>Pub</td>
      <td>Bank</td>
      <td>Beer Bar</td>
      <td>Beer Store</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Bar</td>
      <td>Café</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Coffee Shop</td>
      <td>Vietnamese Restaurant</td>
      <td>Gaming Cafe</td>
      <td>Bakery</td>
      <td>Dessert Shop</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Airport Lounge</td>
      <td>Airport Service</td>
      <td>Harbor / Marina</td>
      <td>Airport Terminal</td>
      <td>Rental Car Location</td>
      <td>Coffee Shop</td>
      <td>Plane</td>
      <td>Bar</td>
      <td>Boutique</td>
      <td>Boat or Ferry</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Seafood Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Pub</td>
      <td>Café</td>
      <td>Beer Bar</td>
      <td>Hotel</td>
      <td>Breakfast Spot</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Japanese Restaurant</td>
      <td>Gym</td>
      <td>Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Seafood Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Salad Place</td>
    </tr>
    <tr>
      <th>71</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Clothing Store</td>
      <td>Women's Store</td>
      <td>Coffee Shop</td>
      <td>Boutique</td>
      <td>Furniture / Home Store</td>
      <td>Gift Shop</td>
      <td>Event Space</td>
      <td>Accessories Store</td>
      <td>Vietnamese Restaurant</td>
      <td>Convenience Store</td>
    </tr>
    <tr>
      <th>72</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Park</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
      <td>Japanese Restaurant</td>
      <td>Pub</td>
      <td>Doner Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
    </tr>
    <tr>
      <th>73</th>
      <td>York</td>
      <td>1.0</td>
      <td>Field</td>
      <td>Hockey Arena</td>
      <td>Tennis Court</td>
      <td>Dog Run</td>
      <td>Trail</td>
      <td>Electronics Store</td>
      <td>Eastern European Restaurant</td>
      <td>Escape Room</td>
      <td>Dumpling Restaurant</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Grocery Store</td>
      <td>Café</td>
      <td>Park</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Restaurant</td>
      <td>Baby Store</td>
      <td>Athletics &amp; Sports</td>
      <td>Candy Store</td>
      <td>Nightclub</td>
    </tr>
    <tr>
      <th>76</th>
      <td>West Toronto</td>
      <td>1.0</td>
      <td>Pharmacy</td>
      <td>Bakery</td>
      <td>Supermarket</td>
      <td>Grocery Store</td>
      <td>Park</td>
      <td>Middle Eastern Restaurant</td>
      <td>Brewery</td>
      <td>Bank</td>
      <td>Bar</td>
      <td>Music Venue</td>
    </tr>
    <tr>
      <th>77</th>
      <td>West Toronto</td>
      <td>1.0</td>
      <td>Bar</td>
      <td>Coffee Shop</td>
      <td>Vietnamese Restaurant</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Asian Restaurant</td>
      <td>Men's Store</td>
      <td>Miscellaneous Shop</td>
      <td>Beer Store</td>
      <td>Brewery</td>
    </tr>
    <tr>
      <th>78</th>
      <td>West Toronto</td>
      <td>1.0</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Breakfast Spot</td>
      <td>Nightclub</td>
      <td>Pet Store</td>
      <td>Stadium</td>
      <td>Burrito Place</td>
      <td>Restaurant</td>
      <td>Climbing Gym</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>80</th>
      <td>York</td>
      <td>1.0</td>
      <td>Restaurant</td>
      <td>Discount Store</td>
      <td>Sandwich Place</td>
      <td>Turkish Restaurant</td>
      <td>Bar</td>
      <td>Doner Restaurant</td>
      <td>Diner</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>81</th>
      <td>York</td>
      <td>1.0</td>
      <td>Bus Line</td>
      <td>Convenience Store</td>
      <td>Breakfast Spot</td>
      <td>Brewery</td>
      <td>Women's Store</td>
      <td>Dumpling Restaurant</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Drugstore</td>
    </tr>
    <tr>
      <th>82</th>
      <td>West Toronto</td>
      <td>1.0</td>
      <td>Thai Restaurant</td>
      <td>Bar</td>
      <td>Mexican Restaurant</td>
      <td>Café</td>
      <td>Discount Store</td>
      <td>Diner</td>
      <td>Bakery</td>
      <td>Fried Chicken Joint</td>
      <td>Speakeasy</td>
      <td>Cajun / Creole Restaurant</td>
    </tr>
    <tr>
      <th>83</th>
      <td>West Toronto</td>
      <td>1.0</td>
      <td>Gift Shop</td>
      <td>Breakfast Spot</td>
      <td>Dessert Shop</td>
      <td>Restaurant</td>
      <td>Bookstore</td>
      <td>Bar</td>
      <td>Dog Run</td>
      <td>Movie Theater</td>
      <td>Cuban Restaurant</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>84</th>
      <td>West Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Sushi Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Pub</td>
      <td>Pizza Place</td>
      <td>Yoga Studio</td>
      <td>Smoothie Shop</td>
      <td>Bookstore</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>85</th>
      <td>Downtown Toronto</td>
      <td>1.0</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Bank</td>
      <td>Beer Bar</td>
      <td>Smoothie Shop</td>
      <td>Sandwich Place</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Portuguese Restaurant</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Mississauga</td>
      <td>1.0</td>
      <td>Hotel</td>
      <td>Coffee Shop</td>
      <td>Gym</td>
      <td>Burrito Place</td>
      <td>Fried Chicken Joint</td>
      <td>Middle Eastern Restaurant</td>
      <td>Gas Station</td>
      <td>Sandwich Place</td>
      <td>Intersection</td>
      <td>American Restaurant</td>
    </tr>
    <tr>
      <th>87</th>
      <td>East Toronto</td>
      <td>1.0</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Butcher</td>
      <td>Recording Studio</td>
      <td>Skate Park</td>
      <td>Auto Workshop</td>
      <td>Burrito Place</td>
      <td>Garden</td>
      <td>Garden Center</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Etobicoke</td>
      <td>1.0</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Gym</td>
      <td>American Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Bakery</td>
      <td>Fried Chicken Joint</td>
      <td>Restaurant</td>
      <td>Liquor Store</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Etobicoke</td>
      <td>1.0</td>
      <td>Pizza Place</td>
      <td>Gym</td>
      <td>Pharmacy</td>
      <td>Coffee Shop</td>
      <td>Sandwich Place</td>
      <td>Pub</td>
      <td>Women's Store</td>
      <td>Dog Run</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Etobicoke</td>
      <td>1.0</td>
      <td>River</td>
      <td>Pool</td>
      <td>Donut Shop</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Drugstore</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Etobicoke</td>
      <td>1.0</td>
      <td>Baseball Field</td>
      <td>Construction &amp; Landscaping</td>
      <td>Women's Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Drugstore</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Etobicoke</td>
      <td>1.0</td>
      <td>Hardware Store</td>
      <td>Tanning Salon</td>
      <td>Wings Joint</td>
      <td>Kids Store</td>
      <td>Fast Food Restaurant</td>
      <td>Discount Store</td>
      <td>Convenience Store</td>
      <td>Gym</td>
      <td>Burrito Place</td>
      <td>Burger Joint</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Etobicoke</td>
      <td>1.0</td>
      <td>Liquor Store</td>
      <td>Pharmacy</td>
      <td>Pizza Place</td>
      <td>Beer Store</td>
      <td>Shopping Plaza</td>
      <td>Coffee Shop</td>
      <td>Convenience Store</td>
      <td>Café</td>
      <td>Doner Restaurant</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>96</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Pizza Place</td>
      <td>Furniture / Home Store</td>
      <td>Doner Restaurant</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Donut Shop</td>
      <td>Falafel Restaurant</td>
    </tr>
    <tr>
      <th>97</th>
      <td>North York</td>
      <td>1.0</td>
      <td>Baseball Field</td>
      <td>Dumpling Restaurant</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Drugstore</td>
      <td>Women's Store</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Etobicoke</td>
      <td>1.0</td>
      <td>Pizza Place</td>
      <td>Discount Store</td>
      <td>Coffee Shop</td>
      <td>Sandwich Place</td>
      <td>Intersection</td>
      <td>Chinese Restaurant</td>
      <td>Women's Store</td>
      <td>Dog Run</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>100</th>
      <td>Etobicoke</td>
      <td>1.0</td>
      <td>Park</td>
      <td>Pizza Place</td>
      <td>Bus Line</td>
      <td>Sandwich Place</td>
      <td>Dog Run</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Doner Restaurant</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Etobicoke</td>
      <td>1.0</td>
      <td>Grocery Store</td>
      <td>Pizza Place</td>
      <td>Liquor Store</td>
      <td>Pharmacy</td>
      <td>Fast Food Restaurant</td>
      <td>Beer Store</td>
      <td>Fried Chicken Joint</td>
      <td>Sandwich Place</td>
      <td>Women's Store</td>
      <td>Dog Run</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Etobicoke</td>
      <td>1.0</td>
      <td>Bar</td>
      <td>Garden Center</td>
      <td>Rental Car Location</td>
      <td>Drugstore</td>
      <td>Women's Store</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



##### Cluster 3


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>North York</td>
      <td>2.0</td>
      <td>Martial Arts School</td>
      <td>Women's Store</td>
      <td>Drugstore</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Dumpling Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



##### Cluster 4


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>94</th>
      <td>Etobicoke</td>
      <td>3.0</td>
      <td>Print Shop</td>
      <td>Women's Store</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Drugstore</td>
      <td>Dessert Shop</td>
    </tr>
  </tbody>
</table>
</div>



##### Cluster 5


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Scarborough</td>
      <td>4.0</td>
      <td>Fast Food Restaurant</td>
      <td>Drugstore</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
      <td>Dumpling Restaurant</td>
      <td>Wings Joint</td>
    </tr>
  </tbody>
</table>
</div>


