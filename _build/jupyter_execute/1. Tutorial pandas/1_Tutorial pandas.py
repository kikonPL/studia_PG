#!/usr/bin/env python
# coding: utf-8

# # Pakiet pandas - tutorial
# 
# ## Wstęp
# Pandas jest wykorzystywany do pracy z danymi tabelarycznymi - np. danymi z arkuszy kalkulacyjnych bądź baz danych. Pakiet jest przydatny w eksploracji, czyszczeniu i przetwarzaniu danych. W pandasie dane tabularyczne nazywa się **DataFrame**. Każda zmienna zapisywana jest w kolumnie, zaś każda obserwacja w wierszu.
# 

# ## Tworzenie DataFrame
# Ramkę danych można stworzyć na wiele sposobów: z plików roznego rodzaju, słownika, listy, numpy.array, itp.
# ### Tworzenie DataFrame ze słownika

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data_dict = {'column_1': ['variable_1', 'variable_2'], 'column_2': [1, 3], 'column_3': ['variable_1', 'variable_2']}
df_from_dict = pd.DataFrame(data=data_dict)


# In[3]:


df_from_dict


# ### Tworzenie DataFrame z list

# In[4]:


df_from_list = pd.DataFrame(data=[['variable_1', 1, 'variable_1'], # przypisanie wierszy w formie listy
                                  ['variable_2', 3, 'variable_2']],
                           columns=['column_1', 'column_2', 'column_3'], # zdefiniowanie nazw kolumn
                           index=['index_1', 'index_2'] # zdefiniowanie nazw indeksow
                           )


# In[5]:


df_from_list


# ### Wczytywanie DataFrame z pliku
# DataFrame można wczytać z wszelkiego rodzaju plików: csv, xls, parquet, json, sql, gbq (tabela z Google BigQuery) i wielu innych. W tym celu wystarczy użyć funkcji **read_*** gdzie * oznacza rodzaj pliku. Na przykład chcąc wczytać plik csv użyjemy funkcji **read_csv**.
# 
# Analogicznie chcąc zapisać DataFrame używa się funkcji **to_*** gdzie * oznacza rodzaj pliku.

# In[6]:


df = pd.read_csv(filepath_or_buffer='mpg.csv', # sciezka do pliku
                sep=',', # separator
                header=0, # naglowek (nazwy kolumn)
                index_col=0 # kolumna z indeksem
                )


# In[7]:


# Wyswietlenie pierwszych 5 wierszy
df.head()


# ## Podstawowe funkcje
# Pandas posiada wiele wbudowanych funkcji, dzięki którym można lepiej poznać dane. Poniżej przedstawiono część z nich.

# In[8]:


# Rozmiar ramki danych
df.shape


# In[9]:


# Podsumowanie DataFrame - typy danych, ilosc, liczba wartosci niezerowych (tych ktore nie sa null)
df.info() 


# In[10]:


# Wyswietlenie typow danych
df.dtypes 


# In[11]:


# Statystyki opisowych
df.describe() 


# In[12]:


# To samo tylko dla okreslonych typow danych (domyslnie bierze tylko zmienne numeryczne)
df.describe(include='object') 


# In[13]:


# Lista nazw kolumn
df.columns 


# In[14]:


# Lista indeksow
df.index


# In[15]:


# Ilosc unikalnych wartosci w okreslonych kolumnach
df.nunique() 


# In[16]:


# Unikalne wartosci w kolumnie cylinders wraz
# z liczba wystapien w znormalizowanej formie (procent ogolu)
df['cylinders'].value_counts(normalize=True)


# **Inne podstawowe funkcje**
# 
# |funkcja|opis|
# |--|--|
# |min()|wartosc minimalna|
# |max()|wartosc maksymalna|
# |count()|liczba wartosci|
# |sum()|suma wartosci|
# |median()|mediana|
# |quantile([0.25, 0.75])|kwantyle (mozna wprowadzic inne niz [0.25, 0.75])|
# |mean()|srednia|
# |std()|odchylenie standardowe|
# |apply(*function*)|dzialanie na DataFrame inna funkcja|

# In[17]:


df['weight'].std()


# In[18]:


df.max()


# In[19]:


# Macierz korelacji
df.corr(method='spearman')


# ## Tworzenie podzbiorów

# In[20]:


# Pierwsze 3 wiersze (nie wpisujac zadnej liczby domyslnie brana jest 5)
df.head(3) 


# In[21]:


# Wiersze z okreslonego zakresu indeksow
df[27:33] 


# In[22]:


# Ostatnie 8 wierzy (nie wpisujac zadnej liczby domyslnie brana jest 5)
df.tail(8) 


# In[23]:


# Ostatnie 8 wierszy (drugi sposob)
df[-8:] 


# In[24]:


# Wybranie specyficznych kolumn
df[['cylinders', 'horsepower']].head()


# In[25]:


# Wartosci specyficznej kolumny z okreslonego zakresu wierszy
df.cylinders[15:23]


# 
# <div class="alert alert-block alert-info">
# <b>Info</b> 
# 
# Chcąc działać na wierszach spełniających specyficzne warunki warto tworzyć zmienną maskujacą (która zwraca wartość True/False dla każdego wiersza) i następnie pisać df[mask] zamiast df[((pierwszy warunek) | (drugi warunek)) & (trzeci warunek)]. Wizualnie wygląda to dużo lepiej.
# </div>

# In[26]:


# Chcemy wiersze gdzie model_year jest co najwyzej 70 a zmienna origin miesci sie w zbiorze 'japan' i 'europe'
mask = (df['model_year']<=70) & (df['origin'].isin(['japan', 'europe']))
mask


# In[27]:


df[mask]


# In[28]:


# Wybieranie losowych wierszy
df.sample(5)


# 
# <div class="alert alert-block alert-info">
# <b>Info</b> 
# 
# Przydatnymi funkcjami w selekcji specyficznych wierszy/kolumn są .loc oraz .iloc. Zwłaszcza wtedy kiedy chcemy nadpisać jakąś wartość w DataFramie.
# </div>

# In[29]:


# Wybranie wszystkich wierszy i kolumn z zakresu 5:9
df.iloc[:,5:9] 


# In[30]:


# Wybranie wszystkich wierszy i kolumn z zakresu 5:9 (po nazwach)
df.loc[:,'acceleration':'name'] 


# ## Operacje na danych
# ### Nadpisanie określonej wartości
# 
# W tym przypadku zmienimy wartość 'horsepower' dla drugiej obserwacji na null

# In[31]:


df.iloc[1,:]


# In[32]:


df.loc[1,'horsepower'] = np.nan


# In[33]:


df.iloc[1,:]


# ### Sortowanie i resetowanie indeksów

# In[34]:


df.sort_values(by=['model_year', 'horsepower', 'name'], # kolumny sortujace
              ascending=False, # kolejnosc malejaca
              inplace=True # nadpisanie aktualnej ramki danych
              )


# In[35]:


# Wyswietlenie posortowanego DataFrame. Zauwazyc mozna, ze indeksy nie ukladaja sie od poczatku. Pozostal stary porzadek
df.head()


# In[36]:


df.reset_index(inplace=True, # nadpisanie aktualnej ramki danych
              drop=True # usuniecie poprzednich indeksow
              )


# In[37]:


df.head()


# ### Grupowanie danych
# Poniżej przedstawiono przykład grupowania kolumny **horsepower** i **model_year** ze względu na kolumne **cylinders**.
# 
# Funkcja max pokazuje jaką funkcją działamy na pogrupowane dane - w tym przypadku zostawiamy wartość maksymalną każdej podgrupy. Równie dobrze można użyć pozostałych funkcji, które zostały wcześniej przedstawione.
# 
# Funkcja reset_index() przenosi indeksy (w tym przypadku cylinders) do kolumny.

# In[38]:


grouped = df.groupby(by='cylinders')[['horsepower','model_year']].max()


# In[39]:


grouped


# In[40]:


grouped = grouped.reset_index()
grouped


# #### Działanie różnymi funkcjami na każdą z kolumn

# In[41]:


def median_mean(x):
    return x.median() - x.mean()

df.groupby(by='cylinders')[['horsepower','model_year']].agg({'horsepower': ['sum', 'max'], # dzialanie lista funkcji
                                                             'model_year': median_mean}) # dzialanie wlasna funkcja                                                           


# ### Funkcja lambda

# 
# <div class="alert alert-block alert-info">
# <b>Info</b> 
# 
# Funkcja lambda jest bardzo przydatnym narzędziem. Działając funkcją lambda na ramkę danych prawdopodobnie nasze obliczenia wykonaja się szybciej niż robienie pętli czy innych operacji.
# </div>

# In[42]:


# Na poczatku przygotujemy pogrupowana ramke danych. 
# Tym razem na pogrupowane dane zadzialamy w taki sposob ze dla kazdej grupy utworzymy liste wartosci ktore naleza do grupy
grouped2 = df.groupby(by='cylinders')['horsepower'].apply(list).reset_index()


# In[43]:


grouped2


# Zauwazyc mozna, ze wartosci w liscie sie powtarzaja. Pozbyc sie ich mozemy dzieki wyrazeniu lambda

# In[44]:


grouped2['unique_horsepower'] = grouped2['horsepower'].apply(lambda x: list(dict.fromkeys(x)))


# In[45]:


grouped2


# In[46]:


len(grouped2.loc[1,'horsepower'])


# In[47]:


len(grouped2.loc[1,'unique_horsepower'])


# ### Tworzenie nowych kolumn

# In[48]:


df[pd.isnull(df.horsepower)]


# 
# <div class="alert alert-block alert-info">
# <b>Info</b> 
# 
# W wyrażeniu lambda tworząc instrukcję warunkową if-else nie używamy standardowej struktury:
# ```    
# if (condition):   
#   do sth    
# else:    
#   do sth
# ```    
# Zamiast tego używamy struktury jednowierszowej:
# ```
# do sth if (condtion) else do sth 
# ```
# </div>

# In[49]:


# Tworzymy nowa kolumne w taki sposob, ze wartosc nowej kolumny jest
# rowna kolumnie 'horsepower' jesli 'acceleration' jest null (lub odwrotnie)
# albo mnozeniu horsepower i aceeleration jesli obie wartosci nie sa null

# W tym przypadku tworzymy wyrazenie lambda na calej ramce danych (nie na kolumnie jak bylo to wyzej).
# Wazne jest aby w tym przypadku na koncu dodac axis=1.

df['horsepower*acceleration'] = df.apply(lambda row: 
                                         row['horsepower'] # Tworzymy wyrazenie warunkowe (Zwroc uwage na strukture!)
                                         if pd.isnull(row['acceleration'])
                                         else
                                             row['acceleration'] 
                                             if pd.isnull(row['horsepower'])
                                             else
                                                 row['horsepower']*row['acceleration'],
                                         axis=1
                                        )


# In[50]:


df.head()


# **II sposób**
# 
# Wyniesienie wnętrza funkcji lambda do osobnej funkcji. Sprawia to, że kod staje się bardziej przejrzysty.

# In[51]:


def row_operation(x):
    if pd.isnull(x['acceleration']):
        return x['horsepower']
    else:
        if pd.isnull(x['horsepower']):
            return x['acceleration']
        else:
            return x['horsepower']*x['acceleration']

df['horsepower*acceleration'] = df.apply(lambda row: row_operation(row), axis=1)
df.head()


# ### Tworzenie nowych wierszy

# In[52]:


# Tworzymy liste wartosci, ktore chcemy dopisac do ramki danych
row_list = [31.0, 4, 119.0, 82.0, 2720, 19.4, 82, 'usa', 'chevy s-10', 1590.8]

# Przy pomocy loc dopisujemy wiersz z indeksem 398
df.loc[398,:] = row_list


# In[53]:


# Tworzymy nowy DataFrame ze slownika po czym dolaczamy do naszej ramki danych
row_dict = {'mpg': [31.0], 'cylinders': [4], 'displacement': [119.0],
            'horsepower': [82.0], 'weight': [2720], 'acceleration': [19.4],
            'model_year': [82], 'origin': ['usa'], 'name': ['chevy s-10'], 'horsepower*acceleration': [1590.8]}
df_to_append = pd.DataFrame(data = row_dict)
df = df.append(df_to_append)


# In[54]:


# Dodajac nowa ramke danych zauwazyc mozna ze indeksy pozostaja takie same jak w oryginalnych ramkach danych

df.tail()


# In[55]:


df.reset_index(inplace=True, drop=True)


# In[56]:


df.tail()


# ### Usuwanie danych

# In[57]:


# Uwsuwanie wierszy
df.drop(index=399, inplace=True)
df.tail()


# In[58]:


# Usuwanie kolumny
df.drop(columns='horsepower*acceleration', inplace=True)
df[:5]


# ### Działania na wartościach nullowych

# In[59]:


# Patrzymy czy ktorykolwiek (any) wiersz (axis=1) jest rowny null
mask = df.isnull().any(axis=1)
mask


# In[60]:


df[mask]


# 
# <div class="alert alert-block alert-info">
# <b>Info</b> 
# 
# Nie ma prostej odpowiedzi jak zastąpić wartości nullowe. Wszystko zależy od tego na jakim zbiorze pracujemy. Czasem wiersze posiadające nulle sa usuwane o ile ich liczebność stanowi odsetek całego zbioru. Prostą metodą jest zastąpienie nulli wartością średnią lub medianą dla danych grup (tj. np. średnia wartosc horsepower dla samochodów z 4 cylindrami gdzie model_year jest powyżej 80). Wszystko zalezy od badanego problemu.
# </div>

# In[61]:


df.describe()


# In[62]:


df[df['weight']<2000].describe()


# Widać, że konie mechaniczne samochodów z wagą poniżej 2000 różnią sie statystykami od reszty zbioru. Z tego względu wartość null dla wiersza 330 nadpiszemy średnią wartoscią 'horsepower' dla samochodów z wagą 2000 (zauważ, że mediana i średnia są do siebie zbliżone).

# In[63]:


low_weight_mean = int(df[df['weight']<2000]['horsepower'].mean()) # int bo nie moze byc polowa konia mechanicznego

# Do nadpisywania wartosci nullowych sluzy funkcja fillna()
df.loc[df['weight']<2000, 'horsepower'] = df.loc[df['weight']<2000, 'horsepower'].fillna(low_weight_mean)


# In[64]:


df[mask]


# In[65]:


# Usuwanie wierszy z nullami
df = df.dropna()
df.reset_index(drop=True, inplace=True)


# ### Usuwanie duplikatow

# In[66]:


# Zauwazmy ze mamy duplikat (indeks 392 i 17)
df[df['name'] == 'chevy s-10']


# In[67]:


df.drop_duplicates(inplace=True)


# In[68]:


df[df['name'] == 'chevy s-10']


# ### Zmiana nazw kolumn

# In[69]:


# Do parametru columns przypisujemy slownik gdzie klucze to aktualna nazwa kolumny a wartosci to nowe nazwy
df.rename(columns = {'mpg': 'miles_per_gallon'}, inplace=True)
df.head()


# ### Zmiana danych liczbowych na kategorie

# 
# <div class="alert alert-block alert-info">
# <b>Info</b> 
# 
# Czasem bardziej miarodajna jest informacja, że dany obiekt należy do jakiejś grupy zamiast przypisanej do niego konkretnej liczby. Np. znając wzrost i wagę danego pacjeta można stworzyć zmienną kategoryczną oznaczającą jak bardzo otyły jest pacjent. 
# </div>

# In[70]:


print(f"light weight count {len(df[df.weight<=2000])}")
print(f"normal weight {len(df[(df.weight>2000) & (df.weight<=3600)])}")
print(f"heavy weight count {len(df[df.weight>3600])}")


# In[71]:


# Stworzenie 3 przedzialow (waga do 2000, miedzy 2000 a 3600 i powyzej 3600)
bins = pd.IntervalIndex.from_tuples([(df['weight'].min()-1, 2000),(2000, 3600),(3600, df['weight'].max()+1)])

# Przypisanie wartosci kolumny 'weight' do odpowiedniego przedzialu
x = pd.cut(df['weight'].to_list(), bins)
x.categories = ['light_weight', 'normal_weight', 'heavy_weight']

# Stworzenie nowej kolumny z kategoriami
df['weight_category'] = x


# In[72]:


df.tail()


# ### Iterowanie po kolejnych wierszach
# 
# Chcąc napisać pętlę, która będzie iterowana po kolejnych wierszach można użyć funkcji iterrows.

# In[73]:


for i, row in df[:5].iterrows():
    print(f'Indeks: {i}, Liczba koni mechanicznych: {row.horsepower}')


# ## Lączenie i transformacja danych

# ### Merge
# 
# Funkcja merge łączy dwie ramki danych w analogiczny sposób jak dzieje się to w sqlu. Definiuje się, które ramki danych chce się połączyć, w jaki sposób (nazwy te same co w sql) oraz ze względu, na które kolumny.

# In[74]:


# Przypisanie pierwszych 50 wartosci df do oddzielnej ramki danych
df2 = df[:50].copy()

# Stworzenie nowej kolumny name_origin
df2['name_origin'] = df2.apply(lambda row: str(row['name']+' from '+row['origin']),axis=1)

# Zostawienie tylko trzech kolumn
df2 = df2[['name', 'origin', 'name_origin']]
df2.head()


# In[75]:


df_merged = df.merge(df2, 
                    how='inner', # parametr how mowi jak dwie ramki danych maja byc polaczone
                    on= ['name', 'origin']# parametr on mowi wg jakich kolumn laczone sa ramki danych 
                    )                     # (jesli ich nazwy sie roznia to uzywamy left_on, right_on
df_merged.head()


# ### Concat
# 
# Funkcja concat działa podobnie do wyżej przedstawionego append lub do merge z parametrem how='inner' (zależy czy działamy na wierszach (axis=0) czy kolumnach (axis=1)).

# In[76]:


# Laczenie kolumn (podobne do merge z parametrem how='inner')
df_concat = pd.concat([df, df2], axis=1)
df_concat.head()


# In[77]:


# Dodawanie wierszy (wracamy do naszej wczesniejszej jednowierszowej ramki danych df_to_append) (podobne do append)
df_to_append.rename(columns={'mpg': 'miles_per_gallon'}, inplace=True)
df_to_append.drop(columns='horsepower*acceleration', inplace=True)
df_concat_row = pd.concat([df_to_append, df])
df_concat_row.head()


# ### Melt
# 
# Funkcja melt sprowadza kolumny do wierszy. Przykład poniżej.

# In[78]:


pd.melt(df)


# ### PivotTable
# 
# Tworzenie tabeli przestawnej analogicznie jak w excelu.

# In[79]:


df.pivot_table(values='weight',columns='cylinders', index='name', aggfunc=[np.median, np.mean])


# ### Explode
# 
# Możliwe, że mając dane w kolumnie zapisane w formie listy będziemy chcieli je rozbić na oddzielne kolumny. W tym przypadku przydatna będzie funkcja explode.

# In[80]:


# Wrocmy do ramki danych grouped2
grouped2.head()


# In[81]:


df_to_unstack = grouped2.drop(columns='horsepower').copy()


# In[82]:


df_to_unstack.explode(column='unique_horsepower')

