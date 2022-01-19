#!/usr/bin/env python
# coding: utf-8

# # Analiza statystyczna - tutorial 
# ## Wstęp
# Przed rozpoczęciem prac nad modelami/ analizami warto poznać dane, na których będziemy pracować. 
# W tym celu wykonujemy analizy statystyczne opisujące nam dane.
# 
# <div class="alert alert-block alert-success">
# <b>Cel</b> 
# Celem analizy statystycznej jest pozyskanie jak największej wiedzy z danych
# </div>
# 

# In[1]:


# importy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import normaltest, anderson


# Pobranie danych 
# 
# Pobieramy przykładowy dataset z biblioteki Scikit-Learn 

# In[2]:


from sklearn.datasets import fetch_california_housing


# In[3]:


df = fetch_california_housing(as_frame = True)['frame']


# Odczytajmy informację o tych danych

# In[4]:


print(fetch_california_housing(as_frame = True)['DESCR'])


# In[5]:


df.head()


# ## Pierwsze spojrzenie na dane

# <div class="alert alert-block alert-success">
# <b>Cel</b> 
# 
# Celem poniższego zadania jest analiza statystyczna danych *fetch_california_housing* przed tworzeniem modelu regresyjnego, w którym celem (zmienną objaśnianą) będzie wyznaczenie Mediany wartości domu (**MedHouseVal**)
#     
# </div>

# ### Ogólna wiedza o danych

# Przy pierwszym spojrzeniu na dane warto sprawdzić jaka jest średnia, mediana, wartosci minimalne i maksymalne. Najprostszy sposób, aby to zbadać to wykorzystać metodę *describe()*. Przedstawia ona liczbę danych, średnią, odchylenie standardowe, minimalną wartość, maksymalną wartośc, mediane oraz 25 i 75 percentyl

# In[6]:


df.describe()


# Wsyztskie te informacje można oczywiście wydobyć osobnymi metodami:

# **Liczba wierszy:**
# - dla wszystkich kolulumn <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.count.html">*pd.DataFrame.count()*</a>
# - dla pojedynczej kolumny <a href = https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.count.html>*pd.Series.count()*</a>

# In[7]:


df.count()


# In[8]:


df['MedHouseVal'].count()


# **Średnia:**
# - dla wszytskich kolulumn <a href=https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html>*pd.DataFrame.mean()*</a>
# - dla pojedynczej kolumny <a href = https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mean.html>*pd.Series.mean()*</a>

# In[9]:


df.mean()


# In[10]:


df['MedHouseVal'].mean()


# **Odchylenie standardowe:**
# - dla wszytskich kolulumn <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.std.html">*pd.DataFrame.std()*</a>
# - dla pojedynczej kolumny <a href = https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.std.html>*pd.Series.std()*</a> 

# In[11]:


df.std()


# In[12]:


df['MedHouseVal'].std()


# **Wartość minimalna:**
# - dla wszytskich kolulumn <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.min.html">*pd.DataFrame.min()*</a>
# - dla pojedynczej kolumny <a href = https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.min.html>*pd.Series.min()*</a> 

# In[13]:


df.min()


# In[14]:


df['MedHouseVal'].min()


# **Wartość maksymalna:**
# - dla wszystkich kolulumn <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.max.html">*pd.DataFrame.max()*</a>
# - dla pojedynczej kolumny <a href = https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.max.html>*pd.Series.max()*</a> 

# In[15]:


df.max()


# In[16]:


df['MedHouseVal'].max()


# **Percentyle:**
# - dla wszystkich kolulumn <a href=https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.quantile.html>*pd.DataFrame.quantile()*</a>
# - dla pojedynczej kolumny <a href = https://numpy.org/doc/stable/reference/generated/numpy.quantile.html>*np.quantile()*</a>, lub <a href = https://numpy.org/doc/stable/reference/generated/numpy.percentile.html> *np.percentile()* </a>

# In[17]:


df.quantile([0.25, 0.50, 0.75])


# In[18]:


np.quantile(df['MedHouseVal'], [0.25,0.50,0.75])


# In[19]:


np.percentile(df['MedHouseVal'], [25,50,75])


# In[20]:


np.percentile(df['MedHouseVal'], 25)


# In[21]:


np.quantile(df['MedHouseVal'], 0.25)


# Mamy już pewną ogólną wiedze na temat naszych danych. Przyjżyjmy się teraz poszczególnym kolumnom i sprawdźmy jak rozkładają się  w nich dane

# ### Analiza poszczególnych kolumn

# W pierwszej kolejności sprawdźmy jak wyglada nasz cel, a więc mediana wartości domu

# Poniważ wartość mieszkania może przyjmować wszystkie wartości większe od 0, najlepszym sposobem na sprawdzenie rozkładu jest stworzenie histogramu

# Najprostszą metodą jest wygenerowanie automatycznego histogramy za pomocą metody <a href = https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.hist.html> *pd.Series.hist()* </a>, lub z pakietu matplotlib <a href = https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html> *plt.hist()* </a>

# In[22]:


df['MedHouseVal'].hist(bins = 20)
plt.xlabel('MedHouseVal')
plt.ylabel('Count')
plt.title('House Value histogram')
plt.show()


# In[23]:


plt.hist(df['MedHouseVal'], bins = 20)
plt.xlabel('MedHouseVal')
plt.ylabel('Count')
plt.title('House Value histogram')
plt.show()


# Najwięcej przypadków znajduje się w przedziale 1-2. Dodatkowo rozkład nie przypomina standardowego <a href = https://en.wikipedia.org/wiki/Normal_distribution> rozkładu normalnego </a>. W kolejnych częściach dowiemy się jak sprawdzić czy dane układają się w rozkład normalny, czy nie.

# **Wartości dyskretne**
# 
# Jeżeli w kolumnie znajdują się wartości dyskretne, warto znać ilość poszczególnych wartości.
# W tym celu można wykorzystać funkcję <a href = https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html>*pd.Series.count_values()*</a>.
# 
# Sprawdżmy jak często występują poszczególne mediany wieku domów w danych obszarach:

# In[24]:


df['HouseAge'].value_counts()


# Otrzymane wyniki są automatycznie ustawiane od przypadków najczęściej do najrzadziej występujacych. 
# 
# Występuje 1273 obszarów z medianą wieku domów równą 52 lata. Co ciekawe mediana równa 51 lat występuje tylko w 48 obszarach. Najmniej, bo tylko 4 obszary, mają medianę wieku domów w wysokości 1 roku.
# 
# Sprawdźmy jak wyglądają takie dane na wykresie

# In[25]:


from matplotlib import pyplot as plt

df_plot = df['HouseAge'].value_counts().reset_index().sort_values( 'index')
#df_plot['index']
plt.bar(x = 'index', height = 'HouseAge', data = df_plot)
plt.xlabel('House Age')
plt.ylabel('Counts')
plt.show()


# Widać, że zdecydowanie 52 lata to najczęściej występująca mediana wieku domów. Widać także zwiększenie ilości danych w środkowych przedziale lat.

# Po wyglądzie można się zastanowić czy ta wartość 52 lata nie jest maksymalną wartością, która mogła zostać wpisana. Należy się zastanowić czy nie warto usunąć tych danych do dalszej analizy. Niestety w źródle danych nie ma informacji na ten temat, ale w rzeczywistych warunkach taka informacja powinna być opisana.

# ## Badanie korelacji

# W analizie statystycznej ważnym elementem jest badanie korelacji zmiennych. To dzięki niej wiemy czy występuje jakaś zależność pomiędzy zbiorami danych. 
# 
# Bardzo istotne jest zbadanie korelacji podczas budowania modelu regresyjnego. Badając korelację naszego celu (w tym przykładzie 'MedHouseVal') z cechami, możemy dowiedzieć się która z cech jest naistotniejsza, a która nie pomoże nam w stworzeniu odpowiedniego modelu

# Najczęściej badanymi korelacjami są:
#  - <a href = https://en.wikipedia.org/wiki/Pearson_correlation_coefficient> **Pearsona** </a> - bada zależność liniową w danych X i Y
#  
#  $$
# r_{p}= \frac{cov(X,Y)}{\sigma_X \sigma_Y}
# $$
# 
#  - <a href = https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient> **Spearmana** </a> - bada zależność rank danych X i Y. Wysoka wartość współczynnika korelacji nie oznacza, że dane są liniowo skorelowane, ale wraz z wzrostem wartosci X, zwiększają się wartości Y, jednak niekoniecznie zawsze o taką samą wartość
#  
#  $$
#  r_s = \frac{cov(R(X)R(Y))}{\sigma_{R(X)}\sigma_{R(Y)}}
#  $$
#  
# *gdzie*:
#  
#  $cov()$ - kowariancja, 
#  
#  $\sigma$ - odchylenie standardowe
#  
#  $R()$ - rangi danych
#  
# 

# Najprostrzym sposobem na zbadanie korelacji w tabeli jest wykorzystanie metody <a href = https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html> *pd.DataFrame.corr()* </a>
# 
# Metoda stworzy macież korelacji pomiędzy wszystkimi kolumnami numerycznymi w DataFrame

# In[26]:


df.corr()


# Jeżeli chcielibyśmy wyliczyć korelację Spearmana wystarczy dodać parametr ***method = 'spearman'***

# In[27]:


df.corr(method = 'spearman')


# Z obu korelacji wynika, że najlepiej skorelowaną cechą z ceną domu (zmienną objaśniana) jest wysokość zarobków (zmienna objaśniająca). Przy budowie modelu regresyjnego, to ta cecha będzie jedną z najistotniejszych. To ona najlepiej swoimi wielkościami opisuje cenę domu.
# 
# Skoro znamy najlepszą zmienną objaśniajacą, spójrzmy na wykres punktowy tych dwóch zmiennych.

# In[28]:


plt.plot(df['MedInc'], df['MedHouseVal'],  linestyle = '', marker = '.', alpha = 0.1) 
# parametr alpha wykorzystywany jest do ustawienia przejrzystosci punktów. 
# Im ciemniejszy obszar na wykresie tym więcej punktów się tam znajduje
plt.ylabel('MedHouseVal')
plt.xlabel('MedInc')
plt.title('Wartość domu vs Wysokość zarobków')
plt.show()


# Na wykresie widać tendencję zwiększania się wartości domów przy zwiększajacych się zarobkach.

# Korelacji nie musimy liczyć dla wszystkich kolumn w tabeli. Możemy ją takze zmierzyć tylko dla dwóch wybranych kolumn. 
# Służą do tego metody:
# - <a href= https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html> np.corrcoef()</a> (korelacja Pearsona)
# - <a href = https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html> scipy.stats.pearsonr()</a> (korelacja Pearsona)
# - <a href = https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html> scipy.stats.spearmanr()</a> (korelacja Spearmana)

# In[29]:


np.corrcoef(df['MedHouseVal'], df['MedInc'])


# In[30]:


from scipy.stats import pearsonr

pearsonr(df['MedHouseVal'], df['MedInc'])


# In[31]:


from scipy.stats import spearmanr

spearmanr(df['MedHouseVal'], df['MedInc'])


# Zwróćmy uwagę, że metody z biblioteki *scipy* zwracają dwie wartości. Pierwsza jest to współczynnik korelacji, a druga to ***P-value***, której teraz dokładnie się przyjrzymy 

# ## Testy statystyczne

# <div class="alert alert-block alert-success">
#    <b> Definicje </b>
#     
# **Test statystyczny** – formuła matematyczna pozwalająca oszacować prawdopodobieństwo spełnienia pewnej hipotezy statystycznej w populacji na podstawie próby losowej z tej populacji.
#     
# $\newline$
# 
# **Hipoteza statystyczna** - dowolne przypuszczenie co do rozkładu populacji
# 
# $\newline$
# 
# **Hipoteza zerowa** $H_0$ - Przypuszczenie, które chcemy sprawdzić za pomocą testów statystycznych. 
# 
# *Przykład:* w metodzie *pearsonr* hipotezą zerową jest założenie, że nie istnieje korelacja pomiędzy zbiorami danych
# $\newline$   
# 
# **P-value** - prawdopodobieństwo kumulatywne wylosowania próby takiej lub bardziej skrajnej jak zaobserwowana, przy założeniu, że hipoteza zerowa jest prawdziwa
#     
# *Przykład:* W korelacji pomiędzy wartoscią domu, a zarobkami P-value wynosi 0. Możemy więc stwierdzić, że nie istnieje możliwość wylosowania takiej próby, aby korelacja była wyższa lub taka sama z jednoczesnym założeniem, że korelacja ta nie istnieje.
#     
# </div>

# <div class="alert alert-block alert-warning">
#     Przy wykorzystywaniu testów statystycznych ustala się pewien próg (<b>poziom istotności</b> $\alpha$ ). Jeżeli P-value jest mniejsze niż ten poziom można odrzucić hipotezę zerową. Zazwyczaj $\alpha$ = 0.05
#     
# *Przykład* W korelacji pomiędzy wartoscią domu, a zarobkami P-value < 0.05, więc odrzucamy hipotezę zerową.
# </div>

# <div class="alert alert-block alert-info">
#     
# Za pomocą P-value oraz poziomu istotności możemy odrzucić hipotezę zerową. Jednak, nie oznacza to, że potwierdzamy hipotezę alternatywną ($H_1$). 
# </div>

# Za pomocą testów statystycznych możemy także sprawdzić czy dane posiadają konkretny rozkład.
#  
# Sprawdźmy więc, czy ceny domów posiadaja rozkład normalny. Możemy to zrobić za pomocą metody <a href = https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html> *scipy.stats.normaltest()* </a>
# 
# Hipotezą zerową w tej metodzie jest załozenie, że dane mają rozkład normalny.
# 
# Przyjmijmy poziom istotności $\alpha$ = 0.05. 
# Jeżeli P-value < 0.05 Będziemy mogli odrzucić tą hipotezę i powiedzieć, że cena domów nie posiada rozkładu normalnego

# In[32]:


normaltest(df['MedHouseVal'])


# P-value = 0, a więc odrzucamy hipotezę, że dane posiadają rozkład normalny
# 

# Sprawdźmy co by jednak było, gdyby dane faktycznie były z rozkładu normalnego.
# 
# Wylosujmy 10 000 próbek z rozkładu normalnego
# 

# In[33]:


np.random.seed(42) #pozwala na wylosowanie zawsze tych samych danych
norm = [np.random.normal() for i in range(100000)]


# Spójrzmy na histogram 

# In[34]:


plt.hist(norm, bins = 20)


# A teraz sprawdźmy, czy za pomocą testu statystycznego możemy stwierdzić, że rozkład jest normalny

# In[35]:


normaltest(norm)


# P-value > 0.05, a więc nie możemy odrzucić hipotezy zerowej. Nie oznacza, to że rozkład musi być normalny, jednak jest to możliwe 

# Sprawdźmy czy cena domu posiada inny rozkład.
# 
# Metoda, która pozwala sprawdzić inne często występujące rozkłady to <a href = https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html> *scipy.stats.anderson()* </a>
# 
# Za jej pomocą możemy sprawdzić takie rozkłady jak:
# - norm - rozkład normalny
# - expon - rozkład wykładniczy
# - logistic - rozkład logistyczny
# - gumbel - rozkład Gumbela
# 

# In[36]:


for test in ['norm', 'expon', 'logistic', 'gumbel']:
    print('Sprawdzamy czy rozkład jest ', test)
    print( anderson(df['MedHouseVal'], dist = test))
    


# W dokumentacji danej metody można wyczytać, że jeżeli wartość testu (*statistic*) jest większa od wartości krytycznych (*critical_values*) to dane nie należą do testowanego rozkładu, co oznacza, że rozkład ceny domow nie należy do żadnego z wyżej wymienonych.

# W bibliotece *scipy.stats* można znaleść inne testy statystyczne.
