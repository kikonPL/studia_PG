#!/usr/bin/env python
# coding: utf-8

# # Wykrywanie anomalii - tutorial 
# ## Wstęp
# Badając dane często mozemy spotkać sie z elementami zbioru danych które róznią się od ogółu danych. Takie przypadki mogą mieć znaczący i zróznicowany wpływ na wnioski jakie wynosimy z danych. Tego typu zjawsiko moze mieć także rózne przyczyny od zwizanych z rzeczywistym rozkładem badanego zjawiska po błedy ludzkie czy tez maszynowe zwiazane z odczytywnaniem czy przekształcaniem danych. Badanie tego typu zjawisk nosi nazwę **wykrywywania anomalii** (ang. *outlier detection*)
# 
# Przykładowymi outlierami mogą być na przykład hity, czyli stylokolory sprzedające sie duzo szybciej od innych. Innym przykładem outlierów mogą być na przykład podjerzane transakcje kartami kredytowymi - nadanie tego typu zdarzeń nosi angielską nazwę *fraud detection*
# <div class="alert alert-block alert-success">
# <b>Definicja</b> 
#     
# Objekt w zbiorze danych nazywamy obserwacją odstającą (outlierem) jeżeli spełnia nastepujące warunki 
#     <li>Odchyla się od normalnego/znanego zachowania danych</li>
#     <li>Przyjmuje wartości dalekie od oczekiwanych bądź też średnich</li>
#     <li>Jego charakterystyka nie jest połączona z, ani podobna do, żadnego innego objektu</li>
# </div>
# 
# Możemy więc powiedzieć że outlier charakteryzuje sie pewnym niezwyczajnymi cechami w zbiorze danych. Wspomniana niezwyczajność nie ma charakteru wartościujacego, poniewaz anomalie mogą być zarówno zjawiskami korzystnymi jak i nie. Anomalią może byc bardzo dobrze sprzedający sie stylokolor, moze też nia być słabo sprzedający salon.

# <div class="alert alert-block alert-info">
# <b>Info</b> 
# 
# To czym jest a czym nie jest wartość odstająca nie ma ścisłej matematycznej definicji. 
# To co jest albo nie outlierem w analizie musi opierać się na biznesowym zrozumieniu danych i zadania przed którym stoimy.
# </div>

# Spróbujmy złapać intuicję na temat tego czym sa elementy odstajace za pomoca prostego syntetycznego przykładu. Na początku zaimportujmy potrzebne pakiety

# In[1]:


# importy
import numpy as np

import plotly.graph_objects as go # pakiet plotly służy do interaktywnych wizualizacji

from plotly.offline import plot

from IPython.display import IFrame, display, HTML


# Wygenerujmy losowe dane 
# 
# <div class="alert alert-block alert-danger">
# <b>Uwaga!</b> 
#     
# Nie polecam używania funkcji <i>np.random.seed</i> do generowania liczb losowych. 
# Ten sposób ustalenia ziarna spowoduje że każde wywołanie tej samej funkcji losującej zwróci identyczny wynik.
# Przy wielokrotnym wywoływaniu w tym samym skrypcie funkcji losującej to nie jest zachowanie którego oczekujemy. 
# Użycie default_rng zapewni powtarzalne wyniki a pozwoli uniknąć wspomnianego problemu
# </div>

# In[2]:


rng = np.random.default_rng(42) # ustawiamy ziarno w celu uzyskanie reprodukowalności


# In[3]:


mu, sigma = 0, .05 # ustalmy wartość oczekiwana i odchylenie standardowe dla rozkładu normalnego
low, high = .35, .65 # ustalamy minmum i maskimum dla rozkładu jednostajnego

x_noise = rng.normal(mu, sigma, 50) # losujemy 50 przykładów szumu
y_noise = rng.normal(mu, sigma, 50) # losujemy 50 przykładów szumu
x_line = rng.uniform(low, high, 50) # losujemy punkty dla prostej y = x
y_line = x_line
x_data = x_line + x_noise # dodajemy szum do prostej
y_data = y_line + y_noise 
outlier_x = [.95] # dodajemy outliera
outlier_y = [.1]


# Tworzymy wykres z wygenerowanymi danymi

# In[4]:


fig = go.Figure(data=[go.Scatter(x=x_data, y=y_data, mode='markers', name='Regularne przypadki'),
                      go.Scatter(
    x=outlier_x, y=outlier_y, mode='markers', name='Outlier')],
    layout=go.Layout(xaxis={'range': [0, 1], 'title': 'Cecha 1', 'tickformat': ',.2f'},
                     yaxis={
        'range': [0, 1], 'title': 'Cecha 2', 'tickformat': ',.2f'},
    title={
        'text': 'Syntetyczny przykład wartości odstających', 'x': 0.5},
    legend={'yanchor': 'top',
            'y': 0.99,
            'xanchor': 'left',
            'x': 0.01
            })) # tworzymy wykres punktowy wygenerowanych punktów
fig.add_annotation(x=.95, y=.1,
                   text='Stąd outlierzy wyszli',
                   showarrow=True,
                   arrowhead=1) # dodajemy adnotacje
fig.show() # wyświetlamy wykres


# Na wykresie widzimy że punkt zaznaczony na czerwono wyraźnie odstaje od punktów w kolorze niebieskim. Ten własnie punkt jest wartością odstajacą. Takich wartości czesto może byc więcej niz 1. Na przykład gdy mówimy o sprzedaży stylokolorów, więcej niz jeden może sprzedawać sie znacznie lepiej niż reszta stylokolorów i przez to być wartością odstającą.

# Po zdefiniowaniu co rozumiemy jako elementy odstające, przejdziemy do sprawdzenia w jaki sposób możemy takie obiekty odnaleźć w badanych danych. Przyjrzyjmy się zatem  więc sposobom odnajdywania outlierów ze wzgledu na typy.

# ## Wykres pudełkowy
# Pierwszym sposobem odnajdywania outlierów z którym się zapoznamy jest wykres pudełkowy (ang. *box-and-whisker plot*). Pozwala ona zwizualizowac jednocześnie statystyki opisowe danych, które badamy jak i wartości odstające, będąc wygodnym narzedziem do badania rozkładu danych które nas interesują. Do stworzenia wykresów wykorzystamy wygenerowane uprzednio syntetyczne dane.

# In[5]:


import pandas as pd
df = pd.DataFrame({'Cecha 1': np.concatenate((outlier_x, x_data)),
                   'Cecha 2': np.concatenate((outlier_y, y_data))})
df_melted = df.melt(var_name='Cecha', value_name= 'Wartość')  # przygotowywyujemy dane do wykresu


# Na wykresie poniżej widzimy przykłady wykresów pudełkowych stworzonych na stworzonych przez nas danych. Zwrócmy uwagę na niebieskie i czerwone pudełka, od których pochodzi nazwa wykresu. Rozciągaja się one od wartości pierwszego kwartyla oznaczanej przez Q<sub>1</sub> do wartości trzeciego kwartyla Q<sub>3</sub>. Linia przecinająca pudełko równolegle do osi x oznacza medianę wartości zmiennej w danych.
# 
# Następnie widzimy wysuwające się w góre i dół "wąsy". Górny wąs zaczyna sie od wartości 3 kwartyla kończąć się na najwiekszej wartości cechy mieszącej się w przedziale (Q<sub>3</sub>,Q<sub>3</sub>+ 1.5(Q<sub>3</sub> - Q<sub>1</sub>)] . Odpowiednio dolny wąs ciągnie się od najmniejszej wartości mieszczącej się w przedziale [Q<sub>1</sub>- 1.5(Q<sub>3</sub> - Q<sub>1</sub>), Q<sub>1</sub>)  do wartości pierwszego kwartyla.
# 
# Oprócz wspomnianych elementów na wykresie pudełkowym dla obu z cech widzimy po jednym punkcie nie mieszczącym się w zakresie wyznaczanym przez końce górnego i dolnego wiersza. Dla cechy 1 (niebieski wykres) jest to wartość 0.95, z kolei dla cechy 2 (czerwony wykres) jest to wartość 0.1. Te punkty to wartości odstające.

# In[6]:


import plotly.express as px
fig = px.box(df_melted, y='Wartość', facet_col='Cecha', color='Cecha',
             boxmode='overlay', )  # tworzymy wykres pudełkowy
fig.add_annotation(x=.25, y=.498,
                   text='Mediana',
                   showarrow=True,
                   ax=50,
                   ay=-25,
                   arrowhead=1)  # dodajemy adnotacje
fig.add_annotation(x=.25, y=.586,
                   text='Trzeci kwartyl',
                   showarrow=True,
                   ax=50,
                   ay=-25,
                   arrowhead=1)  
fig.add_annotation(x=.25, y=.424,
                   text='Pierwszy kwartyl',
                   showarrow=True,
                   ax=50,
                   ay=25,
                   arrowhead=1)  
fig.add_annotation(x=.125, y=.668,
                   text='Górny wąs',
                   showarrow=True,
                   ax=50,
                   ay=-25,
                   arrowhead=1)  
fig.add_annotation(x=.125, y=.316,
                   text='Dolny wąs',
                   showarrow=True,
                   ax=50,
                   ay=25,
                   arrowhead=1)  
fig.add_annotation(x=0, y=.95,
                   text='Wartość odstająca',
                   showarrow=True,
                   ax=50,
                   ay=25,
                   arrowhead=1)  

fig.show() # wyświetlamy wykres


# Wykres pudełkowy pozwala nam w prosty sposób przeanalizować dane i poznać statystyki opisowe takie jak mediana czy wartosci innych kwartyli które charakteryzują badane dane w jasny i łatwy do zrozumienia sposób. Istnieją także inne niż przedstawiony tutaj tradycyjny wykres pudełkowy takie jak na przykład wcięty wykres pudełkowy (ang. *notched box plot*), które mając wcięcia przy medianie obrazując jej przedział ufności.

# <div class="alert alert-block alert-danger">
# <b>Uwaga!</b> 
#     
#    Wykresy pudełkowe nie są pozbawione wad jeśli chodzi o wyznaczanie wartości odstajacych. Naczelnym problemem jest brak możliwości uwzglednia relacji między róznymi badanymi cechami. Jak widzimy na wykresie każdy wykres reprezentuje wartości tylko dla jednej zmiennej. Jest to poważne uproszczenie. Kolejna wadą wykresów pudełkowych jest brak elastyczności przy ustalaniu czym w naszym zadaniu jest wartość odstająca, co utrudnia uwzględnienie naszych założeń biznesowych
# </div>

# ## Las izolacji
# Las izolacji jest metodą wykorzystująca uczenie zespołowe oraz drzewa decyzyjne do wykrywania wartości odstajacych, polegającą na użyciu zespołu wielu drzew izolujących.
# 
# ### Czym jest las izolacji
# 
# Metoda lasu izolującego wykorzystuje dwie ilościowe własności anomalii, mianowicie
# - Niewielką ilość : anomalie stanowią mniejszość przypadków
# - Odmienność: anomalie mają wartości atrybutów diametralnie różne od normalnych przypadków
# 
# Te właśnie cechy sprawiaja że elementy o wartościach odstających są podatne na izolację. W celu wyizolowania elementu, rekurencyjnie dzielimy przypadki aż do wyizolowania ostatniego elementu. Podziały sa generowane przez losowy wybór atrybutu i losowy wybór wartości podziału mieszczącej się pomiędzy najmniejszą i największą wartością wybraną atrybutu. Ponieważ rekurencyjne podziały mogę być reprezentowane za pomocą struktury drzewa, ilość partycji potrzebnych od izolacji punktu jest odpowiednia długości ścieżki od korzenia drzewa do jego końcowego wierzchołka. 
# 
# ![title](1_a.png)
# 
# Intuicyjnie dla anomalii ścięzki będą krótsze niz dla normalnych elementów, ponieważ elementy odstające są łatwiejsze do odzielenia od elementów normalnych, tak jak widzimy to na rysunku powyżej. Element oznaczony na czerwono odróżniający jest dość łatwy to oddzielenia, to jest przeprowadzenia takich cięć aby został jedynym w weźle. Dla elementu zaznaczonego na zielono znajdującego się wśród innych elemntów jest to trudniejsze.
# 
# 
# 

# ### Implementacja lasu izolacji (ang. *isolation forest*)

# Algorytm lasu izolacji podobnie jak inne najpopularniejsze algorytmy jest zaimplementowany w pythonowej bibliotece PyOD. Tej właśnie biblioteki użyjemy aby zaprezentować działanie lasów izolacji. 

# <div class="alert alert-block alert-info">
# <b>Info</b> 
# 
# Dokumentacje popularnego pythonowego pakietu PyOD służącego detekcji wartości odstających znajdziemy na stronie https://pyod.readthedocs.io/en/latest/ 
# </div>

# Przyjrzyjmy się jeszcze raz wygenerowanym uprzednio danym, tym razem w formie tabularycznej:

# In[7]:


with pd.option_context('display.float_format', '{:,.2f}'.format):
    display(df.head())


# Przeprowadzimy na naszych danych wyszukiwanie elementów odstajacych za pomocą lasu izolacji

# Las izolacji jest zaimplementowany w klasie IForest zawartej w pakiecie PyOD. Inicjalizują istancje owej klasy mozemy zdefiniować m.in następujące parametry
# - **n_estimators** - ilość estymatorów (drzew izolacji) w zespole, domyślnie 100
# - **max_samples** - ilość próbek losowanych do uczenia pojedynczego drzewa, domyślnie max(256, \#X)
# - **contamination** - proporcja outlierów w zbiorze danych, domyślnie 0.1
# - **max_features** - ilość cech użytych do uczenia kazdego z drzew, domyślnie 1 (czyli wszystkie)
# - **bootstrap** - czy losowanie elementów do próbki z powtórzeniami czy nie, domyslnie False

# In[8]:


from pyod.models.iforest import IForest  # importujemy klasę lasu izolacji
isolation_forest = IForest(n_estimators=50,  # zmniejszamy ilość estymatorów
                           contamination=.02,  # ustawiamy kontaminacje outlierami na 2%
                           random_state=21)
isolation_forest.fit(df)  # trenujemy model na naszych danych
# prognozujemy prawdopodobieństwo że mamy do czynienia z outlierem
preds = isolation_forest.predict_proba(df)[:,1]
df['outlier_probability'] =  np.round(preds, 2) # zaokrąglamy predykcje do 2 miejsca po przecinku


# Uzyskane wyniki outlierów są skalowane za pomocą przekształcenia min-max w celu uzyskania prawdopodbieństw że dany element jest wartością odstającą. Zwizualizujmy zatem rezultaty tego przekształcenia, w celu weryfikacji czy las izolacji w rzeczywiści odnalazł wartość odstającą

# In[9]:


fig = px.scatter(df, x = 'Cecha 1', y = 'Cecha 2', color= 'outlier_probability', color_continuous_scale='Bluered',
                title='Wyniki wykrywania anomalii dla lasu izolacji')
fig.update_xaxes(range=[0, 1], tickformat = ',.2f')
fig.update_yaxes(range=[0, 1], tickformat = ',.2f')
fig.show()


# Widzimy na wykresie, że wyraźnie odznacza się na czerwono dodana przez nas wartość odstająca. Regularna część danych ma kolory od wyraziście niebieskiego w samym środku skupienia po fioletowy na jego brzegach. Udało nam się zidentyfikować wartość odstającą dzięki algorytmowi lasu izolującego.

# Zobaczmy jak działa stworzony przez nas algorytm na nowych, niewidzianych przez model danych. Predykcja o wartości 1 oznacza outliera, natomiast 0 oznacza brak wartości odstającej.

# In[10]:


print(
    f'''Dla punktu (0,1) algorytm lasu izolacji zwraca wynik {isolation_forest.predict([[0, 1]])},
    natomiast dla punktu (0.5, 0.5) otrzymujemy prognozę {isolation_forest.predict([[.5, .5]])}''')


# ## Local Outlier Factor (LOF)

# Local Outlier Factor (LOF) jest algorytmem wykrywania anomalii, który opiera sie na porównaniu gęstości rozkładu dla danego punktu z gęstością rozkładu dla punktów z nim sąsiadujących. Wykorzystujemy tutaj do wyznaczenia wartości odstających ich następującą cechę, tego typu obiekty znajdują się w obszarach o obniżonej gęstości. Algorytm LOF porównuje gęstość  dla sprawdzanego przypadku do gęstości dla otaczających go najbliszych sąsiadów, stąd właśnie lokalność w nazwie. 

# Stworzymy funkcje wyliczającą dystans osiąganlości rd, dla znanego N<sub>k</sub>

# In[11]:


import numpy as np


def absolute_distance(a, b):
    '''Funkcja zwraca dystans absolutny między 2 wektorami'''
    diff = np.abs(np.array(a)-np.array(b)
                  )  # liczymy absolutne dystanse dla każdego z wymiarów
    dist = np.sum(diff)  # sumujemy
    return dist


def reachability_distance(a, b, nk_b, metric=absolute_distance):
    '''Funkcja wylicza dystans dla a względem b'''
    rd = max(metric(a, b), nk_b)
    return rd


# Zauważmy że wartość N<sub>k</sub> może być różna dla różnych punktów należących do X. Spróbujmy wywołać naszą fukcje dla różnych wartości 

# In[12]:


print(reachability_distance([-1, -2], [2, 3], 1)) # a = (-1, -2), b = (2, 3),  nk_b =1 


# In[13]:


print(reachability_distance([-1, -2], [2, 3], 10)) # a = (-1, -2), b = (2, 3),  nk_b = 10


# ### Implementacja Local Outlier Factor
# 
# Ponownie użyjemy blibioteki PyOD która oferuje bardzo szeroki wybór metod detekcji anomalii. Będziemy pracowali na tym samym zbiorze danych co uprzednio. Spójrzmy więc nań po raz kolejny.

# In[14]:


# usuwamy wyniki  działania lasu izolacji z ramki danych
df.drop(columns=['outlier_probability'], inplace=True)
with pd.option_context('display.float_format', '{:,.2f}'.format):
    display(df.head())


# LocalOutlierFactor został  zaimplementowany w klasie LOF, która możemy znaleźć PyOD. Gdy inicjalizujemy instancje tej klasy mozemy zdefiniować m.in następujące parametry
# - **n_neighbors** - ilość najbliższych sąsiadów k, domyślnie 20
# - **p** - parametr Metryki Minkowskiego, 1 to Manhattan, 2 Euklidesowa, domyslnie 2
# - **contamination** - proporcja outlierów w zbiorze danych, domyślnie 0.1
# 

# In[15]:


from pyod.models.lof import LOF  # importujemy klasę lasu izolacji
lof = LOF(n_neighbors=8,  # zmniejszamy ilość sąsiadów
          contamination=.02,  # ustawiamy kontaminacje outlierami na 2%
          )
lof.fit(df)  # trenujemy model na naszych danych
# prognozujemy prawdopodobieństwo że mamy do czynienia z outlierem
preds = lof.predict_proba(df)[:, 1]
# zaokrąglamy predykcje do 2 miejsca po przecinku
df['outlier_probability'] = np.round(preds, 2)


# In[16]:


fig = px.scatter(df, x = 'Cecha 1', y = 'Cecha 2', color= 'outlier_probability', color_continuous_scale='Bluered',
                title='Wyniki wykrywania anomalii dla LOF')
fig.update_xaxes(range=[0, 1], tickformat = ',.2f')
fig.update_yaxes(range=[0, 1], tickformat = ',.2f')
fig.show()


# In[17]:


df.head()


# <div class="alert alert-block alert-info">
# <b>Info</b> 
# 
# Wykrywanie nowinek kontra wykrywanie wartości odstających : 
#     W wykrywaniu wartości odstających (ang. <i>outlier detection</i>) mamy na celu wykrecie anomalii istniejących w zbiorze treningowym.
#     W wykrywaniu nowinek (ang. <i>novelty detection</i>)trenujmy algorytm na danych niezawierających anomalii i robimy prognozy  na przyszłych, niewidzianych danych czy mają charakter anomalii.
#     
# </div>

# Local Outlier Factor domyślnie służy do wykrywania wartości odstajacych, ale przy odpowiednim ustawieniu parametru *novelty*,
# możemy go użyć do wykrywania nowinek

# In[18]:


df = df.drop(columns = ['outlier_probability'])
lof_novelty = LOF(n_neighbors=7,  # zmniejszamy ilość sąsiadów
          novelty=True,  # ustawiamy kontaminacje outlierami na 2%
          contamination=.02,
          )
lof_novelty.fit(df)


# In[19]:


print(
    f'''Dla punktu (0,1) algorytm lasu izolacji zwraca wynik {lof_novelty.predict([[0, 1]])},
    natomiast dla punktu (0.5, 0.5) otrzymujemy prognozę {lof_novelty.predict([[.5, .5]])}''')


# ## Podsumowanie
# W tutorialu udało nam się wprowadzić tematykę związana z wykrywaniem anomalii w data science. 
# Metod, które zostały tutaj wspomniane należy uzywac świadomie rozumiejąć zarówno ich działanie jak i biznesowe cele do których chemy użyć. Ważna kwestią jest tutaj pytanie czy usuwać wartości odstające ze zbioru treningowego przed uczeniem modeli regresyjnych czy tez klasyfikacyjnych. Aby odpowiedzieć na to pytanie musimy zrozumieć pochodzenie wartości odstającej. Wtedy możemy podjąć decyzję, możemy usunąć anomalię, na przykład jesli jest efektem źle wpisanych danych, czy błednego odczytu. Możemy anomalię zostawić w naszym zbiorze trenignowym, albo możemy zbudować kaskadę modeli w którym jeden będzie klasyfikował objekty jako anomalie lub nie, a następnie dwa kolejne modele które obsługiwałyby osobno odstające i regularne wartości. Mozliwości jest wiele, a najlepszym sposobem na rostrzygnięcie ich właściwości jest empiryczna weryfikacja.
