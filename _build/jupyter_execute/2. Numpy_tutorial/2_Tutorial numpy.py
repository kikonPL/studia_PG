#!/usr/bin/env python
# coding: utf-8

# # Pakiet NumPy - tutorial
# 
# ## Wstep
# **Czym jest pakiet numpy?**
# Pakiet NumPy jest pakietam wykorzystywanym do operacji numerycznych przeprowadzanych na wielowymiarowych obiektach zwanych arrayami. Pakiet ten jest bardzo wydajny i często wykorzystywany - zarówno przez początkujących jak i tych bardziej zaawansowanych. Dobrze współpracuje z pakietami takimi jak pandas, scipy, matplotlib, scikit-learn i wieloma innymi pakietami bazującymi na danych numerycznych.
# 
# **Czym jest array?**
# Array jest obiektem, na którym można wywoływać złożone operacje numeryczne. Z uwagi na wydajność obliczeń array'e sa często wykorzystywane. Swoją strukturą przypominają listę, lecz między tymi obiektami są znaczące różnice. Niektóre z nich zostały przedstawione poniżej.

# ## Różnica między działaniami na listach a numpy array
# 
# Standardowo Python nie potrafi robić obliczeń na listach:

# In[1]:


height = [1.82, 2.01, 1.68]
weight = [100, 89, 53]


# ```python
# weight / height ** 2 # to nie zadziala
# ```

# Jest na to sposób, który też jest powszechnie stosowany - List Comprehension.
# 
# Należy jednak pamiętać, że nie w każdym przypadku List Comprehension zastąpi nam funckje wbudowane w pakiet numpy. Ponadto nadmiar tej struktury wpływa negatywnie na wydajność kodu.

# In[2]:


[weight[i] / height[i] ** 2 for i in range(len(height))]


# Szybszym (i wydaje się, że bardziej przyjaznym) rozwiązaniem jest użycie pakietu Numpy. Dzięki temu obiektowi możemy definiować działania na całej array'i.

# In[3]:


import numpy as np


# In[4]:


np_height = np.array(height)
np_height


# In[5]:


np_weight = np.array(weight)
np_weight


# In[6]:


bmi = np_weight / np_height ** 2
bmi


# <div class="alert alert-block alert-danger">
# <b>Uwaga!</b> 
#     
# W numpy array można przechowywać dane tylko jednego typu (w listach można mieszać typy danych).
# </div>

# In[7]:


[0.2, 'll', True]


# In[8]:


np.array([0.2, 'll', True])


# <div class="alert alert-block alert-danger">
# <b>Uwaga!</b> 
#     
# Uwaga! Dodajac standardowe listy w Pythonie uzyskamy listę elementów wartości z obu list. Robiąc to samo na arrayach dodajemy odpowiednie elementy do siebie. Trzeba więc uważać na to co się robi.
# </div>

# In[9]:


python_list = [1, 2, 3]
numpy_array = np.array(python_list)


# In[10]:


python_list + python_list


# In[11]:


numpy_array + numpy_array


# In[12]:


# Chcac uzyskac efekt powiekszenia listy (jak na standardowych listach) wystarczy uzyc funkcji np.append
np.append(numpy_array, numpy_array)


# ## Operacje na arrayach
# ### Tworzenie Numpy Array

# In[13]:


# Definiowanie array'a na podstawie listy
np.array([1,2,3,4,5])


# In[14]:


# Funkcja np.ones tworzy array skladajacy sie z samych 1
np.ones(16)


# In[15]:


# Funkcja np.ones tworzy array skladajacy sie z samych 0
np.zeros(5)


# In[16]:


# Tworzenie pustej arrayi. Powinna dzialac najszybciej bo nie wymaga zmieniania wartosci liczb
# (wartosci sa wybierane na podstawie miejsca w pamieci gdzie sie znajduja)
np.empty(15, dtype=int)


# In[17]:


# Tworzenie macierzy jednostkowej
np.eye(3)


# In[18]:


# Definiowanie array'a z zadanego zakresu o okreslonej dlugosci
np.linspace(start=0, stop=5, num=21)


# In[19]:


# Definiowanie arraya z zadanego zakresu o okreslonej roznicy miedzy wyrazami
# Zauwaz ze wyraz 5.25 nie zostal juz dopisany
np.arange(start=0, stop=5.25, step=0.25)


# In[20]:


# Jesli nie poda sie parametru stop to array budowany jest od 0 do podanej liczby z zadanym krokiem
np.arange(5.25, step=0.25)


# In[21]:


# Mozna zdefiniowac typ danych przechowywanych w array'u
# Zwroc uwage ze nie definiujac parametru step jest on zdefiniowany jako 1
np.arange(10, dtype=complex)


# ### Filtrowanie elementow

# In[22]:


# Chcac stworzyc dwuwymiarowy array tworzymy array od zagniezdzonych list
np_2d = np.array([[1.1, 5, 3],
                  [4, 5, 6]])
np_2d


# In[23]:


# Wybieranie pierwszego wiersza
np_2d[0]


# In[24]:


# Wybieranie drugiej kolumny
np_2d[:, 1]


# In[25]:


# Wybranie wszystkich kolumn dla pierwszego wiersza (od 0 do 1 (bez 1))
np_2d[0:1, :]


# In[26]:


# pierwsza i druga kolumna od drugiego wiersza wlacznie
np_2d[1:,[0,1]]


# In[27]:


# Wybranie elementow spelniajacych okreslony warunek
np_2d[np_2d<5]


# In[28]:


# Czy ktorykolwiek element array'a jest rowny 5
np.any(np_2d==5)


# In[29]:


# Czy wszystkie elementy array'a sa rowne 5
np.all(np_2d==5)


# In[30]:


np_2d


# In[31]:


np_2d[np.nonzero(np_2d)]


# In[32]:


np_2d[np.where(np_2d<5)]


# ### Wywoływanie funkcji opisujących array
# W celu lepszego poznania obiektu, z którym pracujemy warto wywołać podstawowe funkcje mówiące nam o ilości danych, liczbie wymiarów, itp.
# 
# |funkcja|opis|
# |--|--|
# |ndim|zwraca liczbę wymiarów array'a|
# |shape|zwraca wymiar array'a|
# |size|zwraca liczbę elementów w array'u|
# |dtype|określenie typu danych w array'u|
# |unique|zwraca unikalne elementy w array'u|

# In[33]:


# Utworzmy 16elementowy array. Zauwaz, ze w pakiecie numpy mamy zdefiniowana liczbe pi.
pi_array = np.linspace(start=-2*np.pi, stop=2*np.pi, num=16)


# In[34]:


pi_array


# In[35]:


pi_array.ndim


# In[36]:


# Sprawdzanie wymiaru arraya
pi_array.shape


# In[37]:


# Sprawdzanie liczby elementow w arrayu
pi_array.size


# In[38]:


pi_array.dtype.name


# In[39]:


np.unique(pi_array)


# ### Wywoływanie określonych operacji matematycznych na arrayach
# 
# |funkcja|opis|
# |--|--|
# |reshape|zmienia wymiar array'a|
# |ravel|spłaszczanie array'a|
# |T|transponuje array|
# |@ lub dot|mnożenie macierzowe arrayi|
# | * , **/** , **+**,**-** |mnożenie/dzielenie/dodawanie/odejmowanie poszczególnych elementów między sobą|
# |sin (lub inne funkcje trygonometryczne)/exp/sqrt/log|działanie na array odpowiednia funkcja|
# 
# Stałe matematyczne:
# 
# |funkcja|stała|
# |---|--|
# |np.pi|liczba pi|
# |np.e|liczba e (zobacz także funkcję np.exp())|
# |np.inf|nieskonczoność|
# |np.nan|not a number|
# |np.ninf|minus nieskonczoność|
# |np.log()|funkcja logarytmiczna, np np.log(0)==-inf|

# In[40]:


# Wywolywanie funkcji na arrayu. Zauwaz, ze w pakiecie numpy wystepuja funkcje trygonometryczne
np.sin(pi_array)


# In[41]:


np.arcsin(np.sin(pi_array))


# In[42]:


# Funkcja reshape przeksztalca wymiar array'a do zadanego
# W tym przypadku array o wymiarze (16,) przeksztalcamy na array o wymiarze 4,4
A = np.ones(16).reshape(4,4)


# In[43]:


A


# In[44]:


B = np.arange(16).reshape(4,4)


# In[45]:


B


# In[46]:


# Funkcja splaszczajaca (zauwaz ze B nie zostalo nadpisane)
B.ravel()


# In[47]:


B.ndim


# In[48]:


# Porownywanie poszczegolnych elementow w arrayach
A==B


# In[49]:


# Mnozenie odpowiadajacych sobie wyrazow
A*B


# In[50]:


# Mnozenie macierzowe
A.dot(B)


# In[51]:


# Inny sposob mnozenia macierzowego
A@B


# In[52]:


# Transponowanie
B.T


# In[53]:


# Funkcja e^x. Chac uzyskac liczbe e wystarczy podac np.exp(1)
np.exp(B)


# In[54]:


# Pierwiastkowanie
np.sqrt(B)


# ### Sortowanie array'a

# In[55]:


# Stworzenie arrayi z 10 losowych liczb calkowitych z przedzialu [1,10]
to_sort = np.random.randint(1, 10, 10)


# In[56]:


to_sort


# In[57]:


np.sort(to_sort)


# Sortowanie malejaco

# In[58]:


-np.sort(-to_sort)


# In[59]:


# Funckja flip odwraca kolejnosc elementow w arrayi
np.flip(np.sort(to_sort))


# ### Łączenie i dzielenie array'i, dodawanie i usuwanie elementów
# 
# |funkcja|opis|
# |--|--|
# |vstack/hstack|wierszowe/kolumnowe skladanie dwoch arrayi|
# |vspli/hsplit|wierszowe/kolumnowe rozdzielanie dwoch arrayi|
# |append| dodawanie elementu na koniec arrayi|
# |insert|umieszczanie nowego elementu w arrayi w okreslonym miejscu|
# |delete|usuwanie elementow arrayi|

# In[60]:


# Wertykalne laczanie arrayi
np.vstack((A, B))


# In[61]:


# Horyzontalne laczanie array'i
np.hstack((A, B))


# In[62]:


# Podzial arrayi B na dwie mniejsze
np.hsplit(B, 2)


# In[63]:


# Dodawanie liczby do arraya
np.append(to_sort, 10)


# In[64]:


# Dodawanie listy do arraya
np.append(to_sort, [10, 15, 13])


# In[65]:


# Dodawanie arraya do arraya
np.append(to_sort, np.array([10, 15, 13]))


# In[66]:


# Umieszczenie na 5 miejscu w arrayu listy
np.insert(to_sort, 4, [1, 23])


# In[67]:


# Usuniecie drugiej kolumny w arrayu
np.delete(B, 1, axis=1)


# In[68]:


# Usuniecie drugiego i trzecicego wiersza
np.delete(B, [1, 2], axis=0)


# ### Podstawowe statystyki
# 
# |funkcja|opis|
# |--|--|
# |sum|zsumowanie elementow|
# |max|element maksymalny|
# |min|element minimalny|
# |mean|srednia|
# |sum|zsumowanie elementow|
# |std|odchylenie standardowe|
# |var|wariancja|
# |cov|macierz kowariancji|
# |cumsum|suma skumulowana|
# |argmax|zwraca **indeks** najwiekszego elementu|
# |argmin|zwraca **indeks** najmniejszego elementu|

# In[69]:


# Zsumowanie wszystkich elementow
B.sum()


# In[70]:


# Suma poszczegolnych kolumn
B.sum(axis=0)


# In[71]:


# Suma poszczegolnych wierszy
B.sum(axis=1)


# In[72]:


np.cumsum(B)


# In[73]:


to_sort


# In[74]:


np.argmax(to_sort)


# ## Zdjęcie jako wielowymiarowy array

# In[75]:


# Wczytajmy przykladowe zdjecie i sprawdzmy jego typ
from scipy import misc
raccoon = misc.face()
type(raccoon)


# <div class="alert alert-block alert-info">
# <b>Info!</b> 
#     
# Wczytując kolorowe zdjęcie (kolory w RGB) otrzymamy array wielkości (X, Y, 3). X odpowiada za wysokość obrazu (liczba wierszy w array'i),  Y odpowiada za szerokość (liczba kolumn w array'i), a 3 odpowiada za składowe RGB.
# </div>

# In[76]:


# Sprawdzmy liczbe wymiarow
raccoon.ndim


# In[77]:


# Sprawdzmy rozmiary (zdjecie ma rozmiar 768x1024 pixeli)
raccoon.shape


# In[78]:


# Liczba elementow (768*1024*3)
raccoon.size


# In[79]:


raccoon.dtype


# In[80]:


# Wyswietlenie pierwszego pixela
raccoon[0][0]


# In[81]:


# Wybranie drugiego wiersza.
# Analogicznie: raccoon[1, :, :] lub raccoon[1]
raccoon[1, ...]  


# In[82]:


# Iterowanie po pikselach
for column in raccoon[:5]:
    for rgb in column[:5]:
        print(rgb)


# In[83]:


# Sprawdzenie poprawnosci petli (wyciagniecie 5 pierwszych wierszy i 5 pierwszych kolumn)
raccoon[:5,:5]


# In[84]:


# Wyswietlenie arraya przy pomocy pakietu matplotlib
# Na jednym z kolejnych zajec bedzie wiecej o tym pakiecie i nie tylko :)
import matplotlib.pyplot as plt
plt.imshow(raccoon)


# In[85]:


# Znormalizowanie RGB (zmiana z przedzialu [0, 255] na [0, 1])
raccoon = raccoon / 255


# In[86]:


# Chac uzyskac zdjecie w skali szarosci trzeba pomnozyc macierzowo nasza macierz przez podana ponizej
raccoon_gray = raccoon @ [0.2126, 0.7152, 0.0722]


# In[87]:


raccoon_gray.shape


# In[88]:


plt.imshow(raccoon_gray, cmap="gray")


# ## Pandas DataFrame a Numpy Array
# Chcąc wykonywać obliczenia szybciej warto zamienić DataFrame na Array. Wtedy każdy wiersz arraya będzie odpowiadać za każdy wiersz ramki danych.

# In[89]:


import pandas as pd
df = pd.read_csv(filepath_or_buffer=r'mpg.csv', # sciezka do pliku
                sep=',', # separator
                header=0, # naglowek (nazwy kolumn)
                index_col=0 # kolumna z indeksem
                )


# In[90]:


df.head()


# In[91]:


df_array = df.to_numpy()


# In[92]:


df_array[:5]


# In[93]:


df.shape


# In[94]:


df_array.shape


# Porównajmy czas wykonania działań na trzech kolumnach

# In[95]:


from time import time


# In[96]:


start = time()
fun = lambda row:row['cylinders']*row['weight']**row['acceleration']
df.apply(fun, axis=1)
print(f'Czas wykonania: {time()-start}')


# In[97]:


start = time()
df_array[:,1]*df_array[:,4]**df_array[:,5]
print(f'Czas wykonania: {time()-start}')


# Jak widać już na niewielkim zbiorze danych proste obliczenia na arrayach wykonują się szybciej. Warto o tym pamiętać i to wykorzystywać.

# ## Podsumowanie
# 
# Jak widać NumPy jest pakietem, ktory zdecydowanie potrafi ułatwić życie. Jest często wykorzystywany do złożonych obliczeń. Ma zastosowania przy pracy z różnymi danymi - obrazami, ramkami danych, czy po prostu dużymi listami.
