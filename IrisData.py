import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
fin = pd.read_csv('https://archive.ics.uci.edu/ml/'
                  'machine-learning-databases/iris/iris.data',
                  header=None)
fin.tail()

x = [1,2,3,4,5]
y = fin.iloc[0:100,4].values #ดึงชชื่อดอกไม้มากอันดับ 0->100 ดึงตำแหน่งที่4ออกมา
#print(y)
y = np.where(y == 'Iris-setosa',-1,1) #เช็คค่าที่ดึงมา ถ้าชื่อ Iris-setosa ให้ค่าเป็น -1 ไม่ใช่ให้เป็น 1 (Iris-versicolor)
#print(y)
X = fin.iloc[0:100,[0,2]].values
print(X)
#ใส่ค่าของ setosa<ตำแหน่่ง 0->50>
plt.scatter(X[:50,0],X[0:50,1],
            color='red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],
            color='blue',marker='x',label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='lower right')
plt.show()
