from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# from pandas import DataFrame as df

colnames = ['best',
            'population-1','population-2','population-3',
            'population-4','population-5','population-6',
            'population-7','population-8','population-9',
            'population-10','population-11','population-12',]

data = pd.read_csv(r'./Siskon_UAS/Images/Stresstest_rani.csv',names=colnames,header=0)
# data.to_csv('D:/EdukayshOn/UNAIR/Semester_8/SisKon_Lanjut/Siskon_UAS/Images/Stresstest_rani.csv')
nama = 'Rani'
fig, ax = plt.subplots()
print(data.iloc[2,:])
best = data.iloc[:,0]
rest = data.iloc[:,1:]
amt = len(best) # AMT of columns
row = len(data.iloc[:,0]) 

# Create 2D Mesh to help Plot Data
arr, wow = np.mgrid[0:amt-1:30j, 0:12:12j]
print(arr)
print(wow)
new = np.arange(amt)
print(best) # Best Iteration
print(rest) # Last Population (1-12)

plt.title(f'Kode {nama} Iteration Test')
plt.xticks()
plt.yticks([150, 160, 170])
plt.ylabel('Threshold Values')
plt.xlabel('Iterasi')
plt.plot(new,best)
plt.scatter(arr,rest,color='black')

plt.show()