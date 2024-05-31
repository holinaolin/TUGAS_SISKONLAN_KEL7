import numpy as np
import cv2
# import pandas as pd
from pandas import DataFrame as df
import xlsxwriter
# import random

def threshold_image(im,th):
    threshold_im = np.zeros(im.shape)
    threshold_im[im>th] = 1
    return threshold_im

def compute_otsu_criteria(im:np.ndarray,th):
    thresholded_im = threshold_image(im,th)
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1/nb_pixels
    weight0 = 1-weight1
    if weight1 == 0 or weight0 == 0:
        return np.inf
    val_pixels1 = im[thresholded_im==1]
    val_pixels0 = im[thresholded_im==0]
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    return weight0*var0 + weight1*var1


# path = "D:\SKRIPSI related\PY\gogogo\otsu_test.jpeg"
path = 'D:/EdukayshOn/UNAIR/Semester_8/SisKon_Lanjut/Siskon_UAS/Images/Praise THE SUN BG.jpg'
im = cv2.imread(path,0)
# cv2.imshow("original",im)
# cv2.waitKey(0)

thresholds = []
for t in range(256):
    value = [compute_otsu_criteria(im, t)]
    thresholds = np.append(thresholds, value, axis=0)
    # print(f'{t} Outputs {value}')
# print(thresholds)
thresholds = (np.array([thresholds, np.arange(256)])).T
print(f'Threshold:\n{thresholds}')

excel_file = df(thresholds)
excel_file.to_csv('D:/EdukayshOn/UNAIR/Semester_8/SisKon_Lanjut/Siskon_UAS/Images/OtsuImaging.csv')

excel_file = xlsxwriter.Workbook('D:/EdukayshOn/UNAIR/Semester_8/SisKon_Lanjut/Siskon_UAS/Images/OtsuImaging.xslx')
worksheet = excel_file.add_worksheet()
row = 0
for col, data in enumerate(thresholds):
    worksheet.write_column(row,col,data)

excel_file.close()

print("nuts")
thresholded_im = threshold_image(im,160)
cv2.imshow("thresholded",thresholded_im*255)
cv2.waitKey(0)
cv2.imwrite('D:/EdukayshOn/UNAIR/Semester_8/SisKon_Lanjut/Siskon_UAS/Images/Sun_Optimal.png', thresholded_im*255)