import shutil
import random
import pandas as pd
from openpyxl import load_workbook

dir = "/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4_3_AL4"
data = pd.read_excel('/home/ba4_project/ba4_Hee/Trypanosome/Prediction4_3_AL4.xlsx', sheet_name='Prediction4_3_AL4')
sheet = 'Dataset3_AL4'

pos_list = []
neg_list = []
row = 0
for image in data.Image:
    if data.Class[row] == "pos":
        pos_list.append(image)
    else:
        neg_list.append(image)
    row += 1

random_pos = random.sample(pos_list, k=100)
random_neg = random.sample(neg_list, k=100)

book = load_workbook('/home/ba4_project/ba4_Hee/Trypanosome/Check.xlsx')
writer = pd.ExcelWriter('/home/ba4_project/ba4_Hee/Trypanosome/Check.xlsx', engine='openpyxl')
writer.book = book

df = pd.DataFrame({'pos': random_pos, 'neg': random_neg})
df.to_excel(writer, sheet_name=sheet, index=False)
writer.save()

dstn_path = dir + '/check/pos'
for pos in random_pos:
    shutil.copy(pos, dstn_path)

dstn_path = dir + '/check/neg'
for neg in random_neg:
    shutil.copy(neg, dstn_path)
