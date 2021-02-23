import pandas as pd
import shutil

data = pd.read_excel('/home/ba4_project/ba4_Hee/Trypanosome/Prediction4_3_AL3.xlsx', sheet_name='unlabeled')
print('Dataframe created')

for image in data.Image:
    dstn_path = '/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4_3_AL4/unlabeled/img'
    shutil.copy(image, dstn_path)

print('Redistribution finished')
