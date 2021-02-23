import pandas as pd
import shutil

data = pd.read_excel('/home/ba4_project/ba4_Hee/Trypanosome/Prediction4_2_AL3.xlsx', sheet_name='Random')
print('Dataframe created')

row = 0
for image in data.Image:
    if data.Check[row] == "T":
        if data.Class[row] == "pos":
            dstn_path = '/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4_2_AL4/img/pos'
            shutil.copy(image, dstn_path)
        else:
            dstn_path = '/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4_2_AL4/img/neg'
            shutil.copy(image, dstn_path)
    elif data.Check[row] == "F":
        if data.Class[row] == "neg":
            dstn_path = '/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4_2_AL4/img/pos'
            shutil.copy(image, dstn_path)
        else:
            dstn_path = '/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4_2_AL4/img/neg'
            shutil.copy(image, dstn_path)
    else:
        dstn_path = '/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4_2_AL4/unlabeled/img'
        shutil.copy(image, dstn_path)

    row += 1

print('Redistribution finished')
