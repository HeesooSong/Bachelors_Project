import split_folders

org_path = '/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4_3_AL4/img'
output_path = '/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4_3_AL4'
split_folders.ratio(org_path, output=output_path, ratio=(.7, .3))
