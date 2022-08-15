import pandas as pd 
import argparse
import os 

# This script aims to create csv that will provide scores for calculating the weights 
# for RandomWeightedSampler so balance between positive and negative samples would be preserved. 

FILENAME_COL = "filename"
SCORE_COL = "score-status-combined" 
SCORE_TARGET_COL = "score"

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def fun(csv_path, root_path):
    df = pd.read_csv(csv_path)

    new_df = pd.DataFrame({FILENAME_COL: [], SCORE_TARGET_COL: []})
    for idx, row in df.sort_values(FILENAME_COL).iterrows():
        filename = row[FILENAME_COL]
        score = int(row[SCORE_COL])
        dir_name = os.path.join(root_path, filename)
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            files = os.listdir(dir_name)
            entries = [[os.path.join(filename, f), score] for f in files if is_image_file(f)]
            df_part = pd.DataFrame(entries, columns=[FILENAME_COL, SCORE_TARGET_COL])
            new_df = pd.concat([new_df, df_part])
    return new_df

def main(args):
    new_df = pd.concat([fun(args.csv_a, args.dataroot_a), fun(args.csv_b, args.dataroot_b)])
    new_df.to_csv(args.target, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parset.add_argument('--dataroot_a', required=True, help='path to cropped images root domain A')
    parset.add_argument('--dataroot_b', required=True, help='path to cropped images root domain B')
    parset.add_argument('--csv_a', required=True, help='path to csv of domain A')
    parset.add_argument('--csv_b', required=True, help='path to csv of domain B')
    parset.add_argument('--target', required=True, help='path to save csv with result')
    main(parser.parse_args())
    
