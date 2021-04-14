import os
import glob
import pandas as pd
import numpy as np
import csv
from datetime import datetime as dt

file_extension = 'csv'


def change_tac_to_label():

    all_filenames = [i for i in glob.glob('*_clean_TAC.{}'.format(file_extension))]
    for filename in all_filenames:
        df = pd.read_csv(filename, delimiter=',')
        df['TAC_Reading'] = np.where(df['TAC_Reading'] >= 0.08, 'drunk', 'sober')
        new_filename = filename.replace(".csv", "") + "-labeled." + file_extension
        df.to_csv(new_filename, index=None)


def rename_columns():
    filenames = [i for i in glob.glob('*-accel.{}'.format(file_extension))]
    for filename in filenames:
        df = pd.read_csv(filename, delimiter=',')
        df = df.rename(columns={'TAC_Reading': 'label'})
        df = df.rename(columns={'timestamp': 'datetime'})
        df.to_csv(filename.replace(".csv", "") + "-renamed." + file_extension, index=None)


def separate_accel_by_pid():
    filename = "all_accelerometer_data_pids_13.csv"
    all_df = pd.read_csv(filename, delimiter=',')
    pids = all_df['pid'].unique()
    for index, row in all_df.iterrows():
        if row['pid'] == pids[0]:
            write_to_csv(row)
        elif row['pid'] == pids[1]:
            write_to_csv(row)
        elif row['pid'] == pids[2]:
            write_to_csv(row)
        elif row['pid'] == pids[3]:
            write_to_csv(row)
        elif row['pid'] == pids[4]:
            write_to_csv(row)
        elif row['pid'] == pids[5]:
            write_to_csv(row)
        elif row['pid'] == pids[6]:
            write_to_csv(row)
        elif row['pid'] == pids[7]:
            write_to_csv(row)
        elif row['pid'] == pids[8]:
            write_to_csv(row)
        elif row['pid'] == pids[9]:
            write_to_csv(row)
        elif row['pid'] == pids[10]:
            write_to_csv(row)
        elif row['pid'] == pids[11]:
            write_to_csv(row)
        elif row['pid'] == pids[12]:
            write_to_csv(row)


def add_accel_values_to_tac():
    all_labeled_filenames = [i for i in glob.glob('*-labeled.{}'.format(file_extension))]
    filename = "all_accelerometer_data_pids_13.csv"
    all_df = pd.read_csv(filename, delimiter=',').head(1000)
    first_item = all_labeled_filenames.pop()
    labeled_df = pd.read_csv(first_item, delimiter=',')
    for labeled_index, labeled_row in labeled_df.iterrows():
        for all_index, all_row in all_df.iterrows():
            if labeled_row['timestamp'] == all_row['time']:
                print(labeled_row)
                labeled_row['x'] = all_row['x']
                labeled_row['y'] = all_row['y']
                labeled_row['z'] = all_row['z']
    print(labeled_df)
    #    labeled_df.to_csv(labeled_file.replace(".csv", "")+"-accel."+file_extension, index=False)
    #    print("Done writing to"+labeled_file)


def convert_timestamp_datetime():
    clean_tac_labeled_filenames = [i for i in glob.glob('*_clean_TAC-labeled.{}'.format(file_extension))]
    for clean_tac_labeled_file in clean_tac_labeled_filenames:
        df = pd.read_csv(clean_tac_labeled_file, delimiter=',')
        for index, row in df.iterrows():
            timestamp = dt.fromtimestamp(int(row["timestamp"]))
            formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            df.loc[index, 'timestamp'] = formatted_time
        new_filename = clean_tac_labeled_file.replace(".csv", "") + "-timed." + file_extension
        df.to_csv(new_filename, index=False)


def split_by_10_seconds():
    print("Split by 10 seconds")


def write_to_csv(row):
    with open(row["pid"] + '-accel.csv', mode='a') as new_csv_file:
        writer = csv.writer(new_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([row["pid"], row["time"], row["x"], row["y"], row["z"]])


if __name__ == '__main__':
    os.chdir("datasets/accelerometer")
    # change_tac_to_label()
    add_accel_values_to_tac()
    # convert_timestamp_datetime()
