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


def rename_columns(file_postfix, old_column_name, new_column_name):
    filenames = [i for i in glob.glob('*' + file_postfix + '.{}'.format(file_extension))]
    for filename in filenames:
        df = pd.read_csv(filename, delimiter=',')
        df = df.rename(columns={old_column_name: new_column_name})
        df.to_csv(filename.replace(".csv", "") + "-renamed." + file_extension, index=None)


def separate_accel_by_pid():
    filename = "all_accelerometer_data_pids_13.csv"
    all_df = pd.read_csv(filename, delimiter=',')
    no_rows = all_df.shape[0]
    pids = all_df['pid'].unique()
    for index, row in all_df.iterrows():
        print(str(index) + "/" + str(no_rows))
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


def add_accel_values_to_tac(accel_file_name, tac_file_name):
    accel_df = pd.read_csv(accel_file_name, delimiter=',')
    tac_labeled_df = pd.read_csv(tac_file_name, delimiter=',')
    tac_labeled_no_rows = tac_labeled_df.shape[0]
    for index, tac_labeled_row in tac_labeled_df.iterrows():
        for index2, accel_row in accel_df.iterrows():
            if round(accel_row['timestamp'] / 1000) == tac_labeled_row['timestamp']:
                print("Timestamp match found!")
                tac_labeled_row['x'] = round(accel_row['x'] / 1000)
                tac_labeled_row['y'] = round(accel_row['y'] / 1000)
                tac_labeled_row['z'] = round(accel_row['z'] / 1000)
        print(str(index) + " / " + str(tac_labeled_no_rows))
    tac_labeled_df.to_csv(tac_file_name.replace(".csv", "") + "-acceled." + file_extension, index=False)


def convert_timestamp_datetime(file_name_postfix):
    filenames = [i for i in glob.glob('*_clean_TAC-labeled-renamed-renamed.{}'.format(file_extension))]
    for file_name in filenames:
        df = pd.read_csv(file_name, delimiter=',')
        no_rows = df.shape[0]
        for index, row in df.iterrows():
            print(str(index) + " / " + str(no_rows))
            try:
                timestamp = dt.fromtimestamp(int(row["datetime"]))
                formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                df.loc[index, 'timestamp'] = formatted_time
            except ValueError:
                print("Value Error occurred!")
                continue
        new_filename = file_name.replace(".csv", "") + file_name_postfix + "." + file_extension
        df.to_csv(new_filename, index=False)
        print("Done writing to " + new_filename)


def add_column_names():
    all_accel_filenames = [i for i in glob.glob('*-accel.{}'.format(file_extension))]
    for filename in all_accel_filenames:
        df = pd.read_csv(filename, header=None)
        df.rename(columns={0: 'pid', 1: 'timestamp', 2: 'x', 3: 'y', 4: 'z'}, inplace=True)
        df.to_csv(filename, index=False)


def write_to_csv(row):
    with open(row["pid"] + '-accel.csv', mode='a') as new_csv_file:
        writer = csv.writer(new_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([row["pid"], row["time"], row["x"], row["y"], row["z"]])


def merge_csv_files(category):
    print(os.getcwd())
    os.chdir("accel-labeled/" + category + "/feature-gen")

    all_filenames = [i for i in glob.glob('*.{}'.format(file_extension))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    print(combined_csv.shape[0])
    combined_csv.to_csv("combined-" + category + ".csv", index=False)
    os.chdir('./../../..')
    print(os.getcwd())


def merge_and_shuffle():
    os.chdir('accel-labeled/')
    all_filenames = [i for i in glob.glob('combined-*.{}'.format(file_extension))]
    combined_df = pd.concat([pd.read_csv(f) for f in all_filenames])
    print("Before : ")
    print(combined_df.head(10))
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    print("After : ")
    print(combined_df.head(10))

    combined_df.to_csv("combined-all.csv", index=False)


def check_for_duplicates(category):
    os.chdir("accel-labeled/" + category + '/feature-gen/')
    df = pd.read_csv("combined-" + category + ".csv")
    print("Before : " + str(df.shape[0]))
    df.drop_duplicates(subset=['x_mean', 'x_median', 'x_stdev', 'x_raw_min', 'x_raw_max', 'x_abs_min', 'x_abs_max',
                               'y_mean', 'y_median', 'y_stdev', 'y_raw_min', 'y_raw_max', 'y_abs_min', 'y_abs_max',
                               'z_mean', 'z_median', 'z_stdev', 'z_raw_min', 'z_raw_max', 'z_abs_min', 'z_abs_max'],
                       inplace=True)
    print("After : " + str(df.shape[0]))
    df.to_csv("dedup-" + category + ".csv", index=False)
    os.chdir('./../../..')


def count_no_rows(category):
    os.chdir("accel-labeled/" + category + '/feature-gen')
    df = pd.read_csv('combined-' + category + '.csv', delimiter=',')
    total_no_rows = df.shape[0]
    print("Full Total : " + str(total_no_rows))


def change_string_label_to_int():
    os.chdir('accel-labeled/')
    df = pd.read_csv("combined-all.csv", delimiter=',')
    df['label'] = np.where(df['label'] == "sober", 0, 1)
    df.to_csv("combined-all-labeled.csv", index=None)


if __name__ == '__main__':
    os.chdir("datasets/accelerometer")
    # change_tac_to_label()
    # convert_timestamp_datetime("-at")
    # separate_accel_by_pid()
    add_accel_values_to_tac("BK7610-accel.csv", "BK7610_clean_TAC-labeled.csv")
    # add_column_names()

    # renaming columns names
    # rename_columns("_clean_TAC-labeled", "timestamp", "datetime")
    # rename_columns("-accel", "timestamp", "datetime")
    # rename_columns("_clean_TAC-labeled-renamed", "TAC_Reading", "label")
    # merge_csv_files("drunk")
    # merge_csv_files("sober")

    # check_for_duplicates("drunk")
    # check_for_duplicates("sober")
    # count_no_rows("drunk")
    # count_no_rows("sober")
    # merge_and_shuffle()
    change_string_label_to_int()