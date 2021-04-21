import glob
import os
import pandas as pd
import statistics


def gen_features_per_pid(file_name, label):
    df = pd.read_csv(file_name, delimiter=',')
    df['dt'] = pd.to_datetime(df['datetime'], unit='s')

    rows = []
    column_names = ['group_timestamp', 'label',
                        'x_mean', 'x_median', 'x_stdev', 'x_raw_min', 'x_raw_max', 'x_abs_min', 'x_abs_max',
                        'y_mean', 'y_median', 'y_stdev', 'y_raw_min', 'y_raw_max', 'y_abs_min', 'y_abs_max',
                        'z_mean', 'z_median', 'z_stdev', 'z_raw_min', 'z_raw_max', 'z_abs_min', 'z_abs_max' ]

    for group_name, g in df.groupby(pd.Grouper(freq='10s', key='dt')):
        print(f'Start time {group_name} has {len(g)} records within 10 secs')
        row = []
        group_timestamp = group_name
        label = label

        if len(g) < 50:
            continue
        else:
            x = g['x'].head(50)
            x_mean = x.mean()
            x_median = x.median()
            x_std_dev = statistics.stdev(x)
            x_raw_min = min(x)
            x_raw_max = max(x)
            x_abs_min = min(abs(x))
            x_abs_max = max(abs(x))

            # print(
            #     f'Mean : {x_mean}, Median : {x_median}, Stdev : {x_std_dev}, '
            #     f'X raw Min : {x_raw_min}, X raw Max : {x_raw_max}, '
            #     f'X abs Min : {x_abs_min}, X abs Max : {x_abs_max}'
            # )

            y = g['y'].head(50)
            y_mean = y.mean()
            y_median = y.median()
            y_std_dev = statistics.stdev(y)
            y_raw_min = min(y)
            y_raw_max = max(y)
            y_abs_min = min(abs(y))
            y_abs_max = max(abs(y))

            # print(
            #     f'Mean : {y_mean}, Median : {y_median}, Std dev : {y_std_dev}, '
            #     f'X raw Min : {y_raw_min}, X raw Max : {y_raw_max}, '
            #     f'X abs Min : {y_abs_min}, X abs Max : {y_abs_max}'
            # )

            z = g['z'].head(50)
            z_mean = z.mean()
            z_median = z.median()
            z_std_dev = statistics.stdev(z)
            z_raw_min = min(z)
            z_raw_max = max(z)
            z_abs_min = min(abs(z))
            z_abs_max = max(abs(z))

            # print(
            #     f'Mean : {z_mean}, Median : {z_median}, Std dev : {z_std_dev}, '
            #     f'X raw Min : {z_raw_min}, X raw Max : {z_raw_max}, '
            #     f'X abs Min : {z_abs_min}, X abs Max : {z_abs_max}'
            # )
            row.append(group_timestamp)
            row.append(label)

            row.append(x_mean)
            row.append(x_median)
            row.append(x_std_dev)
            row.append(x_raw_min)
            row.append(x_raw_max)
            row.append(x_abs_min)
            row.append(x_abs_max)

            row.append(y_mean)
            row.append(y_median)
            row.append(y_std_dev)
            row.append(y_raw_min)
            row.append(y_raw_max)
            row.append(y_abs_min)
            row.append(y_abs_max)

            row.append(z_mean)
            row.append(z_median)
            row.append(z_std_dev)
            row.append(z_raw_min)
            row.append(z_raw_max)
            row.append(z_abs_min)
            row.append(z_abs_max)

            rows.append(row)

    group_df = pd.DataFrame(rows, columns=column_names)
    group_df.to_csv("fg-"+file_name, index=False)


def feature_gen_all():
    file_extension = "csv"
    os.chdir("datasets/accelerometer/accel-labeled/drunk/")
    filenames = [i for i in glob.glob('*' + '-accel-labeled.{}'.format(file_extension))]
    for file in filenames:
        gen_features_per_pid(file, "drunk")
    os.chdir('../sober/')
    filenames = [i for i in glob.glob('*' + '-accel-labeled.{}'.format(file_extension))]
    for file in filenames:
        gen_features_per_pid(file, "sober")


if __name__ == '__main__':
    feature_gen_all()
