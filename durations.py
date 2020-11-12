
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np

import click



@click.command('plot')
@click.argument('input-folder')
@click.argument('part-of-csv-name')
@click.option('--to-file', default='')
@click.option('--replace', help="what should be replaced with '' to get number from csv name. '.csv' is replaced automatically")
def plot(input_folder, part_of_csv_name, to_file, replace):
    duration_csvs = [x for x in os.listdir(input_folder) if part_of_csv_name in x]

    series = {}
    for csv in duration_csvs:
        series = {**series, **{csv: pd.read_csv(os.path.join(input_folder, csv))['0']}}

    series_renamed = {}
    for name, durations in series.items():
        series_renamed = {**series_renamed, **{name.replace(replace, '').replace('.csv', ''): durations}}

    df = pd.DataFrame(series_renamed)
    zscores = zscore(df)
    abs_z_scores = np.abs(zscores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df = df[filtered_entries]

    sorted_names = [str(x) for x in list(range(1,len(duration_csvs) + 1))]

    df = df.reindex(sorted_names, axis=1)


    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.set_style("whitegrid")
    plot = sns.violinplot(data=df)
    plt.ylim(0, df.max().max())
    plot.set(
        xlabel='Number of images processed in paralel',
        ylabel='Time in milliseconds',
        title='NX'
        )
    if to_file:
        plot.figure.savefig(f'./models/{to_file}')
    else:
        plt.show()
    plt.clf()


@click.group()
def cli():
    pass

if __name__ == '__main__':
    cli.add_command(plot)
    cli()
