import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()


def vcat_dist(df, idx, save=False, filename=None):

    ncount = len(df)
    plt.figure(figsize=(6, 6))
    ax = sns.countplot(x=idx, data=df)  # , order=[3,4,5,6,7,8,9,10,11,12])
    plt.title('Distribution of {}'.format(idx))
    plt.xlabel('Number of Classes')

    # Make twin axis
    ax2 = ax.twinx()

    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax2.set_ylabel('Frequency [%]')

    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate('{:.1f}%'.format(  # set the alignment of the text
            100.*y/ncount), (x.mean(), y), ha='center', va='bottom')

    # Use a LinearLocator to ensure the correct number of ticks
    ax.yaxis.set_major_locator(ticker.LinearLocator(11))

    # Fix the frequency range to 0-100
    ax2.set_ylim(0, 100)
    ax.set_ylim(0, ncount)

    # And use a MultipleLocator to ensure a tick spacing of 10
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Need to turn the grid on ax2 off,
    # otherwise the gridlines end up on top of the bars
    ax2.grid(None)

    if save:
        plt.savefig('{}.png'.format(filename))
    else:
        plt.show()


def vnum_dist(df):
    for name, values in df.iteritems():
        sns.distplot(df[name], rug=False, hist=True)
        plt.show()


# %matplotlib inline
def lineplot(training_sizes, val_result, test_result):
    import pandas as pd
    dataset = pd.DataFrame({'Training Size': training_sizes + training_sizes,
                            'Accuracy': val_result + test_result,
                            'Dataset': [
                               'Validation' for _ in range(len(training_sizes))] +
                            ['Test' for _ in range(len(training_sizes))],
                            })

    sns.lineplot(x='Training Size', y='Accuracy', data=dataset,
                 markers=True, hue='Dataset')
    plt.show()
