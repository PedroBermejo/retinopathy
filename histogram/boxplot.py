import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd


def main():
    df = pd.read_csv(os.path.join(os.getcwd(), args.path_predicts))

    # create a boxplot using matplotlib
    plt.boxplot([df['inception_good_images_softmax_left'],
                 df['inception_good_images_softmax_right']])

    # add x-axis labels
    plt.xticks([1, 2], ['softmax_left', 'softmax_right'])

    # add y-axis label
    plt.ylabel('Softmax Probability')

    # add title
    plt.title('Inception Softmax Probabilities')

    # display the plot
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-predicts', help='Path to inception csv')
    args = parser.parse_args()
    main()
