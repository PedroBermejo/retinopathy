import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd


def main():
    df = pd.read_csv(os.path.join(os.getcwd(), args.path_predicts))

    models = ['inception', 'mobilenet', 'resnet', 'vgg']
    models_replace = {
        'inception': 'InceptionV3',
        'mobilenet': 'MobilenetV2',
        'resnet': 'Resnet50',
        'vgg': 'Vgg19'}
    datasets = ['good_images', 'blur', 'gauss_noise', 'random_fog']
    replace = {'good_images': 'referencia',
               'blur': 'borrosidad',
               'gauss_noise': 'ruido',
               'random_fog': 'niebla'}
    colors = ['red', 'green', 'blue', 'orange']

    fig, ax = plt.subplots(ncols=4, figsize=(10, 5))
    i = 0

    for model_name in models:
        template_df = []
        for dataset_name in datasets:
            max_df = pd.DataFrame({replace[dataset_name]: df[[f'{model_name}_{dataset_name}_softmax_left']].max(axis=1)})

            if len(template_df) == 0:
                template_df = max_df
            else:
                template_df = max_df.join(template_df, how='inner')

        template_df.boxplot(ax=ax[i], column=list(replace.values()))
        ax[i].set_title(models_replace[model_name])
        i = i + 1

    ax[0].set_ylabel('Intervalo de confianza')

    for ax_i in ax:
        for tick in ax_i.get_xticklabels():
            tick.set_rotation(45)

    fig.text(0.04, 0.5, '', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-predicts', help='Path to inception csv')
    args = parser.parse_args()
    main()
