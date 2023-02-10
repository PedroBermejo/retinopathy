import torch
import argparse
import os
import pandas as pd


def extract_interest_columns(df, prefix):
    result = df[[f"{prefix}_names",
                 f"{prefix}_probability_left",
                 f"{prefix}_probability_right",
                 f"{prefix}_predicts"]]

    result.dropna(how='all', inplace=True)
    return result


def add_softmax(df, prefix):
    df = df.reset_index(drop=True)
    softmax = torch.nn.Softmax(dim=1)
    tensor = torch \
        .from_numpy(df[[f"{prefix}_probability_left",
                        f"{prefix}_probability_right"]].values)

    softmax_tensor = softmax(tensor)

    softmax_df = pd.DataFrame(softmax_tensor.numpy(),
                              columns=[f"{prefix}_softmax_left", f"{prefix}_softmax_right"])

    return pd.merge(df, softmax_df, left_index=True, right_index=True)


def process_df(df):
    good_img_df = extract_interest_columns(df, "good_images")
    good_img_df['good_images_names'] = \
        good_img_df['good_images_names'].str.replace('.jpg', '.png')
    good_img_df = add_softmax(good_img_df, "good_images")

    blur_df = extract_interest_columns(df, "blur")
    blur_df = add_softmax(blur_df, "blur")

    gauss_df = extract_interest_columns(df, "gauss_noise")
    gauss_df = add_softmax(gauss_df, "gauss_noise")

    random_fog_df = extract_interest_columns(df, "random_fog")
    random_fog_df = add_softmax(random_fog_df, "random_fog")

    return good_img_df\
        .merge(blur_df, left_on='good_images_names', right_on='blur_names', how='inner')\
        .merge(gauss_df, left_on='good_images_names', right_on='gauss_noise_names', how='inner')\
        .merge(random_fog_df, left_on='good_images_names', right_on='random_fog_names', how='inner')


def main():
    inceptionDf = pd.read_csv(os.path.join(os.getcwd(), args.inception_predicts))
    mobileNetDf = pd.read_csv(os.path.join(os.getcwd(), args.mobilenet_predicts))
    resnetDf = pd.read_csv(os.path.join(os.getcwd(), args.resnet_predicts))
    vggDf = pd.read_csv(os.path.join(os.getcwd(), args.vgg_predicts))

    inceptionMDf = process_df(inceptionDf).add_prefix("inception_")
    mobileMNetDf = process_df(mobileNetDf).add_prefix("mobilenet_")
    resnetMDf = process_df(resnetDf).add_prefix("resnet_")
    vggMDf = process_df(vggDf).add_prefix("vgg_")

    merged = inceptionMDf\
        .merge(mobileMNetDf, left_on='inception_good_images_names', right_on='mobilenet_good_images_names', how='inner')\
        .merge(resnetMDf, left_on='inception_good_images_names', right_on='resnet_good_images_names', how='inner')\
        .merge(vggMDf, left_on='inception_good_images_names', right_on='vgg_good_images_names', how='inner')

    merged.to_csv(os.path.join(os.getcwd(), args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inception-predicts', help='Path to inception csv')
    parser.add_argument('--mobilenet-predicts', help='Path to mobilenet csv')
    parser.add_argument('--resnet-predicts', help='Path to resnet csv')
    parser.add_argument('--vgg-predicts', help='Path to vgg csv')
    parser.add_argument('--output', help='Path to output csv')

    args = parser.parse_args()
    main()
