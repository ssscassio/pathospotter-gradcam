# -*- coding: utf-8 -*-
import sys
import argparse
from model import load_model
from util import save_model_summary


def main(model_file):
    model = load_model(model_file)

    model_summary_file = model_file + '_summary.txt'
    save_model_summary(model, model_summary_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This will save a model summary under the model folder")
    # Define argument list
    parser.add_argument('--model_file', type=str,
                        help="Path to the model file for the CNN model. Default is './models/glomeruloesclerose'")

    # Set arguments Defaults
    parser.set_defaults(
        model_file="./models/glomeruloesclerose",
    )
    args = parser.parse_args(sys.argv[1:])
    args = vars(args)

    main(
        model_file=args['model_file'],
    )
