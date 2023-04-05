import argparse
import os
from os import environ as env
from pathlib import Path

import data.utils
import model.utils as model_utils
from model.waveunet import Waveunet
from test import predict_song


def main(args):
    # MODEL
    num_features = [args.features * i for i in range(1, args.levels + 1)] if args.feature_growth == "add" else \
        [args.features * 2 ** i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print("Loading model from checkpoint " + str(args.load_model))
    state = model_utils.load_model(model, None, args.load_model, args.cuda)
    print('Step', state['step'])

    args_input = args.input
    if args_input.split("/")[-1].endswith("mixture.wav"):
        preds = predict_song(args, args_input, model)
        output_folder = os.path.dirname(args_input) if args.output is None else args.output
        for inst in preds.keys():
            data.utils.write_wav(os.path.join(output_folder, os.path.basename(args_input) + "_" + inst + ".wav"),
                                 preds[inst], args.sr)
    else:
        walker = sorted(str(p) for p in Path(args_input).rglob("*/mixture.wav"))
        for args_input in walker:
            print(args_input)
            preds = predict_song(args, args_input, model)
            output_folder = os.path.dirname(args_input) if args.output is None else args.output
            for inst in preds.keys():
                data.utils.write_wav(os.path.join(output_folder, os.path.basename(args_input) + "_" + inst + ".wav"),
                                     preds[inst], args.sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, nargs='+', default=["other", "Ru-106"],
                        help="List of instruments to separate (default: \"Ru-106 Co-60 other\")")
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--features', type=int, default=int(env['FEATURES']),
                        help='Number of feature channels per layer')
    parser.add_argument('--load_model', type=str, default=str(env['LOAD_MODEL']),
                        help='Reload a previously trained model')
    parser.add_argument('--batch_size', type=int, default=int(env['BATCH_SIZE']),
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=int(env['DEPTH']),
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=int(env['SR']), help="Sampling rate")
    parser.add_argument('--channels', type=int, default=int(env['CHANNELS']), help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=int(env['KERNEL_SIZE']),
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=int(env['OUTPUT_SIZE']),
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=int(env['STRIDES']),
                        help="Strides in Waveunet")
    parser.add_argument('--conv_type', type=str, default=str(env['CONV_TYPE']),
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default=str(env['RES']),
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=int(env['SEPARATE']),
                        help="Train separate model for each source (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default=str(env['FEATURE_GROWTH']),
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")
    parser.add_argument('--input', type=str, default=str(env['INPUT']),
                        help="Path to input mixture to be separated")
    parser.add_argument('--output', type=str, default=str(env['OUTPUT']),
                        help="Output path (same folder as input path if not set)")

    args = parser.parse_args("--cuda".split())

    main(args)
    print("done")
