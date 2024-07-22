import argparse
import csv
import os
from turtle import ScrolledCanvas
from scipy.stats import spearmanr
import json

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--suffix",
        default="",
        type=str,
        required=False,
        help="Amateur model size"
    )


    args = parser.parse_args()
    tests = {
        'good': ("./commongen_test.txt", 1.0),
        'med': ("./commongen_test_med.txt", 0.5),
        'bad': ("./commongen_test_bad.txt", 0.0),
        }

    results = {test_name: -1 for test_name in tests}

    gt_scores = []
    gt_scores_bin = []
    pred_scores = []
    for test_name in tests:
        ground_truth_file, score = tests[test_name]
        if args.suffix == "":
            fr1 = open(ground_truth_file, 'r')
        else:
            fr1 = open(ground_truth_file.replace('.txt', '_' + args.suffix.strip() + '.txt'), 'r')
        gt_lines=fr1.readlines()
        
        # if "dstc9" not in test_name: 
        for ind, gt_line in enumerate(gt_lines):
            if gt_line.strip() == "":
                continue
            gt_scores.append(score)
            gt_scores_bin.append(float(score > 0.75))
            pred_scores.append(float(gt_lines[ind].strip().split('\t')[-1]))

    sp_score=spearmanr(gt_scores, pred_scores)
    sp_score_bin=spearmanr(gt_scores_bin, pred_scores)

    results[test_name] = sp_score 
    print("###Spearman Correlation {}, p-value: {}###".format(sp_score[0], sp_score[1]))
    print("###Spearman Correlation with Label Binarization {}, p-value: {}###".format(sp_score_bin[0], sp_score_bin[1]))
