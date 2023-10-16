#!/bin/sh
CUDA_VISIBLE_DEVICES=3 python3 main.py \
--video_file data/videos/ayush.mp4 \
--path results/ayush_test/   \
--camera_params "1671.770118, 540, 960" \
--camera_model "SIMPLE_PINHOLE" \
--loss_case 231 \
--make_video \
--straight_line_method True \
--gt_prediction_prepro \
--ref_tgt_depth_diff_weight 0.03 \
--pos_dist_min 0.01 \
--learning_rate 3e-5 \
--confidence_enabled \
--confidence_rendered \
