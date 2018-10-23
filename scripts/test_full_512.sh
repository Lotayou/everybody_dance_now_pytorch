############## To train full model #############
##### Using GPUs with 12G memory (not tested)
# Using labels only (1s per batch)
export CUDA_VISIBLE_DEVICES=0
# python test_video.py --name everybody_dance_now_temporal --model pose2vid --dataroot ./datasets/full_test/ --which_epoch latest --netG local --ngf 32 --niter_fix_global 20 --label_nc 0 --no_instance --resize_or_crop none --clip_length 10000
python test_video.py --name everybody_dance_now_temporal --model pose2vid --dataroot ./datasets/cardio_dance/ --which_epoch 56 --netG local --ngf 32 --niter_fix_global 20 --label_nc 0 --no_instance --resize_or_crop none