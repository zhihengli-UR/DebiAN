{

PYTHONPATH=.:$PYTHONPATH python scene_exp/debian.py --batch_size 128 --epoch 50 --num_workers 28 --dset_name places --ood_dset_name lsun

    exit
}