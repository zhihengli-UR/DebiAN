{

PYTHONPATH=.:$PYTHONPATH python celeba_exp/debian.py --batch_size 64 --attribute_index 20 --lr 1e-4 --arch resnet50 --num_workers 20 --dset_name celeba --criterion BCE --weight_decay 0 --optimizer adam

    exit
}