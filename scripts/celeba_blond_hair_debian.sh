{

PYTHONPATH=.:$PYTHONPATH python celeba_exp/debian.py --batch_size 256 --attribute_index 9 --lr 1e-4 --arch resnet18 --num_workers 10 --dset_name celeba --criterion BCE --weight_decay 1e-4 --optimizer adam --amp

    exit
}