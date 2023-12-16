python orca_cifar.py --dataset cifar10 --labeled-num 5 --labeled-ratio 0.5 --epochs 200 --batch-size 512 --degrade-level 1 --degrade-choice blur 
python orca_cifar.py --dataset cifar10 --labeled-num 5 --labeled-ratio 0.5 --epochs 200 --batch-size 512 --degrade-level 1 --degrade-choice jitter
python orca_cifar.py --dataset cifar10 --labeled-num 5 --labeled-ratio 0.5 --epochs 200 --batch-size 512 --degrade-level 1 --degrade-choice elastic 

python odac_cifar.py --dataset cifar100 --labeled-num 50 --labeled-ratio 0.5 --epochs 200 --batch-size 512 --degrade-level 1 --degrade-choice blur --dsbn
python odac_cifar.py --dataset cifar100 --labeled-num 50 --labeled-ratio 0.5 --epochs 200 --batch-size 512 --degrade-level 1 --degrade-choice jitter --dsbn
python odac_cifar.py --dataset cifar100 --labeled-num 50 --labeled-ratio 0.5 --epochs 200 --batch-size 512 --degrade-level 1 --degrade-choice elastic --dsbn

python orca_cifar.py --dataset cifar100 --labeled-num 50 --labeled-ratio 0.5 --epochs 200 --batch-size 512 --degrade-level 1 --degrade-choice blur 
python orca_cifar.py --dataset cifar100 --labeled-num 50 --labeled-ratio 0.5 --epochs 200 --batch-size 512 --degrade-level 1 --degrade-choice jitter
python orca_cifar.py --dataset cifar100 --labeled-num 50 --labeled-ratio 0.5 --epochs 200 --batch-size 512 --degrade-level 1 --degrade-choice elastic 