#!/usr/bin/env bash

# - airplane : 0
# - automobile : 1
# - bird : 2
# - cat : 3
# - deer : 4
# - dog : 5
# - frog : 6
# - horse : 7
# - ship : 8
# - truck : 9

for cst1 in `echo 1`
do
    for cst2 in `echo 9`
    do
        python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_natural.pth' --target1 ${cst1} --target2 ${cst2} >> test_result.txt
        python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_normal.pth'  --target1 ${cst1} --target2 ${cst2} >> test_result.txt
        python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_trades.pth'  --target1 ${cst1} --target2 ${cst2} >> test_result.txt
        python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_alltar.pth'  --target1 ${cst1} --target2 ${cst2} >> test_result.txt
        # python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_sp_35.pth'   --target1 ${cst1} --target2 ${cst2} >> test_result.txt
        # python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_sp_53.pth'   --target1 ${cst1} --target2 ${cst2} >> test_result.txt
    done
done
