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

python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_natural.pth' --target1 1 --target2 9 >> result.txt
python attack_targeted_cifar10.py --model-path './cp_cifar10/res18_normal.pth'  --target1 1 --target2 9 >>result.txt