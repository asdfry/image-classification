# node4
### local (dali off)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node4 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --disable_dali -le bond0 ./
```
### local (dali with cpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node4 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --dali_cpu -le bond0 ./
```
### local (dali with gpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node4 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode -le bond0 ./
```
### ontap (dali off)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node4 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --disable_dali -le bond0 ./
```
### ontap (dali with cpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node4 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --dali_cpu -le bond0 ./
```
### ontap (dali with gpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node4 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode -le bond0 ./
```
# node5
### local (dali off)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node5 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --disable_dali -le bond0 ./
```
### local (dali with cpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node5 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --dali_cpu -le bond0 ./
```
### local (dali with gpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node5 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode -le bond0 ./
```
### ontap (dali off)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node5 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --disable_dali -le bond0 ./
```
### ontap (dali with cpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node5 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --dali_cpu -le bond0 ./
```
### ontap (dali with gpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node5 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode -le bond0 ./
```
# node7
### local (dali off)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node7 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --disable_dali -le eth1 ./
```
### local (dali with cpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node7 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --dali_cpu -le eth1 ./
```
### local (dali with gpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node7 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode -le eth1 ./
```
### ontap (dali off)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node7 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --disable_dali -le eth1 ./
```
### ontap (dali with cpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node7 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --dali_cpu -le eth1 ./
```
### ontap (dali with gpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node7 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode -le eth1 ./
```
# node8
### local (dali off)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node8 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --disable_dali -le eth1 ./
```
### local (dali with cpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node8 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --dali_cpu -le eth1 ./
```
### local (dali with gpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node8 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode -le eth1 ./
```
### ontap (dali off)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node8 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --disable_dali -le eth1 ./
```
### ontap (dali with cpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node8 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --dali_cpu -le eth1 ./
```
### ontap (dali with gpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node8 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode -le eth1 ./
```
# node9
### local (dali off)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node9 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --disable_dali -le eth1 ./
```
### local (dali with cpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node9 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --dali_cpu -le eth1 ./
```
### local (dali with gpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c local-node9 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode -le eth1 ./
```
### ontap (dali off)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node9 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --disable_dali -le eth1 ./
```
### ontap (dali with cpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node9 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode --dali_cpu -le eth1 ./
```
### ontap (dali with gpu)
```
torchrun --nproc_per_node=4 main.py -b 256 -c ontap-node9 --loss_scale 256.0 --workers 4 --lr=0.4 --fp16_mode -le eth1 ./
```