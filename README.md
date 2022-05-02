[toc]



`hyc undergraduate thesis`实验记录



## conda虚拟环境`new`的搭建

经过多次失败的尝试，我最终搭建的conda虚拟环境命名为`new`，最终能跑通作者提供的代码（在计算队列：72rtxib和62v100ib上都能成功运行ares代码）。

```bash
conda create -n new python=3.8 pip 
conda activate new

#-------------------------------
conda install pytorch=1.5.0 torchvision=0.6.0 cudatoolkit=10.1 -c pytorch

#-------------------------------
/fs00/software/bin/pnju -u <account> -p <password> -i  

pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
pip install torch-sparse==0.6.7 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
pip install torch-cluster==1.5.7 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html

pip install torch-geometric==1.6.3

#-------------------------------
pip install pytorch-lightning==1.2.10 python-dotenv wandb==0.10.15 atom3d==v0.2.4
```

`new_e3nn_ins.lsf`

```bash
#BSUB -q 62v100ib
#BSUB -n 1
#BSUB -gpu "num=1:aff=yes"
#BSUB -J hyc
export PATH=/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin:$PATH
module load cuda/10.1.243
module load cudnn/10.1-v7.6.4.38
module load gcc/7.4.0 
cd /fsa/home/ww_duyy/hyc/data/e3nn_ares/e3nn_ares
rm -rf build dist e3nn.egg-info __pycache__
/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin/python3.8 setup_cuda.py install
```

报错`Network is unreachable`，此时输入以下命令：

```bash
cd ~/hyc/data/e3nn_ares/e3nn_ares/
module load cuda/10.1.243
module load cudnn/10.1-v7.6.4.38
module load gcc/7.4.0 
python setup_cuda.py install
#再次提交
bsub < new_e3nn_ins.lsf
```

`new_e3nn_test.lsf` 测试e3nn是否安装成功（是否可在GPU上计算）

```bash
#BSUB -q 62v100ib
#BSUB -n 1
#BSUB -gpu "num=1:aff=yes"
#BSUB -J hyc
export PATH=/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin:$PATH
python -c "import torch; import e3nn.cuda_rsh"
```



## 代码

关于修改和编写代码的说明

### ares_release源码

`train.py`在`import ares.data as d`一行前加上以下代码

```bash
os.environ["WANDB_API_KEY"] = "22bd032b485c5d9f00edefa3d1bf114e17b0b47f"  #此处换为你的wandb api key
os.environ["WANDB_MODE"] = "offline"

parentPath = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(parentPath)
```

`predict.py`的`root_dir = pathlib.Path(__file__).parent.absolute()`应该为`root_dir = pathlib.Path(__file__).parent.parent.absolute()`

其他代码没有问题，不需要改动



### atom3d库源码

`atom3d/util/rosetta.py`要改一下代码，把`_lookup_helper`函数里的`.head(1)`去掉即可，原因是loc后返回值是`pandas.core.series.Series`，head(1)相当于取原来DataFrame中该行的第一列。只有对于一个pdb文件有多个记录的情况，才需要加head(1)，但在我们的情况下并不会出现这种情况，一个pdb文件对应一行“评分”记录。其实`_lookup_helper`函数应该分为输入变量是否为DataFrame的某些行和Series两种情况来分别处理更加合理。

在用`sc_pred`生成sc文件时，`__call__`函数下`x['id'] = str(key)`改为`x['id'] = str(key[1])`，这样sc文件里`id`字段就只有`XXX.pdb`这一个值，而不是元组`"('XXX','XXX.pdb')"`。（事后发现若改为`x['id'] = str(key[0])`，在测试阶段会方便些，因为测试生成的csv文件里的tag字段的记录就不含有.pdb后缀）




### sc.py

因为作者给的测试结果.csv文件就包含了`rms`，所以不需要此脚本也可以，直接把作者的.csv文件里的ares分数换成自己复现的 :joy::joy_cat::clown_face:，但是训练时必需这一步，否则缺少标签。



### benchmark1_data.py

将原作者csv文件中的ares分数替换为我复现的



### new_benchmark1.ipynb

画图（展示测试效果）：

- Fig. 2A-C
- Fig. S3
- Fig. S4



### benchmark2_data.py

将原作者csv文件中的ares分数替换为我复现的



### make_final_csvs.py

produce `new_benchmark2_nobootstrap.csv` and `new_benchmark2_bootstrap.csv`



### new_benchmark2.ipynb

画图：

- Fig. 2D
- Fig. S5



### translation_data.py

将原作者csv文件中的ares分数替换为我复现的



### new_translation.py

画图：

- Fig. 4A



### rnaome_model.py & rnaome_predict.py

为了获得倒数第二个全连接层的输出，需要修改代码，重新进行测试，生成带有'fe'记录的`new_rnaome.csv`



### new_rnaome.ipynb

画图：

- Fig. 4B




## lsf scripts

*LSF*(Load Sharing Facility)系统下的作业脚本汇总

### `new_train.lsf`

```bash
#BSUB -q 62v100ib
#BSUB -n 8
#BSUB -gpu "num=1:aff=yes"
#BSUB -J hyc
export PATH=/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin:$PATH
module load cuda/10.1.243
module load cudnn/10.1-v7.6.4.38
module load gcc/7.4.0
cd /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/
lr=(0.005 0.0075 0.01)
bs=(8 16)
agb=(1 2)
for i in ${lr[@]};do
	for j in ${bs[@]};do
		for k in ${agb[@]};do
			/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin/python3.8 -m ares.train /fsa/home/ww_duyy/hyc/data/Townshend/classics_train_val/example_train /fsa/home/ww_duyy/hyc/data/Townshend/classics_train_val/example_val -f pdb --label_dir /fsa/home/ww_duyy/hyc/data/Townshend/classics_train_val/sc --batch_size=$j --accumulate_grad_batches=$k --learning_rate=$i --max_epochs=13  --gpus=1 --num_workers=8
		done
	done
done	
```



### `new_aug.lsf`

```bash
#BSUB -q 62v100ib
#BSUB -n 6
#BSUB -gpu "num=1:aff=yes"
#BSUB -J hyc
export PATH=/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin:$PATH
module load cuda/10.1.243
module load cudnn/10.1-v7.6.4.38
module load gcc/7.4.0
# rna_pdb=rna_puzzle_
# sc_rna=sc_rna_puzzle_
file_array=(1 2 3 4 5 6 7 8 9 10 11 12 13 14b 14f 15 17 18 19 20 21)
# cd /fsa/home/ww_duyy/hyc/data/Townshend/augmented_puzzles/decoys
# for i in ${file_array[@]}; do
# 	{
# 	mkdir ${sc_rna}${i}
# 	/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin/python3.8 sc_pred.py ${rna_pdb}${i} ${sc_rna}${i}
# 	}&
# done
# wait

cd /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/
for i in ${file_array[@]}; do
	/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin/python3.8 -m ares.predict /fsa/home/ww_duyy/hyc/data/Townshend/augmented_puzzles/decoys/rna_puzzle_$i  /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/models/ares/2zojmx7u/checkpoints/epoch=4-step=4374.ckpt /fsa/home/ww_duyy/hyc/data/Townshend/augmented_puzzles/decoys/pred_output/new_rna_puzzle_$i.csv -f pdb --label_dir /fsa/home/ww_duyy/hyc/data/Townshend/augmented_puzzles/decoys/sc_rna_puzzle_$i --gpus=1 --num_workers=6 --batch_size=16  
done
```



### `new_aug_natives.lsf`

```bash
#BSUB -q 62v100ib
#BSUB -n 6
#BSUB -gpu "num=1:aff=yes"
#BSUB -J hyc
export PATH=/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin:$PATH
module load cuda/10.1.243
module load cudnn/10.1-v7.6.4.38
module load gcc/7.4.0
cd /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/
/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin/python3.8 -m ares.predict /fsa/home/ww_duyy/hyc/data/Townshend/augmented_puzzles/natives  /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/models/ares/2zojmx7u/checkpoints/epoch=4-step=4374.ckpt /fsa/home/ww_duyy/hyc/data/Townshend/augmented_puzzles/decoys/pred_output/new_rna_puzzle_natives.csv -f pdb --nolabels --gpus=1 --num_workers=6 --batch_size=16
```



### `new_watkins.lsf`

```bash
#BSUB -q 62v100ib
#BSUB -n 6
#BSUB -gpu "num=1:aff=yes"
#BSUB -J hyc
export PATH=/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin:$PATH
module load cuda/10.1.243
module load cudnn/10.1-v7.6.4.38
module load gcc/7.4.0
file_array=(cdigmp cobalamin fmn guanine mn neomycin preq1 thf cgamp gln2_monomer nada nico prpp sam3 thim ykoy)
cd /fsa/home/ww_duyy/hyc/data/Watkins/
# for i in ${file_array[@]}; do
# {
#   mkdir sc_${i}
#   /fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin/python3.8 /fsa/home/ww_duyy/hyc/sc_pred.py ${i} sc_${i}
# }&
# done
# wait

cd /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/
for i in ${file_array[@]}; do
  /fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin/python3.8 -m ares.predict /fsa/home/ww_duyy/hyc/data/Watkins/$i  /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/models/ares/2zojmx7u/checkpoints/epoch=4-step=4374.ckpt  /fsa/home/ww_duyy/hyc/data/Watkins/pred_output/new_$i.csv -f pdb --label_dir /fsa/home/ww_duyy/hyc/data/Watkins/sc_$i --gpus=1 --num_workers=6 --batch_size=16
done
```



### `new_blind.lsf`

```bash
#BSUB -q 62v100ib
#BSUB -n 6
#BSUB -gpu "num=1:aff=yes"
#BSUB -J hyc
export PATH=/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin:$PATH
module load cuda/10.1.243
module load cudnn/10.1-v7.6.4.38
module load gcc/7.4.0
file_array=(puzzle24 puzzle26 puzzle27 puzzle28)
puzzle28_template=(cst nocst full_native)

cd /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/
for i in ${file_array[@]}; do
	if [[ "$i" != "puzzle28" ]]
	then
		/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin/python3.8 -m ares.predict /fsa/home/ww_duyy/hyc/data/Townshend/blind/$i/all_models  /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/models/ares/2zojmx7u/checkpoints/epoch=4-step=4374.ckpt  /fsa/home/ww_duyy/hyc/data/Townshend/blind/pred_output/new_$i.csv -f pdb --nolabels --gpus=1 --num_workers=8 --batch_size=16
	else
		for j in ${puzzle28_template[@]}; do
			/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin/python3.8 -m ares.predict /fsa/home/ww_duyy/hyc/data/Townshend/blind/$i/all_models/$j  /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/models/ares/2zojmx7u/checkpoints/epoch=4-step=4374.ckpt/fsa/home/ww_duyy/hyc/data/Townshend/blind/pred_output/new_${i}_${j}.csv -f pdb --nolabels --gpus=1 --num_workers=8 --batch_size=16   
		#注意new_${i}_${j}如果写成new_$i_$j，bash会认为$i_变量为空，故输出的文件名会像new_cst.csv这样，咳，看来养成对变量加{}的习惯还是有好处的
		done
	fi
done
```



### `new_translation.lsf`

```bash
#BSUB -q 62v100ib
#BSUB -n 6
#BSUB -gpu "num=1:aff=yes"
#BSUB -J hyc
export PATH=/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin:$PATH
module load cuda/10.1.243
module load cudnn/10.1-v7.6.4.38
module load gcc/7.4.0

cd /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/

# translation
/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin/python3.8 -m ares.predict /fsa/home/ww_duyy/hyc/data/Townshend/translation/pdbs /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/models/ares/2zojmx7u/checkpoints/epoch=4-step=4374.ckpt  /fsa/home/ww_duyy/hyc/data/Townshend/translation/new_translation.csv -f pdb --nolabels --gpus=1 --num_workers=8 --batch_size=16
```



### `new_rnaome.lsf`

```bash
#BSUB -q 62v100ib
#BSUB -n 6
#BSUB -gpu "num=1:aff=yes"
#BSUB -J hyc
export PATH=/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin:$PATH
module load cuda/10.1.243
module load cudnn/10.1-v7.6.4.38
module load gcc/7.4.0

cd /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/

# rnaome
/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin/python3.8 -m ares.predict /fsa/home/ww_duyy/hyc/data/Townshend/rnaome/pdbs /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/models/ares/2zojmx7u/checkpoints/epoch=4-step=4374.ckpt  /fsa/home/ww_duyy/hyc/data/Townshend/rnaome/new_rnaome.csv -f pdb --nolabels --gpus=1 --num_workers=8 --batch_size=16
```





### `benchmark1_data.lsf`

```bash
#BSUB -q 5218
#BSUB -n 16
#BSUB -J hyc
export PATH=/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin:$PATH
module load cuda/10.1.243
module load cudnn/10.1-v7.6.4.38
module load gcc/7.4.0
cd /fsa/home/ww_duyy/hyc/data/Townshend/augmented_puzzles/decoys/pred_output
python benchmark1_data.py
```



### `benchmark2_data.lsf`

```bash
#BSUB -q 5218
#BSUB -n 16
#BSUB -J hyc
export PATH=/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin:$PATH
module load cuda/10.1.243
module load cudnn/10.1-v7.6.4.38
module load gcc/7.4.0
cd /fsa/home/ww_duyy/hyc/data/Watkins/pred_output
python benchmark2_data.py
```



### `rnaome_fe.lsf`

```bash
#BSUB -q 62v100ib
#BSUB -n 6
#BSUB -gpu "num=1:aff=yes"
#BSUB -J hyc
export PATH=/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin:$PATH
module load cuda/10.1.243
module load cudnn/10.1-v7.6.4.38
module load gcc/7.4.0

cd /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/

/fsa/home/ww_duyy/hyc/anaconda3/envs/new/bin/python3.8 -m ares.rnaome_predict /fsa/home/ww_duyy/hyc/data/Townshend/rnaome/pdbs /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/models/ares/2zojmx7u/checkpoints/epoch=4-step=4374.ckpt  /fsa/home/ww_duyy/hyc/data/Townshend/rnaome/new_rnaome.csv -f pdb --nolabels --gpus=1 --num_workers=8 --batch_size=16
```



