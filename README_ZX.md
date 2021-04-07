# 使用别人训练好的KITTI2015进行测试
python finetune.py --maxdisp 192 --with_spn --datapath /home/zhangxiao/data/data_scene_flow/training/ \
   --save_path results/kitti2015 --datatype 2015 --pretrained /home/zhangxiao/modelzoo/AnyNet/owner_pretrained/checkpoint/kitti2015_ck/checkpoint.tar \
   --split_file /home/zhangxiao/modelzoo/AnyNet/owner_pretrained/checkpoint/kitti2015_ck/split.txt --evaluate

# 使用自己从头开始训练的KITTI2015进行测试
python finetune.py --maxdisp 192 --with_spn --datapath /home/zhangxiao/data/data_scene_flow/training/ \
   --save_path results/kitti2015 --datatype 2015 --pretrained /home/zhangxiao/code/AnyNet/results/kitti2015/checkpoint.tar \
   --split_file /home/zhangxiao/modelzoo/AnyNet/owner_pretrained/checkpoint/kitti2015_ck/split.txt --evaluate
   
# 使用别人 pretrain的sceneflow模型进行再次训练
python finetune.py --maxdisp 192 --with_spn --datapath /home/zhangxiao/data/data_scene_flow/training/ \
   --save_path results/kitti2015 --datatype 2015 --pretrained /home/zhangxiao/code/AnyNet/results/kitti2015/checkpoint_batch24.tar \
   --split_file /home/zhangxiao/modelzoo/AnyNet/owner_pretrained/checkpoint/kitti2015_ck/split.txt 
   
# 使用别人 pretrain的sceneflow模型来训练杂草_kitti模型
python finetune.py --maxdisp 192 --with_spn --datapath /home/zhangxiao/data/zacao_kitti/ \
   --save_path results/zacao_kitti/models --datatype other --pretrained /home/zhangxiao/modelzoo/AnyNet/owner_pretrained/checkpoint/sceneflow/sceneflow.tar