_base_ = ["1ake.py"]
# Rotation refinement starts from supplied poses; abinit overrides pose_init.
pose_init = "given"
n_imgs_pose_search = 0
use_pose_table = False
optimize_translations = True
model = dict(use_pose_head=True, pose_head_type="mlp")
loss = dict(pose_reg_weight=0.5)
optimizer = dict(pose_lr=0.003)
hps_score = "correlation"
