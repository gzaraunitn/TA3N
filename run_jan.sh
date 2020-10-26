class_file='none'
modality=RGB
train_source_list='none'
train_target_list='none'
val_list='none'
arch=resnet101
pretrained=none
baseline_type=video
frame_aggregation=trn-m
num_segments=5
test_segments=5
add_fc=1
fc_dim=512
use_target=uSv
share_params=Y
dis_DA=JAN
alpha=1
adv_pos_0=Y
adv_DA=none
beta_0=0.75
beta_1=0.75
beta_2=0.5
use_attn=TransAttn
n_attn=1
use_attn_frame=none
use_bn=none
add_loss_DA=none
gamma=0.003
ens_DA=none
mu=0
bS=128
bS_2=128
lr=3e-2
optimizer=SGD
val_segments=$test_segments
lr_decay=10
lr_adaptive=dann
lr_steps_1=10
lr_steps_2=20
epochs=30
gd=20
exp_path='action-experiments/Testexp-SGD-share_params_Y-lr_3e-2-bS_128_74/hmdb_ucf-5seg-disDA_DAN-alpha_1-advDA_none-beta_0.75_0.75_0.5-useBN_none-addlossDA_none-gamma_0.003-ensDA_none-mu_0-useAttn_TransAttn-n_attn_1/'


CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py \
    $class_file \
    $modality \
    $train_source_list \
    $train_target_list \
    $val_list \
    --exp_path $exp_path \
    --arch $arch \
    --pretrained $pretrained \
    --baseline_type $baseline_type \
    --frame_aggregation $frame_aggregation \
    --num_segments $num_segments \
    --val_segments $val_segments \
    --add_fc $add_fc \
    --fc_dim $fc_dim \
    --dropout_i 0.5 \
    --dropout_v 0.5 \
    --use_target $use_target \
    --share_params $share_params \
    --dis_DA $dis_DA \
    --alpha $alpha \
    --place_dis N Y N \
    --adv_DA $adv_DA \
    --beta $beta_0 $beta_1 $beta_2 \
    --place_adv $adv_pos_0 Y Y \
    --use_bn $use_bn \
    --add_loss_DA $add_loss_DA \
    --gamma $gamma \
    --ens_DA $ens_DA \
    --mu $mu \
    --use_attn $use_attn \
    --n_attn $n_attn \
    --use_attn_frame $use_attn_frame \
    --gd $gd \
    --lr $lr \
    --lr_decay $lr_decay \
    --lr_adaptive $lr_adaptive \
    --lr_steps $lr_steps_1 $lr_steps_2 \
    --epochs $epochs \
    --optimizer $optimizer \
    --n_rnn 1 \
    --rnn_cell LSTM \
    --n_directions 1 \
    --n_ts 5 \
    -b $bS $bS_2 $bS \
    -j 0 \
    -ef 1 \
    -pf 50 \
    -sf 50 \
    --copy_list N N \
    --save_model 