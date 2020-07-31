python tools/run_net.py   --cfg configs/myconfig/Kinetics/c2/I3D_8x8_R50.yaml   DATA.PATH_TO_DATA_DIR datas

python tools/run_net.py --predict_source  0 \
--cfg configs/myconfig/Kinetics/c2/SLOWFAST_8x8_R50.yaml

python tools/run_net.py --predict_source  /home/stephen/workspace/Data/hmdb51_org1/wave/Wie_man_winkt_wave_u_cm_np1_fr_med_0.avi \
--cfg configs/myconfig/Kinetics/c2/SLOWFAST_4x16_R50.yaml


python tools/run_net.py   --cfg configs/myconfig/Kinetics/C2D_8x8_R50.yaml \
DATA.PATH_TO_DATA_DIR /home/stephen/workspace/ActionRecognition/SlowFast/datas
TRAIN.BATCH_SIZE 1



python tools/run_net.py --cfg configs/myconfig/Kinetics/SLOWFAST_8x8_R50.yaml



更改：

train_net.py 103   192 line change (1,5)  - > (1,1)
meters.py   (1,5) --> (1, 1)  def finalize_metrics(self, ks=(1, 1)):




python tools/run_net.py --predict_source  /home/stephen/workspace/Data/hmdb51_org1/wave/Wie_man_winkt!!_wave_u_cm_np1_fr_med_0.avi --cfg configs/myconfig/Kinetics/c2/SLOWFAST_4x16_R50.yaml
