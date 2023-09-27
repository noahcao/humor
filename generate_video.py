import os 
import sys 
import subprocess

frame_dir = 'out/humor_qual_jinkun_test_apd_test_60s_1sample/eval_sampling'
out_dir = os.path.join("out_video", frame_dir.split("/")[-2])
os.makedirs(out_dir, exist_ok=True)

seqs_list = os.listdir(frame_dir)

for seq_name in seqs_list:
    seq_dir = os.path.join(frame_dir, seq_name)
    out_path = os.path.join(out_dir, "{}.mp4".format(seq_name))
    cmd = 'ffmpeg -framerate 30 -pattern_type glob -i  \
        "{}/*.png" -c:v libx264 -pix_fmt yuv420p {}'.format(seq_dir, out_path)
    print(seq_name)
    os.system(cmd)