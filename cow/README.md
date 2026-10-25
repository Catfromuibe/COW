# COW

实验主代码。数据读取沿用 HyperPotter 的协议列，模型按论文里的条件语音世界模型、轨迹证据和固定融合来训练。

协议列与 HyperPotter `Default_dataset` 相同：

- 第 5 列：`bonafide` 或 `spoof`
- 第 6 列：音频绝对路径
- 第 9 列：`train`、`dev`、`eval`、`test` 或 `-`

HyperPotter 把 spoof 编成 0。这里按论文把真实语音编成 0，篡改语音编成 1，分数越高越像篡改。

训练先把真实录音分成互不相交的参考集和检测集。参考集只用来拟合世界模型，检测集才做真伪监督。处理后的视图保持长度不变，所以时间邻域可以一一对应，一致性只加在这些对齐位置上，并且用上限截断。

```text
python main.py --debug --stage both
python main.py --stage both --protocols_trn_path protocols/ASVspoof2019LA.txt --protocols_wild_path protocols/InTheWild.txt --comment cow
python main.py --dev --protocols_dev_path protocols/InTheWild.txt --model_path exp/cow/detector.pt
```

五列的官方 ASVspoof 协议可以加 `--audio_root` 指向对应的 flac 目录。
