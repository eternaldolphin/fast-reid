# 在Market-1501上训练
## BoT(R18)

`config`: `configs/Market1501/bagtricks_R18.yml`
```
python3 tools/train_net.py --config-file ./configs/Market1501/bagtricks_R18.yml MODEL.DEVICE "cuda:0"
```

## BoT(RepVGG)

`config`: `configs/Market1501/bagtricks_Repvgg.yml`
```
python3 tools/train_net.py --config-file ./configs/Market1501/bagtricks_Repvgg.yml MODEL.DEVICE "cuda:0"
```

```
conda activate fastreid
python3 tools/train_net.py --config-file ./configs/Market1501/bagtricks_Repvgg.yml --eval-only \
MODEL.WEIGHTS logs/market1501/bagtricks_Repvgg/model_final.pth MODEL.DEVICE "cuda:0"
```