```bash
mkdir data && cd data
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
sudo apt install unzip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip

wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_train.zip -d train
unzip MINDsmall_dev.zip -d test
rm MINDsmall_*.zip

cd ..
python3 src/data_preprocess.py
```

```bash
rm -rf checkpoint/
python3 src/train.py
python3 src/evaluate.py

# or

chmod +x run.sh
./run.sh
```

| classification batch | joint loss | AUC均值 | AUC方差  | MRR均值 | MRR方差  | nDCG@5均值 | nDCG@5方差 | nDCG@10均值 | nDCG@10方差 |
| -------------------- | ---------- | ------- | -------- | ------- | -------- | ---------- | ---------- | ----------- | ----------- |
| 0                    | False      | 0.6508  | 0.000014 | 0.3082  | 0.000006 | 0.3370     | 0.000014   | 0.3989      | 0.000009    |
| 0                    | True       | 0.6491  | 0.000006 | 0.3050  | 0.000000 | 0.3324     | 0.000003   | 0.3962      | 0.000002    |
| 500                  | False      | 0.6479  | 0.000010 | 0.3056  | 0.000011 | 0.3336     | 0.000015   | 0.3967      | 0.000013    |
| 500                  | True       | 0.6496  | 0.000021 | 0.3091  | 0.000001 | 0.3374     | 0.000006   | 0.4002      | 0.000003    |
| 1000                 | False      | 0.6478  | 0.000028 | 0.3056  | 0.000020 | 0.3333     | 0.000031   | 0.3960      | 0.000026    |
| 1000                 | True       | 0.6537  | 0.000009 | 0.3114  | 0.000008 | 0.3408     | 0.000019   | 0.4034      | 0.000012    |