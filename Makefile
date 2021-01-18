train: dataset
	python train.py --no_cuda --epoch 1 --dataset $(shell cat tmp/dataset_id)
.PHONY: train

dataset: tmp/dataset_id
.PHONY: dataset

tmp/dataset_id:
	bin/create-dataset

