train: tmp/dataset_id
	python train.py --no_cuda --epoch 1 --dataset $(shell cat tmp/dataset_id)
.PHONY: train

detect: tmp/model_id
	python detect.py --image_folder data/samples --model $(shell cat tmp/model_id)
.PHONY: detect


tmp/dataset_id:
	bin/create-dataset


tmp/model_id:
	curl -X POST -H "Content-Type: application/json" \
	-d '{"name": "yolov3", "order_by": "created"}' \
	'http://adminuser:12345678@localhost:8008/models.get_all' | \
	jq -r ".data.models[-1].id" > $@
