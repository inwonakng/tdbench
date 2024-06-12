#!/bin/bash

cd "$(dirname "$0")/.."

source ./scripts/activate_conda.sh tdbench

# default values
OPERATION="classifier"
DATASETS="[\
adult,\
amazon_employee_access,\
bank_marketing,\
credit,\
credit_default,\
diabetes,\
electricity,\
elevators,\
higgs,\
home_equity_credit,\
house,\
jannis,\
law_school_admissions,\
magic_telescope,\
medical_appointments,\
mini_boo_ne,\
numer_ai,\
nursery,\
online_shoppers,\
phishing_websites,\
pol,\
road_safety,\
tencent_ctr_small,\
two_d_planes\
]"
DATA_MODE="onehot"
CLASSIFIERS="[xgb,ft_transformer,resnet,mlp,logistic_regression,gaussian_nb,knn]"
DISTILL_METHODS="[original,encoded,decoded,random_sample,agglo,kmeans,kip,gm]"
ENCODERS="[mlp,gnn,tf]"
ENCODER_TRAIN="a100s"
ENCODER_TRAIN_TARGET="[base,multihead]"
LATENT_DIM=16
RESULTS_DIR="tune_classifier_results"
CHECKPOINT_DIR="best_checkpoints"
TUNE_HYPEROPT="false"

for arg in "$@"; do
	case "$arg" in
	--op=*)
		OPERATION=${arg#*=}
		shift 1
		;;
	--datasets=*)
		DATASETS=${arg#*=}
		shift 1
		;;
	--data_mode=*)
		DATA_MODE=${arg#*=}
		shift 1
		;;
	--classifiers=*)
		CLASSIFIERS=${arg#*=}
		shift 1
		;;
	--distill_methods=*)
		DISTILL_METHODS=${arg#*=}
		shift 1
		;;
	--encoders=*)
		ENCODERS=${arg#*=}
		shift 1
		;;
	--encoder_train=*)
		ENCODER_TRAIN=${arg#*=}
		shift 1
		;;
	--encoder_train_target=*)
		ENCODER_TRAIN_TARGET=${arg#*=}
		shift 1
		;;
	--latent_dim=*)
		LATENT_DIM=${arg#*=}
		shift 1
		;;
	--checkpoint_dir=*)
		CHECKPOINT_DIR=${arg#*=}
		shift 1
		;;
	--results_dir=*)
		RESULTS_DIR=${arg#*=}
		shift 1
		;;
	--tune_hyperopt)
		TUNE_HYPEROPT="true"
		shift 1
		;;
	*)
		echo "Unknown argument ${arg}" >&2
		exit 1
		;;
	esac
done
shift

case "$OPERATION" in
"classifier")
	COMMAND="python -m tdbench.tune.classifier"
	;;
"encoder")
	COMMAND="python -m tdbench.tune.encoder"
	;;
"debug-classifier")
	COMMAND="python -m tdbench.debug.classifier"
	;;
"load-classifier-results")
	COMMAND="python -m tdbench.results.load.classifier_performance"
	;;
*)
	echo "Unknown operation" >&2
	exit 1
	;;
esac

ARGS=" data/datasets=$DATASETS"
ARGS+=" distill/methods=$DISTILL_METHODS"
ARGS+=" classifier/models=$CLASSIFIERS"
ARGS+=" classifier.train.results_dir=$RESULTS_DIR"
ARGS+=" classifier.train.tune_hyperopt=$TUNE_HYPEROPT"
ARGS+=" encoder/train=$ENCODER_TRAIN"
ARGS+=" encoder.train.latent_dim=$LATENT_DIM"
ARGS+=" encoder.train.checkpoint_dir=$CHECKPOINT_DIR"
ARGS+=" encoder.train.train_target=$ENCODER_TRAIN_TARGET"

case "$ENCODERS" in
"none")
	if [ "$OPERATION" == "encoder" ]; then
		echo "Tuning encoders requires at least one encoder" >&2
		exit 1
	fi
	ARGS+=" encoder/models=[]"
	;;
*)
	ARGS+=" encoder/models=$ENCODERS"
	;;
esac

case "$DATA_MODE" in
"onehot")
	ARGS+=" data.mode.parse_mode=onehot"
	;;
"mixed")
	ARGS+=" data.mode.parse_mode=mixed"
	;;
"onehot-mixed")
	if [ "$OPERATION" == "encoder" ]; then
		echo "Encoder tuning cannot have changing data mode" >&2
		exit 1
	fi
	ARGS+=" data.mode.parse_mode=onehot"
	ARGS+=" +distill.common.post_data_mode_name=mixed"
	;;
"mixed-onehot")
	if [ "$OPERATION" == "encoder" ]; then
		echo "Encoder tuning cannot have changing data mode" >&2
		exit 1
	fi
	ARGS+=" data.mode.parse_mode=mixed"
	ARGS+=" +distill.common.post_data_mode_name=onehot"
	;;
*)
	echo "Unknown Data Mode" >&2
	exit 1
	;;
esac

$COMMAND $ARGS
