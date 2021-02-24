#!/bin/bash

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5
wave_data=raw_wave
# path of CSJ corpus
CSJDATATOP=/home/lijian/storage/corpus/CSJ_RAW
CSJVER=usb  ## Set your CSJ format (dvd or usb).
            ## Usage    :
            ## Case DVD : We assume CSJ DVDs are copied in this directory with the names dvd1, dvd2,...,dvd17.
            ##            Neccesary directory is dvd3 - dvd17.
            ##            e.g. $ ls $CSJDATATOP(DVD) => 00README.txt dvd1 dvd2 ... dvd17
            ##
            ## Case USB : Neccesary directory is MORPH/SDB and WAV
            ##            e.g. $ ls $CSJDATATOP(USB) => 00README.txt DOC MORPH ... WAV fileList.csv
            ## Case merl :MERL setup. Neccesary directory is WAV and sdb
nj=32
# Optional train_config
# 1. conf/train_transformer_large.yaml: Standard transformer
train_config=conf/train_conformer.yaml
checkpoint=
cmvn=true
do_delta=false

dir=exp/sp_spec_aug
# where to dump the extracted audio clips defined in segments
save_clips_dir=/home/lijian/storage/corpus/csj_audio_clips_extracted_from_segments
feat_dir=raw_wav

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
# maybe you can try to adjust it if you can not get close results as README.md
min_epoch=15
max_epoch=65535
average_num=10
decode_modes="attention_rescoring ctc_greedy_search ctc_prefix_beam_search attention"

. tools/parse_options.sh || exit 1;

# bpemode (unigram or bpe)
nbpe=10000 # according to the config of neural_sp's CSJ example
bpemode=bpe
character_coverage=0.9995

set -e
set -u
set -o pipefail

train_set=train_nodup
train_dev=dev
recog_set="eval1 eval2 eval3"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    if [ ! -e data/csj-data/.done_make_all ]; then
        echo "CSJ transcription file does not exist"
        local/csj_make_trans/csj_autorun.sh ${CSJDATATOP} data/csj-data $CSJVER
    fi

    local/csj_data_prep.sh data/csj-data || exit 1;

    for x in eval1 eval2 eval3; do
        local/csj_eval_data_prep.sh data/csj-data/eval ${x} || exit 1;
    done

    utils/subset_data_dir.sh --first data/train 4000 data/$train_dev # 6hr 31min
    n=$[`cat data/train/segments | wc -l` - 4000]
    utils/subset_data_dir.sh --last data/train $n data/train_nodev

    utils/data/remove_dup_utts.sh 300 data/train_nodev data/train_nodup  # 233hr 36min
    
    # Remove <sp> and POS tag, and lowercase
    for x in $train_set $train_dev ${recog_set}; do
        local/remove_pos.py data/${x}/text | nkf -Z > data/${x}/text.tmp
        mv data/${x}/text.tmp data/${x}/text
    done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    # For wav feature, just copy the data. Fbank extraction is done in training
    mkdir -p $wave_data
    for x in $train_set $train_dev ${recog_set}; do
        cp -r data/$x $wave_data
        if [ -f $wave_data/$x/segments ]; then
            if [ -d $save_clips_dir/$x ]; then
              echo "$0: $save_clips_dir/$x already exists, now delete it."
              rm -r $save_clips_dir/$x || exit 1;
            fi
            mkdir -p $save_clips_dir/$x || exit 1;
            awk -v foo1=${save_clips_dir} -v foo2=$x '{print $1 " " foo1 "/" foo2 "/" $1 ".wav"}' $wave_data/$x/segments > $save_clips_dir/${x}.scp
            extract-segments scp:data/$x/wav.scp data/$x/segments ark:- | wav-copy ark:- scp:$save_clips_dir/${x}.scp
            cp $save_clips_dir/${x}.scp $wave_data/$x/wav.scp
            rm $wave_data/$x/segments
        fi
        num_segments=$(cat $wave_data/$x/wav.scp | wc -l)
        num_audios=$(find $save_clips_dir/$x/ -name "*.wav" | wc -l)
        if [ $num_segments -ne $num_audios ]; then
            echo "$0: It seems that not all segments got converted into audio files successfully!" \
                 "($num_segments != $num_audios)"
            if (( num_audios < num_segments - num_segments/20 )); then
                echo "$0: Less than 95% the segments were successfully generated." \
                     "Probably a serious error."
                exit 1;
            fi
            cp $wave_data/$x/wav.scp $wave_data/$x/wav.scp.old
            find $save_clips_dir/$x/ -name "*.wav" | awk -F'/' '{print $NF " " $0}' | sed 's|.wav | |g' | sort -u > $wave_data/$x/wav.scp
            utils/fix_data_dir.sh $wave_data/$x || exit 1;
            rm $wave_data/$x/wav.scp.old
        fi
    done

    tools/compute_cmvn_stats.py --num_workers 64 --train_config $train_config \
        --in_scp $wave_data/$train_set/wav.scp \
        --out_cmvn $wave_data/$train_set/global_cmvn
fi


dict=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p $wave_data/lang_char/

    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1

    # we borrowed these code and scripts which are related bpe from ESPnet.
    cut -f 2- -d" " $wave_data/${train_set}/text > $wave_data/lang_char/input.txt
    tools/spm_train --input=$wave_data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} \
      --input_sentence_size=100000000 --character_coverage=$character_coverage
    tools/spm_encode --model=${bpemodel}.model --output_format=piece < $wave_data/lang_char/input.txt | tr ' ' '\n' | \
      sort | uniq -c | sort -n -k1 -r | sed -e 's/^[ ]*//g' | cut -d " " -f 2 | grep -v '^\s*$' | awk '{print $1 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict # <eos>
    wc -l ${dict}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Prepare wenet requried data
    echo "Prepare data, prepare requried format"
    for x in $train_dev ${recog_set} $train_set ; do
        tools/format_data.sh --nj ${nj} \
            --feat-type wav --feat $wave_data/$x/wav.scp --bpecode ${bpemodel}.model \
            $wave_data/$x ${dict} > $wave_data/$x/format.data.tmp

        tools/remove_longshortdata.py \
            --min_input_len 0.5 \
            --max_input_len 20 \
            --max_output_len 400 \
            --max_output_input_ratio 10.0 \
            --data_file $wave_data/$x/format.data.tmp \
            --output_data_file $wave_data/$x/format.data
    done

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # Training
    mkdir -p $dir
    INIT_FILE=$dir/ddp_init
    rm -f $INIT_FILE # delete old one before starting
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    # Use "nccl" if it works, otherwise use "gloo"
    dist_backend="nccl"
    cmvn_opts=
    $cmvn && cmvn_opts="--cmvn $wave_data/${train_set}/global_cmvn"
    # train.py will write $train_config to $dir/train.yaml with model input
    # and output dimension, train.yaml will be used for inference or model
    # export later
    for ((i = 0; i < $num_gpus; ++i)); do
    {
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
        python wenet/bin/train.py --gpu $gpu_id \
            --config $train_config \
            --train_data $wave_data/$train_set/format.data \
            --cv_data $wave_data/$train_dev/format.data \
            ${checkpoint:+--checkpoint $checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $num_gpus \
            --ddp.rank $i \
            --ddp.dist_backend $dist_backend \
            --num_workers 1 \
            $cmvn_opts
    } &
    done
    wait
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # Test model, please specify the model you want to test by --checkpoint
    cmvn_opts=
    $cmvn && cmvn_opts="--cmvn data/${train_set}/global_cmvn"
    # TODO, Add model average here
    mkdir -p $dir/test
    if [ ${average_checkpoint} == true ]; then
        decode_checkpoint=$dir/avg_${average_num}.pt
        echo "do model average and final checkpoint is $decode_checkpoint"
        python wenet/bin/average_model.py \
            --dst_model $decode_checkpoint \
            --src_path $dir  \
            --num ${average_num} \
            --min_epoch $min_epoch \
            --max_epoch $max_epoch \
            --val_best
    fi
    # static dataloader is need for attention_rescoring decode
    sed -i 's/dynamic/static/g' $dir/train.yaml
    # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
    # -1 for full chunk
    decoding_chunk_size=
    ctc_weight=0.5
    for test in $recog_set; do
    for mode in ${decode_modes}; do
    {
        test_dir=$dir/${test}_${mode}
        mkdir -p $test_dir
        python wenet/bin/recognize.py --gpu -1 \
            --mode $mode \
            --config $dir/train.yaml \
            --test_data $wave_data/$test/format.data \
            --checkpoint $decode_checkpoint \
            --beam_size 10 \
            --batch_size 1 \
            --penalty 0.0 \
            --dict $dict \
            --result_file $test_dir/text_bpe \
            --ctc_weight $ctc_weight \
            ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
        tools/spm_decode --model=${bpemodel}.model --input_format=piece < $test_dir/text_bpe | sed -e "s/▁/ /g" > $test_dir/text
        python tools/compute-wer.py --char=1 --v=1 \
            $wave_data/$test/text $test_dir/text > $test_dir/wer
    } &
    done
    done
    wait

fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    # Export the best model you want
    python wenet/bin/export_jit.py \
        --config $dir/train.yaml \
        --checkpoint $dir/avg_${average_num}.pt \
        --output_file $dir/final.zip
fi

