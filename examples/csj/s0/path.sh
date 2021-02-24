export PATH=$PWD:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../:$PYTHONPATH

# NOTE(lijian): We need to use Kaldi's binary file 'extract-segments' and 'wav-copy'
#                 to extract segments and save them into separate files.
export KALDI_ROOT=/home/lijian/tools/kaldi
export PATH=${KALDI_ROOT}/src/featbin:${PATH}

# NOTE(lijian): You'll miss many words when using 'tools/spm_encode' to generate dict
#                 if you don't include this.
export LC_ALL=C
