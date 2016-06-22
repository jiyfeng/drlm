# Latent Variable RNN #

Please refer to our [NAACL 2016 Paper](http://arxiv.org/abs/1603.01913) for more technical details.

## Getting Start ##

You need the [Boost C++ libraries](http://www.boost.org/) (>=1.56) to save/load word vocabulary and trained models. 

## Building ##

For Ubuntu user:

1. First you need to fetch the [cnn library](https://github.com/clab/cnn) into the same folder, then follow the instruction to get additional libraries and compile cnn.

2. To compile all DCLMs, run

    make

Note: I haven't tested this code in Windows and Mac

## Data Format ##

Please take a look the [data sample](data-sample.txt) file. In general, each row is one sentence for monologue (*or* utterance for dialogue) and its corresponding label, separated by **TAB**.

For test or the label is unknown, please use $-1$ instead of any integer $>0$

## Command Line Example ##

Please refer to [command-example.sh](command-example.sh)