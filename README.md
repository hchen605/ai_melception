# ai_melception

This repo is for the Melception project on GenAI Music Services.

## Usage

### Setup
Install `arrange` and `re-inst` independently with `requirements_$.txt`

### Download pre-trained weights:   
Please download from
[Google Drive](https://drive.google.com/file/d/10E6F8RbRuSSg9wmYiPv6jyDsaFSSuyte/view?usp=drive_link)    
Put the folder at `./arrange`
### Download data (for training):
```
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xzf lmd_full.tar.gz
```
### 0. Gen init output after upload:   
TBA

### 1. Gen control description:   
```
CUDA_VISIBLE_DEVICES=$ FILE=arrange/data/Honestly_Piano_12.midi MODEL=figaro-expert CHECKPOINT=arrange/checkpoints/figaro-expert.ckpt python arrange/src/sample_desc.py
```   
$Output.txt will be generated in `arrange/desc` folder. User can edit the description text for fine-grained control over music generation.
   
### 2. Gen sample based on user control description:   
```
CUDA_VISIBLE_DEVICES=$ CONTROL=arrange/desc/description_honest.txt MODEL=figaro-expert CHECKPOINT=arrange/checkpoints/figaro-expert.ckpt python arrange/src/gen_sample.py
```    
Find $output.mid at `arrange/sample` folder. Usually it would take a few minutes to run based on input length.
    
### 3. Gen re-instrument sample:  
`cd re-inst`    
```
CUDA_VISIBLE_DEVICES=$ python re-inst/gen_reinstrumentation_sample.py
```
Find $output.mid at `demo/test` folder
