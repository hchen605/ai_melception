# ai_melception

This repo is for the Melception project on GenAI Music Services. The target is controllable user-conditioned music accompaniment generation.     

Some demos may be seen at https://www.melception.com/ 

## Usage

### Setup
Install `arrange` and `re-inst` independently with `requirements_$.txt`    
`init` share the same env with `re-inst`    
p.s. if you find issue training figaro model, you may try `requirements_arrange_train.txt`
TBA: merge to single env.    

### Download pre-trained weights:   
Please download from
[Google Drive]     
-- Init: (https://drive.google.com/drive/folders/17yB-Oae_4eGKJmqRS-LB8PwE2rqwZrUu?usp=sharing)    
put the downloaded folder at `./init`    
-- Arragement: (https://drive.google.com/file/d/10E6F8RbRuSSg9wmYiPv6jyDsaFSSuyte/view?usp=drive_link)   
Put the downloaded folder at `./arrange`     
The weights for re-instrumentation are included in the repo.   
### Download data (for training):
```
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xzf lmd_full.tar.gz
```
### 0. Gen init output after upload:   
```
cd init   
CUDA_VISIBLE_DEVICES=$ python gen_sample_preload.py
```

### 1. Gen control description:   
```
CUDA_VISIBLE_DEVICES=$ FILE=arrange/data/Honestly_Piano_12.midi MODEL=figaro-expert CHECKPOINT=arrange/checkpoints/figaro-expert.ckpt python arrange/src/sample_desc.py
```   
Input midi can be set by `FILE`. Description text file $Output.txt will be generated in `arrange/desc` folder.    
User can edit the description text for fine-grained control over music generation.
   
### 2. Gen sample based on user control description:   
```
CUDA_VISIBLE_DEVICES=$ CONTROL=arrange/desc/description_honest.txt MODEL=figaro-expert CHECKPOINT=arrange/checkpoints/figaro-expert.ckpt python arrange/src/generate_sample.py
```    
`CONTROL` is the user attribute description text file.    
Find midi output at `sample` folder
    
### 3. Gen re-instrument sample:  

```
CUDA_VISIBLE_DEVICES=$ python re-inst/gen_reinstrumentation_sample.py
```
Find $output.mid at `demo/test` folder

## UI (TBA)   

Provide user a fine-grained bar-level attribute control.    
<img src="https://github.com/hchen605/ai_melception/blob/main/fig/melception_ui.png" width="700" height="400" />    


Provide user a fine-grained bar-level attribute control.

## Reference
    
Zhao, Jingwei, Gus Xia, and Ye Wang. "AccoMontage-3: Full-Band Accompaniment Arrangement via Sequential Style Transfer and Multi-Track Function Prior." arXiv preprint arXiv:2310.16334 (2023).     

von Rütte, Dimitri, et al. "Figaro: Generating symbolic music with fine-grained artistic control." arXiv preprint arXiv:2201.10936 (2022).

Zhao, Jingwei, Gus Xia, and Ye Wang. "Q&A: query-based representation learning for multi-track symbolic music re-arrangement." arXiv preprint arXiv:2306.01635 (2023).
