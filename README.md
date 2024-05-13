# ai_melception

This repo is for the Melception project on GenAI Music Services. The target is controllable user-conditioned music accompaniment generation.     

Some demos may be seen at https://www.melception.com/ 

## Usage

### Setup
* This repo runs under `python=3.9`

* Install required dependencies with `requirements_$.txt`    

* Run the following to download training data and pre-trained weights:
  - pre-trained weight for `init` and `arrange` will be saved in the corresponding folders
  ```
  python ./utils/prep.py --data_path=<dir_to_save_training_data>
  ```     
  The weights for re-instrumentation are included in the repo.   

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
