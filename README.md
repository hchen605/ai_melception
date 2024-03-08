# ai_melception

This repo is for the Melception project on GenAI Music Services.

## Usage
### Download pre-trained weights:   
TBA

### Gen control description:   
`CUDA_VISIBLE_DEVICES=$ FILE=arrange/data/Honestly_Piano_12.midi MODEL=figaro-expert CHECKPOINT=arrange/checkpoints/figaro-expert.ckpt python arrange/src/sample_desc.py`   
Find output at `desc` folder
   
### Gen sample based on user control:   
`CUDA_VISIBLE_DEVICES=$ CONTROL=arrange/desc/description_honest.txt MODEL=figaro-expert CHECKPOINT=arrange/checkpoints/figaro-expert.ckpt python arrange/src/gen_sample.py`    
Find output at `sample` folder
    
### Gen re-instrument sample:  
`cd re-inst`    
`CUDA_VISIBLE_DEVICES=$ python re-inst/gen_reinstrumentation_sample.py`
Find output at `demo/test` folder
