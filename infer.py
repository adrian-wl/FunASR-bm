from funasr import AutoModel
import time

# setting
dev_id = 0      ## bm1684x device id
input_path = "./vad_example.wav"        ## input audio path

# offline asr demo
start_time = time.time()
model = AutoModel(model="iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",    ## 语音识别模型
                  vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",                             ## 语音端点检测模型
                  punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",                ## 标点恢复模型
                  dev_id=dev_id,
                  )
res = model.generate(input=input_path,
                     batch_size_s=300,
                     hotword='魔搭')
end_time = time.time()
print(res)
print("total time:", end_time-start_time)
