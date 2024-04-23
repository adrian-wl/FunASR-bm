from funasr import AutoModel
import time

# #offline
start_time = time.time()
model = AutoModel(model="iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
                  vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                  punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                  # spk_model="cam++"
                  dev_id=0,
                  )
# res = model.generate(input="/workspace/tpu-mlir/case/asr/asr_example.wav",
#             batch_size_s=300)
res = model.generate(input="/workspace/tpu-mlir/case/asr/vad_example.wav",
            batch_size_s=300,
            hotword='魔搭')
# res = model.generate(input="/workspace/tpu-mlir/case/asr/vad_example.wav",
#             batch_size_s=300)
end_time = time.time()
print(res)
print(end_time-start_time)
