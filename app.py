import spaces
import gradio as gr
import torch
from TTS.api import TTS
import os
os.environ["COQUI_TTS_AGREED"] = "1"

device = "cuda"

# 定义两个不同的模型
model_A = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
model_B = TTS("tts_models/multilingual/multi-dataset/your_model_B").to(device)  # 替换为实际的模型B路径

@spaces.GPU(enable_queue=True)
def clone(text, audio, model_choice):
    output_paths = []
    
    # 基于选择的模型生成语音
    if model_choice == "模型A" or model_choice == "两者对比":
        model_A.tts_to_file(text=text, speaker_wav=audio, language="en", file_path="./output_A.wav")
        output_paths.append("./output_A.wav")
    
    if model_choice == "模型B" or model_choice == "两者对比":
        model_B.tts_to_file(text=text, speaker_wav=audio, language="en", file_path="./output_B.wav")
        output_paths.append("./output_B.wav")
    
    # 返回一个或两个音频文件
    if len(output_paths) == 1:
        return output_paths[0], None
    else:
        return output_paths[0], output_paths[1]

iface = gr.Interface(fn=clone, 
                     inputs=[
                         gr.Textbox(label='要朗读的文字'),
                         gr.Audio(type='filepath', label='需要克隆的语音文件'),
                         gr.Radio(["模型A", "模型B", "两者对比"], label="选择模型", value="两者对比")
                     ], 
                     outputs=[
                         gr.Audio(type='filepath', label="模型A输出"),
                         gr.Audio(type='filepath', label="模型B输出")
                     ],
                     title='语音克隆模型对比',
                     description="""
                     请选择模型并上传参考语音，对比不同模型的效果。
                     请❤️本项目。
                     """,
                     theme = gr.themes.Base(primary_hue="teal", secondary_hue="teal", neutral_hue="slate"),
                     examples=[
                         ["这是测试文本", "./audio/长切1.wav", "两者对比"],
                         ["语音克隆测试", "./audio/长切2.wav", "两者对比"]
                     ])

iface.launch()