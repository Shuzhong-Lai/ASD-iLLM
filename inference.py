import os

from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template, get_model_tokenizer
from swift.tuners import Swift

last_model_checkpoint = '/home/xxx/projects/llm/ASD-iLLM/output/checkpoint-32'

# 模型
model_id_or_path = 'Qwen/Qwen2.5-7B-Instruct'  # model_id or model_path
GEN_SYS_PROMPT = """"
你是一名经验丰富的儿童自闭症干预医生，你具备丰富的自闭症干预知识。以温暖亲切的语气，表现出共情和对儿童响应的肯定和表扬。请以自然的方式与儿童对话，避免过长的回应和儿童难以理解的内容，确保回应流畅且符合干预医生身份。现在你要和一名自闭症儿童进行主题对话，对话的主题是：{topic_content}, 请围绕主题开始对话。
"""

GEN_SYS_PROMPT1 = """
## 角色设定
您是一名经验丰富的自闭症儿童干预医生，遵循应用行为分析（ABA）原则为儿童提供有效支持。您的目标是帮助他们在主题对话中提高沟通与社交能力。

## 遵循原则
1. 请在交流中应用ABA原则，结合回合制教学原则（DTT）和情景教学原则（NET），在对话中注意以下三个要素：指令、辅助、强化
指令 - 清晰简单地提供指示，引导儿童围绕主题展开对话
辅助 - 在儿童需要帮助时，提供适度的言语支持，以促进正确的回应
强化 - 及时给予积极的反馈和表扬，以鼓励正确和积极的行为
2. 当儿童正确反应时，应该给予强化；当儿童无响应时，应当给予其恰当的辅助，促进其正确回应；当儿童错误反应时，不强化其错误反应，重发指令或给予其适当的辅助，促进其正确回应
3. 请保持温暖亲切的语气，充分表现出共情，对儿童的回应给予肯定和表扬。确保对话自然简洁，以便儿童能轻松理解。

## 开始对话
现在，您将与一名自闭症儿童进行主题对话，主题是：{topic_content}。请围绕此主题开始对话
"""

infer_backend = 'pt'

# 生成参数
max_new_tokens = 2048
temperature = 0
stream = False

def infer_stream(engine: InferEngine, infer_request: InferRequest):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature, stream=True)
    gen = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    print(f'query: {query}\nresponse: ', end='')
    for resp_list in gen:
        print(resp_list[0].choices[0].delta.content, end='', flush=True)
    print()

def infer(engine: InferEngine, infer_request: InferRequest):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature, logprobs=True)
    resp_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    response = resp_list[0].choices[0].message.content
    print(f'query: {query}')
    print(f'response: {response}')


class LLMInference:
    def __init__(self, model_id_or_path, checkpoint_path=None, model_type='qwen2_5',system='You are a helpful assistant.', max_new_tokens=512, temperature=0.2, stream=False):
        self.model_id_or_path = model_id_or_path
        self.checkpoint_path = checkpoint_path
        self.system_prompt = system
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.stream = stream
        self.model_type = model_type

        self.engine = PtEngine(model_id_or_path,model_type=model_type, adapters=checkpoint_path)
        # self.engine = PtEngine(model_id_or_path, adapters=checkpoint_path, model_type='glm_edge')
        # self.template = get_template(self.engine.model.model_meta.template, self.engine.tokenizer, default_system=system, response_prefix='<think>\n\n</think>\n\n')   
        self.template = get_template(self.engine.model.model_meta.template, self.engine.tokenizer, default_system=system)  
        self.engine.default_template = self.template

    def infer(self, query, logprobs=False):
        query = InferRequest(messages=query)
        request_config = RequestConfig(max_tokens=self.max_new_tokens, temperature=self.temperature, stream=self.stream, logprobs=logprobs)
        resp_list = self.engine.infer([query], request_config)
        # response = resp_list[0].choices[0].message.content
        return resp_list[0]

    def infer_stream(self, query):
        query = InferRequest(messages=query)
        request_config = RequestConfig(max_tokens=self.max_new_tokens, temperature=self.temperature, stream=True)
        gen = self.engine.infer([query], request_config)
        query = query.messages[0]['content']
        print(f'query: {query}\nresponse: ', end='')
        for resp_list in gen:
            print(resp_list[0].choices[0].delta.content, end='', flush=True)
        print()
        

def test_llm_infer():
    model_id_or_path = 'Qwen/Qwen2.5-7B-Instruct'  # model_id or model_path
    lora_checkpoint = '/home/xxx/projects/llm/ASD-iLLM/output/Qwen2_5-7B-Instruct/04-24-16:52:44/checkpoint-2475'
    # lora_checkpoint = None
    template_type = None
    default_system = None
    # 加载模型和对话模板
    model, tokenizer = get_model_tokenizer(model_id_or_path, model_type='glm_edge')
    model = Swift.from_pretrained(model_id=model_id_or_path, model=model, adapter_name=lora_checkpoint)
    # model = Swift.from_pretrained(model)
    template_type = template_type or model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=default_system)
    engine = PtEngine.from_model_template(model, template, max_batch_size=2)
    request_config = RequestConfig(max_tokens=512, temperature=0.2)
    history = [{
        "role" : "system", 
        "content" : GEN_SYS_PROMPT1.format(topic_content="颜色")
    }]
    infer_request = [
        InferRequest(messages=history)
    ]
    infer_request = [InferRequest(messages=history)]  
    resp_list = engine.infer(infer_request, request_config)  
    doctor_first = resp_list[0].choices[0].message.content  
    print(f"医生：{doctor_first}")  
    history.append({"role": "assistant", "content": doctor_first})

    while True:  
        user_content = input().strip()  
        if user_content.lower() == 'quit':  
            print("退出")  
            break  
        history.append({"role": "user", "content": user_content})  

        infer_request = [InferRequest(messages=history)]  
        resp_list = engine.infer(infer_request, request_config)  
        response = resp_list[0].choices[0].message.content  
        print(f"医生：{response}")  
        history.append({"role": "assistant", "content": response})  
    

if __name__ == '__main__':
    test_llm_infer()
    