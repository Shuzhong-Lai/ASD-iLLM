# ASD-iLLM
**"ASD-iLLM:An Intervention Large Language Model for Autistic Children based on Real Clinical Dialogue Intervention Dataset" has been accepted by EMNLP2025 Findings!**

## Installation

```
pip install -r requirements.txt
```

build your own .env to add your API_KEY

```
LLM_API_KEY="sk-xxx"
```

## Framework
![ASD-iLLM Framework](/figs/EMNLP2025.png)
we propose a comprehensive framework for training LLMs to conduct dialogue interventions in accordance with the principles of Applied Behavior Analysis (ABA) which is commonly used by clinicians. Specifically, we collected clinical recordings of dialogue interventions for autistic children and constructed the topic dialogue dataset ASD-iLLM-8k. By incorporating the system prompt based on the ABA and ASD-iLLM-8k dataset, we fine-tuned LLMs to develop ASD-iLLM. We also proposed a role-play strategy in which LLMs act as autistic children to comprehensively evaluate the doctor model’s capabilities at the dialogue level.
## Dataset
![ASD-iLLM Framework](/figs/Data_new.png)
We collected 64.2 hours of audio data and transcribed it into 751 instances of topic multi-turn dialogues. After cleaning, we obtained 287 high-quality real multi-turn dialogues. We then used GPT-4.1 for data synthesis, resulting in a total of 8,035 instances of topic multi-turn dialogues. We released this dataset as the ASD-iLLM-8k. 

If you would like to use the open-source dataset from this paper, please click the link below to fill out the application form and guarantee that you will use it solely for academic research.
## Model
Our model weights will be shared on Huggingface, named ASD-iLLM. Here is the link : https://huggingface.co/Shuzhong/ASD-iLLM
## Result
![ASD-iLLM Framework](/figs/Automatic_Evaluation.png)
The 100 multi-turn dialogues from the ASD-iLLM-8k test set are used to create the 1890 single-sentence prediction tasks for sentence-level evaluation. The evaluation results are shown in Table 3. We observed a significant improvement in various metrics for the 7b model fine-tuned on the ASD-iLLM-8k dataset. All fine-tuned models performed better than existing models across most metrics. This indicates that the model's conversational style and intervention strategy at the sentence level closely resemble authentic clinical dialogue interventions.
![ASD-iLLM Framework](/figs/Human_Evaluation.png)
For dialogue-level evaluation, we used the topics from the test set for the role-play strategy to generate multi-turn dialogues. Meanwhile, the multi-turn dialogues of the test set were scored as a baseline to assess their authority through comparison. The results are shown in Table 4, indicating that the various capabilities of ASD-iLLM closely align with those of clinical intervention doctors.
## Case Study
![ASD-iLLM Framework](/figs/Case.png)
Figure illustrates a case of dialogue intervention about the season conducted by ASD-iLLM. We can observe that ASD-iLLM's dialogue style closely resembles clinical interventions, with expressions that are concise and easy for children to understand. Moreover, it follows ABA principles by providing appropriate instruction repetition, assistance, and reinforcement to different types of child responses. This fully demonstrates ASD-iLLM's potential to effectively replace intervention doctors in future dialogue interventions.

## Reference
If you find our paper is valuable, please cite us and feel free to contact us!
```
@inproceedings{lai2025asd,
  title={ASD-iLLM: An Intervention Large Language Model for Autistic Children based on Real Clinical Dialogue Intervention Dataset},
  author={Lai, Shuzhong and Li, Chenxi and Lai, Junhong and Zhong, Yucun and Yan, Chenyu and Li, Xiang and Li, Haifeng and Pan, Gang and Yao, Lin and Wang, Yueming},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2025},
  pages={8058--8079},
  year={2025}
}
```
