import pandas as pd

def prep_dataset(json_file):
    original_dataset_df = pd.read_json(json_file, lines=True)
    samples = original_dataset_df.to_dict()

    print(samples)

    if 'question' in samples and 'answer' in samples:
        text = samples['question'][0] + samples['answer'][0]
    elif 'instruction' in samples and 'response' in samples:
        text = samples['instruction'][0] + samples['response'][0]
    elif 'input' in samples and 'output' in samples:
        text = samples['input'][0] + samples['output'][0]
    else:
        text = samples['text'][0]

    prompt_template = '''### Question:
    {question}

    ### Answer:'''

    num_samples = len(samples['question'])
    finetuning_dataset = []

    for i in range(num_samples):
        question = samples['question'][i]
        answer = samples['answer'][i]
        text_with_prompt_template = prompt_template.format(question=question)
        finetuning_dataset.append({'question': text_with_prompt_template, 'answer': answer})

if __name__ == '__main__':
    #path_to_json_file = './sample_eva_dataset.jsonl'
    path_to_json_file = 'test.jsonl'
    prep_dataset(path_to_json_file)
