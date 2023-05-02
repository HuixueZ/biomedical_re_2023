from logging import log
import openai
import json
import csv
import os
import random
import time
import torch


openai.api_key = "YOUR OPEN AI KEY"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.device_count())
    print('Available:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


#ROLE definition
SYSTEM_PROMPT = "You are a smart and intelligent Relation Recognition (RE) system. I will provide you the definition of the RE labels that you need to label the relation between entities masked with <e1><\e1> and <e2><\e2>, the sentence from where you label the RE and the output format with examples."
USER_PROMPT_1 = "Are you clear about your role?"
ASSISTANT_PROMPT_1 = "Sure, I'm ready to help you with your RE task. Please provide me with the necessary information to get started."


CHEMO_GUIDELINES_PROMPT = (
        "Relation Definition:\n"
        "The following are all the relation types between Chemical Entities Mentions (CEMs) and Gene and Protein Related Objects (GPROs)."
        "1. ACTIVATOR:  CEM that binds to a GPRO (typically a protein) and increases its activity. Conceptual synonyms are Stimulator, Inducer, Potentiator and Enhancer.\n"
        "2. ANTAGONIST:  CEM that reduces the action of another CEM, generally an agonist. Many antagonists act at the same receptor macromolecule as the agonist.\n"
        "3. COFACTOR: CEM that is required for a protein’s biological activity to happen. \n"
        "4. INHIBITOR: CEM that binds to a GPRO (typically a protein) and decreases its activity. \n"
        "5. MODULATOR: CEM that acts as allosteric modulator, compound that increases or decreases the action of an (primary or orthosteric) agonist or antagonist by combining with a distinct (allosteric or allotropic) site on the receptor macromolecule. If no information is available on whether the CEM activates or reduces GPRO activity, this general subclass should be assigned. \n"
        "6. NOT: occurrence of a chemical-protein interaction, without providing any further information on the specific negative CHEMPROT class or class. \n"
        "7. PART-OF: CEM that are structurally related to a GPRO: e.g. specific amino acid residues of a protein.\n"
        "8. REGULATOR: CEM that clearly regulates a GPRO, but for which there is no further information on whether the regulation is direct or indirect.\n"
        "9. SUBSTRATE: CEM upon which a GPRO (typically protein) acts. It should be understood as the substrate of a reaction carried out by a protein (“reactant”) or as transporter substrate\n"
        "\n"
        "Output Format:\n"
        "{{label}}\n"
        "There will have only one positive RE label for each input sentence."
        "\n"
        "Examples:\n"
        "\n")

YOUR_DATASET_GUIDE_PROMPT=()

#get chat gpt responds
def openai_chat_completion_response(final_prompt):
    response = openai.ChatCompletion.create(
        #max_tokens=MAX_TOKENS,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_1},
            {"role": "assistant", "content": ASSISTANT_PROMPT_1},
            {"role": "user", "content": final_prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip(" \n")



def get_guide_prompt(file, out_prompt=CHEMO_GUIDELINES_PROMPT,shot=1):
    """
    select random samples to form the 1shot prompts
    file format: [{'setence': input sentence, 'label': relation label}......]
    """
    # Not sure if we need to set this random.seed
    #random.seed(42)

    #select n samples according to n shot
    samples = random.sample(file, shot)
    for row in samples:
        sentence=row['sentence']
        label=row['label']
        prompt_sample="sentence: "+sentence+" \n output: {{"+label+"}}"
        out_prompt+=prompt_sample

    out_prompt+= "Sentence: {}\n Output: "
    return out_prompt

def finetuning(int_file,out_putfile):
    with open(out_putfile, "w", newline='') as outf:
        writer = csv.writer(outf)

        guide_prompt=get_guide_prompt(int_file)
        flag = 0
        for line in int_file:
            input_sentence=line['sentence']
            label=line['label']
            flag+=1
            # print("-------")
            print("-flag",flag)
            guide_prompt = guide_prompt.format(input_sentence)
            predictions = openai_chat_completion_response(guide_prompt)

            writer.writerow([input_sentence, predictions, label])

            if flag % 60 == 59:
                print(f"have a break!, flag: {flag}")
                time.sleep(60)

def read_file(path):
    "read your input and sample file and format it into [{'setence': input sentence, 'label': relation label}......]"
    pass


if __name__=="__main__":
    input_file_path = "YOUR INFILE PATH"
    output_file_path = " YOUR OUTFILE PATH"
    sample_file_path='YOUR FILE TO PROVIDE THE EXAMPLE'

    input_file=read_file(input_file_path)
    sample_file=read_file(sample_file_path)

    finetuning(input_file,output_file_path)
