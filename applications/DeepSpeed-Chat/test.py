import datetime
import os
import time
from itertools import chain

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM


# model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
from transformers.testing_utils import CaptureLogger


def test_llama():
    tokenizer = AutoTokenizer.from_pretrained("/disk1/models/llama/7B/")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("/disk1/models/llama/7B/")

    input_sentence = "Human: Please tell me about Microsoft in a few sentence? Assistant:"
    chosen_token = tokenizer(input_sentence,
                             max_length=512,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    device = torch.device("cuda")
    chosen_token.to(device)
    model.to(device)
    generate_ids = model.generate(input_ids=chosen_token['input_ids'],
                                  attention_mask=chosen_token['attention_mask'],
                                      num_beams=1,
                                      num_beam_groups=1,
                                      do_sample=False,
                                      num_return_sequences=1,
                                      max_new_tokens=100)

    result = tokenizer.batch_decode(generate_ids,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
    print(result)

def split_model():
    model = AutoModelForCausalLM.from_pretrained("/disk1/work/xiaym/models/dsc/vicuna/actor")
    model.half()
    LlamaForCausalLM.save_pretrained(model, save_directory="/disk1/work/xiaym/models/dsc/vicuna/actor")

def print_model_graph():
    model = AutoModelForCausalLM.from_pretrained("/cloud-model/huggingFace/Models/llama-2/7B")
    # model = AutoModelForCausalLM.from_pretrained("/disk1/models/falcon/falcon-7b", trust_remote_code=True)
    print(sum(p.numel() for p in model.parameters()))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model)

def test_falcon():
    tokenizer = AutoTokenizer.from_pretrained("/disk1/models/falcon/falcon-7b", trust_remote_code=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model="/disk1/models/falcon/falcon-7b",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    sequences = pipeline(
        "Please tell me about Microsoft in a few sentence?",
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


def mask_labels(conversation, target, tokenizer):
    sep = " Assistant: "

    total_len = int(target.ne(tokenizer.pad_token_id).sum())

    rounds = conversation.split('</s>')
    cur_len = 1
    target[:cur_len] = -100
    for i, rou in enumerate(rounds):
        if rou == "":
            break

        parts = rou.split(sep)
        if len(parts) != 2:
            break
        parts[0] += sep
        round_len = len(tokenizer(rou).input_ids)
        instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        target[cur_len: cur_len + instruction_len] = -100

        cur_len += round_len
    target[cur_len:] = -100

    if cur_len < tokenizer.model_max_length:
        if cur_len != total_len:
            target[:] = -100
            print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                f" (ignored)"
            )
class a:

    def __init__(self):
        self.a1 = [1, 3]
        self.b1 = [2, 4]


def test_copy():

    a0 = a()

    params = [
        (a0.a1, a0.b1)
    ]
    for dst, src in params:
        print("dst:", dst)
        print("src:", src)
        dst = src
        print("dst:", dst)

    print(a0.a1)

    a0.a1 = a0.b1

    print(a0.a1)


def load_csv():
    from datasets import load_dataset
    dataset = load_dataset("csv", data_files={"train": "/disk1/work/xiaym/dev/Llama2-Chinese/data/train_sft.csv",
                                              "test": "/disk1/work/xiaym/dev/Llama2-Chinese/data/dev_sft.csv"})

    print(dataset)

def load_txt():
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    tokenizer = AutoTokenizer.from_pretrained("/cloud-model/huggingFace/Models/chinese-alpaca-2-13b", trust_remote_code=True)
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples["text"])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output
    raw_dataset = load_dataset("text", data_files="/disk1/work/xiaym/data/pretain/蔡英文.txt", cache_dir="./temp", keep_in_memory=False)
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns="text",
        load_from_cache_file=True,
        keep_in_memory=False,
        cache_file_names={k: os.path.join("./temp", 'tokenized.arrow') for k in raw_dataset},
        desc="Running tokenizer on dataset",
    )

    block_size = 1024
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped_datasets = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=1,
        load_from_cache_file=True,
        keep_in_memory=False,
        cache_file_names={k: os.path.join("./temp", 'grouped.arrow') for k in tokenized_dataset},
        desc=f"Grouping texts in chunks of {block_size}",
    )
    processed_dataset = grouped_datasets

if __name__ == '__main__':
    load_txt()
    # load_csv()
    # test_copy()
    # t0 = time.time()
    # print(datetime.datetime.now())
    # # split_model()
    # print_model_graph()
    # # test_falcon()
    # t1 = time.time()
    # print(datetime.datetime.now())
    # print(f"{(t1-t0):.2f}s")
    # print(transformers.file_utils.is_torch_bf16_available())

#     from transformers import AutoTokenizer
#
#     tokenizer = AutoTokenizer.from_pretrained(
#         '/disk1/models/llama/7B',
#         # model_max_length=2048,
#         padding_side="right",
#         use_fast=False,
#     )
#     print(tokenizer.model_max_length)
#
#     tokenizer.pad_token = tokenizer.unk_token
#
#     conversation = """Human: which are good professional entry points to the digital and information technology world for someone without prior experience? Assistant: There are several professional entry points to the digital and information technology world for someone without prior experience. Here are a few options:
#
# 1. IT Support Technician: This role involves troubleshooting and resolving technical issues for users in an organization. It's a good entry-level job that can help you gain experience and learn about different aspects of IT.
# 2. Junior Web Developer: As a junior web developer, you will assist with coding, testing, and debugging websites. This role is a good starting point for someone who wants to learn web development.
# 3. Data Analyst: A data analyst collects and analyzes data to help businesses make better decisions. This role requires skills in Excel, SQL, and data visualization tools.
# 4. Digital Marketing Assistant: This role involves assisting with the creation and implementation of digital marketing campaigns. You will need to have skills in social media marketing, SEO, and content creation.
# 5. Cybersecurity Analyst: As a cybersecurity analyst, you will work to protect an organization's computer networks and systems from cyber threats. This role requires knowledge of security protocols, firewalls, and encryption techniques.
# 6. Project Coordinator: A project coordinator assists project managers in organizing and executing projects. This role requires strong communication and organizational skills.
# 7. Technical Writer: Technical writers create documentation for software, hardware, and other technical products. This role requires strong writing and communication skills.
#
# These are just a few examples of entry-level roles in the digital and information technology world. You can also consider getting certified in specific technologies like AWS, Microsoft Azure, Google Cloud, or CompTIA to improve your job prospects.</s> Human: what no code technologies can generate inconme for someone without prior experience Assistant: There are several no-code technologies that can generate income for someone without prior experience. Here are a few options:
#
# 1. Website builders: No-code website builders like Wix, Squarespace, and Webflow allow anyone to create a website without needing to know how to code. You can offer website creation services to businesses or individuals looking to establish an online presence.
# 2. E-commerce platforms: No-code e-commerce platforms like Shopify and BigCommerce allow you to create an online store and sell products without needing to know how to code. You can source products from suppliers or create your own products to sell.
# 3. Social media automation tools: Social media automation tools like Hootsuite and Buffer allow you to manage multiple social media accounts and schedule posts in advance. You can offer social media management services to businesses or individuals looking to improve their online presence.
# 4. Mobile app builders: No-code mobile app builders like BuildFire and Appy Pie allow you to create mobile apps without needing to know how to code. You can offer mobile app creation services to businesses or individuals looking to create a mobile presence.
# 5. Online course platforms: No-code online course platforms like Thinkific and Teachable allow you to create and sell online courses without needing to know how to code. You can create courses on a subject you're knowledgeable about and sell them to people interested in learning.
#
# These are just a few examples of no-code technologies that can generate income for someone without prior experience. However, keep in mind that building a successful business or generating income with these tools still requires effort, marketing, and a willingness to learn and improve your skills.</s> Human: elaborate on low or no code technologies for building integration and or chat bots Assistant: Low or no code technologies have made it easier for non-technical individuals to build integrations and chatbots without needing to know how to code. Here are some examples of these tools:
#
# 1. Zapier: Zapier is a low-code integration platform that allows you to automate workflows between different apps. You can use Zapier to connect apps like Gmail, Slack, Trello, and others, and automate tasks such as sending notifications, creating calendar events, or saving data to a spreadsheet.
# 2. Integromat: Integromat is another low-code integration platform that allows you to automate workflows between different apps. It offers more advanced features than Zapier, such as the ability to create custom functions and use conditional statements in your workflows.
# 3. Microsoft Power Automate: Microsoft Power Automate is a low-code workflow automation tool that allows you to automate tasks and integrate different apps and services. It offers a wide range of connectors to popular apps and services, and allows you to create custom workflows with a visual interface.
# 4. Dialogflow: Dialogflow is a no-code chatbot development platform from Google. It allows you to create conversational interfaces for your website or mobile app, and integrate with popular messaging platforms like Facebook Messenger, Slack, or Telegram. Dialogflow uses natural language processing (NLP) to understand and respond to user requests.
# 5. ManyChat: ManyChat is a no-code chatbot development platform that allows you to create chatbots for Facebook Messenger, WhatsApp, and SMS. It offers a drag-and-drop interface to create conversation flows, and allows you to send broadcasts, drip campaigns, and automated responses.
# 6. Tars: Tars is another no-code chatbot development platform that allows you to create conversational interfaces for your website or mobile app. It offers a visual builder to create conversation flows, and allows you to integrate with popular messaging platforms like Facebook Messenger, WhatsApp, and others.
#
# These low or no-code technologies can help you build integrations and chatbots quickly and easily, without requiring you to know how to code. However, keep in mind that building successful integrations and chatbots still requires a good understanding of your users' needs, as well as a solid strategy and design.</s> Human: how can i build a telegram bot Assistant: Telegram bots can be built using the Telegram Bot API, which provides a set of methods and interfaces to interact with the Telegram server. Here are the steps to build a Telegram bot:
#
# 1. Create a Telegram account if you don't already have one.
# 2. Open Telegram and search for the BotFather bot.
# 3. Start a conversation with BotFather and follow the instructions to create a new bot. You will need to choose a name and a username for your bot.
# 4. Once you've created your bot, BotFather will provide you with an API token. Save this token, as you'll need it to interact with the Telegram server.
# 5. Choose a programming language you are comfortable with to build the bot. You can use languages such as Python, JavaScript, or Ruby, among others.
# 6. Use a Telegram Bot API library for your chosen programming language to interact with the Telegram server. Some popular libraries include python-telegram-bot for Python, node-telegram-bot-api for Node.js, and telegram-bot-ruby for Ruby.
# 7. Write the code for your bot, including the commands and actions you want it to perform. For example, you could create a command to send a message, a command to retrieve information, or a command to perform an action based on user input.
# 8. Deploy your bot to a server or hosting platform so that it's always available to receive messages and respond to user requests.
#
# Once your bot is deployed and running, you can test it by sending messages to it on Telegram. You can also add your bot to Telegram groups or channels to allow other users to interact with it.</s>"""
#     conversation = conversation.replace("\nAssistant:", " Assistant:")
#     conversation_token = tokenizer(conversation,
#                                    max_length=256,
#                                    padding="max_length",
#                                    truncation=True,
#                                    return_tensors="pt")
#
#     print(conversation_token)
#
#     target = conversation_token["input_ids"].clone().squeeze(
#                     0)
#
#     print(target)
#     mask_labels(conversation, target, tokenizer)
#     print("===================")
#     print(target)

