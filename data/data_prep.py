import json
import re
import argparse
from transformers import LlamaTokenizer
import random
import sys
sys.path.append('/home/tianyueo/agent/generate_history')
from generate_history import HistoryGenerator

import numpy as np



def find_number_before_target(s_1, s_2):
    pattern = r"\[(\d+)\]\s+" + re.escape(s_2)
    match = re.search(pattern, s_1)
    if match:
        return int(match.group(1))
    else:
        return None


def extract_lines(s, x, context_front, context_back, total, tokenizer):
    lines_1 = s.split("\\n")
    lines_2 = s.split("\n")
    lines = (lines_1 if len(lines_1) > len(lines_2) else lines_2)
    cur_cnt = 0
    # print('x',x)
    # print('lines_len', len(lines))
    # print(lines[0])

    x_index = -1
    for i, line in enumerate(lines):
        if f"[{x}]" in line:
            x_index = i
            break
    
    if x_index == -1:
        print(f"[{x}]" in s)
        raise ValueError(f"[{x}] not found in string.")

# random chance to append new lines before/after
    start_index = int(max(0, x_index - context_front - 1))
    end_index = int(min(len(lines), x_index + context_back + 2))
    x_lines = lines[start_index:end_index]
    cur_cnt += len(tokenizer.encode("\n".join(x_lines)))

    # if cur_cnt > total:
        # print('start_index', start_index)
        # print('end_index',end_index)
        # print("\n".join(x_lines))
    remaining_lines = lines[:start_index] + lines[end_index:]

    sampled_lines = []

    # while cur_cnt < total:
    #     # print('in_loop_cur_cnt', cur_cnt)  
    #     sampled_line = random.sample(remaining_lines, 1)[0]
    #     cur_cnt += len(tokenizer.encode(sampled_line))
    #     sampled_lines.append(sampled_line)
    if len(sampled_lines) > 0:
        sampled_lines.pop()

    # print(sampled_lines)

    result_lines = []
    for line in lines:
        if line in x_lines or line in sampled_lines:
            result_lines.append(line)

    print('cur_cnt', cur_cnt)
    return "\n".join(result_lines)


def parse_op(data):
    err_cnt = {'not_in_tree':0, 'repeated':0}
    purge_set = set()
    result = {}
    for item in data:
        annotation_id = item["annotation_id"]
        action_uid = item["action_uid"]

        if annotation_id not in result:
            result[annotation_id] = {}
        
        # if str(item["axt_nodeid"]) not in item["axt"]:
        #     print(item)
        #     assert False

        try:
            if type(item["operation"]) == str:
                sub_data = json.loads(item["operation"].replace("'", '"'))
            else:
                sub_data = item["operation"]
            if "original_op" in sub_data:
                if sub_data["original_op"] == "SELECT":
                    target_str = sub_data["value"]
                    tmp_s = item["axt"]
                    start_pos = tmp_s.find(f"[{item['axt_nodeid']}]")
                    substring = tmp_s[start_pos:]
                    ax_node_id = find_number_before_target(substring, target_str)
                    item["operation"] = "CLICK"
                    item["axt_nodeid"] = ax_node_id
                elif sub_data["original_op"] == "TYPE":
                    item["operation"] = (
                        sub_data["original_op"] + "$" + sub_data["value"]
                    )
                else:
                    item["operation"] = sub_data["original_op"]

        except json.JSONDecodeError:
            if "Locator" in item["operation"]:
                if "element_handleget_by_test_id" in item["operation"]:
                    pattern = r"nth\((\d+)\)"
                    match = re.search(pattern, item["operation"])
                    choice = int(match.group(1)) if match else 1
                    tmp_s = item["axt"]
                    start_pos = tmp_s.find(f"[{item['axt_nodeid']}]")
                    substring = tmp_s[start_pos:]
                    numbers = re.findall(r"\[(\d+)\]", substring)
                    item["axt_nodeid"] = int(numbers[choice])
                    item["operation"] = "CLICK"
                elif (
                    "clickget_by_test_id" or "pressget_by_test_id" in item["operation"]
                ):
                    item["operation"] = "CLICK"
        if "CLICK" in item["operation"] or "TYPE" in item["operation"]:
            if annotation_id in result and action_uid in result[annotation_id]:
                purge_set.add(action_uid)
                err_cnt['repeated']+=1
            elif str(item["axt_nodeid"]) not in item["axt"]:
                print('node',item["axt_nodeid"])
                print('axt',item["axt"])
                purge_set.add(action_uid)
                err_cnt['not_in_tree']+=1
            else:
                result[annotation_id][action_uid] = item
        print(err_cnt)
    return result, purge_set


def convert_for_training(result, purge_set):
    chat_processor = HistoryGenerator(model='palm2')
    id = 0
    json_list = []

    for annotation_id in result:
        action_hist = []
        for action_id in result[annotation_id]:
            if action_id in purge_set:
                continue
            json_dict = {}
            prompt_str = "OBJECTIVE :" + result[annotation_id][action_id]["intent"] + " on " + result[annotation_id][action_id]["website"] + ',' + "OBSERVATION : " +  result[annotation_id][action_id]["axt"] + ',' + "PREVIOUS ACTION : " + str(action_hist)
            subdict = {
                "intent": result[annotation_id][action_id]["intent"]
                + " on "
                + result[annotation_id][action_id]["website"],
                "obs": result[annotation_id][action_id]["axt"],
                "action_hist": action_hist,
            }
            json_dict["id"] = id
            id += 1
            json_dict["conversations"] = [
                {"from": "human", "value": prompt_str},
                {
                    "from": "gpt",
                    "value": (
                        result[annotation_id][action_id]["operation"]
                        + " ["
                        + str(result[annotation_id][action_id]["axt_nodeid"])
                        + "]"
                        if "$" not in result[annotation_id][action_id]["operation"]
                        else result[annotation_id][action_id]["operation"].split("$")[0]
                        + " ["
                        + str(result[annotation_id][action_id]["axt_nodeid"])
                        + "] ["
                        + result[annotation_id][action_id]["operation"].split("$")[1]
                        + "]"
                    ),
                },
            ]
            json_list.append(json_dict)

            pattern = (
                r"\["
                + str(result[annotation_id][action_id]["axt_nodeid"])
                + r"\](.*?)\s*\[\d+\]"
            )
            matches = re.search(pattern, result[annotation_id][action_id]["axt"])
            pattern_2 = (

                r"\["
                + str(result[annotation_id][action_id]["axt_nodeid"])
                + r"\].*"
            )
            matches_2 = re.search(pattern_2, result[annotation_id][action_id]["axt"])
            if matches:
                action_descrip = matches.group(1)
            elif matches_2 and len(matches_2.group(0)) < 100:
                action_descrip = matches_2.group(0)
            else:
                # print(annotation_id)
                # print(action_id)
                print(pattern)
                print(result[annotation_id][action_id]["axt_nodeid"])
                print(result[annotation_id][action_id]["axt"])
                print("no match")
                # assert False
                continue
            
            action_descrip = ''.join([ch for ch in action_descrip if ch.isalpha() or ch.isspace()])
            action_descrip = re.sub(r'\b\w+\s+(True|False)\b', '', action_descrip)
            
            try:
                hist_str = chat_processor.process_chat(result[annotation_id][action_id]["axt"],json_dict["conversations"][-1]["value"], action_descrip, result[annotation_id][action_id]["axt_nodeid"])
                hist_str = hist_str.replace('The action is','')
                action_hist.append(hist_str)
            # print(result[annotation_id][action_id]["axt"])
            # print(json_dict["conversations"][-1]["value"])
            # print(hist_str)
            except:           
                action_hist.append(
                    json_dict["conversations"][-1]["value"]
                    + " where "
                    + str(result[annotation_id][action_id]["axt_nodeid"])
                    + " is "
                    + action_descrip
                )
            
    return json_list


def process_data(file_path):
    err_cnt = {'exceed_4000':0}

    tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
    with open(file_path, "r") as file:
        data = json.load(file)

    result, purge_set = parse_op(data)

    for annotation_id, inner_dict in result.items():
        sorted_inner_dict = dict(sorted(inner_dict.items()))
        result[annotation_id] = sorted_inner_dict

    print(len(purge_set))
    # purge_set.add("c9f65ae8-bd67-4a83-82d7-d42587e34e98")
    # purge_set.add("7cd5a347-0e44-4ea2-8fcf-45fec1844279")
    # purge_set.add("1ec300ff-d45f-495f-8fbe-36802fdd8c57")
    print(purge_set)
    
    for annotation_id, inner_dict in result.items():
        for action_id in inner_dict:
            tree = result[annotation_id][action_id]["axt"]
            ax_node_id = result[annotation_id][action_id]["axt_nodeid"]
            context_len = len(tokenizer.encode(tree))
            if context_len > 3870:
                mean, std = 125, 50
                while True:
                    gaussian_number = np.random.normal(mean, std)
                    if 50 <= gaussian_number <= 150:
                        break
                random_number = np.random.randint(1, int(gaussian_number) + 1)

                
                try:
                    result[annotation_id][action_id]["axt"] = extract_lines(
                        tree, str(ax_node_id), random_number, gaussian_number-random_number, 3770, tokenizer
                    )
                    print(len(tokenizer.encode(result[annotation_id][action_id]["axt"])))
                    if len(tokenizer.encode(result[annotation_id][action_id]["axt"])) > 4000:
                        err_cnt['exceed_4000'] += 1
                        # assert False
                except:
                    print("---------")
    print(err_cnt)
    json_list = convert_for_training(result, purge_set)
    

    return json_list


def main():
    parser = argparse.ArgumentParser(description="JSON input")
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        required=True,
        help="Path to the input JSON file.",
    )

    args = parser.parse_args()

    output_data = process_data(args.input_file)

    output_filename = args.input_file.replace(".json", "_llama2.json")
    with open(output_filename, "w") as json_file:
        json.dump(output_data, json_file, indent=4)


if __name__ == "__main__":
    main()
