import argparse
from collections import Counter
from utils import *

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args.model_name, args.model_size, args.model_path)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot" or args.method == "auto_cot" or args.method == "pattern_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        pass

    total = 0
    correct_total = 0
    with open(args.output_dir, "a") as wp:

        for i, data in enumerate(dataloader):
            if i < args.resume_id - 1:
                continue
            output_line = {}
            
            print('*************************')
            print("{}st data".format(i+1))
                    
            # Prepare question template ...
            x, y = data
            x = "Q: " + x[0] + "\n" + "A:"
            #"Initial problem: " + x[0] + "\n" 
            y = y[0].strip()
            
            # print(x, y)
            
            output_line["question"] = x
            output_line["gold_ans"] = y

            if args.method == "zero_shot":
                x = x + " " + args.direct_answer_trigger_for_zeroshot
            elif args.method == "zero_shot_cot":
                x = x + " " + args.cot_trigger
            elif args.method == "few_shot":
                x = demo + x
            elif args.method == "few_shot_cot":
                x = demo + x
            elif args.method == "auto_cot":
                x = demo + x + " " + args.cot_trigger
            elif args.method == "pattern_cot":
                x = demo + x + " " + args.cot_trigger
            else:
                raise ValueError("method is not properly defined ...")

            preds = []
            for _ in range(args.iterations):            
                # Answer experiment by generating text ...
                max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
                z = decoder.decode(x, args.model_name, max_length)

                output_line["rationale"] = z

                # Answer extraction for zero-shot-cot ...
                if args.method == "zero_shot_cot":
                    z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                    max_length = args.max_length_direct
                    pred = decoder.decode(z2, args.model_name, max_length)
                    print(z2 + pred)
                else:
                    pred = z
                    print(x + pred)

                # Clensing of predicted answer ...
                pred = answer_cleansing(args, pred)
                if pred:
                    preds.append(pred)
            
            if not preds:
                preds.append('')
            pred = Counter(preds).most_common(1)[0][0] 
            output_line["pred_ans"] = pred
            output_line["wrap_que"] = x

            output_json = json.dumps(output_line)
            wp.write(output_json + '\n')

            # Choose the most frequent answer from the list ...
            print("pred : {}".format(pred))
            print("GT : " + y)
            print('*************************')
            
            # Checking answer ...
            correct = (np.array([pred]) == np.array([y])).sum().item()
            correct_total += correct
            total += 1 #np.array([y]).size(0)
            accuracy = (correct_total * 1.0 / total) * 100
            print("accuracy : {}".format(accuracy))
            
            if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
                break
                #raise ValueError("Stop !!")

    # Calculate accuracy ...
    accuracy = (correct_total * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="addsub", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters", "object_tracking", "bigbench_date"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--demo_path", type=str, default="demos/addsub", help="pre-generated demos used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0, help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot", "pattern_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment/addsub", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=256, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature"
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="self consistency iterations"
    )
    parser.add_argument(
        "--model_name", type=str, default="qwen", help="model used for decoding."
    )
    parser.add_argument(
        "--model_size", type=str, default="7b", help="model size"
    )
    parser.add_argument(
        "--model_path", type=str, default="", help="path to the model"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is option "
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is "
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is "
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is option "
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is "
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the (Yes or No) answer is: "
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is "
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step. "
    
    return args

if __name__ == "__main__":
    main()
