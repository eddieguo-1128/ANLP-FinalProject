import os
import subprocess
import argparse


def get_checkpoint_dirs(checkpoints_folder):
    print(checkpoints_folder)
    res = []
    for f in os.listdir(checkpoints_folder):
        if os.path.isdir(os.path.join(checkpoints_folder, f)):
            res.append(f)
    return res

def read_record_file(record_file_path):
    recorded_checkpoints = set()
    if os.path.exists(record_file_path):
        with open(record_file_path, 'r') as file:
            for line in file:
                checkpoint = line.split(':')[0].strip()
                recorded_checkpoints.add(checkpoint)
    return recorded_checkpoints

def update_record_file(checkpoint, result ,record_file_path):
    with open(record_file_path, 'a') as file:
        file.write(f"{checkpoint} : {result}\n")

def run_evaluation(checkpoints_folder, checkpoint, eval_file):
    model_path = os.path.join(checkpoints_folder, checkpoint)
    eval_file = eval_file
    
    result = subprocess.run([
        "python", "/home/tianyueo/easyfastchat/fastchat/serve/inf.py",
        "--model-path", model_path,
        "--eval-file", eval_file,
        "--no-save-prediction", 
    ], capture_output=True, text=True)

    last_line = result.stdout.strip().split('\n')[-1]
    return last_line

def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--checkpoints-folder",
        "-c",
        type=str,
        required=True,
        help="Path to the checkpoint folder.",
    )
    parser.add_argument(
        "--record-file-path",
        "-r",
        type=str,
        required=True,
        help="Path to the record file.",
    )
    parser.add_argument(
        "--eval-file",
        "-e",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--prediction-file",
        "-p",
        type=str,
        required=False,
    )

    args = parser.parse_args()

    existing_checkpoints = set(read_record_file(args.record_file_path))
    all_checkpoints = set(get_checkpoint_dirs(args.checkpoints_folder))

    new_checkpoints = all_checkpoints - existing_checkpoints

    for checkpoint in new_checkpoints:
        result = run_evaluation(args.checkpoints_folder, checkpoint, args.eval_file)
        update_record_file(checkpoint, result, args.record_file_path)


if __name__ == "__main__":
    main()
