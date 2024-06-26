import os
import json
import pandas as pd


def collect_experiment_data(dump_location):
    # 初始化列表以存储实验数据
    experiments = []

    # 遍历保存的实验结果目录
    for exp_name in os.listdir(dump_location):
        exp_dir = os.path.join(dump_location, exp_name)
        if os.path.isdir(exp_dir):
            accuracy_path = os.path.join(exp_dir, 'accuracy.png')
            loss_path = os.path.join(exp_dir, 'loss.png')
            model_path = os.path.join(exp_dir, 'pytorch_model.bin')
            tokenizer_path = os.path.join(exp_dir, 'dual-bert-fake-news-tokenizer')

            experiments.append({
                'Experiment Name': exp_name,
                'Accuracy Plot': accuracy_path if os.path.exists(accuracy_path) else 'N/A',
                'Loss Plot': loss_path if os.path.exists(loss_path) else 'N/A',
                'Model Path': model_path if os.path.exists(model_path) else 'N/A',
                'Tokenizer Path': tokenizer_path if os.path.exists(tokenizer_path) else 'N/A'
            })

    return experiments


def save_experiment_data_to_csv(experiments, output_file):
    df = pd.DataFrame(experiments)
    df.to_csv(output_file, index=False)


def main():
    # 从配置文件中读取 dump_location
    config = json.load(open("config.json"))
    dump_location = config["dump_location"]

    # 收集实验数据
    experiments = collect_experiment_data(dump_location)

    # 保存数据到CSV文件
    output_file = os.path.join(dump_location, 'experiment_summary.csv')
    save_experiment_data_to_csv(experiments, output_file)
    print(f"Experiment data saved to {output_file}")


if __name__ == "__main__":
    main()
