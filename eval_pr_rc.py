import re

def parse_log_file(file_path):
    # Regular expressions to extract required information
    model_pattern = re.compile(r"Evaluating Model:\s+(\S+)")
    accuracy_pattern = re.compile(r"Test Accuracy:\s+([\d.]+)%")
    confusion_pattern = re.compile(
        r"Confusion Matrix - TN:\s+(\d+),\s+FP:\s+(\d+),\s+FN:\s+(\d+),\s+TP:\s+(\d+)"
    )

    models = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_model = {}
    for line in lines:
        line = line.strip()

        # Check for model evaluation start
        model_match = model_pattern.search(line)
        if model_match:
            if current_model:
                models.append(current_model)
                current_model = {}
            current_model['Model'] = model_match.group(1)
            continue

        # Check for Test Accuracy
        acc_match = accuracy_pattern.search(line)
        if acc_match and 'Test Accuracy' not in current_model:
            current_model['Test Accuracy'] = float(acc_match.group(1))
            continue

        # Check for Confusion Matrix
        conf_match = confusion_pattern.search(line)
        if conf_match:
            current_model['TN'] = int(conf_match.group(1))
            current_model['FP'] = int(conf_match.group(2))
            current_model['FN'] = int(conf_match.group(3))
            current_model['TP'] = int(conf_match.group(4))
            continue

    # Don't forget to add the last model
    if current_model:
        models.append(current_model)

    return models

def calculate_precision_recall(model):
    TN = model['TN']
    FP = model['FP']
    FN = model['FN']
    TP = model['TP']
    # As per user's definitions
    precision = TN / (TN + FP) if (TN + FP) > 0 else 0
    recall = TN / (TN + FN) if (TN + FN) > 0 else 0

    #precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    #recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return precision, recall

def main():
    # log_file_path = "/home/minseok/forensic/evaluation_logs/evaluation_log_20241104_200021.log"
    # log_file_path2 = "/home/minseok/forensic/evaluation_logs2/evaluation_log_20241104_210055.log"
    # log_file_path3 = "/home/minseok/forensic/evaluation_logs3/evaluation_log_20241104_210151.log"
    # log_file_path4 = "/home/minseok/forensic/evaluation_logs4/evaluation_log_20241104_210219.log"
    # log_file_path5 = "/home/minseok/forensic/evaluation_logs5/evaluation_log_20241104_210256.log"
    log_file_path = "/home/minseok/forensic/evaluation_logs_rb1/evaluation_log_20241105_110023.log"
    log_file_path2 = "/home/minseok/forensic/evaluation_logs_rb2/evaluation_log_20241105_110136.log"
    log_file_path3 = "/home/minseok/forensic/evaluation_logs_rb3/evaluation_log_20241105_110213.log"
    log_file_path4 = "/home/minseok/forensic/evaluation_logs_rb4/evaluation_log_20241105_110318.log"
    log_file_path5 = "/home/minseok/forensic/evaluation_logs_rb5/evaluation_log_20241105_110347.log"

    log_lst = [log_file_path, log_file_path2, log_file_path3, log_file_path4, log_file_path5]

    # # Parse the log file
    # models = parse_log_file(log_file_path)

    # if not models:
    #     print("No models found in the log file.")
    #     return

    # # Find the model with the highest Test Accuracy
    # best_model = max(models, key=lambda x: x.get('Test Accuracy', 0))

    # # Calculate Precision and Recall
    # precision, recall = calculate_precision_recall(best_model)

    # # Display the results
    # print(f"Best Model: {best_model['Model']}")
    # print(f"Test Accuracy: {best_model['Test Accuracy']}%")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    avg_precision = 0
    avg_recall = 0
    avg_accuracy = 0
    for log_file_path in log_lst:
        models = parse_log_file(log_file_path)

        if not models:
            print("No models found in the log file.")
            return

        # Find the model with the highest Test Accuracy
        best_model = max(models, key=lambda x: x.get('Test Accuracy', 0))

        # Calculate Precision and Recall
        precision, recall = calculate_precision_recall(best_model)

        # Display the results
        print(f"Best Model: {best_model['Model']}")
        print(f"Test Accuracy: {best_model['Test Accuracy']}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        avg_precision += precision
        avg_recall += recall
        avg_accuracy += best_model['Test Accuracy']
    avg_precision /= len(log_lst)
    avg_recall /= len(log_lst)
    avg_accuracy /= len(log_lst)
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")

if __name__ == "__main__":
    main()
