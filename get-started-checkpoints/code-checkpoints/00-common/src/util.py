from ruamel.yaml import YAML

def load_params():
    yaml = YAML(typ="safe")
    with open("params.yaml") as f:
        params = yaml.load(f)
    return params

def history_list_to_csv(history_list):
    "Converts a list of history dicts to a CSV string"
    keys = list(history_list[0].history.keys())
    csv_string = ", ".join(["epoch"] + keys) + "\n"
    list_len = len(history_list)
    for i in range(list_len):
        row = (str(i+1) + ", " + ", ".join([str(history_list[i].history[k][0]) for k in keys]) + "\n")
        csv_string += row
    return csv_string

def history_to_csv(history):
    keys = list(history.history.keys())
    csv_string = ", ".join(["epoch"] + keys) + "\n"
    list_len = len(history.history[keys[0]])
    for i in range(list_len):
        row = (
            str(i + 1)
            + ", "
            + ", ".join([str(history.history[k][i]) for k in keys])
            + "\n"
        )
        csv_string += row

    return csv_string

def logs_to_csv(logs):
    keys = list(logs.keys())
    csv_string = ", ".join(["epoch"] + keys) + "\n"
    list_len = len(logs[keys[0]])
    for i in range(list_len):
        row = (
            str(i + 1)
            + ", "
            + ", ".join([str(logs[k][i]) for k in keys])
            + "\n"
        )
        csv_string += row

    return csv_string
