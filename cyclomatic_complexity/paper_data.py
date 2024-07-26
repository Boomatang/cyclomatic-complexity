from pathlib import Path

from radon.complexity import cc_rank
import matplotlib.pyplot as plt
from tabulate import tabulate
from datetime import datetime
import json

timestamp_format = "%a %b %d %H:%M:%S %Y %z"


def load_data(path, name):
    data_file_name = name
    location = Path(path)

    if not location.is_dir():
        print("Directory does not exist, do not continue")
        exit(1)

    data_file = Path(location, data_file_name)
    if not data_file.is_file():
        print("File does not exist, do not continue")
        exit(1)

    with open(data_file) as df:
        data = json.load(df)
    return data


def project_table(data, save_file):

    history_dates = []
    for project in data:
        if not len(data[project]):
            continue
        project_history = {"project": project, "first_merge": datetime.now().date(), "last_merge": datetime(2000, 1, 1).date(), "last_commit": ""}
        for merge in data[project]:
            timestamp = datetime.strptime(merge['timestamp'], timestamp_format).date()

            if timestamp < project_history['first_merge']:
                project_history['first_merge'] = timestamp

            if timestamp > project_history['last_merge']:
                project_history['last_merge'] = timestamp
                project_history['last_commit'] = merge.get('commit', None)

        history_dates.append(project_history)

    history_dates = sorted(history_dates, key=lambda x: x['first_merge'])
    headers = {"project": 'Project', "first_merge": 'First Merge', "last_merge": 'Last Merge', "last_commit": 'Last Commit'}
    table = tabulate(history_dates, headers, tablefmt='latex_raw')
    with open(save_file, "w") as f:
        f.write(table)


def language_list(data, save_file):
    lang = set()
    for project in data:
        for entry in data[project]:
            for scc in entry['scc']:
                lang.add(scc['Name'])

    lang = sorted(lang)
    lang_list = [f"\t\\item {a}\n" for a in lang]
    formated_lang_list = "\\begin{itemize}\n" + "".join(lang_list) + "\\end{itemize}\n"
    with open(save_file, "w") as f:
        f.write(formated_lang_list)


def xx(dataset):
    youngest = {'project': None, 'timestamp': datetime(2000, 1, 1)}
    project_youngest = {}
    for project in dataset:
        if project not in project_youngest:
            project_youngest.setdefault(project, {'commit': None, 'timestamp': datetime(2000, 1, 1)})
        for entry in dataset[project]:
            if entry['timestamp'].timestamp() > youngest['timestamp'].timestamp():
                youngest['project'] = project
                youngest['timestamp'] = entry['timestamp']
            if entry['timestamp'].timestamp() > project_youngest[project]['timestamp'].timestamp():
                project_youngest[project] = entry
    if youngest['project'] is not None:
        dataset[youngest['project']].remove(project_youngest[youngest['project']])
    return youngest, project_youngest, dataset


def get_row_set(data):
    sample = {}
    for project in data:
        if not len(data[project]):
            continue
        if project not in sample:
            sample.setdefault(project, [])
        for entry in data[project]:
            sample[project].append({'commit': entry['commit'], 'timestamp': datetime.strptime(entry['timestamp'], timestamp_format)})

    for project in sample:
        sample[project] = sorted(sample[project], key=lambda x: x['timestamp'])

    counter = 1
    rows = []

    while counter:
        counter = 0
        for project in sample:
            if len(sample[project]) > counter:
                counter = len(sample[project])

        youngest, project_youngest, sample = xx(sample)

        if youngest['project'] is None:
            continue

        row = {'timestamp': youngest['timestamp']}
        for project in project_youngest:
            row[project] = project_youngest[project]['commit']
        rows.append(row)
    return rows


def scc_total(scc):
    total = 0
    for entry in scc:
        total += entry['Lines']
    return total


def project_loc(data, save_file):
    loc_rows = get_row_set(data)

    for row in loc_rows:
        for key in row:
            if key == 'timestamp' or row[key] is None:
                continue

            commit = filter(lambda x: x['commit'] == row[key], data[key])
            commit = next(commit, None)
            if commit:
                row[key] = scc_total(commit['scc'])

    timestamps = []
    projects = {}
    for row in loc_rows:
        for key in row:
            if key == 'timestamp':
                timestamps.append(row[key])
                continue
            if key not in projects:
                projects.setdefault(key, [])
            value = row[key]
            if value is None:
                value = 0
            projects[key].append(value)

    fig, ax = plt.subplots(figsize=(40, 15))

    ax.stackplot(timestamps, projects.values(),
                 labels=projects.keys(), alpha=0.8)
    ax.legend(loc='upper left', reverse=True)
    ax.set_title('LoC for Org')
    ax.set_xlabel('Year')
    ax.set_ylabel('Lines Of Code')

    plt.tight_layout()
    plt.savefig(save_file)


def scc_was_scanned(source, filters):
    not_scanned = 0
    scanned = 0

    for s in source:
        if s['Name'].lower() in filters:
            scanned += s['Lines']
        else:
            not_scanned += s['Lines']

    return scanned, not_scanned


def project_scanned_loc(data, save_file):
    cc_lang = ['rust', 'go', 'python', 'javascript', 'ruby']
    loc_cc = get_row_set(data)

    results = []
    for row in loc_cc:
        result_row = {'timestamp': '', 'scanned': 0, 'not_scanned': 0}
        for key in row:
            if key == 'timestamp':
                result_row['timestamp'] = row[key]
                continue
            if row[key] is None:
                continue

            commit = filter(lambda x: x['commit'] == row[key], data[key])
            commit = next(commit, None)
            if commit:
                scanned, not_scanned = scc_was_scanned(commit['scc'], cc_lang)
                result_row['scanned'] += scanned
                result_row['not_scanned'] += not_scanned
        results.append(result_row)

    timestamps = []
    status = {}
    for row in results:
        for key in row:
            if key == 'timestamp':
                timestamps.append(row[key])
                continue
            if key not in status:
                status.setdefault(key, [])
            value = row[key]
            if value is None:
                value = 0
            status[key].append(value)

    fig, ax = plt.subplots(figsize=(40, 15))

    ax.stackplot(timestamps, status.values(),
                 labels=status.keys(), alpha=0.8)
    ax.legend(loc='upper left', reverse=True)
    ax.set_title('Scanned lines of code Vs Non Scanned for Cyclomatic Complexity')
    ax.set_xlabel('Year')
    ax.set_ylabel('Lines Of Code')

    plt.tight_layout()
    plt.savefig(save_file)


def total_CC(dataset):
    total = 0
    for entry in dataset:
        total += entry['score']

    return total


def project_cc(data, save_file):
    total_cc_rows = get_row_set(data)
    total_cc_results = []
    for row in total_cc_rows:
        counter = 0
        for key in row:
            if key == 'timestamp' or row[key] is None:
                continue

            commit = filter(lambda x: x['commit'] == row[key], data[key])
            commit = next(commit, None)
            if commit:
                counter += total_CC(commit['cc'])

        total_cc_results.append({row['timestamp']: counter})

    timestamp = []
    values = []
    for entry in total_cc_results:
        for key in entry:
            timestamp.append(key)
            values.append(entry[key])

    fig, ax = plt.subplots(figsize=(40, 12))

    ax.stackplot(timestamp, values)
    ax.set_title('Total Cyclomatic Complexity For Org')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Cyclomatic Complexity Score')

    plt.tight_layout()
    plt.savefig(save_file)


def total_cc_grouped(data, save_file):
    categorised_cc_rows = get_row_set(data)
    categorised_cc_results = []
    for row in categorised_cc_rows:
        tmp = {row['timestamp']: 0}
        counter = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
        for key in row:
            if key == 'timestamp' or row[key] is None:
                continue

            commit = filter(lambda x: x['commit'] == row[key], data[key])
            commit = next(commit, None)
            if commit:
                result = categorised_cc(commit['cc'])
                for r in result:
                    counter[r] += result[r]

        categorised_cc_results.append({row['timestamp']: counter})

    timestamp = []
    values = {}
    for entry in categorised_cc_results:
        for key in entry:
            timestamp.append(key)
            for k in entry[key]:
                if k not in values:
                    values.setdefault(k, [])
                values[k].append(entry[key][k])

    keys = ["A 1-5", "B 6-10", "C 11-20", "D 21-30", "E 31-40", "F 41+"]

    fig, ax = plt.subplots(figsize=(40, 12))

    ax.stackplot(timestamp, values.values(),
                 labels=keys, alpha=0.8)
    ax.legend(loc='upper left', reverse=True)
    ax.set_title('Categorised Cyclomatic Complexity For Org')
    ax.set_xlabel('Year')
    ax.set_ylabel('Cyclomatic Complexity Score')

    plt.tight_layout()
    plt.savefig(save_file)


def categorised_cc_func_count(dataset):
    total = {}
    for entry in dataset:
        rank = cc_rank(entry['score'])
        if rank not in total:
            total[rank] = 0
        total[rank] += 1

    return total


def project_function_count_chart(data, save_file):
    categorised_cc_rows = get_row_set(data)
    categorised_cc_results = []
    for row in categorised_cc_rows:
        counter = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
        for key in row:
            if key == 'timestamp' or row[key] is None:
                continue

            commit = filter(lambda x: x['commit'] == row[key], data[key])
            commit = next(commit, None)
            if commit:
                result = categorised_cc_func_count(commit['cc'])
                for r in result:
                    counter[r] += result[r]

        categorised_cc_results.append({row['timestamp']: counter})


    timestamp = []
    values = {}
    for entry in categorised_cc_results:
        for key in entry:
            timestamp.append(key)
            for k in entry[key]:
                if k not in values:
                    values.setdefault(k, [])
                values[k].append(entry[key][k])

    fig, ax = plt.subplots(figsize=(40, 12))

    keys_value = {"A 1-5": "A", "B 6-10": "B", "C 11-20": "C", "D 21-30": "D", "E 31-40": "E", "F 41+": "F"}
    for key in keys_value:
        ax.plot(timestamp, values[keys_value[key]], label=key)
    ax.legend(loc='upper left', reverse=True)
    ax.set_title('Function Cyclomatic Complexity Rank Count For Org')
    ax.set_xlabel('Year')
    ax.set_ylabel('Cyclomatic Complexity Score As Percent')


    plt.tight_layout()
    plt.savefig(save_file)


def categorised_cc(dataset):
    total = {}
    for entry in dataset:
        rank = cc_rank(entry['score'])
        if rank not in total:
            total[rank] = 0
        total[rank] += entry['score']

    return total


def total_project_cc(project, data, save_file):
    project_data = {}
    project_data[project] = {}
    for row in data[project]:
        if not bool(row['cc']):
            continue
        counter = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
        result = categorised_cc(row['cc'])
        for r in result:
            counter[r] += result[r]
        project_data[project][row['timestamp']] = counter

    if not bool(project_data[project]):
        project_data.pop(project)

    keys = ["A 1-5", "B 6-10", "C 11-20", "D 21-30", "E 31-40", "F 41+"]

    fig, ax = plt.subplots(figsize=(40, 12))
    timestamp = []
    values = {}
    for entry in project_data[project]:
        timestamp.append(datetime.strptime(entry, timestamp_format))
        for k in project_data[project][entry]:
            if k not in values:
                values.setdefault(k, [])
            values[k].append(project_data[project][entry][k])

    ax.stackplot(timestamp, values.values(), labels=keys, alpha=0.8)
    ax.set_title(f'Total Cyclomatic Complexity For {project}')
    ax.legend(loc='upper left', reverse=True)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Cyclomatic Complexity Score')

    plt.tight_layout()
    plt.savefig(save_file)


def current_data(pd) -> dict:
    sorted_pd = sorted(pd, key=lambda x: datetime.strptime(x['timestamp'], timestamp_format))
    if not sorted_pd:
        return {}
    return sorted_pd[-1]


def project_pie_chart(project, data, save_file):
    project_data = {}
    class_a_data = {}

    c_data = current_data(data[project])
    project_data[project] = {}
    counter = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
    result = categorised_cc(c_data['cc'])
    for r in result:
        counter[r] += result[r]
    project_data[project] = counter

    a_counter = {}
    for row in c_data['cc']:
        if cc_rank(row['score']) == "A":
            if row['score'] in a_counter.keys():
                a_counter[row['score']] += 1
            else:
                a_counter[row['score']] = 1

    class_a_data[project] = a_counter

    has_data = False
    drop = []
    for k in project_data[project]:
        if project_data[project][k] > 0:
            has_data = True
        else:
            drop.append(k)
    for d in drop:
        project_data[project].pop(d)
    if not has_data:
        project_data.pop(project)

    keys = {"A": "A 1-5", "B": "B 6-10", "C": "C 11-20", "D": "D 21-30", "E": "E 31-40", "F": "F 41+"}
    fig, ax = plt.subplots(1, 2, figsize=(25, 12))
    fig.suptitle(f"Brake down of ranks for {project}")
    ax[0].pie(project_data[project].values(), labels=[keys[i] for i in project_data[project].keys()], autopct="%1.1f%%", startangle=90)
    ax[0].set_title('Percentage of ranked functions')

    ax[1].pie(class_a_data[project].values(), labels=class_a_data[project].keys(), autopct="%1.1f%%", startangle=90)
    ax[1].set_title('Break down of A rank')

    plt.tight_layout()
    plt.savefig(save_file)


if __name__ == '__main__':
    file_name = "20240725-1459.json"
    file_location = "../data"
    dataset = load_data(file_location, file_name)

    project_table(dataset, "../paper/data/case_study/generated/project_table.tex")
    language_list(dataset, "../paper/data/case_study/generated/lang_list.tex")

    project_loc(dataset, "../paper/images/project_loc.png")
    project_scanned_loc(dataset, "../paper/images/project_scanned_loc.png")
    project_cc(dataset, "../paper/images/project_cc.png")
    project_function_count_chart(dataset, "../paper/images/project_function_count_chart.png")

    total_project_cc('wasm-shim', dataset, "../paper/images/total_cc_wasm_shim.png")
    total_project_cc('limitador', dataset, "../paper/images/total_cc_limitador.png")
    total_project_cc('testsuite', dataset, "../paper/images/total_cc_testsuite.png")
    total_project_cc('multicluster-gateway-controller', dataset, "../paper/images/total_cc_mgc.png")

    project_pie_chart('testsuite', dataset, '../paper/images/testsuite_pie_chart.png')

    total_cc_grouped(dataset, "../paper/images/project_group_cc.png")
