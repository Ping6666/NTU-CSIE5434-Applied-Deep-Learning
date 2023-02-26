import os, sys
import json

# sys.argv[1]: preprocess type: mc, qa.
# sys.argv[2]: path to the context file.
# sys.argv[3]: path to the testing file.
# sys.argv[4]: path to the "adjustment" testing file.


def read_file(c_file, d_file):
    # read file
    with open(c_file, 'r') as f_in:
        context = json.load(f_in)
    with open(d_file, 'r') as f_in:
        data = json.load(f_in)
    return context, data


# preprocess
def mc_preprocess(context, data):
    _list = []
    for d in data:
        _d = {
            "id": d["id"],
            "sent1": d["question"],  # start of first sentence
            "sent2": "",  # start of second sentence
            "ending0": context[d["paragraphs"][0]],
            "ending1": context[d["paragraphs"][1]],
            "ending2": context[d["paragraphs"][2]],
            "ending3": context[d["paragraphs"][3]],
        }

        if "relevant" in d:
            _d["label"] = d["paragraphs"].index(d["relevant"])

        _list.append(_d)
    return _list


def qa_preprocess(context, data):
    _list = []
    for d in data:
        _d = {
            "id": d["id"],
            "question": d["question"],
            "context": context[d["relevant"]],
            "answers": {
                'answer_start': [d["answer"]["start"]],
                'text': [d["answer"]["text"]],
            },
        }
        _list.append(_d)
    return _list


def write_file(o_file, data):
    # make output dir
    dir_path = os.path.dirname(o_file)
    os.makedirs(dir_path, exist_ok=True)

    # dump file
    with open(o_file, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False)
    return


def main():
    context, data = read_file(sys.argv[2], sys.argv[3])

    if sys.argv[1] == 'mc':
        o_list = mc_preprocess(context, data)
    elif sys.argv[1] == 'qa':
        o_list = qa_preprocess(context, data)
    else:
        raise ValueError

    write_file(sys.argv[4], o_list)
    return


if __name__ == "__main__":
    main()
