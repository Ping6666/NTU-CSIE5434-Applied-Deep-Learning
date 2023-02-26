import os, sys
import jsonlines, json

# sys.argv[1] (.jsonl): path to the testing file.
# sys.argv[2] (.csv): path to the "adjustment" testing file.


def mk_dir(o_file):
    # make output dir
    dir_path = os.path.dirname(o_file)
    os.makedirs(dir_path, exist_ok=True)
    return


# preprocess
def preprocess(i_file, o_file):
    _list = []

    # read file
    with jsonlines.open(i_file) as f_in:
        for obj in f_in:
            _d = {
                "id": obj["id"],
                "text": "" if "maintext" not in obj else obj["maintext"],
                "summary": "" if "title" not in obj else obj["title"],
            }
            _list.append(_d)

    # dump file
    with open(o_file, 'w', encoding='utf-8') as f_out:
        json.dump(_list, f_out, ensure_ascii=False)
    return


def main():
    mk_dir(sys.argv[2])
    preprocess(sys.argv[1], sys.argv[2])
    return


if __name__ == "__main__":
    main()
