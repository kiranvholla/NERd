import gzip
import itertools


def get_ner_reader(data):
    fin = gzip.open(data, 'rt',  encoding="utf8") if data.endswith('.gz') else open(data, 'rt',  encoding="utf8")
    for is_divider, lines in itertools.groupby(fin, _is_divider):
        if is_divider:
            continue
        lines = [line.strip().replace('\u200d', '').replace('\u200c', '').replace('\u200b', '') for line in lines]
        metadata = lines[0].strip() if lines[0].strip().startswith('# id') else None
        fields = [line.split() for line in lines if not line.startswith('# id')]
        fields = [list(field) for field in zip(*fields)]
        yield fields, metadata



def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    if line.split()[0] == "-DOCSTART-":
        return True
    return False
    
    

def _assign_ner_tags(ner_tag, rep_):
    ner_tags_rep = []
    sub_token_len = len(rep_)
    mask_ = [False] * sub_token_len
    if len(mask_):
        mask_[0] = True
    if ner_tag[0] == 'B':
        in_tag = 'I' + ner_tag[1:]
        ner_tags_rep.append(ner_tag)
        ner_tags_rep.extend([in_tag] * (sub_token_len - 1))
    else:
        ner_tags_rep.extend([ner_tag] * sub_token_len)
    return ner_tags_rep, mask_



def extract_spans(tags):
    cur_tag = None
    cur_start = None
    gold_spans = {}
    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        if _cur_start is None:
            return _gold_spans
        _gold_spans[(_cur_start, _cur_id - 1)] = _cur_tag
        return _gold_spans
    for _id, nt in enumerate(tags):
        indicator = nt[0]
        if indicator == 'B':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]
            pass
        elif indicator == 'I':
            pass
        elif indicator == 'O':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_tag = 'O'
            cur_start = _id
            pass
    _save_span(cur_tag, cur_start, _id + 1, gold_spans)
    return gold_spans