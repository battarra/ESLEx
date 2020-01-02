from typing import Dict


def plt_style(*style_words: str) -> Dict[str, object]:
    style = {}

    for word in style_words:
        _parse_plt_style_word(word, style)

    return style


def _parse_plt_style_word(style_word: str, style_dic: Dict[str, object]) -> None:
    colors_dic = {'color1': 'tab:blue', 'color2': 'tab:red'}
    if style_word in colors_dic:
        style_dic['color'] = colors_dic[style_word]
        return

    if style_word == 'dashed':
        style_dic['linestyle'] = '--'
        return

    thickn_dic = {'thin': 1, 'thick': 1.5}
    if style_word in thickn_dic:
        style_dic['linewidth'] = thickn_dic[style_word]
        return

    marks_dic = {'mark_small': 1, 'mark_large': 2}
    if style_word in marks_dic:
        style_dic['marker'] = 'o'
        style_dic['markersize'] = marks_dic[style_word]
        return

    raise Exception("Unknown style word {word}".format(word=style_word))