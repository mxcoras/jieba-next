"""Jieba command line interface."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import jieba_next as jieba


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        usage="%(prog)s [options] filename",
        description="Jieba command line interface.",
        epilog="If no filename specified, use STDIN instead.",
    )

    parser.add_argument(
        "-d",
        "--delimiter",
        metavar="DELIM",
        default=" / ",
        nargs="?",
        const=" ",
        help=(
            "use DELIM instead of ' / ' for word delimiter; "
            "or a space if it is used without DELIM"
        ),
    )
    parser.add_argument(
        "-p",
        "--pos",
        metavar="DELIM",
        nargs="?",
        const="_",
        help=(
            "enable POS tagging; if DELIM is specified, use DELIM instead of '_' "
            "for POS delimiter"
        ),
    )
    parser.add_argument("-D", "--dict", help="use DICT as dictionary")
    parser.add_argument(
        "-u",
        "--user-dict",
        help=(
            "use USER_DICT together with the default dictionary or DICT (if specified)"
        ),
    )
    parser.add_argument(
        "-a",
        "--cut-all",
        action="store_true",
        dest="cutall",
        default=False,
        help="full pattern cutting (ignored with POS tagging)",
    )
    parser.add_argument(
        "-n",
        "--no-hmm",
        dest="hmm",
        action="store_false",
        default=True,
        help="don't use the Hidden Markov Model",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="don't print loading messages to stderr",
    )
    parser.add_argument(
        "-V", "--version", action="version", version="Jieba " + jieba.__version__
    )
    parser.add_argument("filename", nargs="?", help="input file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.quiet:
        jieba.set_log_level(60)

    if args.dict:
        jieba.set_dictionary(args.dict)

    if args.user_dict:
        jieba.load_userdict(args.user_dict)

    if args.pos:
        import jieba_next.posseg

        pos_delim = args.pos

        def cut_generator(sentence: str):
            for w, f in jieba_next.posseg.cut(sentence, HMM=args.hmm):
                yield w + pos_delim + f

    else:

        def cut_generator(sentence: str):
            yield from jieba.cut(sentence, cut_all=args.cutall, HMM=args.hmm)

    try:
        input_file = (
            Path(args.filename).open(encoding="utf-8") if args.filename else sys.stdin
        )

        for line in input_file:
            line_content = line.removesuffix("\r\n")
            words = cut_generator(line_content)
            result = str(args.delimiter).join(words)
            print(result)
        return 0
    finally:
        if args.filename and "input_file" in locals():
            input_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
