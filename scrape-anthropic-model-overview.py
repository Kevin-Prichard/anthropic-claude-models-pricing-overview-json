#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict as dd
import dateparser
from datetime import datetime
from enum import Enum
import json
import requests

import lxml.html
import regex


MODELS_OVERVIEW_URL = "https://docs.anthropic.com/en/docs/about-claude/models/overview"
TABLE_SEL = ("//{title_tag}[child::span[normalize-space()='{model_name}']]"
             "/following::table")
NUMBER_RX = regex.compile(r"\D*(?<number>[0-9]*\.?[0-9]+)?\D*")
ANTH_DATE_RX = regex.compile(r"^.*?(?P<date>\w+\s\d{4}).*?$")
MODEL_ID_RX = regex.compile(r"^(?P<model_id>\S+).*")
MODEL_TAG_PUBLISH_DATE_RX = regex.compile(
    r".*?\D(:?(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2}))$")
MODEL_TAG_PARTS_RX = regex.compile(
    r"^(?P<family>\w+)\s+"
    r"(?P<branch>\w+)\s+"
    r"(?P<version>[0-9.]+)"
)


COMP_CONVERSIONS = {
    "Multilingual": lambda x: x.lower() == "yes",
    "Vision": lambda x: x.lower() == "yes",
    "Extended thinking": lambda x: x.lower() == "yes",
    "Priority Tier": lambda x: x.lower() == "yes",
    "Context window": lambda x: int(NUMBER_RX.match(x).group("number")) * 1000,
    "Max output": lambda x: int(NUMBER_RX.match(x).group("number")),
    "Training data cut-off": lambda x: dateparser.parse(
        ANTH_DATE_RX.match(x).group("date")).isoformat(),
}

TABLE_IDENTIFIERS = {
    "names": {
        "JSON": "model_names",
        "PYTHON": "model_names",
    },
    "aliases": {
        "JSON": "model_aliases",
        "PYTHON": "model_aliases",
    },
    "comparison": {
        "JSON": "model_facts",
        "PYTHON": "model_facts",
    },
    "pricing": {
        "JSON": "model_pricing",
        "PYTHON": "model_pricing",
    },
}


class OutputFormat(Enum):
    JSON = "json"
    CSV = "csv"
    JAVASCRIPT = "javascript"
    PYTHON = "python"
    YAML = "yaml"

    def __str__(self):
        return self.value


class KeyCase(Enum):
    ASIS = "asis"
    SNAKE = "snake"
    LOWER = "lower"
    UPPER = "upper"

    def __str__(self):
        return self.value


class TrainingOrPublishDate(Enum):
    TRAINING = "training"
    PUBLISHED = "published"

    def __str__(self):
        return self.value


def key_case_conv(key, key_case: KeyCase):
    key = key.replace("&", "and")
    if key_case == KeyCase.SNAKE:
        return key.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    elif key_case == KeyCase.LOWER:
        return key.lower()
    elif key_case == KeyCase.UPPER:
        return key.upper()
    else:
        return key  # asis, no conversion


def get_args():
    case_serial = ", ".join([case.value for case in KeyCase])
    parser = argparse.ArgumentParser(
        prog='scrape_anthropic_model_overview.py',
        description='Scrape Anthropic\'s overview page of model metadata')
    parser.add_argument(
        '-u', '--url', action='store', default=MODELS_OVERVIEW_URL,
        dest='url', help='URL of Anthropic models overview page')
    parser.add_argument(
        '-c', '--case', action='store', default=KeyCase.SNAKE, dest='case',
        type=KeyCase, choices=list(KeyCase),
        help='Case of dict keys in output JSON: %s' % case_serial)
    parser.add_argument(
        '-o', '--output', action='store', required=False,
        dest='output_file', help='File containing sample data')
    parser.add_argument(
        '-f', '--format', action='store', default=OutputFormat.PYTHON,
        dest='output_format', type=OutputFormat, choices=list(OutputFormat),
        help='Output format of the data: %s' % ", ".join(
            [fmt.value for fmt in OutputFormat]))
    parser.add_argument(
        '-m', '--subst-id', action='store_true', default=False,
        dest='subst_id',
        help='Substitute Model IDs in place of Model Names for dict keys')
    parser.add_argument(
        '-a', '--model-aliases', action='store_true', default=False,
        dest='model_aliases',
        help='Include model aliases in output JSON; a redundancy but useful '
             'for the appearance of completeness without performing two accesses'
    )
    args = parser.parse_args()
    return args


def extract_model_names_table(html_tree):
    result = {}

    for table in html_tree.xpath(TABLE_SEL.format(
        title_tag="h2", model_name="Model names"))[:2]:
        rows = table.xpath(".//tr")
        for row in rows[1:]:
            cells = [td.text_content().strip() for td in row.xpath(".//td")]
            if not cells or len(cells) < 4:
                continue
            result[cells[0]] = {
                "anthropic": MODEL_ID_RX.match(cells[1]).group("model_id"),
                "aws_bedrock": MODEL_ID_RX.match(cells[2]).group("model_id"),
                "gcp_vertex_ai": MODEL_ID_RX.match(cells[3]).group("model_id"),
            }
    return result


def extract_model_aliases_table(html_tree):  # , model_renames=None):
    table = html_tree.xpath(TABLE_SEL.format(
        title_tag="h3", model_name="Model aliases"))[0]
    rows = table.xpath(".//tr")
    result = {}
    for row in rows[1:]:
        cells = [td.text_content().strip() for td in row.xpath(".//td")]
        if not cells or len(cells) < 3:
            continue
        # if (model_key := cells[0]) and model_renames:
        #     model_key = model_renames.get(model_key, {}).get("anthropic",
        #                                                      model_key)
        result[cells[0]] = {
            "alias": cells[1],
            "model_id": cells[2],
        }
    return result


def extract_model_comparison_table(html_tree, key_case: KeyCase, model_renames=None):
    # AKA Model Facts
    table = html_tree.xpath(TABLE_SEL.format(
        title_tag="h3", model_name="Model comparison table"))[0]
    rows = table.xpath(".//tr")
    result = dd(dict)
    models = [th.text_content().strip() for th in rows[0].xpath(".//th")[1:]]
    for row in rows[1:]:
        cells = [td.text_content().strip() for td in row.xpath(".//td")]
        info_kind = cells[0].strip()
        handler = COMP_CONVERSIONS.get(info_kind, lambda x: x)
        for model_nr, cell in enumerate(cells[1:]):
            if (model_key := models[model_nr]) and model_renames:
                model_key = model_renames.get(model_key,
                                              {}).get("anthropic", model_key)
            result[model_key][key_case_conv(info_kind, key_case)] = handler(cell.strip())

    return result


def extract_model_pricing_table(html_tree, key_case: KeyCase, model_renames=None):
    table = html_tree.xpath(TABLE_SEL.format(
        title_tag="h3", model_name="Model pricing"))[0]
    rows = table.xpath(".//tr")
    token_kinds = [th.text_content().strip() for th in rows[0].xpath(".//th")]
    result_reg = dd(dict)

    for row in rows[1:]:
        cells = [td.text_content().strip() for td in row.xpath(".//td")]
        for token_kind_nr, cell in enumerate(cells[1:]):
            value = f'{float(NUMBER_RX.match(cell).group("number"))}'
            if (model_key := cells[0]) and model_renames:
                model_key = model_renames.get(model_key, {}).get("anthropic", model_key)
            result_reg[model_key][key_case_conv(token_kinds[token_kind_nr + 1], key_case)] = value

    return result_reg


def model_tag_to_name(model_names, aliases):
    tag2name = dict()
    for model_name, model_info in model_names.items():
        for platform, model_id in model_info.items():
            tag2name[model_id] = model_name
    for alias, alias_info in aliases.items():
        tag2name[alias_info["alias"]] = tag2name[alias_info["model_id"]]
    return tag2name


def combine_limits_and_pricing(model_names, pricing, comparison):
    combined = {}
    for model_key, model_info in model_names.items():
        # Special case, where model_key is a tag with " v2" suffix,
        # but referenced in other places without such as pricing
        if model_key.endswith(" v2"):
            model_key = model_key[:-3]
        model_tag = model_info["anthropic"]
        combined[model_key] = {
            "pricing": pricing.get(model_tag),
            "limits": {
                "context_window": comparison[model_key]["context_window"],
                "max_output": comparison[model_key]["max_output"],
            },
        }
    return combined
def fetch_and_parse_model_metadata(url,
                                   output_file=None,
                                   key_case: KeyCase=KeyCase.SNAKE,
                                   subst_id=False,
                                   format: OutputFormat=OutputFormat.JSON):

    with requests.get(url) as response:
        if response.status_code != 200:
            raise Exception(f"Failed to fetch {url}: {response.status_code}")
        html_tree = lxml.html.fromstring(response.content)
        model_names = extract_model_names_table(html_tree)
        model_renames = model_names if subst_id else None
        aliases = extract_model_aliases_table(
            html_tree)  # , model_renames=model_names)
        tag2name = model_tag_to_name(model_names, aliases)
        comparison = extract_model_comparison_table(
                html_tree, key_case)  #, model_renames=model_names)
        # latest = platform_by_name_by_latest(model_names, comparison)
        pricing = extract_model_pricing_table(
            html_tree, key_case, model_renames=model_names)
        limits_and_pricing = combine_limits_and_pricing(model_names,
                                                        pricing,
                                                        comparison)
        latest = None
        result = {
            "names": model_names,
            "aliases": aliases,
            "tag_to_name": tag2name,
            "comparison": comparison,
            # "latest": latest,
            "pricing": pricing,
            "limits_and_pricing": limits_and_pricing,
        }
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(convert_output_format(result, format))
                # json.dump(result, f, indent=4, ensure_ascii=False)
        else:
            return result


def convert_output_format(data, output_format: OutputFormat):

    if output_format == OutputFormat.JSON:
        return json.dumps(data, indent=4, ensure_ascii=False)
    elif output_format == OutputFormat.CSV:
        # Implement CSV conversion logic here
        pass
    elif output_format == OutputFormat.JAVASCRIPT:
        # Implement JavaScript conversion logic here
        pass
    elif output_format == OutputFormat.PYTHON:
        # Convert prices from string to Decimal
        import black
        from decimal import Decimal
        for model_key, prices in data["pricing"].items():
            for price_key, price_value in prices.items():
                prices[price_key] = Decimal(price_value)

        for model_key, attribs in data["comparison"].items():
            attribs["training_data_cut_off"] = datetime.fromisoformat(
                attribs["training_data_cut_off"])

        buf = []
        # bust out the Decimal class and use ast.parse/unparse to convert the data
        for model_info_type, model_info in data.items():
            buf.append(
                black.format_str(
                    f"{TABLE_IDENTIFIERS[model_info_type]['PYTHON']} = " +
                    repr(dict(model_info)).replace("'", '"'),
                    mode=black.Mode(
                        line_length=80, string_normalization=True, is_pyi=False
                    )
                )
            )
        return "\n\n".join(buf)

    elif output_format == OutputFormat.YAML:
        # Implement YAML conversion logic here
        pass
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def main():
    args = get_args()
    result = fetch_and_parse_model_metadata(
        MODELS_OVERVIEW_URL,
        output_file=args.output_file,
        key_case=args.case,
        subst_id=args.subst_id,
        format=args.output_format
    )
    if result:
        print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
