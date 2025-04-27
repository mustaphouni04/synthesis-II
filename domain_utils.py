import datasets
from datasets import load_dataset
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import lxml.etree as etree
import PythonTmx as tmx
import pandas as pd
from tqdm import tqdm
import os

class DomainProcessing:
    def __init__(self, paths: Union[str,Path, List[Union[str, Path]]]):
        self.paths = [Path(p) for p in (paths if isinstance(paths,list) else [paths])]
        self.domain = None
        self.sentences = []

        if str(self.paths[0]).endswith(".csv") and isinstance(str(self.paths[0]), str):
            self.domain = pd.read_csv(self.paths[0])
        else:
            for path in self.paths:
                try:
                    self.domain = load_dataset(str(path))
                    self.content = None
                except:
                    if type(path) == Path:
                        path = str(path)
                    else:
                        try:
                            # Load a TMX file
                            tmx_file: etree._ElementTree = etree.parse(
                                    str(path), etree.XMLParser(encoding="utf-16le")
                            )
                            tmx_root: etree._Element = tmx_file.getroot()
                            self.sentences.extend(self.extract_translations(tmx_root))
                        except etree.XMLSyntaxError as e:
                            raise ValueError(f"Failed to parse TMX file: {e}")
                        except Exception as e:
                            raise RuntimeError(f"Unexpected error while processing TMX file: {e}")

    def extract_translations(self, root: etree._Element) -> List[Dict[str,str]]:
        translations = []

        for tu in tqdm(root.findall(".//tu")):
            en_sentence = None
            es_sentence = None

            for tuv in tu.findall(".//tuv"):
                lang = tuv.get("lang")
                seg = tuv.find("seg")

                if seg is not None and lang is not None:
                    if lang.startswith("EN-GB"):
                        en_sentence = seg.text.strip() if seg.text else ""
                    elif lang.startswith("ES-ES"):
                        es_sentence = seg.text.strip() if seg.text else ""
            if en_sentence and es_sentence:
                translations.append({"en":en_sentence, "es":es_sentence})
        return translations

    @property
    def type(self):
        if self.domain is not None:
            return type(self.domain)
        else:
            return type(self.sentences)
    
    def process(self):
        if self.sentences:
            return pd.DataFrame(self.sentences)
        return self.domain





