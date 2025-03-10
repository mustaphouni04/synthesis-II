from lxml import etree
from lxml.etree import tostring
import pandas as pd 
import os
import sys

input_file = os.path.basename(sys.argv[1:][0])

df = pd.DataFrame(columns=["Source", "Target", "Context", "Limit", "Status"])
xml_tree = etree.parse(input_file)

namespace = {"ns": "urn:oasis:names:tc:xliff:document:1.2", "mq": "MQXliff"}
trans_units = xml_tree.findall(".//ns:trans-unit", namespaces=namespace)

def stringify_children(node):
    if node == None:
        return ""
    s = node.text
    if s is None:
        s = ''
    for child in node:
        s += etree.tostring(child, encoding='unicode')
    return s 

sources = []
targets = []
contexts = []
statuses = []
limits = []

# Extract source, target, context, status, and limit information from the trans-unit element
for trans_unit in trans_units:
    # Extract the source text
    source_element = trans_unit.find(".//ns:source", namespaces=namespace)
    source = stringify_children(source_element) if source_element is not None else ""
    sources.append(source)

    # Extract the target text
    target_element = trans_unit.find(".//ns:target", namespaces=namespace)
    target = stringify_children(target_element) if target_element is not None else ""
    targets.append(target)

    # Extract the context information
    context_element = trans_unit.find(".//ns:context", namespaces=namespace)
    context = context_element.text if context_element is not None else ""
    contexts.append(context)

    # Extract the status information
    status = trans_unit.get("{MQXliff}status") or ""
    statuses.append(status)

    # Extract the limit information
    limit = trans_unit.get("{MQXliff}maxlengthchars") or ""
    limits.append(limit)



df['Source'] = sources
df['Target'] = targets
df['Context'] = contexts
df['Status'] = statuses 
df['Limit'] = limits 

df.to_excel("EXPORT_" + os.path.splitext(input_file)[0] + ".xlsx",index=False)

