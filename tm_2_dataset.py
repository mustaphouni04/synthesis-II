import csv
import re
import os
import sys
from xml.sax.saxutils import unescape

def read_file(file_path):
    """
    Read file with correct encoding (tries UTF-16 first, then falls back to UTF-8)
    """
    encodings = ['utf-16', 'utf-8', 'latin-1']
    content = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                if content:
                    print(f"Successfully read file using {encoding} encoding")
                    break
        except Exception as e:
            print(f"Failed to read with {encoding} encoding: {e}")
    
    if not content:
        raise ValueError(f"Could not read file {file_path} with any of the attempted encodings")
    
    return content

def extract_translation_pairs(xml_content):
    """
    Extract translation pairs using regex pattern matching
    """
    translations = []
    counter = 1
    
    # Extract source and target languages
    srclang_match = re.search(r'srclang=["\']([^"\']+)["\']', xml_content)
    targetlang_match = re.search(r'type=["\']targetlang["\']>([^<]+)<', xml_content)
    
    source_language = srclang_match.group(1) if srclang_match else "en"
    target_language = targetlang_match.group(1) if targetlang_match else "es"
    
    print(f"Source language: {source_language}, Target language: {target_language}")
    
    # Find all translation units
    tu_pattern = r'<tu[^>]*>.*?</tu>'
    tus = re.findall(tu_pattern, xml_content, re.DOTALL)
    
    print(f"Found {len(tus)} translation units")
    
    for tu in tus:
        # Extract the source segment
        source_pattern = r'<tuv\s+xml:lang=["\']' + source_language + r'["\'][^>]*>.*?<seg>(.*?)</seg>'
        source_match = re.search(source_pattern, tu, re.DOTALL)
        
        # Extract the target segment
        target_pattern = r'<tuv\s+xml:lang=["\']' + target_language + r'["\'][^>]*>.*?<seg>(.*?)</seg>'
        target_match = re.search(target_pattern, tu, re.DOTALL)
        
        # Extract context information
        context_pre = ""
        context_post = ""
        
        # Search for context in the source tuv
        source_tuv_pattern = r'<tuv\s+xml:lang=["\']' + source_language + r'["\'][^>]*>(.*?)</tuv>'
        source_tuv_match = re.search(source_tuv_pattern, tu, re.DOTALL)
        
        if source_tuv_match:
            source_tuv_content = source_tuv_match.group(1)
            
            pre_context_pattern = r'<prop\s+type=["\']x-context-pre["\']>(.*?)</prop>'
            post_context_pattern = r'<prop\s+type=["\']x-context-post["\']>(.*?)</prop>'
            
            pre_context_match = re.search(pre_context_pattern, source_tuv_content, re.DOTALL)
            post_context_match = re.search(post_context_pattern, source_tuv_content, re.DOTALL)
            
            if pre_context_match:
                context_pre = pre_context_match.group(1).strip()
            
            if post_context_match:
                context_post = post_context_match.group(1).strip()
        
        if source_match and target_match:
            source_text = source_match.group(1).strip()
            target_text = target_match.group(1).strip()
            
            # Unescape XML entities
            source_text = unescape(source_text)
            target_text = unescape(target_text)
            context_pre = unescape(context_pre)
            context_post = unescape(context_post)
            
            translations.append({
                'ID': counter,
                'source': source_text,
                'source_language': source_language,
                'target': target_text,
                'target_language': target_language,
                'x-context-pre': context_pre,
                'x-context-post': context_post
            })
            
            counter += 1
    
    return translations

def save_to_csv(translations, output_file):
    """
    Save translations to CSV file
    """
    if not translations:
        print("No translations found!")
        return False
    
    fieldnames = ['ID', 'source', 'source_language', 'target', 'target_language', 'x-context-pre', 'x-context-post']
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(translations)
        
        print(f"Successfully saved {len(translations)} translations to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False


input_file = r"C:\Users\Miguel\OneDrive\Escritorio\4t curs\second_semester\synthetsis_project_II\AI_Translation_Engines\sample data\TM\[EN-ES] CF - iDISC.tmx"
output_file = "translations.csv"

if len(sys.argv) > 1:
    input_file = sys.argv[1]

if len(sys.argv) > 2:
    output_file = sys.argv[2]

try:
    # Read the TMX file
    xml_content = read_file(input_file)
    
    # Extract translation pairs
    translations = extract_translation_pairs(xml_content)
    
    # Save to CSV
    success = save_to_csv(translations, output_file)
    
    if success:
        print(f"Conversion completed successfully. Output saved to {output_file}")
        # Print the first few translations as a preview
        for i, trans in enumerate(translations[:3]):
            print(f"\nSample {i+1}:")
            for key, value in trans.items():
                print(f"{key}: {value}")
    else:
        print("Conversion failed.")

except Exception as e:
    print(f"Error processing TMX file: {e}")
    import traceback
    traceback.print_exc()

