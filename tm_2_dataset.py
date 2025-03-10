"""
TMX to CSV Converter
Este script convierte archivos TMX (Translation Memory eXchange) a formato CSV,
extrayendo pares de traducción junto con información de contexto.
"""

import re
import os
import sys
from xml.sax.saxutils import unescape

def read_file(file_path):
    """
    Lee un archivo con la codificación correcta (primero intenta UTF-16, luego UTF-8)
    """
    encodings = ['utf-16', 'utf-8', 'latin-1']
    content = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                if content:
                    print(f"Archivo leído correctamente con codificación {encoding}")
                    break
        except Exception as e:
            print(f"Error al leer con codificación {encoding}: {e}")
    
    if not content:
        raise ValueError(f"No se pudo leer el archivo {file_path} con ninguna de las codificaciones intentadas")
    
    return content

def extract_translation_pairs(xml_content):
    """
    Extrae pares de traducción utilizando patrones regex
    """
    translations = []
    counter = 1
    
    # Extrae los idiomas origen y destino
    srclang_match = re.search(r'srclang=["\']([^"\']+)["\']', xml_content)
    targetlang_match = re.search(r'type=["\']targetlang["\']>([^<]+)<', xml_content)
    
    source_language = srclang_match.group(1) if srclang_match else "en"
    target_language = targetlang_match.group(1) if targetlang_match else "es"
    
    print(f"Idioma origen: {source_language}, Idioma destino: {target_language}")
    
    # Encuentra todas las unidades de traducción
    tu_pattern = r'<tu[^>]*>.*?</tu>'
    tus = re.findall(tu_pattern, xml_content, re.DOTALL)
    
    print(f"Se encontraron {len(tus)} unidades de traducción")
    
    for tu in tus:
        # Extrae el segmento origen
        source_pattern = r'<tuv\s+xml:lang=["\']' + source_language + r'["\'][^>]*>.*?<seg>(.*?)</seg>'
        source_match = re.search(source_pattern, tu, re.DOTALL)
        
        # Extrae el segmento destino
        target_pattern = r'<tuv\s+xml:lang=["\']' + target_language + r'["\'][^>]*>.*?<seg>(.*?)</seg>'
        target_match = re.search(target_pattern, tu, re.DOTALL)
        
        # Extrae información de contexto
        context_pre = ""
        context_post = ""
        
        # Busca contexto en el tuv origen
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
            
            # Decodifica entidades XML
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
    Guarda las traducciones en un archivo CSV sin comas adicionales para campos vacíos
    """
    if not translations:
        print("¡No se encontraron traducciones!")
        return False
    
    # Define campos requeridos y opcionales
    required_fields = ['ID', 'source', 'source_language', 'target', 'target_language']
    optional_fields = ['x-context-pre', 'x-context-post']
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            # Primero, escribe el encabezado - incluimos todos los campos
            header = ','.join(required_fields + optional_fields)
            csvfile.write(header + '\n')
            
            # Procesa cada fila de traducción manualmente
            for translation in translations:
                row_values = []
                
                # Añade los campos requeridos con comillas si es necesario
                for field in required_fields:
                    value = translation.get(field, "")
                    # Añade comillas si el valor contiene comas, comillas o saltos de línea
                    if isinstance(value, str) and (',' in value or '"' in value or '\n' in value):
                        # Reemplaza comillas dobles con comillas dobles escapadas
                        value = '"' + str(value).replace('"', '""') + '"'
                    row_values.append(str(value))
                
                # Verifica si los campos opcionales tienen contenido
                optional_content = False
                for field in optional_fields:
                    if field in translation and translation[field] and translation[field].strip():
                        optional_content = True
                        break
                
                # Solo añade los campos opcionales si al menos uno tiene contenido
                if optional_content:
                    for field in optional_fields:
                        value = translation.get(field, "")
                        if isinstance(value, str) and (',' in value or '"' in value or '\n' in value):
                            value = '"' + str(value).replace('"', '""') + '"'
                        row_values.append(str(value))
                
                # Escribe la fila
                csvfile.write(','.join(row_values) + '\n')
        
        print(f"Se guardaron correctamente {len(translations)} traducciones en {output_file}")
        return True
    except Exception as e:
        print(f"Error al guardar en CSV: {e}")
        return False


input_file = r"C:\Users\Miguel\OneDrive\Escritorio\4t curs\second_semester\synthetsis_project_II\Files\TM\MM-Mitsubishi MUT EN-ES.tmx"
output_file = "translations.csv"

if len(sys.argv) > 1:
    input_file = sys.argv[1]

if len(sys.argv) > 2:
    output_file = sys.argv[2]

try:
    # Lee el archivo TMX
    xml_content = read_file(input_file)
    
    # Extrae los pares de traducción
    translations = extract_translation_pairs(xml_content)
    
    # Guarda en CSV
    success = save_to_csv(translations, output_file)
    
    if success:
        print(f"Conversión completada exitosamente. Salida guardada en {output_file}")
        # Muestra las primeras traducciones como vista previa
        for i, trans in enumerate(translations[:3]):
            print(f"\nEjemplo {i+1}:")
            for key, value in trans.items():
                if value:  # Solo muestra campos con contenido
                    print(f"{key}: {value}")
    else:
        print("La conversión falló.")

except Exception as e:
    print(f"Error al procesar el archivo TMX: {e}")
    import traceback
    traceback.print_exc()

