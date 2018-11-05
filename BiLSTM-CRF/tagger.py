import utils
from collections import defaultdict
import numpy as np, random, re, math, pickle

def tag_brand_name(pt_text, attr_vals):
    output = None
    
    if 'brand' in attr_vals:
        output = str(utils.lcs(pt_text, attr_vals['brand']))
    
    return output

def tag_screen_size(pt_text, attr_vals):
    output = None
    
    if 'screen_size' in attr_vals:
        screen = re.findall('\d+\.\d+|\d+', attr_vals['screen_size'], re.IGNORECASE)
        if len(screen) > 0:
            screen = str(screen[0])
            pattern = re.compile(screen+'\"|'+screen+'[\- ]inch|'+screen+'[\- ]in', re.IGNORECASE)

            for m in pattern.finditer(pt_text):
                output = pt_text[m.start():m.start()+len(m.group(0))]
                break
    
    return output

def tag_hdd_capacity(pt_text, attr_vals):
    output = None
    
    if 'hard_drive_capacity' in attr_vals:
        hdd = re.findall('\d+\.\d+|\d+', attr_vals['hard_drive_capacity'], re.IGNORECASE)
        if len(hdd) > 0:
            hdd[0] = hdd[0][0] if float(hdd[0]) >= 1000 else hdd[0]
            hdd = str(hdd[0])
            pattern = hdd+'\s*GB|'+hdd+'\s*gb|'+hdd+'\s*TB|'+hdd+'\s*tb|'+hdd+'\s*giga|'+hdd+'\s*tera'
            pattern = re.compile(pattern, re.IGNORECASE)

            for m in pattern.finditer(pt_text):
                output = pt_text[m.start():m.start()+len(m.group(0))]
                break
    
    return output

def tag_ram_size(pt_text, attr_vals):
    output = None
    
    if 'ram_memory' in attr_vals:
        ram = re.findall('\d+\.\d+|\d+', attr_vals['ram_memory'], re.IGNORECASE)
        if len(ram) > 0:
            ram = str(ram[0])
            pattern = ram+'\s*GB|'+ram+'\s*gb|'+ram+'\s*giga'
            pattern = re.compile(pattern, re.IGNORECASE)

            for m in pattern.finditer(pt_text):
                output = pt_text[m.start():m.start()+len(m.group(0))]
                break
    
    return output

def tag_screen_resolution(pt_text, attr_vals):
    output = None
    pattern = re.compile(r'\b[0-9]{3,}\s*x\s*[0-9]{3,}|[0-9]{3,}p\b', re.IGNORECASE)
    
    for m in pattern.finditer(pt_text):
        output = pt_text[m.start():m.start()+len(m.group(0))]
        break
    
    return output

def tag_processor_speed(pt_text, attr_vals):
    output = None
    pattern = re.compile(r'\b(\d+\.\d+|\d+)\s*GHz\b', re.IGNORECASE)
    
    for m in pattern.finditer(pt_text):
        output = pt_text[m.start():m.start()+len(m.group(0))]
        break
    
    return output

def tag_processor_type(pt_text, attr_vals):
    output = None
    
    if 'processor_type' in attr_vals:
        proc = attr_vals['processor_type']
        if len(proc) > 0:
            pattern = '('+utils.lcs(pt_text, proc)+')\s*(([A-Za-z][0-9]+[- ])?\s*[0-9]{4}([A-Za-z]+)?)?'
            pattern = re.compile(pattern, re.IGNORECASE)

            for m in pattern.finditer(pt_text):
                output = pt_text[m.start():m.start()+len(m.group(0))]
                break
    
    return output