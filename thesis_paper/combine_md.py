import re
import os

def process_includes(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    result = []
    for line in lines:
        match = re.match(r'@include\s+"(.+?)"', line.strip())
        if match:
            include_path = match.group(1)
            if os.path.exists(include_path):
                with open(include_path, 'r', encoding='utf-8') as inc:
                    result.append(f"\n<!-- BEGIN: {include_path} -->\n")
                    result.extend(inc.readlines())
                    result.append(f"\n<!-- END: {include_path} -->\n")
            else:
                result.append(f"<!-- ERROR: File '{include_path}' not found -->\n")
        else:
            result.append(line)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(result)

# Call the function
process_includes("thesis.md", "full_thesis.md")
